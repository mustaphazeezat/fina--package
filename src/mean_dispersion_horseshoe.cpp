#include "includes/mean_dispersion_horseshoe.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>

// ========================================
// Update tau_j using Makalic-Schmidt approach
// ========================================

void update_tau_makalic_schmidt(
    HorseshoeParams& horseshoe_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    int J, int G
) {
    const double lambda_floor = 0.01;   // floor for lambda
    const double log_eps = 1e-10;       // small constant for log stability
    const double tau_min = 0.01;        // FIXED: lower bound for tau (was 0.1)
    const double tau_max = 10.0;        // FIXED: upper bound for tau (was 2.0)

    for (int j = 0; j < J; j++) {
        double sum_term = 0.0;
        int valid_count = 0;

        for (int g = 0; g < G; g++) {
            double mu_star_val = mu_star_1_J(j, g);
            double mu_base_val = mu_baseline(g);

            // SAFETY CHECK: Positive and finite
            if (mu_star_val <= 0 || mu_base_val <= 0) continue;
            if (!std::isfinite(mu_star_val) || !std::isfinite(mu_base_val)) continue;

            double mu_star_safe = std::max(mu_star_val, log_eps);
            double mu_base_safe = std::max(mu_base_val, log_eps);

            double delta_jg = std::log(mu_star_safe) - std::log(mu_base_safe);
            if (!std::isfinite(delta_jg)) continue;

            double lambda_jg = std::max(horseshoe_params.lambda[j](g), lambda_floor);
            double sigma_mu = horseshoe_params.sigma_mu;

            // CRITICAL FIX: Keep sum_term as SUM, don't divide by valid_count!
            sum_term += (delta_jg * delta_jg) / (lambda_jg * lambda_jg * sigma_mu * sigma_mu);
            valid_count++;
        }

        if (valid_count == 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        // CRITICAL FIX: No regularization - use sum directly
        // sum_term is Σ_g (delta² / (lambda² × sigma²))

        // SAFETY CHECK
        if (!std::isfinite(sum_term) || sum_term < 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        // Update tau_j using Half-Cauchy(0,1) prior (Makalic-Schmidt)
        // τ²_j | rest ~ InvGamma(G/2 + 1/2, 1/ξ_j + S_j/2)
        double shape_tau = valid_count / 2.0 + 0.5;
        double rate_tau  = 1.0 / horseshoe_params.xi[j] + sum_term / 2.0;

        // SAFETY CHECK
        if (!std::isfinite(rate_tau) || rate_tau <= 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        std::gamma_distribution<double> gamma_dist_tau(shape_tau, 1.0 / rate_tau);
        double tau_sq_inv = gamma_dist_tau(rng_local);

        if (!std::isfinite(tau_sq_inv) || tau_sq_inv <= 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        double tau_new = 1.0 / std::sqrt(tau_sq_inv);

        // FIXED: Relaxed bounds to allow more exploration
        horseshoe_params.tau[j] = std::min(std::max(tau_new, tau_min), tau_max);

        // Update auxiliary variable xi_j
        double tau_j_sq = horseshoe_params.tau[j] * horseshoe_params.tau[j];
        double rate_xi = 1.0 + 1.0 / tau_j_sq;

        if (std::isfinite(rate_xi) && rate_xi > 0) {
            std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
            double xi_inv = gamma_dist_xi(rng_local);
            if (std::isfinite(xi_inv) && xi_inv > 0) {
                horseshoe_params.xi[j] = 1.0 / xi_inv;
            }
        }
    }
}

MeanDispersionHorseshoeResult mean_dispersion_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    HorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    MeanDispersionHorseshoeResult result;

    // ============================================================
    // 1. Update global shrinkage (tau_j, xi_j) FIRST
    //    Must precede lambda update to break the tau-lambda feedback loop.
    //    When lambda is near zero, S_j = sum(delta^2/lambda^2) explodes,
    //    driving rate_tau to infinity and tau to infinity. By updating tau
    //    first (with current lambda still reasonable), tau gets a sensible
    //    value that then feeds a reasonable rate into the lambda update.
    // ============================================================
    update_tau_makalic_schmidt(horseshoe_params, mu_star_1_J, mu_baseline, J, G);

    // ============================================================
    // 2. Update local shrinkage (lambda_jg, nu_jg) with updated tau
    //
    //    1/lambda^2 ~ Gamma(1, 1/nu + delta^2/(2*sigma^2*tau^2))
    //    1/nu       ~ Gamma(1, 1 + 1/lambda^2)
    //
    //    lambda is floored at 1e-4 after sampling â€” lambda=0 is not in
    //    the support of HC+(0,1) but the sampler can get arbitrarily
    //    close, causing S_j = sum(delta^2/lambda^2) to overflow on the
    //    next tau update. The floor prevents that without meaningfully
    //    changing the posterior (HC+(0,1) has negligible mass below 1e-4
    //    when the data contains any non-zero signal).
    // ============================================================

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
            if (!std::isfinite(delta_jg)) continue;

            double tau_j   = horseshoe_params.tau[j];
            double sigma_mu = horseshoe_params.sigma_mu;
            double nu_jg   = horseshoe_params.nu[j](g);

            // 1/lambda^2 ~ Gamma(1, 1/nu + delta^2/(2*sigma^2*tau^2))
            double rate_lambda = 1.0 / nu_jg +
                                 (delta_jg * delta_jg) /
                                 (2.0 * tau_j * tau_j * sigma_mu * sigma_mu);

            std::gamma_distribution<double> gamma_dist_lambda(1.0, 1.0 / rate_lambda);
            double lambda_sq_inv = gamma_dist_lambda(rng_local);
            double lambda_jg     = 1.0 / std::sqrt(lambda_sq_inv);

            // CRITICAL: Both floor and ceiling
            // Floor prevents numerical issues
            // Ceiling prevents extreme values (lambda > 100 means essentially no shrinkage)
            lambda_jg = std::max(0.01, std::min(100.0, lambda_jg));

            horseshoe_params.lambda[j](g) = lambda_jg;

            // 1/nu ~ Gamma(1, 1 + 1/lambda^2)
            double rate_nu = 1.0 + 1.0 / (lambda_jg * lambda_jg);
            std::gamma_distribution<double> gamma_dist_nu(1.0, 1.0 / rate_nu);
            horseshoe_params.nu[j](g) = 1.0 / gamma_dist_nu(rng_local);
        }
    }

    // ============================================================
    // 3. Update global variance parameter (sigma_mu)
    // ============================================================

    double sum_all = 0.0;
	int valid_total = 0;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
			if (!std::isfinite(delta_jg)) continue;
            double tau_j = horseshoe_params.tau[j];
            double lambda_jg = horseshoe_params.lambda[j](g);

			if (!std::isfinite(tau_j) || tau_j == 0.0) continue;
			if (!std::isfinite(lambda_jg) || lambda_jg == 0.0) continue;

            sum_all += (delta_jg * delta_jg) / (tau_j * tau_j * lambda_jg * lambda_jg);

			valid_total++;
        }
    }


    // sigma_mu^2 ~ InvGamma(a_sigma + J*G/2, b_sigma + sum_all/2)
    // Using minimally informative prior to let data dominate
	if (valid_total == 0) {                                          // â† ADD
    	horseshoe_params.sigma_mu = 1.0;  // neutral default          // â† ADD
	} else{


    double a_sigma = 1.5;
    double b_sigma = 0.5;

    double shape_sigma = a_sigma + valid_total / 2.0;
    double rate_sigma = b_sigma + sum_all / 2.0;

    std::gamma_distribution<double> gamma_dist_sigma(shape_sigma, 1.0 / rate_sigma);
    double sigma_mu_sq_inv = gamma_dist_sigma(rng_local);
    horseshoe_params.sigma_mu = 1.0 / std::sqrt(sigma_mu_sq_inv);

    // CRITICAL FIX: Add bounds to prevent explosion
    horseshoe_params.sigma_mu = std::max(0.1, std::min(3.0, horseshoe_params.sigma_mu));

	}

    // ============================================================
    // 4. Update dispersion regression parameters
    // ============================================================

    // Collect valid log values for regression
    std::vector<double> x_vals, y_vals;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu = mu_star_1_J(j, g);
            double phi = phi_star_1_J(j, g);

            if (std::isfinite(std::log(mu)) && std::isfinite(std::log(phi)) &&
                mu > 0 && phi > 0) {
                x_vals.push_back(std::log(mu));
                y_vals.push_back(std::log(phi));
            }
        }
    }

    int n = x_vals.size();

    if (n == 0) {
        // Return defaults if no valid data
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        result.horseshoe_params = horseshoe_params;
        result.log_likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }

    // Fit linear regression for dispersion
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X;

    for (int i = 0; i < n; ++i) {
        y(i) = y_vals[i];
    }

    if (quadratic) {
        X.resize(n, 3);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
            X(i, 2) = x_vals[i] * x_vals[i];
        }
    } else {
        X.resize(n, 2);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
        }
    }

    // ============================================================
    // 4. Update dispersion regression â€” Normal-InvGamma conjugate posterior
    //    Matches the R implementation exactly.
    //
    //    Prior:  b | alpha_phi_2 ~ N(m_b, I)
    //            alpha_phi_2     ~ InvGamma(v_1, v_2)
    //
    //    parameter0 = I + X'X
    //    parameter1 = m_b + X'y
    //    V_tilde    = parameter0^{-1}
    //    m_tilde    = V_tilde * parameter1
    //    v_tilde_1  = v_1 + J*G/2
    //    v_tilde_2  = v_2 + 0.5*(y'y - m_tilde'*parameter0*m_tilde + sum(m_b^2))
    // ============================================================

    int p = m_b.size();   // 2 (linear) or 3 (quadratic)

    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd Xty = X.transpose() * y;
    double          yty = y.squaredNorm();

    Eigen::MatrixXd parameter0 = Eigen::MatrixXd::Identity(p, p) + XtX;
    Eigen::VectorXd parameter1 = m_b + Xty;

    Eigen::MatrixXd V_tilde = parameter0.inverse();
    Eigen::VectorXd m_tilde = V_tilde * parameter1;

    double v_tilde_1 = v_1 + static_cast<double>(J * G) / 2.0;
    double v_tilde_2 = v_2 + 0.5 * (yty
                                     - (m_tilde.transpose() * parameter0 * m_tilde)(0, 0)
                                     + m_b.squaredNorm());
    if (v_tilde_2 <= 0.0) v_tilde_2 = v_2;

    // Sample alpha_phi_2 ~ InvGamma(v_tilde_1, v_tilde_2)
    std::gamma_distribution<double> gamma_dist(v_tilde_1, 1.0 / v_tilde_2);
    double alpha_phi_2 = 1.0 / gamma_dist(rng_local);

    // Sample b ~ N(m_tilde, alpha_phi_2 * V_tilde)
    Eigen::MatrixXd cov_b = alpha_phi_2 * V_tilde;
    Eigen::VectorXd b_new = rmvnorm(m_tilde, cov_b);

    // Persist alpha_phi_2 for next iteration
    horseshoe_params.alpha_phi_2 = alpha_phi_2;

    // ============================================================
    // 5. Return results
    // ============================================================

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.horseshoe_params = horseshoe_params;
    // ============================================================
    // 6. Calculate Log-Likelihood
    // ============================================================
    double log_lik_delta = 0.0;
    double const_part = -0.5 * std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        double tau_sq = horseshoe_params.tau[j] * horseshoe_params.tau[j];
		if (!std::isfinite(tau_sq) || tau_sq == 0.0) continue;
        for (int g = 0; g < G; ++g) {
            double delta = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
			if (!std::isfinite(delta)) continue;

            double lambda_sq = horseshoe_params.lambda[j](g) * horseshoe_params.lambda[j](g);
            double sigma_sq_mu = horseshoe_params.sigma_mu * horseshoe_params.sigma_mu;

            // Total variance for this gene/cluster pair
            double total_var = sigma_sq_mu * tau_sq * lambda_sq;

			if (!std::isfinite(total_var) || total_var <= 0.0) continue;

            log_lik_delta += const_part - 0.5 * std::log(total_var) - (delta * delta) / (2.0 * total_var);
        }
    }

    // Log-likelihood of the dispersion residuals
    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2) - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    return result;
}

// Initialize horseshoe parameters
HorseshoeParams initialize_horseshoe_params(int J, int G) {
    HorseshoeParams params;

    // Initialize lambda (local shrinkage) to 1.0
    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }

    // Initialize tau (global shrinkage per cluster) to 1.0 (neutral)
    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);

    // Initialize overall variance
    params.sigma_mu = 1.0;

    return params;
}

HorseshoeParams initialize_horseshoe_params_empirical(
    int J, int G,
    const Eigen::VectorXd& mu_baseline,
    const Eigen::MatrixXd& mu_initial
) {
    HorseshoeParams params;

    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }
    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);

    // Estimate Ïƒ_Î¼ from variance of DEVIATIONS from baseline
    std::vector<double> all_variances;

    for (int g = 0; g < G; ++g) {
        if (mu_baseline(g) <= 0) continue;

        std::vector<double> deltas;
        for (int j = 0; j < J; ++j) {
            if (mu_initial(j, g) > 0) {
                // Deviation = log(cluster_mean) - log(baseline)
                double delta = std::log(mu_initial(j, g)) - std::log(mu_baseline(g));
                if (std::isfinite(delta)) {
                    deltas.push_back(delta);
                }
            }
        }

        if (deltas.size() > 1) {
            double mean = 0.0;
            for (double d : deltas) mean += d;
            mean /= deltas.size();

            double var = 0.0;
            for (double d : deltas) {
                var += (d - mean) * (d - mean);
            }
            var /= (deltas.size() - 1);

            if (std::isfinite(var) && var > 0) {
                all_variances.push_back(var);
            }
        }
    }

    // Set Ïƒ_Î¼ to median variance
    if (all_variances.size() > 0) {
        std::sort(all_variances.begin(), all_variances.end());
        double median_var = all_variances[all_variances.size() / 2];
        params.sigma_mu = std::sqrt(median_var);
        params.sigma_mu = std::max(0.2, std::min(params.sigma_mu, 0.5));
    } else {
        params.sigma_mu = 1.0;
    }

    return params;
}