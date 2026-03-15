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
    const double lambda_floor = 0.01;
    const double log_eps = 1e-10;
    const double tau_min = 0.01;
    const double tau_max = 10.0;

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

            double lambda_jg = horseshoe_params.lambda[j](g);
            double sigma_mu = horseshoe_params.sigma_mu;

            // CRITICAL: Validate parameters before division
            if (!std::isfinite(lambda_jg) || lambda_jg <= 0) {
                lambda_jg = 1.0;
            }
            if (!std::isfinite(sigma_mu) || sigma_mu <= 0) {
                sigma_mu = 1.0;
            }

            lambda_jg = std::max(lambda_jg, lambda_floor);

            double term = (delta_jg * delta_jg) / (lambda_jg * lambda_jg * sigma_mu * sigma_mu);

            // Only add if finite
            if (std::isfinite(term) && term >= 0) {
                sum_term += term;
                valid_count++;
            }
        }

        if (valid_count == 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        // Cap sum_term to prevent overflow
        sum_term = std::min(sum_term, 1e6);

        if (!std::isfinite(sum_term) || sum_term < 0) {
            horseshoe_params.tau[j] = 0.5;
            continue;
        }

        // Makalic-Schmidt update
        double shape_tau = valid_count / 2.0 + 0.5;

        double xi_j = horseshoe_params.xi[j];
        if (!std::isfinite(xi_j) || xi_j <= 0) {
            xi_j = 1.0;
            horseshoe_params.xi[j] = 1.0;
        }

        double rate_tau = 1.0 / xi_j + sum_term / 2.0;

        // Cap rate
        rate_tau = std::min(rate_tau, 1e6);

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
    // 1. Update tau first
    // ============================================================
    update_tau_makalic_schmidt(horseshoe_params, mu_star_1_J, mu_baseline, J, G);

    // ============================================================
    // 2. Update lambda WITH CRITICAL SAFETY CHECKS
    // ============================================================

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
            if (!std::isfinite(delta_jg)) continue;

            double tau_j   = horseshoe_params.tau[j];
            double sigma_mu = horseshoe_params.sigma_mu;
            double nu_jg   = horseshoe_params.nu[j](g);

            // CRITICAL: Validate ALL parameters before division
            if (!std::isfinite(tau_j) || tau_j <= 0) {
                tau_j = 1.0;
            }
            if (!std::isfinite(sigma_mu) || sigma_mu <= 0) {
                sigma_mu = 1.0;
            }
            if (!std::isfinite(nu_jg) || nu_jg <= 0) {
                nu_jg = 1.0;
                horseshoe_params.nu[j](g) = 1.0;
            }

            // 1/lambda^2 ~ Gamma(1, 1/nu + delta^2/(2*sigma^2*tau^2))
            double rate_lambda = 1.0 / nu_jg +
                                 (delta_jg * delta_jg) /
                                 (2.0 * tau_j * tau_j * sigma_mu * sigma_mu);

            // CRITICAL: Validate rate before using
            if (!std::isfinite(rate_lambda) || rate_lambda <= 0) {
                horseshoe_params.lambda[j](g) = 1.0;
                horseshoe_params.nu[j](g) = 1.0;
                continue;
            }

            // Cap rate to prevent extreme values
            rate_lambda = std::min(rate_lambda, 1e6);

            std::gamma_distribution<double> gamma_dist_lambda(1.0, 1.0 / rate_lambda);
            double lambda_sq_inv = gamma_dist_lambda(rng_local);

            // Check sampled value
            if (!std::isfinite(lambda_sq_inv) || lambda_sq_inv <= 0) {
                horseshoe_params.lambda[j](g) = 1.0;
                horseshoe_params.nu[j](g) = 1.0;
                continue;
            }

            double lambda_jg = 1.0 / std::sqrt(lambda_sq_inv);

            // CRITICAL: Both floor and ceiling
            lambda_jg = std::max(0.01, std::min(100.0, lambda_jg));

            horseshoe_params.lambda[j](g) = lambda_jg;

            // 1/nu ~ Gamma(1, 1 + 1/lambda^2)
            double rate_nu = 1.0 + 1.0 / (lambda_jg * lambda_jg);

            if (std::isfinite(rate_nu) && rate_nu > 0) {
                std::gamma_distribution<double> gamma_dist_nu(1.0, 1.0 / rate_nu);
                double nu_inv = gamma_dist_nu(rng_local);

                // CRITICAL: Check sampled value
                if (std::isfinite(nu_inv) && nu_inv > 0) {
                    horseshoe_params.nu[j](g) = 1.0 / nu_inv;
                } else {
                    horseshoe_params.nu[j](g) = 1.0;  // Safe default
                }
            } else {
                horseshoe_params.nu[j](g) = 1.0;  // Safe default
            }
        }
    }

    // ============================================================
    // 3. Update sigma_mu
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

    if (valid_total == 0) {
        horseshoe_params.sigma_mu = 1.0;
    } else {
        double a_sigma = 1.5;
        double b_sigma = 0.5;

        double shape_sigma = a_sigma + valid_total / 2.0;
        double rate_sigma = b_sigma + sum_all / 2.0;

        // CRITICAL: Validate gamma parameters
        if (!std::isfinite(shape_sigma) || shape_sigma <= 0 ||
            !std::isfinite(rate_sigma) || rate_sigma <= 0) {
            horseshoe_params.sigma_mu = 1.0;
        } else {
            std::gamma_distribution<double> gamma_dist_sigma(shape_sigma, 1.0 / rate_sigma);
            double sigma_mu_sq_inv = gamma_dist_sigma(rng_local);

            // CRITICAL: Check sampled value before sqrt and division
            if (std::isfinite(sigma_mu_sq_inv) && sigma_mu_sq_inv > 0) {
                horseshoe_params.sigma_mu = 1.0 / std::sqrt(sigma_mu_sq_inv);
                // Bounds
                horseshoe_params.sigma_mu = std::max(0.1, std::min(3.0, horseshoe_params.sigma_mu));
            } else {
                horseshoe_params.sigma_mu = 1.0;  // Safe default
            }
        }
    }

    // ============================================================
    // 4. Update dispersion regression parameters
    // ============================================================

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
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        result.horseshoe_params = horseshoe_params;
        result.log_likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }

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

    int p = m_b.size();

    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd Xty = X.transpose() * y;
    double yty = y.squaredNorm();

    Eigen::MatrixXd parameter0 = Eigen::MatrixXd::Identity(p, p) + XtX;
    Eigen::VectorXd parameter1 = m_b + Xty;

    Eigen::MatrixXd V_tilde = parameter0.inverse();
    Eigen::VectorXd m_tilde = V_tilde * parameter1;

    double v_tilde_1 = v_1 + static_cast<double>(J * G) / 2.0;
    double v_tilde_2 = v_2 + 0.5 * (yty
                                     - (m_tilde.transpose() * parameter0 * m_tilde)(0, 0)
                                     + m_b.squaredNorm());
    if (v_tilde_2 <= 0.0) v_tilde_2 = v_2;

    std::gamma_distribution<double> gamma_dist(v_tilde_1, 1.0 / v_tilde_2);
    double alpha_phi_2 = 1.0 / gamma_dist(rng_local);

    Eigen::MatrixXd cov_b = alpha_phi_2 * V_tilde;
    Eigen::VectorXd b_new = rmvnorm(m_tilde, cov_b);

    horseshoe_params.alpha_phi_2 = alpha_phi_2;

    // ============================================================
    // 5. Calculate log-likelihood
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

            double total_var = sigma_sq_mu * tau_sq * lambda_sq;

            if (!std::isfinite(total_var) || total_var <= 0.0) continue;

            log_lik_delta += const_part - 0.5 * std::log(total_var) - (delta * delta) / (2.0 * total_var);
        }
    }

    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2) - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    // ============================================================
    // 6. Return results
    // ============================================================

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.horseshoe_params = horseshoe_params;

    return result;
}

// Initialize horseshoe parameters
HorseshoeParams initialize_horseshoe_params(int J, int G) {
    HorseshoeParams params;

    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }

    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);
    params.sigma_mu = 1.0;
    params.alpha_phi_2 = 1.0;  // CRITICAL: Initialize this!

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

    std::vector<double> all_variances;

    for (int g = 0; g < G; ++g) {
        if (mu_baseline(g) <= 0) continue;

        std::vector<double> deltas;
        for (int j = 0; j < J; ++j) {
            if (mu_initial(j, g) > 0) {
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

    if (all_variances.size() > 0) {
        std::sort(all_variances.begin(), all_variances.end());
        double median_var = all_variances[all_variances.size() / 2];
        params.sigma_mu = std::sqrt(median_var);
        params.sigma_mu = std::max(0.2, std::min(params.sigma_mu, 0.5));
    } else {
        params.sigma_mu = 1.0;
    }

    params.alpha_phi_2 = 1.0;  // CRITICAL: Initialize this!

    return params;
}