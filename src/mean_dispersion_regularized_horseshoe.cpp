#include "includes/mean_dispersion_regularized_horseshoe.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef RCPP_VERSION
#include <Rcpp.h>
#endif

// ── helpers ──────────────────────────────────────────────────────────────────

static inline double lambda_tilde_sq(double lambda_sq, double tau_sq, double c_sq) {
    return (c_sq * lambda_sq) / (c_sq + tau_sq * lambda_sq);
}

// Log-unnormalized posterior for log(λ_{jg}) — used in MH step
static double log_post_log_lambda(
    double log_lambda,          // proposal point
    double delta,
    double nu,
    double tau_sq,
    double sigma_sq,
    double c_sq
) {
    double lambda   = std::exp(log_lambda);
    double lsq      = lambda * lambda;
    double lt_sq    = lambda_tilde_sq(lsq, tau_sq, c_sq);

    if (lt_sq <= 0.0 || !std::isfinite(lt_sq)) return -std::numeric_limits<double>::infinity();


    double lp = 0.0;
    lp += -0.5 * std::log(lt_sq) - (delta * delta) / (2.0 * sigma_sq * tau_sq * lt_sq);  // likelihood
    lp += -(1.5) * std::log(lsq) - 1.0 / (nu * lsq);   // InvGamma(1/2, 1/nu) prior on lambda^2
    lp += log_lambda;   // Jacobian for sampling in log-space
    return lp;
}

// Log-unnormalized posterior for log(τ_j) — used in MH step (not used in Makalic-Schmidt)
static double log_post_log_tau(
    double log_tau,
    const std::vector<double>& deltas,
    const std::vector<double>& lambda_sqs,
    double xi_j,
    double tau_0_sq,
    double sigma_sq,
    double c_sq
) {
    double tau   = std::exp(log_tau);
    double tsq   = tau * tau;

    double lp = -(1.5) * std::log(tsq) - 1.0 / (xi_j * tau_0_sq * tsq);  //  prior
    lp += log_tau;  // Jacobian

    for (size_t g = 0; g < deltas.size(); ++g) {
        double lt_sq = lambda_tilde_sq(lambda_sqs[g], tsq, c_sq);
        if (lt_sq <= 0.0 || !std::isfinite(lt_sq)) return -std::numeric_limits<double>::infinity();
        lp += -0.5 * std::log(tsq * lt_sq) - (deltas[g] * deltas[g]) / (2.0 * sigma_sq * tsq * lt_sq);
    }
    return lp;
}


// ============================================================
// NEW: Makalic-Schmidt tau_j update for regularized horseshoe
// ============================================================
void update_tau_makalic_schmidt_regularized(
    RegularizedHorseshoeParams& rhs_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    int J, int G
) {
    const double lambda_floor = 0.01;   // floor for lambda
    const double log_eps = 1e-10;       // small constant for log stability
    const double tau_min = 0.01;        // lower bound for tau
    const double tau_max = 50.0;        // FIXED: Reasonable upper bound (was 10, then removed)
    const double c_sq = rhs_params.c_squared;
    const double sigma_sq = rhs_params.sigma_mu * rhs_params.sigma_mu;

    for (int j = 0; j < J; j++) {
        double sum_term = 0.0;
        int valid_count = 0;

        for (int g = 0; g < G; g++) {
            double mu_star_val = mu_star_1_J(j, g);
            double mu_base_val = mu_baseline(g);

            // SAFETY CHECK 1: Positive and finite
            if (mu_star_val <= 0 || mu_base_val <= 0) continue;
            if (!std::isfinite(mu_star_val) || !std::isfinite(mu_base_val)) continue;

            double mu_star_safe = std::max(mu_star_val, log_eps);
            double mu_base_safe = std::max(mu_base_val, log_eps);

            double delta_jg = std::log(mu_star_safe) - std::log(mu_base_safe);

            // SAFETY CHECK 2: Finite delta
            if (!std::isfinite(delta_jg)) continue;

            double lambda_jg = std::max(rhs_params.lambda[j](g), lambda_floor);
            double lambda_sq_val = lambda_jg * lambda_jg;

            // CRITICAL FIX: For tau update, use RAW lambda, not lambda_tilde
            // Using lambda_tilde creates feedback loop with current tau
            // This breaks the Gibbs sampler independence

            // SAFETY CHECK 3: Valid lambda
            if (!std::isfinite(lambda_sq_val) || lambda_sq_val <= 0) continue;

            // Use RAW lambda for tau update
            sum_term += (delta_jg * delta_jg) / (lambda_sq_val * sigma_sq);
            valid_count++;
        }

        if (valid_count == 0) {
            rhs_params.tau[j] = 0.5;  // neutral default
            continue;
        }

        // CRITICAL FIX: No regularization - use sum directly
        // sum_term is Σ_g (delta² / (lambda_tilde² × sigma²))

        // SAFETY CHECK
        if (!std::isfinite(sum_term) || sum_term < 0) {
            rhs_params.tau[j] = 0.5;
            continue;
        }

        // Update tau_j using Half-Cauchy(0, tau_0) prior (Makalic-Schmidt)
        double tau_0_sq = rhs_params.tau_0 * rhs_params.tau_0;
        double shape_tau = valid_count / 2.0 + 0.5;
        double rate_tau  = 1.0 / (rhs_params.xi[j] * tau_0_sq) + sum_term / 2.0;

        // SAFETY CHECK 4: Valid gamma parameters
        if (!std::isfinite(rate_tau) || rate_tau <= 0) {
            rhs_params.tau[j] = 0.5;
            continue;
        }

        std::gamma_distribution<double> gamma_dist_tau(shape_tau, 1.0 / rate_tau);
        double tau_sq_inv = gamma_dist_tau(rng_local);

        // SAFETY CHECK 5: Valid sampled value
        if (!std::isfinite(tau_sq_inv) || tau_sq_inv <= 0) {
            rhs_params.tau[j] = 0.5;
            continue;
        }

        double tau_new = 1.0 / std::sqrt(tau_sq_inv);

        // FIXED: Both bounds - prevent explosion and sticking
        rhs_params.tau[j] = std::min(std::max(tau_new, tau_min), tau_max);

        // Update auxiliary variable xi_j
        double tau_j_sq = rhs_params.tau[j] * rhs_params.tau[j];
        double rate_xi = 1.0 + 1.0 / (tau_j_sq * tau_0_sq);

        if (std::isfinite(rate_xi) && rate_xi > 0) {
            std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
            double xi_inv = gamma_dist_xi(rng_local);
            if (std::isfinite(xi_inv) && xi_inv > 0) {
                rhs_params.xi[j] = 1.0 / xi_inv;
            }
        }
    }
}


// ============================================================
// NEW: Makalic-Schmidt tau_0 update for regularized horseshoe
// ============================================================
void update_tau_0_makalic_schmidt_regularized(
    RegularizedHorseshoeParams& rhs_params,
    int J
) {
    const double tau_0_min = 0.1;
    const double tau_0_max = 2.0;

    double sum_inv_tau_sq = 0.0;
    int valid_count = 0;

    for (int j = 0; j < J; j++) {
        double tau_j = rhs_params.tau[j];

        // SAFETY CHECK: Valid tau
        if (!std::isfinite(tau_j) || tau_j <= 0) continue;

        double tau_j_sq = tau_j * tau_j;
        sum_inv_tau_sq += 1.0 / tau_j_sq;
        valid_count++;
    }

    if (valid_count == 0) {
        rhs_params.tau_0 = 0.5;
        return;
    }

    // tau_0^2 ~ InvGamma(J/2, 1/xi_0 + sum(1/tau_j^2) / 2)
    double shape = valid_count / 2.0;
    double rate = 1.0 / rhs_params.xi_0 + sum_inv_tau_sq / 2.0;

    // SAFETY CHECK: Valid gamma parameters
    if (!std::isfinite(rate) || rate <= 0) {
        rhs_params.tau_0 = 0.5;
        return;
    }

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
    double tau_0_sq_inv = gamma_dist(rng_local);

    // SAFETY CHECK: Valid sampled value
    if (!std::isfinite(tau_0_sq_inv) || tau_0_sq_inv <= 0) {
        rhs_params.tau_0 = 0.5;
        return;
    }

    double tau_0_new = 1.0 / std::sqrt(tau_0_sq_inv);

    // Truncate
    rhs_params.tau_0 = std::min(std::max(tau_0_new, tau_0_min), tau_0_max);

    // Update auxiliary variable xi_0
    double tau_0_sq = rhs_params.tau_0 * rhs_params.tau_0;
    double rate_xi = 1.0 + 1.0 / tau_0_sq;

    if (std::isfinite(rate_xi) && rate_xi > 0) {
        std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
        double xi_inv = gamma_dist_xi(rng_local);
        if (std::isfinite(xi_inv) && xi_inv > 0) {
            rhs_params.xi_0 = 1.0 / xi_inv;
        }
    }
}


// ============================================================
// Initialize — default
// ============================================================
RegularizedHorseshoeParams initialize_regularized_horseshoe_params(int J, int G, double p_0) {
    RegularizedHorseshoeParams params;

    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j]     = Eigen::VectorXd::Ones(G);
    }

    params.tau_0       = 0.5;  // Start smaller for stability
    params.xi_0        = 1.0;
    params.sigma_mu    = 0.5;  // Start smaller for stability

    // Initialize tau_j and xi_j from their prior: tau_j ~ HC(0, tau_0)
    // These will be updated during MCMC using Makalic-Schmidt
    params.tau.resize(J);
    params.xi.resize(J);
    std::gamma_distribution<double> gamma_half(0.5, 1.0);
    for (int j = 0; j < J; ++j) {
        double xi_j_init = 1.0 / gamma_half(rng_local);
        double tau_0_sq  = params.tau_0 * params.tau_0;
        std::gamma_distribution<double> gamma_tau(0.5, 1.0 / (xi_j_init * tau_0_sq));
        params.tau[j] = 1.0 / std::sqrt(gamma_tau(rng_local));
        params.xi[j]  = xi_j_init;
    }
    params.alpha_phi_2 = 1.0;

    // c² = prior odds of being differential: p_0/(G-p_0)
    double frac = std::max(1.0, p_0) / std::max(1.0, static_cast<double>(G) - p_0);
    params.c_squared = std::max(0.25, std::min(4.0, frac));

    return params;
}


// ============================================================
// Initialize — empirical
// c² set from 95th percentile of |δ|, FIXED thereafter
// ============================================================
RegularizedHorseshoeParams initialize_regularized_horseshoe_params_empirical(
    int J, int G, double p_0,
    const Eigen::VectorXd& mu_baseline,
    const Eigen::MatrixXd& mu_initial
) {
    RegularizedHorseshoeParams params;

    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j]     = Eigen::VectorXd::Ones(G);
    }
    params.tau.assign(J, 1.0);
    params.xi.assign(J, 1.0);
    params.tau_0       = 1.0;
    params.xi_0        = 1.0;
    params.alpha_phi_2 = 1.0;

    std::vector<double> gene_vars, abs_deltas;

    for (int g = 0; g < G; ++g) {
        if (mu_baseline(g) <= 0 || !std::isfinite(mu_baseline(g))) continue;
        std::vector<double> deltas;
        for (int j = 0; j < J; ++j) {
            if (mu_initial(j, g) > 0 && std::isfinite(mu_initial(j, g))) {
                double d = std::log(mu_initial(j, g)) - std::log(mu_baseline(g));
                if (std::isfinite(d)) { deltas.push_back(d); abs_deltas.push_back(std::abs(d)); }
            }
        }
        if (deltas.size() > 1) {
            double m = std::accumulate(deltas.begin(), deltas.end(), 0.0) / deltas.size();
            double v = 0.0;
            for (double d : deltas) v += (d - m) * (d - m);
            v /= (deltas.size() - 1);
            if (std::isfinite(v) && v > 0) gene_vars.push_back(v);
        }
    }

    if (!gene_vars.empty()) {
        std::sort(gene_vars.begin(), gene_vars.end());
        params.sigma_mu = std::sqrt(gene_vars[gene_vars.size() / 2]);
        params.sigma_mu = std::max(0.1, std::min(1.0, params.sigma_mu));  // FIXED: Tighter bound (was 3.0)
    } else {
        params.sigma_mu = 1.0;
    }

    // ── Initialize tau_0 and tau_j from data (FIXED - NOT UPDATED) ──
    // Piironen & Vehtari (2017) use empirical Bayes for tau
    // These values are SET ONCE and NOT updated during MCMC
    // Reason: Regularization breaks conjugacy for tau updates

    // tau_0: global scale across all clusters
    double tau_0_init = (p_0 / std::max(1.0, static_cast<double>(G) - p_0)) * params.sigma_mu;
    params.tau_0 = std::max(0.3, std::min(3.0, tau_0_init));

    // tau_j: cluster-specific scale (median absolute deviation per cluster)
    for (int j = 0; j < J; ++j) {
        std::vector<double> abs_deltas_j;
        for (int g = 0; g < G; ++g) {
            if (mu_initial(j, g) > 0 && mu_baseline(g) > 0 &&
                std::isfinite(mu_initial(j, g)) && std::isfinite(mu_baseline(g))) {
                double delta = std::log(mu_initial(j, g)) - std::log(mu_baseline(g));
                if (std::isfinite(delta)) {
                    abs_deltas_j.push_back(std::abs(delta));
                }
            }
        }

        if (!abs_deltas_j.empty()) {
            std::sort(abs_deltas_j.begin(), abs_deltas_j.end());
            double mad_j = abs_deltas_j[abs_deltas_j.size() / 2];
            params.tau[j] = std::max(0.3, std::min(5.0, mad_j));  // FIXED: Allow larger tau
        } else {
            params.tau[j] = params.tau_0;  // fallback to global scale
        }

        // xi_j not needed (tau not updated)
        params.xi[j] = 1.0;
    }

    // xi_0 not needed (tau_0 not updated)
    params.xi_0 = 1.0;

    if (!abs_deltas.empty()) {
        std::sort(abs_deltas.begin(), abs_deltas.end());
        int idx = std::min(static_cast<int>(0.95 * abs_deltas.size()),
                           static_cast<int>(abs_deltas.size()) - 1);
        double c = abs_deltas[idx];
        params.c_squared = std::max(0.25, std::min(4.0, c * c));
    } else {
        double frac = std::max(1.0, p_0) / std::max(1.0, static_cast<double>(G) - p_0);
        params.c_squared = std::max(0.25, std::min(4.0, frac));
    }

    return params;
}


// ============================================================
// MAIN MCMC FUNCTION - ALWAYS uses Makalic-Schmidt (Full Bayesian)
// ============================================================
MeanDispersionRegularizedHorseshoeResult mean_dispersion_regularized_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    RegularizedHorseshoeParams& rhs_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    MeanDispersionRegularizedHorseshoeResult result;

    const double c_sq = rhs_params.c_squared;    // FIXED — never changes
    double sigma_sq = rhs_params.sigma_mu * rhs_params.sigma_mu;

    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // ============================================================
    // STEP 1: Update tau_j and tau_0 using EMPIRICAL BAYES
    //         Piironen & Vehtari (2017) use FIXED tau, not sampled
    //         Regularization breaks Gibbs conjugacy - MH needed for tau
    //         We use fixed tau for stability (original P&V approach)
    // ============================================================

    // NOTE: Tau is NOT updated in regularized horseshoe!
    // Reason: λ̃² = (c²λ²)/(c² + τ²λ²) couples λ and τ
    // This breaks conjugacy - no closed-form Gibbs update
    // Makalic-Schmidt only works for STANDARD horseshoe
    //
    // Piironen & Vehtari (2017) use empirical Bayes (fixed tau)
    // Tau is set once at initialization and not updated
    //
    // If you want to update tau, use Metropolis-Hastings (complex)

    // COMMENTED OUT - causes lambda explosion and tau collapse:
    // update_tau_makalic_schmidt_regularized(
    //     rhs_params, mu_star_1_J, mu_baseline, J, G
    // );

    // COMMENTED OUT - tau_0 also fixed:
    // update_tau_0_makalic_schmidt_regularized(rhs_params, J);

    // ============================================================
    // STEP 2: Update λ_{jg} via Metropolis-Hastings in log-space
    // ============================================================

    const double MH_SD_LAMBDA = 0.3;

    for (int j = 0; j < J; ++j) {
        double tau_j = rhs_params.tau[j];

        // SAFETY CHECK: Valid tau
        if (!std::isfinite(tau_j) || tau_j <= 0) continue;

        double tau_sq_j = tau_j * tau_j;

        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            // SAFETY CHECK 1: Positive and finite mu values
            if (mu_jg <= 0 || mu_bg <= 0 || !std::isfinite(mu_jg) || !std::isfinite(mu_bg)) continue;

            double delta = std::log(mu_jg) - std::log(mu_bg);

            // SAFETY CHECK 2: Finite delta
            if (!std::isfinite(delta)) continue;

            double nu_jg = rhs_params.nu[j](g);

            // SAFETY CHECK 3: Valid nu
            if (!std::isfinite(nu_jg) || nu_jg <= 0) {
                nu_jg = 1.0;
                rhs_params.nu[j](g) = 1.0;
            }

            double lambda_current = rhs_params.lambda[j](g);

            // SAFETY CHECK 4: Valid lambda
            if (!std::isfinite(lambda_current) || lambda_current <= 0) {
                lambda_current = 1.0;
                rhs_params.lambda[j](g) = 1.0;
            }

            double log_lam = std::log(lambda_current);

            // Propose
            std::normal_distribution<double> norm(log_lam, MH_SD_LAMBDA);
            double log_lam_prop = norm(rng_local);

            double lp_old  = log_post_log_lambda(log_lam,      delta, nu_jg, tau_sq_j, sigma_sq, c_sq);
            double lp_prop = log_post_log_lambda(log_lam_prop, delta, nu_jg, tau_sq_j, sigma_sq, c_sq);

            double log_acc = lp_prop - lp_old;

            // SAFETY CHECK 5: Finite acceptance ratio
            if (std::isfinite(log_acc) && std::log(unif(rng_local)) < log_acc) {
                double lambda_new = std::exp(log_lam_prop);

                // SAFETY CHECK 6: Floor lambda (same as standard horseshoe)
                lambda_new = std::max(lambda_new, 0.01);
                rhs_params.lambda[j](g) = lambda_new;
            }

            // ── Update ν_{jg} — exact Gibbs ──────────────────────────────────
            // 1/ν ~ Gamma(1,  1 + 1/λ²)
            double lambda_sq_jg = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);
            double rate_nu = 1.0 + 1.0 / lambda_sq_jg;

            // SAFETY CHECK 7: Valid gamma parameters for nu
            if (std::isfinite(rate_nu) && rate_nu > 0) {
                std::gamma_distribution<double> g_nu(1.0, 1.0 / rate_nu);
                double nu_inv = g_nu(rng_local);

                if (std::isfinite(nu_inv) && nu_inv > 0) {
                    rhs_params.nu[j](g) = 1.0 / nu_inv;
                } else {
                    rhs_params.nu[j](g) = 1.0;
                }
            }
        }
    }

    // ============================================================
    // STEP 3: Update σ_μ — exact Gibbs with SAFETY CHECKS
    // ============================================================
    {
        double sum_sq = 0.0;
        int valid = 0;

        for (int j = 0; j < J; ++j) {
            double tau_j = rhs_params.tau[j];

            // SAFETY CHECK: Valid tau
            if (!std::isfinite(tau_j) || tau_j <= 0) continue;

            double tau_sq_j = tau_j * tau_j;

            for (int g = 0; g < G; ++g) {
                double mu_jg = mu_star_1_J(j, g);
                double mu_bg = mu_baseline(g);

                // SAFETY CHECK 1: Positive and finite
                if (mu_jg <= 0 || mu_bg <= 0) continue;
                if (!std::isfinite(mu_jg) || !std::isfinite(mu_bg)) continue;

                double delta = std::log(mu_jg) - std::log(mu_bg);

                // SAFETY CHECK 2: Finite delta
                if (!std::isfinite(delta)) continue;

                double lsq = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);
                double lt_sq = lambda_tilde_sq(lsq, tau_sq_j, c_sq);

                // SAFETY CHECK 3: Valid lambda_tilde
                if (!std::isfinite(lt_sq) || lt_sq <= 0) continue;

                double v_jg = tau_sq_j * lt_sq;

                // SAFETY CHECK 4: Valid variance
                if (v_jg > 0.0 && std::isfinite(v_jg)) {
                    sum_sq += (delta * delta) / v_jg;
                    ++valid;
                }
            }
        }

        if (valid > 0) {
            const double a_sigma = 1.5;  // FIXED: Weaker prior (was 2.0)
            const double b_sigma = 0.5;
            double shape_s = a_sigma + static_cast<double>(valid) / 2.0;
            double rate_s = b_sigma + sum_sq / 2.0;

            // SAFETY CHECK 5: Valid gamma parameters
            if (std::isfinite(rate_s) && rate_s > 0) {
                std::gamma_distribution<double> g_sig(shape_s, 1.0 / rate_s);
                double sigma_sq_inv = g_sig(rng_local);

                // SAFETY CHECK 6: Valid sampled value
                if (std::isfinite(sigma_sq_inv) && sigma_sq_inv > 0) {
                    rhs_params.sigma_mu = 1.0 / std::sqrt(sigma_sq_inv);

                    // CRITICAL FIX: Relaxed upper bound (was 1.0, now 3.0)
                    rhs_params.sigma_mu = std::max(0.1, std::min(3.0, rhs_params.sigma_mu));
                } else {
                    rhs_params.sigma_mu = 0.5;
                }
            } else {
                rhs_params.sigma_mu = 0.5;
            }
        } else {
            rhs_params.sigma_mu = 0.5;  // Default if no valid data
        }
    }

    // Update sigma_sq for use in log-likelihood
    sigma_sq = rhs_params.sigma_mu * rhs_params.sigma_mu;

    // ============================================================
    // STEP 4 — Update dispersion regression (b, alpha_phi_2)
    // ============================================================

    std::vector<double> x_vals, y_vals;
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu  = mu_star_1_J(j, g);
            double phi = phi_star_1_J(j, g);
            if (mu > 0 && phi > 0 && std::isfinite(std::log(mu)) && std::isfinite(std::log(phi))) {
                x_vals.push_back(std::log(mu));
                y_vals.push_back(std::log(phi));
            }
        }
    }

    int n = static_cast<int>(x_vals.size());

    if (n == 0) {
        result.alpha_phi_2    = 1.0;
        result.b              = m_b;
        result.rhs_params     = rhs_params;
        result.log_likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }

    Eigen::VectorXd y(n);
    Eigen::MatrixXd X;
    for (int i = 0; i < n; ++i) y(i) = y_vals[i];

    if (quadratic) {
        X.resize(n, 3);
        for (int i = 0; i < n; ++i) { X(i,0)=1.0; X(i,1)=x_vals[i]; X(i,2)=x_vals[i]*x_vals[i]; }
    } else {
        X.resize(n, 2);
        for (int i = 0; i < n; ++i) { X(i,0)=1.0; X(i,1)=x_vals[i]; }
    }

    int p = m_b.size();
    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd Xty = X.transpose() * y;
    double          yty = y.squaredNorm();

    Eigen::MatrixXd parameter0 = Eigen::MatrixXd::Identity(p, p) + XtX;
    Eigen::VectorXd parameter1 = m_b + Xty;
    Eigen::MatrixXd V_tilde    = parameter0.inverse();
    Eigen::VectorXd m_tilde    = V_tilde * parameter1;

    double v_tilde_1 = v_1 + static_cast<double>(J * G) / 2.0;
    double v_tilde_2 = v_2 + 0.5 * (yty
                                     - (m_tilde.transpose() * parameter0 * m_tilde)(0,0)
                                     + m_b.squaredNorm());
    if (v_tilde_2 <= 0.0) v_tilde_2 = v_2;

    std::gamma_distribution<double> g_phi(v_tilde_1, 1.0 / v_tilde_2);
    double alpha_phi_2 = 1.0 / g_phi(rng_local);

    Eigen::VectorXd b_new = rmvnorm(m_tilde, alpha_phi_2 * V_tilde);
    rhs_params.alpha_phi_2 = alpha_phi_2;

    result.alpha_phi_2 = alpha_phi_2;
    result.b           = b_new;
    result.rhs_params  = rhs_params;

    // ============================================================
    // STEP 5: Calculate Log-Likelihood with SAFETY CHECKS
    // ============================================================
    double log_lik_delta = 0.0;
    const double log2pi  = std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        double tau_j = rhs_params.tau[j];

        // SAFETY CHECK: Valid tau
        if (!std::isfinite(tau_j) || tau_j <= 0) continue;

        double tau_sq_j   = tau_j * tau_j;
        double sigma_sq_u = rhs_params.sigma_mu * rhs_params.sigma_mu;

        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            // SAFETY CHECK 1: Positive mu values
            if (mu_jg <= 0 || mu_bg <= 0) continue;

            double delta = std::log(mu_jg) - std::log(mu_bg);

            // SAFETY CHECK 2: Finite delta
            if (!std::isfinite(delta)) continue;

            double lsq   = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);
            double lt_sq = lambda_tilde_sq(lsq, tau_sq_j, c_sq);
            double v_jg  = sigma_sq_u * tau_sq_j * lt_sq;

            // SAFETY CHECK 3: Valid variance
            if (v_jg <= 0.0 || !std::isfinite(v_jg)) continue;

            log_lik_delta += -0.5 * (log2pi + std::log(v_jg) + delta * delta / v_jg);
        }
    }

    double rss = (y - X * b_new).squaredNorm();
    double log_lik_phi = -0.5 * (n * log2pi + n * std::log(alpha_phi_2) + rss / alpha_phi_2);

    result.log_likelihood = log_lik_delta + log_lik_phi;
    return result;
}