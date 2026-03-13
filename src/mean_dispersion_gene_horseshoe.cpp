//
// Created by Azeezat Mustapha on 22.02.26.
//
#include "includes/mean_dispersion_gene_horseshoe.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>
#include <algorithm>

//==============================================================================
// GENE-LEVEL HORSESHOE IMPLEMENTATION
//==============================================================================

/*
KEY DIFFERENCES from cluster-gene horseshoe:

OLD (cluster-gene):
  δ_{jg} ~ N(0, σ²_μ τ²_j λ²_{jg})  ← J × G separate parameters
  → Each cluster-gene pair can deviate independently
  → No reason for shrinkage, all genes can be "differential" somewhere

NEW (gene-level):
  δ_{jg} ~ N(0, σ²_μ ω²_g)  ← Only G parameters (one per gene)
  → SAME ω_g for all clusters j
  → If ω_g small, gene g is constant across ALL clusters (shrunk)
  → If ω_g large, gene g varies across clusters (marker)

This creates ACTUAL sparsity: genes either used or not used for clustering.
*/

//==============================================================================
// Initialize Parameters
//==============================================================================

GeneLevelHorseshoeParams initialize_gene_level_horseshoe_params(
    int G,
    const Eigen::MatrixXd& mu_initial,      // J × G
    const Eigen::VectorXd& mu_baseline      // G
) {
    GeneLevelHorseshoeParams params;

    int J = mu_initial.rows();

    params.lambda_sq.resize(G);
    params.xi.resize(G);
    params.omega.resize(G);

    // Initialize lambda based on cross-cluster variance (empirical Bayes)
    for (int g = 0; g < G; ++g) {
        // Compute variance of log(μ_{jg}) across clusters
        double mean_log_mu = 0.0;
        int count = 0;

        for (int j = 0; j < J; ++j) {
            if (mu_initial(j, g) > 0 && mu_baseline(g) > 0) {
                double log_mu = std::log(mu_initial(j, g));
                if (std::isfinite(log_mu)) {
                    mean_log_mu += log_mu;
                    count++;
                }
            }
        }

        if (count > 1) {
            mean_log_mu /= count;

            double var_log_mu = 0.0;
            for (int j = 0; j < J; ++j) {
                if (mu_initial(j, g) > 0 && mu_baseline(g) > 0) {
                    double log_mu = std::log(mu_initial(j, g));
                    if (std::isfinite(log_mu)) {
                        double dev = log_mu - mean_log_mu;
                        var_log_mu += dev * dev;
                    }
                }
            }
            var_log_mu /= (count - 1);

            // Initialize λ²_g ≈ observed variance
            // Clamp to reasonable range to avoid extreme initial values
            params.lambda_sq[g] = std::max(0.1, std::min(2.0, var_log_mu));
        } else {
            params.lambda_sq[g] = 1.0;
        }

        params.xi[g] = 1.0;
        params.omega[g] = 1.0;
    }

    // Global shrinkage: start at moderate value
    params.tau_0_sq = 0.2;
    params.xi_0 = 0.2;

    // Overall variance scale
    params.sigma_mu_sq = 0.1;

    return params;
}

//==============================================================================
// Main MCMC Update
//==============================================================================

GeneLevelHorseshoeResult mean_dispersion_gene_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,   // J × G current cluster means
    const Eigen::MatrixXd& phi_star_1_J,  // J × G dispersions
    const Eigen::VectorXd& mu_baseline,   // G baseline means
    GeneLevelHorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    GeneLevelHorseshoeResult result;

    // ══════════════════════════════════════════════════════════════════════
    // STEP 1: Compute deviations δ_{jg} = log(μ_{jg}) - log(μ_baseline_g)
    // ══════════════════════════════════════════════════════════════════════

    Eigen::MatrixXd delta(J, G);
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            if (mu_star_1_J(j, g) > 0 && mu_baseline(g) > 0 &&
                std::isfinite(mu_star_1_J(j, g)) && std::isfinite(mu_baseline(g))) {
                delta(j, g) = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
                if (!std::isfinite(delta(j, g))) {
                    delta(j, g) = 0.0;
                }
            } else {
                delta(j, g) = 0.0;
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 2: Update λ²_g for each gene
    // ══════════════════════════════════════════════════════════════════════

    for (int g = 0; g < G; ++g) {
        // Sum of squared deviations across clusters for gene g
        double sum_delta_sq = 0.0;
        for (int j = 0; j < J; ++j) {
            sum_delta_sq += delta(j, g) * delta(j, g);
        }

        if (sum_delta_sq <= 0 || !std::isfinite(sum_delta_sq)) {
            horseshoe_params.lambda_sq[g] = 0.001;  // Very small (shrunk out)
            continue;
        }

        // Full conditional:
        // λ²_g | · ~ InvGamma(J/2 + 1/2, Σ_j δ²_{jg}/(2σ²_μ τ²_0) + 1/ξ_g)
        double shape = static_cast<double>(J) / 2.0 + 0.5;
        double rate = sum_delta_sq / (2.0 * horseshoe_params.sigma_mu_sq * horseshoe_params.tau_0_sq)
                      + 1.0 / horseshoe_params.xi[g];

        if (!std::isfinite(rate) || rate <= 0) {
            horseshoe_params.lambda_sq[g] = 1.0;
            continue;
        }

        // Sample from InvGamma
        std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
        double lambda_sq_inv = gamma_dist(rng_local);
        horseshoe_params.lambda_sq[g] = 1.0 / lambda_sq_inv;

        // CRITICAL: Bound to reasonable range
        // This prevents numerical overflow but still allows strong shrinkage
        horseshoe_params.lambda_sq[g] = std::max(0.001, std::min(100.0, horseshoe_params.lambda_sq[g]));
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 3: Update ξ_g for each gene
    // ══════════════════════════════════════════════════════════════════════

    for (int g = 0; g < G; ++g) {
        // Full conditional: ξ_g | · ~ InvGamma(1, 1 + 1/λ²_g)
        double shape = 1.0;
        double rate = 1.0 + 1.0 / horseshoe_params.lambda_sq[g];

        std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
        double xi_inv = gamma_dist(rng_local);
        horseshoe_params.xi[g] = 1.0 / xi_inv;
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 4: Update τ²_0 (global shrinkage scale)
    // ══════════════════════════════════════════════════════════════════════

    double sum_term = 0.0;
    int valid_count = 0;

    for (int g = 0; g < G; ++g) {
        for (int j = 0; j < J; ++j) {
            if (std::isfinite(delta(j, g)) && horseshoe_params.lambda_sq[g] > 0) {
                sum_term += (delta(j, g) * delta(j, g)) /
                           (2.0 * horseshoe_params.sigma_mu_sq * horseshoe_params.lambda_sq[g]);
                valid_count++;
            }
        }
    }

    if (valid_count > 0) {
        // Full conditional: τ²_0 | · ~ InvGamma(GJ/2 + 1/2, Σ_{g,j} δ²_{jg}/(2σ²_μ λ²_g) + 1/ξ_0)
        double shape = static_cast<double>(valid_count) / 2.0 + 0.5;
        double rate = sum_term + 1.0 / horseshoe_params.xi_0;

        if (std::isfinite(rate) && rate > 0) {
            std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
            double tau_0_sq_inv = gamma_dist(rng_local);
            horseshoe_params.tau_0_sq = 1.0 / tau_0_sq_inv;

            // CRITICAL: Bound τ_0 to prevent explosion
            // Keep it moderate to ensure actual shrinkage happens
            horseshoe_params.tau_0_sq = std::max(0.01, std::min(0.5, horseshoe_params.tau_0_sq));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 5: Update ξ_0
    // ══════════════════════════════════════════════════════════════════════

    // Full conditional: ξ_0 | · ~ InvGamma(1, 1 + 1/τ²_0)
    double shape = 1.0;
    double rate = 1.0 + 1.0 / horseshoe_params.tau_0_sq;

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
    double xi_0_inv = gamma_dist(rng_local);
    horseshoe_params.xi_0 = 1.0 / xi_0_inv;

    // ══════════════════════════════════════════════════════════════════════
    // STEP 6: Update σ²_μ (overall variance scale)
    // ══════════════════════════════════════════════════════════════════════

    double sum_delta_omega = 0.0;
    int count = 0;

    for (int g = 0; g < G; ++g) {
        double omega_sq = horseshoe_params.tau_0_sq * horseshoe_params.lambda_sq[g];
        if (omega_sq > 0 && std::isfinite(omega_sq)) {
            for (int j = 0; j < J; ++j) {
                if (std::isfinite(delta(j, g))) {
                    sum_delta_omega += (delta(j, g) * delta(j, g)) / (2.0 * omega_sq);
                    count++;
                }
            }
        }
    }

    if (count > 0) {
        // Full conditional: σ²_μ | · ~ InvGamma(GJ/2 + a_σ, Σ_{g,j} δ²_{jg}/(2ω²_g) + b_σ)
        double a_sigma = 2.0;  // Prior shape
        double b_sigma = 1.0;  // Prior scale

        double shape = static_cast<double>(count) / 2.0 + a_sigma;
        double rate = sum_delta_omega + b_sigma;

        if (std::isfinite(rate) && rate > 0) {
            std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
            double sigma_mu_sq_inv = gamma_dist(rng_local);
            horseshoe_params.sigma_mu_sq = 1.0 / sigma_mu_sq_inv;

            // Bound to reasonable range
            horseshoe_params.sigma_mu_sq = std::max(0.1, std::min(2.0, horseshoe_params.sigma_mu_sq));
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 7: Compute ω_g = τ_0 × λ_g (for output/interpretation)
    // ══════════════════════════════════════════════════════════════════════

    double tau_0 = std::sqrt(horseshoe_params.tau_0_sq);
    for (int g = 0; g < G; ++g) {
        double lambda_g = std::sqrt(horseshoe_params.lambda_sq[g]);
        horseshoe_params.omega[g] = tau_0 * lambda_g;
    }

    // ══════════════════════════════════════════════════════════════════════
    // STEP 8: Update dispersion regression (same as normal model)
    // ══════════════════════════════════════════════════════════════════════

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

    // Build design matrix
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

    Eigen::VectorXd b_hat = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    double shape_2 = v_1 / 2.0;
    double scale_3 = v_2 / 2.0;

    std::gamma_distribution<double> gamma_dist_phi(shape_2, 1.0 / scale_3);
    double gamma_sample = gamma_dist_phi(rng_local);
    double alpha_phi_2 = 1.0 / gamma_sample;

    Eigen::MatrixXd cov_b = alpha_phi_2 * (X.transpose() * X).inverse();
    Eigen::VectorXd b_new = rmvnorm(b_hat, cov_b);

    // ══════════════════════════════════════════════════════════════════════
    // STEP 9: Calculate log-likelihood
    // ══════════════════════════════════════════════════════════════════════

    double log_lik_delta = 0.0;
    double const_part = -0.5 * std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            if (mu_star_1_J(j, g) > 0 && mu_baseline(g) > 0) {
                double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
                if (std::isfinite(delta_jg)) {
                    // Prior variance for gene g (same for all clusters)
                    double omega_sq = horseshoe_params.tau_0_sq * horseshoe_params.lambda_sq[g];
                    double total_var = horseshoe_params.sigma_mu_sq * omega_sq;

                    if (total_var > 0 && std::isfinite(total_var)) {
                        log_lik_delta += const_part - 0.5 * std::log(total_var)
                                       - (delta_jg * delta_jg) / (2.0 * total_var);
                    }
                }
            }
        }
    }

    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2)
                    - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    // ══════════════════════════════════════════════════════════════════════
    // STEP 10: Return results
    // ══════════════════════════════════════════════════════════════════════

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.horseshoe_params = horseshoe_params;

    return result;
}

//==============================================================================
// Helper: Get Prior Variance for Each Gene
//==============================================================================

Eigen::VectorXd get_gene_prior_variances(const GeneLevelHorseshoeParams& params) {
    /*
    Returns prior variance for each gene: σ²_μ × ω²_g

    This is used when sampling μ_{jg}:
      Prior: log(μ_{jg}) ~ N(log(μ_baseline_g), gene_prior_variance[g])
    */

    int G = params.omega.size();
    Eigen::VectorXd variances(G);

    for (int g = 0; g < G; ++g) {
        double omega_sq = params.tau_0_sq * params.lambda_sq[g];
        variances(g) = params.sigma_mu_sq * omega_sq;
    }

    return variances;
}