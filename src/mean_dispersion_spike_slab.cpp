#include "includes/mean_dispersion_spike_slab.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>
#include <algorithm>

// ========================================
// Initialize Spike-and-Slab Parameters
// ========================================

SpikeSlabParams initialize_spike_slab_params(int J, int G) {
    SpikeSlabParams params;

    // Initialize gamma (indicators) - start with 50% in slab
    params.gamma.resize(J);
    params.theta.resize(J);

    std::uniform_real_distribution<double> unif(0, 1);

    for (int j = 0; j < J; j++) {
        params.gamma[j] = Eigen::MatrixXd::Zero(G, 1);
        params.theta[j] = Eigen::VectorXd::Zero(G);

        for (int g = 0; g < G; g++) {
            // Random initialization: 50% in slab
            params.gamma[j](g, 0) = (unif(rng_local) < 0.5) ? 1.0 : 0.0;

            // Small random effect if in slab
            if (params.gamma[j](g, 0) == 1.0) {
                std::normal_distribution<double> norm(0, 0.5);
                params.theta[j](g) = norm(rng_local);
            }
        }
    }

    // Initialize hyperparameters
    params.pi = 0.1;              // Prior: 30% of genes differential
    params.sigma_slab = 2.0;      // Slab standard deviation
    params.sigma_spike = 0.1;    // Spike standard deviation (very small)

    return params;
}

// ========================================
// Spike-and-Slab MCMC Update
// ========================================

MeanDispersionSpikeSlabResult mean_dispersion_spike_slab_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    SpikeSlabParams& spike_slab_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    MeanDispersionSpikeSlabResult result;

    // ============================================================
    // 1. Update spike-and-slab parameters
    // ============================================================

    double sigma_spike_sq = spike_slab_params.sigma_spike * spike_slab_params.sigma_spike;
    double sigma_slab_sq = spike_slab_params.sigma_slab * spike_slab_params.sigma_slab;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            // Observed log-deviation from baseline
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            // Skip if invalid
            if (mu_jg <= 0 || mu_bg <= 0 || !std::isfinite(mu_jg) || !std::isfinite(mu_bg)) {
                spike_slab_params.gamma[j](g, 0) = 0.0;
                spike_slab_params.theta[j](g) = 0.0;
                continue;
            }

            double delta_obs = std::log(mu_jg) - std::log(mu_bg);

            if (!std::isfinite(delta_obs)) {
                spike_slab_params.gamma[j](g, 0) = 0.0;
                spike_slab_params.theta[j](g) = 0.0;
                continue;
            }

            // ===== Update gamma (indicator variable) =====

            // Log-likelihood under spike (gamma = 0): delta ~ N(0, sigma_spike²)
            double log_lik_spike = -0.5 * std::log(2.0 * M_PI * sigma_spike_sq)
                                   - 0.5 * (delta_obs * delta_obs) / sigma_spike_sq;

            // Log-likelihood under slab (gamma = 1): delta ~ N(0, sigma_slab²)
            double log_lik_slab = -0.5 * std::log(2.0 * M_PI * sigma_slab_sq)
                                  - 0.5 * (delta_obs * delta_obs) / sigma_slab_sq;

            // Log-prior
            double log_prior_slab = std::log(spike_slab_params.pi);
            double log_prior_spike = std::log(1.0 - spike_slab_params.pi);

            // Log-posterior (unnormalized)
            double log_post_slab = log_lik_slab + log_prior_slab;
            double log_post_spike = log_lik_spike + log_prior_spike;

            // Normalize using log-sum-exp trick
            double max_log = std::max(log_post_slab, log_post_spike);
            double exp_slab = std::exp(log_post_slab - max_log);
            double exp_spike = std::exp(log_post_spike - max_log);

            double prob_slab = exp_slab / (exp_slab + exp_spike);

            // Handle numerical issues
            if (!std::isfinite(prob_slab)) {
                prob_slab = 0.5;
            }
            prob_slab = std::max(0.0, std::min(1.0, prob_slab));

            // Sample gamma
            std::uniform_real_distribution<double> unif(0, 1);
            double u = unif(rng_local);
            spike_slab_params.gamma[j](g, 0) = (u < prob_slab) ? 1.0 : 0.0;

            // ===== Update theta (effect size) given gamma =====


        }
    }

    // ============================================================
    // 2. Update pi (proportion of differential genes)
    // ============================================================

    double sum_gamma = 0.0;
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            sum_gamma += spike_slab_params.gamma[j](g, 0);
        }
    }

    // pi ~ Beta(a_pi + sum(gamma), b_pi + J*G - sum(gamma))
    double a_pi = 1.0;  // Weak prior
    double b_pi = 1.0;

    double alpha_beta = a_pi + sum_gamma;
    double beta_beta = b_pi + (J * G) - sum_gamma;

    // Sample from Beta using two Gamma samples
    std::gamma_distribution<double> gamma1(alpha_beta, 1.0);
    std::gamma_distribution<double> gamma2(beta_beta, 1.0);

    double g1 = gamma1(rng_local);
    double g2 = gamma2(rng_local);

    if (g1 + g2 > 0) {
        spike_slab_params.pi = g1 / (g1 + g2);
    } else {
        spike_slab_params.pi = 0.5;  // Fallback
    }

    // Constrain pi to reasonable range
    spike_slab_params.pi = std::max(0.01, std::min(0.99, spike_slab_params.pi));

    // ============================================================
    // 3. Update sigma_slab (slab variance)
    // ============================================================

    spike_slab_params.sigma_slab = 5.0;

    // sigma_spike stays fixed (very small)
    // spike_slab_params.sigma_spike = 0.01;

    // ============================================================
    // 4. Update dispersion regression parameters (same as before)
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
        result.spike_slab_params = spike_slab_params;
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

    Eigen::VectorXd b_hat = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    double shape = v_1 / 2.0;
    double scale = v_2 / 2.0;

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / scale);
    double gamma_sample = gamma_dist(rng_local);
    double alpha_phi_2 = 1.0 / gamma_sample;

    Eigen::MatrixXd cov_b = alpha_phi_2 * (X.transpose() * X).inverse();
    Eigen::VectorXd b_new = rmvnorm(b_hat, cov_b);

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.spike_slab_params = spike_slab_params;

    // ============================================================
    // 5. Calculate Log-Likelihood
    // ============================================================

    double log_lik_delta = 0.0;
    double const_part = -0.5 * std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0) continue;

            double delta = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta)) continue;

            // Likelihood depends on gamma
            double variance;
            if (spike_slab_params.gamma[j](g, 0) == 1.0) {
                variance = sigma_slab_sq;
            } else {
                variance = sigma_spike_sq;
            }

            log_lik_delta += const_part - 0.5 * std::log(variance)
                           - (delta * delta) / (2.0 * variance);
        }
    }

    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2)
                     - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    return result;
}
//
// Created by Azeezat Mustapha on 25.01.26.
//