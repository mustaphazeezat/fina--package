//
// Created by Azeezat Mustapha on 12.02.26.
//

#include "includes/marker_selection.h"
#include "includes/utils.h"
#include <cmath>

MarkerParams initialize_marker_params(int J, int G) {
    MarkerParams params;

    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; ++j) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }

    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);
    params.sigma_beta = 1.0;

    params.beta = Eigen::MatrixXd::Zero(J, G);
    params.marker_prob = Eigen::MatrixXd::Zero(J, G);
    params.gene_specificity = Eigen::VectorXd::Zero(G);

    return params;
}

void update_marker_selection(
    MarkerParams& marker_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_grand
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    // ========================================
    // 1. Compute log fold changes (β_jg)
    // ========================================
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            if (mu_star_1_J(j, g) > 0 && mu_grand(g) > 0) {
                marker_params.beta(j, g) = std::log(mu_star_1_J(j, g)) -
                                           std::log(mu_grand(g));
            } else {
                marker_params.beta(j, g) = 0.0;
            }
        }
    }

    // ========================================
    // 2. Update local shrinkage (λ_jg, ν_jg)
    // ========================================
    for (int j = 0; j < J; ++j) {
        double tau_j = marker_params.tau[j];
        double sigma_beta = marker_params.sigma_beta;

        for (int g = 0; g < G; ++g) {
            double beta_jg = marker_params.beta(j, g);
            double nu_jg = marker_params.nu[j](g);

            // Update λ_jg
            double rate_lambda = 1.0 / nu_jg +
                (beta_jg * beta_jg) / (2.0 * tau_j * tau_j * sigma_beta * sigma_beta);

            if (rate_lambda > 1e-6 && rate_lambda < 1e6) {
                std::gamma_distribution<double> gamma_dist(1.0, 1.0 / rate_lambda);
                double lambda_sq_inv = gamma_dist(rng_local);
                marker_params.lambda[j](g) = 1.0 / std::sqrt(lambda_sq_inv);
                marker_params.lambda[j](g) = std::max(0.01, std::min(100.0, marker_params.lambda[j](g)));
            }

            // Update ν_jg
            double lambda_jg = marker_params.lambda[j](g);
            double rate_nu = 1.0 + 1.0 / (lambda_jg * lambda_jg);

            std::gamma_distribution<double> gamma_dist_nu(1.0, 1.0 / rate_nu);
            marker_params.nu[j](g) = 1.0 / gamma_dist_nu(rng_local);
        }
    }

    // ========================================
    // 3. Update global shrinkage (τ_j, ξ_j)
    // ========================================
    for (int j = 0; j < J; ++j) {
        double sum_term = 0.0;
        int valid_count = 0;

        for (int g = 0; g < G; ++g) {
            double beta_jg = marker_params.beta(j, g);
            double lambda_jg = marker_params.lambda[j](g);

            if (std::isfinite(beta_jg) && lambda_jg > 0) {
                sum_term += (beta_jg * beta_jg) / (lambda_jg * lambda_jg);
                valid_count++;
            }
        }

        if (valid_count > 0) {
            double sigma_beta_sq = marker_params.sigma_beta * marker_params.sigma_beta;
            double xi_j = marker_params.xi[j];

            double shape_tau = valid_count / 2.0;
            double rate_tau = 1.0 / xi_j + sum_term / (2.0 * sigma_beta_sq);

            if (rate_tau > 1e-6 && rate_tau < 1e6) {
                std::gamma_distribution<double> gamma_dist(shape_tau, 1.0 / rate_tau);
                double tau_sq_inv = gamma_dist(rng_local);
                marker_params.tau[j] = 1.0 / std::sqrt(tau_sq_inv);
                marker_params.tau[j] = std::max(0.01, std::min(10.0, marker_params.tau[j]));
            }

            // Update ξ_j
            double tau_sq = marker_params.tau[j] * marker_params.tau[j];
            double rate_xi = 1.0 + 1.0 / tau_sq;

            std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
            marker_params.xi[j] = 1.0 / gamma_dist_xi(rng_local);
        }
    }

    // ========================================
    // 4. Update σ_β (variance of marker effects)
    // ========================================
    double sum_all = 0.0;
    int total_valid = 0;

    for (int j = 0; j < J; ++j) {
        double tau_sq = marker_params.tau[j] * marker_params.tau[j];

        for (int g = 0; g < G; ++g) {
            double beta_jg = marker_params.beta(j, g);
            double lambda_sq = marker_params.lambda[j](g) * marker_params.lambda[j](g);

            if (std::isfinite(beta_jg)) {
                sum_all += (beta_jg * beta_jg) / (tau_sq * lambda_sq);
                total_valid++;
            }
        }
    }

    if (total_valid > 0) {
        double a_sigma = 2.0;
        double b_sigma = 1.0;

        double shape = a_sigma + total_valid / 2.0;
        double rate = b_sigma + sum_all / 2.0;

        std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
        double sigma_sq_inv = gamma_dist(rng_local);
        marker_params.sigma_beta = 1.0 / std::sqrt(sigma_sq_inv);
        marker_params.sigma_beta = std::max(0.1, std::min(5.0, marker_params.sigma_beta));
    }

    // ========================================
    // 5. Compute marker probabilities
    // ========================================
    marker_params.marker_prob = compute_marker_probabilities(marker_params);

    // ========================================
    // 6. Compute gene specificity scores
    // ========================================
    for (int g = 0; g < G; ++g) {
        double max_prob = 0.0;
        for (int j = 0; j < J; ++j) {
            max_prob = std::max(max_prob, marker_params.marker_prob(j, g));
        }
        marker_params.gene_specificity(g) = max_prob;
    }
}

Eigen::MatrixXd compute_marker_probabilities(const MarkerParams& marker_params) {
    int J = marker_params.beta.rows();
    int G = marker_params.beta.cols();

    Eigen::MatrixXd probs(J, G);

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            // Posterior shrinkage factor: κ = 1 / (1 + λ²τ²)
            // High κ (near 1) = strong shrinkage = NOT a marker
            // Low κ (near 0) = weak shrinkage = IS a marker

            double lambda_sq = marker_params.lambda[j](g) * marker_params.lambda[j](g);
            double tau_sq = marker_params.tau[j] * marker_params.tau[j];

            double kappa = 1.0 / (1.0 + lambda_sq * tau_sq);

            // Marker probability = 1 - shrinkage
            probs(j, g) = 1.0 - kappa;

            // Alternative: use effect size magnitude
            // double effect_size = std::abs(marker_params.beta(j, g));
            // probs(j, g) = 1.0 / (1.0 + std::exp(-2.0 * effect_size));  // Sigmoid
        }
    }

    return probs;
}