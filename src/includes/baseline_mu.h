//
// Created by Azeezat Mustapha on 11.02.26.
//

#ifndef BASELINE_MU_H
#define BASELINE_MU_H

#include <RcppEigen.h>
#include <random>
#include "utils.h"

// Prior parameters for baseline
struct BaselinePrior {
    Eigen::VectorXd log_mean;    // Mean of log(baseline) - CHANGED
    Eigen::VectorXd log_sd;      // SD of log(baseline) - CHANGED
};

// Initialize baseline prior from data (empirical Bayes)
inline BaselinePrior initialize_baseline_prior(
    const Eigen::MatrixXd& mu_star_1_J,
    int J, int G
) {
    BaselinePrior prior;
    prior.log_mean = Eigen::VectorXd::Zero(G);
    prior.log_sd = Eigen::VectorXd::Ones(G);

    // For each gene, estimate prior from geometric mean across clusters
    for (int g = 0; g < G; ++g) {
        double log_sum = 0.0;
        double log_sq_sum = 0.0;
        int valid_count = 0;

        for (int j = 0; j < J; ++j) {
            double mu_jg = mu_star_1_J(j, g);
            if (mu_jg > 0 && std::isfinite(mu_jg)) {
                double log_mu = std::log(mu_jg);
                log_sum += log_mu;
                log_sq_sum += log_mu * log_mu;
                valid_count++;
            }
        }

        if (valid_count > 1) {
            // Mean and variance of log(mu) across clusters
            double log_mean_g = log_sum / valid_count;
            double log_var_g = (log_sq_sum / valid_count) - (log_mean_g * log_mean_g);

            // Use weak prior centered at observed log-mean
            prior.log_mean(g) = log_mean_g;
            prior.log_sd(g) = std::sqrt(log_var_g + 1.0);  // Add 1 for uncertainty

            // Bounds for stability
            prior.log_sd(g) = std::max(0.5, std::min(prior.log_sd(g), 3.0));
        } else {
            // Default weak prior
            prior.log_mean(g) = 0.0;  // exp(0) = 1
            prior.log_sd(g) = 2.0;
        }
    }

    return prior;
}

// Sample mu_baseline with LOGNORMAL prior
inline Eigen::VectorXd sample_mu_baseline(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& delta,
    const BaselinePrior& prior,
    int J, int G
) {
    Eigen::VectorXd mu_baseline(G);

    for (int g = 0; g < G; ++g) {
        // For each gene, sample log(mu_baseline_g)

        // Likelihood: log(mu_jg) = log(mu_baseline_g) + delta_jg
        // So: log(mu_baseline_g) = log(mu_jg) - delta_jg

        // Collect all observations of log(baseline) from clusters
        double sum_log_baseline = 0.0;
        int valid_count = 0;

        for (int j = 0; j < J; ++j) {
            double mu_jg = mu_star_1_J(j, g);
            double delta_jg = delta(j, g);

            if (mu_jg > 0 && std::isfinite(mu_jg) && std::isfinite(delta_jg)) {
                // Implied log(baseline) from this cluster
                double log_baseline_implied = std::log(mu_jg) - delta_jg;

                if (std::isfinite(log_baseline_implied)) {
                    sum_log_baseline += log_baseline_implied;
                    valid_count++;
                }
            }
        }

        if (valid_count == 0) {
            // No valid data - sample from prior
            std::normal_distribution<double> norm_dist(prior.log_mean(g), prior.log_sd(g));
            double log_baseline_sample = norm_dist(rng_local);
            mu_baseline(g) = std::exp(log_baseline_sample);
            continue;
        }

        // Posterior for log(mu_baseline_g)
        // Prior: log(baseline) ~ N(prior_log_mean, prior_log_sd²)
        // Likelihood: Each cluster gives observation of log(baseline)
        //             log(baseline) ~ N(log(mu_jg) - delta_jg, small variance)

        // For simplicity, use conjugate Normal posterior:
        // posterior_mean = weighted average of prior and data

        double prior_log_sd_g = prior.log_sd(g);
        double prior_precision = 1.0 / (prior_log_sd_g * prior_log_sd_g);
        double data_mean = sum_log_baseline / valid_count;
        double data_precision = valid_count / 1.0;  // Assume variance=1 for each observation

        double posterior_precision = prior_precision + data_precision;
        double posterior_mean = (prior_precision * prior.log_mean(g) +
                                data_precision * data_mean) / posterior_precision;
        double posterior_sd = 1.0 / std::sqrt(posterior_precision);

        // Sample log(mu_baseline_g) from posterior
        std::normal_distribution<double> norm_dist(posterior_mean, posterior_sd);
        double log_baseline_sample = norm_dist(rng_local);

        // Transform to original scale
        mu_baseline(g) = std::exp(log_baseline_sample);

        // Safety check
        if (!std::isfinite(mu_baseline(g)) || mu_baseline(g) <= 0) {
            // Fallback: use geometric mean
            double geom_mean = std::exp(data_mean);
            mu_baseline(g) = std::max(geom_mean, 0.01);
        }
    }

    return mu_baseline;
}

#endif // BASELINE_MU_H