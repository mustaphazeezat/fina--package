#include "includes/unique_parameters.h"
#include "includes/utils.h"
#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <thread>
#include <random>
#include <vector>

// ============================================================
// Thread-safe utility functions
// ============================================================

inline Eigen::VectorXd rmvnorm_safe(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, std::mt19937& rng) {
    int n = mean.size();

    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        Eigen::VectorXd z(n);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            z(i) = dist(rng) * std::sqrt(std::max(cov(i, i), 1e-10));
        }
        return mean + z;
    }

    Eigen::MatrixXd L = llt.matrixL();
    Eigen::VectorXd z(n);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        z(i) = dist(rng);
    }

    return mean + L * z;
}

inline double rlnorm_safe(double meanlog, double sdlog, std::mt19937& rng) {
    std::lognormal_distribution<double> dist(meanlog, sdlog);
    return dist(rng);
}

// ============================================================
// Cluster cache
// ============================================================

struct ClusterCache {
    std::vector<bool> is_nonempty;
    std::vector<std::vector<std::pair<int, int>>> cell_indices;

    ClusterCache(int J, const std::vector<std::vector<int>>& Z)
        : is_nonempty(J, false), cell_indices(J) {
        int D = Z.size();
        for (int d = 0; d < D; ++d) {
            for (size_t c = 0; c < Z[d].size(); ++c) {
                int j = Z[d][c];
                if (j >= 0 && j < J) {
                    is_nonempty[j] = true;
                    cell_indices[j].push_back({d, c});
                }
            }
        }
    }
};

// ============================================================
// Negative Binomial log-likelihood (shared)
// ============================================================

inline double compute_nb_log_likelihood(
    double mu_star,
    double phi_star,
    const ClusterCache& cache,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g
) {
    if (!cache.is_nonempty[j]) {
        return 0.0;
    }

    double log_lik = 0.0;

    for (const auto& dc : cache.cell_indices[j]) {
        int d = dc.first;
        int c = dc.second;

        double y = Y[d](g, c);
        double beta = Beta[d](c);
        double mu_beta = mu_star * beta;
        double denom = mu_beta + phi_star;

        log_lik += phi_star * std::log(phi_star / denom) +
                   std::lgamma(y + phi_star) - std::lgamma(phi_star) - std::lgamma(y + 1.0) +
                   y * std::log(mu_beta / denom);
    }

    return log_lik;
}

// ============================================================
// PRIOR 1: Lognormal (ORIGINAL - models log(μ*) directly)
// log(μ*) ~ N(0, α²_μ)
// log(φ*) ~ N(b₀ + b₁ log(μ*), α²_φ)
// ============================================================

double unique_parameters_log_prob(
    double mu_star,
    double phi_star,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    double alpha_mu_2,
    bool quadratic
) {
    double log_mu = std::log(mu_star);
    double log_phi = std::log(phi_star);

    double lprod1;
    if (quadratic) {
        double log_mu_sq = log_mu * log_mu;
        lprod1 = -log_mu - log_phi -
                 (0.5 / alpha_mu_2) * log_mu_sq -
                 std::pow(log_phi - (b(0) + b(1) * log_mu + b(2) * log_mu_sq), 2) / (2.0 * alpha_phi_2);
    } else {
        lprod1 = -log_mu - log_phi -
                 (0.5 / alpha_mu_2) * log_mu * log_mu -
                 std::pow(log_phi - (b(0) + b(1) * log_mu), 2) / (2.0 * alpha_phi_2);
    }

    // Likelihood (non-cached version)
    int D = Y.size();
    bool has_cells = false;
    for (int d = 0; d < D && !has_cells; ++d) {
        for (size_t c = 0; c < Z[d].size() && !has_cells; ++c) {
            if (Z[d][c] == j) {
                has_cells = true;
            }
        }
    }

    if (!has_cells) {
        return lprod1;
    }

    double lprod2 = 0.0;
    for (int d = 0; d < D; ++d) {
        for (size_t c = 0; c < Z[d].size(); ++c) {
            if (Z[d][c] == j) {
                double y = Y[d](g, c);
                double beta = Beta[d](c);
                double mu_beta = mu_star * beta;
                double denom = mu_beta + phi_star;

                lprod2 += phi_star * std::log(phi_star / denom) +
                          std::lgamma(y + phi_star) - std::lgamma(phi_star) - std::lgamma(y + 1.0) +
                          y * std::log(mu_beta / denom);
            }
        }
    }

    return lprod1 + lprod2;
}

// Cached version
double unique_parameters_log_prob_fast(
    double mu_star,
    double phi_star,
    const ClusterCache& cache,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    double alpha_mu_2,
    bool quadratic
) {
    double log_mu = std::log(mu_star);
    double log_phi = std::log(phi_star);

    double lprod1;
    if (quadratic) {
        double log_mu_sq = log_mu * log_mu;
        lprod1 = -log_mu - log_phi -
                 (0.5 / alpha_mu_2) * log_mu_sq -
                 std::pow(log_phi - (b(0) + b(1) * log_mu + b(2) * log_mu_sq), 2) / (2.0 * alpha_phi_2);
    } else {
        lprod1 = -log_mu - log_phi -
                 (0.5 / alpha_mu_2) * log_mu * log_mu -
                 std::pow(log_phi - (b(0) + b(1) * log_mu), 2) / (2.0 * alpha_phi_2);
    }

    double lprod2 = compute_nb_log_likelihood(mu_star, phi_star, cache, Y, Beta, j, g);

    return lprod1 + lprod2;
}

// ============================================================
// PRIOR 2: Horseshoe (models DEVIATIONS from baseline)
// δ_{jg} = log(μ*) - log(μ_baseline) ~ N(0, σ²_μ τ²_j λ²_{jg})
// log(φ*) ~ N(b₀ + b₁ log(μ*), α²_φ)
// ============================================================

double unique_parameters_log_prob_horseshoe_fast(
    double mu_star,
    double phi_star,
    const ClusterCache& cache,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const HorseshoeParams& horseshoe_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic
) {
    double log_mu = std::log(mu_star);
    double log_phi = std::log(phi_star);

    // Horseshoe prior on μ: models DEVIATION from baseline
    double delta_jg = log_mu - std::log(mu_baseline(g));
    double tau_j = horseshoe_params.tau[j];
    double lambda_jg = horseshoe_params.lambda[j](g);
    double sigma_mu = horseshoe_params.sigma_mu;
    double prior_var_mu = sigma_mu * sigma_mu * tau_j * tau_j * lambda_jg * lambda_jg;

    if (prior_var_mu < 1e-20) prior_var_mu = 1e-20;

    double lprod1 = -log_mu - log_phi;
    lprod1 += -0.5 * (delta_jg * delta_jg) / prior_var_mu;

    // Prior on φ
    double mean_log_phi;
    if (quadratic) {
        mean_log_phi = b(0) + b(1) * log_mu + b(2) * log_mu * log_mu;
    } else {
        mean_log_phi = b(0) + b(1) * log_mu;
    }
    lprod1 += -0.5 * std::pow(log_phi - mean_log_phi, 2) / alpha_phi_2;

    double lprod2 = compute_nb_log_likelihood(mu_star, phi_star, cache, Y, Beta, j, g);

    return lprod1 + lprod2;
}

// ============================================================
// PRIOR 3: Regularized Horseshoe (models DEVIATIONS from baseline)
// δ_{jg} ~ N(0, σ²_μ τ²_j λ̃²_{jg})
// where λ̃²_{jg} = (c² λ²_{jg}) / (c² + τ²_j λ²_{jg})
// ============================================================

double unique_parameters_log_prob_reg_horseshoe_fast(
    double mu_star,
    double phi_star,
    const ClusterCache& cache,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const RegularizedHorseshoeParams& rhs_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic
) {
    double log_mu = std::log(mu_star);
    double log_phi = std::log(phi_star);

    // Regularized horseshoe: models DEVIATION from baseline
    double delta_jg = log_mu - std::log(mu_baseline(g));
    double tau_j = rhs_params.tau[j];
    double lambda_jg = rhs_params.lambda[j](g);
    double sigma_mu = rhs_params.sigma_mu;
    double c_squared = rhs_params.c_squared;

    double tau_sq = tau_j * tau_j;
    double lambda_sq = lambda_jg * lambda_jg;
    double lambda_tilde_sq = (c_squared * lambda_sq) / (c_squared + tau_sq * lambda_sq);

    double prior_var_mu = sigma_mu * sigma_mu * tau_sq * lambda_tilde_sq;
    if (prior_var_mu < 1e-20) prior_var_mu = 1e-20;

    double lprod1 = -log_mu - log_phi;
    lprod1 += -0.5 * (delta_jg * delta_jg) / prior_var_mu;

    // Prior on φ
    double mean_log_phi;
    if (quadratic) {
        mean_log_phi = b(0) + b(1) * log_mu + b(2) * log_mu * log_mu;
    } else {
        mean_log_phi = b(0) + b(1) * log_mu;
    }
    lprod1 += -0.5 * std::pow(log_phi - mean_log_phi, 2) / alpha_phi_2;

    double lprod2 = compute_nb_log_likelihood(mu_star, phi_star, cache, Y, Beta, j, g);

    return lprod1 + lprod2;
}
// ============================================================
// PRIOR 4: Spike-and-Slab (models DEVIATIONS from baseline)
// δ_{jg} ~ γ_{jg} * N(0, σ²_slab) + (1-γ_{jg}) * N(0, σ²_spike)
// ============================================================

double unique_parameters_log_prob_spike_slab_fast(
    double mu_star,
    double phi_star,
    const ClusterCache& cache,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const SpikeSlabParams& spike_slab_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic
) {
    double log_mu = std::log(mu_star);
    double log_phi = std::log(phi_star);

    // Spike-and-slab prior on μ: models DEVIATION from baseline
    double delta_jg = log_mu - std::log(mu_baseline(g));

    // Get gamma (indicator: 1=slab, 0=spike)
    double gamma_jg = spike_slab_params.gamma[j](g, 0);

    // Variance depends on spike vs slab
    double prior_var_mu;
    if (gamma_jg > 0.5) {  // In slab (differential gene)
        double sigma_slab = spike_slab_params.sigma_slab;
        prior_var_mu = sigma_slab * sigma_slab;
    } else {  // In spike (non-differential gene)
        double sigma_spike = spike_slab_params.sigma_spike;
        prior_var_mu = sigma_spike * sigma_spike;
    }

    // Safety check
    if (prior_var_mu < 1e-20) prior_var_mu = 1e-20;

    // Jacobian and prior on log(μ*)
    double lprod1 = -log_mu - log_phi;
    lprod1 += -0.5 * (delta_jg * delta_jg) / prior_var_mu;

    // Prior on log(φ*)
    double mean_log_phi;
    if (quadratic) {
        mean_log_phi = b(0) + b(1) * log_mu + b(2) * log_mu * log_mu;
    } else {
        mean_log_phi = b(0) + b(1) * log_mu;
    }
    lprod1 += -0.5 * std::pow(log_phi - mean_log_phi, 2) / alpha_phi_2;

    // Likelihood
    double lprod2 = compute_nb_log_likelihood(mu_star, phi_star, cache, Y, Beta, j, g);

    return lprod1 + lprod2;
}

// ============================================================
// Helper struct
// ============================================================

struct JGUpdate {
    double mu_star_new;
    double phi_star_new;
    Eigen::Vector2d X_mu_phi_star_new;
    int accept_count;
};

// ============================================================
// MAIN MCMC FUNCTION
// ============================================================

UniqueParamsResult unique_parameters_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const std::vector<std::vector<Eigen::RowVectorXd>>& mean_X_mu_phi,
    const std::vector<std::vector<Eigen::MatrixXd>>& tilde_s_mu_phi,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::VectorXd>& Beta,
    double alpha_mu_2,
    const std::vector<std::vector<Eigen::MatrixXd>>& covariance,
    int iter_num,
    bool quadratic,
    const std::vector<Eigen::MatrixXd>& Y,
    double adaptive_prop,
    int num_cores,
    MuPriorType prior_type,
    const HorseshoeParams* horseshoe_params,
    const RegularizedHorseshoeParams* reg_horseshoe_params,
	const SpikeSlabParams* spike_slab_params,
    const Eigen::VectorXd* mu_baseline
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();
    int n = iter_num;

    // Build cluster cache
    ClusterCache cache(J, Z);

    // Initialize result matrices
    Eigen::MatrixXd mu_star_1_J_new = mu_star_1_J;
    Eigen::MatrixXd phi_star_1_J_new = phi_star_1_J;

    auto covariance_new = covariance;
    auto tilde_s_mu_phi_new = tilde_s_mu_phi;
    auto mean_X_mu_phi_new = mean_X_mu_phi;

    int accept_count_tot = 0;

    // Determine actual number of cores
    int actual_cores = num_cores;
    if (actual_cores <= 0) {
        actual_cores = std::thread::hardware_concurrency();
    }
    actual_cores = std::max(1, actual_cores);

    // Create (j,g) pairs - prioritize non-empty clusters
    std::vector<std::pair<int, int>> jg_pairs;
    jg_pairs.reserve(J * G);

    for (int j = 0; j < J; ++j) {
        if (cache.is_nonempty[j]) {
            for (int g = 0; g < G; ++g) {
                jg_pairs.push_back({j, g});
            }
        }
    }
    for (int j = 0; j < J; ++j) {
        if (!cache.is_nonempty[j]) {
            for (int g = 0; g < G; ++g) {
                jg_pairs.push_back({j, g});
            }
        }
    }

    int total_pairs = jg_pairs.size();
    std::vector<JGUpdate> all_results(total_pairs);

    // Thread configuration
    int pairs_per_thread = std::max(100, total_pairs / actual_cores);
    int num_batches = (total_pairs + pairs_per_thread - 1) / pairs_per_thread;

    std::vector<std::thread> threads;
    threads.reserve(std::min(actual_cores, num_batches));

    for (int batch = 0; batch < num_batches; ++batch) {
        int start = batch * pairs_per_thread;
        int end = std::min(start + pairs_per_thread, total_pairs);

        if (start >= total_pairs) break;

        threads.emplace_back([=, &all_results, &jg_pairs, &cache,
                              &b, &Y, &Beta, &covariance,
                              &mu_star_1_J, &phi_star_1_J]() {

            std::mt19937 local_rng(std::random_device{}() + batch * 1000);
            std::uniform_real_distribution<double> unif(0.0, 1.0);

            const bool use_adaptive = (n > 100);
            const double prop_scale = (2.4 * 2.4) / 2.0;

            for (int i = start; i < end; ++i) {
                int j = jg_pairs[i].first;
                int g = jg_pairs[i].second;

                double mu_star_old = mu_star_1_J(j, g);
                double phi_star_old = phi_star_1_J(j, g);

                double mu_star_new_val, phi_star_new_val;
                Eigen::Vector2d X_mu_phi_star_new;

                // ============================================================
                // CASE 1: Empty cluster - sample from prior
                // ============================================================
                if (!cache.is_nonempty[j]) {
                    switch (prior_type) {
                        case MuPriorType::HORSESHOE: {
                            // Sample from horseshoe: baseline + deviation
                            double tau_j = horseshoe_params->tau[j];
                            double lambda_jg = horseshoe_params->lambda[j](g);
                            double sigma_mu = horseshoe_params->sigma_mu;
                            double prior_sd = sigma_mu * tau_j * lambda_jg;

                            std::normal_distribution<double> norm_dist(0.0, prior_sd);
                            double delta = norm_dist(local_rng);
                            double log_mu_new = std::log((*mu_baseline)(g)) + delta;
                            mu_star_new_val = std::exp(log_mu_new);

                            double mean_log_phi = b(0) + b(1) * log_mu_new;
                            if (quadratic && b.size() > 2) {
                                mean_log_phi += b(2) * log_mu_new * log_mu_new;
                            }
                            phi_star_new_val = rlnorm_safe(mean_log_phi, std::sqrt(alpha_phi_2), local_rng);
                            break;
                        }
                        case MuPriorType::REGULARIZED_HORSESHOE: {
                            // Sample from regularized horseshoe: baseline + deviation
                            double tau_j = reg_horseshoe_params->tau[j];
                            double lambda_jg = reg_horseshoe_params->lambda[j](g);
                            double sigma_mu = reg_horseshoe_params->sigma_mu;
                            double c_squared = reg_horseshoe_params->c_squared;

                            double tau_sq = tau_j * tau_j;
                            double lambda_sq = lambda_jg * lambda_jg;
                            double lambda_tilde_sq = (c_squared * lambda_sq) / (c_squared + tau_sq * lambda_sq);
                            double prior_sd = sigma_mu * tau_j * std::sqrt(lambda_tilde_sq);

                            std::normal_distribution<double> norm_dist(0.0, prior_sd);
                            double delta = norm_dist(local_rng);
                            double log_mu_new = std::log((*mu_baseline)(g)) + delta;
                            mu_star_new_val = std::exp(log_mu_new);

                            double mean_log_phi = b(0) + b(1) * log_mu_new;
                            if (quadratic && b.size() > 2) {
                                mean_log_phi += b(2) * log_mu_new * log_mu_new;
                            }
                            phi_star_new_val = rlnorm_safe(mean_log_phi, std::sqrt(alpha_phi_2), local_rng);
                            break;
                        }
						case MuPriorType::SPIKE_SLAB: {
            				// Sample from spike-and-slab prior
            				double gamma_jg = spike_slab_params->gamma[j](g, 0);
            				double prior_sd;

            				// Use slab or spike variance
            				if (gamma_jg > 0.5) {  // In slab (differential)
                				prior_sd = spike_slab_params->sigma_slab;
            				} else {  // In spike (non-differential)
                				prior_sd = spike_slab_params->sigma_spike;
            				}

            					// Sample deviation from baseline
            					std::normal_distribution<double> norm_dist(0.0, prior_sd);
            					double delta = norm_dist(local_rng);
            					double log_mu_new = std::log((*mu_baseline)(g)) + delta;
            					mu_star_new_val = std::exp(log_mu_new);

            						// Sample phi given mu
            					double mean_log_phi = b(0) + b(1) * log_mu_new;
            					if (quadratic && b.size() > 2) {
                					mean_log_phi += b(2) * log_mu_new * log_mu_new;
            					}
            					phi_star_new_val = rlnorm_safe(mean_log_phi, std::sqrt(alpha_phi_2), local_rng);
            					break;
        				}
                        case MuPriorType::LOGNORMAL:
                        default: {
                            // Sample from lognormal: ABSOLUTE log-expression (ORIGINAL behavior)
                            mu_star_new_val = rlnorm_safe(0.0, std::sqrt(alpha_mu_2), local_rng);
                            phi_star_new_val = rlnorm_safe(b(0) + b(1) * std::log(mu_star_new_val),
                                                           std::sqrt(alpha_phi_2), local_rng);
                            break;
                        }
                    }

                    X_mu_phi_star_new << std::log(mu_star_new_val), std::log(phi_star_new_val);
                    all_results[i] = {mu_star_new_val, phi_star_new_val, X_mu_phi_star_new, 1};
                    continue;
                }

                // ============================================================
                // CASE 2: Non-empty cluster - MH update
                // ============================================================
                Eigen::Vector2d X_old;
                X_old << std::log(mu_star_old), std::log(phi_star_old);

                // Propose new values
                if (use_adaptive) {
                    Eigen::Matrix2d cov = prop_scale *
                        (covariance[j][g] + adaptive_prop * Eigen::Matrix2d::Identity());
                    X_mu_phi_star_new = rmvnorm_safe(X_old, cov, local_rng);
                } else {
                    Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
                    X_mu_phi_star_new = rmvnorm_safe(X_old, cov, local_rng);
                }

                mu_star_new_val = std::exp(X_mu_phi_star_new(0));
                phi_star_new_val = std::exp(X_mu_phi_star_new(1));

                // Early rejection for invalid values
                if (X_mu_phi_star_new(0) > 10.0 || !std::isfinite(mu_star_new_val) ||
                    !std::isfinite(phi_star_new_val)) {
                    all_results[i] = {mu_star_old, phi_star_old, X_old, 0};
                    continue;
                }

                // ============================================================
                // Compute log acceptance ratio based on prior type
                // ============================================================
                double log_prob_new, log_prob_old;

                switch (prior_type) {
                    case MuPriorType::HORSESHOE:
                        log_prob_new = unique_parameters_log_prob_horseshoe_fast(
                            mu_star_new_val, phi_star_new_val, cache,
                            b, alpha_phi_2, Y, Beta, j, g,
                            *horseshoe_params, *mu_baseline, quadratic
                        );
                        log_prob_old = unique_parameters_log_prob_horseshoe_fast(
                            mu_star_old, phi_star_old, cache,
                            b, alpha_phi_2, Y, Beta, j, g,
                            *horseshoe_params, *mu_baseline, quadratic
                        );
                        break;

                    case MuPriorType::REGULARIZED_HORSESHOE:
                        log_prob_new = unique_parameters_log_prob_reg_horseshoe_fast(
                            mu_star_new_val, phi_star_new_val, cache,
                            b, alpha_phi_2, Y, Beta, j, g,
                            *reg_horseshoe_params, *mu_baseline, quadratic
                        );
                        log_prob_old = unique_parameters_log_prob_reg_horseshoe_fast(
                            mu_star_old, phi_star_old, cache,
                            b, alpha_phi_2, Y, Beta, j, g,
                            *reg_horseshoe_params, *mu_baseline, quadratic
                        );
                        break;
					case MuPriorType::SPIKE_SLAB:  // ← ADD THIS ENTIRE CASE
        				log_prob_new = unique_parameters_log_prob_spike_slab_fast(
            				mu_star_new_val, phi_star_new_val, cache,
            				b, alpha_phi_2, Y, Beta, j, g,
            				*spike_slab_params, *mu_baseline, quadratic
        				);
        				log_prob_old = unique_parameters_log_prob_spike_slab_fast(
            				mu_star_old, phi_star_old, cache,
            				b, alpha_phi_2, Y, Beta, j, g,
            				*spike_slab_params, *mu_baseline, quadratic
        				);
        				break;
                    case MuPriorType::LOGNORMAL:
                    default:
                        log_prob_new = unique_parameters_log_prob_fast(
                            mu_star_new_val, phi_star_new_val, cache,
                            b, alpha_phi_2, Y, Beta, j, g, alpha_mu_2, quadratic
                        );
                        log_prob_old = unique_parameters_log_prob_fast(
                            mu_star_old, phi_star_old, cache,
                            b, alpha_phi_2, Y, Beta, j, g, alpha_mu_2, quadratic
                        );
                        break;
                }

                // Acceptance ratio (includes Jacobian correction)
                double log_accept = log_prob_new - log_prob_old +
                                   X_mu_phi_star_new(0) + X_mu_phi_star_new(1) - X_old(0) - X_old(1);

                int accept = 0;
                if (std::isfinite(log_accept) && unif(local_rng) < std::min(1.0, std::exp(log_accept))) {
                    accept = 1;
                } else {
                    X_mu_phi_star_new = X_old;
                    mu_star_new_val = mu_star_old;
                    phi_star_new_val = phi_star_old;
                }

                all_results[i] = {mu_star_new_val, phi_star_new_val, X_mu_phi_star_new, accept};
            }
        });

        if ((int)threads.size() >= actual_cores) {
            for (auto& t : threads) {
                if (t.joinable()) t.join();
            }
            threads.clear();
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    // Collect results
    for (int i = 0; i < total_pairs; ++i) {
        int j = jg_pairs[i].first;
        int g = jg_pairs[i].second;

        mu_star_1_J_new(j, g) = all_results[i].mu_star_new;
        phi_star_1_J_new(j, g) = all_results[i].phi_star_new;

        Eigen::Vector2d X_new = all_results[i].X_mu_phi_star_new;

        tilde_s_mu_phi_new[j][g] = tilde_s_mu_phi[j][g] + X_new * X_new.transpose();
        mean_X_mu_phi_new[j][g] = mean_X_mu_phi[j][g] * (1.0 - 1.0 / n) +
                                  (1.0 / n) * X_new.transpose();
        covariance_new[j][g] = (1.0 / (n - 1)) * tilde_s_mu_phi_new[j][g] -
                               (n / (double)(n - 1)) * mean_X_mu_phi_new[j][g].transpose() *
                               mean_X_mu_phi_new[j][g];

        accept_count_tot += all_results[i].accept_count;
    }

    UniqueParamsResult result;
    result.mu_star_1_J_new = mu_star_1_J_new;
    result.phi_star_1_J_new = phi_star_1_J_new;
    result.accept_count = accept_count_tot;
    result.tilde_s_mu_phi_new = tilde_s_mu_phi_new;
    result.mean_X_mu_phi_new = mean_X_mu_phi_new;
    result.covariance_new = covariance_new;

    return result;
}