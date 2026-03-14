#include "includes/normHDP_mcmc.h"
#include "includes/allocation_z.h"
#include "includes/dataset_specific.h"
#include "includes/component_probabilities.h"
#include "includes/alpha.h"
#include "includes/alpha_zero.h"
#include "includes/unique_parameters.h"
#include "includes/capture_efficiencies.h"
#include "includes/mean_dispersion.h"
#include "includes/mean_dispersion_horseshoe.h"
#include "includes/mean_dispersion_spike_slab.h"
#include "includes/mean_dispersion_regularized_horseshoe.h"
#include "includes/baseline_mu.h"
#include "includes/utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>

NormHDPResult normHDP_mcmc(
    const std::vector<Eigen::MatrixXd>& Y,
    int J,
    int number_iter,
    int thinning,
    bool empirical,
    int burn_in,
    bool quadratic,
    int iter_update,
    double beta_mean,
    double alpha_mu_2,
    double adaptive_prop,
    bool print_Z,
    int num_cores,
    bool save_only_z,
    bool use_sparse_prior,
	bool use_spike_slab,
	bool use_reg_horseshoe,
    double p_0,
    const Eigen::VectorXd& baynorm_mu_estimate,
    const Eigen::VectorXd& baynorm_phi_estimate,
    const std::vector<Eigen::VectorXd>& baynorm_beta
) {
    // ============ Dimensions ============
    int D = Y.size();
    int G = Y[0].rows();
    std::vector<int> C(D);
    for (int d = 0; d < D; ++d) {
        C[d] = Y[d].cols();
    }

    // ============ BayNorm Estimates (passed from R) ============
    Eigen::VectorXd mu_estimate = baynorm_mu_estimate;
    Eigen::VectorXd phi_estimate = baynorm_phi_estimate;
    std::vector<Eigen::VectorXd> Beta_estimate = baynorm_beta;

    // Replace zeros with small values
    for (int g = 0; g < G; ++g) {
        if (mu_estimate(g) == 0) mu_estimate(g) = 0.01;
        if (phi_estimate(g) == 0) phi_estimate(g) = 0.01;
    }


     // Hyperparameters for σ²_η prior
     double a_eta = 2.0;
     double b_eta = 1.0;

     // Proposal SD for η_g MH updates (tune later)
     double eta_prop_sd = 0.1;

    // ============ Hyper-parameters ============

    // alpha_mu_2
    if (alpha_mu_2 < 0) {
    std::vector<double> log_mu_vals;

    for (int g = 0; g < G; ++g) {
        if (mu_estimate(g) > 0 && std::isfinite(std::log(mu_estimate(g)))) {
            log_mu_vals.push_back(std::log(mu_estimate(g)));
        }
    }

    if (log_mu_vals.size() > 1) {
        // Compute mean
        double mean_log_mu = 0.0;
        for (double lm : log_mu_vals) mean_log_mu += lm;
        mean_log_mu /= log_mu_vals.size();

        // Compute variance
        double var_log_mu = 0.0;
        for (double lm : log_mu_vals) {
            var_log_mu += (lm - mean_log_mu) * (lm - mean_log_mu);
        }
        var_log_mu /= (log_mu_vals.size() - 1);

        // Set alpha_mu_2 to empirical variance (or scaled version)
        alpha_mu_2 = var_log_mu;

    } else {
        alpha_mu_2 = 4.0;
    }
}

    // a_d_beta, b_d_beta
    Eigen::VectorXd a_d_beta(D);
    Eigen::VectorXd b_d_beta(D);

    for (int d = 0; d < D; ++d) {
        double baynorm_mean_capeff = beta_mean;
        double baynorm_var_capeff = 0.5;
        double a_beta, b_beta;

        do {
            a_beta = ((1.0 - baynorm_mean_capeff) / baynorm_var_capeff -
                     1.0 / baynorm_mean_capeff) * baynorm_mean_capeff * baynorm_mean_capeff;
            b_beta = a_beta * (1.0 / baynorm_mean_capeff - 1.0);

            if (baynorm_var_capeff >= baynorm_mean_capeff * (1.0 - baynorm_mean_capeff) ||
                a_beta < 1.0 || b_beta < 1.0) {
                baynorm_var_capeff /= 2.0;
            } else {
                break;
            }
        } while (true);

        a_d_beta(d) = a_beta;
        b_d_beta(d) = b_beta;
    }

    // Regression parameters v_1, v_2, m_b
    double v_1, v_2;
    Eigen::VectorXd m_b;

    // Fit linear model (simplified - could use weighted least squares)
    std::vector<double> x_vals, y_vals;
    for (int g = 0; g < G; ++g) {
        if (std::isfinite(std::log(mu_estimate(g))) &&
            std::isfinite(std::log(phi_estimate(g)))) {
            x_vals.push_back(std::log(mu_estimate(g)));
            y_vals.push_back(std::log(phi_estimate(g)));
        }
    }

    int n_valid = x_vals.size();
    Eigen::VectorXd y_vec(n_valid);
    Eigen::MatrixXd X_mat;

    for (int i = 0; i < n_valid; ++i) {
        y_vec(i) = y_vals[i];
    }

    if (quadratic) {
        X_mat.resize(n_valid, 3);
        for (int i = 0; i < n_valid; ++i) {
            X_mat(i, 0) = 1.0;
            X_mat(i, 1) = x_vals[i];
            X_mat(i, 2) = x_vals[i] * x_vals[i];
        }
        m_b.resize(3);
        m_b << -1.0, 2.0, 0.0;
    } else {
        X_mat.resize(n_valid, 2);
        for (int i = 0; i < n_valid; ++i) {
            X_mat(i, 0) = 1.0;
            X_mat(i, 1) = x_vals[i];
        }
        m_b.resize(2);
        m_b << -1.0, 2.0;
    }

    Eigen::VectorXd coef = (X_mat.transpose() * X_mat).ldlt().solve(X_mat.transpose() * y_vec);
    Eigen::VectorXd resid = y_vec - X_mat * coef;
    double rse_squared = resid.squaredNorm() / (n_valid - coef.size());

    if (empirical) {
        m_b = coef;
        double variance = 5.0;
        v_1 = rse_squared * rse_squared / variance + 2.0;
        v_2 = (v_1 - 1.0) * rse_squared;
    } else {
        v_1 = 2.0;
        v_2 = 1.0;
    }

    // ============ Initial Values ============
    Eigen::VectorXd b_initial = m_b;
    double alpha_phi_2_initial = rse_squared;

    // Random allocation
    std::vector<std::vector<int>> Z_initial(D);
    for (int d = 0; d < D; ++d) {
        Z_initial[d].resize(C[d]);
        std::uniform_int_distribution<int> dist(0, J - 1);
        for (int c = 0; c < C[d]; ++c) {
            Z_initial[d][c] = dist(rng_local);
        }
    }

    Eigen::MatrixXd P_J_D_initial = Eigen::MatrixXd::Constant(J, D, 1.0 / J);
    Eigen::VectorXd P_initial = Eigen::VectorXd::Constant(J, 1.0 / J);
    double alpha_initial = 1.0;
    double alpha_zero_initial = 1.0;

    Eigen::MatrixXd mu_star_1_J_initial(J, G);
    Eigen::MatrixXd phi_star_1_J_initial(J, G);
    for (int j = 0; j < J; ++j) {
        mu_star_1_J_initial.row(j) = mu_estimate.transpose();
        phi_star_1_J_initial.row(j) = phi_estimate.transpose();
    }
    // ============ Compute Baseline Expression (for Horseshoe) ============
    // Baseline = overall mean expression, used as REFERENCE for deviations
    Eigen::VectorXd mu_baseline(G);
    BaselinePrior baseline_prior;
    Eigen::MatrixXd delta_horseshoe;
    if (use_sparse_prior || use_spike_slab || use_reg_horseshoe) {
    // Initialize prior from data (weak)
    baseline_prior = initialize_baseline_prior(mu_estimate, J, G);

    // Initial baseline: sample from prior (NOT mu_estimate)
    std::normal_distribution<double> std_normal(0.0, 1.0);
    for (int g = 0; g < G; ++g) {
        double log_base = baseline_prior.log_mean(g) +
                          std::sqrt(baseline_prior.log_sd(g)) * std_normal(rng_local);
        mu_baseline(g) = std::exp(log_base);
        if (mu_baseline(g) <= 0 || !std::isfinite(mu_baseline(g))) {
            mu_baseline(g) = 0.01;
        }
    }

    // Initialize delta as deviation from baseline
    delta_horseshoe = Eigen::MatrixXd::Zero(J, G);
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            delta_horseshoe(j, g) = std::log(mu_star_1_J_initial(j, g)) -
                                     std::log(mu_baseline(g));
        }
    }

} else {
    // For lognormal prior, baseline not needed (but set to avoid issues)
    mu_baseline = mu_estimate;
}
    // ============ Initialize ALL Parameters (prevent uninitialized on Windows) ============
    SpikeSlabParams spike_slab_params;
    HorseshoeParams horseshoe_params;
    RegularizedHorseshoeParams reg_horseshoe_params;

    // CRITICAL: ALWAYS initialize to prevent Windows crashes from garbage values
    // Even if not using horseshoe, we need valid (empty) params for safety
    horseshoe_params = initialize_horseshoe_params(J, G);
    reg_horseshoe_params = initialize_regularized_horseshoe_params(J, G, p_0);

    // Override with empirical initialization if requested
    if (use_sparse_prior && empirical) {
        horseshoe_params = initialize_horseshoe_params_empirical(
            J, G, mu_baseline, mu_star_1_J_initial
        );
    }

    if (use_reg_horseshoe && empirical) {
        reg_horseshoe_params = initialize_regularized_horseshoe_params_empirical(
            J, G, p_0, mu_baseline, mu_star_1_J_initial
        );
    }

    // Initialize spike-slab parameters (if using)
    if (use_spike_slab) {
        spike_slab_params = initialize_spike_slab_params(J, G);
        std::cout << "Using spike-and-slab prior for cluster-specific means\n";
    }


    // ============ Prepare Outputs ============
    int num_saved = (number_iter - burn_in) / thinning;

    NormHDPResult result;
    result.D = D;
    result.C = C;
    result.G = G;
    result.use_sparse_prior = use_sparse_prior;  // Store flag
	result.use_spike_slab = use_spike_slab;
	result.use_reg_horseshoe = use_reg_horseshoe;

    if (!save_only_z) {
        result.b_output.reserve(num_saved);
        result.alpha_phi2_output.reserve(num_saved);
        result.P_J_D_output.reserve(num_saved);
        result.P_output.reserve(num_saved);
        result.alpha_output.reserve(num_saved);
        result.alpha_zero_output.reserve(num_saved);
        result.mu_star_1_J_output.reserve(num_saved);
        result.phi_star_1_J_output.reserve(num_saved);
        result.Beta_output.reserve(num_saved);
        result.mu_baseline_output.reserve(num_saved);

        // Reserve space for horseshoe parameters if using sparse prior
        if (use_sparse_prior) {
            result.horseshoe_output.reserve(num_saved);
        }
		if (use_reg_horseshoe) {  // ADD THIS BLOCK
            result.reg_horseshoe_output.reserve(num_saved);
        }
    }
    result.Z_output.reserve(num_saved);

    // Acceptance rates
    result.acceptance_rates.P_accept.resize(number_iter - 1, 0.0);
    result.acceptance_rates.alpha_accept.resize(number_iter - 1, 0.0);
    result.acceptance_rates.alpha_zero_accept.resize(number_iter - 1, 0.0);
    result.acceptance_rates.unique_accept.resize(number_iter - 1, 0.0);
    result.acceptance_rates.Beta_accept.resize(number_iter - 1, 0.0);

    // ============ Set Initial Values as Current ============
    Eigen::VectorXd b_new = b_initial;
    double alpha_phi_2_new = alpha_phi_2_initial;
    std::vector<std::vector<int>> Z_new = Z_initial;
    Eigen::MatrixXd P_J_D_new = P_J_D_initial;
    Eigen::VectorXd P_new = P_initial;
    double alpha_new = alpha_initial;
    double alpha_zero_new = alpha_zero_initial;
    Eigen::MatrixXd mu_star_1_J_new = mu_star_1_J_initial;
    Eigen::MatrixXd phi_star_1_J_new = phi_star_1_J_initial;
    std::vector<Eigen::VectorXd> Beta_new = Beta_estimate;

    // Acceptance counters
    int P_count = 0, alpha_count = 0, alpha_zero_count = 0;
    int unique_count = 0, Beta_count = 0;

    // ============ Covariance Structures ============

    // Component probabilities
    Eigen::RowVectorXd mean_X_component_new(J - 1);
    for (int j = 0; j < J - 1; ++j) {
        mean_X_component_new(j) = std::log(P_initial(j) / P_initial(J - 1));
    }
    Eigen::MatrixXd tilde_s_component_new = mean_X_component_new.transpose() * mean_X_component_new;
    Eigen::MatrixXd covariance_component_new = Eigen::MatrixXd::Zero(J - 1, J - 1);

    // Alpha
    double mean_X_alpha_new = std::log(alpha_new);
    double M_2_alpha_new = 0.0;
    double variance_alpha_new = 0.0;

    // Alpha zero
    double mean_X_alpha_zero_new = std::log(alpha_zero_new);
    double M_2_alpha_zero_new = 0.0;
    double variance_alpha_zero_new = 0.0;

    // Unique parameters
    std::vector<std::vector<Eigen::MatrixXd>> covariance_unique_new(J);
    std::vector<std::vector<Eigen::MatrixXd>> tilde_s_unique_new(J);
    std::vector<std::vector<Eigen::RowVectorXd>> mean_X_unique_new(J);

    for (int j = 0; j < J; ++j) {
        covariance_unique_new[j].resize(G);
        tilde_s_unique_new[j].resize(G);
        mean_X_unique_new[j].resize(G);

        for (int g = 0; g < G; ++g) {
            covariance_unique_new[j][g] = Eigen::Matrix2d::Zero();

            Eigen::Vector2d x_vec;
            x_vec << std::log(mu_star_1_J_new(j, g)), std::log(phi_star_1_J_new(j, g));
            tilde_s_unique_new[j][g] = x_vec * x_vec.transpose();
            mean_X_unique_new[j][g] = x_vec.transpose();
        }
    }

    // Capture efficiencies
    std::vector<Eigen::VectorXd> variance_capture_new(D);
    std::vector<Eigen::VectorXd> mean_X_capture_new(D);
    std::vector<Eigen::VectorXd> M_2_capture_new(D);

    for (int d = 0; d < D; ++d) {
        variance_capture_new[d] = Eigen::VectorXd::Zero(C[d]);
        M_2_capture_new[d] = Eigen::VectorXd::Zero(C[d]);
        mean_X_capture_new[d].resize(C[d]);
        for (int c = 0; c < C[d]; ++c) {
            mean_X_capture_new[d](c) = std::log(Beta_new[d](c) / (1.0 - Beta_new[d](c)));
        }
    }

    // ============ MCMC Iterations ============
    std::cout << "Starting MCMC with " << number_iter << " iterations...\n";

    for (int iter = 2; iter <= number_iter; ++iter) {

        if (iter % iter_update == 0) {
            std::cout << "Iteration: " << iter << " / " << number_iter << std::endl;
        }
		// Regression parameters - CHOOSE MODEL
		if (use_reg_horseshoe) {
            // Use regularized horseshoe version
            auto mean_disp_output = mean_dispersion_regularized_horseshoe_mcmc(
                mu_star_1_J_new,
                phi_star_1_J_new,
                mu_baseline,  // Baseline
                reg_horseshoe_params,
                v_1, v_2, m_b, quadratic
            );
            alpha_phi_2_new = mean_disp_output.alpha_phi_2;
            b_new = mean_disp_output.b;
            reg_horseshoe_params = mean_disp_output.rhs_params;

        } else if (use_spike_slab) {
            // Use spike-and-slab version
            auto mean_disp_output = mean_dispersion_spike_slab_mcmc(
                mu_star_1_J_new,
                phi_star_1_J_new,
                mu_baseline,  // Baseline
                spike_slab_params,
                v_1, v_2, m_b, quadratic
            );
            alpha_phi_2_new = mean_disp_output.alpha_phi_2;
            b_new = mean_disp_output.b;
            spike_slab_params = mean_disp_output.spike_slab_params;

        } else if (use_sparse_prior) {
            // Use horseshoe prior version
            auto mean_disp_output = mean_dispersion_horseshoe_mcmc(
                mu_star_1_J_new,
                phi_star_1_J_new,
                mu_baseline,  // Baseline expression
                horseshoe_params,
                v_1, v_2, m_b, quadratic
            );
            alpha_phi_2_new = mean_disp_output.alpha_phi_2;
            b_new = mean_disp_output.b;
            horseshoe_params = mean_disp_output.horseshoe_params;  // Update horseshoe params
        } else {
            // Use standard version (no sparsity)
            auto mean_disp_output = mean_dispersion_mcmc(
                mu_star_1_J_new,
                phi_star_1_J_new,
                v_1, v_2, m_b, quadratic
            );
            alpha_phi_2_new = mean_disp_output.alpha_phi_2;
            b_new = mean_disp_output.b;
        }
        if (use_sparse_prior || use_spike_slab || use_reg_horseshoe) {
        // Compute current delta from mu_star and mu_baseline
        for (int j = 0; j < J; ++j) {
            for (int g = 0; g < G; ++g) {
                delta_horseshoe(j, g) = std::log(mu_star_1_J_new(j, g)) -
                                         std::log(mu_baseline(g));
            }
        }

        // Sample new baseline
        mu_baseline = sample_mu_baseline(
            mu_star_1_J_new,
            delta_horseshoe,
            baseline_prior,
            J, G
        );

        // Update delta with new baseline
        for (int j = 0; j < J; ++j) {
            for (int g = 0; g < G; ++g) {
                delta_horseshoe(j, g) = std::log(mu_star_1_J_new(j, g)) -
                                         std::log(mu_baseline(g));
            }
        }
    }


        // 2) Allocation variables
        auto alloc_output = allocation_variables_mcmc(P_J_D_new, mu_star_1_J_new,
                                                     phi_star_1_J_new, Y, Beta_new,
                                                     iter, num_cores);
        Z_new = alloc_output.Z;

        if (print_Z) {
            for (int d = 0; d < D; ++d) {
                std::cout << "Dataset " << d << " cluster counts: ";
                std::vector<int> counts(J, 0);
                for (int z : Z_new[d]) counts[z]++;
                for (int j = 0; j < J; ++j) std::cout << counts[j] << " ";
                std::cout << std::endl;
            }
        }

        // 3) Dataset-specific probabilities
        P_J_D_new = dataset_specific_mcmc(Z_new, P_new, alpha_new);

        // 4) Component probabilities
        auto comp_output = component_probabilities_mcmc(P_new, P_J_D_new, alpha_zero_new,
                                                       alpha_new, covariance_component_new,
                                                       mean_X_component_new, tilde_s_component_new,
                                                       iter, adaptive_prop);
        P_new = comp_output.P_new;
        tilde_s_component_new = comp_output.tilde_s_new;
        mean_X_component_new = comp_output.mean_x_new;
        covariance_component_new = comp_output.covariance_new;
        P_count += comp_output.accept;
        result.acceptance_rates.P_accept[iter - 2] = (double)P_count / (iter - 1);

        // 5) Alpha
        auto alpha_output = alpha_mcmc(P_J_D_new, P_new, alpha_new, mean_X_alpha_new,
                                      M_2_alpha_new, variance_alpha_new, iter, adaptive_prop);
        alpha_new = alpha_output.alpha_new;
        mean_X_alpha_new = alpha_output.X_mean_new;
        M_2_alpha_new = alpha_output.M_2_new;
        variance_alpha_new = alpha_output.variance_new;
        alpha_count += alpha_output.accept;
        result.acceptance_rates.alpha_accept[iter - 2] = (double)alpha_count / (iter - 1);

        // 6) Alpha zero
        auto alpha_zero_output = alpha_zero_mcmc(P_new, alpha_zero_new, mean_X_alpha_zero_new,
                                                M_2_alpha_zero_new, variance_alpha_zero_new,
                                                iter, adaptive_prop);
        alpha_zero_new = alpha_zero_output.alpha_zero_new;
        mean_X_alpha_zero_new = alpha_zero_output.X_mean_new;
        M_2_alpha_zero_new = alpha_zero_output.M_2_new;
        variance_alpha_zero_new = alpha_zero_output.variance_new;
        alpha_zero_count += alpha_zero_output.accept;
        result.acceptance_rates.alpha_zero_accept[iter - 2] = (double)alpha_zero_count / (iter - 1);

        // 7) Unique parameters
        MuPriorType prior_type = MuPriorType::LOGNORMAL;
        const HorseshoeParams* hs_ptr = nullptr;
        const RegularizedHorseshoeParams* rhs_ptr = nullptr;
		const SpikeSlabParams* ss_ptr = nullptr;
        const Eigen::VectorXd* baseline_ptr = nullptr;

        if (use_reg_horseshoe) {
                prior_type = MuPriorType::REGULARIZED_HORSESHOE;
                rhs_ptr = &reg_horseshoe_params;
                baseline_ptr = &mu_baseline;
            } else if (use_sparse_prior) {
                prior_type = MuPriorType::HORSESHOE;
                hs_ptr = &horseshoe_params;
                baseline_ptr = &mu_baseline;
            }else if (use_spike_slab) {
    			prior_type   = MuPriorType::SPIKE_SLAB;
    			ss_ptr       = &spike_slab_params;
    			baseline_ptr = &mu_baseline;
    		}

        auto unique_output = unique_parameters_mcmc(mu_star_1_J_new, phi_star_1_J_new,
                                                   mean_X_unique_new, tilde_s_unique_new,
                                                   Z_new, b_new, alpha_phi_2_new, Beta_new,
                                                   alpha_mu_2, covariance_unique_new, iter,
                                                   quadratic, Y, adaptive_prop, num_cores, prior_type,
                                                    hs_ptr,rhs_ptr,ss_ptr,baseline_ptr);
        mu_star_1_J_new = unique_output.mu_star_1_J_new;
        phi_star_1_J_new = unique_output.phi_star_1_J_new;
        tilde_s_unique_new = unique_output.tilde_s_mu_phi_new;
        mean_X_unique_new = unique_output.mean_X_mu_phi_new;
        covariance_unique_new = unique_output.covariance_new;
        unique_count += unique_output.accept_count;
        result.acceptance_rates.unique_accept[iter - 2] = (double)unique_count / ((iter - 1) * J * G);

        // 8) Capture efficiencies
        auto capture_output = capture_efficiencies_mcmc(Beta_new, Y, Z_new, mu_star_1_J_new,
                                                       phi_star_1_J_new, a_d_beta, b_d_beta,
                                                       iter, M_2_capture_new, mean_X_capture_new,
                                                       variance_capture_new, adaptive_prop);
        Beta_new = capture_output.Beta_new;
        mean_X_capture_new = capture_output.mean_X_new;
        M_2_capture_new = capture_output.M_2_new;
        variance_capture_new = capture_output.variance_new;
        Beta_count += capture_output.accept_count;
        int total_cells = 0;
        for (int c_val : C) total_cells += c_val;
        result.acceptance_rates.Beta_accept[iter - 2] = (double)Beta_count / ((iter - 1) * total_cells);

        // 9) Save outputs
        if (iter >= burn_in && (iter - burn_in) % thinning == 0) {
            result.Z_output.push_back(Z_new);

            if (!save_only_z) {
                result.b_output.push_back(b_new);
                result.alpha_phi2_output.push_back(alpha_phi_2_new);
                result.P_J_D_output.push_back(P_J_D_new);
                result.P_output.push_back(P_new);
                result.alpha_output.push_back(alpha_new);
                result.alpha_zero_output.push_back(alpha_zero_new);
                result.mu_star_1_J_output.push_back(mu_star_1_J_new);
                result.phi_star_1_J_output.push_back(phi_star_1_J_new);
                result.Beta_output.push_back(Beta_new);
                result.mu_baseline_output.push_back(mu_baseline);

                // Save horseshoe parameters if using sparse prior
                if (use_sparse_prior) {
                    result.horseshoe_output.push_back(horseshoe_params);
                }
				if (use_spike_slab) {
                    result.spike_slab_output.push_back(spike_slab_params);
                }
				if (use_reg_horseshoe) {
                    result.reg_horseshoe_output.push_back(reg_horseshoe_params);
                }
            }
        }
    }

    std::cout << "MCMC completed!\n";

    return result;
}