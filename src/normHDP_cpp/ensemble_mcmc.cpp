#include <Rcpp.h>
#include <RcppEigen.h>
#include "includes/normHDP_mcmc.h"
#include "includes/mean_dispersion_horseshoe.h"
#include "includes/mean_dispersion_spike_slab.h"
#include "includes/mean_dispersion_regularized_horseshoe.h"
#include "includes/utils.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <cmath>
#include <map>
#include <chrono>
#include <iomanip>

using namespace Rcpp;

// Thread-safe progress tracking
std::mutex progress_mutex;


struct ChainResult {
    // SAVE FULL TRACES instead of just final values
    std::vector<Eigen::VectorXd> b_output;
    std::vector<double> alpha_phi2_output;
    std::vector<Eigen::MatrixXd> P_J_D_output;
    std::vector<Eigen::VectorXd> P_output;
    std::vector<double> alpha_output;
    std::vector<double> alpha_zero_output;
    std::vector<Eigen::MatrixXd> mu_star_1_J_output;
    std::vector<Eigen::MatrixXd> phi_star_1_J_output;
    std::vector<std::vector<Eigen::VectorXd>> Beta_output;
    std::vector<std::vector<std::vector<int>>> Z_output;
    std::vector<HorseshoeParams> horseshoe_output;
	std::vector<SpikeSlabParams> spike_slab_output;
	std::vector<RegularizedHorseshoeParams> reg_horseshoe_output;
    std::vector<Eigen::VectorXd> mu_baseline_output;
    bool use_sparse_prior;
	bool use_spike_slab;
	bool use_reg_horseshoe;
    int D;
};

// Run a single chain - returns plain C++ struct, no R objects
ChainResult run_single_chain_safe(
    const std::vector<Eigen::MatrixXd>& Y,
    int J,
    int chain_length,
    int thinning,
    bool empirical,
    int burn_in,
    bool quadratic,
    double beta_mean,
    double alpha_mu_2,
    double adaptive_prop,
    bool save_only_z,
    bool use_sparse_prior,
	bool use_spike_slab,
	bool use_reg_horseshoe,
    double p_0,
    const Eigen::VectorXd& baynorm_mu_estimate,
    const Eigen::VectorXd& baynorm_phi_estimate,
    const std::vector<Eigen::VectorXd>& baynorm_beta,
    int chain_id,
    bool print_progress
) {
    if (print_progress) {
        std::lock_guard<std::mutex> lock(progress_mutex);
       	std::cout << "Starting chain " << (chain_id + 1) << std::endl;
    	std::cout.flush();
    }

    // Initialize RNG with different seed for each chain
    init_rng(std::random_device{}() + chain_id * 1000);

    NormHDPResult result = normHDP_mcmc(
        Y, J, chain_length, thinning, empirical, burn_in, quadratic,
        999999,  // iter_update (no printing within chain)
        beta_mean, alpha_mu_2, adaptive_prop,
        false,   // print_Z
        1,       // num_cores per chain (parallelism at chain level)
        save_only_z,
        use_sparse_prior,
		use_spike_slab,
		use_reg_horseshoe,
        p_0,
        baynorm_mu_estimate,
        baynorm_phi_estimate,
        baynorm_beta
    );

    if (print_progress) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        std::cout << "Starting chain " << (chain_id + 1) << std::endl;
    	std::cout.flush();
    }

    // Copy to plain struct - WITH SAFETY CHECKS
    ChainResult chain_result;

    // Initialize flags first
    chain_result.use_sparse_prior = result.use_sparse_prior;
    chain_result.use_spike_slab = result.use_spike_slab;
    chain_result.use_reg_horseshoe = result.use_reg_horseshoe;
    chain_result.D = result.D;

    // Copy outputs with try-catch for safety
    try {
        chain_result.b_output = result.b_output;
        chain_result.alpha_phi2_output = result.alpha_phi2_output;
        chain_result.P_J_D_output = result.P_J_D_output;
        chain_result.P_output = result.P_output;
        chain_result.alpha_output = result.alpha_output;
        chain_result.alpha_zero_output = result.alpha_zero_output;
        chain_result.mu_star_1_J_output = result.mu_star_1_J_output;
        chain_result.phi_star_1_J_output = result.phi_star_1_J_output;
        chain_result.Beta_output = result.Beta_output;
        chain_result.Z_output = result.Z_output;
        chain_result.mu_baseline_output = result.mu_baseline_output;

        // CRITICAL: Only copy horseshoe if actually used and populated
        if (result.use_sparse_prior && !result.horseshoe_output.empty()) {
            chain_result.horseshoe_output = result.horseshoe_output;
        }

        if (result.use_spike_slab && !result.spike_slab_output.empty()) {
            chain_result.spike_slab_output = result.spike_slab_output;
        }

        if (result.use_reg_horseshoe && !result.reg_horseshoe_output.empty()) {
            chain_result.reg_horseshoe_output = result.reg_horseshoe_output;
        }

    } catch (const std::exception& e) {
        #ifdef RCPP_VERSION
        Rcpp::Rcout << "Warning: Error copying chain results: " << e.what() << "\n";
        #endif
        // Chain result will be incomplete but won't crash
    }

    return chain_result;
}

// Compute mean and standard deviation
template<typename T>
void compute_mean_sd(const std::vector<T>& samples, T& mean, T& sd) {
    int n = samples.size();
    if (n == 0) {

        return;
    }

    mean = samples[0];
    for (int i = 1; i < n; ++i) {
        mean += samples[i];
    }
    mean /= n;

    // Zero-initialize sd with correct dimensions
    sd = T::Zero(mean.rows(), mean.cols());  // for Matrix types

    if (n == 1) {
        // sd stays zero — only one sample, variance undefined
        return;
    }

    for (int i = 0; i < n; ++i) {
        sd += (samples[i] - mean).cwiseProduct(samples[i] - mean);
    }
    sd = (sd / (n - 1)).cwiseSqrt();
}

void compute_scalar_mean_sd(const std::vector<double>& samples, double& mean, double& sd) {
    int n = samples.size();
    if (n == 0) { mean = 0.0; sd = 0.0; return; }

    mean = 0.0;
    for (double s : samples) mean += s;
    mean /= n;

    if (n == 1) { sd = 0.0; return; }  // ← add this

    sd = 0.0;
    for (double s : samples) sd += (s - mean) * (s - mean);
    sd = std::sqrt(sd / (n - 1));
}

// [[Rcpp::export]]
Rcpp::List ensemble_mcmc_R(
    Rcpp::List Y,
    int J,
    int num_chains,
    int chain_length,
    int thinning,
    bool empirical,
    int burn_in,
    bool quadratic,
    int iter_update,
    double beta_mean,
    double alpha_mu_2,
    double adaptive_prop,
    bool print_progress,
    int num_cores,
    bool save_only_z,
    bool use_sparse_prior,
	bool use_spike_slab,
	bool use_reg_horseshoe,
    double p_0,
    Rcpp::NumericVector baynorm_mu_estimate,
    Rcpp::NumericVector baynorm_phi_estimate,
    Rcpp::List baynorm_beta_list
) {
		if ((int)use_sparse_prior + (int)use_spike_slab + (int)use_reg_horseshoe > 1) {
    Rcpp::stop("Only one of use_sparse_prior, use_spike_slab, use_reg_horseshoe may be TRUE at a time.");
}
    // Convert all R objects to C++ BEFORE threading
    int D = Y.size();

    std::vector<Eigen::MatrixXd> Y_vec(D);
    for (int d = 0; d < D; d++) {
        Y_vec[d] = Rcpp::as<Eigen::MatrixXd>(Y[d]);
    }

    std::vector<Eigen::VectorXd> baynorm_beta(D);
    for (int d = 0; d < D; d++) {
        baynorm_beta[d] = Rcpp::as<Eigen::VectorXd>(baynorm_beta_list[d]);
    }

    Eigen::VectorXd mu_vec = Rcpp::as<Eigen::VectorXd>(baynorm_mu_estimate);
    Eigen::VectorXd phi_vec = Rcpp::as<Eigen::VectorXd>(baynorm_phi_estimate);

    int G = Y_vec[0].rows();
    std::vector<int> C(D);
    for (int d = 0; d < D; d++) {
        C[d] = Y_vec[d].cols();
    }

    if (num_cores <= 0) {
        num_cores = std::thread::hardware_concurrency();
    }

    Rcpp::Rcout << "\n" << std::string(70, '=') << "\n";
    Rcpp::Rcout << "ENSEMBLE MCMC" << (use_sparse_prior ? " (WITH HORSESHOE PRIOR)" : "") << "\n";
    Rcpp::Rcout << std::string(70, '=') << "\n";
    Rcpp::Rcout << "Configuration:\n";
    Rcpp::Rcout << "  Number of chains:  " << num_chains << "\n";
    Rcpp::Rcout << "  Chain length:      " << chain_length << " iterations\n";
    Rcpp::Rcout << "  Burn-in:           " << burn_in << " iterations\n";
    Rcpp::Rcout << "  Thinning:          " << thinning << "\n";
    Rcpp::Rcout << "  Parallel cores:    " << num_cores << "\n";
    Rcpp::Rcout << "  Datasets:          " << D << "\n";
    Rcpp::Rcout << "  Genes:             " << G << "\n";
    Rcpp::Rcout << "  Clusters:          " << J << "\n";
    Rcpp::Rcout << "  Sparse prior:      " << (use_sparse_prior ? "Horseshoe" : "NO") << "\n";
    Rcpp::Rcout << "  Spike-Slab prior:  " << (use_spike_slab ? "spike slab" : "NO") << "\n";
	Rcpp::Rcout << "  Reg-Horseshoe:     " << (use_reg_horseshoe ? "reg. horseshoe" : "NO") << "\n";
    if (use_reg_horseshoe) {
        Rcpp::Rcout << "    p_0 (sparsity):  " << p_0 << "\n";
    }
    Rcpp::Rcout << std::string(70, '=') << "\n\n";

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Use plain C++ vector, no R objects in threads
    std::vector<ChainResult> chain_results(num_chains);

    // Process chains in batches, join threads before converting to R
    for (int batch_start = 0; batch_start < num_chains; batch_start += num_cores) {
        int batch_end = std::min(batch_start + num_cores, num_chains);
        std::vector<std::thread> threads;

        if (print_progress) {
			std::cout << "Running chains " << (batch_start + 1)
                        << " to " << batch_end << "...\n" << std::endl;
    		std::cout.flush();
        }

        // Launch threads
        for (int i = batch_start; i < batch_end; i++) {
            threads.emplace_back([&, i]() {
				try {
                chain_results[i] = run_single_chain_safe(
                    Y_vec, J, chain_length, thinning, empirical, burn_in, quadratic,
                    beta_mean, alpha_mu_2, adaptive_prop, save_only_z, use_sparse_prior,
                    use_spike_slab, use_reg_horseshoe, p_0, mu_vec, phi_vec, baynorm_beta,
                    i, print_progress
                );
			} catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        std::cerr << "Chain " << i << " failed: " << e.what() << std::endl;
        // Leave chain_results[i] in default state
    } catch (...) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        std::cerr << "Chain " << i << " failed with unknown error" << std::endl;
    }
            });
        }

        // FIXED: Wait for ALL threads to complete before continuing
        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }

        // FIXED: Check for user interrupt between batches, not during threading
        Rcpp::checkUserInterrupt();

        if (print_progress) {
            Rcpp::Rcout << "Batch completed (" << batch_end << "/" << num_chains
                        << " chains done)\n\n";

        }
    }

    // convert to R objects AFTER all threads are done
    Rcpp::List Z_trace_all(num_chains);

    // save parameter traces if not save_only_z
    Rcpp::List b_trace_all, alpha_trace_all, alpha_zero_trace_all;
    Rcpp::List alpha_phi2_trace_all, P_trace_all;
    Rcpp::List mu_trace_all, phi_trace_all, Beta_trace_all;
    Rcpp::List lambda_trace_all, tau_trace_all, sigma_mu_trace_all;

    if (!save_only_z) {
        b_trace_all = Rcpp::List(num_chains);
        alpha_trace_all = Rcpp::List(num_chains);
        alpha_zero_trace_all = Rcpp::List(num_chains);
        alpha_phi2_trace_all = Rcpp::List(num_chains);
        P_trace_all = Rcpp::List(num_chains);
        mu_trace_all = Rcpp::List(num_chains);
        phi_trace_all = Rcpp::List(num_chains);
        Beta_trace_all = Rcpp::List(num_chains);

        if (use_sparse_prior || use_reg_horseshoe) {
            lambda_trace_all = Rcpp::List(num_chains);
            tau_trace_all = Rcpp::List(num_chains);
            sigma_mu_trace_all = Rcpp::List(num_chains);
        }
    }

    for (int i = 0; i < num_chains; i++) {
        const auto& result = chain_results[i];
        int n_samples = result.Z_output.size();

        // Save Z traces
        Rcpp::List Z_trace_i(n_samples);
        for (int t = 0; t < n_samples; t++) {
            Rcpp::List Z_t(D);
            for (int d = 0; d < D; d++) {
                Z_t[d] = Rcpp::wrap(result.Z_output[t][d]);
            }
            Z_trace_i[t] = Z_t;
        }
        Z_trace_all[i] = Z_trace_i;

        // Save parameter traces (if not save_only_z)
        if (!save_only_z && n_samples > 0) {
            // b trace
            b_trace_all[i] = Rcpp::wrap(result.b_output);

            // alpha trace
            alpha_trace_all[i] = Rcpp::wrap(result.alpha_output);

            // alpha_zero trace
            alpha_zero_trace_all[i] = Rcpp::wrap(result.alpha_zero_output);

            // alpha_phi2 trace
            alpha_phi2_trace_all[i] = Rcpp::wrap(result.alpha_phi2_output);

            // P trace
            P_trace_all[i] = Rcpp::wrap(result.P_output);

            // mu trace
            mu_trace_all[i] = Rcpp::wrap(result.mu_star_1_J_output);

            // phi trace
            phi_trace_all[i] = Rcpp::wrap(result.phi_star_1_J_output);

            // Beta trace
            Beta_trace_all[i] = Rcpp::wrap(result.Beta_output);

            // Horseshoe traces (if sparse prior) - WITH BOUNDS CHECKING FOR WINDOWS
            if (use_sparse_prior && !result.horseshoe_output.empty()) {
                try {
                    int n_hs_samples = result.horseshoe_output.size();

                    // CRITICAL: Validate structure before accessing (Windows needs this!)
                    bool valid_structure = (n_hs_samples > 0);

                    if (valid_structure && n_hs_samples > 0) {
                        // Check first sample has proper dimensions
                        if (result.horseshoe_output[0].lambda.size() != static_cast<size_t>(J)) {
                            valid_structure = false;
                        } else {
                            for (int j = 0; j < J && valid_structure; j++) {
                                if (result.horseshoe_output[0].lambda[j].size() != static_cast<size_t>(G)) {
                                    valid_structure = false;
                                }
                            }
                        }
                    }

                    if (valid_structure) {
                        // Lambda trace
                        Rcpp::List lambda_chain(n_hs_samples);
                        for (int t = 0; t < n_hs_samples; t++) {
                            if (result.horseshoe_output[t].lambda.size() == static_cast<size_t>(J)) {
                                Eigen::MatrixXd lambda_mat(J, G);
                                for (int j = 0; j < J; j++) {
                                    if (result.horseshoe_output[t].lambda[j].size() == static_cast<size_t>(G)) {
                                        lambda_mat.row(j) = result.horseshoe_output[t].lambda[j].transpose();
                                    } else {
                                        lambda_mat.row(j).setConstant(1.0);
                                    }
                                }
                                lambda_chain[t] = lambda_mat;
                            }
                        }
                        lambda_trace_all[i] = lambda_chain;

                        // Tau trace
                        Eigen::MatrixXd tau_mat(n_hs_samples, J);
                        for (int t = 0; t < n_hs_samples; t++) {
                            if (result.horseshoe_output[t].tau.size() == static_cast<size_t>(J)) {
                                for (int j = 0; j < J; j++) {
                                    tau_mat(t, j) = result.horseshoe_output[t].tau[j];
                                }
                            } else {
                                tau_mat.row(t).setConstant(1.0);
                            }
                        }
                        tau_trace_all[i] = tau_mat;

                        // Sigma_mu trace
                        Eigen::VectorXd sigma_mu_vec(n_hs_samples);
                        for (int t = 0; t < n_hs_samples; t++) {
                            sigma_mu_vec(t) = result.horseshoe_output[t].sigma_mu;
                        }
                        sigma_mu_trace_all[i] = sigma_mu_vec;
                    }
                } catch (const std::exception& e) {
                    Rcpp::Rcout << "Warning: Error processing horseshoe for chain " << (i+1)
                                << ": " << e.what() << "\n";
                }
            }

            // Regularized Horseshoe traces - WITH BOUNDS CHECKING FOR WINDOWS
            if (use_reg_horseshoe && !result.reg_horseshoe_output.empty()) {
                try {
                    int n_reg_samples = result.reg_horseshoe_output.size();

                    // CRITICAL: Validate structure
                    bool valid_structure = (n_reg_samples > 0);

                    if (valid_structure && n_reg_samples > 0) {
                        if (result.reg_horseshoe_output[0].lambda.size() != static_cast<size_t>(J)) {
                            valid_structure = false;
                        } else {
                            for (int j = 0; j < J && valid_structure; j++) {
                                if (result.reg_horseshoe_output[0].lambda[j].size() != static_cast<size_t>(G)) {
                                    valid_structure = false;
                                }
                            }
                        }
                    }

                    if (valid_structure) {
                        // Lambda trace
                        Rcpp::List lambda_chain(n_reg_samples);
                        for (int t = 0; t < n_reg_samples; t++) {
                            if (result.reg_horseshoe_output[t].lambda.size() == static_cast<size_t>(J)) {
                                Eigen::MatrixXd lambda_mat(J, G);
                                for (int j = 0; j < J; j++) {
                                    if (result.reg_horseshoe_output[t].lambda[j].size() == static_cast<size_t>(G)) {
                                        lambda_mat.row(j) = result.reg_horseshoe_output[t].lambda[j].transpose();
                                    } else {
                                        lambda_mat.row(j).setConstant(1.0);
                                    }
                                }
                                lambda_chain[t] = lambda_mat;
                            }
                        }
                        lambda_trace_all[i] = lambda_chain;

                        // Tau trace
                        Eigen::MatrixXd tau_mat(n_reg_samples, J);
                        for (int t = 0; t < n_reg_samples; t++) {
                            if (result.reg_horseshoe_output[t].tau.size() == static_cast<size_t>(J)) {
                                for (int j = 0; j < J; j++) {
                                    tau_mat(t, j) = result.reg_horseshoe_output[t].tau[j];
                                }
                            } else {
                                tau_mat.row(t).setConstant(1.0);
                            }
                        }
                        tau_trace_all[i] = tau_mat;

                        // Sigma_mu trace
                        Eigen::VectorXd sigma_mu_vec(n_reg_samples);
                        for (int t = 0; t < n_reg_samples; t++) {
                            sigma_mu_vec(t) = result.reg_horseshoe_output[t].sigma_mu;
                        }
                        sigma_mu_trace_all[i] = sigma_mu_vec;
                    }
                } catch (const std::exception& e) {
                    Rcpp::Rcout << "Warning: Error processing regularized horseshoe for chain " << (i+1)
                                << ": " << e.what() << "\n";
                }
            }



        }
    }

    Rcpp::Rcout << "All chains completed. Computing ensemble statistics...\n";

    // ===== AVERAGING & POSTERIOR SUMMARY =====
    std::vector<Eigen::VectorXd> b_samples;
    std::vector<double> alpha_phi2_samples;
    std::vector<Eigen::MatrixXd> P_J_D_samples;
    std::vector<Eigen::VectorXd> P_samples;
    std::vector<double> alpha_samples;
    std::vector<double> alpha_zero_samples;
    std::vector<Eigen::MatrixXd> mu_samples;
    std::vector<Eigen::MatrixXd> phi_samples;
    std::vector<std::vector<Eigen::VectorXd>> Beta_samples(D);

    for (int i = 0; i < num_chains; i++) {
        const auto& r = chain_results[i];
        int last = r.b_output.size() - 1;
        if (last < 0) continue;

        b_samples.push_back(r.b_output[last]);
        alpha_phi2_samples.push_back(r.alpha_phi2_output[last]);
        P_J_D_samples.push_back(r.P_J_D_output[last]);
        P_samples.push_back(r.P_output[last]);
        alpha_samples.push_back(r.alpha_output[last]);
        alpha_zero_samples.push_back(r.alpha_zero_output[last]);
        mu_samples.push_back(r.mu_star_1_J_output[last]);
        phi_samples.push_back(r.phi_star_1_J_output[last]);

        for (int d = 0; d < D; d++) {
            Beta_samples[d].push_back(r.Beta_output[last][d]);
        }
    }

    Eigen::VectorXd b_mean, b_sd;
    Eigen::MatrixXd P_J_D_mean, P_J_D_sd;
    Eigen::VectorXd P_mean, P_sd;
    Eigen::MatrixXd mu_mean, mu_sd;
    Eigen::MatrixXd phi_mean, phi_sd;

    double alpha_phi2_mean, alpha_phi2_sd;
    double alpha_mean, alpha_sd;
    double alpha_zero_mean, alpha_zero_sd;

    compute_mean_sd(b_samples, b_mean, b_sd);
    compute_scalar_mean_sd(alpha_phi2_samples, alpha_phi2_mean, alpha_phi2_sd);
    compute_mean_sd(P_J_D_samples, P_J_D_mean, P_J_D_sd);
    compute_mean_sd(P_samples, P_mean, P_sd);
    compute_scalar_mean_sd(alpha_samples, alpha_mean, alpha_sd);
    compute_scalar_mean_sd(alpha_zero_samples, alpha_zero_mean, alpha_zero_sd);
    compute_mean_sd(mu_samples, mu_mean, mu_sd);
    compute_mean_sd(phi_samples, phi_mean, phi_sd);

    std::vector<Eigen::VectorXd> Beta_mean(D);
    std::vector<Eigen::VectorXd> Beta_sd(D);
    for (int d = 0; d < D; d++) {
        compute_mean_sd(Beta_samples[d], Beta_mean[d], Beta_sd[d]);
    }

    // ===== HORSESHOE PARAMETER AVERAGING (if using sparse prior) =====
    Eigen::MatrixXd lambda_mean, lambda_sd;
    Eigen::VectorXd tau_mean, tau_sd;
    double sigma_mu_mean = 0.0, sigma_mu_sd = 0.0;

    if (use_sparse_prior && num_chains > 0 &&
    !chain_results.empty() &&
    !chain_results[0].horseshoe_output.empty()) {
        Rcpp::Rcout << "Computing horseshoe parameter statistics...\n";

        // Collect horseshoe samples from all chains (last sample from each chain)
        std::vector<Eigen::MatrixXd> lambda_samples;
        std::vector<Eigen::VectorXd> tau_samples;
        std::vector<double> sigma_mu_samples;

        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            int last = r.horseshoe_output.size() - 1;
            if (last < 0) continue;

            const auto& hs = r.horseshoe_output[last];

            // Convert lambda vectors to matrix
            Eigen::MatrixXd lambda_mat(J, G);
            for (int j = 0; j < J; j++) {
                lambda_mat.row(j) = hs.lambda[j].transpose();
            }
            lambda_samples.push_back(lambda_mat);

            // Convert tau vector
            Eigen::VectorXd tau_vec(J);
            for (int j = 0; j < J; j++) {
                tau_vec(j) = hs.tau[j];
            }
            tau_samples.push_back(tau_vec);

            sigma_mu_samples.push_back(hs.sigma_mu);
        }

        // Compute means and SDs
        compute_mean_sd(lambda_samples, lambda_mean, lambda_sd);
        compute_mean_sd(tau_samples, tau_mean, tau_sd);
        compute_scalar_mean_sd(sigma_mu_samples, sigma_mu_mean, sigma_mu_sd);
    }
	 // ===== SPIKE-AND-SLAB PARAMETER AVERAGING =====
    Eigen::MatrixXd gamma_mean, gamma_sd;
    double pi_mean = 0.0, pi_sd = 0.0;
    double sigma_slab_mean = 0.0, sigma_slab_sd = 0.0;

    if (use_spike_slab && !chain_results[0].spike_slab_output.empty()) {
        Rcpp::Rcout << "Computing spike-and-slab parameter statistics...\n";

        // Collect spike-slab samples from all chains (last sample from each chain)
        std::vector<Eigen::MatrixXd> gamma_samples;
        std::vector<double> pi_samples;
        std::vector<double> sigma_slab_samples;

        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            int last = r.spike_slab_output.size() - 1;
            if (last < 0) continue;

            const auto& ss = r.spike_slab_output[last];

            // Convert gamma vectors to matrix
            Eigen::MatrixXd gamma_mat(J, G);
            for (int j = 0; j < J; j++) {
                gamma_mat.row(j) = ss.gamma[j].col(0).transpose();
            }
            gamma_samples.push_back(gamma_mat);

            pi_samples.push_back(ss.pi);
            sigma_slab_samples.push_back(ss.sigma_slab);
        }

        // Compute means and SDs
        compute_mean_sd(gamma_samples, gamma_mean, gamma_sd);
        compute_scalar_mean_sd(pi_samples, pi_mean, pi_sd);
        compute_scalar_mean_sd(sigma_slab_samples, sigma_slab_mean, sigma_slab_sd);
    }

	// ===== REGULARIZED HORSESHOE PARAMETER AVERAGING =====
    Eigen::MatrixXd rhs_lambda_mean, rhs_lambda_sd;
    Eigen::VectorXd rhs_tau_mean, rhs_tau_sd;
    double rhs_tau_0_mean = 0.0, rhs_tau_0_sd = 0.0;
    double rhs_sigma_mu_mean = 0.0, rhs_sigma_mu_sd = 0.0;

    if (use_reg_horseshoe && !chain_results[0].reg_horseshoe_output.empty()) {
        Rcpp::Rcout << "Computing regularized horseshoe parameter statistics...\n";

        std::vector<Eigen::MatrixXd> rhs_lambda_samples;
        std::vector<Eigen::VectorXd> rhs_tau_samples;
        std::vector<double> rhs_tau_0_samples;
        std::vector<double> rhs_sigma_mu_samples;

        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            int last = r.reg_horseshoe_output.size() - 1;
            if (last < 0) continue;

            const auto& rhs = r.reg_horseshoe_output[last];

            // Convert lambda
            Eigen::MatrixXd lambda_mat(J, G);
            for (int j = 0; j < J; j++) {
                lambda_mat.row(j) = rhs.lambda[j].transpose();
            }
            rhs_lambda_samples.push_back(lambda_mat);

            // Convert tau
            Eigen::VectorXd tau_vec(J);
            for (int j = 0; j < J; j++) {
                tau_vec(j) = rhs.tau[j];
            }
            rhs_tau_samples.push_back(tau_vec);

            rhs_tau_0_samples.push_back(rhs.tau_0);
            rhs_sigma_mu_samples.push_back(rhs.sigma_mu);
        }

        // Compute means and SDs
        compute_mean_sd(rhs_lambda_samples, rhs_lambda_mean, rhs_lambda_sd);
        compute_mean_sd(rhs_tau_samples, rhs_tau_mean, rhs_tau_sd);
        compute_scalar_mean_sd(rhs_tau_0_samples, rhs_tau_0_mean, rhs_tau_0_sd);
        compute_scalar_mean_sd(rhs_sigma_mu_samples, rhs_sigma_mu_mean, rhs_sigma_mu_sd);
    }

    // ===== CONSENSUS CLUSTERING =====
    Rcpp::Rcout << "Computing consensus clustering...\n";

    std::vector<std::vector<int>> Z_consensus(D);
    std::vector<Eigen::MatrixXd> coclustering(D);

    for (int d = 0; d < D; d++) {
        int C_d = C[d];
        Eigen::MatrixXd cocluster = Eigen::MatrixXd::Zero(C_d, C_d);

        for (int chain = 0; chain < num_chains; chain++) {
            const auto& Z_last = chain_results[chain].Z_output.back()[d];
            for (int c1 = 0; c1 < C_d; c1++) {
                for (int c2 = c1; c2 < C_d; c2++) {
                    if (Z_last[c1] == Z_last[c2]) {
                        cocluster(c1, c2) += 1.0;
                        if (c1 != c2) cocluster(c2, c1) += 1.0;
                    }
                }
            }
        }

        cocluster /= num_chains;
        coclustering[d] = cocluster;

        Z_consensus[d].resize(C_d);
        for (int c = 0; c < C_d; c++) {
            std::map<int, int> freq;
            for (int chain = 0; chain < num_chains; chain++) {
                freq[chain_results[chain].Z_output.back()[d][c]]++;
            }

            int best = 0, maxc = 0;
            for (auto& kv : freq) {
                if (kv.second > maxc) {
                    maxc = kv.second;
                    best = kv.first;
                }
            }

            Z_consensus[d][c] = best;
        }
    }

    Rcpp::Rcout << "\n" << std::string(70, '=') << "\n";
    Rcpp::Rcout << "ENSEMBLE COMPLETE\n";
    Rcpp::Rcout << std::string(70, '=') << "\n";

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Convert to hours, minutes, seconds
    int total_seconds = duration.count() / 1000;
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;

    Rcpp::Rcout << "Timing Summary:\n";
    Rcpp::Rcout << "  Total wall time:   ";
    if (hours > 0) {
        Rcpp::Rcout << hours << "h " << minutes << "m " << seconds << "s\n";
    } else if (minutes > 0) {
        Rcpp::Rcout << minutes << "m " << seconds << "s\n";
    } else {
        Rcpp::Rcout << seconds << "." << (duration.count() % 1000) << "s\n";
    }

    // Compute per-chain average
    double avg_time_per_chain = duration.count() / (double)num_chains / 1000.0;
    Rcpp::Rcout << "  Avg per chain:     " << std::fixed << std::setprecision(2)
                << avg_time_per_chain << "s\n";

    // Compute speedup from parallelization
    double theoretical_serial_time = avg_time_per_chain * num_chains;
    double actual_time = duration.count() / 1000.0;
    double speedup = theoretical_serial_time / actual_time;
    Rcpp::Rcout << "  Parallel speedup:  " << std::fixed << std::setprecision(2)
                << speedup << "x (using " << num_cores << " cores)\n";

    Rcpp::Rcout << std::string(70, '=') << "\n\n";

    // ===== BUILD RETURN LIST =====
    Rcpp::List result_list = Rcpp::List::create(
        Rcpp::_["b_mean"] = b_mean,
        Rcpp::_["b_sd"] = b_sd,
        Rcpp::_["alpha_phi2_mean"] = alpha_phi2_mean,
        Rcpp::_["alpha_phi2_sd"] = alpha_phi2_sd,
        Rcpp::_["P_J_D_mean"] = P_J_D_mean,
        Rcpp::_["P_J_D_sd"] = P_J_D_sd,
        Rcpp::_["P_mean"] = P_mean,
        Rcpp::_["P_sd"] = P_sd,
        Rcpp::_["alpha_mean"] = alpha_mean,
        Rcpp::_["alpha_sd"] = alpha_sd,
        Rcpp::_["alpha_zero_mean"] = alpha_zero_mean,
        Rcpp::_["alpha_zero_sd"] = alpha_zero_sd,
        Rcpp::_["mu_star_1_J_mean"] = mu_mean,
        Rcpp::_["mu_star_1_J_sd"] = mu_sd,
        Rcpp::_["phi_star_1_J_mean"] = phi_mean,
        Rcpp::_["phi_star_1_J_sd"] = phi_sd,
        Rcpp::_["Beta_mean"] = Beta_mean,
        Rcpp::_["Beta_sd"] = Beta_sd,
        Rcpp::_["Z_consensus"] = Z_consensus,
        Rcpp::_["coclustering"] = coclustering,
		Rcpp::_["b_trace_all"] = b_trace_all,
    	Rcpp::_["alpha_trace_all"] = alpha_trace_all,
    	Rcpp::_["alpha_zero_trace_all"] = alpha_zero_trace_all,
    	Rcpp::_["mu_trace_all"] = mu_trace_all,
    	Rcpp::_["phi_trace_all"] = phi_trace_all,
        Rcpp::_["Z_trace_all"] = Z_trace_all,
        Rcpp::_["use_sparse_prior"] = use_sparse_prior,
		Rcpp::_["use_spike_slab"] = use_spike_slab,
		Rcpp::_["use_reg_horseshoe"] = use_reg_horseshoe
    );

    // Add horseshoe parameters if sparse prior was used
    if (use_sparse_prior && lambda_mean.size() > 0) {
        result_list["lambda_mean"] = lambda_mean;
        result_list["lambda_sd"] = lambda_sd;
        result_list["tau_mean"] = tau_mean;
        result_list["tau_sd"] = tau_sd;
        result_list["sigma_mu_mean"] = sigma_mu_mean;
        result_list["sigma_mu_sd"] = sigma_mu_sd;
		result_list["lambda_trace_all"] = lambda_trace_all;
    	result_list["tau_trace_all"] = tau_trace_all;
    	result_list["sigma_mu_trace_all"] = sigma_mu_trace_all;
        
    }

	 if (use_spike_slab && gamma_mean.size() > 0) {
        result_list["gamma_mean"] = gamma_mean;
        result_list["gamma_sd"] = gamma_sd;
        result_list["pi_mean"] = pi_mean;
        result_list["pi_sd"] = pi_sd;
        result_list["sigma_slab_mean"] = sigma_slab_mean;
        result_list["sigma_slab_sd"] = sigma_slab_sd;
    }

	if (use_reg_horseshoe && rhs_lambda_mean.size() > 0) {
        result_list["rhs_lambda_mean"] = rhs_lambda_mean;
        result_list["rhs_lambda_sd"] = rhs_lambda_sd;
        result_list["rhs_tau_mean"] = rhs_tau_mean;
        result_list["rhs_tau_sd"] = rhs_tau_sd;
        result_list["rhs_tau_0_mean"] = rhs_tau_0_mean;
        result_list["rhs_tau_0_sd"] = rhs_tau_0_sd;
        result_list["rhs_sigma_mu_mean"] = rhs_sigma_mu_mean;
        result_list["rhs_sigma_mu_sd"] = rhs_sigma_mu_sd;
		result_list["lambda_trace_all"] = lambda_trace_all;
    	result_list["tau_trace_all"] = tau_trace_all;
    	result_list["sigma_mu_trace_all"] = sigma_mu_trace_all;

    }

    return result_list;
}