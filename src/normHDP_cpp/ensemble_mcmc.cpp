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

// ---------------------------------------------------------------------------
// THREADING NOTES
//
// The R API (anything in Rcpp:: / SEXP / R_alloc etc.) is strictly
// single-threaded. Every Rcpp call must happen on the main thread only.
// Threads may only use plain C++ and Eigen.
//
// The mutex is local to each call of ensemble_mcmc_R so it cannot survive
// in a corrupt/locked state between R-level calls. The old file-scope global
// mutex was the primary cause of the batch-3 crash: if R ever unwinds the
// call stack (exception, interrupt, GC), the global mutex is left in a
// locked or destroyed state, and the next call deadlocks or crashes.
// ---------------------------------------------------------------------------

struct ChainResult {
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
    bool use_sparse_prior  = false;
    bool use_spike_slab    = false;
    bool use_reg_horseshoe = false;
    int  D                 = 0;
    bool failed            = false;  // set true when thread catches an exception
};

// ---------------------------------------------------------------------------
// run_single_chain_safe
// Pure C++ / Eigen only — ZERO Rcpp calls are allowed in this function.
// The mutex is owned by the caller (ensemble_mcmc_R stack frame).
// ---------------------------------------------------------------------------
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
    bool print_progress,
    std::mutex& io_mutex   // caller-owned stack-local mutex, NOT a global
) {
    if (print_progress) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Starting chain " << (chain_id + 1) << std::endl;
    }

    ChainResult chain_result;

    try {
        // rng_local is thread_local (utils.cpp) so each chain has its own RNG.
        // Serialise only the seed call, then run the full MCMC in parallel.
        // NOTE: Do NOT redirect std::cout here — swapping rdbuf from multiple
        // threads concurrently is itself not thread-safe and causes crashes.
        // iter_update=999999 already suppresses per-iteration prints inside
        // normHDP_mcmc; the few remaining cout calls (start/end messages)
        // may interleave in output but are harmless.
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            init_rng(std::random_device{}() + chain_id * 1000);
        }

        NormHDPResult result = normHDP_mcmc(
            Y, J, chain_length, thinning, empirical, burn_in, quadratic,
            999999, beta_mean, alpha_mu_2, adaptive_prop,
            false, 1, save_only_z,
            use_sparse_prior, use_spike_slab, use_reg_horseshoe, p_0,
            baynorm_mu_estimate, baynorm_phi_estimate, baynorm_beta
        );

        chain_result.use_sparse_prior  = result.use_sparse_prior;
        chain_result.use_spike_slab    = result.use_spike_slab;
        chain_result.use_reg_horseshoe = result.use_reg_horseshoe;
        chain_result.D                 = result.D;
        chain_result.b_output            = result.b_output;
        chain_result.alpha_phi2_output   = result.alpha_phi2_output;
        chain_result.P_J_D_output        = result.P_J_D_output;
        chain_result.P_output            = result.P_output;
        chain_result.alpha_output        = result.alpha_output;
        chain_result.alpha_zero_output   = result.alpha_zero_output;
        chain_result.mu_star_1_J_output  = result.mu_star_1_J_output;
        chain_result.phi_star_1_J_output = result.phi_star_1_J_output;
        chain_result.Beta_output         = result.Beta_output;
        chain_result.Z_output            = result.Z_output;
        chain_result.mu_baseline_output  = result.mu_baseline_output;
        if (result.use_sparse_prior  && !result.horseshoe_output.empty())
            chain_result.horseshoe_output     = result.horseshoe_output;
        if (result.use_spike_slab    && !result.spike_slab_output.empty())
            chain_result.spike_slab_output    = result.spike_slab_output;
        if (result.use_reg_horseshoe && !result.reg_horseshoe_output.empty())
            chain_result.reg_horseshoe_output = result.reg_horseshoe_output;

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Chain " << (chain_id + 1) << " failed: " << e.what() << std::endl;
        chain_result.failed = true;
    } catch (...) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cerr << "Chain " << (chain_id + 1) << " failed with unknown error" << std::endl;
        chain_result.failed = true;
    }

    if (print_progress) {
        std::lock_guard<std::mutex> lock(io_mutex);
        std::cout << "Finished chain " << (chain_id + 1) << std::endl;
    }

    return chain_result;
}

// ---------------------------------------------------------------------------
// compute_mean_sd
//
// Bug fix: original used T::Zero(rows, cols). This is fine for MatrixXd but
// VectorXd::Zero() only takes one argument — calling it with two is UB and
// crashes at runtime. Using `sd = mean; sd.setZero()` works for both types.
// ---------------------------------------------------------------------------
template<typename T>
void compute_mean_sd(const std::vector<T>& samples, T& mean_out, T& sd_out) {
    int n = static_cast<int>(samples.size());
    if (n == 0) return;

    mean_out = samples[0];
    for (int i = 1; i < n; ++i) mean_out += samples[i];
    mean_out /= n;

    sd_out = mean_out;   // same shape as mean, works for Vector and Matrix
    sd_out.setZero();

    if (n == 1) return;

    for (int i = 0; i < n; ++i)
        sd_out += (samples[i] - mean_out).cwiseProduct(samples[i] - mean_out);
    sd_out = (sd_out / (n - 1)).cwiseSqrt();
}

void compute_scalar_mean_sd(const std::vector<double>& samples,
                             double& mean_out, double& sd_out) {
    int n = static_cast<int>(samples.size());
    if (n == 0) { mean_out = 0.0; sd_out = 0.0; return; }

    mean_out = 0.0;
    for (double s : samples) mean_out += s;
    mean_out /= n;

    if (n == 1) { sd_out = 0.0; return; }

    sd_out = 0.0;
    for (double s : samples) sd_out += (s - mean_out) * (s - mean_out);
    sd_out = std::sqrt(sd_out / (n - 1));
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
    // DIAGNOSTIC: flush immediately so we know the function was entered
    REprintf("ensemble_mcmc_R: entered, D=%d J=%d num_chains=%d sparse=%d ss=%d rhs=%d\n",
             (int)Y.size(), J, num_chains,
             (int)use_sparse_prior, (int)use_spike_slab, (int)use_reg_horseshoe);

    if ((int)use_sparse_prior + (int)use_spike_slab + (int)use_reg_horseshoe > 1)
        Rcpp::stop("Only one of use_sparse_prior, use_spike_slab, use_reg_horseshoe may be TRUE.");

    // ------------------------------------------------------------------
    // Convert all R objects to plain C++ BEFORE any threading begins.
    // No SEXP / Rcpp object may be read or written inside a thread.
    // ------------------------------------------------------------------
    int D = Y.size();
    REprintf("ensemble_mcmc_R: converting Y (%d datasets)...\n", D);

    std::vector<Eigen::MatrixXd> Y_vec(D);
    for (int d = 0; d < D; d++) {
        REprintf("  Y[%d]...\n", d);
        Y_vec[d] = Rcpp::as<Eigen::MatrixXd>(Y[d]);
    }
    REprintf("ensemble_mcmc_R: Y converted\n");

    std::vector<Eigen::VectorXd> baynorm_beta(D);
    for (int d = 0; d < D; d++) {
        REprintf("  beta[%d]...\n", d);
        baynorm_beta[d] = Rcpp::as<Eigen::VectorXd>(baynorm_beta_list[d]);
    }
    REprintf("ensemble_mcmc_R: beta converted\n");

    Eigen::VectorXd mu_vec  = Rcpp::as<Eigen::VectorXd>(baynorm_mu_estimate);
    Eigen::VectorXd phi_vec = Rcpp::as<Eigen::VectorXd>(baynorm_phi_estimate);

    int G = Y_vec[0].rows();
    std::vector<int> C(D);
    for (int d = 0; d < D; d++) C[d] = Y_vec[d].cols();

    if (num_cores <= 0)
        num_cores = static_cast<int>(std::thread::hardware_concurrency());

    Rcpp::Rcout << "\n" << std::string(70, '=') << "\n";
    Rcpp::Rcout << "ENSEMBLE MCMC"
                << (use_sparse_prior  ? " (WITH HORSESHOE PRIOR)"       : "")
                << (use_spike_slab    ? " (WITH SPIKE-AND-SLAB PRIOR)"  : "")
                << (use_reg_horseshoe ? " (WITH REG. HORSESHOE PRIOR)"  : "")
                << "\n";
    Rcpp::Rcout << std::string(70, '=') << "\n";
    Rcpp::Rcout << "Configuration:\n";
    Rcpp::Rcout << "  Number of chains:  " << num_chains   << "\n";
    Rcpp::Rcout << "  Chain length:      " << chain_length << " iterations\n";
    Rcpp::Rcout << "  Burn-in:           " << burn_in      << " iterations\n";
    Rcpp::Rcout << "  Thinning:          " << thinning     << "\n";
    Rcpp::Rcout << "  Parallel cores:    " << num_cores    << "\n";
    Rcpp::Rcout << "  Datasets:          " << D            << "\n";
    Rcpp::Rcout << "  Genes:             " << G            << "\n";
    Rcpp::Rcout << "  Clusters:          " << J            << "\n";
    Rcpp::Rcout << "  Sparse prior:      " << (use_sparse_prior  ? "Horseshoe"      : "NO") << "\n";
    Rcpp::Rcout << "  Spike-Slab prior:  " << (use_spike_slab    ? "spike slab"     : "NO") << "\n";
    Rcpp::Rcout << "  Reg-Horseshoe:     " << (use_reg_horseshoe ? "reg. horseshoe" : "NO") << "\n";
    if (use_reg_horseshoe)
        Rcpp::Rcout << "    p_0 (sparsity):  " << p_0 << "\n";
    Rcpp::Rcout << std::string(70, '=') << "\n\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<ChainResult> chain_results(num_chains);

    // ------------------------------------------------------------------
    // BATCH LOOP
    //
    // io_mutex is stack-local to this function call. It cannot be left
    // locked or corrupted between R-level invocations of ensemble_mcmc_R,
    // which was the root cause of the crash at batch 3.
    // ------------------------------------------------------------------
    std::mutex io_mutex;

    for (int batch_start = 0; batch_start < num_chains; batch_start += num_cores) {
        int batch_end = std::min(batch_start + num_cores, num_chains);

        Rcpp::Rcout << "Running chains " << (batch_start + 1)
                    << " to " << batch_end << "...\n";

        std::vector<std::thread> threads;
        threads.reserve(batch_end - batch_start);

        for (int i = batch_start; i < batch_end; i++) {
            threads.emplace_back([&, i]() {
                chain_results[i] = run_single_chain_safe(
                    Y_vec, J, chain_length, thinning, empirical, burn_in, quadratic,
                    beta_mean, alpha_mu_2, adaptive_prop, save_only_z,
                    use_sparse_prior, use_spike_slab, use_reg_horseshoe, p_0,
                    mu_vec, phi_vec, baynorm_beta,
                    i, print_progress,
                    io_mutex
                );
            });
        }

        // Join ALL threads before touching results or calling into the R API.
        for (auto& t : threads)
            if (t.joinable()) t.join();

        Rcpp::Rcout << "Batch completed (" << batch_end << "/" << num_chains
                    << " chains done)\n\n";

        // checkUserInterrupt may longjmp — only safe after all threads are
        // joined and we are back on the single main R thread.
        Rcpp::checkUserInterrupt();
    }

    // ------------------------------------------------------------------
    // From here onward: single-threaded, main R thread only.
    // Convert plain C++ results to R objects.
    // ------------------------------------------------------------------
    Rcpp::Rcout << "All chains completed. Computing ensemble statistics...\n";

    // Z traces — always collected regardless of save_only_z
    Rcpp::List Z_trace_all(num_chains);
    for (int i = 0; i < num_chains; i++) {
        const auto& res = chain_results[i];
        int n_samp = static_cast<int>(res.Z_output.size());
        Rcpp::List Z_trace_i(n_samp);
        for (int t = 0; t < n_samp; t++) {
            Rcpp::List Z_t(D);
            for (int d = 0; d < D; d++)
                Z_t[d] = Rcpp::wrap(res.Z_output[t][d]);
            Z_trace_i[t] = Z_t;
        }
        Z_trace_all[i] = Z_trace_i;
    }

    // Parameter traces — only populated when save_only_z is false.
    // Bug fix: previously these were default-constructed empty Rcpp::List
    // objects and then unconditionally stuffed into result_list. Wrapping a
    // null/empty SEXP into another list segfaults when save_only_z=true.
    Rcpp::List b_trace_all, alpha_trace_all, alpha_zero_trace_all;
    Rcpp::List mu_trace_all, phi_trace_all;
    Rcpp::List lambda_trace_all, tau_trace_all, sigma_mu_trace_all;

    if (!save_only_z) {
        b_trace_all          = Rcpp::List(num_chains);
        alpha_trace_all      = Rcpp::List(num_chains);
        alpha_zero_trace_all = Rcpp::List(num_chains);
        mu_trace_all         = Rcpp::List(num_chains);
        phi_trace_all        = Rcpp::List(num_chains);

        if (use_sparse_prior || use_reg_horseshoe) {
            lambda_trace_all   = Rcpp::List(num_chains);
            tau_trace_all      = Rcpp::List(num_chains);
            sigma_mu_trace_all = Rcpp::List(num_chains);
        }

        for (int i = 0; i < num_chains; i++) {
            const auto& res = chain_results[i];
            if (res.failed || res.b_output.empty()) continue;

            b_trace_all[i]          = Rcpp::wrap(res.b_output);
            alpha_trace_all[i]      = Rcpp::wrap(res.alpha_output);
            alpha_zero_trace_all[i] = Rcpp::wrap(res.alpha_zero_output);
            mu_trace_all[i]         = Rcpp::wrap(res.mu_star_1_J_output);
            phi_trace_all[i]        = Rcpp::wrap(res.phi_star_1_J_output);

            // Horseshoe traces
            if (use_sparse_prior && !res.horseshoe_output.empty()) {
                try {
                    int n_hs = static_cast<int>(res.horseshoe_output.size());
                    bool ok  = (res.horseshoe_output[0].lambda.size() == static_cast<size_t>(J));
                    for (int j = 0; j < J && ok; j++)
                        ok = (res.horseshoe_output[0].lambda[j].size() == static_cast<size_t>(G));

                    if (ok) {
                        Rcpp::List lchain(n_hs);
                        Eigen::MatrixXd tau_mat(n_hs, J);
                        Eigen::VectorXd smu_vec(n_hs);
                        for (int t = 0; t < n_hs; t++) {
                            Eigen::MatrixXd lm(J, G);
                            for (int j = 0; j < J; j++) {
                                if ((int)res.horseshoe_output[t].lambda[j].size() == G)
                                    lm.row(j) = res.horseshoe_output[t].lambda[j].transpose();
                                else
                                    lm.row(j).setOnes();
                            }
                            lchain[t] = lm;
                            for (int j = 0; j < J; j++)
                                tau_mat(t, j) = ((int)res.horseshoe_output[t].tau.size() == J)
                                    ? res.horseshoe_output[t].tau[j] : 1.0;
                            smu_vec(t) = res.horseshoe_output[t].sigma_mu;
                        }
                        lambda_trace_all[i]   = lchain;
                        tau_trace_all[i]      = tau_mat;
                        sigma_mu_trace_all[i] = smu_vec;
                    }
                } catch (const std::exception& e) {
                    Rcpp::Rcout << "Warning: horseshoe trace chain " << (i+1)
                                << ": " << e.what() << "\n";
                }
            }

            // Regularised horseshoe traces
            if (use_reg_horseshoe && !res.reg_horseshoe_output.empty()) {
                try {
                    int n_rh = static_cast<int>(res.reg_horseshoe_output.size());
                    bool ok  = (res.reg_horseshoe_output[0].lambda.size() == static_cast<size_t>(J));
                    for (int j = 0; j < J && ok; j++)
                        ok = (res.reg_horseshoe_output[0].lambda[j].size() == static_cast<size_t>(G));

                    if (ok) {
                        Rcpp::List lchain(n_rh);
                        Eigen::MatrixXd tau_mat(n_rh, J);
                        Eigen::VectorXd smu_vec(n_rh);
                        for (int t = 0; t < n_rh; t++) {
                            Eigen::MatrixXd lm(J, G);
                            for (int j = 0; j < J; j++) {
                                if ((int)res.reg_horseshoe_output[t].lambda[j].size() == G)
                                    lm.row(j) = res.reg_horseshoe_output[t].lambda[j].transpose();
                                else
                                    lm.row(j).setOnes();
                            }
                            lchain[t] = lm;
                            for (int j = 0; j < J; j++)
                                tau_mat(t, j) = ((int)res.reg_horseshoe_output[t].tau.size() == J)
                                    ? res.reg_horseshoe_output[t].tau[j] : 1.0;
                            smu_vec(t) = res.reg_horseshoe_output[t].sigma_mu;
                        }
                        lambda_trace_all[i]   = lchain;
                        tau_trace_all[i]      = tau_mat;
                        sigma_mu_trace_all[i] = smu_vec;
                    }
                } catch (const std::exception& e) {
                    Rcpp::Rcout << "Warning: reg-horseshoe trace chain " << (i+1)
                                << ": " << e.what() << "\n";
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Posterior summaries — mean / SD across chains using last sample
    // ------------------------------------------------------------------
    std::vector<Eigen::VectorXd> b_samp, P_samp;
    std::vector<Eigen::MatrixXd> PJD_samp, mu_samp, phi_samp;
    std::vector<double> aphi2_samp, alpha_samp, azero_samp;
    std::vector<std::vector<Eigen::VectorXd>> Beta_samp(D);

    for (int i = 0; i < num_chains; i++) {
        const auto& r = chain_results[i];
        if (r.failed || r.b_output.empty()) continue;
        int last = static_cast<int>(r.b_output.size()) - 1;

        b_samp.push_back(r.b_output[last]);
        aphi2_samp.push_back(r.alpha_phi2_output[last]);
        PJD_samp.push_back(r.P_J_D_output[last]);
        P_samp.push_back(r.P_output[last]);
        alpha_samp.push_back(r.alpha_output[last]);
        azero_samp.push_back(r.alpha_zero_output[last]);
        mu_samp.push_back(r.mu_star_1_J_output[last]);
        phi_samp.push_back(r.phi_star_1_J_output[last]);
        for (int d = 0; d < D; d++)
            Beta_samp[d].push_back(r.Beta_output[last][d]);
    }

    Eigen::VectorXd b_mean, b_sd, P_mean, P_sd;
    Eigen::MatrixXd PJD_mean, PJD_sd, mu_mean, mu_sd, phi_mean, phi_sd;
    double aphi2_mean, aphi2_sd, alpha_mean, alpha_sd, azero_mean, azero_sd;

    compute_mean_sd(b_samp,   b_mean,   b_sd);
    compute_mean_sd(PJD_samp, PJD_mean, PJD_sd);
    compute_mean_sd(P_samp,   P_mean,   P_sd);
    compute_mean_sd(mu_samp,  mu_mean,  mu_sd);
    compute_mean_sd(phi_samp, phi_mean, phi_sd);
    compute_scalar_mean_sd(aphi2_samp, aphi2_mean, aphi2_sd);
    compute_scalar_mean_sd(alpha_samp, alpha_mean, alpha_sd);
    compute_scalar_mean_sd(azero_samp, azero_mean, azero_sd);

    std::vector<Eigen::VectorXd> Beta_mean(D), Beta_sd(D);
    for (int d = 0; d < D; d++)
        compute_mean_sd(Beta_samp[d], Beta_mean[d], Beta_sd[d]);

    // Horseshoe summary
    Eigen::MatrixXd lambda_mean, lambda_sd;
    Eigen::VectorXd tau_mean, tau_sd;
    double sigma_mu_mean = 0.0, sigma_mu_sd = 0.0;

    if (use_sparse_prior && !chain_results.empty() &&
        !chain_results[0].horseshoe_output.empty()) {
        Rcpp::Rcout << "Computing horseshoe parameter statistics...\n";
        std::vector<Eigen::MatrixXd> lsamp;
        std::vector<Eigen::VectorXd> tsamp;
        std::vector<double> ssamp;
        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            if (r.failed || r.horseshoe_output.empty()) continue;
            const auto& hs = r.horseshoe_output.back();
            Eigen::MatrixXd lm(J, G);
            for (int j = 0; j < J; j++) lm.row(j) = hs.lambda[j].transpose();
            lsamp.push_back(lm);
            Eigen::VectorXd tv(J);
            for (int j = 0; j < J; j++) tv(j) = hs.tau[j];
            tsamp.push_back(tv);
            ssamp.push_back(hs.sigma_mu);
        }
        compute_mean_sd(lsamp, lambda_mean, lambda_sd);
        compute_mean_sd(tsamp, tau_mean,    tau_sd);
        compute_scalar_mean_sd(ssamp, sigma_mu_mean, sigma_mu_sd);
    }

    // Spike-and-slab summary
    Eigen::MatrixXd gamma_mean, gamma_sd;
    double pi_mean = 0.0, pi_sd = 0.0, ss_sigma_mean = 0.0, ss_sigma_sd = 0.0;

    if (use_spike_slab && !chain_results.empty() &&
        !chain_results[0].spike_slab_output.empty()) {
        Rcpp::Rcout << "Computing spike-and-slab parameter statistics...\n";
        std::vector<Eigen::MatrixXd> gsamp;
        std::vector<double> pisamp, sigsamp;
        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            if (r.failed || r.spike_slab_output.empty()) continue;
            const auto& ss = r.spike_slab_output.back();
            Eigen::MatrixXd gm(J, G);
            for (int j = 0; j < J; j++) gm.row(j) = ss.gamma[j].col(0).transpose();
            gsamp.push_back(gm);
            pisamp.push_back(ss.pi);
            sigsamp.push_back(ss.sigma_slab);
        }
        compute_mean_sd(gsamp, gamma_mean, gamma_sd);
        compute_scalar_mean_sd(pisamp,  pi_mean,       pi_sd);
        compute_scalar_mean_sd(sigsamp, ss_sigma_mean, ss_sigma_sd);
    }

    // Regularised horseshoe summary
    Eigen::MatrixXd rhs_lambda_mean, rhs_lambda_sd;
    Eigen::VectorXd rhs_tau_mean, rhs_tau_sd;
    double rhs_tau0_mean = 0.0, rhs_tau0_sd = 0.0;
    double rhs_sigma_mean = 0.0, rhs_sigma_sd = 0.0;

    if (use_reg_horseshoe && !chain_results.empty() &&
        !chain_results[0].reg_horseshoe_output.empty()) {
        Rcpp::Rcout << "Computing regularized horseshoe parameter statistics...\n";
        std::vector<Eigen::MatrixXd> lsamp;
        std::vector<Eigen::VectorXd> tsamp;
        std::vector<double> t0samp, ssamp;
        for (int i = 0; i < num_chains; i++) {
            const auto& r = chain_results[i];
            if (r.failed || r.reg_horseshoe_output.empty()) continue;
            const auto& rhs = r.reg_horseshoe_output.back();
            Eigen::MatrixXd lm(J, G);
            for (int j = 0; j < J; j++) lm.row(j) = rhs.lambda[j].transpose();
            lsamp.push_back(lm);
            Eigen::VectorXd tv(J);
            for (int j = 0; j < J; j++) tv(j) = rhs.tau[j];
            tsamp.push_back(tv);
            t0samp.push_back(rhs.tau_0);
            ssamp.push_back(rhs.sigma_mu);
        }
        compute_mean_sd(lsamp, rhs_lambda_mean, rhs_lambda_sd);
        compute_mean_sd(tsamp, rhs_tau_mean,    rhs_tau_sd);
        compute_scalar_mean_sd(t0samp, rhs_tau0_mean,  rhs_tau0_sd);
        compute_scalar_mean_sd(ssamp,  rhs_sigma_mean, rhs_sigma_sd);
    }

    // ------------------------------------------------------------------
    // Consensus clustering
    // Bug fix: guard against chains that failed (Z_output is empty).
    // Calling .back() on an empty vector is UB and crashes.
    // ------------------------------------------------------------------
    Rcpp::Rcout << "Computing consensus clustering...\n";

    std::vector<std::vector<int>> Z_consensus(D);
    std::vector<Eigen::MatrixXd>  coclustering(D);

    for (int d = 0; d < D; d++) {
        int C_d = C[d];
        Eigen::MatrixXd cocluster = Eigen::MatrixXd::Zero(C_d, C_d);
        int valid_chains = 0;

        for (int chain = 0; chain < num_chains; chain++) {
            if (chain_results[chain].Z_output.empty()) continue;  // guard
            ++valid_chains;
            const auto& Z_last = chain_results[chain].Z_output.back()[d];
            for (int c1 = 0; c1 < C_d; c1++)
                for (int c2 = c1; c2 < C_d; c2++)
                    if (Z_last[c1] == Z_last[c2]) {
                        cocluster(c1, c2) += 1.0;
                        if (c1 != c2) cocluster(c2, c1) += 1.0;
                    }
        }

        if (valid_chains > 0) cocluster /= valid_chains;
        coclustering[d] = cocluster;

        Z_consensus[d].resize(C_d, 0);
        for (int c = 0; c < C_d; c++) {
            std::map<int, int> freq;
            for (int chain = 0; chain < num_chains; chain++) {
                if (chain_results[chain].Z_output.empty()) continue;  // guard
                freq[chain_results[chain].Z_output.back()[d][c]]++;
            }
            int best = 0, maxc = 0;
            for (auto& kv : freq)
                if (kv.second > maxc) { maxc = kv.second; best = kv.first; }
            Z_consensus[d][c] = best;
        }
    }

    // ------------------------------------------------------------------
    // Timing report
    // ------------------------------------------------------------------
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    int tsec = static_cast<int>(duration.count()) / 1000;

    Rcpp::Rcout << "\n" << std::string(70, '=') << "\n";
    Rcpp::Rcout << "ENSEMBLE COMPLETE\n";
    Rcpp::Rcout << std::string(70, '=') << "\n";
    Rcpp::Rcout << "Timing Summary:\n  Total wall time:   ";
    if (tsec >= 3600)
        Rcpp::Rcout << (tsec/3600) << "h " << ((tsec%3600)/60) << "m " << (tsec%60) << "s\n";
    else if (tsec >= 60)
        Rcpp::Rcout << (tsec/60) << "m " << (tsec%60) << "s\n";
    else
        Rcpp::Rcout << tsec << "." << (duration.count() % 1000) << "s\n";

    double avg     = duration.count() / (double)num_chains / 1000.0;
    double speedup = (avg * num_chains) / (duration.count() / 1000.0);
    Rcpp::Rcout << "  Avg per chain:     " << std::fixed << std::setprecision(2) << avg << "s\n";
    Rcpp::Rcout << "  Parallel speedup:  " << std::setprecision(2) << speedup
                << "x (using " << num_cores << " cores)\n";
    Rcpp::Rcout << std::string(70, '=') << "\n\n";

    // ------------------------------------------------------------------
    // Build and return result list
    // ------------------------------------------------------------------
    Rcpp::List result_list = Rcpp::List::create(
        Rcpp::_["b_mean"]            = b_mean,
        Rcpp::_["b_sd"]              = b_sd,
        Rcpp::_["alpha_phi2_mean"]   = aphi2_mean,
        Rcpp::_["alpha_phi2_sd"]     = aphi2_sd,
        Rcpp::_["P_J_D_mean"]        = PJD_mean,
        Rcpp::_["P_J_D_sd"]          = PJD_sd,
        Rcpp::_["P_mean"]            = P_mean,
        Rcpp::_["P_sd"]              = P_sd,
        Rcpp::_["alpha_mean"]        = alpha_mean,
        Rcpp::_["alpha_sd"]          = alpha_sd,
        Rcpp::_["alpha_zero_mean"]   = azero_mean,
        Rcpp::_["alpha_zero_sd"]     = azero_sd,
        Rcpp::_["mu_star_1_J_mean"]  = mu_mean,
        Rcpp::_["mu_star_1_J_sd"]    = mu_sd,
        Rcpp::_["phi_star_1_J_mean"] = phi_mean,
        Rcpp::_["phi_star_1_J_sd"]   = phi_sd,
        Rcpp::_["Beta_mean"]         = Beta_mean,
        Rcpp::_["Beta_sd"]           = Beta_sd,
        Rcpp::_["Z_consensus"]       = Z_consensus,
        Rcpp::_["coclustering"]      = coclustering,
        Rcpp::_["Z_trace_all"]       = Z_trace_all,
        Rcpp::_["use_sparse_prior"]  = use_sparse_prior,
        Rcpp::_["use_spike_slab"]    = use_spike_slab,
        Rcpp::_["use_reg_horseshoe"] = use_reg_horseshoe
    );

    // Traces only added when save_only_z=false (otherwise lists are
    // uninitialised and wrapping them into result_list segfaults).
    if (!save_only_z) {
        result_list["b_trace_all"]          = b_trace_all;
        result_list["alpha_trace_all"]      = alpha_trace_all;
        result_list["alpha_zero_trace_all"] = alpha_zero_trace_all;
        result_list["mu_trace_all"]         = mu_trace_all;
        result_list["phi_trace_all"]        = phi_trace_all;
    }

    if (use_sparse_prior && lambda_mean.size() > 0) {
        result_list["lambda_mean"]   = lambda_mean;
        result_list["lambda_sd"]     = lambda_sd;
        result_list["tau_mean"]      = tau_mean;
        result_list["tau_sd"]        = tau_sd;
        result_list["sigma_mu_mean"] = sigma_mu_mean;
        result_list["sigma_mu_sd"]   = sigma_mu_sd;
        if (!save_only_z) {
            result_list["lambda_trace_all"]   = lambda_trace_all;
            result_list["tau_trace_all"]      = tau_trace_all;
            result_list["sigma_mu_trace_all"] = sigma_mu_trace_all;
        }
    }

    if (use_spike_slab && gamma_mean.size() > 0) {
        result_list["gamma_mean"]      = gamma_mean;
        result_list["gamma_sd"]        = gamma_sd;
        result_list["pi_mean"]         = pi_mean;
        result_list["pi_sd"]           = pi_sd;
        result_list["sigma_slab_mean"] = ss_sigma_mean;
        result_list["sigma_slab_sd"]   = ss_sigma_sd;
    }

    if (use_reg_horseshoe && rhs_lambda_mean.size() > 0) {
        result_list["rhs_lambda_mean"]   = rhs_lambda_mean;
        result_list["rhs_lambda_sd"]     = rhs_lambda_sd;
        result_list["rhs_tau_mean"]      = rhs_tau_mean;
        result_list["rhs_tau_sd"]        = rhs_tau_sd;
        result_list["rhs_tau_0_mean"]    = rhs_tau0_mean;
        result_list["rhs_tau_0_sd"]      = rhs_tau0_sd;
        result_list["rhs_sigma_mu_mean"] = rhs_sigma_mean;
        result_list["rhs_sigma_mu_sd"]   = rhs_sigma_sd;
        if (!save_only_z) {
            result_list["lambda_trace_all"]   = lambda_trace_all;
            result_list["tau_trace_all"]      = tau_trace_all;
            result_list["sigma_mu_trace_all"] = sigma_mu_trace_all;
        }
    }

    return result_list;
}