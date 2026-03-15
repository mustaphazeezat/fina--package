//
// Created by Azeezat Mustapha on 15.03.26.
//

#include <Rcpp.h>
#include <RcppEigen.h>
#include "includes/normHDP_mcmc.h"
#include "includes/utils.h"

// [[Rcpp::depends(RcppEigen)]]

// ---------------------------------------------------------------------------
// normHDP_mcmc_R
//
// Thin R-callable wrapper around the internal normHDP_mcmc() C++ function.
// All parallelism (running multiple chains) is handled in R via
// parallel::mclapply or future/furrr — no threading inside this function.
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List normHDP_mcmc_R(
    Rcpp::List          Y_list,
    int                 J,
    int                 chain_length,
    int                 thinning,
    bool                empirical,
    int                 burn_in,
    bool                quadratic,
    int                 iter_update,
    double              beta_mean,
    double              alpha_mu_2,
    double              adaptive_prop,
    bool                print_Z,
    bool                save_only_z,
    bool                use_sparse_prior,
    bool                use_spike_slab,
    bool                use_reg_horseshoe,
    double              p_0,
    Rcpp::NumericVector baynorm_mu_estimate,
    Rcpp::NumericVector baynorm_phi_estimate,
    Rcpp::List          baynorm_beta_list,
    int                 seed = -1          // -1 = random seed
) {
    // Seed the thread-local RNG before doing anything else.
    // Called on the main R thread, no concurrency concern here.
    if (seed < 0) {
        init_rng(std::random_device{}());
    } else {
        init_rng(static_cast<unsigned int>(seed));
    }

    // ── Convert R objects to plain C++ ──────────────────────────────────────
    int D = Y_list.size();

    std::vector<Eigen::MatrixXd> Y(D);
    for (int d = 0; d < D; d++)
        Y[d] = Rcpp::as<Eigen::MatrixXd>(Y_list[d]);

    std::vector<Eigen::VectorXd> baynorm_beta(D);
    for (int d = 0; d < D; d++)
        baynorm_beta[d] = Rcpp::as<Eigen::VectorXd>(baynorm_beta_list[d]);

    Eigen::VectorXd mu_vec  = Rcpp::as<Eigen::VectorXd>(baynorm_mu_estimate);
    Eigen::VectorXd phi_vec = Rcpp::as<Eigen::VectorXd>(baynorm_phi_estimate);

    // ── Run single chain ────────────────────────────────────────────────────
    // num_cores = 1: no sub-threading inside the chain. Parallelism across
    // chains is handled entirely by R (mclapply / furrr).
    NormHDPResult res = normHDP_mcmc(
        Y, J, chain_length, thinning, empirical, burn_in, quadratic,
        iter_update, beta_mean, alpha_mu_2, adaptive_prop,
        print_Z,
        /*num_cores=*/ 1,
        save_only_z,
        use_sparse_prior, use_spike_slab, use_reg_horseshoe,
        p_0,
        mu_vec, phi_vec, baynorm_beta
    );

    // ── Convert C++ result to R list ────────────────────────────────────────
    Rcpp::List out = Rcpp::List::create(
        Rcpp::_["Z_output"]          = res.Z_output,
        Rcpp::_["use_sparse_prior"]  = res.use_sparse_prior,
        Rcpp::_["use_spike_slab"]    = res.use_spike_slab,
        Rcpp::_["use_reg_horseshoe"] = res.use_reg_horseshoe,
        Rcpp::_["D"]                 = res.D,
        Rcpp::_["G"]                 = res.G
    );

    if (!save_only_z) {
        out["b_output"]            = Rcpp::wrap(res.b_output);
        out["alpha_phi2_output"]   = Rcpp::wrap(res.alpha_phi2_output);
        out["P_J_D_output"]        = Rcpp::wrap(res.P_J_D_output);
        out["P_output"]            = Rcpp::wrap(res.P_output);
        out["alpha_output"]        = Rcpp::wrap(res.alpha_output);
        out["alpha_zero_output"]   = Rcpp::wrap(res.alpha_zero_output);
        out["mu_star_1_J_output"]  = Rcpp::wrap(res.mu_star_1_J_output);
        out["phi_star_1_J_output"] = Rcpp::wrap(res.phi_star_1_J_output);
        out["Beta_output"]         = Rcpp::wrap(res.Beta_output);
        out["mu_baseline_output"]  = Rcpp::wrap(res.mu_baseline_output);

        if (res.use_sparse_prior && !res.horseshoe_output.empty()) {
            // Convert horseshoe traces: list of lambda matrices + tau/sigma vecs
            int n  = res.horseshoe_output.size();
            int Jj = res.horseshoe_output[0].tau.size();
            int G  = res.horseshoe_output[0].lambda[0].size();

            Rcpp::List lambda_trace(n);
            Eigen::MatrixXd tau_trace(n, Jj);
            Eigen::VectorXd sigma_trace(n);

            for (int t = 0; t < n; t++) {
                Eigen::MatrixXd lm(Jj, G);
                for (int j = 0; j < Jj; j++)
                    lm.row(j) = res.horseshoe_output[t].lambda[j].transpose();
                lambda_trace[t] = lm;
                for (int j = 0; j < Jj; j++)
                    tau_trace(t, j) = res.horseshoe_output[t].tau[j];
                sigma_trace(t) = res.horseshoe_output[t].sigma_mu;
            }
            out["horseshoe_lambda_trace"] = lambda_trace;
            out["horseshoe_tau_trace"]    = tau_trace;
            out["horseshoe_sigma_trace"]  = sigma_trace;
        }

        if (res.use_spike_slab && !res.spike_slab_output.empty()) {
            int n  = res.spike_slab_output.size();
            int Jj = res.spike_slab_output[0].gamma.size();
            int G  = res.spike_slab_output[0].gamma[0].rows();

            Rcpp::List gamma_trace(n);
            Eigen::VectorXd pi_trace(n), sigma_slab_trace(n);

            for (int t = 0; t < n; t++) {
                Eigen::MatrixXd gm(Jj, G);
                for (int j = 0; j < Jj; j++)
                    gm.row(j) = res.spike_slab_output[t].gamma[j].col(0).transpose();
                gamma_trace[t]       = gm;
                pi_trace(t)          = res.spike_slab_output[t].pi;
                sigma_slab_trace(t)  = res.spike_slab_output[t].sigma_slab;
            }
            out["spike_slab_gamma_trace"]      = gamma_trace;
            out["spike_slab_pi_trace"]         = pi_trace;
            out["spike_slab_sigma_slab_trace"] = sigma_slab_trace;
        }

        if (res.use_reg_horseshoe && !res.reg_horseshoe_output.empty()) {
            int n  = res.reg_horseshoe_output.size();
            int Jj = res.reg_horseshoe_output[0].tau.size();
            int G  = res.reg_horseshoe_output[0].lambda[0].size();

            Rcpp::List lambda_trace(n);
            Eigen::MatrixXd tau_trace(n, Jj);
            Eigen::VectorXd tau0_trace(n), sigma_trace(n);

            for (int t = 0; t < n; t++) {
                Eigen::MatrixXd lm(Jj, G);
                for (int j = 0; j < Jj; j++)
                    lm.row(j) = res.reg_horseshoe_output[t].lambda[j].transpose();
                lambda_trace[t] = lm;
                for (int j = 0; j < Jj; j++)
                    tau_trace(t, j) = res.reg_horseshoe_output[t].tau[j];
                tau0_trace(t)  = res.reg_horseshoe_output[t].tau_0;
                sigma_trace(t) = res.reg_horseshoe_output[t].sigma_mu;
            }
            out["rhs_lambda_trace"] = lambda_trace;
            out["rhs_tau_trace"]    = tau_trace;
            out["rhs_tau0_trace"]   = tau0_trace;
            out["rhs_sigma_trace"]  = sigma_trace;
        }
    }

    return out;
}