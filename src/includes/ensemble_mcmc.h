#ifndef ENSEMBLE_MCMC_H
#define ENSEMBLE_MCMC_H

#include <Eigen/Dense>
#include <vector>
#include "normHDP_mcmc.h"

// Structure for ensemble averaged results
struct EnsembleResult {
    // Averaged posterior estimates
    Eigen::VectorXd b_mean;
    Eigen::VectorXd b_sd;
    double alpha_phi2_mean;
    double alpha_phi2_sd;
    Eigen::MatrixXd P_J_D_mean;
    Eigen::MatrixXd P_J_D_sd;
    Eigen::VectorXd P_mean;
    Eigen::VectorXd P_sd;
    double alpha_mean;
    double alpha_sd;
    double alpha_zero_mean;
    double alpha_zero_sd;
    Eigen::MatrixXd mu_star_1_J_mean;
    Eigen::MatrixXd mu_star_1_J_sd;
    Eigen::MatrixXd phi_star_1_J_mean;
    Eigen::MatrixXd phi_star_1_J_sd;
    std::vector<Eigen::VectorXd> Beta_mean;
    std::vector<Eigen::VectorXd> Beta_sd;

    // --- Horseshoe Summary (NEW) ---
    Eigen::MatrixXd lambda_mean;
    Eigen::MatrixXd lambda_sd;
    Eigen::VectorXd tau_mean;
    Eigen::VectorXd tau_sd;
    double sigma_mu_mean;
    double sigma_mu_sd;

    // --- Trace Storage (NEW: Required for R traces) ---
    // These will hold the raw samples for every chain
    std::vector<std::vector<Eigen::VectorXd>> b_trace_all;
    std::vector<std::vector<double>> alpha_trace_all;
    std::vector<std::vector<double>> alpha_zero_trace_all;
    std::vector<std::vector<Eigen::MatrixXd>> mu_trace_all;
    std::vector<std::vector<Eigen::MatrixXd>> phi_trace_all;

    // Horseshoe Traces
    std::vector<std::vector<Eigen::MatrixXd>> lambda_trace_all; // Chains x Samples x Matrix
    std::vector<Eigen::MatrixXd> tau_trace_all;                // Chains x (Samples x J matrix)
    std::vector<Eigen::VectorXd> sigma_mu_trace_all;           // Chains x Samples vector

	// --- Spike-and-Slab Summary (NEW) ---
    Eigen::MatrixXd gamma_mean;      // Posterior Inclusion Probabilities (J x G)
    Eigen::MatrixXd gamma_sd;
    double pi_mean;                  // Averaged sparsity parameter
    double pi_sd;
    double sigma_slab_mean;          // Averaged slab variance
    double sigma_slab_sd;

    // --- Spike-and-Slab Traces (NEW) ---
    // Chains x Samples x Matrix/Value
    std::vector<std::vector<Eigen::MatrixXd>> gamma_trace_all;
    std::vector<std::vector<double>> pi_trace_all;
    std::vector<std::vector<double>> sigma_slab_trace_all;

    // --- Clustering & Dimensions ---
    // Consensus cluster assignments (most frequent across chains)
    std::vector<std::vector<int>> Z_consensus;
    // Co-clustering matrix for each dataset (probability two cells in same cluster)
    std::vector<Eigen::MatrixXd> coclustering_matrices;
    int D;
    std::vector<int> C;
    int G;
    int J;
    int num_chains;
    bool use_sparse_prior;
	bool use_spike_slab;
};

// Run ensemble MCMC: multiple chains in parallel, then average
EnsembleResult ensemble_mcmc(
    const std::vector<Eigen::MatrixXd>& Y,
    int J,
    int num_chains = 100,
    int chain_length = 100,
    int thinning = 1,
    bool empirical = true,
    int burn_in = 30,
    bool quadratic = false,
    int iter_update = 10,
    double beta_mean = 0.06,
    double alpha_mu_2 = -1.0,
    double adaptive_prop = 0.1,
    bool print_progress = true,
    int num_cores = -1,  // -1 means use all available
    const Eigen::VectorXd& baynorm_mu_estimate = Eigen::VectorXd(),
    const Eigen::VectorXd& baynorm_phi_estimate = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& baynorm_beta = std::vector<Eigen::VectorXd>()
);

#endif // ENSEMBLE_MCMC_H