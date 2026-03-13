#ifndef NORMHDP_MCMC_H
#define NORMHDP_MCMC_H

#include <Eigen/Dense>
#include <vector>
#include "mean_dispersion_horseshoe.h"
#include "mean_dispersion_spike_slab.h"
#include "mean_dispersion_regularized_horseshoe.h"

// Structure for acceptance probabilities
struct AcceptanceRates {
    std::vector<double> P_accept;
    std::vector<double> alpha_accept;
    std::vector<double> alpha_zero_accept;
    std::vector<double> unique_accept;
    std::vector<double> Beta_accept;
};

// Structure for MCMC output
struct NormHDPResult {
    // Main outputs
    std::vector<Eigen::VectorXd> b_output;
    std::vector<double> alpha_phi2_output;
    std::vector<std::vector<std::vector<int>>> Z_output;
    std::vector<Eigen::MatrixXd> P_J_D_output;
    std::vector<Eigen::VectorXd> P_output;
    std::vector<double> alpha_output;
    std::vector<double> alpha_zero_output;
    std::vector<Eigen::MatrixXd> mu_star_1_J_output;
    std::vector<Eigen::MatrixXd> phi_star_1_J_output;
    std::vector<std::vector<Eigen::VectorXd>> Beta_output;

	 std::vector<SpikeSlabParams> spike_slab_output;
     std::vector<Eigen::VectorXd> mu_baseline_output;

    //Horseshoe parameters output (only saved if use_sparse_prior = true)
    std::vector<HorseshoeParams> horseshoe_output;
	std::vector<RegularizedHorseshoeParams> reg_horseshoe_output;
    std::vector<Eigen::VectorXd> eta_output;
    std::vector<double> sigma_eta_sq_output;

    // Diagnostics
    AcceptanceRates acceptance_rates;

    // Dimensions
    int D;
    std::vector<int> C;
    int G;
    bool use_sparse_prior;
	bool use_spike_slab;
	bool use_reg_horseshoe;
};

// Main MCMC function
NormHDPResult normHDP_mcmc(
    const std::vector<Eigen::MatrixXd>& Y,
    int J,
    int number_iter,
    int thinning = 5,
    bool empirical = true,
    int burn_in = 3000,
    bool quadratic = false,
    int iter_update = 100,
    double beta_mean = 0.06,
    double alpha_mu_2 = -1.0,  // -1 means auto-compute
    double adaptive_prop = 0.1,
    bool print_Z = false,
    int num_cores = 4,
    bool save_only_z = false,
    bool use_sparse_prior = true,
	bool use_spike_slab = false,
	bool use_reg_horseshoe = false,
    double p_0 = 100,
    const Eigen::VectorXd& baynorm_mu_estimate = Eigen::VectorXd(),
    const Eigen::VectorXd& baynorm_phi_estimate = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& baynorm_beta = std::vector<Eigen::VectorXd>()
);

#endif // NORMHDP_MCMC_H