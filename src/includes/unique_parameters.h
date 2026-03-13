#ifndef UNIQUE_PARAMETERS_H
#define UNIQUE_PARAMETERS_H

#include <Eigen/Dense>
#include <vector>
#include "mean_dispersion_horseshoe.h"
#include "mean_dispersion_regularized_horseshoe.h"
#include "mean_dispersion_spike_slab.h"
#include "time_effects.h"

// ============================================================
// Enum for prior type selection
// ============================================================
enum class MuPriorType {
    LOGNORMAL,              // Default: log(μ*) ~ N(0, α²_μ)
    HORSESHOE,              // δ_{jg} ~ N(0, σ²_μ τ²_j λ²_{jg})
    REGULARIZED_HORSESHOE ,  // δ_{jg} ~ N(0, σ²_μ τ²_j λ̃²_{jg}) with slab
	SPIKE_SLAB,
	GENE_HORSESHOE
};

// ============================================================
// Structure for unique parameters MCMC result
// ============================================================
struct UniqueParamsResult {
    Eigen::MatrixXd mu_star_1_J_new;
    Eigen::MatrixXd phi_star_1_J_new;
    int accept_count;
    std::vector<std::vector<Eigen::MatrixXd>> tilde_s_mu_phi_new;
    std::vector<std::vector<Eigen::RowVectorXd>> mean_X_mu_phi_new;
    std::vector<std::vector<Eigen::MatrixXd>> covariance_new;
};

// ============================================================
// Log probability functions for each prior type
// ============================================================

// Original: Lognormal prior
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
    bool quadratic = false
);

// Horseshoe prior
double unique_parameters_log_prob_horseshoe(
    double mu_star,
    double phi_star,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const HorseshoeParams& horseshoe_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic = false
);

// Regularized horseshoe prior
double unique_parameters_log_prob_reg_horseshoe(
    double mu_star,
    double phi_star,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const RegularizedHorseshoeParams& rhs_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic = false
);

// Spike and slab
// Spike and slab
double unique_parameters_log_prob_spike_slab(
    double mu_star,
    double phi_star,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    const SpikeSlabParams& spike_slab_params,
    const Eigen::VectorXd& mu_baseline,
    bool quadratic = false
);

// ============================================================
// Main MCMC function with prior selection
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
    double adaptive_prop = 0.01,
    int num_cores = 1,
    MuPriorType prior_type = MuPriorType::LOGNORMAL,
    const HorseshoeParams* horseshoe_params = nullptr,
    const RegularizedHorseshoeParams* reg_horseshoe_params = nullptr,
	const SpikeSlabParams* spike_slab_params = nullptr,
    const Eigen::VectorXd* mu_baseline = nullptr
);

#endif // UNIQUE_PARAMETERS_H