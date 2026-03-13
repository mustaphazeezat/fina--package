//
// Created by Azeezat Mustapha on 25.01.26.
//

#ifndef MEAN_DISPERSION_REGULARIZED_HORSESHOE_H
#define MEAN_DISPERSION_REGULARIZED_HORSESHOE_H

#include <Eigen/Dense>
#include <vector>

// Regularized Horseshoe Parameters (Piironen & Vehtari 2017)
struct RegularizedHorseshoeParams {
    std::vector<Eigen::VectorXd> lambda;  // Local shrinkage (J vectors of length G)
    std::vector<Eigen::VectorXd> nu;      // Auxiliary for lambda
    std::vector<double> tau;              // Global shrinkage per cluster (J values)
    std::vector<double> xi;               // Auxiliary for tau
    double tau_0;                         // Overall global scale
    double xi_0;                          // Auxiliary for tau_0 (FIXED: was xi_tau_0)
    double sigma_mu;                      // Overall variance scale
    double c_squared;                     // Slab variance (regularization parameter)
    double alpha_phi_2;
};

// ========================================
// Result structure
// ========================================
struct MeanDispersionRegularizedHorseshoeResult {
    double alpha_phi_2;
    Eigen::VectorXd b;
    RegularizedHorseshoeParams rhs_params;
    double log_likelihood;
};

// ========================================
// Function declarations
// ========================================

// Standard initialization (default hyperparameters)
RegularizedHorseshoeParams initialize_regularized_horseshoe_params(
    int J,
    int G,
    double p_0
);

// Empirical initialization (data-driven hyperparameters)
RegularizedHorseshoeParams initialize_regularized_horseshoe_params_empirical(
    int J,
    int G,
    double p_0,
    const Eigen::VectorXd& mu_baseline,
    const Eigen::MatrixXd& mu_initial
);

// MCMC update function
MeanDispersionRegularizedHorseshoeResult mean_dispersion_regularized_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    RegularizedHorseshoeParams& rhs_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
);

#endif // MEAN_DISPERSION_REGULARIZED_HORSESHOE_H