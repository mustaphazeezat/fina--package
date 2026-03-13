#ifndef MEAN_DISPERSION_HORSESHOE_H
#define MEAN_DISPERSION_HORSESHOE_H

#include <RcppEigen.h>
#include <vector>

// tau_j ~ Half-Cauchy(0, tau_0)
// tau_0 ~ Half-Cauchy(0, 1)  ← LEARN THIS FROM DATA

// Updated structure with tau hyperprior
struct HorseshoeParams {
    std::vector<Eigen::VectorXd> lambda;  // J x G local shrinkage
    std::vector<Eigen::VectorXd> nu;      // J x G auxiliary for lambda
    std::vector<double> tau;               // J global shrinkage per cluster
    std::vector<double> xi;                // J auxiliary for tau
    double sigma_mu;                       // Overall variance
    double alpha_phi_2;
};

// Result structure
struct MeanDispersionHorseshoeResult {
    double alpha_phi_2;
    Eigen::VectorXd b;
    HorseshoeParams horseshoe_params;
	double log_likelihood;
};

// Main MCMC update function with horseshoe prior
MeanDispersionHorseshoeResult mean_dispersion_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    HorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
);

// Initialize horseshoe parameters
HorseshoeParams initialize_horseshoe_params(int J, int G);

HorseshoeParams initialize_horseshoe_params_empirical(
    int J,
    int G,
    const Eigen::VectorXd& mu_baseline,
    const Eigen::MatrixXd& mu_initial
);;

#endif // MEAN_DISPERSION_HORSESHOE_H