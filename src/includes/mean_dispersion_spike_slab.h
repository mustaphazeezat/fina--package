//
// Created by Azeezat Mustapha on 25.01.26.
//

#ifndef NORMHDPCPP_PACKAGE_MEAN_DISPERSION_SPIKE_SLAB_H
#define NORMHDPCPP_PACKAGE_MEAN_DISPERSION_SPIKE_SLAB_H

#include <Eigen/Dense>
#include <vector>

// Spike-and-Slab Parameters
struct SpikeSlabParams {
    std::vector<Eigen::MatrixXd> gamma;  // Binary indicators (J vectors of length G)
    std::vector<Eigen::VectorXd> theta;  // Effect sizes (J vectors of length G)
    double pi;                           // Sparsity level (prob of being in slab)
    double sigma_slab;                   // Slab variance
    double sigma_spike;                  // Spike variance (very small)
};

// Result structure
struct MeanDispersionSpikeSlabResult {
    double alpha_phi_2;
    Eigen::VectorXd b;
    SpikeSlabParams spike_slab_params;
    double log_likelihood;
};

// Function declarations
SpikeSlabParams initialize_spike_slab_params(int J, int G);

MeanDispersionSpikeSlabResult mean_dispersion_spike_slab_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    SpikeSlabParams& spike_slab_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
);

#endif//NORMHDPCPP_PACKAGE_MEAN_DISPERSION_SPIKE_SLAB_H