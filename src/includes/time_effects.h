//
// Created by Azeezat Mustapha on 07.02.26.
//

#ifndef TIME_EFFECTS_H
#define TIME_EFFECTS_H

#include <Eigen/Dense>
#include <vector>

// ============================================================
// Time effect parameters: η_g and centered pseudotime t'_{dc}
// ============================================================
struct TimeEffectParams {
    // Centered pseudotime for each dataset: [D][C_d]
    std::vector<Eigen::VectorXd> pseudotime;

    // Gene-specific time effects η_g (length G)
    Eigen::VectorXd eta;

    // Prior variance σ^2_η
    double sigma_eta_sq;
};

// ============================================================
// Initialization
// ============================================================

// G: number of genes
// pseudotime: vector of length D, each is C_d-length centered pseudotime
TimeEffectParams initialize_time_effects(
    int G,
    const std::vector<Eigen::VectorXd>& pseudotime
);

// ============================================================
// MCMC updates
// ============================================================

// Update all η_g via Metropolis–Hastings
// mu_star: J x G matrix of cluster-specific means μ*_{jg}
// phi_star: J x G matrix of dispersions φ*_{jg}
// Y: length-D vector of G x C_d count matrices
// Z: length-D vector of cluster assignments (size C_d), values 0..J-1
// Beta: length-D vector of C_d-length size factors β_{dc}
// prop_sd: proposal standard deviation for η_g
void update_eta_mcmc(
    TimeEffectParams& params,
    const Eigen::MatrixXd& mu_star,
    const Eigen::MatrixXd& phi_star,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& Z,
    const std::vector<Eigen::VectorXd>& Beta,
    double prop_sd
);

// Gibbs update for σ^2_η
// a_eta, b_eta: Inv-Gamma hyperparameters
void update_sigma_eta(
    TimeEffectParams& params,
    double a_eta,
    double b_eta
);

#endif // TIME_EFFECTS_H