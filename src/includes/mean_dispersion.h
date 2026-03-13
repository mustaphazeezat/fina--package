#ifndef MEAN_DISPERSION_H
#define MEAN_DISPERSION_H

#include <Eigen/Dense>

// Structure for mean dispersion result
struct MeanDispersionResult {
    double alpha_phi_2;
    Eigen::VectorXd b;
};

// MCMC update for regression parameters (b and alpha_phi_2)
MeanDispersionResult mean_dispersion_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic = false
);

#endif // MEAN_DISPERSION_H