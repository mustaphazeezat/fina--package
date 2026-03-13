#ifndef ALPHA_H
#define ALPHA_H

#include <Eigen/Dense>

// Structure for alpha MCMC result
struct AlphaResult {
    double alpha_new;
    double X_mean_new;
    double M_2_new;
    double variance_new;
    int accept;
};

// Log probability for alpha
double alpha_log_prob(
    const Eigen::MatrixXd& P_J_D,
    const Eigen::VectorXd& P,
    double alpha
);

// MCMC update for alpha
AlphaResult alpha_mcmc(
    const Eigen::MatrixXd& P_J_D,
    const Eigen::VectorXd& P,
    double alpha,
    double X_mean,
    double M_2,
    double variance,
    int iter_num,
    double adaptive_prop = 0.01
);

#endif // ALPHA_H