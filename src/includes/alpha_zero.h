#ifndef ALPHA_ZERO_H
#define ALPHA_ZERO_H

#include <Eigen/Dense>

// Structure for alpha_zero MCMC result (same as AlphaResult)
struct AlphaZeroResult {
    double alpha_zero_new;
    double X_mean_new;
    double M_2_new;
    double variance_new;
    int accept;
};

// Log probability for alpha_zero
double alpha_zero_log_prob(
    const Eigen::VectorXd& P,
    double alpha_zero
);

// MCMC update for alpha_zero
AlphaZeroResult alpha_zero_mcmc(
    const Eigen::VectorXd& P,
    double alpha_zero,
    double X_mean,
    double M_2,
    double variance,
    int iter_num,
    double adaptive_prop = 0.01
);

#endif // ALPHA_ZERO_H