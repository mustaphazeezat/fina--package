#ifndef COMPONENT_PROBABILITIES_H
#define COMPONENT_PROBABILITIES_H

#include <Eigen/Dense>

// Structure for component probabilities MCMC result
struct ComponentProbResult {
    Eigen::VectorXd P_new;
    Eigen::MatrixXd tilde_s_new;
    Eigen::RowVectorXd mean_x_new;
    Eigen::MatrixXd covariance_new;
    int accept;
};

// Log probability for component probabilities
double component_log_prob(
    const Eigen::VectorXd& P,
    const Eigen::MatrixXd& P_J_D,
    double alpha_zero,
    double alpha
);

// MCMC update for component probabilities
ComponentProbResult component_probabilities_mcmc(
    const Eigen::VectorXd& P,
    const Eigen::MatrixXd& P_J_D,
    double alpha_zero,
    double alpha,
    const Eigen::MatrixXd& covariance,
    const Eigen::RowVectorXd& mean_x,
    const Eigen::MatrixXd& tilde_s,
    int iter_num,
    double adaptive_prop = 0.01
);

#endif // COMPONENT_PROBABILITIES_H