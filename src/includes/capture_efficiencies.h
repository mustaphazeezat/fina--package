#ifndef CAPTURE_EFFICIENCIES_H
#define CAPTURE_EFFICIENCIES_H

#include <Eigen/Dense>
#include <vector>

// Structure for capture efficiencies result
struct CaptureEfficienciesResult {
    std::vector<Eigen::VectorXd> Beta_new;
    int accept_count;
    std::vector<Eigen::VectorXd> mean_X_new;
    std::vector<Eigen::VectorXd> M_2_new;
    std::vector<Eigen::VectorXd> variance_new;
};

// Log probability for capture efficiencies (returns vector for all cells)
Eigen::VectorXd capture_efficiencies_log_prob(
    const Eigen::VectorXd& Beta_d,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& Z,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    int d,
    const Eigen::VectorXd& a_d_beta,
    const Eigen::VectorXd& b_d_beta
);

// MCMC update for capture efficiencies (Beta parameters)
CaptureEfficienciesResult capture_efficiencies_mcmc(
    const std::vector<Eigen::VectorXd>& Beta,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& Z,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& a_d_beta,
    const Eigen::VectorXd& b_d_beta,
    int iter_num,
    const std::vector<Eigen::VectorXd>& M_2,
    const std::vector<Eigen::VectorXd>& mean_X,
    const std::vector<Eigen::VectorXd>& variance,
    double adaptive_prop = 0.01
);

#endif // CAPTURE_EFFICIENCIES_H