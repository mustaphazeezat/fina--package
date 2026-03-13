#include "includes/component_probabilities.h"
#include "includes/utils.h"
#include <cmath>
#include <algorithm>

double component_log_prob(
    const Eigen::VectorXd& P,
    const Eigen::MatrixXd& P_J_D,
    double alpha_zero,
    double alpha
) {
    int J = P_J_D.rows();
    int D = P_J_D.cols();
    
    double lprod = 0.0;
    
    // First term: sum((alpha_zero/J - 1) * log(P))
    for (int j = 0; j < J; ++j) {
        lprod += (alpha_zero / J - 1.0) * std::log(P(j));
    }
    
    // Second term: sum over datasets
    for (int d = 0; d < D; ++d) {
        for (int j = 0; j < J; ++j) {
            lprod += alpha * P(j) * std::log(P_J_D(j, d)) - std::lgamma(alpha * P(j));
        }
    }
    
    return lprod;
}

ComponentProbResult component_probabilities_mcmc(
    const Eigen::VectorXd& P,
    const Eigen::MatrixXd& P_J_D,
    double alpha_zero,
    double alpha,
    const Eigen::MatrixXd& covariance,
    const Eigen::RowVectorXd& mean_x,
    const Eigen::MatrixXd& tilde_s,
    int iter_num,
    double adaptive_prop
) {
    int J = P.size();
    int n = iter_num;
    
    Eigen::VectorXd P_old = P;
    Eigen::VectorXd X_d_old(J - 1);
    
    // X_d_old = log(P_old[0:J-2] / P_old[J-1])
    for (int j = 0; j < J - 1; ++j) {
        X_d_old(j) = std::log(P_old(j) / P_old(J - 1));
    }
    
    Eigen::VectorXd X_new;
    
    // Adaptive proposal
    if (n <= 20) {
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(J - 1, J - 1);
        X_new = rmvnorm(X_d_old, cov);
    } else {
        Eigen::MatrixXd cov = (2.4 * 2.4) / (J - 1) * 
            (covariance + adaptive_prop * Eigen::MatrixXd::Identity(J - 1, J - 1));
        X_new = rmvnorm(X_d_old, cov);
    }
    
    // Compute P_new from X_new
    Eigen::VectorXd P_new(J);
    double sum_exp = 0.0;
    for (int j = 0; j < J - 1; ++j) {
        sum_exp += std::exp(X_new(j));
    }
    
    for (int j = 0; j < J - 1; ++j) {
        P_new(j) = std::exp(X_new(j)) / (1.0 + sum_exp);
    }
    P_new(J - 1) = 1.0 / (1.0 + sum_exp);
    
    // Compute log acceptance probability
    double log_acceptance = component_log_prob(P_new, P_J_D, alpha_zero, alpha) -
                           component_log_prob(P_old, P_J_D, alpha_zero, alpha);
    
    for (int j = 0; j < J; ++j) {
        log_acceptance += std::log(P_new(j)) - std::log(P_old(j));
    }
    
    // Accept/reject
    int accept = 0;
    if (!std::isnan(log_acceptance) && runif() < std::min(1.0, std::exp(log_acceptance))) {
        accept = 1;
    } else {
        X_new = X_d_old;
        P_new = P_old;
    }
    
    // Update statistics
    Eigen::MatrixXd tilde_s_new = tilde_s + X_new * X_new.transpose();
    Eigen::RowVectorXd mean_x_new = mean_x * (1.0 - 1.0 / n) + (1.0 / n) * X_new.transpose();
    Eigen::MatrixXd covariance_new = (1.0 / (n - 1)) * tilde_s_new - 
                                     (n / (double)(n - 1)) * mean_x_new.transpose() * mean_x_new;
    
    ComponentProbResult result;
    result.P_new = P_new;
    result.tilde_s_new = tilde_s_new;
    result.mean_x_new = mean_x_new;
    result.covariance_new = covariance_new;
    result.accept = accept;
    
    return result;
}