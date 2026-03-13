#include "includes/alpha_zero.h"
#include "includes/utils.h"
#include <cmath>
#include <algorithm>

double alpha_zero_log_prob(
    const Eigen::VectorXd& P,
    double alpha_zero
) {
    int J = P.size();
    
    double lprob = -alpha_zero + std::lgamma(alpha_zero) - J * std::lgamma(alpha_zero / J);
    
    for (int j = 0; j < J; ++j) {
        lprob += (alpha_zero / J) * std::log(P(j));
    }
    
    return lprob;
}

AlphaZeroResult alpha_zero_mcmc(
    const Eigen::VectorXd& P,
    double alpha_zero,
    double X_mean,
    double M_2,
    double variance,
    int iter_num,
    double adaptive_prop
) {
    int n = iter_num;
    double alpha_zero_old = alpha_zero;
    double X_d_old = std::log(alpha_zero_old);
    
    double X_new;
    
    if (n <= 20) {
        X_new = X_d_old + rnorm();
    } else {
        X_new = X_d_old + std::sqrt(2.4 * 2.4 * (variance + adaptive_prop)) * rnorm();
    }
    
    double alpha_zero_new = std::exp(X_new);
    
    double log_acceptance = alpha_zero_log_prob(P, alpha_zero_new) -
                           alpha_zero_log_prob(P, alpha_zero_old) +
                           std::log(alpha_zero_new) - std::log(alpha_zero_old);
    
    int accept = 0;
    
    if (!std::isnan(log_acceptance) && runif() < std::min(1.0, std::exp(log_acceptance))) {
        accept = 1;
    } else {
        X_new = X_d_old;
        alpha_zero_new = alpha_zero_old;
    }
    
    double X_mean_new = (1.0 - 1.0 / n) * X_mean + (1.0 / n) * X_new;
    double M_2_new = M_2 + (X_new - X_mean) * (X_new - X_mean_new);
    double variance_new = (1.0 / (n - 1)) * M_2_new;
    
    AlphaZeroResult result;
    result.alpha_zero_new = alpha_zero_new;
    result.X_mean_new = X_mean_new;
    result.M_2_new = M_2_new;
    result.variance_new = variance_new;
    result.accept = accept;
    
    return result;
}