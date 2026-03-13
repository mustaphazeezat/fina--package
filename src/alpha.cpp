#include "includes/alpha.h"
#include "includes/utils.h"
#include <cmath>
#include <algorithm>

double alpha_log_prob(
    const Eigen::MatrixXd& P_J_D,
    const Eigen::VectorXd& P,
    double alpha
) {
    int D = P_J_D.cols();
    int J = P.size();

    double lprod = -alpha + D * std::lgamma(alpha);

    for (int d = 0; d < D; ++d) {
        for (int j = 0; j < J; ++j) {
            lprod += alpha * P(j) * std::log(P_J_D(j, d)) - std::lgamma(alpha * P(j));
        }
    }

    return lprod;
}

AlphaResult alpha_mcmc(
    const Eigen::MatrixXd& P_J_D,
    const Eigen::VectorXd& P,
    double alpha,
    double X_mean,
    double M_2,
    double variance,
    int iter_num,
    double adaptive_prop
) {
    int n = iter_num;
    double alpha_old = alpha;
    double X_d_old = std::log(alpha_old);

    double X_new;

    if (n <= 20) {
        X_new = X_d_old + rnorm();
    } else {
        X_new = X_d_old + std::sqrt(2.4 * 2.4 * (variance + adaptive_prop)) * rnorm();
    }

    double alpha_new = std::exp(X_new);

    double log_acceptance = alpha_log_prob(P_J_D, P, alpha_new) -
                           alpha_log_prob(P_J_D, P, alpha_old) +
                           std::log(alpha_new) - std::log(alpha_old);

    int accept = 0;

    if (!std::isnan(log_acceptance) && runif() < std::min(1.0, std::exp(log_acceptance))) {
        accept = 1;
    } else {
        X_new = X_d_old;
        alpha_new = alpha_old;
    }

    double X_mean_new = (1.0 - 1.0 / n) * X_mean + (1.0 / n) * X_new;
    double M_2_new = M_2 + (X_new - X_mean) * (X_new - X_mean_new);
    double variance_new = (1.0 / (n - 1)) * M_2_new;

    AlphaResult result;
    result.alpha_new = alpha_new;
    result.X_mean_new = X_mean_new;
    result.M_2_new = M_2_new;
    result.variance_new = variance_new;
    result.accept = accept;

    return result;
}