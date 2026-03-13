#include "includes/capture_efficiencies.h"
#include "includes/utils.h"
#include <cmath>
#include <algorithm>

Eigen::VectorXd capture_efficiencies_log_prob(
    const Eigen::VectorXd& Beta_d,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& Z,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    int d,
    const Eigen::VectorXd& a_d_beta,
    const Eigen::VectorXd& b_d_beta
) {
    int C_d = Z[d].size();
    int G = Y[d].rows();

    Eigen::VectorXd lprod(C_d);

    for (int c = 0; c < C_d; ++c) {
        int j = Z[d][c];  // Cluster assignment for cell c
        double beta = Beta_d(c);

        // Prior contribution
        double lprod1 = (a_d_beta(d) - 1.0) * std::log(beta) +
                        (b_d_beta(d) - 1.0) * std::log(1.0 - beta);

        // Likelihood contribution
        double lprod2 = 0.0;
        for (int g = 0; g < G; ++g) {
            double y = Y[d](g, c);
            double mu_star = mu_star_1_J(j, g);
            double phi_star = phi_star_1_J(j, g);
            double mu_beta = mu_star * beta;

            lprod2 += (phi_star + y) * std::log(phi_star + mu_beta) - y * std::log(beta);
        }

        lprod(c) = lprod1 - lprod2;
    }

    return lprod;
}

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
    double adaptive_prop
) {
    int D = Z.size();
    int n = iter_num;

    std::vector<Eigen::VectorXd> Beta_new(D);
    std::vector<Eigen::VectorXd> X_new(D);
    std::vector<Eigen::VectorXd> mean_X_new(D);
    std::vector<Eigen::VectorXd> M_2_new(D);
    std::vector<Eigen::VectorXd> variance_new(D);

    int accept_count_tot = 0;

    for (int d = 0; d < D; ++d) {
        int C_d = Z[d].size();

        Eigen::VectorXd Beta_d_old = Beta[d];
        Eigen::VectorXd X_d_old(C_d);

        // Transform to unconstrained space
        for (int c = 0; c < C_d; ++c) {
            X_d_old(c) = std::log(Beta_d_old(c) / (1.0 - Beta_d_old(c)));
        }

        Eigen::VectorXd X_d_new(C_d);

        // Propose new values
        if (n <= 20) {
            for (int c = 0; c < C_d; ++c) {
                X_d_new(c) = X_d_old(c) + rnorm();
            }
        } else {
            for (int c = 0; c < C_d; ++c) {
                X_d_new(c) = X_d_old(c) +
                    std::sqrt(2.4 * 2.4 * (variance[d](c) + adaptive_prop)) * rnorm();
            }
        }

        // Transform back to constrained space
        Eigen::VectorXd Beta_d_new(C_d);
        for (int c = 0; c < C_d; ++c) {
            Beta_d_new(c) = std::exp(X_d_new(c)) / (1.0 + std::exp(X_d_new(c)));
            if (Beta_d_new(c) >= 1.0) {
                Beta_d_new(c) = 0.99;
            }
        }

        // Compute log acceptance probabilities
        Eigen::VectorXd acceptance_prob_log =
            capture_efficiencies_log_prob(Beta_d_new, Y, Z, mu_star_1_J, phi_star_1_J,
                                         d, a_d_beta, b_d_beta) -
            capture_efficiencies_log_prob(Beta_d_old, Y, Z, mu_star_1_J, phi_star_1_J,
                                         d, a_d_beta, b_d_beta);

        // Add Jacobian adjustment
        for (int c = 0; c < C_d; ++c) {
            acceptance_prob_log(c) += std::log(Beta_d_new(c)) + std::log(1.0 - Beta_d_new(c)) -
                                      std::log(Beta_d_old(c)) - std::log(1.0 - Beta_d_old(c));
        }

        // Accept/reject for each cell
        int accept_num = 0;
        for (int c = 0; c < C_d; ++c) {
            if (std::isnan(acceptance_prob_log(c)) ||
                runif() >= std::min(1.0, std::exp(acceptance_prob_log(c)))) {
                X_d_new(c) = X_d_old(c);
                Beta_d_new(c) = Beta_d_old(c);
            } else {
                accept_num++;
            }
        }

        Beta_new[d] = Beta_d_new;
        X_new[d] = X_d_new;

        // Update adaptive statistics
        mean_X_new[d] = mean_X[d] * (1.0 - 1.0 / n) + (1.0 / n) * X_new[d];
        M_2_new[d] = M_2[d] + (X_new[d] - mean_X[d]).cwiseProduct(X_new[d] - mean_X_new[d]);
        variance_new[d] = (1.0 / (n - 1)) * M_2_new[d];

        accept_count_tot += accept_num;
    }

    CaptureEfficienciesResult result;
    result.Beta_new = Beta_new;
    result.accept_count = accept_count_tot;
    result.mean_X_new = mean_X_new;
    result.M_2_new = M_2_new;
    result.variance_new = variance_new;

    return result;
}