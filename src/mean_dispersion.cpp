#include "includes/mean_dispersion.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>

MeanDispersionResult mean_dispersion_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    // Collect valid log values
    std::vector<double> x_vals, y_vals;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu  = mu_star_1_J(j, g);
            double phi = phi_star_1_J(j, g);

            if (std::isfinite(std::log(mu)) && std::isfinite(std::log(phi)) &&
                mu > 0 && phi > 0) {
                x_vals.push_back(std::log(mu));
                y_vals.push_back(std::log(phi));
            }
        }
    }

    int n = x_vals.size();

    if (n == 0) {
        MeanDispersionResult result;
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        return result;
    }

    // Build design matrix X and response y
    Eigen::VectorXd y(n);
    Eigen::MatrixXd X;

    for (int i = 0; i < n; ++i) {
        y(i) = y_vals[i];
    }

    if (quadratic) {
        X.resize(n, 3);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
            X(i, 2) = x_vals[i] * x_vals[i];
        }
    } else {
        X.resize(n, 2);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
        }
    }

    // Normal-InvGamma conjugate posterior — matches the R implementation exactly.
    //
    // Prior:  b | alpha_phi_2 ~ N(m_b, I)
    //         alpha_phi_2     ~ InvGamma(v_1, v_2)
    //
    // Sufficient statistics (R names in comments):
    //   parameter0 = I + X'X
    //   parameter1 = m_b + X'y
    //   parameter2 = y'y
    //
    // Posterior:
    //   V_tilde   = parameter0^{-1}
    //   m_tilde   = V_tilde * parameter1
    //   v_tilde_1 = v_1 + J*G / 2
    //   v_tilde_2 = v_2 + 0.5*(y'y - m_tilde' * parameter0 * m_tilde + sum(m_b^2))

    int p = m_b.size();   // 2 (linear) or 3 (quadratic)

    Eigen::MatrixXd XtX = X.transpose() * X;
    Eigen::VectorXd Xty = X.transpose() * y;
    double          yty = y.squaredNorm();

    // parameter0 = I + X'X
    Eigen::MatrixXd parameter0 = Eigen::MatrixXd::Identity(p, p) + XtX;

    // parameter1 = m_b + X'y
    Eigen::VectorXd parameter1 = m_b + Xty;

    // Posterior covariance and mean of b
    Eigen::MatrixXd V_tilde = parameter0.inverse();
    Eigen::VectorXd m_tilde = V_tilde * parameter1;

    // Posterior hyperparameters for alpha_phi_2
    double v_tilde_1 = v_1 + static_cast<double>(J * G) / 2.0;
    double v_tilde_2 = v_2 + 0.5 * (yty
                                     - (m_tilde.transpose() * parameter0 * m_tilde)(0, 0)
                                     + m_b.squaredNorm());

    if (v_tilde_2 <= 0.0) v_tilde_2 = v_2;   // numerical guard

    // Sample alpha_phi_2 ~ InvGamma(v_tilde_1, v_tilde_2)
    std::gamma_distribution<double> gamma_dist(v_tilde_1, 1.0 / v_tilde_2);
    double alpha_phi_2 = 1.0 / gamma_dist(rng_local);

    // Sample b ~ N(m_tilde,  alpha_phi_2 * V_tilde)
    Eigen::MatrixXd cov_b = alpha_phi_2 * V_tilde;
    Eigen::VectorXd b_new = rmvnorm(m_tilde, cov_b);

    MeanDispersionResult result;
    result.alpha_phi_2 = alpha_phi_2;
    result.b           = b_new;

    return result;
}