//
// Created by Azeezat Mustapha on 07.02.26.
//


#include "includes/time_effects.h"
#include "includes/utils.h"   // for rng_local
#include <cmath>
#include <random>
#include <iostream>

// ============================================================
// Initialization
// ============================================================
TimeEffectParams initialize_time_effects(
    int G,
    const std::vector<Eigen::VectorXd>& pseudotime
) {
    TimeEffectParams params;
    params.pseudotime = pseudotime;             // already centered in R
    params.eta = Eigen::VectorXd::Zero(G);      // start with no time effect
    params.sigma_eta_sq = 1.0;                  // initial prior variance

    std::cout << "TimeEffectParams initialized: G = " << G
              << ", D = " << pseudotime.size() << "\n";

    return params;
}

// ============================================================
// Negative Binomial log-likelihood
// p(y | mu, phi) with mean mu, size phi
// ============================================================
static inline double nb_loglik(double y, double mu, double phi) {
    if (mu <= 0.0 || phi <= 0.0 || !std::isfinite(mu) || !std::isfinite(phi)) {
        return -1e300;  // heavy penalty for invalid values
    }

    double log_term1 = std::lgamma(y + phi) - std::lgamma(phi) - std::lgamma(y + 1.0);
    double log_term2 = phi * std::log(phi / (phi + mu));
    double log_term3 = y   * std::log(mu  / (phi + mu));

    double ll = log_term1 + log_term2 + log_term3;
    if (!std::isfinite(ll)) return -1e300;
    return ll;
}

// ============================================================
// Log posterior for a single η_g
// ============================================================

static double log_p_eta_for_gene(
    double eta_g,
    double sigma_eta_sq,
    int g,
    const Eigen::MatrixXd& mu_star,                   // J x G
    const Eigen::MatrixXd& phi_star,                  // J x G
    const std::vector<Eigen::MatrixXd>& Y,            // [D] G x C_d
    const std::vector<std::vector<int>>& Z,           // [D][C_d] cluster indices 0..J-1
    const std::vector<Eigen::VectorXd>& Beta,         // [D][C_d]
    const std::vector<Eigen::VectorXd>& pseudotime    // [D][C_d], centered
) {
    // Prior: N(0, sigma_eta_sq)
    double log_prior = -0.5 * eta_g * eta_g / sigma_eta_sq;

    double log_lik = 0.0;
    int D = Y.size();

    for (int d = 0; d < D; ++d) {
        int C_d = Y[d].cols();
        for (int c = 0; c < C_d; ++c) {
            int j = Z[d][c];               // cluster index
            double t = pseudotime[d](c);   // centered pseudotime
            double beta = Beta[d](c);

            double mu_eff = beta * mu_star(j, g) * std::exp(eta_g * t);
            double phi    = phi_star(j, g);
            double y      = Y[d](g, c);

            log_lik += nb_loglik(y, mu_eff, phi);
        }
    }

    return log_prior + log_lik;
}

// ============================================================
// Metropolis–Hastings update for all η_g
// ============================================================
void update_eta_mcmc(
    TimeEffectParams& params,
    const Eigen::MatrixXd& mu_star,
    const Eigen::MatrixXd& phi_star,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& Z,
    const std::vector<Eigen::VectorXd>& Beta,
    double prop_sd
) {
    int G = params.eta.size();
    int D = Y.size();

    if (D == 0 || G == 0) return;

    std::normal_distribution<double> proposal(0.0, prop_sd);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    for (int g = 0; g < G; ++g) {
        double eta_old = params.eta(g);
        double eta_new = eta_old + proposal(rng_local);

        double log_post_old = log_p_eta_for_gene(
            eta_old,
            params.sigma_eta_sq,
            g,
            mu_star,
            phi_star,
            Y,
            Z,
            Beta,
            params.pseudotime
        );

        double log_post_new = log_p_eta_for_gene(
            eta_new,
            params.sigma_eta_sq,
            g,
            mu_star,
            phi_star,
            Y,
            Z,
            Beta,
            params.pseudotime
        );

        double log_accept = log_post_new - log_post_old;

        if (std::log(unif(rng_local)) < log_accept) {
            params.eta(g) = eta_new;  // accept
        }
        // else: reject, keep old
    }
}

// ============================================================
// Gibbs update for σ^2_η
// σ^2_η ~ Inv-Gamma(a_eta, b_eta)
// ============================================================
void update_sigma_eta(
    TimeEffectParams& params,
    double a_eta,
    double b_eta
) {
    double ss = params.eta.squaredNorm();          // sum of η_g^2
    double shape_post = a_eta + params.eta.size() / 2.0;
    double rate_post  = b_eta + ss / 2.0;

    std::gamma_distribution<double> gamma_dist(shape_post, 1.0 / rate_post);
    double inv_var = gamma_dist(rng_local);
    params.sigma_eta_sq = 1.0 / inv_var;
}