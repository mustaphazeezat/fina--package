//
// Created by Azeezat Mustapha on 22.02.26.
//

#ifndef NORMHDPCPP_PACKAGE_MEAN_DISPERSION_GENE_HORSESHOE_H
#define NORMHDPCPP_PACKAGE_MEAN_DISPERSION_GENE_HORSESHOE_H

#ifndef MEAN_DISPERSION_GENE_HORSESHOE_H
#define MEAN_DISPERSION_GENE_HORSESHOE_H

#include <Eigen/Dense>
#include <vector>

//==============================================================================
// GENE-LEVEL HORSESHOE: One ω_g per gene (shared across all clusters)
//==============================================================================

/*
Mathematical Model:
───────────────────

For clustering with FIXED Z (Stage 2):

Cluster means:
  log(μ_{jg}) = log(μ_baseline_g) + δ_{jg}

  δ_{jg} | σ²_μ, ω_g ~ N(0, σ²_μ ω²_g)  ← SAME ω_g for ALL clusters j

Gene-level horseshoe:
  ω_g = τ_0 λ_g
  λ²_g ~ InvGamma(1/2, 1/ξ_g)
  ξ_g ~ InvGamma(1/2, 1)
  τ²_0 ~ InvGamma(1/2, 1/ξ_0)
  ξ_0 ~ InvGamma(1/2, 1)
  σ²_μ ~ InvGamma(a_σ, b_σ)

Interpretation:
  Large ω_g: Gene g varies ACROSS clusters → informative for clustering
  Small ω_g: Gene g constant ACROSS clusters → noise, shrunk to baseline
*/

struct GeneLevelHorseshoeParams {
    std::vector<double> lambda_sq;  // λ²_g for each gene [G]
    std::vector<double> xi;         // ξ_g for each gene [G]
    double tau_0_sq;                // τ²_0 global shrinkage
    double xi_0;                    // ξ_0 auxiliary for τ_0
    double sigma_mu_sq;             // σ²_μ overall variance scale
    std::vector<double> omega;      // ω_g = τ_0 × λ_g (output) [G]
};

struct GeneLevelHorseshoeResult {
    GeneLevelHorseshoeParams horseshoe_params;
    double alpha_phi_2;    // Dispersion variance
    Eigen::VectorXd b;     // Dispersion regression coefficients
    double log_likelihood; // For model comparison
};

// Initialize gene-level horseshoe parameters
GeneLevelHorseshoeParams initialize_gene_level_horseshoe_params(
    int G,
    const Eigen::MatrixXd& mu_initial,
    const Eigen::VectorXd& mu_baseline
);

// MCMC update for gene-level horseshoe (call at each iteration)
GeneLevelHorseshoeResult mean_dispersion_gene_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    GeneLevelHorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
);

// Get prior variance for each gene (for use when sampling μ_{jg})
Eigen::VectorXd get_gene_prior_variances(const GeneLevelHorseshoeParams& params);

#endif // MEAN_DISPERSION_GENE_HORSESHOE_H

#endif //NORMHDPCPP_PACKAGE_MEAN_DISPERSION_GENE_HORSESHOE_H