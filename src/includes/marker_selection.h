//
// Created by Azeezat Mustapha on 12.02.26.
//

#ifndef NORMHDPCPP_PACKAGE_MARKER_SELECTION_H
#define NORMHDPCPP_PACKAGE_MARKER_SELECTION_H

#pragma once
#include <Eigen/Dense>
#include <vector>

struct MarkerParams {
    // Sparse effects (deviations from cluster grand mean)
    std::vector<Eigen::VectorXd> lambda;  // [J][G] local shrinkage
    std::vector<Eigen::VectorXd> nu;      // [J][G] local auxiliary
    std::vector<double> tau;              // [J] global shrinkage per cluster
    std::vector<double> xi;               // [J] global auxiliary

    double sigma_beta;  // Variance of marker effects

    // Derived quantities (for reporting)
    Eigen::MatrixXd beta;           // [J x G] log fold changes
    Eigen::MatrixXd marker_prob;    // [J x G] posterior prob of being marker
    Eigen::VectorXd gene_specificity; // [G] how cluster-specific is each gene
};

struct MarkerSelectionResult {
    MarkerParams marker_params;
    Eigen::MatrixXd posterior_inclusion_prob;  // P(gene g is marker for cluster j)
    std::vector<int> top_markers_per_cluster;   // Gene indices
};

// Initialize marker parameters
MarkerParams initialize_marker_params(int J, int G);

// Update marker parameters (AFTER updating cluster means)
void update_marker_selection(
    MarkerParams& marker_params,
    const Eigen::MatrixXd& mu_star_1_J,  // Current cluster means
    const Eigen::VectorXd& mu_grand       // Grand mean per gene
);

// Compute posterior probability that gene g is a marker for cluster j
Eigen::MatrixXd compute_marker_probabilities(const MarkerParams& marker_params);

#endif //NORMHDPCPP_PACKAGE_MARKER_SELECTION_H