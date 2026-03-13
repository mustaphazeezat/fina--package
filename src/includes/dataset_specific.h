#ifndef DATASET_SPECIFIC_H
#define DATASET_SPECIFIC_H

#include <Eigen/Dense>
#include <vector>

// Dataset-specific MCMC update for component probabilities
// Updates P_J_D (J x D matrix) based on cluster assignments Z
Eigen::MatrixXd dataset_specific_mcmc(
    const std::vector<std::vector<int>>& Z,  // Z[d][c] = cluster assignment for cell c in dataset d
    const Eigen::VectorXd& P,                 // J-dimensional global probability vector
    double alpha                              // Concentration parameter
);

#endif // DATASET_SPECIFIC_H