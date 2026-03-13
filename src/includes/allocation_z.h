#ifndef ALLOCATION_Z_H
#define ALLOCATION_Z_H

#include <Eigen/Dense>
#include <vector>
#include <random>

// Structure to hold the result
struct AllocationResult {
    std::vector<std::vector<int>> Z;  // Z[d][c] = cluster assignment
};

// Main allocation function
AllocationResult allocation_variables_mcmc(
    const Eigen::MatrixXd& P_J_D,           // J x D matrix of prior probabilities
    const Eigen::MatrixXd& mu_star_1_J,     // J x G matrix of means
    const Eigen::MatrixXd& phi_star_1_J,    // J x G matrix of dispersions
    const std::vector<Eigen::MatrixXd>& Y,  // D datasets, each G x C[d]
    const std::vector<Eigen::VectorXd>& Beta, // D vectors of size factors
    int iter_num,
    int num_cores = 1
);

#endif // ALLOCATION_Z_H