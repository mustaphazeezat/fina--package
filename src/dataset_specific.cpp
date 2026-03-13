#include "includes/dataset_specific.h"
#include <random>
#include <algorithm>
#include <cmath>

// Sample from Dirichlet distribution
Eigen::VectorXd sample_dirichlet(const Eigen::VectorXd& alpha, std::mt19937& rng) {
    const int K = alpha.size();
    Eigen::VectorXd samples(K);
    
    // Sample from Gamma distributions
    for (int k = 0; k < K; ++k) {
        std::gamma_distribution<double> gamma(alpha(k), 1.0);
        samples(k) = gamma(rng);
    }
    
    // Normalize to get Dirichlet sample
    samples /= samples.sum();
    
    return samples;
}

Eigen::MatrixXd dataset_specific_mcmc(
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& P,
    double alpha
) {
    const int J = P.size();
    const int D = Z.size();
    
    // Initialize result matrix
    Eigen::MatrixXd P_J_D(J, D);
    
    // Random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Process each dataset
    for (int d = 0; d < D; ++d) {
        // Count cluster assignments for dataset d
        Eigen::VectorXd counts = Eigen::VectorXd::Zero(J);
        
        for (int c = 0; c < Z[d].size(); ++c) {
            int cluster = Z[d][c];
            if (cluster >= 0 && cluster < J) {
                counts(cluster) += 1.0;
            }
        }
        
        // Compute Dirichlet parameters: counts + alpha * P
        Eigen::VectorXd parameters = counts + alpha * P;
        
        // Sample from Dirichlet distribution
        Eigen::VectorXd P_J_d = sample_dirichlet(parameters, rng);
        
        // Handle zero probabilities (set to small value)
        const double min_prob = 0.001;
        for (int j = 0; j < J; ++j) {
            if (P_J_d(j) < min_prob) {
                P_J_d(j) = min_prob;
            }
        }
        
        // Renormalize
        P_J_d /= P_J_d.sum();
        
        // Store in result matrix
        P_J_D.col(d) = P_J_d;
    }
    
    return P_J_D;
}