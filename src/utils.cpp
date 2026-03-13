#include "includes/utils.h"
#include <cmath>

// Thread-local random number generator
thread_local std::mt19937 rng_local(std::random_device{}());

void init_rng(unsigned int seed) {
    rng_local.seed(seed);
}

double rnorm(double mean, double sd) {
    std::normal_distribution<double> dist(mean, sd);
    return dist(rng_local);
}

double runif() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_local);
}

double rlnorm(double meanlog, double sdlog) {
    std::lognormal_distribution<double> dist(meanlog, sdlog);
    return dist(rng_local);
}

Eigen::VectorXd rmvnorm(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov) {
    int n = mean.size();
    
    // Cholesky decomposition
    Eigen::LLT<Eigen::MatrixXd> llt(cov);
    Eigen::MatrixXd L = llt.matrixL();
    
    // Generate standard normal samples
    Eigen::VectorXd z(n);
    for (int i = 0; i < n; ++i) {
        z(i) = rnorm();
    }
    
    // Transform to multivariate normal
    return mean + L * z;
}

Eigen::VectorXd rdirichlet(const Eigen::VectorXd& alpha) {
    int K = alpha.size();
    Eigen::VectorXd samples(K);
    
    // Sample from Gamma distributions
    for (int k = 0; k < K; ++k) {
        std::gamma_distribution<double> gamma(alpha(k), 1.0);
        samples(k) = gamma(rng_local);
    }
    
    // Normalize
    samples /= samples.sum();
    
    return samples;
}