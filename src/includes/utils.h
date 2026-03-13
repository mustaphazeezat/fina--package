#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <random>

// Thread-local random number generator
extern thread_local std::mt19937 rng_local;

// Initialize RNG with seed
void init_rng(unsigned int seed = std::random_device{}());

// Sample from standard normal distribution
double rnorm(double mean = 0.0, double sd = 1.0);

// Sample from uniform [0, 1] distribution
double runif();

// Sample from log-normal distribution
double rlnorm(double meanlog, double sdlog);

// Sample from multivariate normal distribution
Eigen::VectorXd rmvnorm(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov);

// Sample from Dirichlet distribution
Eigen::VectorXd rdirichlet(const Eigen::VectorXd& alpha);

#endif // UTILS_H