#include "includes/allocation_z.h"
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <numeric>

// Negative binomial log probability density
inline double dnbinom_log(double x, double mu, double size) {
    if (mu <= 0 || size <= 0) return -std::numeric_limits<double>::infinity();

    double p = size / (size + mu);
    return std::lgamma(x + size) - std::lgamma(size) - std::lgamma(x + 1.0) +
           size * std::log(p) + x * std::log(1.0 - p);
}

// Categorical sampling from probability vector
int sample_categorical(const Eigen::VectorXd& probs, std::mt19937& rng) {
    std::discrete_distribution<int> dist(probs.data(), probs.data() + probs.size());
    return dist(rng);
}

// Process a chunk of cells for one dataset
std::vector<int> process_chunk(
    const Eigen::MatrixXd& Y_chunk,        // G x C_chunk
    const Eigen::MatrixXd& mu_star_1_J,    // J x G
    const Eigen::MatrixXd& phi_star_1_J,   // J x G
    const Eigen::VectorXd& Beta_chunk,     // C_chunk
    const Eigen::VectorXd& P_J,            // J (prior for this dataset)
    unsigned int seed
) {
    const int G = Y_chunk.rows();
    const int C_chunk = Y_chunk.cols();
    const int J = mu_star_1_J.rows();

    std::mt19937 rng(seed);
    std::vector<int> Z_chunk(C_chunk);

    // Pre-compute log priors
    Eigen::VectorXd log_prior = P_J.array().log();

    // For each cell in this chunk
    for (int c = 0; c < C_chunk; ++c) {
        Eigen::VectorXd log_prob(J);

        // For each cluster j
        for (int j = 0; j < J; ++j) {
            double lp = 0.0;

            // For each gene g
            for (int g = 0; g < G; ++g) {
                double y = Y_chunk(g, c);
                double mu = mu_star_1_J(j, g) * Beta_chunk(c);
                double phi = phi_star_1_J(j, g);

                lp += dnbinom_log(y, mu, phi);
            }

            log_prob(j) = lp + log_prior(j);
        }

        // Numerical stability: subtract max
        double max_log_prob = log_prob.maxCoeff();
        Eigen::VectorXd prob = (log_prob.array() - max_log_prob).exp();

        // Normalize
        prob /= prob.sum();

        // Sample
        Z_chunk[c] = sample_categorical(prob, rng);
    }

    return Z_chunk;
}

AllocationResult allocation_variables_mcmc(
    const Eigen::MatrixXd& P_J_D,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int iter_num,
    int num_cores
) {
    const int D = Y.size();
    const int J = P_J_D.rows();

    AllocationResult result;
    result.Z.resize(D);

    // Process each dataset
    for (int d = 0; d < D; ++d) {
        const int C_d = Y[d].cols();
        const int G = Y[d].rows();

        // Determine chunk sizes for parallel processing
        int chunk_size = (C_d + num_cores - 1) / num_cores;
        int actual_chunks = (C_d + chunk_size - 1) / chunk_size;

        std::vector<std::thread> threads;
        std::vector<std::vector<int>> chunk_results(actual_chunks);

        // Create random seeds for each thread
        std::random_device rd;
        std::vector<unsigned int> seeds(actual_chunks);
        for (int i = 0; i < actual_chunks; ++i) {
            seeds[i] = rd() + i;
        }

        // Launch threads
        for (int i = 0; i < actual_chunks; ++i) {
            int start_col = i * chunk_size;
            int end_col = std::min(start_col + chunk_size, C_d);
            int this_chunk_size = end_col - start_col;

            // Extract chunk
            Eigen::MatrixXd Y_chunk = Y[d].middleCols(start_col, this_chunk_size);
            Eigen::VectorXd Beta_chunk = Beta[d].segment(start_col, this_chunk_size);
            Eigen::VectorXd P_J = P_J_D.col(d);

            threads.emplace_back([&chunk_results, i, Y_chunk,
                                 mu_star_1_J, phi_star_1_J,
                                 Beta_chunk, P_J, seeds]() {
                chunk_results[i] = process_chunk(
                    Y_chunk, mu_star_1_J, phi_star_1_J,
                    Beta_chunk, P_J, seeds[i]
                );
            });
        }

        // Wait for all threads
        for (auto& t : threads) {
            t.join();
        }

        // Combine results
        result.Z[d].reserve(C_d);
        for (const auto& chunk : chunk_results) {
            result.Z[d].insert(result.Z[d].end(), chunk.begin(), chunk.end());
        }
    }

    return result;
}