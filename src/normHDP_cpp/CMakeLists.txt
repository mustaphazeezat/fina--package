cmake_minimum_required(VERSION 4.0)
project(normHDP_cpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/opt/homebrew/opt/eigen/include/eigen3")

include_directories(/opt/homebrew/opt/eigen/include/eigen3)

# Find threading library
find_package(Threads REQUIRED)

add_executable(normHDP_cpp
        main.cpp
        allocation_z.cpp
        mean_dispersion.cpp
        includes/allocation_z.h
        includes/dataset_specific.h
        dataset_specific.cpp
        includes/component_probabilities.h
        component_probabilities.cpp
        includes/utils.h
        utils.cpp
        includes/alpha.h
        alpha.cpp
        includes/alpha_zero.h
        alpha_zero.cpp
        unique_parameters.cpp
        includes/unique_parameters.h
        includes/capture_efficiencies.h
        capture_efficiencies.cpp
        includes/normHDP_mcmc.h
        normHDP_mcmc.cpp
        includes/ensemble_mcmc.h
        ensemble_mcmc.cpp
)

# Link threading library
target_link_libraries(normHDP_cpp Threads::Threads)

# Enable warnings
target_compile_options(normHDP_cpp PRIVATE -Wall -Wextra)