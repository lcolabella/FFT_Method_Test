#include "app/PermeabilityRunner.hpp"

#include <algorithm>
#include <future>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef PERMEABILITY_USE_OPENMP
#include <omp.h>
#endif

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalOperator.hpp"
#include "bignonnet/VariationalRHS.hpp"
#include "bignonnet/VelocityRecovery.hpp"
#include "fft/FFTWBackend.hpp"
#include "io/VoxelIO.hpp"
#include "post/PermeabilityTensor.hpp"
#include "solver/MinresSolver.hpp"

namespace permeability {

namespace {

std::vector<std::pair<std::size_t, Real3>> directions_from_mode(GradientDirection mode) {
    if (mode == GradientDirection::X) {
        return {{0, Real3{1.0, 0.0, 0.0}}};
    }
    if (mode == GradientDirection::Y) {
        return {{1, Real3{0.0, 1.0, 0.0}}};
    }
    if (mode == GradientDirection::Z) {
        return {{2, Real3{0.0, 0.0, 1.0}}};
    }
    return {
        {0, Real3{1.0, 0.0, 0.0}},
        {1, Real3{0.0, 1.0, 0.0}},
        {2, Real3{0.0, 0.0, 1.0}},
    };
}

struct DirectionSolveResult {
    std::size_t col{0};
    SolverResult solve_result{};
    Real3 avg_velocity{0.0, 0.0, 0.0};
};

DirectionSolveResult solve_one_direction(const SimulationConfig& config,
                                         const BinaryMedium& medium,
                                         const ForceSupport& support,
                                         std::size_t col,
                                         const Real3& grad) {
#ifdef PERMEABILITY_USE_OPENMP
    if (config.omp_threads > 0) {
        omp_set_num_threads(config.omp_threads);
    }
#endif

    FFTWBackend fftw;
    GreenOperator green(medium.grid(), config.green, config.mu, fftw, config.fft_threads, config.p_radius);
    BackgroundForce f0_builder(medium, support);
    VariationalRHS rhs_builder(support, green, f0_builder);
    VariationalOperator op(support, green);
    MinresSolver minres(config.tolerance, config.max_iterations);
    VelocityRecovery recovery(support, green, f0_builder);

    const std::vector<double> b = rhs_builder.build(grad);
    const SolverResult result = minres.solve(op, b);
    const VectorField3D velocity = recovery.recover(grad, result.solution);

    DirectionSolveResult out;
    out.col = col;
    out.solve_result = result;
    out.avg_velocity = recovery.average_over_domain(velocity);
    return out;
}

}  // namespace

PermeabilityRunner::PermeabilityRunner(SimulationConfig config) : config_(std::move(config)) {}

int PermeabilityRunner::run() {
#ifdef PERMEABILITY_USE_OPENMP
    if (!config_.parallel_load_cases && config_.omp_threads > 0) {
        omp_set_num_threads(config_.omp_threads);
    }
#endif

    BinaryMedium medium = VoxelIO::load_text(config_.input_path, config_.voxel_size);
    ForceSupport support(medium, config_.support_mode);
    if (support.num_active_voxels() == 0) {
        throw std::runtime_error("Support set is empty; cannot run variational solve");
    }

    PermeabilityTensor k_tensor;

    const std::vector<std::pair<std::size_t, Real3>> gradients = directions_from_mode(config_.gradient_mode);

    std::cout << "Grid: " << medium.grid().nx() << " x " << medium.grid().ny() << " x " << medium.grid().nz() << "\n";
    std::cout << "Voxel size: " << config_.voxel_size << "\n";
    std::cout << "Porosity: " << medium.porosity() << "\n";
    std::cout << "Support voxels (NB): " << support.num_active_voxels() << "\n";

    std::vector<DirectionSolveResult> direction_results;
    direction_results.reserve(gradients.size());

    if (config_.parallel_load_cases && gradients.size() > 1) {
        std::vector<std::future<DirectionSolveResult>> futures;
        futures.reserve(gradients.size());

        for (const auto& item : gradients) {
            const std::size_t col = item.first;
            const Real3 grad = item.second;
            futures.push_back(std::async(std::launch::async, [&, col, grad]() {
                return solve_one_direction(config_, medium, support, col, grad);
            }));
        }

        for (auto& fut : futures) {
            direction_results.push_back(fut.get());
        }
    } else {
        for (const auto& item : gradients) {
            const std::size_t col = item.first;
            const Real3 grad = item.second;
            direction_results.push_back(solve_one_direction(config_, medium, support, col, grad));
        }
    }

    for (const auto& item : gradients) {
        const std::size_t col = item.first;
        const auto it = std::find_if(direction_results.begin(), direction_results.end(),
                                     [col](const DirectionSolveResult& r) { return r.col == col; });
        if (it == direction_results.end()) {
            throw std::runtime_error("Internal error: missing direction result");
        }

        if (!it->solve_result.converged) {
            std::cerr << "Warning: solve did not converge for load case " << col
                      << ", final rel residual=" << it->solve_result.final_rel_residual << "\n";
        }

        k_tensor.set_column(col, it->avg_velocity);

        std::cout << "Direction " << col << ": iterations=" << it->solve_result.iterations
                  << ", relres=" << it->solve_result.final_rel_residual << "\n";
    }

    std::cout << "Permeability tensor K:\n" << k_tensor.to_string() << "\n";
    return 0;
}

}  // namespace permeability
