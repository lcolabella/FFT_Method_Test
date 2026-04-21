#include "app/PermeabilityRunner.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

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

}  // namespace

PermeabilityRunner::PermeabilityRunner(SimulationConfig config) : config_(std::move(config)) {}

int PermeabilityRunner::run() {
    BinaryMedium medium = VoxelIO::load_text(config_.input_path);
    ForceSupport support(medium, config_.support_mode);
    if (support.num_active_voxels() == 0) {
        throw std::runtime_error("Support set is empty; cannot run variational solve");
    }

    FFTWBackend fftw;
    GreenOperator green(medium.grid(), config_.green, config_.mu, fftw, config_.fft_threads, config_.p_radius);

    BackgroundForce f0_builder(medium, support);
    VariationalRHS rhs_builder(support, green, f0_builder);
    VariationalOperator op(support, green);
    MinresSolver minres(config_.tolerance, config_.max_iterations);
    VelocityRecovery recovery(support, green, f0_builder);
    PermeabilityTensor k_tensor;

    const std::vector<std::pair<std::size_t, Real3>> gradients = directions_from_mode(config_.gradient_mode);

    std::cout << "Grid: " << medium.grid().nx() << " x " << medium.grid().ny() << " x " << medium.grid().nz() << "\n";
    std::cout << "Porosity: " << medium.porosity() << "\n";
    std::cout << "Support voxels (NB): " << support.num_active_voxels() << "\n";

    for (const auto& item : gradients) {
        const std::size_t col = item.first;
        const Real3 grad = item.second;

        const std::vector<double> b = rhs_builder.build(grad);
        const SolverResult result = minres.solve(op, b);

        if (!result.converged) {
            std::cerr << "Warning: solve did not converge for load case " << col
                      << ", final rel residual=" << result.final_rel_residual << "\n";
        }

        const VectorField3D velocity = recovery.recover(grad, result.solution);
        const Real3 avg = recovery.average_over_domain(velocity);
        k_tensor.set_column(col, avg);

        std::cout << "Direction " << col << ": iterations=" << result.iterations
                  << ", relres=" << result.final_rel_residual << "\n";
    }

    std::cout << "Permeability tensor K:\n" << k_tensor.to_string() << "\n";
    return 0;
}

}  // namespace permeability
