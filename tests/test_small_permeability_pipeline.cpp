#include <cassert>
#include <cmath>
#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalOperator.hpp"
#include "bignonnet/VariationalRHS.hpp"
#include "bignonnet/VelocityRecovery.hpp"
#include "fft/FFTWBackend.hpp"
#include "geometry/BinaryMedium.hpp"
#include "post/PermeabilityTensor.hpp"
#include "solver/MinresSolver.hpp"

int main() {
    permeability::Grid3D grid(4, 4, 4);
    std::vector<unsigned char> mask(grid.total_size(), 1);
    mask[grid.flat_index(1, 1, 1)] = 0;
    mask[grid.flat_index(1, 2, 1)] = 0;
    mask[grid.flat_index(2, 1, 2)] = 0;
    mask[grid.flat_index(2, 2, 2)] = 0;

    permeability::BinaryMedium medium(grid, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::InterfaceOnly);
    assert(support.num_active_voxels() > 0);

    permeability::FFTWBackend fft;
    permeability::GreenDiscretization disc{permeability::GreenDiscretizationType::EnergyConsistent};
    permeability::GreenOperator green(grid, disc, 1.0, fft, 1, 2);

    permeability::BackgroundForce bg(medium, support);
    permeability::VariationalRHS rhs_builder(support, green, bg);
    permeability::VariationalOperator op(support, green);
    permeability::MinresSolver solver(1e-8, 60);
    permeability::VelocityRecovery recovery(support, green, bg);

    const permeability::Real3 g{1.0, 0.0, 0.0};
    const std::vector<permeability::Scalar> b = rhs_builder.build(g);
    const permeability::SolverResult sr = solver.solve(op, b);

    assert(sr.solution.size() == op.size());
    assert(!sr.residual_history.empty());
    assert(std::isfinite(sr.final_rel_residual));

    const permeability::VectorField3D v = recovery.recover(g, sr.solution);
    const permeability::Real3 avg = recovery.average_over_domain(v);

    permeability::PermeabilityTensor K;
    K.set_column(0, avg);

    const auto data = K.data();
    for (auto e : data) {
        assert(std::isfinite(e));
    }

    return 0;
}
