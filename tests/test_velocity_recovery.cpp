#include <cassert>
#include <cmath>
#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalRHS.hpp"
#include "bignonnet/VelocityRecovery.hpp"
#include "fft/FFTWBackend.hpp"
#include "geometry/BinaryMedium.hpp"

int main() {
    permeability::Grid3D grid(4, 4, 4);
    std::vector<unsigned char> mask(grid.total_size(), 1);
    mask[grid.flat_index(1, 1, 1)] = 0;
    mask[grid.flat_index(2, 2, 2)] = 0;
    mask[grid.flat_index(2, 1, 2)] = 0;

    permeability::BinaryMedium medium(grid, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::InterfaceOnly);

    permeability::FFTWBackend fft;
    permeability::GreenDiscretization disc{permeability::GreenDiscretizationType::EnergyConsistent};
    permeability::GreenOperator green(grid, disc, 1.0, fft, 1, 2);
    permeability::BackgroundForce bg(medium, support);
    permeability::VelocityRecovery recovery(support, green, bg);

    std::vector<double> x(3 * support.num_active_voxels(), 0.0);
    for (std::size_t n = 0; n < support.num_active_voxels(); ++n) {
        x[3 * n + 0] = 0.1 * static_cast<double>(n + 1);
        x[3 * n + 1] = -0.05 * static_cast<double>(n + 2);
        x[3 * n + 2] = 0.03 * static_cast<double>(n + 3);
    }

    const permeability::Real3 g{1.0, 0.0, 0.0};
    permeability::VectorField3D v = recovery.recover(g, x);

    const permeability::Real3 mean_support = recovery.average_over_support(v);
    assert(std::abs(mean_support[0]) < 1e-8);
    assert(std::abs(mean_support[1]) < 1e-8);
    assert(std::abs(mean_support[2]) < 1e-8);

    for (std::size_t i = 0; i < v.size(); ++i) {
        assert(std::isfinite(v.x()[i]));
        assert(std::isfinite(v.y()[i]));
        assert(std::isfinite(v.z()[i]));
    }

    return 0;
}
