#include <cassert>
#include <cmath>
#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalRHS.hpp"
#include "fft/FFTWBackend.hpp"
#include "geometry/BinaryMedium.hpp"

int main() {
    permeability::Grid3D grid(3, 3, 3);
    std::vector<unsigned char> mask(grid.total_size(), 1);
    mask[grid.flat_index(1, 1, 1)] = 0;
    mask[grid.flat_index(2, 1, 1)] = 0;

    permeability::BinaryMedium medium(grid, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::InterfaceOnly);

    permeability::FFTWBackend fft;
    permeability::GreenDiscretization disc{permeability::GreenDiscretizationType::EnergyConsistent};
    permeability::GreenOperator green(grid, disc, 1.0, fft, 1, 2);
    permeability::BackgroundForce bg(medium, support);
    permeability::VariationalRHS rhs_builder(support, green, bg);

    const permeability::Real3 g{1.0, 0.0, 0.0};
    const std::vector<double> b1 = rhs_builder.build(g);
    const std::vector<double> b2 = rhs_builder.build(g);

    assert(b1.size() == 3 * support.num_active_voxels());
    assert(b1.size() == b2.size());

    for (std::size_t i = 0; i < b1.size(); ++i) {
        assert(std::abs(b1[i] - b2[i]) < 1e-12);
    }

    for (std::size_t comp = 0; comp < 3; ++comp) {
        double m = 0.0;
        for (std::size_t n = 0; n < support.num_active_voxels(); ++n) {
            m += b1[3 * n + comp];
        }
        m /= static_cast<double>(support.num_active_voxels());
        assert(std::abs(m) < 1e-10);
    }

    return 0;
}
