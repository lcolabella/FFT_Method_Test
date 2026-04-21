#include <cassert>
#include <cmath>
#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "core/Types.hpp"
#include "geometry/BinaryMedium.hpp"

int main() {
    permeability::Grid3D g(2, 2, 1);

    // Mask values: 1 = fluid, 0 = solid
    // layout (k=0): [0,1;1,0]
    std::vector<unsigned char> mask(g.total_size(), 1);
    mask[g.flat_index(0, 0, 0)] = 0;
    mask[g.flat_index(1, 1, 0)] = 0;

    permeability::BinaryMedium medium(g, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::FullSolid);
    permeability::BackgroundForce bg(medium, support);

    const permeability::Real3 grad{1.0, 0.0, 0.0};
    permeability::VectorField3D f0 = bg.build(grad);

    // NF = 2 fluids, NB = 2 solids in support => NF/NB = 1.
    // fluid voxels: -g; support-solid voxels: -(NF/NB)g = -g.
    const std::size_t s0 = g.flat_index(0, 0, 0);
    const std::size_t f1 = g.flat_index(1, 0, 0);
    const std::size_t f2 = g.flat_index(0, 1, 0);
    const std::size_t s1 = g.flat_index(1, 1, 0);

    assert(std::abs(f0.x()[f1] + 1.0) < 1e-12);
    assert(std::abs(f0.x()[f2] + 1.0) < 1e-12);
    assert(std::abs(f0.x()[s0] + 1.0) < 1e-12);
    assert(std::abs(f0.x()[s1] + 1.0) < 1e-12);

    assert(std::abs(f0.y()[f1]) < 1e-12);
    assert(std::abs(f0.z()[f1]) < 1e-12);

    return 0;
}
