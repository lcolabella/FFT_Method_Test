#include <cassert>
#include <vector>

#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/TrialForceField.hpp"
#include "core/Types.hpp"
#include "geometry/BinaryMedium.hpp"

int main() {
    permeability::Grid3D g(4, 4, 4);
    std::vector<unsigned char> mask(g.total_size(), 1);
    mask[g.flat_index(1, 1, 1)] = 0;
    mask[g.flat_index(2, 2, 2)] = 0;

    permeability::BinaryMedium medium(g, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::FullSolid);

    assert(support.num_active_voxels() == 2);
    for (std::size_t v : support.active_voxels()) {
        assert(support.contains_voxel(v));
        const std::size_t local = support.local_index(v);
        assert(support.global_index(local) == v);
    }

    permeability::TrialForceField compact(support.num_active_voxels());
    compact.fill_zero();
    compact.x(0) = 1.0;
    compact.y(0) = 2.0;
    compact.z(0) = 3.0;
    compact.x(1) = 4.0;
    compact.y(1) = 5.0;
    compact.z(1) = 6.0;

    permeability::VectorField3D full(g);
    full.fill_zero();
    compact.scatter_to_full(full, support);

    permeability::TrialForceField gathered(support.num_active_voxels());
    gathered.gather_from_full(full, support);
    assert(gathered.raw() == compact.raw());

    return 0;
}
