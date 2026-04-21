#include <cassert>
#include <algorithm>
#include <vector>

#include "geometry/BinaryMedium.hpp"
#include "geometry/InterfaceVoxels.hpp"

int main() {
    permeability::Grid3D g(4, 4, 4);
    std::vector<unsigned char> mask(g.total_size(), 1);

    // A 2x2x2 solid block in the center; all these solids are interface for this tiny case.
    for (std::size_t k = 1; k <= 2; ++k) {
        for (std::size_t j = 1; j <= 2; ++j) {
            for (std::size_t i = 1; i <= 2; ++i) {
                mask[g.flat_index(i, j, k)] = 0;
            }
        }
    }

    permeability::BinaryMedium medium(g, std::move(mask));
    const auto fi = permeability::InterfaceVoxels::compute_interface_only(medium);

    assert(!fi.empty());
    assert(fi.size() == 8);
    assert(std::is_sorted(fi.begin(), fi.end()));
    return 0;
}
