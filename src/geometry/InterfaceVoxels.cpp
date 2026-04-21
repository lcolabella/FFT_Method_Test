#include "geometry/InterfaceVoxels.hpp"

#include <algorithm>

namespace permeability {

std::vector<std::size_t> InterfaceVoxels::compute_interface_only(const BinaryMedium& medium) {
    const Grid3D& g = medium.grid();
    std::vector<std::size_t> out;
    out.reserve(g.total_size() / 4);

    for (std::size_t flat = 0; flat < g.total_size(); ++flat) {
        if (!medium.is_solid(flat)) {
            continue;
        }

        auto [i, j, k] = g.unflatten(flat);
        bool interface = false;

        for (int dk = -1; dk <= 1 && !interface; ++dk) {
            for (int dj = -1; dj <= 1 && !interface; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (di == 0 && dj == 0 && dk == 0) {
                        continue;
                    }
                    const std::size_t nidx = g.periodic_neighbor(i, j, k, di, dj, dk);
                    if (medium.is_fluid(nidx)) {
                        interface = true;
                        break;
                    }
                }
            }
        }

        if (interface) {
            out.push_back(flat);
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

}  // namespace permeability
