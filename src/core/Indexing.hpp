#pragma once

#include <array>
#include <cstddef>

#include "core/Grid3D.hpp"

namespace permeability {

struct VoxelCoord {
    std::size_t i;
    std::size_t j;
    std::size_t k;
};

class Indexing {
public:
    static VoxelCoord to_coord(const Grid3D& grid, std::size_t flat) {
        auto [i, j, k] = grid.unflatten(flat);
        return VoxelCoord{i, j, k};
    }

    static std::size_t to_flat(const Grid3D& grid, const VoxelCoord& c) {
        return grid.flat_index(c.i, c.j, c.k);
    }

    static VoxelCoord periodic_neighbor(const Grid3D& grid, const VoxelCoord& c, int di, int dj, int dk) {
        const auto [i, j, k] = grid.periodic_neighbor_coord(c.i, c.j, c.k, di, dj, dk);
        return VoxelCoord{i, j, k};
    }
};

}  // namespace permeability
