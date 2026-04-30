#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace common {

struct Geometry {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    std::vector<std::uint32_t> materialIds;

    std::size_t voxelCount() const {
        return nx * ny * nz;
    }
};

} // namespace common
