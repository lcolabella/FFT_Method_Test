#pragma once

#include <string>

#include "geometry/BinaryMedium.hpp"

namespace permeability {

class VoxelIO {
public:
    static BinaryMedium load_text(const std::string& path, double voxel_size = 1.0);
};

}  // namespace permeability
