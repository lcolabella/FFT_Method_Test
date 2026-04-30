#pragma once

#include <cstdint>
#include <string>

#include "core/Geometry.hpp"
#include "core/Types.hpp"
#include "geometry/BinaryMedium.hpp"

namespace permeability {

class VoxelIO {
public:
    // Auto-dispatch based on file extension: .fgeo -> load_fgeo, everything else -> load_text.
    static BinaryMedium load(const std::string& path,
                             Scalar voxel_size = Scalar(1),
                             int fluid_id = 1);

    // Load a text voxel geometry file.  Voxels matching fluid_id are treated as
    // fluid (mask=1); all other values are treated as solid (mask=0).
    // fluid_id defaults to 1 for backward compatibility with binary 0/1 files.
    static BinaryMedium load_text(const std::string& path,
                                  Scalar voxel_size = Scalar(1),
                                  int fluid_id = 1);

    // Load a binary .fgeo geometry file.
    // The file stores material IDs; voxels matching fluid_id are fluid.
    static BinaryMedium load_fgeo(const std::string& path,
                                  Scalar voxel_size = Scalar(1),
                                  int fluid_id = 1);

    // Load geometry preserving raw material IDs (for Brinkman solver).
    // Returns common::Geometry with uint32 material IDs; dispatches on .fgeo extension.
    static common::Geometry load_geometry(const std::string& path);
};

}  // namespace permeability
