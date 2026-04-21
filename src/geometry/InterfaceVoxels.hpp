#pragma once

#include <cstddef>
#include <vector>

#include "geometry/BinaryMedium.hpp"

namespace permeability {

class InterfaceVoxels {
public:
    static std::vector<std::size_t> compute_interface_only(const BinaryMedium& medium);
};

}  // namespace permeability
