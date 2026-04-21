#include "bignonnet/ForceSupport.hpp"

#include <stdexcept>

#include "geometry/InterfaceVoxels.hpp"

namespace permeability {

ForceSupport::ForceSupport(const BinaryMedium& medium, SupportMode mode)
    : grid_(medium.grid()), voxel_to_local_(grid_.total_size(), kInvalid) {
    if (mode == SupportMode::InterfaceOnly) {
        active_voxels_ = InterfaceVoxels::compute_interface_only(medium);
    } else {
        active_voxels_.reserve(grid_.total_size());
        for (std::size_t idx = 0; idx < grid_.total_size(); ++idx) {
            if (medium.is_solid(idx)) {
                active_voxels_.push_back(idx);
            }
        }
    }

    for (std::size_t local = 0; local < active_voxels_.size(); ++local) {
        voxel_to_local_[active_voxels_[local]] = local;
    }
}

bool ForceSupport::contains_voxel(std::size_t voxel_idx) const {
    return voxel_idx < voxel_to_local_.size() && voxel_to_local_[voxel_idx] != kInvalid;
}

std::size_t ForceSupport::size() const noexcept {
    return 3 * active_voxels_.size();
}

std::size_t ForceSupport::local_index(std::size_t global_idx) const {
    if (!contains_voxel(global_idx)) {
        throw std::out_of_range("Voxel is not in active support");
    }
    return voxel_to_local_[global_idx];
}

std::size_t ForceSupport::global_index(std::size_t local_idx) const {
    if (local_idx >= active_voxels_.size()) {
        throw std::out_of_range("Support local index out of range");
    }
    return active_voxels_[local_idx];
}

}  // namespace permeability
