#pragma once

#include <cstddef>
#include <vector>

#include "core/Types.hpp"
#include "geometry/BinaryMedium.hpp"

namespace permeability {

class ForceSupport {
public:
    ForceSupport(const BinaryMedium& medium, SupportMode mode);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool contains_voxel(std::size_t voxel_idx) const;
    [[nodiscard]] std::size_t local_index(std::size_t global_idx) const;
    [[nodiscard]] std::size_t global_index(std::size_t local_idx) const;

    [[nodiscard]] const std::vector<std::size_t>& active_voxels() const noexcept { return active_voxels_; }
    [[nodiscard]] std::size_t num_active_voxels() const noexcept { return active_voxels_.size(); }
    [[nodiscard]] const Grid3D& grid() const noexcept { return grid_; }

private:
    Grid3D grid_;
    std::vector<std::size_t> active_voxels_;
    std::vector<std::size_t> voxel_to_local_;
    static constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
};

}  // namespace permeability
