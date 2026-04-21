#pragma once

#include <cstddef>
#include <vector>

#include "core/Grid3D.hpp"

namespace permeability {

class BinaryMedium {
public:
    BinaryMedium(Grid3D grid, std::vector<unsigned char> fluid_mask);

    [[nodiscard]] const Grid3D& grid() const noexcept { return grid_; }
    [[nodiscard]] std::size_t size() const noexcept { return mask_.size(); }

    [[nodiscard]] bool is_fluid(std::size_t flat_idx) const noexcept;
    [[nodiscard]] bool is_solid(std::size_t flat_idx) const noexcept;
    [[nodiscard]] bool is_fluid(std::size_t i, std::size_t j, std::size_t k) const noexcept;
    [[nodiscard]] bool is_solid(std::size_t i, std::size_t j, std::size_t k) const noexcept;
    [[nodiscard]] double porosity() const;

    [[nodiscard]] const std::vector<unsigned char>& mask() const noexcept { return mask_; }

private:
    Grid3D grid_;
    std::vector<unsigned char> mask_;
};

}  // namespace permeability
