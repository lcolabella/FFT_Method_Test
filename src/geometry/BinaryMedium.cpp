#include "geometry/BinaryMedium.hpp"

#include <stdexcept>

namespace permeability {

BinaryMedium::BinaryMedium(Grid3D grid, std::vector<unsigned char> fluid_mask)
    : grid_(std::move(grid)), mask_(std::move(fluid_mask)) {
    if (mask_.size() != grid_.total_size()) {
        throw std::invalid_argument("BinaryMedium mask size must match grid size");
    }
}

bool BinaryMedium::is_fluid(std::size_t flat_idx) const noexcept {
    return mask_[flat_idx] != 0;
}

bool BinaryMedium::is_solid(std::size_t flat_idx) const noexcept {
    return !is_fluid(flat_idx);
}

bool BinaryMedium::is_fluid(std::size_t i, std::size_t j, std::size_t k) const noexcept {
    const std::size_t idx = grid_.flat_index(
        grid_.periodic_index(static_cast<long long>(i), grid_.nx()),
        grid_.periodic_index(static_cast<long long>(j), grid_.ny()),
        grid_.periodic_index(static_cast<long long>(k), grid_.nz()));
    return is_fluid(idx);
}

bool BinaryMedium::is_solid(std::size_t i, std::size_t j, std::size_t k) const noexcept {
    return !is_fluid(i, j, k);
}

double BinaryMedium::porosity() const {
    if (mask_.empty()) {
        return 0.0;
    }
    std::size_t fluid_count = 0;
    for (unsigned char v : mask_) {
        fluid_count += (v != 0) ? 1 : 0;
    }
    return static_cast<double>(fluid_count) / static_cast<double>(mask_.size());
}

}  // namespace permeability
