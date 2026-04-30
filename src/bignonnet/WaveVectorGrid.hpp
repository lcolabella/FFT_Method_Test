#pragma once

#include <cstddef>
#include <vector>

#include "core/Grid3D.hpp"
#include "core/Types.hpp"

namespace permeability {

class WaveVectorGrid {
public:
    struct Components {
        Scalar kx;
        Scalar ky;
        Scalar kz;
        Scalar k2;
    };

    explicit WaveVectorGrid(Grid3D grid);

    [[nodiscard]] std::size_t spectral_size() const noexcept { return grid_.total_size(); }
    [[nodiscard]] std::size_t spectral_flat_index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return grid_.flat_index(i, j, k);
    }

    [[nodiscard]] const Grid3D& grid() const noexcept { return grid_; }
    [[nodiscard]] Scalar kx(std::size_t idx) const noexcept { return kx_[idx]; }
    [[nodiscard]] Scalar ky(std::size_t idx) const noexcept { return ky_[idx]; }
    [[nodiscard]] Scalar kz(std::size_t idx) const noexcept { return kz_[idx]; }
    [[nodiscard]] Scalar norm2(std::size_t idx) const noexcept { return norm2_[idx]; }
    [[nodiscard]] bool is_zero_mode(std::size_t idx) const noexcept { return norm2_[idx] == Scalar(0); }
    [[nodiscard]] Components components(std::size_t idx) const noexcept {
        return Components{kx_[idx], ky_[idx], kz_[idx], norm2_[idx]};
    }

private:
    Grid3D grid_;
    std::vector<Scalar> kx_;
    std::vector<Scalar> ky_;
    std::vector<Scalar> kz_;
    std::vector<Scalar> norm2_;
};

}  // namespace permeability
