#pragma once

#include <cstddef>
#include <vector>

#include "core/Grid3D.hpp"

namespace permeability {

class WaveVectorGrid {
public:
    struct Components {
        double kx;
        double ky;
        double kz;
        double k2;
    };

    explicit WaveVectorGrid(Grid3D grid);

    [[nodiscard]] std::size_t spectral_size() const noexcept { return grid_.total_size(); }
    [[nodiscard]] std::size_t spectral_flat_index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return grid_.flat_index(i, j, k);
    }

    [[nodiscard]] const Grid3D& grid() const noexcept { return grid_; }
    [[nodiscard]] double kx(std::size_t idx) const noexcept { return kx_[idx]; }
    [[nodiscard]] double ky(std::size_t idx) const noexcept { return ky_[idx]; }
    [[nodiscard]] double kz(std::size_t idx) const noexcept { return kz_[idx]; }
    [[nodiscard]] double norm2(std::size_t idx) const noexcept { return norm2_[idx]; }
    [[nodiscard]] bool is_zero_mode(std::size_t idx) const noexcept { return norm2_[idx] == 0.0; }
    [[nodiscard]] Components components(std::size_t idx) const noexcept {
        return Components{kx_[idx], ky_[idx], kz_[idx], norm2_[idx]};
    }

private:
    Grid3D grid_;
    std::vector<double> kx_;
    std::vector<double> ky_;
    std::vector<double> kz_;
    std::vector<double> norm2_;
};

}  // namespace permeability
