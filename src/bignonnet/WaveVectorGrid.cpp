#include "bignonnet/WaveVectorGrid.hpp"

#include <cmath>

namespace permeability {

WaveVectorGrid::WaveVectorGrid(Grid3D grid)
    : grid_(std::move(grid)),
      kx_(grid_.total_size(), 0.0),
      ky_(grid_.total_size(), 0.0),
      kz_(grid_.total_size(), 0.0),
      norm2_(grid_.total_size(), 0.0) {
    const double two_pi = 2.0 * 3.14159265358979323846;

    for (std::size_t idx = 0; idx < grid_.total_size(); ++idx) {
        auto [i, j, k] = grid_.unflatten(idx);

        const long long ii = (2 * i <= grid_.nx()) ? static_cast<long long>(i)
                                                   : static_cast<long long>(i) - static_cast<long long>(grid_.nx());
        const long long jj = (2 * j <= grid_.ny()) ? static_cast<long long>(j)
                                                   : static_cast<long long>(j) - static_cast<long long>(grid_.ny());
        const long long kk = (2 * k <= grid_.nz()) ? static_cast<long long>(k)
                                                   : static_cast<long long>(k) - static_cast<long long>(grid_.nz());

        kx_[idx] = two_pi * static_cast<double>(ii) / grid_.lx();
        ky_[idx] = two_pi * static_cast<double>(jj) / grid_.ly();
        kz_[idx] = two_pi * static_cast<double>(kk) / grid_.lz();
        norm2_[idx] = kx_[idx] * kx_[idx] + ky_[idx] * ky_[idx] + kz_[idx] * kz_[idx];
    }
}

}  // namespace permeability
