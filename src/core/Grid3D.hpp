#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace permeability {

class Grid3D {
public:
    Grid3D(std::size_t nx, std::size_t ny, std::size_t nz,
           double hx = 1.0, double hy = 1.0, double hz = 1.0,
           double lx = 0.0, double ly = 0.0, double lz = 0.0)
        : nx_(nx), ny_(ny), nz_(nz),
          hx_(hx), hy_(hy), hz_(hz),
          lx_(lx > 0.0 ? lx : hx * static_cast<double>(nx)),
          ly_(ly > 0.0 ? ly : hy * static_cast<double>(ny)),
          lz_(lz > 0.0 ? lz : hz * static_cast<double>(nz)) {
        if (nx_ == 0 || ny_ == 0 || nz_ == 0) {
            throw std::invalid_argument("Grid dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t nx() const noexcept { return nx_; }
    [[nodiscard]] std::size_t ny() const noexcept { return ny_; }
    [[nodiscard]] std::size_t nz() const noexcept { return nz_; }

    [[nodiscard]] double hx() const noexcept { return hx_; }
    [[nodiscard]] double hy() const noexcept { return hy_; }
    [[nodiscard]] double hz() const noexcept { return hz_; }

    [[nodiscard]] double lx() const noexcept { return lx_; }
    [[nodiscard]] double ly() const noexcept { return ly_; }
    [[nodiscard]] double lz() const noexcept { return lz_; }

    [[nodiscard]] std::size_t total_size() const noexcept { return nx_ * ny_ * nz_; }

    [[nodiscard]] std::size_t flat_index(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return (k * ny_ + j) * nx_ + i;
    }

    [[nodiscard]] std::tuple<std::size_t, std::size_t, std::size_t> unflatten(std::size_t idx) const noexcept {
        const std::size_t i = idx % nx_;
        idx /= nx_;
        const std::size_t j = idx % ny_;
        idx /= ny_;
        const std::size_t k = idx;
        return {i, j, k};
    }

    [[nodiscard]] std::size_t wrap_i(long long i) const noexcept {
        return wrap(i, nx_);
    }

    [[nodiscard]] std::size_t wrap_j(long long j) const noexcept {
        return wrap(j, ny_);
    }

    [[nodiscard]] std::size_t wrap_k(long long k) const noexcept {
        return wrap(k, nz_);
    }

    [[nodiscard]] std::size_t periodic_neighbor(std::size_t i, std::size_t j, std::size_t k,
                                                int di, int dj, int dk) const noexcept {
        return flat_index(
            wrap_i(static_cast<long long>(i) + di),
            wrap_j(static_cast<long long>(j) + dj),
            wrap_k(static_cast<long long>(k) + dk));
    }

    [[nodiscard]] std::size_t periodic_index(long long i, std::size_t n) const noexcept {
        return wrap(i, n);
    }

    [[nodiscard]] std::tuple<std::size_t, std::size_t, std::size_t> periodic_neighbor_coord(
        std::size_t i, std::size_t j, std::size_t k, int di, int dj, int dk) const noexcept {
        return {
            wrap_i(static_cast<long long>(i) + di),
            wrap_j(static_cast<long long>(j) + dj),
            wrap_k(static_cast<long long>(k) + dk)};
    }

private:
    static std::size_t wrap(long long v, std::size_t n) noexcept {
        long long m = v % static_cast<long long>(n);
        if (m < 0) {
            m += static_cast<long long>(n);
        }
        return static_cast<std::size_t>(m);
    }

    std::size_t nx_;
    std::size_t ny_;
    std::size_t nz_;
    double hx_;
    double hy_;
    double hz_;
    double lx_;
    double ly_;
    double lz_;
};

}  // namespace permeability
