#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>

#include "core/Types.hpp"

namespace permeability {

class Grid3D {
public:
    Grid3D(std::size_t nx, std::size_t ny, std::size_t nz,
           Scalar hx = Scalar(1), Scalar hy = Scalar(1), Scalar hz = Scalar(1),
           Scalar lx = Scalar(0), Scalar ly = Scalar(0), Scalar lz = Scalar(0))
        : nx_(nx), ny_(ny), nz_(nz),
          hx_(hx), hy_(hy), hz_(hz),
          lx_(lx > Scalar(0) ? lx : hx * static_cast<Scalar>(nx)),
          ly_(ly > Scalar(0) ? ly : hy * static_cast<Scalar>(ny)),
          lz_(lz > Scalar(0) ? lz : hz * static_cast<Scalar>(nz)) {
        if (nx_ == 0 || ny_ == 0 || nz_ == 0) {
            throw std::invalid_argument("Grid dimensions must be positive");
        }
    }

    [[nodiscard]] std::size_t nx() const noexcept { return nx_; }
    [[nodiscard]] std::size_t ny() const noexcept { return ny_; }
    [[nodiscard]] std::size_t nz() const noexcept { return nz_; }

    [[nodiscard]] Scalar hx() const noexcept { return hx_; }
    [[nodiscard]] Scalar hy() const noexcept { return hy_; }
    [[nodiscard]] Scalar hz() const noexcept { return hz_; }

    [[nodiscard]] Scalar lx() const noexcept { return lx_; }
    [[nodiscard]] Scalar ly() const noexcept { return ly_; }
    [[nodiscard]] Scalar lz() const noexcept { return lz_; }

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
    Scalar hx_;
    Scalar hy_;
    Scalar hz_;
    Scalar lx_;
    Scalar ly_;
    Scalar lz_;
};

}  // namespace permeability
