#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/Grid3D.hpp"

namespace permeability {

template <typename T>
class Array3D {
public:
    explicit Array3D(Grid3D grid, const T& value = T())
        : grid_(std::move(grid)), data_(grid_.total_size(), value) {}

    [[nodiscard]] const Grid3D& grid() const noexcept { return grid_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] T& operator[](std::size_t idx) noexcept { return data_[idx]; }
    [[nodiscard]] const T& operator[](std::size_t idx) const noexcept { return data_[idx]; }

    [[nodiscard]] T& at(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return data_[grid_.flat_index(i, j, k)];
    }

    [[nodiscard]] const T& at(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return data_[grid_.flat_index(i, j, k)];
    }

    [[nodiscard]] T& operator()(std::size_t i, std::size_t j, std::size_t k) noexcept {
        return at(i, j, k);
    }

    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j, std::size_t k) const noexcept {
        return at(i, j, k);
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    [[nodiscard]] std::vector<T>& storage() noexcept { return data_; }
    [[nodiscard]] const std::vector<T>& storage() const noexcept { return data_; }

private:
    Grid3D grid_;
    std::vector<T> data_;
};

}  // namespace permeability
