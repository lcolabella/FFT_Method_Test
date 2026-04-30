#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/Array3D.hpp"
#include "core/Types.hpp"

namespace permeability {

using ScalarField3D = Array3D<Scalar>;

class VectorField3D {
public:
    explicit VectorField3D(const Grid3D& grid)
        : x_(grid, 0.0), y_(grid, 0.0), z_(grid, 0.0) {}

    [[nodiscard]] const Grid3D& grid() const noexcept { return x_.grid(); }
    [[nodiscard]] std::size_t size() const noexcept { return x_.size(); }

    [[nodiscard]] ScalarField3D& x() noexcept { return x_; }
    [[nodiscard]] ScalarField3D& y() noexcept { return y_; }
    [[nodiscard]] ScalarField3D& z() noexcept { return z_; }
    [[nodiscard]] const ScalarField3D& x() const noexcept { return x_; }
    [[nodiscard]] const ScalarField3D& y() const noexcept { return y_; }
    [[nodiscard]] const ScalarField3D& z() const noexcept { return z_; }

    void fill(Scalar value) {
        x_.fill(value);
        y_.fill(value);
        z_.fill(value);
    }

    void fill_zero() {
        fill(Scalar(0));
    }

    void scale(Scalar a) {
        for (std::size_t i = 0; i < size(); ++i) {
            x_[i] *= a;
            y_[i] *= a;
            z_[i] *= a;
        }
    }

    void axpy(Scalar a, const VectorField3D& other) {
        ensure_same_grid(other);
        for (std::size_t i = 0; i < size(); ++i) {
            x_[i] += a * other.x_[i];
            y_[i] += a * other.y_[i];
            z_[i] += a * other.z_[i];
        }
    }

    [[nodiscard]] Real3 mean_on_indices(const std::vector<std::size_t>& indices) const {
        Real3 m{Scalar(0), Scalar(0), Scalar(0)};
        if (indices.empty()) {
            return m;
        }
        for (std::size_t idx : indices) {
            m[0] += x_[idx];
            m[1] += y_[idx];
            m[2] += z_[idx];
        }
        const Scalar inv = Scalar(1) / static_cast<Scalar>(indices.size());
        m[0] *= inv;
        m[1] *= inv;
        m[2] *= inv;
        return m;
    }

    [[nodiscard]] Real3 mean_over_all() const {
        Real3 m{Scalar(0), Scalar(0), Scalar(0)};
        if (size() == 0) {
            return m;
        }
        for (std::size_t i = 0; i < size(); ++i) {
            m[0] += x_[i];
            m[1] += y_[i];
            m[2] += z_[i];
        }
        const Scalar inv = Scalar(1) / static_cast<Scalar>(size());
        m[0] *= inv;
        m[1] *= inv;
        m[2] *= inv;
        return m;
    }

    [[nodiscard]] Scalar norm2() const {
        Scalar s = Scalar(0);
        for (std::size_t i = 0; i < size(); ++i) {
            s += x_[i] * x_[i] + y_[i] * y_[i] + z_[i] * z_[i];
        }
        return std::sqrt(s);
    }

    [[nodiscard]] Real3 mean_componentwise() const {
        return mean_over_all();
    }

private:
    void ensure_same_grid(const VectorField3D& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("VectorField3D grid mismatch");
        }
    }

    ScalarField3D x_;
    ScalarField3D y_;
    ScalarField3D z_;
};

class ComplexVectorField3D {
public:
    explicit ComplexVectorField3D(const Grid3D& grid)
        : x_(grid, ScalarComplex(0, 0)),
          y_(grid, ScalarComplex(0, 0)),
          z_(grid, ScalarComplex(0, 0)) {}

    [[nodiscard]] const Grid3D& grid() const noexcept { return x_.grid(); }
    [[nodiscard]] std::size_t size() const noexcept { return x_.size(); }

    [[nodiscard]] Array3D<ScalarComplex>& x() noexcept { return x_; }
    [[nodiscard]] Array3D<ScalarComplex>& y() noexcept { return y_; }
    [[nodiscard]] Array3D<ScalarComplex>& z() noexcept { return z_; }
    [[nodiscard]] const Array3D<ScalarComplex>& x() const noexcept { return x_; }
    [[nodiscard]] const Array3D<ScalarComplex>& y() const noexcept { return y_; }
    [[nodiscard]] const Array3D<ScalarComplex>& z() const noexcept { return z_; }

    void fill(ScalarComplex value) {
        x_.fill(value);
        y_.fill(value);
        z_.fill(value);
    }

    void fill_zero() {
        fill(ScalarComplex(0, 0));
    }

    void scale(Scalar alpha) {
        for (std::size_t i = 0; i < size(); ++i) {
            x_[i] *= alpha;
            y_[i] *= alpha;
            z_[i] *= alpha;
        }
    }

    void axpy(Scalar alpha, const ComplexVectorField3D& other) {
        if (size() != other.size()) {
            throw std::invalid_argument("ComplexVectorField3D grid mismatch");
        }
        for (std::size_t i = 0; i < size(); ++i) {
            x_[i] += alpha * other.x_[i];
            y_[i] += alpha * other.y_[i];
            z_[i] += alpha * other.z_[i];
        }
    }

    [[nodiscard]] Scalar norm2() const {
        Scalar s = Scalar(0);
        for (std::size_t i = 0; i < size(); ++i) {
            s += std::norm(x_[i]) + std::norm(y_[i]) + std::norm(z_[i]);
        }
        return std::sqrt(s);
    }

    [[nodiscard]] Complex3 mean_componentwise() const {
        Complex3 m{ScalarComplex(0, 0),
                   ScalarComplex(0, 0),
                   ScalarComplex(0, 0)};
        if (size() == 0) {
            return m;
        }
        for (std::size_t i = 0; i < size(); ++i) {
            m[0] += x_[i];
            m[1] += y_[i];
            m[2] += z_[i];
        }
        const Scalar inv = Scalar(1) / static_cast<Scalar>(size());
        m[0] *= inv;
        m[1] *= inv;
        m[2] *= inv;
        return m;
    }

private:
    Array3D<ScalarComplex> x_;
    Array3D<ScalarComplex> y_;
    Array3D<ScalarComplex> z_;
};

}  // namespace permeability
