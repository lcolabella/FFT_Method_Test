#pragma once

#include <cstddef>

#include "core/Field3D.hpp"
#include "core/Grid3D.hpp"

namespace permeability {

class IFFTBackend {
public:
    virtual ~IFFTBackend() = default;

    virtual void initialize(const Grid3D& grid, int thread_count) = 0;

    virtual void forward(const VectorField3D& real_in,
                         ComplexVectorField3D& complex_out) = 0;

    virtual void inverse(const ComplexVectorField3D& complex_in,
                         VectorField3D& real_out) = 0;

    [[nodiscard]] virtual double normalization_factor() const noexcept = 0;
    [[nodiscard]] virtual const Grid3D& grid() const noexcept = 0;
};

}  // namespace permeability
