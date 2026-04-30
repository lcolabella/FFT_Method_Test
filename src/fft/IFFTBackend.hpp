#pragma once

#include <cstddef>
#include <string>
#include <vector>

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

    // Optional acceleration hook for Green tensor application in spectral space.
    // Returns true if the backend applied the operation and wrote spectral_velocity.
    virtual bool apply_spectral_green_tensor(const std::vector<Scalar>& spectral_tensor_xx,
                                             const std::vector<Scalar>& spectral_tensor_xy,
                                             const std::vector<Scalar>& spectral_tensor_xz,
                                             const std::vector<Scalar>& spectral_tensor_yy,
                                             const std::vector<Scalar>& spectral_tensor_yz,
                                             const std::vector<Scalar>& spectral_tensor_zz,
                                             const ComplexVectorField3D& spectral_force,
                                             ComplexVectorField3D& spectral_velocity) {
        (void)spectral_tensor_xx;
        (void)spectral_tensor_xy;
        (void)spectral_tensor_xz;
        (void)spectral_tensor_yy;
        (void)spectral_tensor_yz;
        (void)spectral_tensor_zz;
        (void)spectral_force;
        (void)spectral_velocity;
        return false;
    }

    [[nodiscard]] virtual double normalization_factor() const noexcept = 0;
    [[nodiscard]] virtual const Grid3D& grid() const noexcept = 0;

    // Optional: human-readable device description for logging (empty = CPU/default).
    [[nodiscard]] virtual std::string device_info() const { return {}; }
};

}  // namespace permeability
