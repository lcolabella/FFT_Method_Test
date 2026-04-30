#pragma once

#if defined(PERMEABILITY_USE_SYCL)

#include <memory>
#include <string>

#include "fft/IFFTBackend.hpp"

namespace permeability {

class SYCLFFTBackend final : public IFFTBackend {
public:
    SYCLFFTBackend();
    ~SYCLFFTBackend() override;

    SYCLFFTBackend(const SYCLFFTBackend&) = delete;
    SYCLFFTBackend& operator=(const SYCLFFTBackend&) = delete;

    SYCLFFTBackend(SYCLFFTBackend&&) noexcept;
    SYCLFFTBackend& operator=(SYCLFFTBackend&&) noexcept;

    void initialize(const Grid3D& grid, int thread_count) override;
    void forward(const VectorField3D& real_in,
                 ComplexVectorField3D& complex_out) override;
    void inverse(const ComplexVectorField3D& complex_in,
                 VectorField3D& real_out) override;

    bool apply_spectral_green_tensor(const std::vector<Scalar>& spectral_tensor_xx,
                                     const std::vector<Scalar>& spectral_tensor_xy,
                                     const std::vector<Scalar>& spectral_tensor_xz,
                                     const std::vector<Scalar>& spectral_tensor_yy,
                                     const std::vector<Scalar>& spectral_tensor_yz,
                                     const std::vector<Scalar>& spectral_tensor_zz,
                                     const ComplexVectorField3D& spectral_force,
                                     ComplexVectorField3D& spectral_velocity) override;

    [[nodiscard]] double normalization_factor() const noexcept override;
    [[nodiscard]] const Grid3D& grid() const noexcept override;

    // Returns the SYCL device name string (useful for startup logging).
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] std::string device_info() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace permeability

#endif  // PERMEABILITY_USE_SYCL
