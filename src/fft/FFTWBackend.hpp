#pragma once

#include <memory>

#include "fft/IFFTBackend.hpp"

namespace permeability {

class FFTWBackend final : public IFFTBackend {
public:
    FFTWBackend();
    ~FFTWBackend() override;

    FFTWBackend(const FFTWBackend&) = delete;
    FFTWBackend& operator=(const FFTWBackend&) = delete;

    FFTWBackend(FFTWBackend&&) noexcept;
    FFTWBackend& operator=(FFTWBackend&&) noexcept;

    void initialize(const Grid3D& grid, int thread_count) override;
    void forward(const VectorField3D& real_in,
                 ComplexVectorField3D& complex_out) override;
    void inverse(const ComplexVectorField3D& complex_in,
                 VectorField3D& real_out) override;

    [[nodiscard]] double normalization_factor() const noexcept override;
    [[nodiscard]] const Grid3D& grid() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace permeability
