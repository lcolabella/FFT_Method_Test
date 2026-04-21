#pragma once

#include <array>
#include <vector>

#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/WaveVectorGrid.hpp"
#include "core/Field3D.hpp"
#include "fft/IFFTBackend.hpp"

namespace permeability {

class GreenOperator {
public:
    GreenOperator(const Grid3D& grid,
                  GreenDiscretization discretization,
                  double viscosity,
                  IFFTBackend& fft_backend,
                  int fft_threads,
                  int p_radius);

    void apply(const VectorField3D& force, VectorField3D& velocity_like);

    [[nodiscard]] const WaveVectorGrid& wave_vectors() const noexcept { return wave_vectors_; }
    [[nodiscard]] GreenDiscretizationType discretization_type() const noexcept { return discretization_.type; }
    [[nodiscard]] std::size_t spectral_size() const noexcept { return spectral_tensor_xx_.size(); }
    [[nodiscard]] bool has_nan_in_tensor() const noexcept;
    [[nodiscard]] std::array<double, 6> tensor_components(std::size_t idx) const;

private:
    Grid3D grid_;
    GreenDiscretization discretization_;
    double viscosity_;
    int p_radius_;
    IFFTBackend* fft_;
    WaveVectorGrid wave_vectors_;
    ComplexVectorField3D spectral_force_;
    ComplexVectorField3D spectral_velocity_;
    std::vector<double> spectral_tensor_xx_;
    std::vector<double> spectral_tensor_xy_;
    std::vector<double> spectral_tensor_xz_;
    std::vector<double> spectral_tensor_yy_;
    std::vector<double> spectral_tensor_yz_;
    std::vector<double> spectral_tensor_zz_;

    void apply_spectral_green_tensor();
    void precompute_energy_consistent_tensor();
};

}  // namespace permeability
