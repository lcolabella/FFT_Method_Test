#include "bignonnet/GreenOperator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace permeability {

namespace {

constexpr double kPi = 3.14159265358979323846;

double sinc(double x) {
    const double ax = std::abs(x);
    if (ax < 1e-12) {
        return 1.0;
    }
    return std::sin(x) / x;
}

double q_component(long long m, double period_length) {
    return (2.0 * kPi * static_cast<double>(m)) / period_length;
}

}  // namespace

GreenOperator::GreenOperator(const Grid3D& grid,
                             GreenDiscretization discretization,
                             double viscosity,
                             IFFTBackend& fft_backend,
                             int fft_threads,
                             int p_radius)
    : grid_(grid),
      discretization_(discretization),
      viscosity_(viscosity),
      p_radius_(p_radius < 0 ? 0 : p_radius),
      fft_(&fft_backend),
      wave_vectors_(grid),
      spectral_force_(grid),
      spectral_velocity_(grid),
      spectral_tensor_xx_(grid.total_size(), 0.0),
      spectral_tensor_xy_(grid.total_size(), 0.0),
      spectral_tensor_xz_(grid.total_size(), 0.0),
      spectral_tensor_yy_(grid.total_size(), 0.0),
      spectral_tensor_yz_(grid.total_size(), 0.0),
      spectral_tensor_zz_(grid.total_size(), 0.0) {
    if (viscosity_ <= 0.0) {
        throw std::invalid_argument("GreenOperator viscosity must be positive");
    }
    fft_->initialize(grid_, fft_threads);
    precompute_energy_consistent_tensor();
}

void GreenOperator::apply(const VectorField3D& force, VectorField3D& velocity_like) {
    if (force.size() != grid_.total_size() || velocity_like.size() != grid_.total_size()) {
        throw std::invalid_argument("GreenOperator::apply grid mismatch");
    }

    fft_->forward(force, spectral_force_);
    spectral_velocity_ = spectral_force_;

    apply_spectral_green_tensor();

    fft_->inverse(spectral_velocity_, velocity_like);
}

void GreenOperator::apply_spectral_green_tensor() {
    const std::size_t n = spectral_velocity_.size();
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::complex<double> fx = spectral_force_.x()[idx];
        const std::complex<double> fy = spectral_force_.y()[idx];
        const std::complex<double> fz = spectral_force_.z()[idx];

        spectral_velocity_.x()[idx] = spectral_tensor_xx_[idx] * fx + spectral_tensor_xy_[idx] * fy + spectral_tensor_xz_[idx] * fz;
        spectral_velocity_.y()[idx] = spectral_tensor_xy_[idx] * fx + spectral_tensor_yy_[idx] * fy + spectral_tensor_yz_[idx] * fz;
        spectral_velocity_.z()[idx] = spectral_tensor_xz_[idx] * fx + spectral_tensor_yz_[idx] * fy + spectral_tensor_zz_[idx] * fz;
    }
}

void GreenOperator::precompute_energy_consistent_tensor() {
    // Stage 4: Eq. (14) energy-consistent Green discretization.
    // The p-sum is infinite in theory; in practice we truncate to |p_i| <= p_radius_.
    // TODO(Stage 5): benchmark/validate p-radius convergence for production runs.
    const std::size_t n = wave_vectors_.spectral_size();
    const std::size_t nx = grid_.nx();
    const std::size_t ny = grid_.ny();
    const std::size_t nz = grid_.nz();

#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::size_t idx = 0; idx < n; ++idx) {
        auto [ki, kj, kk] = grid_.unflatten(idx);

        if (ki == 0 && kj == 0 && kk == 0) {
            spectral_tensor_xx_[idx] = 0.0;
            spectral_tensor_xy_[idx] = 0.0;
            spectral_tensor_xz_[idx] = 0.0;
            spectral_tensor_yy_[idx] = 0.0;
            spectral_tensor_yz_[idx] = 0.0;
            spectral_tensor_zz_[idx] = 0.0;
            continue;
        }

        double gxx = 0.0;
        double gxy = 0.0;
        double gxz = 0.0;
        double gyy = 0.0;
        double gyz = 0.0;
        double gzz = 0.0;

        for (int px = -p_radius_; px <= p_radius_; ++px) {
            for (int py = -p_radius_; py <= p_radius_; ++py) {
                for (int pz = -p_radius_; pz <= p_radius_; ++pz) {
                    const long long mx = static_cast<long long>(ki) + static_cast<long long>(px) * static_cast<long long>(nx);
                    const long long my = static_cast<long long>(kj) + static_cast<long long>(py) * static_cast<long long>(ny);
                    const long long mz = static_cast<long long>(kk) + static_cast<long long>(pz) * static_cast<long long>(nz);

                    const double sx = sinc(kPi * static_cast<double>(mx) / static_cast<double>(nx));
                    const double sy = sinc(kPi * static_cast<double>(my) / static_cast<double>(ny));
                    const double sz = sinc(kPi * static_cast<double>(mz) / static_cast<double>(nz));
                    const double weight = (sx * sx) * (sy * sy) * (sz * sz);
                    if (weight == 0.0) {
                        continue;
                    }

                    const double qx = q_component(mx, grid_.lx());
                    const double qy = q_component(my, grid_.ly());
                    const double qz = q_component(mz, grid_.lz());
                    const double q2 = qx * qx + qy * qy + qz * qz;
                    if (q2 <= std::numeric_limits<double>::epsilon()) {
                        continue;
                    }

                    const double inv_mu_q2 = 1.0 / (viscosity_ * q2);
                    const double inv_q2 = 1.0 / q2;

                    const double p11 = 1.0 - qx * qx * inv_q2;
                    const double p12 = -qx * qy * inv_q2;
                    const double p13 = -qx * qz * inv_q2;
                    const double p22 = 1.0 - qy * qy * inv_q2;
                    const double p23 = -qy * qz * inv_q2;
                    const double p33 = 1.0 - qz * qz * inv_q2;

                    const double fac = weight * inv_mu_q2;
                    gxx += fac * p11;
                    gxy += fac * p12;
                    gxz += fac * p13;
                    gyy += fac * p22;
                    gyz += fac * p23;
                    gzz += fac * p33;
                }
            }
        }

        spectral_tensor_xx_[idx] = gxx;
        spectral_tensor_xy_[idx] = gxy;
        spectral_tensor_xz_[idx] = gxz;
        spectral_tensor_yy_[idx] = gyy;
        spectral_tensor_yz_[idx] = gyz;
        spectral_tensor_zz_[idx] = gzz;
    }
}

bool GreenOperator::has_nan_in_tensor() const noexcept {
    const auto is_bad = [](double v) {
        return !std::isfinite(v);
    };
    for (std::size_t i = 0; i < spectral_tensor_xx_.size(); ++i) {
        if (is_bad(spectral_tensor_xx_[i]) || is_bad(spectral_tensor_xy_[i]) ||
            is_bad(spectral_tensor_xz_[i]) || is_bad(spectral_tensor_yy_[i]) ||
            is_bad(spectral_tensor_yz_[i]) || is_bad(spectral_tensor_zz_[i])) {
            return true;
        }
    }
    return false;
}

std::array<double, 6> GreenOperator::tensor_components(std::size_t idx) const {
    if (idx >= spectral_tensor_xx_.size()) {
        throw std::out_of_range("GreenOperator::tensor_components index out of range");
    }
    return {spectral_tensor_xx_[idx],
            spectral_tensor_xy_[idx],
            spectral_tensor_xz_[idx],
            spectral_tensor_yy_[idx],
            spectral_tensor_yz_[idx],
            spectral_tensor_zz_[idx]};
}

}  // namespace permeability
