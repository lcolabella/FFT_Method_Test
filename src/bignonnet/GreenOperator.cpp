#include "bignonnet/GreenOperator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace permeability {

namespace {

constexpr double kPi = 3.14159265358979323846;

Scalar sinc(Scalar x) {
    const Scalar ax = std::abs(x);
    if (ax < Scalar(1e-12)) {
        return Scalar(1);
    }
    return std::sin(x) / x;
}

Scalar q_component(long long m, Scalar period_length) {
    return (Scalar(2) * Scalar(kPi) * static_cast<Scalar>(m)) / period_length;
}

}  // namespace

GreenOperator::GreenOperator(const Grid3D& grid,
                             GreenDiscretization discretization,
                             Scalar viscosity,
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
      spectral_tensor_xx_(grid.total_size(), Scalar(0)),
      spectral_tensor_xy_(grid.total_size(), Scalar(0)),
      spectral_tensor_xz_(grid.total_size(), Scalar(0)),
      spectral_tensor_yy_(grid.total_size(), Scalar(0)),
      spectral_tensor_yz_(grid.total_size(), Scalar(0)),
      spectral_tensor_zz_(grid.total_size(), Scalar(0)) {
    if (viscosity_ <= Scalar(0)) {
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
    if (fft_->apply_spectral_green_tensor(
            spectral_tensor_xx_,
            spectral_tensor_xy_,
            spectral_tensor_xz_,
            spectral_tensor_yy_,
            spectral_tensor_yz_,
            spectral_tensor_zz_,
            spectral_force_,
            spectral_velocity_)) {
        return;
    }

    const std::size_t n = spectral_velocity_.size();
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t idx = 0; idx < n; ++idx) {
        const ScalarComplex fx = spectral_force_.x()[idx];
        const ScalarComplex fy = spectral_force_.y()[idx];
        const ScalarComplex fz = spectral_force_.z()[idx];

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
            spectral_tensor_xx_[idx] = Scalar(0);
            spectral_tensor_xy_[idx] = Scalar(0);
            spectral_tensor_xz_[idx] = Scalar(0);
            spectral_tensor_yy_[idx] = Scalar(0);
            spectral_tensor_yz_[idx] = Scalar(0);
            spectral_tensor_zz_[idx] = Scalar(0);
            continue;
        }

        Scalar gxx = Scalar(0);
        Scalar gxy = Scalar(0);
        Scalar gxz = Scalar(0);
        Scalar gyy = Scalar(0);
        Scalar gyz = Scalar(0);
        Scalar gzz = Scalar(0);

        for (int px = -p_radius_; px <= p_radius_; ++px) {
            for (int py = -p_radius_; py <= p_radius_; ++py) {
                for (int pz = -p_radius_; pz <= p_radius_; ++pz) {
                    const long long mx = static_cast<long long>(ki) + static_cast<long long>(px) * static_cast<long long>(nx);
                    const long long my = static_cast<long long>(kj) + static_cast<long long>(py) * static_cast<long long>(ny);
                    const long long mz = static_cast<long long>(kk) + static_cast<long long>(pz) * static_cast<long long>(nz);

                    const Scalar sx = sinc(Scalar(kPi) * static_cast<Scalar>(mx) / static_cast<Scalar>(nx));
                    const Scalar sy = sinc(Scalar(kPi) * static_cast<Scalar>(my) / static_cast<Scalar>(ny));
                    const Scalar sz = sinc(Scalar(kPi) * static_cast<Scalar>(mz) / static_cast<Scalar>(nz));
                    const Scalar weight = (sx * sx) * (sy * sy) * (sz * sz);
                    if (weight == Scalar(0)) {
                        continue;
                    }

                    const Scalar qx = q_component(mx, grid_.lx());
                    const Scalar qy = q_component(my, grid_.ly());
                    const Scalar qz = q_component(mz, grid_.lz());
                    const Scalar q2 = qx * qx + qy * qy + qz * qz;
                    if (q2 <= std::numeric_limits<Scalar>::epsilon()) {
                        continue;
                    }

                    const Scalar inv_mu_q2 = Scalar(1) / (viscosity_ * q2);
                    const Scalar inv_q2 = Scalar(1) / q2;

                    const Scalar p11 = Scalar(1) - qx * qx * inv_q2;
                    const Scalar p12 = -qx * qy * inv_q2;
                    const Scalar p13 = -qx * qz * inv_q2;
                    const Scalar p22 = Scalar(1) - qy * qy * inv_q2;
                    const Scalar p23 = -qy * qz * inv_q2;
                    const Scalar p33 = Scalar(1) - qz * qz * inv_q2;

                    const Scalar fac = weight * inv_mu_q2;
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
    const auto is_bad = [](Scalar v) {
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

std::array<Scalar, 6> GreenOperator::tensor_components(std::size_t idx) const {
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
