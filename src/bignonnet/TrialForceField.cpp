#include "bignonnet/TrialForceField.hpp"

#include <cmath>
#include <stdexcept>

namespace permeability {

namespace {
void check_comp(std::size_t comp) {
    if (comp >= 3) {
        throw std::out_of_range("TrialForceField component must be in [0,2]");
    }
}
}  // namespace

TrialForceField::TrialForceField(std::size_t num_active_voxels)
    : nb_(num_active_voxels), values_(3 * num_active_voxels, 0.0) {}

Scalar& TrialForceField::x(std::size_t local) { return (*this)(local, 0); }
Scalar& TrialForceField::y(std::size_t local) { return (*this)(local, 1); }
Scalar& TrialForceField::z(std::size_t local) { return (*this)(local, 2); }
const Scalar& TrialForceField::x(std::size_t local) const { return (*this)(local, 0); }
const Scalar& TrialForceField::y(std::size_t local) const { return (*this)(local, 1); }
const Scalar& TrialForceField::z(std::size_t local) const { return (*this)(local, 2); }

Scalar& TrialForceField::operator()(std::size_t local, std::size_t comp) {
    check_comp(comp);
    if (local >= nb_) {
        throw std::out_of_range("TrialForceField local index out of range");
    }
    return values_[3 * local + comp];
}

const Scalar& TrialForceField::operator()(std::size_t local, std::size_t comp) const {
    check_comp(comp);
    if (local >= nb_) {
        throw std::out_of_range("TrialForceField local index out of range");
    }
    return values_[3 * local + comp];
}

Real3 TrialForceField::mean_vector() const {
    Real3 m{0.0, 0.0, 0.0};
    if (nb_ == 0) {
        return m;
    }

    Scalar sx = Scalar(0);
    Scalar sy = Scalar(0);
    Scalar sz = Scalar(0);
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for reduction(+:sx,sy,sz)
#endif
    for (std::size_t n = 0; n < nb_; ++n) {
        sx += values_[3 * n + 0];
        sy += values_[3 * n + 1];
        sz += values_[3 * n + 2];
    }
    const Scalar inv = Scalar(1) / static_cast<Scalar>(nb_);
    m[0] = sx * inv;
    m[1] = sy * inv;
    m[2] = sz * inv;
    return m;
}

void TrialForceField::subtract_mean() {
    if (nb_ == 0) {
        return;
    }

    const Real3 m = mean_vector();
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t n = 0; n < nb_; ++n) {
        values_[3 * n + 0] -= m[0];
        values_[3 * n + 1] -= m[1];
        values_[3 * n + 2] -= m[2];
    }
}

Scalar TrialForceField::norm2() const {
    Scalar s = Scalar(0);
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (std::size_t i = 0; i < values_.size(); ++i) {
        s += values_[i] * values_[i];
    }
    return std::sqrt(s);
}

void TrialForceField::fill_zero() {
    std::fill(values_.begin(), values_.end(), 0.0);
}

void TrialForceField::scatter_to_full(VectorField3D& out_full, const ForceSupport& support) const {
    if (support.num_active_voxels() != nb_) {
        throw std::invalid_argument("TrialForceField::scatter_to_full support size mismatch");
    }
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t n = 0; n < nb_; ++n) {
        const std::size_t voxel = support.global_index(n);
        out_full.x()[voxel] = values_[3 * n + 0];
        out_full.y()[voxel] = values_[3 * n + 1];
        out_full.z()[voxel] = values_[3 * n + 2];
    }
}

void TrialForceField::gather_from_full(const VectorField3D& in_full, const ForceSupport& support) {
    if (support.num_active_voxels() != nb_) {
        throw std::invalid_argument("TrialForceField::gather_from_full support size mismatch");
    }
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t n = 0; n < nb_; ++n) {
        const std::size_t voxel = support.global_index(n);
        values_[3 * n + 0] = in_full.x()[voxel];
        values_[3 * n + 1] = in_full.y()[voxel];
        values_[3 * n + 2] = in_full.z()[voxel];
    }
}

}  // namespace permeability
