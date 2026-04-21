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

double& TrialForceField::x(std::size_t local) { return (*this)(local, 0); }
double& TrialForceField::y(std::size_t local) { return (*this)(local, 1); }
double& TrialForceField::z(std::size_t local) { return (*this)(local, 2); }
const double& TrialForceField::x(std::size_t local) const { return (*this)(local, 0); }
const double& TrialForceField::y(std::size_t local) const { return (*this)(local, 1); }
const double& TrialForceField::z(std::size_t local) const { return (*this)(local, 2); }

double& TrialForceField::operator()(std::size_t local, std::size_t comp) {
    check_comp(comp);
    if (local >= nb_) {
        throw std::out_of_range("TrialForceField local index out of range");
    }
    return values_[3 * local + comp];
}

const double& TrialForceField::operator()(std::size_t local, std::size_t comp) const {
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
    for (std::size_t n = 0; n < nb_; ++n) {
        m[0] += x(n);
        m[1] += y(n);
        m[2] += z(n);
    }
    const double inv = 1.0 / static_cast<double>(nb_);
    m[0] *= inv;
    m[1] *= inv;
    m[2] *= inv;
    return m;
}

void TrialForceField::subtract_mean() {
    if (nb_ == 0) {
        return;
    }

    const Real3 m = mean_vector();
    for (std::size_t n = 0; n < nb_; ++n) {
        x(n) -= m[0];
        y(n) -= m[1];
        z(n) -= m[2];
    }
}

double TrialForceField::norm2() const {
    double s = 0.0;
    for (double v : values_) {
        s += v * v;
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
    for (std::size_t n = 0; n < nb_; ++n) {
        const std::size_t voxel = support.global_index(n);
        out_full.x()[voxel] = x(n);
        out_full.y()[voxel] = y(n);
        out_full.z()[voxel] = z(n);
    }
}

void TrialForceField::gather_from_full(const VectorField3D& in_full, const ForceSupport& support) {
    if (support.num_active_voxels() != nb_) {
        throw std::invalid_argument("TrialForceField::gather_from_full support size mismatch");
    }
    for (std::size_t n = 0; n < nb_; ++n) {
        const std::size_t voxel = support.global_index(n);
        x(n) = in_full.x()[voxel];
        y(n) = in_full.y()[voxel];
        z(n) = in_full.z()[voxel];
    }
}

}  // namespace permeability
