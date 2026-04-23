#include "bignonnet/BackgroundForce.hpp"

#include <stdexcept>

namespace permeability {

BackgroundForce::BackgroundForce(const BinaryMedium& medium, const ForceSupport& support)
    : medium_(&medium), support_(&support) {
    if (medium_->grid().total_size() != support_->grid().total_size()) {
        throw std::invalid_argument("BackgroundForce requires medium and support on same grid");
    }
}

VectorField3D BackgroundForce::build(const Real3& macroscopic_gradient) const {
    const Grid3D& grid = medium_->grid();
    VectorField3D f0(grid);
    f0.fill_zero();

    std::size_t nf = 0;
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for reduction(+:nf)
#endif
    for (std::size_t idx = 0; idx < grid.total_size(); ++idx) {
        if (medium_->is_fluid(idx)) {
            ++nf;
        }
    }

    const std::size_t nb = support_->num_active_voxels();
    if (nb == 0) {
        return f0;
    }

    const double c_fluid = -1.0;
    const double c_support = -static_cast<double>(nf) / static_cast<double>(nb);

#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t idx = 0; idx < grid.total_size(); ++idx) {
        if (medium_->is_fluid(idx)) {
            f0.x()[idx] += c_fluid * macroscopic_gradient[0];
            f0.y()[idx] += c_fluid * macroscopic_gradient[1];
            f0.z()[idx] += c_fluid * macroscopic_gradient[2];
        }
    }

    const std::vector<std::size_t>& active = support_->active_voxels();
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t n = 0; n < active.size(); ++n) {
        const std::size_t voxel = active[n];
        f0.x()[voxel] += c_support * macroscopic_gradient[0];
        f0.y()[voxel] += c_support * macroscopic_gradient[1];
        f0.z()[voxel] += c_support * macroscopic_gradient[2];
    }

    return f0;
}

}  // namespace permeability
