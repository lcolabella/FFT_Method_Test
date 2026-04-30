#include "bignonnet/VelocityRecovery.hpp"

#include <stdexcept>

namespace permeability {

VelocityRecovery::VelocityRecovery(const ForceSupport& support,
                                   GreenOperator& green,
                                   const BackgroundForce& background_force)
    : support_(&support), green_(&green), background_force_(&background_force) {}

VectorField3D VelocityRecovery::recover(const Real3& macroscopic_gradient,
                                        const std::vector<Scalar>& compact_solution) const {
    if (compact_solution.size() != 3 * support_->num_active_voxels()) {
        throw std::invalid_argument("VelocityRecovery::recover compact solution size mismatch");
    }

    TrialForceField fx(support_->num_active_voxels());
    fx.raw() = compact_solution;
    fx.subtract_mean();

    // Stage 1 Eq. (8): f = f0 + fx.
    VectorField3D f = background_force_->build(macroscopic_gradient);
    VectorField3D full_fx(support_->grid());
    full_fx.fill_zero();
    fx.scatter_to_full(full_fx, *support_);
    f.axpy(1.0, full_fx);

    // Green convolution U = G*f.
    VectorField3D u(support_->grid());
    green_->apply(f, u);

    // Section 9: choose V' = -average_B(U) so the support-average velocity is zero.
    const Real3 mean_b = u.mean_on_indices(support_->active_voxels());
    const Real3 v_prime{-mean_b[0], -mean_b[1], -mean_b[2]};

#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < u.size(); ++i) {
        u.x()[i] += v_prime[0];
        u.y()[i] += v_prime[1];
        u.z()[i] += v_prime[2];
    }

    return u;
}

Real3 VelocityRecovery::average_over_domain(const VectorField3D& velocity) const {
    return velocity.mean_over_all();
}

Real3 VelocityRecovery::average_over_support(const VectorField3D& velocity) const {
    return velocity.mean_on_indices(support_->active_voxels());
}

}  // namespace permeability
