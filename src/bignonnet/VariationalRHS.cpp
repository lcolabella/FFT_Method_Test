#include "bignonnet/VariationalRHS.hpp"

namespace permeability {

VariationalRHS::VariationalRHS(const ForceSupport& support,
                               GreenOperator& green,
                               const BackgroundForce& background_force)
    : support_(&support),
      green_(&green),
      background_force_(&background_force),
      force_buffer_(support.grid()),
      response_buffer_(support.grid()),
      compact_buffer_(support.num_active_voxels()) {}

std::vector<double> VariationalRHS::build(const Real3& macroscopic_gradient) const {
    // Stage 1 Eq. (11) RHS: b_n = -(G*f0)_n + average_B(G*f0).
    force_buffer_ = background_force_->build(macroscopic_gradient);
    green_->apply(force_buffer_, response_buffer_);

    compact_buffer_.gather_from_full(response_buffer_, *support_);
    compact_buffer_.subtract_mean();

    std::vector<double> b = compact_buffer_.raw();
    for (double& v : b) {
        v = -v;
    }
    return b;
}

}  // namespace permeability
