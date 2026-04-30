#include "bignonnet/VariationalOperator.hpp"

#include <stdexcept>

namespace permeability {

VariationalOperator::VariationalOperator(const ForceSupport& support, GreenOperator& green)
        : support_(&support),
            green_(&green),
            full_force_buffer_(support.grid()),
            full_velocity_buffer_(support.grid()),
            compact_in_(support.num_active_voxels()),
            compact_out_(support.num_active_voxels()) {}

std::size_t VariationalOperator::size() const {
    return support_->size();
}

void VariationalOperator::apply(const std::vector<Scalar>& in, std::vector<Scalar>& out) const {
    if (in.size() != size()) {
        throw std::invalid_argument("VariationalOperator::apply input size mismatch");
    }

    // Algorithm 1 step 1: subtract support-average from compact input.
    compact_in_.raw() = in;
    compact_in_.subtract_mean();

    // Algorithm 1 steps 2-8: scatter compact support vector to full grid.
    full_force_buffer_.fill_zero();
    compact_in_.scatter_to_full(full_force_buffer_, *support_);

    // Algorithm 1 steps 9-13: apply Green convolution via FFT.
    green_->apply(full_force_buffer_, full_velocity_buffer_);

    // Algorithm 1 steps 14-16: gather back to support.
    compact_out_.gather_from_full(full_velocity_buffer_, *support_);

    // Algorithm 1 step 17: subtract support-average from output.
    compact_out_.subtract_mean();
    out = compact_out_.raw();
}

}  // namespace permeability
