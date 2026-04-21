#pragma once

#include <vector>

#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/TrialForceField.hpp"
#include "solver/ILinearOperator.hpp"

namespace permeability {

class VariationalOperator final : public ILinearOperator {
public:
    VariationalOperator(const ForceSupport& support, GreenOperator& green);

    [[nodiscard]] std::size_t size() const override;
    void apply(const std::vector<double>& in, std::vector<double>& out) const override;

private:
    const ForceSupport* support_;
    GreenOperator* green_;
    mutable VectorField3D full_force_buffer_;
    mutable VectorField3D full_velocity_buffer_;
    mutable TrialForceField compact_in_;
    mutable TrialForceField compact_out_;
};

}  // namespace permeability
