#pragma once

#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/TrialForceField.hpp"

namespace permeability {

class VariationalRHS {
public:
    VariationalRHS(const ForceSupport& support,
                   GreenOperator& green,
                   const BackgroundForce& background_force);

    [[nodiscard]] std::vector<Scalar> build(const Real3& macroscopic_gradient) const;

private:
    const ForceSupport* support_;
    GreenOperator* green_;
    const BackgroundForce* background_force_;
    mutable VectorField3D force_buffer_;
    mutable VectorField3D response_buffer_;
    mutable TrialForceField compact_buffer_;
};

}  // namespace permeability
