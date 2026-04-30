#pragma once

#include <vector>

#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/TrialForceField.hpp"
#include "core/Field3D.hpp"

namespace permeability {

class VelocityRecovery {
public:
    VelocityRecovery(const ForceSupport& support,
                     GreenOperator& green,
                     const BackgroundForce& background_force);

    [[nodiscard]] VectorField3D recover(const Real3& macroscopic_gradient,
                                        const std::vector<Scalar>& compact_solution) const;

    [[nodiscard]] Real3 average_over_domain(const VectorField3D& velocity) const;
    [[nodiscard]] Real3 average_over_support(const VectorField3D& velocity) const;

private:
    const ForceSupport* support_;
    GreenOperator* green_;
    const BackgroundForce* background_force_;
};

}  // namespace permeability
