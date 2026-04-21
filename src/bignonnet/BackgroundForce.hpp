#pragma once

#include "bignonnet/ForceSupport.hpp"
#include "core/Field3D.hpp"
#include "core/Types.hpp"
#include "geometry/BinaryMedium.hpp"

namespace permeability {

class BackgroundForce {
public:
    BackgroundForce(const BinaryMedium& medium, const ForceSupport& support);

    [[nodiscard]] VectorField3D build(const Real3& macroscopic_gradient) const;

private:
    const BinaryMedium* medium_;
    const ForceSupport* support_;
};

}  // namespace permeability
