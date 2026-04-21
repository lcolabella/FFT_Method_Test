#pragma once

#include "core/Types.hpp"

namespace permeability {

struct GreenDiscretization {
    GreenDiscretizationType type{GreenDiscretizationType::EnergyConsistent};
};

}  // namespace permeability
