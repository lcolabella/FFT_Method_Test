#pragma once

#include <array>
#include <complex>
#include <cstddef>

namespace permeability {

using Index = std::size_t;

using Real3 = std::array<double, 3>;
using Complex3 = std::array<std::complex<double>, 3>;

enum class SupportMode {
    FullSolid,
    InterfaceOnly,
};

enum class GreenDiscretizationType {
    EnergyConsistent,
};

enum class GradientDirection {
    X,
    Y,
    Z,
    All,
};

}  // namespace permeability
