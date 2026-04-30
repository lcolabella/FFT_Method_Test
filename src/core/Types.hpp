#pragma once

#include <array>
#include <complex>
#include <cstddef>

namespace permeability {

using Index = std::size_t;

// Phase 5: Precision is now configurable via CMake FFT_METHOD_PRECISION option
// Default (Phase 4 state): single precision
// Can be overridden with -DFFT_METHOD_PRECISION_DOUBLE for double precision
#if defined(FFT_METHOD_PRECISION_DOUBLE)
using Scalar        = double;
using ScalarComplex = std::complex<double>;
#else
// Default: single precision (float)
using Scalar        = float;
using ScalarComplex = std::complex<float>;
#endif

using Real3    = std::array<Scalar, 3>;
using Complex3 = std::array<ScalarComplex, 3>;

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

enum class ComputeBackend {
    CPU,
    GPU,
};

}  // namespace permeability
