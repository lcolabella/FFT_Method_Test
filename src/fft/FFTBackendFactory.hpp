#pragma once

#include <memory>

#include "core/Types.hpp"
#include "fft/IFFTBackend.hpp"

namespace permeability {

std::unique_ptr<IFFTBackend> create_fft_backend(ComputeBackend backend);
const char* compute_backend_name(ComputeBackend backend) noexcept;

}  // namespace permeability
