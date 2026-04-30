#include "fft/FFTBackendFactory.hpp"

#include <stdexcept>

#include "fft/FFTWBackend.hpp"

#if defined(PERMEABILITY_USE_SYCL)
#include "fft/SYCLFFTBackend.hpp"
#endif

namespace permeability {

std::unique_ptr<IFFTBackend> create_fft_backend(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::CPU:
#if defined(PERMEABILITY_HAVE_FFTW3F) || defined(PERMEABILITY_HAVE_FFTW3)
            return std::make_unique<FFTWBackend>();
#else
            throw std::runtime_error(
                "CPU backend requested, but this binary was built without FFTW support. "
                "Install FFTW and rebuild, or run with --compute-backend gpu.");
#endif
        case ComputeBackend::GPU:
#if defined(PERMEABILITY_USE_SYCL)
            return std::make_unique<SYCLFFTBackend>();
#else
            throw std::runtime_error(
                "GPU backend requested, but this binary was built without SYCL support. "
                "Reconfigure with -DFFT_METHOD_GPU_BACKEND=SYCL");
#endif
    }

    throw std::runtime_error("Unknown compute backend");
}

const char* compute_backend_name(ComputeBackend backend) noexcept {
    switch (backend) {
        case ComputeBackend::CPU:
            return "cpu";
        case ComputeBackend::GPU:
            return "gpu";
    }
    return "unknown";
}

}  // namespace permeability
