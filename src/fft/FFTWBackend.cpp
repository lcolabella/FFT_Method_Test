#include "fft/FFTWBackend.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(PERMEABILITY_HAVE_FFTW3)
#include <fftw3.h>
#endif

namespace permeability {

#if defined(PERMEABILITY_HAVE_FFTW3)
namespace {

std::mutex& fftw_planner_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace
#endif

struct FFTWBackend::Impl {
    Grid3D grid{1, 1, 1};
    int threads = 1;
    bool initialized = false;
    double inv_n = 1.0;

#if defined(PERMEABILITY_HAVE_FFTW3)
    std::vector<std::complex<double>> in;
    std::vector<std::complex<double>> out;
    fftw_plan forward_plan = nullptr;
    fftw_plan inverse_plan = nullptr;
#endif

    ~Impl() {
#if defined(PERMEABILITY_HAVE_FFTW3)
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());
        if (forward_plan != nullptr) {
            fftw_destroy_plan(forward_plan);
            forward_plan = nullptr;
        }
        if (inverse_plan != nullptr) {
            fftw_destroy_plan(inverse_plan);
            inverse_plan = nullptr;
        }
#endif
    }
};

#if defined(PERMEABILITY_HAVE_FFTW3)
namespace {
std::mutex g_fftw_init_mutex;
bool g_fftw_threads_initialized = false;
}  // namespace
#endif

FFTWBackend::FFTWBackend() : impl_(std::make_unique<Impl>()) {}
FFTWBackend::~FFTWBackend() = default;
FFTWBackend::FFTWBackend(FFTWBackend&&) noexcept = default;
FFTWBackend& FFTWBackend::operator=(FFTWBackend&&) noexcept = default;

void FFTWBackend::initialize(const Grid3D& grid, int thread_count) {
    impl_->grid = grid;
    impl_->threads = std::max(1, thread_count);
    impl_->inv_n = 1.0 / static_cast<double>(grid.total_size());

#if defined(PERMEABILITY_HAVE_FFTW3)
    {
        std::lock_guard<std::mutex> lock(g_fftw_init_mutex);
        if (!g_fftw_threads_initialized) {
            if (fftw_init_threads() == 0) {
                throw std::runtime_error("fftw_init_threads failed");
            }
            g_fftw_threads_initialized = true;
        }
    }

    const std::size_t n = grid.total_size();
    impl_->in.assign(n, std::complex<double>(0.0, 0.0));
    impl_->out.assign(n, std::complex<double>(0.0, 0.0));

    {
        // FFTW planning API is not thread-safe; serialize plan create/destroy.
        std::lock_guard<std::mutex> lock(fftw_planner_mutex());

        fftw_plan_with_nthreads(impl_->threads);

        if (impl_->forward_plan != nullptr) {
            fftw_destroy_plan(impl_->forward_plan);
            impl_->forward_plan = nullptr;
        }
        if (impl_->inverse_plan != nullptr) {
            fftw_destroy_plan(impl_->inverse_plan);
            impl_->inverse_plan = nullptr;
        }

        // Plans are created once, outside apply paths.
        impl_->forward_plan = fftw_plan_dft_3d(
            static_cast<int>(grid.nz()),
            static_cast<int>(grid.ny()),
            static_cast<int>(grid.nx()),
            reinterpret_cast<fftw_complex*>(impl_->in.data()),
            reinterpret_cast<fftw_complex*>(impl_->out.data()),
            FFTW_FORWARD,
            FFTW_MEASURE);

        impl_->inverse_plan = fftw_plan_dft_3d(
            static_cast<int>(grid.nz()),
            static_cast<int>(grid.ny()),
            static_cast<int>(grid.nx()),
            reinterpret_cast<fftw_complex*>(impl_->in.data()),
            reinterpret_cast<fftw_complex*>(impl_->out.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE);
    }

    if (impl_->forward_plan == nullptr || impl_->inverse_plan == nullptr) {
        throw std::runtime_error("FFTW plan creation failed");
    }
#endif

    impl_->initialized = true;
}

void FFTWBackend::forward(const VectorField3D& real_in,
                          ComplexVectorField3D& complex_out) {
    if (!impl_->initialized) {
        throw std::runtime_error("FFTWBackend must be initialized before forward transform");
    }

    if (real_in.size() != complex_out.size()) {
        throw std::invalid_argument("FFTWBackend::forward grid size mismatch");
    }

#if defined(PERMEABILITY_HAVE_FFTW3)
    const std::size_t n = real_in.size();
    auto run_component = [&](const ScalarField3D& src, Array3D<std::complex<double>>& dst) {
#if defined(PERMEABILITY_USE_OPENMP)
#pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            impl_->in[i] = {src[i], 0.0};
        }
        fftw_execute(impl_->forward_plan);
#if defined(PERMEABILITY_USE_OPENMP)
#pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = impl_->out[i];
        }
    };

    run_component(real_in.x(), complex_out.x());
    run_component(real_in.y(), complex_out.y());
    run_component(real_in.z(), complex_out.z());
#else
    for (std::size_t i = 0; i < real_in.size(); ++i) {
        complex_out.x()[i] = {real_in.x()[i], 0.0};
        complex_out.y()[i] = {real_in.y()[i], 0.0};
        complex_out.z()[i] = {real_in.z()[i], 0.0};
    }
#endif
}

void FFTWBackend::inverse(const ComplexVectorField3D& complex_in,
                          VectorField3D& real_out) {
    if (!impl_->initialized) {
        throw std::runtime_error("FFTWBackend must be initialized before inverse transform");
    }

    if (real_out.size() != complex_in.size()) {
        throw std::invalid_argument("FFTWBackend::inverse grid size mismatch");
    }

#if defined(PERMEABILITY_HAVE_FFTW3)
    const std::size_t n = real_out.size();
    auto run_component = [&](const Array3D<std::complex<double>>& src, ScalarField3D& dst) {
#if defined(PERMEABILITY_USE_OPENMP)
#pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            impl_->in[i] = src[i];
        }
        fftw_execute(impl_->inverse_plan);
#if defined(PERMEABILITY_USE_OPENMP)
#pragma omp parallel for
#endif
        for (std::size_t i = 0; i < n; ++i) {
            // FFTW does not normalize inverse transforms; apply explicit 1/N.
            dst[i] = impl_->out[i].real() * impl_->inv_n;
        }
    };

    run_component(complex_in.x(), real_out.x());
    run_component(complex_in.y(), real_out.y());
    run_component(complex_in.z(), real_out.z());
#else
    for (std::size_t i = 0; i < real_out.size(); ++i) {
        real_out.x()[i] = complex_in.x()[i].real();
        real_out.y()[i] = complex_in.y()[i].real();
        real_out.z()[i] = complex_in.z()[i].real();
    }
#endif
}

double FFTWBackend::normalization_factor() const noexcept {
    return impl_->inv_n;
}

const Grid3D& FFTWBackend::grid() const noexcept {
    return impl_->grid;
}

}  // namespace permeability
