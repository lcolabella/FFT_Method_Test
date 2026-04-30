#include "brinkman/FFT.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(COMMON_USE_FFTW)
#include <fftw3.h>
#endif

#if defined(COMMON_USE_OPENMP)
#include <omp.h>
#endif

namespace common::fft {

std::vector<Complex> fft1d(const std::vector<Complex>& input) {
    const std::size_t n = input.size();
    if (n == 0) {
        throw std::runtime_error("FFT input size must be > 0");
    }

#if defined(COMMON_USE_FFTW)
    fftw_complex* in = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * n));
    fftw_complex* out = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * n));
    if (in == nullptr || out == nullptr) {
        if (in != nullptr) {
            fftw_free(in);
        }
        if (out != nullptr) {
            fftw_free(out);
        }
        throw std::runtime_error("FFTW allocation failed");
    }

    for (std::size_t i = 0; i < n; ++i) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }

    fftw_plan plan = fftw_plan_dft_1d(
        static_cast<int>(n),
        in,
        out,
        FFTW_FORWARD,
        FFTW_ESTIMATE);
    if (plan == nullptr) {
        fftw_free(in);
        fftw_free(out);
        throw std::runtime_error("Failed to create FFTW plan");
    }

    fftw_execute(plan);

    std::vector<Complex> spectrum(n);
    for (std::size_t i = 0; i < n; ++i) {
        spectrum[i] = Complex(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return spectrum;
#else
    throw std::runtime_error("FFTW backend is required but not enabled at build time");
#endif
}

std::vector<double> magnitude(const std::vector<Complex>& spectrum) {
    std::vector<double> magnitudes(spectrum.size(), 0.0);

#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(spectrum.size()); ++i) {
        magnitudes[static_cast<std::size_t>(i)] = std::abs(spectrum[static_cast<std::size_t>(i)]);
    }

    return magnitudes;
}

} // namespace common::fft
