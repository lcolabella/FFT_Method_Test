#pragma once

#include <complex>
#include <vector>

namespace common::fft {

using Complex = std::complex<double>;

std::vector<Complex> fft1d(const std::vector<Complex>& input);
std::vector<double> magnitude(const std::vector<Complex>& spectrum);

} // namespace common::fft
