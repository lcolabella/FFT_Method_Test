#include <cassert>
#include <cmath>

#include "core/Field3D.hpp"
#include "fft/FFTWBackend.hpp"

int main() {
    permeability::Grid3D g(4, 4, 4);
    permeability::VectorField3D real(g);
    permeability::ComplexVectorField3D freq(g);
    permeability::VectorField3D back(g);

    for (std::size_t i = 0; i < real.size(); ++i) {
        real.x()[i] = static_cast<double>(i);
        real.y()[i] = static_cast<double>(2 * i);
        real.z()[i] = static_cast<double>(3 * i);
    }

    permeability::FFTWBackend fft;
    fft.initialize(g, 1);
    fft.forward(real, freq);
    fft.inverse(freq, back);

    for (std::size_t i = 0; i < real.size(); ++i) {
        assert(std::abs(back.x()[i] - real.x()[i]) < 1e-9);
        assert(std::abs(back.y()[i] - real.y()[i]) < 1e-9);
        assert(std::abs(back.z()[i] - real.z()[i]) < 1e-9);
    }

    return 0;
}
