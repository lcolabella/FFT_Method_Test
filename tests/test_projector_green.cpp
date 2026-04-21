#include <cassert>
#include <cmath>

#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "core/Field3D.hpp"
#include "fft/FFTWBackend.hpp"

int main() {
    permeability::Grid3D grid(4, 4, 4);
    permeability::FFTWBackend fft;
    permeability::GreenDiscretization disc{permeability::GreenDiscretizationType::EnergyConsistent};
    permeability::GreenOperator green(grid, disc, 1.0, fft, 1, 2);

    assert(!green.has_nan_in_tensor());

    const std::size_t zero_idx = grid.flat_index(0, 0, 0);
    const auto t0 = green.tensor_components(zero_idx);
    for (double v : t0) {
        assert(std::abs(v) < 1e-15);
    }

    for (std::size_t idx = 0; idx < green.spectral_size(); ++idx) {
        const auto t = green.tensor_components(idx);
        assert(std::isfinite(t[0]));
        assert(std::isfinite(t[1]));
        assert(std::isfinite(t[2]));
        assert(std::isfinite(t[3]));
        assert(std::isfinite(t[4]));
        assert(std::isfinite(t[5]));

        // Packed ordering: xx, xy, xz, yy, yz, zz. Symmetry requires ij=ji.
        assert(std::abs(t[1] - t[1]) < 1e-15);
        assert(std::abs(t[2] - t[2]) < 1e-15);
        assert(std::abs(t[4] - t[4]) < 1e-15);
    }

    permeability::VectorField3D in(grid);
    in.fill_zero();
    permeability::VectorField3D out(grid);
    out.fill(1.0);
    green.apply(in, out);

    for (std::size_t i = 0; i < out.size(); ++i) {
        assert(std::abs(out.x()[i]) < 1e-14);
        assert(std::abs(out.y()[i]) < 1e-14);
        assert(std::abs(out.z()[i]) < 1e-14);
    }

    return 0;
}
