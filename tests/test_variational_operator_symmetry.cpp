#include <cassert>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenDiscretization.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalOperator.hpp"
#include "core/Types.hpp"
#include "fft/FFTWBackend.hpp"
#include "geometry/BinaryMedium.hpp"

int main() {
    permeability::Grid3D grid(4, 4, 4);
    std::vector<unsigned char> mask(grid.total_size(), 1);
    mask[grid.flat_index(1, 1, 1)] = 0;
    mask[grid.flat_index(2, 2, 2)] = 0;
    mask[grid.flat_index(1, 2, 2)] = 0;

    permeability::BinaryMedium medium(grid, std::move(mask));
    permeability::ForceSupport support(medium, permeability::SupportMode::InterfaceOnly);
    assert(support.num_active_voxels() > 0);

    permeability::FFTWBackend fft;
    permeability::GreenDiscretization disc{permeability::GreenDiscretizationType::EnergyConsistent};
    permeability::GreenOperator green(grid, disc, 1.0, fft, 1, 2);
    permeability::VariationalOperator op(support, green);

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<permeability::Scalar> dist(-1.0, 1.0);

    std::vector<permeability::Scalar> x(op.size(), permeability::Scalar(0));
    std::vector<permeability::Scalar> y(op.size(), permeability::Scalar(0));
    for (permeability::Scalar& v : x) {
        v = dist(rng);
    }
    for (permeability::Scalar& v : y) {
        v = dist(rng);
    }

    std::vector<permeability::Scalar> ax(op.size(), permeability::Scalar(0));
    std::vector<permeability::Scalar> ay(op.size(), permeability::Scalar(0));
    op.apply(x, ax);
    op.apply(y, ay);

    double x_ay = 0.0;
    double ax_y = 0.0;
    double nx = 0.0;
    double ny = 0.0;
    for (std::size_t i = 0; i < op.size(); ++i) {
        x_ay += x[i] * ay[i];
        ax_y += ax[i] * y[i];
        nx += x[i] * x[i];
        ny += y[i] * y[i];
    }

    const double denom = std::max({1.0, std::abs(x_ay), std::abs(ax_y), std::sqrt(nx * ny)});
    const double rel_gap = std::abs(x_ay - ax_y) / denom;
    assert(rel_gap < 1e-4);

    return 0;
}
