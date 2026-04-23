#pragma once

#include <string>

#include "bignonnet/GreenDiscretization.hpp"
#include "core/Types.hpp"

namespace permeability {

struct SimulationConfig {
    std::string input_path{"data/sample_geometry.geom"};

    SupportMode support_mode{SupportMode::InterfaceOnly};
    GradientDirection gradient_mode{GradientDirection::All};
    GreenDiscretization green{};

    double mu{1.0};
    double voxel_size{1.0};
    double tolerance{1e-10};
    std::size_t max_iterations{400};
    int p_radius{2};

    int fft_threads{1};
    int omp_threads{0};
    bool parallel_load_cases{false};

    static SimulationConfig from_cli(int argc, char** argv);
};

}  // namespace permeability
