#pragma once

#include <array>
#include <string>

#include "bignonnet/GreenDiscretization.hpp"
#include "core/Types.hpp"

namespace permeability {

enum class AnalysisType {
    Fluid,
    Elastic,
};

enum class FluidAnalysisMode {
    Permeability,
    PressureGradient,
};

struct SimulationConfig {
    // Shared / top-level fields (also readable from [global] section of a .cfg file)
    std::string geometry_file{"data/sample_geometry.geom"};
    std::string material_file{};
    std::string output_file{"output.txt"};
    std::string log_file{};

    // High-level analysis selection (readable from [analysis] section)
    AnalysisType analysis{AnalysisType::Fluid};

    // Fluid common settings (readable from [fluid] section)
    FluidAnalysisMode fluid_mode{FluidAnalysisMode::Permeability};
    ComputeBackend compute_backend{ComputeBackend::CPU};
    std::array<Scalar, 3> pressure_gradient{Scalar(1), Scalar(0), Scalar(0)};

    // Bignonnet solver parameters (readable from [bignonnet] section)
    SupportMode support_mode{SupportMode::InterfaceOnly};
    GradientDirection gradient_mode{GradientDirection::All};
    GreenDiscretization green{};

    Scalar mu{Scalar(1)};
    bool   mu_explicit{false};  // true when mu was set via CLI or config file
    int    fluid_id{1};         // voxel value in geometry file that represents fluid
    bool   fluid_id_explicit{false};
    Scalar voxel_size{Scalar(1)};
    Scalar tolerance{Scalar(1e-10)};
    std::size_t max_iterations{400};    // maximum solver iterations (applies to all solvers)
    int p_radius{2};
    int progress_interval{0};           // 0 = disabled; log residual every N iterations (all solvers)

    int threads{0};      // OpenMP worker threads (0 = use OMP defaults)
    int fft_threads{0};  // FFTW threads (0 = inherit from threads)
    bool parallel_load_cases{false};
    bool write_velocity_fields{false};  // write microscopic velocity fields to file

    // Solver selection: "bignonnet" | "brinkman"
    std::string solver{"bignonnet"};

    // Brinkman solver parameters (readable from [brinkman] section)
    Scalar brinkman_solid_penalty{Scalar(1.0e8)};
    Scalar brinkman_relaxation{Scalar(0.9)};
    Scalar brinkman_tolerance{Scalar(1.0e-8)};
    Scalar brinkman_forcing_magnitude{Scalar(1)};
    bool brinkman_write_velocity_fields{false};  // legacy alias for write_velocity_fields
    bool brinkman_require_single_solid_id{true};

    // Load from a --config .cfg file, then apply any extra CLI overrides
    static SimulationConfig from_cli(int argc, char** argv);
};

}  // namespace permeability
