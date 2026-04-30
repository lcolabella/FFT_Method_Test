#include "app/PermeabilityRunner.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef PERMEABILITY_USE_OPENMP
#include <omp.h>
#endif

#include "app/config/Config.hpp"
#include "bignonnet/BackgroundForce.hpp"
#include "bignonnet/ForceSupport.hpp"
#include "bignonnet/GreenOperator.hpp"
#include "bignonnet/VariationalOperator.hpp"
#include "bignonnet/VariationalRHS.hpp"
#include "bignonnet/VelocityRecovery.hpp"
#include "brinkman/FluidBrinkman.hpp"
#include "fft/FFTBackendFactory.hpp"
#include "io/VoxelIO.hpp"
#include "logging/Logger.hpp"
#include "materials/MaterialDatabase.hpp"
#include "post/PermeabilityTensor.hpp"
#include "solver/MinresSolver.hpp"

namespace permeability {

namespace {

const char* fluid_mode_name(FluidAnalysisMode mode) {
    return (mode == FluidAnalysisMode::Permeability) ? "permeability" : "pressure_gradient";
}

Real3 pressure_gradient_from_cfg(const SimulationConfig& cfg) {
    return Real3{cfg.pressure_gradient[0], cfg.pressure_gradient[1], cfg.pressure_gradient[2]};
}

std::vector<std::pair<std::size_t, Real3>> directions_from_mode(GradientDirection mode) {
    if (mode == GradientDirection::X) {
        return {{0, Real3{1.0, 0.0, 0.0}}};
    }
    if (mode == GradientDirection::Y) {
        return {{1, Real3{0.0, 1.0, 0.0}}};
    }
    if (mode == GradientDirection::Z) {
        return {{2, Real3{0.0, 0.0, 1.0}}};
    }
    return {
        {0, Real3{1.0, 0.0, 0.0}},
        {1, Real3{0.0, 1.0, 0.0}},
        {2, Real3{0.0, 0.0, 1.0}},
    };
}

struct DirectionSolveResult {
    std::size_t col{0};
    SolverResult solve_result{};
    Real3 avg_velocity{0.0, 0.0, 0.0};
    std::optional<VectorField3D> velocity_field{};
    double elapsed_seconds{0.0};
};

void write_raw(std::ofstream& out, const void* data, std::size_t bytes) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error("Binary output write failed");
    }
}

std::string base_output_stem(const std::string& base_output,
                             std::filesystem::path& parent_out) {
    const std::filesystem::path base = base_output.empty()
                                           ? std::filesystem::path("output.txt")
                                           : std::filesystem::path(base_output);
    parent_out = base.parent_path();
    return base.stem().empty() ? std::string("output") : base.stem().string();
}

std::string velocity_field_output_path(const std::string& base_output,
                                       const std::string& suffix) {
    std::filesystem::path parent;
    const std::string stem = base_output_stem(base_output, parent);
    const std::string file_name = suffix.empty() ? (stem + ".fvel") : (stem + "_" + suffix + ".fvel");
    return (parent.empty() ? std::filesystem::path(file_name) : (parent / file_name)).string();
}

std::string velocity_field_output_path_axis(const std::string& base_output,
                                            std::size_t axis) {
    return velocity_field_output_path(base_output, "axis" + std::to_string(axis + 1));
}

std::string text_output_path(const std::string& base_output, const std::string& suffix) {
    std::filesystem::path parent;
    const std::string stem = base_output_stem(base_output, parent);
    const std::string file_name = stem + "_" + suffix + ".txt";
    return (parent.empty() ? std::filesystem::path(file_name) : (parent / file_name)).string();
}

void write_velocity_field_binary(const std::string& output_file,
                                 const Grid3D& grid,
                                 const Real3& driving_vector,
                                 Scalar porosity,
                                 Scalar viscosity,
                                 const VectorField3D& velocity) {
    const std::filesystem::path p(output_file);
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream out(output_file, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open velocity output file: " + output_file);
    }

    const char magic[4] = {'F', 'V', 'E', 'L'};
    const std::uint8_t version = 1;
    const std::uint8_t reserved8 = 0;
    const std::uint16_t reserved16 = 0;
    const std::uint64_t nx = static_cast<std::uint64_t>(grid.nx());
    const std::uint64_t ny = static_cast<std::uint64_t>(grid.ny());
    const std::uint64_t nz = static_cast<std::uint64_t>(grid.nz());
    const double por = static_cast<double>(porosity);
    const double mu = static_cast<double>(viscosity);
    const double driving[3] = {
        static_cast<double>(driving_vector[0]),
        static_cast<double>(driving_vector[1]),
        static_cast<double>(driving_vector[2]),
    };

    write_raw(out, magic, sizeof(magic));
    write_raw(out, &version, sizeof(version));
    write_raw(out, &reserved8, sizeof(reserved8));
    write_raw(out, &reserved16, sizeof(reserved16));
    write_raw(out, &nx, sizeof(nx));
    write_raw(out, &ny, sizeof(ny));
    write_raw(out, &nz, sizeof(nz));
    write_raw(out, &por, sizeof(por));
    write_raw(out, &mu, sizeof(mu));
    write_raw(out, driving, sizeof(driving));

    for (std::size_t i = 0; i < velocity.size(); ++i) {
        const double xyz[3] = {
            static_cast<double>(velocity.x()[i]),
            static_cast<double>(velocity.y()[i]),
            static_cast<double>(velocity.z()[i]),
        };
        write_raw(out, xyz, sizeof(xyz));
    }
}

void log_load_case_completed(const DirectionSolveResult& result) {
    auto& log = common::Logger::instance();
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2)
        << "Completed load case axis=" << result.col
        << " iterations=" << result.solve_result.iterations
        << " relres=" << result.solve_result.final_rel_residual
        << " converged=" << (result.solve_result.converged ? "yes" : "no")
        << " elapsed=" << std::fixed << std::setprecision(2) << result.elapsed_seconds << " s";
    log.info(oss.str());
}

DirectionSolveResult solve_one_direction(const SimulationConfig& config,
                                         const BinaryMedium& medium,
                                         const ForceSupport& support,
                                         std::size_t col,
                                         const Real3& grad) {
    const auto t0 = std::chrono::steady_clock::now();

#ifdef PERMEABILITY_USE_OPENMP
    if (config.threads > 0) {
        omp_set_num_threads(config.threads);
    }
#endif

    std::unique_ptr<IFFTBackend> fft_backend = create_fft_backend(config.compute_backend);
    {
        const std::string dinfo = fft_backend->device_info();
        if (!dinfo.empty()) {
            common::Logger::instance().info("FFT backend device: " + dinfo);
        }
    }
    GreenOperator green(medium.grid(),
                        config.green,
                        config.mu,
                        *fft_backend,
                        config.fft_threads,
                        config.p_radius);
    BackgroundForce f0_builder(medium, support);
    VariationalRHS rhs_builder(support, green, f0_builder);
    VariationalOperator op(support, green);
    MinresSolver minres(config.tolerance, config.max_iterations);
    VelocityRecovery recovery(support, green, f0_builder);

    const std::vector<Scalar> b = rhs_builder.build(grad);
    MinresSolver::ProgressCallback progress_cb;
    if (config.progress_interval > 0) {
        struct ProgressState {
            std::chrono::steady_clock::time_point last_time;
            std::size_t last_iter{0};
        };
        auto state = std::make_shared<ProgressState>();
        state->last_time = std::chrono::steady_clock::now();
        progress_cb = [col, state](std::size_t iter, Scalar rel_res) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - state->last_time).count();
            const std::size_t iters_done = iter - state->last_iter;
            const double s_per_iter = (iters_done > 0) ? elapsed / static_cast<double>(iters_done) : 0.0;
            state->last_time = now;
            state->last_iter = iter;
            std::ostringstream oss;
            oss << "axis=" << col << " iter=" << iter
                << " relres=" << std::scientific << std::setprecision(4) << rel_res
                << std::fixed << std::setprecision(3) << " s/iter=" << s_per_iter;
            common::Logger::instance().info(oss.str());
        };
    }
    const SolverResult result = minres.solve(op, b, {}, config.progress_interval, progress_cb);
    VectorField3D velocity = recovery.recover(grad, result.solution);

    DirectionSolveResult out;
    out.col = col;
    out.solve_result = result;
    out.avg_velocity = recovery.average_over_domain(velocity);
    if (config.write_velocity_fields) {
        out.velocity_field = std::move(velocity);
    }
    out.elapsed_seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    return out;
}

}  // namespace

PermeabilityRunner::PermeabilityRunner(SimulationConfig config) : config_(std::move(config)) {}

// ── Brinkman solver path ───────────────────────────────────────────────────────

static int run_brinkman(SimulationConfig& cfg) {
    auto& log = common::Logger::instance();
    log.info("Execution route: fluid -> brinkman");

    // Load geometry preserving full material IDs
    log.info("Loading geometry: " + cfg.geometry_file);
    const common::Geometry geometry = VoxelIO::load_geometry(cfg.geometry_file);
    log.info("Grid: " + std::to_string(geometry.nx) + " x " +
             std::to_string(geometry.ny) + " x " + std::to_string(geometry.nz));

    // Load material database (may be empty if no file given)
    common::MaterialDatabase materialDb;
    if (!cfg.material_file.empty()) {
        log.info("Loading material database: " + cfg.material_file);
        materialDb = common::MaterialDatabase::read(cfg.material_file);
        log.info("Loaded material database: " + cfg.material_file);
    } else {
        // No material file: build a minimal DB from --mu / config mu value.
        // Brinkman solver needs at least viscosity for the fluid material.
        if (cfg.mu <= 0.0) {
            throw std::runtime_error(
                "Brinkman solver requires mu > 0; "
                "provide --mu or a material_file with viscosity");
        }
        common::MaterialProperties fluidProps;
        fluidProps.values["viscosity"] = cfg.mu;
        // Do NOT add phase tag here: if any entry has a phase tag, the Brinkman
        // solver requires ALL geometry material IDs to have one.  Without the tag
        // the solver falls back to using fluidMaterialId directly.
        materialDb.insert(static_cast<std::uint32_t>(cfg.fluid_id), std::move(fluidProps));
        log.info("Auto-built material DB: fluid_id=" + std::to_string(cfg.fluid_id) +
                 " viscosity=" + std::to_string(cfg.mu));
    }

    // Build AppConfig for the Brinkman solver
    common::AppConfig appCfg;
    appCfg.geometryFile  = cfg.geometry_file;
    appCfg.materialFile  = cfg.material_file;
    appCfg.outputFile    = cfg.output_file;
    appCfg.logFile       = cfg.log_file;
    appCfg.fluid.mode = (cfg.fluid_mode == FluidAnalysisMode::Permeability)
                            ? common::FluidRunMode::Permeability
                            : common::FluidRunMode::Response;
    appCfg.fluid.fluidMaterialId            = static_cast<std::uint32_t>(cfg.fluid_id);
    appCfg.fluid.voxelSize                  = cfg.voxel_size;
    appCfg.fluid.solidPenalty               = cfg.brinkman_solid_penalty;
    appCfg.fluid.relaxation                 = cfg.brinkman_relaxation;
    appCfg.fluid.tolerance                  = cfg.brinkman_tolerance;
    appCfg.fluid.maxIterations              = static_cast<int>(cfg.max_iterations);
    appCfg.fluid.forcingMagnitude           = cfg.brinkman_forcing_magnitude;
    appCfg.fluid.macroPressureGradient      = {
        static_cast<double>(cfg.pressure_gradient[0]),
        static_cast<double>(cfg.pressure_gradient[1]),
        static_cast<double>(cfg.pressure_gradient[2]),
    };
    appCfg.fluid.progressInterval           = cfg.progress_interval;
    appCfg.fluid.writeDirectionalVelocityFields = cfg.write_velocity_fields;
    appCfg.fluid.requireSingleSolidMaterialId   = cfg.brinkman_require_single_solid_id;

    log.info("Solver: brinkman");
    log.info(std::string("Fluid mode: ") + fluid_mode_name(cfg.fluid_mode));
    {
        std::ostringstream oss;
        oss << "Brinkman parameters: solid_penalty=" << cfg.brinkman_solid_penalty
            << " relaxation=" << cfg.brinkman_relaxation
            << " tolerance=" << cfg.brinkman_tolerance
            << " max_iterations=" << cfg.max_iterations;
        log.info(oss.str());
    }

    if (cfg.fluid_mode == FluidAnalysisMode::PressureGradient) {
        const std::array<double, 3> force{
            -static_cast<double>(cfg.pressure_gradient[0]),
            -static_cast<double>(cfg.pressure_gradient[1]),
            -static_cast<double>(cfg.pressure_gradient[2]),
        };

        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(4)
                << "Pressure-gradient response: grad_p=("
                << cfg.pressure_gradient[0] << ", "
                << cfg.pressure_gradient[1] << ", "
                << cfg.pressure_gradient[2] << ")"
                << " equivalent_force=(-grad_p)=("
                << force[0] << ", " << force[1] << ", " << force[2] << ")";
            log.info(oss.str());
        }

        const common::fluid::SolveResult result =
            common::fluid::solveBrinkman(appCfg, geometry, materialDb, force);

        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6)
                << "Macroscopic velocity U=("
                << result.superficialVelocity[0] << ", "
                << result.superficialVelocity[1] << ", "
                << result.superficialVelocity[2] << ")"
                << " iterations=" << result.diagnostics.iterations
                << " residual=" << result.diagnostics.residual
                << " converged=" << (result.diagnostics.converged ? "yes" : "no");
            log.info(oss.str());
        }

        if (cfg.write_velocity_fields) {
            const std::string path = velocity_field_output_path(cfg.output_file, "");
            common::fluid::writeVelocityFieldBinary(path, geometry, force, result);
            log.info("Velocity field written to " + path);
        }

        if (!cfg.output_file.empty()) {
            const std::string txt_path = text_output_path(cfg.output_file, "velocity");
            std::ofstream out(txt_path);
            if (!out) {
                throw std::runtime_error("Cannot open output file: " + txt_path);
            }
            out << "# FFT_Method fluid pressure-gradient response\n";
            out << "# solver = brinkman\n";
            out << "# geometry_file = " << cfg.geometry_file << "\n";
            out << "# grid = " << geometry.nx << " x " << geometry.ny << " x " << geometry.nz << "\n";
            out << "# voxel_size = " << cfg.voxel_size << "\n";
            out << "# porosity = " << result.porosity << "\n";
            out << "# viscosity = " << result.viscosity << "\n";
            out << "# pressure_gradient = "
                << cfg.pressure_gradient[0] << " "
                << cfg.pressure_gradient[1] << " "
                << cfg.pressure_gradient[2] << "\n";
            out << "# macroscopic_velocity = "
                << result.superficialVelocity[0] << " "
                << result.superficialVelocity[1] << " "
                << result.superficialVelocity[2] << "\n";
            out << "# iterations = " << result.diagnostics.iterations << "\n";
            out << "# residual = " << result.diagnostics.residual << "\n";
            out << "# converged = " << (result.diagnostics.converged ? "true" : "false") << "\n";
            log.info("Results written to " + txt_path);
        }

        return 0;
    }

    const common::fluid::PermeabilityResult result =
        common::fluid::computePermeabilityTensor(
            appCfg,
            geometry,
            materialDb,
            [&](std::size_t axis, const std::array<double, 3>& force, const common::fluid::SolveResult& solve) {
                if (!cfg.write_velocity_fields) {
                    return;
                }
                const std::string path = velocity_field_output_path_axis(cfg.output_file, axis);
                common::fluid::writeVelocityFieldBinary(path, geometry, force, solve);
                common::Logger::instance().info("Velocity field written to " + path);
            });

    // Build a formatted tensor string matching the bignonnet output style
    const Scalar scale = cfg.voxel_size * cfg.voxel_size;
    // (voxel_size^2 scaling is already applied inside computePermeabilityTensor)
    (void)scale;

    // Prefix width: "[YYYY-MM-DD HH:MM:SS] [INFO] " = 29 chars
    static constexpr std::size_t kLogPrefixWidth = 29;
    const std::string kIndent(kLogPrefixWidth, ' ');

    std::ostringstream tensor_str;
    tensor_str << std::scientific << std::setprecision(6);
    for (int row = 0; row < 3; ++row) {
        if (row > 0) tensor_str << "\n" << kIndent;
        for (int col = 0; col < 3; ++col) {
            tensor_str << result.tensor[static_cast<std::size_t>(row) * 3U +
                                        static_cast<std::size_t>(col)];
            if (col < 2) tensor_str << "  ";
        }
    }

    log.info("Permeability tensor K (Darcy sign, K > 0):\n" + kIndent + tensor_str.str());

    if (!cfg.output_file.empty()) {
        const std::string txt_path = text_output_path(cfg.output_file, "permeability");
        std::ofstream out(txt_path);
        if (!out) {
            throw std::runtime_error("Cannot open output file: " + txt_path);
        }
        out << "# FFT_Method permeability result\n";
        out << "# solver = brinkman\n";
        out << "# geometry_file = " << cfg.geometry_file << "\n";
        out << "# grid = " << geometry.nx << " x " << geometry.ny << " x " << geometry.nz << "\n";
        out << "# voxel_size = " << cfg.voxel_size << "\n";
        out << "# porosity = " << result.porosity << "\n";
        out << "# viscosity = " << result.viscosity << "\n";
        out << "#\n";
        out << "# Permeability tensor K (Darcy sign, K > 0):\n";
        out << tensor_str.str() << "\n";
        log.info("Results written to " + txt_path);
    }

    return 0;
}

// ── Bignonnet solver path ──────────────────────────────────────────────────────

int PermeabilityRunner::run() {
    const auto bignonnet_t0 = std::chrono::steady_clock::now();
    auto& log = common::Logger::instance();
    log.info("Entering PermeabilityRunner");

    if (config_.analysis != AnalysisType::Fluid) {
        throw std::runtime_error("Only fluid analysis is implemented in this version");
    }

#ifdef PERMEABILITY_USE_OPENMP
    if (!config_.parallel_load_cases && config_.threads > 0) {
        omp_set_num_threads(config_.threads);
    }
#endif

    if (config_.solver == "brinkman") {
        if (config_.compute_backend == ComputeBackend::GPU) {
            throw std::runtime_error(
                "compute_backend=gpu is currently supported only for solver=bignonnet");
        }
        log.info(std::string("Execution route: fluid -> brinkman -> ") + fluid_mode_name(config_.fluid_mode));
        return run_brinkman(config_);
    }

    log.info(std::string("Execution route: fluid -> bignonnet -> ") + fluid_mode_name(config_.fluid_mode));
    log.info(std::string("Execution backend route: ") + compute_backend_name(config_.compute_backend));
    log.info("Loading geometry: " + config_.geometry_file);
    BinaryMedium medium = VoxelIO::load(config_.geometry_file, config_.voxel_size,
                                             config_.fluid_id);

    // Resolve mu and fluid_id from material file if not explicitly set
    if (!config_.material_file.empty()) {
        const common::MaterialDatabase db = common::MaterialDatabase::read(config_.material_file);
        std::uint32_t fluid_mat_id = 0;
        Scalar fluid_viscosity = Scalar(0);
        for (std::uint32_t id = 1; id <= 65535; ++id) {
            if (!db.has(id)) continue;
            const auto& props = db.at(id);
            const auto it = props.values.find("viscosity");
            if (it != props.values.end() && it->second > 0.0) {
                fluid_mat_id = id;
                fluid_viscosity = static_cast<Scalar>(it->second);
                break;
            }
        }
        if (fluid_mat_id == 0) {
            throw std::runtime_error(
                "material_file '" + config_.material_file +
                "' has no material with viscosity > 0 (fluid phase)");
        }
        if (!config_.mu_explicit) {
            config_.mu = fluid_viscosity;
            std::ostringstream oss;
            oss << "mu set from material " << fluid_mat_id
                << " (viscosity = " << fluid_viscosity << ")";
            common::Logger::instance().info(oss.str());
        }
        if (!config_.fluid_id_explicit) {
            config_.fluid_id = static_cast<int>(fluid_mat_id);
            // Re-load now that fluid_id is known
            medium = VoxelIO::load(config_.geometry_file, config_.voxel_size,
                                        config_.fluid_id);
            std::ostringstream oss;
            oss << "fluid_id set from material file: " << config_.fluid_id;
            common::Logger::instance().info(oss.str());
        }
    }

    if (config_.mu <= 0.0) {
        throw std::runtime_error("mu must be positive; provide --mu or set viscosity in material_file");
    }

    ForceSupport support(medium, config_.support_mode);
    if (support.num_active_voxels() == 0) {
        throw std::runtime_error("Support set is empty; cannot run variational solve");
    }

    PermeabilityTensor k_tensor;

    const std::vector<std::pair<std::size_t, Real3>> gradients = directions_from_mode(config_.gradient_mode);

    {
        std::ostringstream oss;
        oss << "Solver: bignonnet";
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Fluid mode: " << fluid_mode_name(config_.fluid_mode);
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Compute backend: " << compute_backend_name(config_.compute_backend);
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Input geometry file: " << config_.geometry_file;
        log.info(oss.str());
    }
    if (!config_.material_file.empty()) {
        std::ostringstream oss;
        oss << "Input material file: " << config_.material_file;
        log.info(oss.str());
    } else {
        log.info("Input material file: (none)");
    }
    {
        std::ostringstream oss;
        oss << "Output file: " << config_.output_file;
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Fluid properties: fluid_id=" << config_.fluid_id
            << " mu=" << config_.mu;
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Stop criteria (MINRES): tolerance=" << config_.tolerance
            << " max_iterations=" << config_.max_iterations;
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Bignonnet parameters: support="
            << (config_.support_mode == SupportMode::InterfaceOnly ? "interface" : "fullsolid")
            << " gradient=";
        switch (config_.gradient_mode) {
            case GradientDirection::X: oss << "x"; break;
            case GradientDirection::Y: oss << "y"; break;
            case GradientDirection::Z: oss << "z"; break;
            case GradientDirection::All: oss << "all"; break;
        }
        oss << " p_radius=" << config_.p_radius;
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Threading: threads=" << config_.threads
            << " fft_threads=" << config_.fft_threads
            << " parallel_load_cases=" << (config_.parallel_load_cases ? "true" : "false");
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Grid: " << medium.grid().nx() << " x " << medium.grid().ny()
            << " x " << medium.grid().nz();
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Voxel size: " << config_.voxel_size;
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Porosity: " << medium.porosity();
        log.info(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Support voxels (NB): " << support.num_active_voxels();
        log.info(oss.str());
    }

    if (config_.fluid_mode == FluidAnalysisMode::PressureGradient) {
        const Real3 grad = pressure_gradient_from_cfg(config_);
        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(4)
                << "Starting pressure-gradient solve grad=("
                << grad[0] << ", " << grad[1] << ", " << grad[2] << ")";
            log.info(oss.str());
        }

        DirectionSolveResult result = solve_one_direction(config_, medium, support, 0, grad);
        log_load_case_completed(result);

        if (config_.write_velocity_fields && result.velocity_field.has_value()) {
            const std::string path = velocity_field_output_path(config_.output_file, "");
            write_velocity_field_binary(path,
                                        medium.grid(),
                                        grad,
                                        medium.porosity(),
                                        config_.mu,
                                        result.velocity_field.value());
            log.info("Velocity field written to " + path);
        }

        if (!result.solve_result.converged) {
            std::ostringstream oss;
            oss << "Solve did not converge for pressure-gradient response"
                << ", final rel residual=" << result.solve_result.final_rel_residual;
            log.warn(oss.str());
        }

        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(6)
                << "Macroscopic velocity U=("
                << result.avg_velocity[0] << ", "
                << result.avg_velocity[1] << ", "
                << result.avg_velocity[2] << ")";
            log.info(oss.str());
        }

        if (!config_.output_file.empty()) {
            const std::string txt_path = text_output_path(config_.output_file, "velocity");
            std::ofstream out(txt_path);
            if (!out) {
                throw std::runtime_error("Cannot open output file: " + txt_path);
            }
            out << "# FFT_Method fluid pressure-gradient response\n";
            out << "# solver = bignonnet\n";
            out << "# geometry_file = " << config_.geometry_file << "\n";
            out << "# grid = " << medium.grid().nx() << " x " << medium.grid().ny()
                << " x " << medium.grid().nz() << "\n";
            out << "# voxel_size = " << config_.voxel_size << "\n";
            out << "# porosity = " << medium.porosity() << "\n";
            out << "# mu = " << config_.mu << "\n";
            out << "# pressure_gradient = "
                << grad[0] << " " << grad[1] << " " << grad[2] << "\n";
            out << "# macroscopic_velocity = "
                << result.avg_velocity[0] << " "
                << result.avg_velocity[1] << " "
                << result.avg_velocity[2] << "\n";
            out << "# iterations = " << result.solve_result.iterations << "\n";
            out << "# final_rel_residual = " << result.solve_result.final_rel_residual << "\n";
            out << "# converged = " << (result.solve_result.converged ? "true" : "false") << "\n";
            log.info("Results written to " + txt_path);
        }

        {
            const double total_elapsed = std::chrono::duration<double>(
                                           std::chrono::steady_clock::now() - bignonnet_t0)
                                           .count();
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << "Bignonnet total elapsed=" << total_elapsed << " s";
            log.info(oss.str());
        }

        return 0;
    }

    std::vector<DirectionSolveResult> direction_results;
    direction_results.reserve(gradients.size());

    if (config_.parallel_load_cases && gradients.size() > 1) {
        struct Pending {
            std::future<DirectionSolveResult> future;
        };
        std::vector<Pending> pending;
        pending.reserve(gradients.size());

        for (const auto& item : gradients) {
            const std::size_t col = item.first;
            const Real3 grad = item.second;
            {
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(2)
                    << "Starting load case axis=" << col
                    << " grad=(" << grad[0] << ", " << grad[1] << ", " << grad[2] << ")";
                log.info(oss.str());
            }
            pending.push_back(Pending{std::async(std::launch::async, [&, col, grad]() {
                return solve_one_direction(config_, medium, support, col, grad);
            })});
        }

        while (!pending.empty()) {
            bool consumed_one = false;
            for (auto it = pending.begin(); it != pending.end(); ++it) {
                if (it->future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    DirectionSolveResult result = it->future.get();
                    log_load_case_completed(result);
                    direction_results.push_back(std::move(result));
                    pending.erase(it);
                    consumed_one = true;
                    break;
                }
            }
            if (!consumed_one) {
                std::this_thread::yield();
            }
        }
    } else {
        for (const auto& item : gradients) {
            const std::size_t col = item.first;
            const Real3 grad = item.second;
            {
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(2)
                    << "Starting load case axis=" << col
                    << " grad=(" << grad[0] << ", " << grad[1] << ", " << grad[2] << ")";
                log.info(oss.str());
            }
            DirectionSolveResult result = solve_one_direction(config_, medium, support, col, grad);
            log_load_case_completed(result);
            direction_results.push_back(std::move(result));
        }
    }

    for (const auto& item : gradients) {
        const std::size_t col = item.first;
        const auto it = std::find_if(direction_results.begin(), direction_results.end(),
                                     [col](const DirectionSolveResult& r) { return r.col == col; });
        if (it == direction_results.end()) {
            throw std::runtime_error("Internal error: missing direction result");
        }

        if (!it->solve_result.converged) {
            std::ostringstream oss;
            oss << "Solve did not converge for load case " << col
                << ", final rel residual=" << it->solve_result.final_rel_residual;
            log.warn(oss.str());
        }

        const Real3 colK{
            -config_.mu * it->avg_velocity[0],
            -config_.mu * it->avg_velocity[1],
            -config_.mu * it->avg_velocity[2],
        };
        k_tensor.set_column(col, colK);

        if (config_.write_velocity_fields && it->velocity_field.has_value()) {
            const std::string path = velocity_field_output_path_axis(config_.output_file, col);
            write_velocity_field_binary(path,
                                        medium.grid(),
                                        item.second,
                                        medium.porosity(),
                                        config_.mu,
                                        it->velocity_field.value());
            log.info("Velocity field written to " + path);
        }

        // Completion for each load case is logged immediately after solve return.
    }

    // Prefix width: "[YYYY-MM-DD HH:MM:SS] [INFO] " = 29 chars
    static constexpr std::size_t kLogPrefixWidth = 29;
    const std::string kIndent(kLogPrefixWidth, ' ');
    log.info("Permeability tensor K (Darcy sign, K > 0):\n" + kIndent + k_tensor.to_string(kIndent));

    if (!config_.output_file.empty()) {
        const std::string txt_path = text_output_path(config_.output_file, "permeability");
        std::ofstream out(txt_path);
        if (!out) {
            throw std::runtime_error("Cannot open output file: " + txt_path);
        }
        out << "# FFT_Method permeability result\n";
        out << "# geometry_file = " << config_.geometry_file << "\n";
        out << "# grid = " << medium.grid().nx() << " x " << medium.grid().ny()
            << " x " << medium.grid().nz() << "\n";
        out << "# voxel_size = " << config_.voxel_size << "\n";
        out << "# porosity = " << medium.porosity() << "\n";
        out << "# mu = " << config_.mu << "\n";
        out << "#\n";
        out << "# Permeability tensor K (Darcy sign, K > 0):\n";
        out << k_tensor.to_string() << "\n";
        {
            std::ostringstream oss;
            oss << "Results written to " << txt_path;
            log.info(oss.str());
        }
    }

    {
        const double total_elapsed = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - bignonnet_t0)
                                       .count();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "Bignonnet total elapsed=" << total_elapsed << " s";
        log.info(oss.str());
    }

    return 0;
}

}  // namespace permeability
