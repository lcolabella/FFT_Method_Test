#include "app/SimulationConfig.hpp"

#include <stdexcept>
#include <string>

namespace permeability {

namespace {

bool is_flag(const char* arg, const char* flag) {
    return arg != nullptr && std::string(arg) == flag;
}

const char* require_value(int argc, char** argv, int& i, const char* flag_name) {
    if (i + 1 >= argc || argv[i + 1] == nullptr) {
        throw std::invalid_argument(std::string("Missing value for ") + flag_name);
    }
    ++i;
    return argv[i];
}

}  // namespace

SimulationConfig SimulationConfig::from_cli(int argc, char** argv) {
    SimulationConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (is_flag(argv[i], "--input")) {
            cfg.input_path = require_value(argc, argv, i, "--input");
        } else if (is_flag(argv[i], "--support")) {
            const std::string v = require_value(argc, argv, i, "--support");
            if (v == "interface") {
                cfg.support_mode = SupportMode::InterfaceOnly;
            } else if (v == "fullsolid") {
                cfg.support_mode = SupportMode::FullSolid;
            } else {
                throw std::invalid_argument("Invalid --support value. Use interface|fullsolid");
            }
        } else if (is_flag(argv[i], "--mu")) {
            cfg.mu = std::stod(require_value(argc, argv, i, "--mu"));
        } else if (is_flag(argv[i], "--voxel-size")) {
            cfg.voxel_size = std::stod(require_value(argc, argv, i, "--voxel-size"));
        } else if (is_flag(argv[i], "--tol")) {
            cfg.tolerance = std::stod(require_value(argc, argv, i, "--tol"));
        } else if (is_flag(argv[i], "--maxit")) {
            cfg.max_iterations = static_cast<std::size_t>(std::stoull(require_value(argc, argv, i, "--maxit")));
        } else if (is_flag(argv[i], "--threads")) {
            cfg.fft_threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (is_flag(argv[i], "--omp-threads")) {
            cfg.omp_threads = std::stoi(require_value(argc, argv, i, "--omp-threads"));
        } else if (is_flag(argv[i], "--p-radius")) {
            cfg.p_radius = std::stoi(require_value(argc, argv, i, "--p-radius"));
        } else if (is_flag(argv[i], "--gradient")) {
            const std::string v = require_value(argc, argv, i, "--gradient");
            if (v == "x") {
                cfg.gradient_mode = GradientDirection::X;
            } else if (v == "y") {
                cfg.gradient_mode = GradientDirection::Y;
            } else if (v == "z") {
                cfg.gradient_mode = GradientDirection::Z;
            } else if (v == "all") {
                cfg.gradient_mode = GradientDirection::All;
            } else {
                throw std::invalid_argument("Invalid --gradient value. Use x|y|z|all");
            }
        } else if (is_flag(argv[i], "--parallel-load-cases")) {
            cfg.parallel_load_cases = true;
        } else {
            throw std::invalid_argument(std::string("Unknown CLI option: ") + argv[i]);
        }
    }

    if (cfg.mu <= 0.0) {
        throw std::invalid_argument("--mu must be positive");
    }
    if (cfg.voxel_size <= 0.0) {
        throw std::invalid_argument("--voxel-size must be > 0");
    }
    if (cfg.tolerance <= 0.0) {
        throw std::invalid_argument("--tol must be positive");
    }
    if (cfg.max_iterations == 0) {
        throw std::invalid_argument("--maxit must be >= 1");
    }
    if (cfg.fft_threads <= 0) {
        throw std::invalid_argument("--threads must be >= 1");
    }
    if (cfg.omp_threads < 0) {
        throw std::invalid_argument("--omp-threads must be >= 0");
    }
    if (cfg.p_radius < 0) {
        throw std::invalid_argument("--p-radius must be >= 0");
    }

    return cfg;
}

}  // namespace permeability
