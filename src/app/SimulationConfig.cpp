#include "app/SimulationConfig.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace permeability {

namespace {

// ── string helpers ────────────────────────────────────────────────────────────

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parseBool(const std::string& value) {
    const std::string lo = toLower(trim(value));
    return (lo == "1" || lo == "true" || lo == "yes" || lo == "on");
}

AnalysisType parseAnalysisType(const std::string& value) {
    const std::string lo = toLower(trim(value));
    if (lo == "fluid") return AnalysisType::Fluid;
    if (lo == "elastic") return AnalysisType::Elastic;
    throw std::invalid_argument("Invalid analysis type: " + value);
}

FluidAnalysisMode parseFluidAnalysisMode(const std::string& value) {
    const std::string lo = toLower(trim(value));
    if (lo == "permeability") return FluidAnalysisMode::Permeability;
    if (lo == "pressure_gradient") return FluidAnalysisMode::PressureGradient;
    throw std::invalid_argument("Invalid fluid analysis mode: " + value);
}

ComputeBackend parseComputeBackend(const std::string& value) {
    const std::string lo = toLower(trim(value));
    if (lo == "cpu") return ComputeBackend::CPU;
    if (lo == "gpu") return ComputeBackend::GPU;
    throw std::invalid_argument("Invalid compute backend: " + value + ". Use cpu|gpu");
}

GradientDirection parseGradientDirection(const std::string& value, const std::string& source_name) {
    const std::string lo = toLower(trim(value));
    if (lo == "x") return GradientDirection::X;
    if (lo == "y") return GradientDirection::Y;
    if (lo == "z") return GradientDirection::Z;
    if (lo == "all") return GradientDirection::All;
    throw std::invalid_argument("Invalid " + source_name + " value: " + value +
                                ". Use x|y|z|all");
}

std::array<Scalar, 3> parseVector3(const std::string& value, const std::string& source_name) {
    std::string normalized = value;
    for (char& c : normalized) {
        if (c == ',') c = ' ';
    }

    std::istringstream iss(normalized);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(iss >> x >> y >> z)) {
        throw std::invalid_argument("Invalid " + source_name + " value: " + value +
                                    ". Expected three numbers");
    }
    std::string extra;
    if (iss >> extra) {
        throw std::invalid_argument("Invalid " + source_name + " value: " + value +
                                    ". Expected exactly three numbers");
    }

    return {
        static_cast<Scalar>(x),
        static_cast<Scalar>(y),
        static_cast<Scalar>(z),
    };
}

// ── section-aware cfg file reader ─────────────────────────────────────────────
//
// Format:
//   key = value           (global section)
//   [section]
//   key = value           (keyed as "section.key")
//
// Comments start with # or ; and are stripped.

using SectionMap = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

SectionMap parseConfigFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    SectionMap sections;
    std::string section;  // empty = global
    std::string line;
    std::size_t lineNo = 0;

    while (std::getline(f, line)) {
        ++lineNo;
        const auto commentPos = line.find_first_of("#;");
        if (commentPos != std::string::npos) line = line.substr(0, commentPos);
        line = trim(line);
        if (line.empty()) continue;

        if (line.front() == '[') {
            // Section header
            const auto close = line.find(']');
            if (close == std::string::npos) {
                throw std::runtime_error("Config line " + std::to_string(lineNo) +
                                         ": missing closing ']'");
            }
            section = toLower(trim(line.substr(1, close - 1)));
            continue;
        }

        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Config line " + std::to_string(lineNo) +
                                     ": expected key = value");
        }
        const std::string key = toLower(trim(line.substr(0, eq)));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) {
            throw std::runtime_error("Config line " + std::to_string(lineNo) +
                                     ": empty key or value");
        }
        sections[section][key] = val;
    }
    return sections;
}

// ── helpers to apply sections to a config ─────────────────────────────────────

auto lookup(const SectionMap& m, const std::string& section, const std::string& key)
    -> const std::string* {
    const auto sit = m.find(section);
    if (sit == m.end()) return nullptr;
    const auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return nullptr;
    return &kit->second;
}

void applyGlobal(SimulationConfig& cfg, const SectionMap& m) {
    // Legacy/global keys for backward compatibility.
    if (const auto* v = lookup(m, "", "geometry_file"))    cfg.geometry_file  = *v;
    if (const auto* v = lookup(m, "", "material_file"))    cfg.material_file  = *v;
    if (const auto* v = lookup(m, "", "output_file"))      cfg.output_file    = *v;
    if (const auto* v = lookup(m, "", "log_file"))         cfg.log_file       = *v;
    if (const auto* v = lookup(m, "", "solver"))           cfg.solver         = *v;
    if (const auto* v = lookup(m, "", "compute_backend"))  cfg.compute_backend = parseComputeBackend(*v);
    if (const auto* v = lookup(m, "", "threads"))          cfg.threads        = std::stoi(*v);
    if (const auto* v = lookup(m, "", "max_iterations"))   cfg.max_iterations = static_cast<std::size_t>(std::stoull(*v));
    if (const auto* v = lookup(m, "", "progress_interval")) cfg.progress_interval = std::stoi(*v);
    if (const auto* v = lookup(m, "", "voxel_size"))        cfg.voxel_size = static_cast<Scalar>(std::stod(*v));
}

void applyAnalysis(SimulationConfig& cfg, const SectionMap& m) {
    const std::string sec = "analysis";
    if (const auto* v = lookup(m, sec, "type")) {
        cfg.analysis = parseAnalysisType(*v);
    }
}

void applyFluidCommon(SimulationConfig& cfg, const SectionMap& m) {
    const std::string sec = "fluid";
    if (const auto* v = lookup(m, sec, "mode")) {
        cfg.fluid_mode = parseFluidAnalysisMode(*v);
    }
    if (const auto* v = lookup(m, sec, "solver")) {
        cfg.solver = *v;
    }
    if (const auto* v = lookup(m, sec, "compute_backend")) {
        cfg.compute_backend = parseComputeBackend(*v);
    }
    if (const auto* v = lookup(m, sec, "pressure_gradient")) {
        cfg.pressure_gradient = parseVector3(*v, "fluid.pressure_gradient");
    }
    if (const auto* v = lookup(m, sec, "mu")) {
        cfg.mu = static_cast<Scalar>(std::stod(*v));
        cfg.mu_explicit = true;
    }
    if (const auto* v = lookup(m, sec, "fluid_id")) {
        cfg.fluid_id = std::stoi(*v);
        cfg.fluid_id_explicit = true;
    }
    if (const auto* v = lookup(m, sec, "voxel_size")) {
        cfg.voxel_size = static_cast<Scalar>(std::stod(*v));
    }
    if (const auto* v = lookup(m, sec, "tolerance")) {
        cfg.tolerance = static_cast<Scalar>(std::stod(*v));
        cfg.brinkman_tolerance = cfg.tolerance;
    }
    if (const auto* v = lookup(m, sec, "max_iterations")) {
        cfg.max_iterations = static_cast<std::size_t>(std::stoull(*v));
    }
    if (const auto* v = lookup(m, sec, "progress_interval")) {
        cfg.progress_interval = std::stoi(*v);
    }
    if (const auto* v = lookup(m, sec, "threads")) {
        cfg.threads = std::stoi(*v);
    }
    if (const auto* v = lookup(m, sec, "fft_threads")) {
        cfg.fft_threads = std::stoi(*v);
    }
    if (const auto* v = lookup(m, sec, "parallel_load_cases")) {
        cfg.parallel_load_cases = parseBool(*v);
    }
    if (const auto* v = lookup(m, sec, "write_velocity_fields")) {
        cfg.write_velocity_fields = parseBool(*v);
        cfg.brinkman_write_velocity_fields = cfg.write_velocity_fields;
    }
}

void applyBignonnetSection(SimulationConfig& cfg, const SectionMap& m, const std::string& sec) {
    if (const auto* v = lookup(m, sec, "mu")) {
        cfg.mu = static_cast<Scalar>(std::stod(*v));
        cfg.mu_explicit = true;
    }
    if (const auto* v = lookup(m, sec, "fluid_id")) {
        cfg.fluid_id = std::stoi(*v);
        cfg.fluid_id_explicit = true;
    }
    if (const auto* v = lookup(m, sec, "tolerance")) {
        cfg.tolerance = static_cast<Scalar>(std::stod(*v));
    }
    if (const auto* v = lookup(m, sec, "p_radius")) {
        cfg.p_radius = std::stoi(*v);
    }
    if (const auto* v = lookup(m, sec, "fft_threads")) {
        cfg.fft_threads = std::stoi(*v);
    }
    if (const auto* v = lookup(m, sec, "parallel_load_cases")) {
        cfg.parallel_load_cases = parseBool(*v);
    }
    if (const auto* v = lookup(m, sec, "support")) {
        const std::string lo = toLower(*v);
        if (lo == "interface")        cfg.support_mode = SupportMode::InterfaceOnly;
        else if (lo == "fullsolid")   cfg.support_mode = SupportMode::FullSolid;
        else throw std::invalid_argument("Invalid support value: " + *v);
    }
    if (const auto* v = lookup(m, sec, "gradient")) {
        cfg.gradient_mode = parseGradientDirection(*v, sec + ".gradient");
    }
}

void applyBignonnet(SimulationConfig& cfg, const SectionMap& m) {
    applyBignonnetSection(cfg, m, "bignonnet");
}

void applyFluidBignonnet(SimulationConfig& cfg, const SectionMap& m) {
    applyBignonnetSection(cfg, m, "fluid.bignonnet");
}

void applyBrinkmanSection(SimulationConfig& cfg, const SectionMap& m, const std::string& sec) {
    if (const auto* v = lookup(m, sec, "solid_penalty"))
        cfg.brinkman_solid_penalty = static_cast<Scalar>(std::stod(*v));
    if (const auto* v = lookup(m, sec, "relaxation"))
        cfg.brinkman_relaxation = static_cast<Scalar>(std::stod(*v));
    if (const auto* v = lookup(m, sec, "tolerance"))
        cfg.brinkman_tolerance = static_cast<Scalar>(std::stod(*v));
    if (const auto* v = lookup(m, sec, "forcing_magnitude"))
        cfg.brinkman_forcing_magnitude = static_cast<Scalar>(std::stod(*v));
    if (const auto* v = lookup(m, sec, "write_velocity_fields")) {
        cfg.brinkman_write_velocity_fields = parseBool(*v);
        cfg.write_velocity_fields = cfg.brinkman_write_velocity_fields;
    }
    if (const auto* v = lookup(m, sec, "require_single_solid_id")) {
        cfg.brinkman_require_single_solid_id = parseBool(*v);
    }
}

void applyBrinkman(SimulationConfig& cfg, const SectionMap& m) {
    applyBrinkmanSection(cfg, m, "brinkman");
}

void applyFluidBrinkman(SimulationConfig& cfg, const SectionMap& m) {
    applyBrinkmanSection(cfg, m, "fluid.brinkman");
}

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

    // First pass: check for --config and load the file
    for (int i = 1; i < argc; ++i) {
        if (is_flag(argv[i], "--config")) {
            const std::string path = require_value(argc, argv, i, "--config");
            const SectionMap m = parseConfigFile(path);

            // Legacy keys first, then new hierarchical keys override them.
            applyGlobal(cfg, m);
            applyBignonnet(cfg, m);
            applyBrinkman(cfg, m);
            applyAnalysis(cfg, m);
            applyFluidCommon(cfg, m);
            applyFluidBignonnet(cfg, m);
            applyFluidBrinkman(cfg, m);
        }
    }

    // Second pass: CLI overrides (everything except --config which was already consumed)
    for (int i = 1; i < argc; ++i) {
        if (is_flag(argv[i], "--config")) {
            ++i;  // skip the path value
        } else if (is_flag(argv[i], "--input")) {
            cfg.geometry_file = require_value(argc, argv, i, "--input");
        } else if (is_flag(argv[i], "--material-file")) {
            cfg.material_file = require_value(argc, argv, i, "--material-file");
        } else if (is_flag(argv[i], "--output")) {
            cfg.output_file = require_value(argc, argv, i, "--output");
        } else if (is_flag(argv[i], "--log")) {
            cfg.log_file = require_value(argc, argv, i, "--log");
        } else if (is_flag(argv[i], "--analysis")) {
            cfg.analysis = parseAnalysisType(require_value(argc, argv, i, "--analysis"));
        } else if (is_flag(argv[i], "--fluid-mode")) {
            cfg.fluid_mode = parseFluidAnalysisMode(require_value(argc, argv, i, "--fluid-mode"));
        } else if (is_flag(argv[i], "--fluid-solver")) {
            cfg.solver = require_value(argc, argv, i, "--fluid-solver");
        } else if (is_flag(argv[i], "--compute-backend")) {
            cfg.compute_backend = parseComputeBackend(require_value(argc, argv, i, "--compute-backend"));
        } else if (is_flag(argv[i], "--pressure-gradient")) {
            cfg.pressure_gradient =
                parseVector3(require_value(argc, argv, i, "--pressure-gradient"),
                             "--pressure-gradient");
        } else if (is_flag(argv[i], "--write-velocity-fields")) {
            cfg.write_velocity_fields = true;
            cfg.brinkman_write_velocity_fields = true;
        } else if (is_flag(argv[i], "--support")) {
            const std::string v = require_value(argc, argv, i, "--support");
            if (v == "interface")       cfg.support_mode = SupportMode::InterfaceOnly;
            else if (v == "fullsolid")  cfg.support_mode = SupportMode::FullSolid;
            else throw std::invalid_argument("Invalid --support value. Use interface|fullsolid");
        } else if (is_flag(argv[i], "--mu")) {
            cfg.mu = static_cast<Scalar>(std::stod(require_value(argc, argv, i, "--mu")));
            cfg.mu_explicit = true;
        } else if (is_flag(argv[i], "--fluid-id")) {
            cfg.fluid_id = std::stoi(require_value(argc, argv, i, "--fluid-id"));
            cfg.fluid_id_explicit = true;
        } else if (is_flag(argv[i], "--voxel-size")) {
            cfg.voxel_size = static_cast<Scalar>(std::stod(require_value(argc, argv, i, "--voxel-size")));
        } else if (is_flag(argv[i], "--tol")) {
            cfg.tolerance = static_cast<Scalar>(std::stod(require_value(argc, argv, i, "--tol")));
            cfg.brinkman_tolerance = cfg.tolerance;  // applies to both solvers
        } else if (is_flag(argv[i], "--maxit")) {
            cfg.max_iterations = static_cast<std::size_t>(
                std::stoull(require_value(argc, argv, i, "--maxit")));
        } else if (is_flag(argv[i], "--threads")) {
            cfg.threads = std::stoi(require_value(argc, argv, i, "--threads"));
        } else if (is_flag(argv[i], "--fft-threads")) {
            cfg.fft_threads = std::stoi(require_value(argc, argv, i, "--fft-threads"));
        } else if (is_flag(argv[i], "--p-radius")) {
            cfg.p_radius = std::stoi(require_value(argc, argv, i, "--p-radius"));
        } else if (is_flag(argv[i], "--gradient")) {
            cfg.gradient_mode =
                parseGradientDirection(require_value(argc, argv, i, "--gradient"), "--gradient");
        } else if (is_flag(argv[i], "--parallel-load-cases")) {
            cfg.parallel_load_cases = true;
        } else if (is_flag(argv[i], "--solver")) {
            cfg.solver = require_value(argc, argv, i, "--solver");
        } else if (is_flag(argv[i], "--solid-penalty")) {
            cfg.brinkman_solid_penalty = static_cast<Scalar>(std::stod(require_value(argc, argv, i, "--solid-penalty")));
        } else if (is_flag(argv[i], "--relaxation")) {
            cfg.brinkman_relaxation = static_cast<Scalar>(std::stod(require_value(argc, argv, i, "--relaxation")));
        } else if (is_flag(argv[i], "--forcing-magnitude")) {
            cfg.brinkman_forcing_magnitude = static_cast<Scalar>(std::stod(
                require_value(argc, argv, i, "--forcing-magnitude")));
        } else if (is_flag(argv[i], "--progress-interval")) {
            cfg.progress_interval = std::stoi(
                require_value(argc, argv, i, "--progress-interval"));
        } else {
            throw std::invalid_argument(std::string("Unknown CLI option: ") + argv[i]);
        }
    }

    // Validate
    if (cfg.fft_threads == 0) {
        cfg.fft_threads = cfg.threads;
    }
    if (cfg.fft_threads == 0) {
        cfg.fft_threads = 1;
    }

    if (cfg.analysis != AnalysisType::Fluid) {
        throw std::invalid_argument("analysis must be 'fluid' (other analyses are not implemented yet)");
    }
    if (cfg.solver != "bignonnet" && cfg.solver != "brinkman") {
        throw std::invalid_argument("solver must be 'bignonnet' or 'brinkman'");
    }
    if (cfg.fluid_mode == FluidAnalysisMode::PressureGradient) {
        const Scalar grad_mag2 =
            cfg.pressure_gradient[0] * cfg.pressure_gradient[0] +
            cfg.pressure_gradient[1] * cfg.pressure_gradient[1] +
            cfg.pressure_gradient[2] * cfg.pressure_gradient[2];
        if (std::abs(grad_mag2) <= Scalar(0)) {
            throw std::invalid_argument(
                "pressure_gradient must be non-zero when fluid mode is pressure_gradient");
        }
    }
    if (cfg.mu_explicit && cfg.mu <= 0.0) throw std::invalid_argument("mu must be positive");
    if (cfg.voxel_size <= 0.0)     throw std::invalid_argument("voxel_size must be > 0");
    if (cfg.tolerance <= 0.0)      throw std::invalid_argument("tolerance must be positive");
    if (cfg.max_iterations == 0)   throw std::invalid_argument("max_iterations must be >= 1");
    if (cfg.fft_threads <= 0)      throw std::invalid_argument("fft_threads must be >= 1");
    if (cfg.threads < 0)           throw std::invalid_argument("threads must be >= 0");
    if (cfg.p_radius < 0)          throw std::invalid_argument("p_radius must be >= 0");

    return cfg;
}

}  // namespace permeability



