#include "common/ConfigParser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

std::string trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::unordered_map<std::string, std::string> parseConfigFile(const std::string& filePath) {
    std::ifstream input(filePath);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open input file: " + filePath);
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    std::size_t lineNo = 0;

    while (std::getline(input, line)) {
        ++lineNo;
        const auto commentPos = line.find_first_of("#;");
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const auto eqPos = line.find('=');
        if (eqPos == std::string::npos) {
            throw std::runtime_error(
                "Invalid input file line " + std::to_string(lineNo) + ": expected key = value");
        }

        const std::string key = toLower(trim(line.substr(0, eqPos)));
        const std::string value = trim(line.substr(eqPos + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error(
                "Invalid input file line " + std::to_string(lineNo) + ": empty key or value");
        }

        values[key] = value;
    }

    return values;
}

int parseInt(const std::string& value, const std::string& fieldName) {
    try {
        size_t used = 0;
        const int parsed = std::stoi(value, &used);
        if (used != value.size()) {
            throw std::runtime_error("");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error("Invalid integer for " + fieldName + ": " + value);
    }
}

double parseDouble(const std::string& value, const std::string& fieldName) {
    try {
        size_t used = 0;
        const double parsed = std::stod(value, &used);
        if (used != value.size()) {
            throw std::runtime_error("");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error("Invalid floating-point value for " + fieldName + ": " + value);
    }
}

std::uint32_t parseUInt32(const std::string& value, const std::string& fieldName) {
    try {
        size_t used = 0;
        const unsigned long parsed = std::stoul(value, &used);
        if (used != value.size() || parsed > static_cast<unsigned long>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("");
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (...) {
        throw std::runtime_error("Invalid unsigned integer value for " + fieldName + ": " + value);
    }
}

std::array<double, 3> parseVector3(const std::string& value, const std::string& fieldName) {
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream in(normalized);

    std::array<double, 3> vec{};
    if (!(in >> vec[0] >> vec[1] >> vec[2])) {
        throw std::runtime_error("Invalid vector value for " + fieldName + ": expected 3 numbers");
    }

    double extra = 0.0;
    if (in >> extra) {
        throw std::runtime_error("Invalid vector value for " + fieldName + ": expected exactly 3 numbers");
    }
    return vec;
}

bool parseBool(const std::string& value, const std::string& fieldName) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
        return false;
    }
    throw std::runtime_error("Invalid boolean for " + fieldName + ": " + value);
}

} // namespace

namespace common {

std::string toString(AnalysisType analysis) {
    switch (analysis) {
    case AnalysisType::Elastic:
        return "elastic";
    case AnalysisType::Fluid:
        return "fluid";
    default:
        return "unknown";
    }
}

std::string toString(BackendType backend) {
    switch (backend) {
    case BackendType::CPU:
        return "cpu";
    case BackendType::CUDA:
        return "cuda";
    case BackendType::Intel:
        return "intel";
    default:
        return "cpu";
    }
}

std::string toString(GeometryFormat format) {
    switch (format) {
    case GeometryFormat::Auto:
        return "auto";
    case GeometryFormat::Text:
        return "text";
    case GeometryFormat::Binary:
        return "binary";
    default:
        return "auto";
    }
}

std::string toString(FluidRunMode mode) {
    switch (mode) {
    case FluidRunMode::Permeability:
        return "permeability";
    case FluidRunMode::Response:
        return "response";
    default:
        return "permeability";
    }
}

GeometryFormat geometryFormatFromString(const std::string& value) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "auto") {
        return GeometryFormat::Auto;
    }
    if (lowered == "text") {
        return GeometryFormat::Text;
    }
    if (lowered == "binary") {
        return GeometryFormat::Binary;
    }
    throw std::runtime_error("Unknown geometry format: " + value);
}

FluidRunMode fluidRunModeFromString(const std::string& value) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "permeability") {
        return FluidRunMode::Permeability;
    }
    if (lowered == "response") {
        return FluidRunMode::Response;
    }
    throw std::runtime_error("Unknown fluid mode: " + value);
}

AnalysisType analysisFromString(const std::string& value) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "elastic") {
        return AnalysisType::Elastic;
    }
    if (lowered == "fluid") {
        return AnalysisType::Fluid;
    }
    return AnalysisType::Unknown;
}

BackendType backendFromString(const std::string& value) {
    const std::string lowered = toLower(trim(value));
    if (lowered == "cpu") {
        return BackendType::CPU;
    }
    if (lowered == "cuda") {
        return BackendType::CUDA;
    }
    if (lowered == "intel") {
        return BackendType::Intel;
    }
    throw std::runtime_error("Unknown backend type: " + value);
}

std::vector<std::string> validateConfig(const AppConfig& config) {
    std::vector<std::string> errors;

    if (config.analysis == AnalysisType::Unknown) {
        errors.emplace_back("analysis must be set to elastic or fluid");
    }
    if (config.geometryFile.empty()) {
        errors.emplace_back("geometry_file must be provided");
    }
    if (config.materialFile.empty()) {
        errors.emplace_back("material_file must be provided");
    }
    if (config.outputFile.empty()) {
        errors.emplace_back("output_file must be provided");
    }
    if (config.logFile.empty()) {
        errors.emplace_back("log_file must be provided");
    }
    if (config.resources.cpus < 1) {
        errors.emplace_back("cpus must be >= 1");
    }
    if (config.resources.gpus < 0) {
        errors.emplace_back("gpus must be >= 0");
    }

    if (config.analysis == AnalysisType::Fluid) {
        if (config.fluid.solidPenalty <= 0.0) {
            errors.emplace_back("solid_penalty must be > 0");
        }
        if (config.fluid.forcingMagnitude <= 0.0) {
            errors.emplace_back("forcing_magnitude must be > 0");
        }
        if (config.fluid.voxelSize <= 0.0) {
            errors.emplace_back("voxel_size must be > 0");
        }
        if (config.fluid.maxIterations < 1) {
            errors.emplace_back("max_iterations must be >= 1");
        }
        if (config.fluid.progressInterval < 0) {
            errors.emplace_back("progress_interval must be >= 0");
        }
        if (config.fluid.tolerance <= 0.0) {
            errors.emplace_back("tolerance must be > 0");
        }
        if (config.fluid.relaxation <= 0.0 || config.fluid.relaxation > 1.0) {
            errors.emplace_back("relaxation must be in (0, 1]");
        }
        if (config.fluid.mode == FluidRunMode::Response) {
            const auto& g = config.fluid.macroPressureGradient;
            if (g[0] == 0.0 && g[1] == 0.0 && g[2] == 0.0) {
                errors.emplace_back("macro_pressure_gradient cannot be all zeros in response mode");
            }
        }
    }

    return errors;
}

CliOptions parseCommandLine(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + flag);
            }
            ++i;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h") {
            options.helpRequested = true;
        } else if (arg == "--input" || arg == "-i") {
            options.inputFile = nextValue(arg);
        } else if (arg == "--analysis") {
            options.analysis = nextValue(arg);
        } else if (arg == "--geometry") {
            options.geometryFile = nextValue(arg);
        } else if (arg == "--geometry-format") {
            options.geometryFormat = nextValue(arg);
        } else if (arg == "--materials") {
            options.materialFile = nextValue(arg);
        } else if (arg == "--output") {
            options.outputFile = nextValue(arg);
        } else if (arg == "--log") {
            options.logFile = nextValue(arg);
        } else if (arg == "--backend") {
            options.backend = nextValue(arg);
        } else if (arg == "--cpus") {
            options.cpus = parseInt(nextValue(arg), "cpus");
        } else if (arg == "--gpus") {
            options.gpus = parseInt(nextValue(arg), "gpus");
        } else if (arg == "--fluid-mode") {
            options.fluidMode = nextValue(arg);
        } else if (arg == "--fluid-material-id") {
            options.fluidMaterialId = parseUInt32(nextValue(arg), "fluid-material-id");
        } else if (arg == "--solid-penalty") {
            options.solidPenalty = parseDouble(nextValue(arg), "solid-penalty");
        } else if (arg == "--forcing") {
            options.forcingMagnitude = parseDouble(nextValue(arg), "forcing");
        } else if (arg == "--voxel-size") {
            options.voxelSize = parseDouble(nextValue(arg), "voxel-size");
        } else if (arg == "--macro-grad") {
            options.macroPressureGradient = parseVector3(nextValue(arg), "macro-grad");
        } else if (arg == "--max-iter") {
            options.maxIterations = parseInt(nextValue(arg), "max-iter");
        } else if (arg == "--progress-interval") {
            options.progressInterval = parseInt(nextValue(arg), "progress-interval");
        } else if (arg == "--write-directional-fields") {
            options.writeDirectionalVelocityFields = parseBool(nextValue(arg), "write-directional-fields");
        } else if (arg == "--require-single-solid-id") {
            options.requireSingleSolidMaterialId = parseBool(nextValue(arg), "require-single-solid-id");
        } else if (arg == "--tol") {
            options.tolerance = parseDouble(nextValue(arg), "tol");
        } else if (arg == "--omega") {
            options.relaxation = parseDouble(nextValue(arg), "omega");
        } else if (arg == "--field-output") {
            options.fieldOutputFile = nextValue(arg);
        } else if (arg == "--reference") {
            options.referenceFile = nextValue(arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return options;
}

AppConfig buildConfig(const CliOptions& cli) {
    AppConfig config;

    if (cli.inputFile.has_value()) {
        const auto fileValues = parseConfigFile(*cli.inputFile);

        auto readIfPresent = [&](const std::string& key) -> std::optional<std::string> {
            const auto it = fileValues.find(key);
            if (it == fileValues.end()) {
                return std::nullopt;
            }
            return it->second;
        };

        if (const auto value = readIfPresent("analysis"); value.has_value()) {
            config.analysis = analysisFromString(*value);
        }
        if (const auto value = readIfPresent("geometry_file"); value.has_value()) {
            config.geometryFile = *value;
        }
        if (const auto value = readIfPresent("geometry_format"); value.has_value()) {
            config.geometryFormat = geometryFormatFromString(*value);
        }
        if (const auto value = readIfPresent("material_file"); value.has_value()) {
            config.materialFile = *value;
        }
        if (const auto value = readIfPresent("output_file"); value.has_value()) {
            config.outputFile = *value;
        }
        if (const auto value = readIfPresent("log_file"); value.has_value()) {
            config.logFile = *value;
        }
        if (const auto value = readIfPresent("backend"); value.has_value()) {
            config.resources.backend = backendFromString(*value);
        }
        if (const auto value = readIfPresent("cpus"); value.has_value()) {
            config.resources.cpus = parseInt(*value, "cpus");
        }
        if (const auto value = readIfPresent("gpus"); value.has_value()) {
            config.resources.gpus = parseInt(*value, "gpus");
        }
        if (const auto value = readIfPresent("fluid_mode"); value.has_value()) {
            config.fluid.mode = fluidRunModeFromString(*value);
        }
        if (const auto value = readIfPresent("fluid_material_id"); value.has_value()) {
            config.fluid.fluidMaterialId = parseUInt32(*value, "fluid_material_id");
        }
        if (const auto value = readIfPresent("solid_penalty"); value.has_value()) {
            config.fluid.solidPenalty = parseDouble(*value, "solid_penalty");
        }
        if (const auto value = readIfPresent("forcing_magnitude"); value.has_value()) {
            config.fluid.forcingMagnitude = parseDouble(*value, "forcing_magnitude");
        }
        if (const auto value = readIfPresent("voxel_size"); value.has_value()) {
            config.fluid.voxelSize = parseDouble(*value, "voxel_size");
        }
        if (const auto value = readIfPresent("macro_pressure_gradient"); value.has_value()) {
            config.fluid.macroPressureGradient = parseVector3(*value, "macro_pressure_gradient");
        }
        if (const auto value = readIfPresent("max_iterations"); value.has_value()) {
            config.fluid.maxIterations = parseInt(*value, "max_iterations");
        }
        if (const auto value = readIfPresent("progress_interval"); value.has_value()) {
            config.fluid.progressInterval = parseInt(*value, "progress_interval");
        }
        if (const auto value = readIfPresent("write_directional_velocity_fields"); value.has_value()) {
            config.fluid.writeDirectionalVelocityFields = parseBool(*value, "write_directional_velocity_fields");
        }
        if (const auto value = readIfPresent("require_single_solid_material_id"); value.has_value()) {
            config.fluid.requireSingleSolidMaterialId = parseBool(*value, "require_single_solid_material_id");
        }
        if (const auto value = readIfPresent("tolerance"); value.has_value()) {
            config.fluid.tolerance = parseDouble(*value, "tolerance");
        }
        if (const auto value = readIfPresent("relaxation"); value.has_value()) {
            config.fluid.relaxation = parseDouble(*value, "relaxation");
        }
        if (const auto value = readIfPresent("field_output_file"); value.has_value()) {
            config.fluid.fieldOutputFile = *value;
        }
        if (const auto value = readIfPresent("reference_file"); value.has_value()) {
            config.fluid.referenceFile = *value;
        }
    }

    if (cli.analysis.has_value()) {
        config.analysis = analysisFromString(*cli.analysis);
    }
    if (cli.geometryFile.has_value()) {
        config.geometryFile = *cli.geometryFile;
    }
    if (cli.geometryFormat.has_value()) {
        config.geometryFormat = geometryFormatFromString(*cli.geometryFormat);
    }
    if (cli.materialFile.has_value()) {
        config.materialFile = *cli.materialFile;
    }
    if (cli.outputFile.has_value()) {
        config.outputFile = *cli.outputFile;
    }
    if (cli.logFile.has_value()) {
        config.logFile = *cli.logFile;
    }
    if (cli.backend.has_value()) {
        config.resources.backend = backendFromString(*cli.backend);
    }
    if (cli.cpus.has_value()) {
        config.resources.cpus = *cli.cpus;
    }
    if (cli.gpus.has_value()) {
        config.resources.gpus = *cli.gpus;
    }
    if (cli.fluidMode.has_value()) {
        config.fluid.mode = fluidRunModeFromString(*cli.fluidMode);
    }
    if (cli.fluidMaterialId.has_value()) {
        config.fluid.fluidMaterialId = *cli.fluidMaterialId;
    }
    if (cli.solidPenalty.has_value()) {
        config.fluid.solidPenalty = *cli.solidPenalty;
    }
    if (cli.forcingMagnitude.has_value()) {
        config.fluid.forcingMagnitude = *cli.forcingMagnitude;
    }
    if (cli.voxelSize.has_value()) {
        config.fluid.voxelSize = *cli.voxelSize;
    }
    if (cli.macroPressureGradient.has_value()) {
        config.fluid.macroPressureGradient = *cli.macroPressureGradient;
    }
    if (cli.maxIterations.has_value()) {
        config.fluid.maxIterations = *cli.maxIterations;
    }
    if (cli.progressInterval.has_value()) {
        config.fluid.progressInterval = *cli.progressInterval;
    }
    if (cli.writeDirectionalVelocityFields.has_value()) {
        config.fluid.writeDirectionalVelocityFields = *cli.writeDirectionalVelocityFields;
    }
    if (cli.requireSingleSolidMaterialId.has_value()) {
        config.fluid.requireSingleSolidMaterialId = *cli.requireSingleSolidMaterialId;
    }
    if (cli.tolerance.has_value()) {
        config.fluid.tolerance = *cli.tolerance;
    }
    if (cli.relaxation.has_value()) {
        config.fluid.relaxation = *cli.relaxation;
    }
    if (cli.fieldOutputFile.has_value()) {
        config.fluid.fieldOutputFile = *cli.fieldOutputFile;
    }
    if (cli.referenceFile.has_value()) {
        config.fluid.referenceFile = *cli.referenceFile;
    }

    return config;
}

void printHelp(const std::string& programName) {
    std::cout
        << "Usage: " << programName << " [options]\n\n"
        << "Options:\n"
        << "  -h, --help              Show this help message\n"
        << "  -i, --input <file>      Input config file (key = value)\n"
        << "  --analysis <type>       Analysis type: elastic | fluid\n"
        << "  --geometry <file>       Geometry file path\n"
        << "  --geometry-format <f>   Geometry format: auto | text | binary\n"
        << "  --materials <file>      Material properties file path\n"
        << "  --output <file>         Output file path\n"
        << "  --log <file>            Log file path\n"
        << "  --backend <type>        Backend: cpu | cuda | intel\n"
        << "  --cpus <n>              Number of CPU threads (>= 1)\n"
        << "  --gpus <n>              Number of GPUs (>= 0)\n"
        << "  --fluid-mode <m>        Fluid mode: permeability | response\n"
        << "  --fluid-material-id <i> Material ID representing fluid phase\n"
        << "  --solid-penalty <a>     Brinkman penalty in solid voxels\n"
        << "  --forcing <f>           Forcing magnitude for tensor runs\n"
        << "  --voxel-size <h>        Physical voxel size used to scale permeability by h^2\n"
        << "  --macro-grad <gx,gy,gz> Macro pressure gradient for response mode\n"
        << "  --max-iter <n>          Maximum iterations for fluid solve\n"
        << "  --progress-interval <n> Report progress every n iterations (0 disables)\n"
        << "  --write-directional-fields <b> Write vx/vy/vz field files in permeability mode\n"
        << "  --require-single-solid-id <b> Require exactly one solid material id when using phase tags\n"
        << "  --tol <v>               Convergence tolerance for fluid solve\n"
        << "  --omega <v>             Relaxation factor in (0,1]\n"
        << "  --field-output <file>   Binary velocity output path for response mode\n"
        << "  --reference <file>      Optional reference file path for comparison\n\n"
        << "CLI values override values from --input file.\n";
}

} // namespace common
