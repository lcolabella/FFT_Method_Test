#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace common {

// Analysis and backend type enumerations (formerly in core/Types.hpp)
enum class AnalysisType { Elastic, Fluid, Unknown };
enum class BackendType  { CPU, CUDA, Intel };

std::string toString(AnalysisType analysis);
std::string toString(BackendType backend);
AnalysisType analysisFromString(const std::string& value);
BackendType  backendFromString(const std::string& value);

struct RuntimeResources {
    int cpus = 1;
    int gpus = 0;
    BackendType backend = BackendType::CPU;
};

enum class GeometryFormat {
    Auto,
    Text,
    Binary
};

enum class FluidRunMode {
    Permeability,
    Response
};

struct FluidConfig {
    FluidRunMode mode = FluidRunMode::Permeability;
    std::uint32_t fluidMaterialId = 0;
    double solidPenalty = 1.0e8;
    double forcingMagnitude = 1.0;
    double voxelSize = 1.0;
    std::array<double, 3> macroPressureGradient{1.0, 0.0, 0.0};
    int maxIterations = 600;
    int progressInterval = 0;
    bool writeDirectionalVelocityFields = false;
    bool requireSingleSolidMaterialId = true;
    double tolerance = 1.0e-8;
    double relaxation = 0.9;
    std::string fieldOutputFile;
    std::string referenceFile;
};

struct AppConfig {
    AnalysisType analysis = AnalysisType::Unknown;
    std::string geometryFile;
    GeometryFormat geometryFormat = GeometryFormat::Auto;
    std::string materialFile;
    std::string outputFile = "output.txt";
    std::string logFile = "run.log";
    RuntimeResources resources;
    FluidConfig fluid;
};

std::string toString(GeometryFormat format);
GeometryFormat geometryFormatFromString(const std::string& value);
std::string toString(FluidRunMode mode);
FluidRunMode fluidRunModeFromString(const std::string& value);

std::vector<std::string> validateConfig(const AppConfig& config);

} // namespace common
