#include "common/StudyRunner.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "common/FFT.hpp"
#include "common/FluidBrinkman.hpp"
#include "common/Logger.hpp"
#include "common/Types.hpp"

namespace {

std::size_t nextPowerOfTwo(std::size_t value) {
    if (value <= 1) {
        return 1;
    }

    std::size_t result = 1;
    while (result < value) {
        result <<= 1U;
    }
    return result;
}

void writeMagnitudes(const std::string& outputFile, const std::vector<double>& magnitudes) {
    std::ofstream out(outputFile, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output file: " + outputFile);
    }

    out << "# k magnitude\n";
    for (std::size_t i = 0; i < magnitudes.size(); ++i) {
        out << i << " " << magnitudes[i] << "\n";
    }
}

double valueFromMaterialOrId(
    const common::MaterialDatabase& materialDb,
    std::uint32_t materialId,
    std::initializer_list<const char*> preferredKeys,
    bool& usedProperty) {
    const auto& values = materialDb.at(materialId).values;
    for (const char* key : preferredKeys) {
        const auto it = values.find(key);
        if (it != values.end()) {
            usedProperty = true;
            return it->second;
        }
    }

    usedProperty = false;
    return static_cast<double>(materialId);
}

std::vector<common::fft::Complex> geometryToMaterialSignal(
    const common::Geometry& geometry,
    const common::MaterialDatabase& materialDb,
    std::initializer_list<const char*> preferredKeys,
    std::size_t& fallbackCount,
    double scale) {
    const std::size_t n = nextPowerOfTwo(std::max<std::size_t>(1, geometry.materialIds.size()));
    std::vector<common::fft::Complex> signal(n, common::fft::Complex(0.0, 0.0));

    fallbackCount = 0;
    for (std::size_t i = 0; i < geometry.materialIds.size(); ++i) {
        bool usedProperty = false;
        const double value = valueFromMaterialOrId(
            materialDb,
            geometry.materialIds[i],
            preferredKeys,
            usedProperty);
        if (!usedProperty) {
            ++fallbackCount;
        }
        signal[i] = common::fft::Complex(value * scale, 0.0);
    }

    return signal;
}

std::size_t countDistinctMaterials(const common::Geometry& geometry) {
    std::unordered_set<std::uint32_t> uniqueIds;
    uniqueIds.reserve(geometry.materialIds.size());
    for (std::uint32_t materialId : geometry.materialIds) {
        uniqueIds.insert(materialId);
    }
    return uniqueIds.size();
}

bool usesPhaseTags(const common::Geometry& geometry, const common::MaterialDatabase& materialDb) {
    std::unordered_set<std::uint32_t> uniqueIds;
    uniqueIds.reserve(geometry.materialIds.size());
    for (std::uint32_t materialId : geometry.materialIds) {
        if (uniqueIds.insert(materialId).second && materialDb.hasTag(materialId, "phase")) {
            return true;
        }
    }
    return false;
}

std::string withSuffix(const std::string& prefix, const std::string& suffix) {
    return prefix + suffix;
}

std::string formatScientific(double value, int decimals) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(decimals) << value;
    return out.str();
}

std::string formatFixed(double value, int decimals) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

} // namespace

namespace common {

void StudyRunner::run(const AppConfig& config, const Geometry& geometry, const MaterialDatabase& materialDb) const {
    Logger::instance().info(
        "Running analysis='" + toString(config.analysis) +
        "' backend='" + toString(config.resources.backend) +
        "' cpus=" + std::to_string(config.resources.cpus) +
        " gpus=" + std::to_string(config.resources.gpus));
    Logger::instance().info(
        "Material validation passed for " + std::to_string(countDistinctMaterials(geometry)) +
        " material ids against database with " + std::to_string(materialDb.size()) + " entries");

    switch (config.analysis) {
    case AnalysisType::Elastic:
        runElastic(config, geometry, materialDb);
        break;
    case AnalysisType::Fluid:
        runFluid(config, geometry, materialDb);
        break;
    default:
        throw std::runtime_error("Unsupported analysis type");
    }
}

void StudyRunner::runElastic(const AppConfig& config, const Geometry& geometry, const MaterialDatabase& materialDb) const {
    Logger::instance().info("Elastic study placeholder: FFT of material-aware scalar field.");

    std::size_t fallbackCount = 0;
    const auto signal = geometryToMaterialSignal(
        geometry,
        materialDb,
        {"young", "density"},
        fallbackCount,
        1.0);
    Logger::instance().info(
        "Elastic placeholder mapping: preferred keys=[young,density], fallback-to-id=" +
        std::to_string(fallbackCount) + " voxels");
    const auto spectrum = fft::fft1d(signal);
    const auto magnitudes = fft::magnitude(spectrum);

    writeMagnitudes(config.outputFile, magnitudes);
    Logger::instance().info("Elastic output written to: " + config.outputFile);
}

void StudyRunner::runFluid(const AppConfig& config, const Geometry& geometry, const MaterialDatabase& materialDb) const {
    const std::string fluidSelector = usesPhaseTags(geometry, materialDb)
        ? "fluid_material_id=ignored(phase tags)"
        : "fluid_material_id=" + std::to_string(config.fluid.fluidMaterialId);

    Logger::instance().info(
        "Fluid solve: solver='brinkman' mode='" + toString(config.fluid.mode) +
        "' " + fluidSelector +
        " penalty=" + formatScientific(config.fluid.solidPenalty, 0) +
        " max_iter=" + std::to_string(config.fluid.maxIterations) +
        " tol=" + formatScientific(config.fluid.tolerance, 0) +
        " omega=" + formatFixed(config.fluid.relaxation, 2));

    if (config.fluid.mode == FluidRunMode::Permeability) {
        const std::string outputPrefix = config.outputFile;
        fluid::PermeabilityResult result;

        if (config.fluid.writeDirectionalVelocityFields) {
            const std::array<std::string, 3> labels{"x", "y", "z"};
            result = fluid::computePermeabilityTensor(
                config,
                geometry,
                materialDb,
                [&](std::size_t axis, const std::array<double, 3>& force, const fluid::SolveResult& directional) {
                    const std::string directionalPath = withSuffix(outputPrefix, ".v" + labels[axis] + ".fvel");
                    fluid::writeVelocityFieldBinary(directionalPath, geometry, force, directional);
                    Logger::instance().info(
                        "Directional velocity field written: " + directionalPath +
                        " iterations=" + std::to_string(directional.diagnostics.iterations) +
                        " residual=" + formatScientific(directional.diagnostics.residual, 2) +
                        " converged=" + std::string(directional.diagnostics.converged ? "yes" : "no"));
                });
        } else {
            result = fluid::computePermeabilityTensor(
                config,
                geometry,
                materialDb);
        }

        const std::string permeabilityPath = withSuffix(outputPrefix, ".perm.txt");

        fluid::writePermeabilityTensorText(
            config,
            materialDb,
            permeabilityPath,
            geometry,
            result);

        Logger::instance().info(
            "Permeability tensor written (text): " + permeabilityPath +
            " porosity=" + formatFixed(result.porosity, 2) +
            " viscosity=" + formatFixed(result.viscosity, 2) +
            " elapsed=" + formatFixed(result.elapsedSeconds, 2) + " s");
        Logger::instance().info(
            "Tensor rows: [" +
            formatScientific(result.tensor[0], 2) + ", " + formatScientific(result.tensor[1], 2) + ", " + formatScientific(result.tensor[2], 2) + "] [" +
            formatScientific(result.tensor[3], 2) + ", " + formatScientific(result.tensor[4], 2) + ", " + formatScientific(result.tensor[5], 2) + "] [" +
            formatScientific(result.tensor[6], 2) + ", " + formatScientific(result.tensor[7], 2) + ", " + formatScientific(result.tensor[8], 2) + "]");
        return;
    }

    const std::array<double, 3> force{
        -config.fluid.macroPressureGradient[0],
        -config.fluid.macroPressureGradient[1],
        -config.fluid.macroPressureGradient[2]
    };
    const fluid::SolveResult result = fluid::solveBrinkman(config, geometry, materialDb, force);

    const std::string outputPath =
        config.fluid.fieldOutputFile.empty()
            ? withSuffix(config.outputFile, ".response.fvel")
            : config.fluid.fieldOutputFile;
    fluid::writeVelocityFieldBinary(outputPath, geometry, force, result);

    Logger::instance().info(
        "Velocity field written (binary FVEL): " + outputPath +
        " iterations=" + std::to_string(result.diagnostics.iterations) +
        " residual=" + formatScientific(result.diagnostics.residual, 2) +
        " converged=" + std::string(result.diagnostics.converged ? "yes" : "no") +
        " elapsed=" + formatFixed(result.diagnostics.elapsedSeconds, 2) + " s");
    Logger::instance().info(
        "Superficial velocity = (" +
        std::to_string(result.superficialVelocity[0]) + ", " +
        std::to_string(result.superficialVelocity[1]) + ", " +
        std::to_string(result.superficialVelocity[2]) + ")");
    if (!result.diagnostics.converged) {
        Logger::instance().warn("Fluid response solve reached maximum iterations before tolerance.");
    }
}

} // namespace common
