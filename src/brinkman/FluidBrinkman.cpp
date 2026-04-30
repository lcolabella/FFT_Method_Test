#include "brinkman/FluidBrinkman.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "logging/Logger.hpp"

namespace {

std::size_t flatten(std::size_t x, std::size_t y, std::size_t z, std::size_t nx, std::size_t ny) {
    return z * nx * ny + y * nx + x;
}

std::size_t wrapIndex(std::size_t i, int delta, std::size_t n) {
    if (delta > 0) {
        return (i + 1U) % n;
    }
    return (i + n - 1U) % n;
}

struct PhaseMask {
    std::unordered_set<std::uint32_t> fluidIds;
    std::unordered_set<std::uint32_t> solidIds;
    bool usesPhaseTags = false;
};

std::string joinIds(const std::unordered_set<std::uint32_t>& ids) {
    std::vector<std::uint32_t> sorted(ids.begin(), ids.end());
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream out;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << sorted[i];
    }
    return out.str();
}

std::vector<std::uint32_t> geometryMaterialIds(const common::Geometry& geometry) {
    std::unordered_set<std::uint32_t> unique;
    unique.reserve(geometry.materialIds.size());
    for (std::uint32_t id : geometry.materialIds) {
        unique.insert(id);
    }

    std::vector<std::uint32_t> ids(unique.begin(), unique.end());
    std::sort(ids.begin(), ids.end());
    return ids;
}

PhaseMask resolvePhaseMask(
    const common::AppConfig& config,
    const common::Geometry& geometry,
    const common::MaterialDatabase& materialDb) {
    PhaseMask mask;
    const auto ids = geometryMaterialIds(geometry);

    for (std::uint32_t id : ids) {
        if (materialDb.hasTag(id, "phase")) {
            mask.usesPhaseTags = true;
            break;
        }
    }

    if (mask.usesPhaseTags) {
        for (std::uint32_t id : ids) {
            if (!materialDb.hasTag(id, "phase")) {
                throw std::runtime_error(
                    "Material id " + std::to_string(id) +
                    " is missing required tag phase=fluid|solid");
            }
            const std::string phase = materialDb.tag(id, "phase");
            if (phase == "fluid") {
                mask.fluidIds.insert(id);
            } else if (phase == "solid") {
                mask.solidIds.insert(id);
            } else {
                throw std::runtime_error(
                    "Unsupported phase tag for material id " + std::to_string(id) + ": " + phase);
            }
        }
    } else {
        mask.fluidIds.insert(config.fluid.fluidMaterialId);
        for (std::uint32_t id : ids) {
            if (id != config.fluid.fluidMaterialId) {
                mask.solidIds.insert(id);
            }
        }
    }

    if (mask.fluidIds.empty()) {
        throw std::runtime_error("No fluid material ids found. Set phase=fluid or fluid_material_id.");
    }
    if (mask.solidIds.empty()) {
        throw std::runtime_error("No solid material ids found for Brinkman solve.");
    }
    if (config.fluid.requireSingleSolidMaterialId && mask.solidIds.size() != 1) {
        throw std::runtime_error(
            "Expected exactly one solid material id, found " + std::to_string(mask.solidIds.size()));
    }

    if (mask.usesPhaseTags) {
        static bool warnedPhaseOverridesLegacy = false;
        if (!warnedPhaseOverridesLegacy) {
            warnedPhaseOverridesLegacy = true;
            common::Logger::instance().warn(
                "Phase tags detected in materials: fluid_material_id=" +
                std::to_string(config.fluid.fluidMaterialId) +
                " is ignored. Using phase-tag sets fluid={" + joinIds(mask.fluidIds) +
                "} solid={" + joinIds(mask.solidIds) + "}.");
        }
    }

    return mask;
}

double readFluidViscosity(
    const common::MaterialDatabase& materialDb,
    std::uint32_t fluidId) {
    const auto& props = materialDb.at(fluidId).values;
    const auto it = props.find("viscosity");
    if (it == props.end()) {
        throw std::runtime_error(
            "Missing viscosity for fluid material id: " + std::to_string(fluidId));
    }
    return it->second;
}

struct CoefficientFields {
    std::vector<double> alpha;
    std::vector<double> viscosity;
    double porosity = 0.0;
    double meanFluidViscosity = 1.0;
};

CoefficientFields buildCoefficientFields(
    const common::AppConfig& config,
    const common::Geometry& geometry,
    const common::MaterialDatabase& materialDb,
    const PhaseMask& phaseMask) {
    CoefficientFields fields;
    fields.alpha.assign(geometry.materialIds.size(), config.fluid.solidPenalty);
    fields.viscosity.assign(geometry.materialIds.size(), 1.0);

    std::unordered_map<std::uint32_t, double> fluidViscosity;
    fluidViscosity.reserve(phaseMask.fluidIds.size());
    for (std::uint32_t fluidId : phaseMask.fluidIds) {
        fluidViscosity[fluidId] = readFluidViscosity(materialDb, fluidId);
    }

    std::size_t fluidCount = 0;
    double viscositySum = 0.0;
    for (std::size_t i = 0; i < geometry.materialIds.size(); ++i) {
        const std::uint32_t id = geometry.materialIds[i];
        const auto it = fluidViscosity.find(id);
        if (it == fluidViscosity.end()) {
            continue;
        }

        fields.alpha[i] = 0.0;
        fields.viscosity[i] = it->second;
        ++fluidCount;
        viscositySum += it->second;
    }

    if (!geometry.materialIds.empty()) {
        fields.porosity = static_cast<double>(fluidCount) /
            static_cast<double>(geometry.materialIds.size());
    }
    if (fluidCount > 0) {
        fields.meanFluidViscosity = viscositySum / static_cast<double>(fluidCount);
    }

    return fields;
}

std::string formatSeconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << " s";
    return out.str();
}

std::string formatSecondsNoDecimals(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << seconds << " s";
    return out.str();
}

std::string formatResidual(double residual) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(2) << residual;
    return out.str();
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

std::string axisName(std::size_t axis) {
    if (axis == 0) {
        return "X";
    }
    if (axis == 1) {
        return "Y";
    }
    return "Z";
}

std::array<double, 3> computeSuperficialVelocity(const common::fluid::VelocityField& field) {
    const std::size_t n = field.ux.size();
    std::array<double, 3> avg{0.0, 0.0, 0.0};
    if (n == 0) {
        return avg;
    }

    double sumX = 0.0;
    double sumY = 0.0;
    double sumZ = 0.0;

#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for reduction(+:sumX,sumY,sumZ)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const std::size_t id = static_cast<std::size_t>(i);
        sumX += field.ux[id];
        sumY += field.uy[id];
        sumZ += field.uz[id];
    }

    avg[0] = sumX / static_cast<double>(n);
    avg[1] = sumY / static_cast<double>(n);
    avg[2] = sumZ / static_cast<double>(n);
    return avg;
}

bool isFinite3(const std::array<double, 3>& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

void writeRaw(std::ofstream& out, const void* data, std::size_t bytes) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error("Binary output write failed");
    }
}

void ensureParentDirectory(const std::string& outputFile) {
    const std::filesystem::path path(outputFile);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

} // namespace

namespace common::fluid {

SolveResult solveBrinkman(
    const AppConfig& config,
    const Geometry& geometry,
    const MaterialDatabase& materialDb,
    const std::array<double, 3>& force) {
    const PhaseMask phaseMask = resolvePhaseMask(config, geometry, materialDb);
    const CoefficientFields coeff = buildCoefficientFields(config, geometry, materialDb, phaseMask);

    SolveResult result;
    result.porosity = coeff.porosity;
    result.viscosity = coeff.meanFluidViscosity;

    const std::size_t n = geometry.voxelCount();
    result.velocity.ux.assign(n, 0.0);
    result.velocity.uy.assign(n, 0.0);
    result.velocity.uz.assign(n, 0.0);

    std::vector<double> pressure(n, 0.0);
    std::vector<double> uStarX(n, 0.0);
    std::vector<double> uStarY(n, 0.0);
    std::vector<double> uStarZ(n, 0.0);
    std::vector<double> divergence(n, 0.0);
    std::vector<double> phi(n, 0.0);
    std::vector<double> phiNext(n, 0.0);

    const auto start = std::chrono::steady_clock::now();
    constexpr int pressureIterationsPerStep = 35;

    for (int iter = 0; iter < config.fluid.maxIterations; ++iter) {
        double divergenceL2 = 0.0;

#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for collapse(3)
#endif
        for (std::size_t z = 0; z < geometry.nz; ++z) {
            const std::size_t zm = wrapIndex(z, -1, geometry.nz);
            const std::size_t zp = wrapIndex(z, +1, geometry.nz);

            for (std::size_t y = 0; y < geometry.ny; ++y) {
                const std::size_t ym = wrapIndex(y, -1, geometry.ny);
                const std::size_t yp = wrapIndex(y, +1, geometry.ny);

                for (std::size_t x = 0; x < geometry.nx; ++x) {
                    const std::size_t xm = wrapIndex(x, -1, geometry.nx);
                    const std::size_t xp = wrapIndex(x, +1, geometry.nx);

                    const std::size_t id = flatten(x, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXm = flatten(xm, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXp = flatten(xp, y, z, geometry.nx, geometry.ny);
                    const std::size_t idYm = flatten(x, ym, z, geometry.nx, geometry.ny);
                    const std::size_t idYp = flatten(x, yp, z, geometry.nx, geometry.ny);
                    const std::size_t idZm = flatten(x, y, zm, geometry.nx, geometry.ny);
                    const std::size_t idZp = flatten(x, y, zp, geometry.nx, geometry.ny);

                    const double mu = coeff.viscosity[id];
                    const double alpha = coeff.alpha[id];

                    const double lapUx =
                        result.velocity.ux[idXm] + result.velocity.ux[idXp] +
                        result.velocity.ux[idYm] + result.velocity.ux[idYp] +
                        result.velocity.ux[idZm] + result.velocity.ux[idZp];
                    const double lapUy =
                        result.velocity.uy[idXm] + result.velocity.uy[idXp] +
                        result.velocity.uy[idYm] + result.velocity.uy[idYp] +
                        result.velocity.uy[idZm] + result.velocity.uy[idZp];
                    const double lapUz =
                        result.velocity.uz[idXm] + result.velocity.uz[idXp] +
                        result.velocity.uz[idYm] + result.velocity.uz[idYp] +
                        result.velocity.uz[idZm] + result.velocity.uz[idZp];

                    const double dpdx = 0.5 * (pressure[idXp] - pressure[idXm]);
                    const double dpdy = 0.5 * (pressure[idYp] - pressure[idYm]);
                    const double dpdz = 0.5 * (pressure[idZp] - pressure[idZm]);

                    const double denom = alpha + 6.0 * mu;
                    const double jacUx = (force[0] - dpdx + mu * lapUx) / denom;
                    const double jacUy = (force[1] - dpdy + mu * lapUy) / denom;
                    const double jacUz = (force[2] - dpdz + mu * lapUz) / denom;

                    uStarX[id] = config.fluid.relaxation * jacUx +
                        (1.0 - config.fluid.relaxation) * result.velocity.ux[id];
                    uStarY[id] = config.fluid.relaxation * jacUy +
                        (1.0 - config.fluid.relaxation) * result.velocity.uy[id];
                    uStarZ[id] = config.fluid.relaxation * jacUz +
                        (1.0 - config.fluid.relaxation) * result.velocity.uz[id];
                }
            }
        }

#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for collapse(3) reduction(+:divergenceL2)
#endif
        for (std::size_t z = 0; z < geometry.nz; ++z) {
            const std::size_t zm = wrapIndex(z, -1, geometry.nz);
            const std::size_t zp = wrapIndex(z, +1, geometry.nz);
            for (std::size_t y = 0; y < geometry.ny; ++y) {
                const std::size_t ym = wrapIndex(y, -1, geometry.ny);
                const std::size_t yp = wrapIndex(y, +1, geometry.ny);
                for (std::size_t x = 0; x < geometry.nx; ++x) {
                    const std::size_t xm = wrapIndex(x, -1, geometry.nx);
                    const std::size_t xp = wrapIndex(x, +1, geometry.nx);

                    const std::size_t id = flatten(x, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXm = flatten(xm, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXp = flatten(xp, y, z, geometry.nx, geometry.ny);
                    const std::size_t idYm = flatten(x, ym, z, geometry.nx, geometry.ny);
                    const std::size_t idYp = flatten(x, yp, z, geometry.nx, geometry.ny);
                    const std::size_t idZm = flatten(x, y, zm, geometry.nx, geometry.ny);
                    const std::size_t idZp = flatten(x, y, zp, geometry.nx, geometry.ny);

                    const double div =
                        0.5 * (uStarX[idXp] - uStarX[idXm]) +
                        0.5 * (uStarY[idYp] - uStarY[idYm]) +
                        0.5 * (uStarZ[idZp] - uStarZ[idZm]);
                    divergence[id] = div;
                    divergenceL2 += div * div;
                }
            }
        }

        std::fill(phi.begin(), phi.end(), 0.0);
        for (int k = 0; k < pressureIterationsPerStep; ++k) {
#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for collapse(3)
#endif
            for (std::size_t z = 0; z < geometry.nz; ++z) {
                const std::size_t zm = wrapIndex(z, -1, geometry.nz);
                const std::size_t zp = wrapIndex(z, +1, geometry.nz);
                for (std::size_t y = 0; y < geometry.ny; ++y) {
                    const std::size_t ym = wrapIndex(y, -1, geometry.ny);
                    const std::size_t yp = wrapIndex(y, +1, geometry.ny);
                    for (std::size_t x = 0; x < geometry.nx; ++x) {
                        const std::size_t xm = wrapIndex(x, -1, geometry.nx);
                        const std::size_t xp = wrapIndex(x, +1, geometry.nx);

                        const std::size_t id = flatten(x, y, z, geometry.nx, geometry.ny);
                        const std::size_t idXm = flatten(xm, y, z, geometry.nx, geometry.ny);
                        const std::size_t idXp = flatten(xp, y, z, geometry.nx, geometry.ny);
                        const std::size_t idYm = flatten(x, ym, z, geometry.nx, geometry.ny);
                        const std::size_t idYp = flatten(x, yp, z, geometry.nx, geometry.ny);
                        const std::size_t idZm = flatten(x, y, zm, geometry.nx, geometry.ny);
                        const std::size_t idZp = flatten(x, y, zp, geometry.nx, geometry.ny);

                        const double neighbor =
                            phi[idXm] + phi[idXp] + phi[idYm] +
                            phi[idYp] + phi[idZm] + phi[idZp];
                        phiNext[id] = (neighbor - divergence[id]) / 6.0;
                    }
                }
            }
            phi.swap(phiNext);
        }

#if defined(COMMON_USE_OPENMP)
#pragma omp parallel for collapse(3)
#endif
        for (std::size_t z = 0; z < geometry.nz; ++z) {
            const std::size_t zm = wrapIndex(z, -1, geometry.nz);
            const std::size_t zp = wrapIndex(z, +1, geometry.nz);
            for (std::size_t y = 0; y < geometry.ny; ++y) {
                const std::size_t ym = wrapIndex(y, -1, geometry.ny);
                const std::size_t yp = wrapIndex(y, +1, geometry.ny);
                for (std::size_t x = 0; x < geometry.nx; ++x) {
                    const std::size_t xm = wrapIndex(x, -1, geometry.nx);
                    const std::size_t xp = wrapIndex(x, +1, geometry.nx);

                    const std::size_t id = flatten(x, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXm = flatten(xm, y, z, geometry.nx, geometry.ny);
                    const std::size_t idXp = flatten(xp, y, z, geometry.nx, geometry.ny);
                    const std::size_t idYm = flatten(x, ym, z, geometry.nx, geometry.ny);
                    const std::size_t idYp = flatten(x, yp, z, geometry.nx, geometry.ny);
                    const std::size_t idZm = flatten(x, y, zm, geometry.nx, geometry.ny);
                    const std::size_t idZp = flatten(x, y, zp, geometry.nx, geometry.ny);

                    const double dphidx = 0.5 * (phi[idXp] - phi[idXm]);
                    const double dphidy = 0.5 * (phi[idYp] - phi[idYm]);
                    const double dphidz = 0.5 * (phi[idZp] - phi[idZm]);

                    result.velocity.ux[id] = uStarX[id] - dphidx;
                    result.velocity.uy[id] = uStarY[id] - dphidy;
                    result.velocity.uz[id] = uStarZ[id] - dphidz;
                    pressure[id] += phi[id];
                }
            }
        }

        result.diagnostics.iterations = iter + 1;
        result.diagnostics.residual = std::sqrt(divergenceL2 / static_cast<double>(n));
        result.diagnostics.elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        if (!std::isfinite(result.diagnostics.residual)) {
            throw std::runtime_error(
                "Brinkman solver became unstable: non-finite residual at iteration " +
                std::to_string(result.diagnostics.iterations));
        }

        if (config.fluid.progressInterval > 0 &&
            ((result.diagnostics.iterations % config.fluid.progressInterval) == 0 ||
             result.diagnostics.iterations == config.fluid.maxIterations)) {
            const double spi = result.diagnostics.elapsedSeconds /
                static_cast<double>(result.diagnostics.iterations);
            const double eta = spi * static_cast<double>(
                config.fluid.maxIterations - result.diagnostics.iterations);
            common::Logger::instance().info(
                "Fluid progress [coupled] iter=" + std::to_string(result.diagnostics.iterations) +
                "/" + std::to_string(config.fluid.maxIterations) +
                " residual=" + formatResidual(result.diagnostics.residual) +
                " elapsed=" + formatSeconds(result.diagnostics.elapsedSeconds) +
                " eta_to_max=" + formatSeconds(eta));
        }

        if (result.diagnostics.residual < config.fluid.tolerance) {
            result.diagnostics.converged = true;
            break;
        }
    }

    result.componentDiagnostics = {result.diagnostics, result.diagnostics, result.diagnostics};
    result.superficialVelocity = computeSuperficialVelocity(result.velocity);
    if (!isFinite3(result.superficialVelocity)) {
        throw std::runtime_error(
            "Brinkman solver produced non-finite macroscopic velocity");
    }
    return result;
}

PermeabilityResult computePermeabilityTensor(
    const AppConfig& config,
    const Geometry& geometry,
    const MaterialDatabase& materialDb,
    const std::function<void(std::size_t, const std::array<double, 3>&, const SolveResult&)>& onLoadCaseSolved) {
    PermeabilityResult result;
    const auto start = std::chrono::steady_clock::now();
    const double permeabilityScale = config.fluid.voxelSize * config.fluid.voxelSize;

    for (int j = 0; j < 3; ++j) {
        std::array<double, 3> force{0.0, 0.0, 0.0};
        force[static_cast<std::size_t>(j)] = config.fluid.forcingMagnitude;

        common::Logger::instance().info(
            "Starting permeability load case axis=" + std::to_string(j) +
            " force=(" + formatScientific(force[0], 2) + ", " +
            formatScientific(force[1], 2) + ", " + formatScientific(force[2], 2) + ")");

        const SolveResult solveResult = solveBrinkman(config, geometry, materialDb, force);
        result.porosity = solveResult.porosity;
        result.viscosity = solveResult.viscosity;
        result.loadDiagnostics[static_cast<std::size_t>(j)] = solveResult.diagnostics;

        for (int i = 0; i < 3; ++i) {
            result.tensor[static_cast<std::size_t>(i) * 3U + static_cast<std::size_t>(j)] =
                solveResult.viscosity * solveResult.superficialVelocity[static_cast<std::size_t>(i)] /
                force[static_cast<std::size_t>(j)] * permeabilityScale;
        }

        if (onLoadCaseSolved) {
            onLoadCaseSolved(static_cast<std::size_t>(j), force, solveResult);
        }

        common::Logger::instance().info(
            "Completed permeability load case axis=" + std::to_string(j) +
            " iterations=" + std::to_string(solveResult.diagnostics.iterations) +
            " residual=" + formatResidual(solveResult.diagnostics.residual) +
            " converged=" + std::string(solveResult.diagnostics.converged ? "yes" : "no") +
            " elapsed=" + formatSeconds(solveResult.diagnostics.elapsedSeconds));
    }

    result.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

void writePermeabilityTensorText(
    const AppConfig& config,
    const MaterialDatabase& materialDb,
    const std::string& outputFile,
    const Geometry& geometry,
    const PermeabilityResult& result) {
    ensureParentDirectory(outputFile);
    std::ofstream out(outputFile, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open permeability output file: " + outputFile);
    }

    out << "Permeability Tensor Report\n";
    out << "==========================\n\n";

    out << "Geometry dimensions (nx, ny, nz): "
        << geometry.nx << ", " << geometry.ny << ", " << geometry.nz << "\n";
    out << "Pore volume fraction (porosity): " << formatFixed(result.porosity, 2) << "\n";

    std::vector<std::uint32_t> fluidIds = materialDb.idsForTag("phase", "fluid");
    if (fluidIds.empty()) {
        fluidIds.push_back(config.fluid.fluidMaterialId);
    }

    out << "Fluid dynamic viscosity by phase material id:\n";
    for (std::uint32_t fluidId : fluidIds) {
        const auto& props = materialDb.at(fluidId).values;
        const auto it = props.find("viscosity");
        if (it == props.end()) {
            out << "  - material " << fluidId << ": not provided\n";
        } else {
            out << "  - material " << fluidId << ": " << formatScientific(it->second, 6) << "\n";
        }
    }

    const double scaleH2 = config.fluid.voxelSize * config.fluid.voxelSize;
    out << "Voxel size (physical length per voxel): " << formatScientific(config.fluid.voxelSize, 2) << "\n";
    out << "Permeability scaling factor (voxel_size^2): " << formatScientific(scaleH2, 2) << "\n";
    out << "Total solver elapsed time: " << formatSecondsNoDecimals(result.elapsedSeconds) << "\n\n";

    out << "Load-case summary\n";
    out << "-----------------\n";
    out << std::left
        << std::setw(10) << "Case"
        << std::setw(38) << "Applied force (Fx,Fy,Fz)"
        << std::setw(12) << "Iterations"
        << std::setw(16) << "Residual"
        << std::setw(12) << "Converged"
        << "\n";

    for (std::size_t axis = 0; axis < result.loadDiagnostics.size(); ++axis) {
        const auto& diagnostics = result.loadDiagnostics[axis];
        std::array<double, 3> force{0.0, 0.0, 0.0};
        force[axis] = config.fluid.forcingMagnitude;
        const std::string forceText =
            "(" + formatScientific(force[0], 2) + ", " +
            formatScientific(force[1], 2) + ", " +
            formatScientific(force[2], 2) + ")";

        out << std::left
            << std::setw(10) << (std::string("Axis ") + axisName(axis))
            << std::setw(38) << forceText
            << std::setw(12) << diagnostics.iterations
            << std::setw(16) << formatResidual(diagnostics.residual)
            << std::setw(12) << (diagnostics.converged ? "yes" : "no")
            << "\n";
    }

    out << "\nPermeability tensor K (scientific notation, 6 decimals)\n";
    out << "-------------------------------------------------------\n";
    out << std::left << std::setw(12) << ""
        << std::setw(18) << "X"
        << std::setw(18) << "Y"
        << std::setw(18) << "Z" << "\n";

    out << std::left << std::setw(12) << "X"
        << std::setw(18) << formatScientific(result.tensor[0], 6)
        << std::setw(18) << formatScientific(result.tensor[1], 6)
        << std::setw(18) << formatScientific(result.tensor[2], 6) << "\n";

    out << std::left << std::setw(12) << "Y"
        << std::setw(18) << formatScientific(result.tensor[3], 6)
        << std::setw(18) << formatScientific(result.tensor[4], 6)
        << std::setw(18) << formatScientific(result.tensor[5], 6) << "\n";

    out << std::left << std::setw(12) << "Z"
        << std::setw(18) << formatScientific(result.tensor[6], 6)
        << std::setw(18) << formatScientific(result.tensor[7], 6)
        << std::setw(18) << formatScientific(result.tensor[8], 6) << "\n";
}

void writeVelocityFieldBinary(
    const std::string& outputFile,
    const Geometry& geometry,
    const std::array<double, 3>& force,
    const SolveResult& result) {
    ensureParentDirectory(outputFile);
    std::ofstream out(outputFile, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open velocity output file: " + outputFile);
    }

    const char magic[4] = {'F', 'V', 'E', 'L'};
    const std::uint8_t version = 1;
    const std::uint8_t reserved8 = 0;
    const std::uint16_t reserved16 = 0;
    const std::uint64_t nx = static_cast<std::uint64_t>(geometry.nx);
    const std::uint64_t ny = static_cast<std::uint64_t>(geometry.ny);
    const std::uint64_t nz = static_cast<std::uint64_t>(geometry.nz);

    writeRaw(out, magic, sizeof(magic));
    writeRaw(out, &version, sizeof(version));
    writeRaw(out, &reserved8, sizeof(reserved8));
    writeRaw(out, &reserved16, sizeof(reserved16));
    writeRaw(out, &nx, sizeof(nx));
    writeRaw(out, &ny, sizeof(ny));
    writeRaw(out, &nz, sizeof(nz));
    writeRaw(out, &result.porosity, sizeof(result.porosity));
    writeRaw(out, &result.viscosity, sizeof(result.viscosity));
    writeRaw(out, force.data(), force.size() * sizeof(double));

    for (std::size_t i = 0; i < result.velocity.ux.size(); ++i) {
        const double xyz[3] = {
            result.velocity.ux[i],
            result.velocity.uy[i],
            result.velocity.uz[i]
        };
        writeRaw(out, xyz, sizeof(xyz));
    }
}

} // namespace common::fluid
