#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "app/config/Config.hpp"
#include "core/Geometry.hpp"
#include "materials/MaterialDatabase.hpp"

namespace common::fluid {

struct SolveDiagnostics {
    int iterations = 0;
    double residual = 0.0;
    bool converged = false;
    double elapsedSeconds = 0.0;
};

struct VelocityField {
    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<double> uz;
};

struct SolveResult {
    VelocityField velocity;
    std::array<double, 3> superficialVelocity{0.0, 0.0, 0.0};
    double porosity = 0.0;
    SolveDiagnostics diagnostics;
    std::array<SolveDiagnostics, 3> componentDiagnostics{};
    double viscosity = 1.0;
};

struct PermeabilityResult {
    std::array<double, 9> tensor{};
    std::array<SolveDiagnostics, 3> loadDiagnostics{};
    double porosity = 0.0;
    double viscosity = 1.0;
    double elapsedSeconds = 0.0;
};

SolveResult solveBrinkman(
    const AppConfig& config,
    const Geometry& geometry,
    const MaterialDatabase& materialDb,
    const std::array<double, 3>& force);

PermeabilityResult computePermeabilityTensor(
    const AppConfig& config,
    const Geometry& geometry,
    const MaterialDatabase& materialDb,
    const std::function<void(std::size_t, const std::array<double, 3>&, const SolveResult&)>& onLoadCaseSolved = {});

void writePermeabilityTensorText(
    const AppConfig& config,
    const MaterialDatabase& materialDb,
    const std::string& outputFile,
    const Geometry& geometry,
    const PermeabilityResult& result);

void writeVelocityFieldBinary(
    const std::string& outputFile,
    const Geometry& geometry,
    const std::array<double, 3>& force,
    const SolveResult& result);

} // namespace common::fluid
