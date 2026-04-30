#include <exception>
#include <iostream>
#include <string>

#include "app/PermeabilityRunner.hpp"
#include "app/SimulationConfig.hpp"
#include "logging/Logger.hpp"

namespace {

std::string find_cli_value(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] != nullptr && std::string(argv[i]) == flag) {
            if (i + 1 < argc && argv[i + 1] != nullptr) {
                return argv[i + 1];
            }
            break;
        }
    }
    return {};
}

const char* analysis_name(permeability::AnalysisType analysis) {
    switch (analysis) {
        case permeability::AnalysisType::Fluid:
            return "fluid";
        case permeability::AnalysisType::Elastic:
            return "elastic";
    }
    return "unknown";
}

const char* fluid_mode_name(permeability::FluidAnalysisMode mode) {
    switch (mode) {
        case permeability::FluidAnalysisMode::Permeability:
            return "permeability";
        case permeability::FluidAnalysisMode::PressureGradient:
            return "pressure_gradient";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string cli_log_file = find_cli_value(argc, argv, "--log");
        if (!cli_log_file.empty()) {
            common::Logger::instance().open(cli_log_file);
        }

        auto& log = common::Logger::instance();
        log.info("============================================================");
        log.info("FFT_Method startup");
        log.info("Parsing command line and configuration");

        permeability::SimulationConfig cfg = permeability::SimulationConfig::from_cli(argc, argv);

        if (!cfg.log_file.empty()) {
            if (cfg.log_file != cli_log_file) {
                log.open(cfg.log_file);
                log.info("Log file opened from configuration: " + cfg.log_file);
            }
        }

        log.info(std::string("Analysis path: ") + analysis_name(cfg.analysis));
        if (cfg.analysis == permeability::AnalysisType::Fluid) {
            log.info("Fluid path: solver=" + cfg.solver +
                     " mode=" + fluid_mode_name(cfg.fluid_mode));
            log.info("Compute backend: " + std::string(
                cfg.compute_backend == permeability::ComputeBackend::GPU ? "gpu" : "cpu"));
        }
        log.info("Initializing runner");

        permeability::PermeabilityRunner runner(cfg);
        return runner.run();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        std::cerr << "Usage: fft_method [--config <file>] [--input <geometry>] [--material-file <file>] "
                  << "[--output <file>] [--log <file>] "
                  << "[--analysis fluid] [--fluid-mode permeability|pressure_gradient] [--fluid-solver bignonnet|brinkman] "
                  << "[--compute-backend cpu|gpu] "
                  << "[--pressure-gradient <gx,gy,gz>] "
                  << "[--write-velocity-fields] "
                  << "[--support interface|fullsolid] [--mu <value>] "
                  << "[--voxel-size <value>] [--tol <value>] [--maxit <value>] [--threads <value>] [--fft-threads <value>] [--p-radius <value>] "
                  << "[--gradient x|y|z|all] [--parallel-load-cases]\n";
        return 1;
    }
}
