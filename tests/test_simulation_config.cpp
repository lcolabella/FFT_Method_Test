#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "app/SimulationConfig.hpp"

namespace {

std::string write_temp_cfg(const std::string& name, const std::string& content) {
    const std::string path = "/tmp/" + name;
    std::ofstream out(path, std::ios::trunc);
    assert(out.is_open());
    out << content;
    out.close();
    return path;
}

}  // namespace

int main() {
    const std::string cfg_path = write_temp_cfg(
        "fft_method_step8_config.cfg",
        "solver = brinkman\n"
        "threads = 4\n"
        "max_iterations = 321\n"
        "geometry_file = examples/sample_inputs/geometry.geom\n"
        "\n"
        "[bignonnet]\n"
        "fft_threads = 0\n"
        "\n"
        "[brinkman]\n"
        "tolerance = 1e-6\n");

    {
        std::string a0 = "fft_method";
        std::string a1 = "--config";
        std::string a2 = cfg_path;
        char* argv[] = {a0.data(), a1.data(), a2.data()};
        const int argc = 3;

        const permeability::SimulationConfig cfg =
            permeability::SimulationConfig::from_cli(argc, argv);

        assert(cfg.solver == "brinkman");
        assert(cfg.threads == 4);
        assert(cfg.fft_threads == 4);  // inherited from threads when fft_threads is 0
        assert(cfg.max_iterations == 321);
        assert(cfg.brinkman_tolerance == permeability::Scalar(1e-6));
    }

    {
        std::string a0 = "fft_method";
        std::string a1 = "--config";
        std::string a2 = cfg_path;
        std::string a3 = "--threads";
        std::string a4 = "6";
        std::string a5 = "--fft-threads";
        std::string a6 = "2";
        char* argv[] = {
            a0.data(), a1.data(), a2.data(), a3.data(), a4.data(), a5.data(), a6.data()};
        const int argc = 7;

        const permeability::SimulationConfig cfg =
            permeability::SimulationConfig::from_cli(argc, argv);

        assert(cfg.threads == 6);
        assert(cfg.fft_threads == 2);
    }

    std::remove(cfg_path.c_str());

    const std::string cfg_new_schema = write_temp_cfg(
        "fft_method_step9_new_schema.cfg",
        "solver = bignonnet\n"
        "\n"
        "[analysis]\n"
        "type = fluid\n"
        "\n"
        "[fluid]\n"
        "mode = pressure_gradient\n"
        "solver = brinkman\n"
        "compute_backend = gpu\n"
        "pressure_gradient = 0.0, 2.5, -1.0\n"
        "threads = 3\n"
        "fft_threads = 0\n"
        "write_velocity_fields = true\n"
        "max_iterations = 120\n"
        "tolerance = 2e-7\n"
        "\n"
        "[fluid.brinkman]\n"
        "tolerance = 3e-7\n"
        "forcing_magnitude = 4.0\n");

    {
        std::string a0 = "fft_method";
        std::string a1 = "--config";
        std::string a2 = cfg_new_schema;
        char* argv[] = {a0.data(), a1.data(), a2.data()};
        const int argc = 3;

        const permeability::SimulationConfig cfg =
            permeability::SimulationConfig::from_cli(argc, argv);

        assert(cfg.analysis == permeability::AnalysisType::Fluid);
        assert(cfg.fluid_mode == permeability::FluidAnalysisMode::PressureGradient);
        assert(cfg.solver == "brinkman");
        assert(cfg.compute_backend == permeability::ComputeBackend::GPU);
        assert(cfg.threads == 3);
        assert(cfg.fft_threads == 3);
        assert(cfg.write_velocity_fields);
        assert(cfg.max_iterations == 120);
        assert(cfg.tolerance == permeability::Scalar(2e-7));
        assert(cfg.brinkman_tolerance == permeability::Scalar(3e-7));
        assert(cfg.brinkman_forcing_magnitude == permeability::Scalar(4.0));
        assert(cfg.pressure_gradient[0] == permeability::Scalar(0));
        assert(cfg.pressure_gradient[1] == permeability::Scalar(2.5));
        assert(cfg.pressure_gradient[2] == permeability::Scalar(-1.0));
    }

    {
        std::string a0 = "fft_method";
        std::string a1 = "--config";
        std::string a2 = cfg_new_schema;
        std::string a3 = "--fluid-mode";
        std::string a4 = "permeability";
        std::string a5 = "--fluid-solver";
        std::string a6 = "bignonnet";
        std::string a7 = "--compute-backend";
        std::string a8 = "cpu";
        char* argv[] = {
            a0.data(), a1.data(), a2.data(), a3.data(), a4.data(), a5.data(), a6.data(),
            a7.data(), a8.data()};
        const int argc = 9;

        const permeability::SimulationConfig cfg =
            permeability::SimulationConfig::from_cli(argc, argv);

        assert(cfg.fluid_mode == permeability::FluidAnalysisMode::Permeability);
        assert(cfg.solver == "bignonnet");
        assert(cfg.compute_backend == permeability::ComputeBackend::CPU);
    }

    const std::string cfg_bad_pressure_gradient = write_temp_cfg(
        "fft_method_step9_bad_grad.cfg",
        "[analysis]\n"
        "type = fluid\n"
        "\n"
        "[fluid]\n"
        "mode = pressure_gradient\n"
        "pressure_gradient = 0, 0, 0\n");

    {
        bool threw = false;
        try {
            std::string a0 = "fft_method";
            std::string a1 = "--config";
            std::string a2 = cfg_bad_pressure_gradient;
            char* argv[] = {a0.data(), a1.data(), a2.data()};
            const int argc = 3;
            (void)permeability::SimulationConfig::from_cli(argc, argv);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    const std::string cfg_bad_analysis = write_temp_cfg(
        "fft_method_step9_bad_analysis.cfg",
        "[analysis]\n"
        "type = elastic\n");

    {
        bool threw = false;
        try {
            std::string a0 = "fft_method";
            std::string a1 = "--config";
            std::string a2 = cfg_bad_analysis;
            char* argv[] = {a0.data(), a1.data(), a2.data()};
            const int argc = 3;
            (void)permeability::SimulationConfig::from_cli(argc, argv);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    std::remove(cfg_new_schema.c_str());
    std::remove(cfg_bad_pressure_gradient.c_str());
    std::remove(cfg_bad_analysis.c_str());
    return 0;
}
