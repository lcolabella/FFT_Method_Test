#include <exception>
#include <iostream>

#include "app/PermeabilityRunner.hpp"
#include "app/SimulationConfig.hpp"

int main(int argc, char** argv) {
    try {
        permeability::SimulationConfig cfg = permeability::SimulationConfig::from_cli(argc, argv);
        permeability::PermeabilityRunner runner(cfg);
        return runner.run();
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << "\n";
        std::cerr << "Usage: fft_method --input <path> [--support interface|fullsolid] [--mu <value>] "
                  << "[--voxel-size <value>] [--tol <value>] [--maxit <value>] [--threads <value>] [--omp-threads <value>] [--p-radius <value>] "
                  << "[--gradient x|y|z|all] [--parallel-load-cases]\n";
        return 1;
    }
}
