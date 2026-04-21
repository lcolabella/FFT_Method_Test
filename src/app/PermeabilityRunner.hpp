#pragma once

#include "app/SimulationConfig.hpp"

namespace permeability {

class PermeabilityRunner {
public:
    explicit PermeabilityRunner(SimulationConfig config);
    int run();

private:
    SimulationConfig config_;
};

}  // namespace permeability
