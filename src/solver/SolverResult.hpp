#pragma once

#include <cstddef>
#include <vector>

namespace permeability {

struct SolverResult {
    std::vector<double> solution;
    bool converged{false};
    std::size_t iterations{0};
    double final_rel_residual{1.0};
    std::vector<double> residual_history;
};

}  // namespace permeability
