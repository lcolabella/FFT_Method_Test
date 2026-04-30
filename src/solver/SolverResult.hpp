#pragma once

#include <cstddef>
#include <vector>

#include "core/Types.hpp"

namespace permeability {

struct SolverResult {
    std::vector<Scalar> solution;
    bool converged{false};
    std::size_t iterations{0};
    Scalar final_rel_residual{Scalar(1)};
    std::vector<Scalar> residual_history;
};

}  // namespace permeability
