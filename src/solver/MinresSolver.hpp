#pragma once

#include <cstddef>
#include <vector>

#include "solver/ILinearOperator.hpp"
#include "solver/SolverResult.hpp"

namespace permeability {

class MinresSolver {
public:
    // MINRES for symmetric (possibly semidefinite) matrix-free operators.
    MinresSolver(double tolerance, std::size_t max_iterations);

    [[nodiscard]] SolverResult solve(const ILinearOperator& op,
                                     const std::vector<double>& rhs,
                                     const std::vector<double>& initial_guess = {}) const;

private:
    double tolerance_;
    std::size_t max_iterations_;
};

}  // namespace permeability
