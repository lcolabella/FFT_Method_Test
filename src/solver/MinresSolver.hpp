#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "core/Types.hpp"
#include "solver/ILinearOperator.hpp"
#include "solver/SolverResult.hpp"

namespace permeability {

class MinresSolver {
public:
    // Callback invoked every progress_interval iterations: (iteration, relative_residual).
    using ProgressCallback = std::function<void(std::size_t, Scalar)>;

    // MINRES for symmetric (possibly semidefinite) matrix-free operators.
    MinresSolver(Scalar tolerance, std::size_t max_iterations);

    [[nodiscard]] SolverResult solve(const ILinearOperator& op,
                                     const std::vector<Scalar>& rhs,
                                     const std::vector<Scalar>& initial_guess = {},
                                     int progress_interval = 0,
                                     ProgressCallback progress_callback = {}) const;

private:
    Scalar tolerance_;
    std::size_t max_iterations_;
};

}  // namespace permeability
