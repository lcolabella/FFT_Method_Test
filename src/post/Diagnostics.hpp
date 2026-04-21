#pragma once

#include <vector>

#include "solver/ILinearOperator.hpp"

namespace permeability {

class Diagnostics {
public:
    static double residual_norm(const ILinearOperator& op,
                                const std::vector<double>& x,
                                const std::vector<double>& b);

    static double relative_symmetry_gap(const ILinearOperator& op,
                                        const std::vector<double>& x,
                                        const std::vector<double>& y);

    static double mean_component(const std::vector<double>& compact_vec, std::size_t component);
    static bool is_zero_mean_compact(const std::vector<double>& compact_vec, double tol);
};

}  // namespace permeability
