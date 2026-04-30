#pragma once

#include <vector>

#include "core/Types.hpp"
#include "solver/ILinearOperator.hpp"

namespace permeability {

class Diagnostics {
public:
    static Scalar residual_norm(const ILinearOperator& op,
                                const std::vector<Scalar>& x,
                                const std::vector<Scalar>& b);

    static Scalar relative_symmetry_gap(const ILinearOperator& op,
                                        const std::vector<Scalar>& x,
                                        const std::vector<Scalar>& y);

    static Scalar mean_component(const std::vector<Scalar>& compact_vec, std::size_t component);
    static bool is_zero_mean_compact(const std::vector<Scalar>& compact_vec, Scalar tol);
};

}  // namespace permeability
