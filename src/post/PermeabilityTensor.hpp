#pragma once

#include <array>
#include <string>

#include "core/Types.hpp"

namespace permeability {

class PermeabilityTensor {
public:
    PermeabilityTensor();

    // Darcy sign convention: V = -K * grad(P), so K column j is -<v^(j)>.
    void set_column(std::size_t column, const Real3& avg_velocity);
    [[nodiscard]] const std::array<Scalar, 9>& data() const noexcept { return k_; }
    [[nodiscard]] std::string to_string(const std::string& indent = "") const;

private:
    std::array<Scalar, 9> k_;
};

}  // namespace permeability
