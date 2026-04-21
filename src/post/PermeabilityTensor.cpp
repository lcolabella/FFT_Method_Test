#include "post/PermeabilityTensor.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace permeability {

PermeabilityTensor::PermeabilityTensor() : k_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} {}

void PermeabilityTensor::set_column(std::size_t column, const Real3& avg_velocity) {
    if (column >= 3) {
        throw std::out_of_range("PermeabilityTensor::set_column index out of range");
    }
    for (std::size_t row = 0; row < 3; ++row) {
        k_[row * 3 + column] = -avg_velocity[row];
    }
}

std::string PermeabilityTensor::to_string() const {
    std::ostringstream oss;
    oss << std::setprecision(10);
    for (std::size_t r = 0; r < 3; ++r) {
        oss << k_[3 * r + 0] << " " << k_[3 * r + 1] << " " << k_[3 * r + 2];
        if (r + 1 < 3) {
            oss << "\n";
        }
    }
    return oss.str();
}

}  // namespace permeability
