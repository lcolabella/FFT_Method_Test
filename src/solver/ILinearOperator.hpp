#pragma once

#include <cstddef>
#include <vector>

#include "core/Types.hpp"

namespace permeability {

class ILinearOperator {
public:
    virtual ~ILinearOperator() = default;

    [[nodiscard]] virtual std::size_t size() const = 0;
    virtual void apply(const std::vector<Scalar>& in, std::vector<Scalar>& out) const = 0;
};

}  // namespace permeability
