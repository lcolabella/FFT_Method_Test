#pragma once

#include <cstddef>
#include <vector>

namespace permeability {

class ILinearOperator {
public:
    virtual ~ILinearOperator() = default;

    [[nodiscard]] virtual std::size_t size() const = 0;
    virtual void apply(const std::vector<double>& in, std::vector<double>& out) const = 0;
};

}  // namespace permeability
