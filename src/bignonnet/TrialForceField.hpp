#pragma once

#include <cstddef>
#include <vector>

#include "bignonnet/ForceSupport.hpp"
#include "core/Field3D.hpp"
#include "core/Types.hpp"

namespace permeability {

class TrialForceField {
public:
    explicit TrialForceField(std::size_t num_active_voxels);

    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] std::size_t num_active_voxels() const noexcept { return nb_; }

    [[nodiscard]] double& x(std::size_t local);
    [[nodiscard]] double& y(std::size_t local);
    [[nodiscard]] double& z(std::size_t local);

    [[nodiscard]] const double& x(std::size_t local) const;
    [[nodiscard]] const double& y(std::size_t local) const;
    [[nodiscard]] const double& z(std::size_t local) const;

    [[nodiscard]] double& operator()(std::size_t local, std::size_t comp);
    [[nodiscard]] const double& operator()(std::size_t local, std::size_t comp) const;

    void subtract_mean();
    [[nodiscard]] Real3 mean_vector() const;
    [[nodiscard]] double norm2() const;
    void fill_zero();

    void scatter_to_full(VectorField3D& out_full, const ForceSupport& support) const;
    void gather_from_full(const VectorField3D& in_full, const ForceSupport& support);

    [[nodiscard]] const std::vector<double>& raw() const noexcept { return values_; }
    [[nodiscard]] std::vector<double>& raw() noexcept { return values_; }

private:
    std::size_t nb_;
    std::vector<double> values_;
};

}  // namespace permeability
