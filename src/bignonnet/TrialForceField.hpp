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

    [[nodiscard]] Scalar& x(std::size_t local);
    [[nodiscard]] Scalar& y(std::size_t local);
    [[nodiscard]] Scalar& z(std::size_t local);

    [[nodiscard]] const Scalar& x(std::size_t local) const;
    [[nodiscard]] const Scalar& y(std::size_t local) const;
    [[nodiscard]] const Scalar& z(std::size_t local) const;

    [[nodiscard]] Scalar& operator()(std::size_t local, std::size_t comp);
    [[nodiscard]] const Scalar& operator()(std::size_t local, std::size_t comp) const;

    void subtract_mean();
    [[nodiscard]] Real3 mean_vector() const;
    [[nodiscard]] Scalar norm2() const;
    void fill_zero();

    void scatter_to_full(VectorField3D& out_full, const ForceSupport& support) const;
    void gather_from_full(const VectorField3D& in_full, const ForceSupport& support);

    [[nodiscard]] const std::vector<Scalar>& raw() const noexcept { return values_; }
    [[nodiscard]] std::vector<Scalar>& raw() noexcept { return values_; }

private:
    std::size_t nb_;
    std::vector<Scalar> values_;
};

}  // namespace permeability
