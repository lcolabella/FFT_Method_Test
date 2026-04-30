#include "post/Diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace permeability {

namespace {

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
    Scalar s = Scalar(0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

Scalar norm2(const std::vector<Scalar>& v) {
    return std::sqrt(std::max(Scalar(0), dot(v, v)));
}

}  // namespace

Scalar Diagnostics::residual_norm(const ILinearOperator& op,
                                  const std::vector<Scalar>& x,
                                  const std::vector<Scalar>& b) {
    if (x.size() != op.size() || b.size() != op.size()) {
        throw std::invalid_argument("Diagnostics::residual_norm size mismatch");
    }

    std::vector<Scalar> ax(op.size(), Scalar(0));
    op.apply(x, ax);

    Scalar s = Scalar(0);
    for (std::size_t i = 0; i < op.size(); ++i) {
        const Scalar r = ax[i] - b[i];
        s += r * r;
    }
    return std::sqrt(s);
}

Scalar Diagnostics::relative_symmetry_gap(const ILinearOperator& op,
                                          const std::vector<Scalar>& x,
                                          const std::vector<Scalar>& y) {
    if (x.size() != op.size() || y.size() != op.size()) {
        throw std::invalid_argument("Diagnostics::relative_symmetry_gap size mismatch");
    }

    std::vector<Scalar> ax(op.size(), Scalar(0));
    std::vector<Scalar> ay(op.size(), Scalar(0));
    op.apply(x, ax);
    op.apply(y, ay);

    Scalar xay = Scalar(0);
    Scalar axy = Scalar(0);
    for (std::size_t i = 0; i < op.size(); ++i) {
        xay += x[i] * ay[i];
        axy += ax[i] * y[i];
    }
    const Scalar denom = std::max({Scalar(1), std::abs(xay), std::abs(axy), norm2(x) * norm2(y)});
    return std::abs(xay - axy) / denom;
}

Scalar Diagnostics::mean_component(const std::vector<Scalar>& compact_vec, std::size_t component) {
    if (component >= 3 || compact_vec.size() % 3 != 0) {
        throw std::invalid_argument("Diagnostics::mean_component invalid compact vector layout");
    }

    const std::size_t n = compact_vec.size() / 3;
    if (n == 0) {
        return Scalar(0);
    }

    Scalar s = Scalar(0);
    for (std::size_t i = 0; i < n; ++i) {
        s += compact_vec[3 * i + component];
    }
    return s / static_cast<Scalar>(n);
}

bool Diagnostics::is_zero_mean_compact(const std::vector<Scalar>& compact_vec, Scalar tol) {
    if (tol < Scalar(0)) {
        throw std::invalid_argument("Diagnostics::is_zero_mean_compact requires non-negative tolerance");
    }
    if (compact_vec.empty()) {
        return true;
    }
    if (compact_vec.size() % 3 != 0) {
        throw std::invalid_argument("Diagnostics::is_zero_mean_compact invalid compact layout");
    }
    const Scalar m0 = std::abs(mean_component(compact_vec, 0));
    const Scalar m1 = std::abs(mean_component(compact_vec, 1));
    const Scalar m2 = std::abs(mean_component(compact_vec, 2));
    const Scalar lim = std::max(tol, Scalar(10) * std::numeric_limits<Scalar>::epsilon());
    return m0 <= lim && m1 <= lim && m2 <= lim;
}

}  // namespace permeability
