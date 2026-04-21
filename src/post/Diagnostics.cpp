#include "post/Diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace permeability {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm2(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

}  // namespace

double Diagnostics::residual_norm(const ILinearOperator& op,
                                  const std::vector<double>& x,
                                  const std::vector<double>& b) {
    if (x.size() != op.size() || b.size() != op.size()) {
        throw std::invalid_argument("Diagnostics::residual_norm size mismatch");
    }

    std::vector<double> ax(op.size(), 0.0);
    op.apply(x, ax);

    double s = 0.0;
    for (std::size_t i = 0; i < op.size(); ++i) {
        const double r = ax[i] - b[i];
        s += r * r;
    }
    return std::sqrt(s);
}

double Diagnostics::relative_symmetry_gap(const ILinearOperator& op,
                                          const std::vector<double>& x,
                                          const std::vector<double>& y) {
    if (x.size() != op.size() || y.size() != op.size()) {
        throw std::invalid_argument("Diagnostics::relative_symmetry_gap size mismatch");
    }

    std::vector<double> ax(op.size(), 0.0);
    std::vector<double> ay(op.size(), 0.0);
    op.apply(x, ax);
    op.apply(y, ay);

    double xay = 0.0;
    double axy = 0.0;
    for (std::size_t i = 0; i < op.size(); ++i) {
        xay += x[i] * ay[i];
        axy += ax[i] * y[i];
    }
    const double denom = std::max({1.0, std::abs(xay), std::abs(axy), norm2(x) * norm2(y)});
    return std::abs(xay - axy) / denom;
}

double Diagnostics::mean_component(const std::vector<double>& compact_vec, std::size_t component) {
    if (component >= 3 || compact_vec.size() % 3 != 0) {
        throw std::invalid_argument("Diagnostics::mean_component invalid compact vector layout");
    }

    const std::size_t n = compact_vec.size() / 3;
    if (n == 0) {
        return 0.0;
    }

    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += compact_vec[3 * i + component];
    }
    return s / static_cast<double>(n);
}

bool Diagnostics::is_zero_mean_compact(const std::vector<double>& compact_vec, double tol) {
    if (tol < 0.0) {
        throw std::invalid_argument("Diagnostics::is_zero_mean_compact requires non-negative tolerance");
    }
    if (compact_vec.empty()) {
        return true;
    }
    if (compact_vec.size() % 3 != 0) {
        throw std::invalid_argument("Diagnostics::is_zero_mean_compact invalid compact layout");
    }
    const double m0 = std::abs(mean_component(compact_vec, 0));
    const double m1 = std::abs(mean_component(compact_vec, 1));
    const double m2 = std::abs(mean_component(compact_vec, 2));
    const double lim = std::max(tol, 10.0 * std::numeric_limits<double>::epsilon());
    return m0 <= lim && m1 <= lim && m2 <= lim;
}

}  // namespace permeability
