#include "solver/MinresSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace permeability {

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm2(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += a * x[i];
    }
}

}  // namespace

MinresSolver::MinresSolver(double tolerance, std::size_t max_iterations)
    : tolerance_(tolerance), max_iterations_(max_iterations) {
    if (tolerance_ <= 0.0) {
        throw std::invalid_argument("MinresSolver tolerance must be positive");
    }
}

SolverResult MinresSolver::solve(const ILinearOperator& op,
                                 const std::vector<double>& rhs,
                                 const std::vector<double>& initial_guess) const {
    const std::size_t n = op.size();
    if (rhs.size() != n) {
        throw std::invalid_argument("MinresSolver::solve rhs size mismatch");
    }
    if (!initial_guess.empty() && initial_guess.size() != n) {
        throw std::invalid_argument("MinresSolver::solve initial guess size mismatch");
    }

    // Working vectors (allocated once, reused in all iterations).
    std::vector<double> x(n, 0.0);
    std::vector<double> v_old(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> z(n, 0.0);
    std::vector<double> w_prev(n, 0.0);
    std::vector<double> w(n, 0.0);
    std::vector<double> w_next(n, 0.0);

    if (!initial_guess.empty()) {
        x = initial_guess;
    }

    SolverResult result;
    result.solution = x;

    const double rhs_raw_norm = norm2(rhs);
    if (rhs_raw_norm == 0.0) {
        result.converged = true;
        result.iterations = 0;
        result.final_rel_residual = 0.0;
        result.residual_history.push_back(0.0);
        return result;
    }

    const double rhs_norm = std::max(rhs_raw_norm, std::numeric_limits<double>::min());

    // r0 = b - A*x0, stored in z then normalized to v.
    op.apply(x, z);
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = rhs[i] - z[i];
    }

    const double beta1 = norm2(z);
    const double rel0 = beta1 / rhs_norm;
    result.residual_history.push_back(rel0);
    result.final_rel_residual = rel0;

    if (rel0 <= tolerance_ || beta1 == 0.0) {
        result.converged = true;
        result.iterations = 0;
        result.solution = x;
        return result;
    }

    for (std::size_t i = 0; i < n; ++i) {
        v[i] = z[i] / beta1;
    }
    std::fill(v_old.begin(), v_old.end(), 0.0);
    std::fill(w_prev.begin(), w_prev.end(), 0.0);
    std::fill(w.begin(), w.end(), 0.0);
    std::fill(w_next.begin(), w_next.end(), 0.0);

    double beta = beta1;

    double cs_old = -1.0;
    double sn_old = 0.0;
    double cs = -1.0;
    double sn = 0.0;
    double phi_bar = beta1;

    constexpr double kBreakdownEps = 1e-14;
    double rel_res = rel0;

    for (std::size_t k = 0; k < max_iterations_; ++k) {
        // 1) z = A*v
        op.apply(v, z);

        // 2) alpha = dot(v, z)
        const double alpha = dot(v, z);

        // 3) z -= alpha*v; z -= beta*v_old
        for (std::size_t i = 0; i < n; ++i) {
            z[i] -= alpha * v[i] + beta * v_old[i];
        }

        // 4) beta_new = ||z||
        const double beta_new = norm2(z);

        // 5) v_old = v; v = z / beta_new
        v_old = v;
        if (beta_new > kBreakdownEps) {
            for (std::size_t i = 0; i < n; ++i) {
                v[i] = z[i] / beta_new;
            }
        } else {
            std::fill(v.begin(), v.end(), 0.0);
        }

        // 6) Apply old/current Givens.
        const double eps = sn_old * beta;
        const double delta_raw = -cs_old * beta;               // dbar in Paige-Saunders
        const double delta     = cs * delta_raw + sn * alpha;  // off-diagonal entry of R
        const double phi       = sn * delta_raw - cs * alpha;  // epln_bar: pre-rotation diagonal

        // 7) New Givens to eliminate beta_new.
        const double rho = std::hypot(phi, beta_new);

        double cs_new = 1.0;
        double sn_new = 0.0;
        if (rho > kBreakdownEps) {
            cs_new = phi / rho;
            sn_new = beta_new / rho;
        }

        // 8) w_next = (v_old - eps*w_prev - delta*w)/rho
        if (rho > kBreakdownEps) {
            for (std::size_t i = 0; i < n; ++i) {
                w_next[i] = (v_old[i] - eps * w_prev[i] - delta * w[i]) / rho;
            }
        } else {
            std::fill(w_next.begin(), w_next.end(), 0.0);
        }

        // 9) x += (cs_new * phi_bar) * w_next
        axpy(cs_new * phi_bar, w_next, x);

        // 10) phi_bar *= sn_new
        phi_bar *= sn_new;

        // 11) Shift recurrence state.
        w_prev = w;
        w = w_next;
        cs_old = cs;
        sn_old = sn;
        cs = cs_new;
        sn = sn_new;
        beta = beta_new;

        // 12) Relative residual estimate.
        rel_res = std::abs(phi_bar) / rhs_norm;
        result.residual_history.push_back(rel_res);
        result.final_rel_residual = rel_res;
        result.iterations = k + 1;

        if (rel_res <= tolerance_) {
            result.converged = true;
            break;
        }

        // Invariant subspace reached.
        if (beta_new < kBreakdownEps) {
            result.converged = (rel_res <= tolerance_);
            break;
        }

    }

    if (!result.converged) {
        result.converged = (result.final_rel_residual <= tolerance_);
    }

    result.solution = x;
    return result;
}

}  // namespace permeability
