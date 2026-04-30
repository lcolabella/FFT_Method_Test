#include "solver/MinresSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace permeability {

namespace {

Scalar dot(const std::vector<Scalar>& a, const std::vector<Scalar>& b) {
    Scalar s = Scalar(0);
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for reduction(+:s)
#endif
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

Scalar norm2(const std::vector<Scalar>& v) {
    return std::sqrt(std::max(Scalar(0), dot(v, v)));
}

void axpy(Scalar a, const std::vector<Scalar>& x, std::vector<Scalar>& y) {
#ifdef PERMEABILITY_USE_OPENMP
#pragma omp parallel for
#endif
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += a * x[i];
    }
}

}  // namespace

MinresSolver::MinresSolver(Scalar tolerance, std::size_t max_iterations)
    : tolerance_(tolerance), max_iterations_(max_iterations) {
    if (tolerance_ <= Scalar(0)) {
        throw std::invalid_argument("MinresSolver tolerance must be positive");
    }
}

SolverResult MinresSolver::solve(const ILinearOperator& op,
                                 const std::vector<Scalar>& rhs,
                                 const std::vector<Scalar>& initial_guess,
                                 int progress_interval,
                                 ProgressCallback progress_callback) const {
    const std::size_t n = op.size();
    if (rhs.size() != n) {
        throw std::invalid_argument("MinresSolver::solve rhs size mismatch");
    }
    if (!initial_guess.empty() && initial_guess.size() != n) {
        throw std::invalid_argument("MinresSolver::solve initial guess size mismatch");
    }

    // Working vectors (allocated once, reused in all iterations).
    std::vector<Scalar> x(n, Scalar(0));
    std::vector<Scalar> v_old(n, Scalar(0));
    std::vector<Scalar> v(n, Scalar(0));
    std::vector<Scalar> z(n, Scalar(0));
    std::vector<Scalar> w_prev(n, Scalar(0));
    std::vector<Scalar> w(n, Scalar(0));
    std::vector<Scalar> w_next(n, Scalar(0));

    if (!initial_guess.empty()) {
        x = initial_guess;
    }

    SolverResult result;
    result.solution = x;

    const Scalar rhs_raw_norm = norm2(rhs);
    if (rhs_raw_norm == Scalar(0)) {
        result.converged = true;
        result.iterations = 0;
        result.final_rel_residual = Scalar(0);
        result.residual_history.push_back(Scalar(0));
        return result;
    }

    const Scalar rhs_norm = std::max(rhs_raw_norm, std::numeric_limits<Scalar>::min());

    // r0 = b - A*x0, stored in z then normalized to v.
    op.apply(x, z);
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = rhs[i] - z[i];
    }

    const Scalar beta1 = norm2(z);
    const Scalar rel0 = beta1 / rhs_norm;
    result.residual_history.push_back(rel0);
    result.final_rel_residual = rel0;

    if (rel0 <= tolerance_ || beta1 == Scalar(0)) {
        result.converged = true;
        result.iterations = 0;
        result.solution = x;
        return result;
    }

    for (std::size_t i = 0; i < n; ++i) {
        v[i] = z[i] / beta1;
    }
    std::fill(v_old.begin(), v_old.end(), Scalar(0));
    std::fill(w_prev.begin(), w_prev.end(), Scalar(0));
    std::fill(w.begin(), w.end(), Scalar(0));
    std::fill(w_next.begin(), w_next.end(), Scalar(0));

    Scalar beta = beta1;

    Scalar cs_old = Scalar(-1);
    Scalar sn_old = Scalar(0);
    Scalar cs = Scalar(-1);
    Scalar sn = Scalar(0);
    Scalar phi_bar = beta1;

    constexpr Scalar kBreakdownEps = static_cast<Scalar>(1e-14);
    Scalar rel_res = rel0;

    for (std::size_t k = 0; k < max_iterations_; ++k) {
        // 1) z = A*v
        op.apply(v, z);

        // 2) alpha = dot(v, z)
        const Scalar alpha = dot(v, z);

        // 3) z -= alpha*v; z -= beta*v_old
        for (std::size_t i = 0; i < n; ++i) {
            z[i] -= alpha * v[i] + beta * v_old[i];
        }

        // 4) beta_new = ||z||
        const Scalar beta_new = norm2(z);

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
        const Scalar eps = sn_old * beta;
        const Scalar delta_raw = -cs_old * beta;               // dbar in Paige-Saunders
        const Scalar delta     = cs * delta_raw + sn * alpha;  // off-diagonal entry of R
        const Scalar phi       = sn * delta_raw - cs * alpha;  // epln_bar: pre-rotation diagonal

        // 7) New Givens to eliminate beta_new.
        const Scalar rho = std::hypot(phi, beta_new);

        Scalar cs_new = Scalar(1);
        Scalar sn_new = Scalar(0);
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

        if (progress_interval > 0 && progress_callback &&
            (k + 1) % static_cast<std::size_t>(progress_interval) == 0) {
            progress_callback(k + 1, rel_res);
        }

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
