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
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double norm2(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void axpy(double a, const std::vector<double>& x, std::vector<double>& y) {
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += a * x[i];
    }
}

std::vector<double> solve_dense_system(std::vector<double> a,
                                       std::vector<double> b,
                                       std::size_t n) {
    const double eps = 1e-14;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double best = std::abs(a[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double cand = std::abs(a[i * n + k]);
            if (cand > best) {
                best = cand;
                pivot = i;
            }
        }

        if (best < eps) {
            a[k * n + k] += eps;
            best = std::abs(a[k * n + k]);
            if (best < eps) {
                throw std::runtime_error("MINRES normal equation is singular");
            }
        }

        if (pivot != k) {
            for (std::size_t j = k; j < n; ++j) {
                std::swap(a[k * n + j], a[pivot * n + j]);
            }
            std::swap(b[k], b[pivot]);
        }

        const double diag = a[k * n + k];
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = a[i * n + k] / diag;
            a[i * n + k] = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j) {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
    return x;
}

std::vector<double> minres_projected_step(const std::vector<double>& alphas,
                                          const std::vector<double>& betas,
                                          double beta1) {
    const std::size_t k = alphas.size();
    std::vector<double> normal(k * k, 0.0);
    std::vector<double> rhs(k, 0.0);

    for (std::size_t i = 0; i < k; ++i) {
        const double d = alphas[i];
        const double bd = (i > 0 ? betas[i - 1] : 0.0);
        const double bu = (i + 1 < k ? betas[i] : 0.0);

        normal[i * k + i] += d * d + bd * bd + bu * bu;
        if (i > 0) {
            const double v = d * bd;
            normal[i * k + (i - 1)] += v;
            normal[(i - 1) * k + i] += v;
        }
        if (i + 1 < k) {
            const double v = d * bu;
            normal[i * k + (i + 1)] += v;
            normal[(i + 1) * k + i] += v;
        }
        if (i + 2 < k) {
            const double v = bu * betas[i + 1];
            normal[i * k + (i + 2)] += v;
            normal[(i + 2) * k + i] += v;
        }
    }

    rhs[0] = beta1 * alphas[0];
    if (k > 1) {
        rhs[1] = beta1 * betas[0];
    }

    return solve_dense_system(std::move(normal), std::move(rhs), k);
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
    if (rhs.size() != op.size()) {
        throw std::invalid_argument("MinresSolver::solve rhs size mismatch");
    }

    SolverResult result;
    result.solution.assign(op.size(), 0.0);
    if (!initial_guess.empty()) {
        if (initial_guess.size() != op.size()) {
            throw std::invalid_argument("MinresSolver::solve initial guess size mismatch");
        }
        result.solution = initial_guess;
    }

    const std::size_t n = op.size();
    std::vector<double> ax(n, 0.0);
    op.apply(result.solution, ax);

    std::vector<double> r(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = rhs[i] - ax[i];
    }

    const double rhs_norm = std::max(norm2(rhs), std::numeric_limits<double>::min());
    const double beta1 = norm2(r);
    const double rel0 = beta1 / rhs_norm;
    result.residual_history.push_back(rel0);
    result.final_rel_residual = rel0;

    if (rel0 <= tolerance_) {
        result.converged = true;
        result.iterations = 0;
        return result;
    }

    std::vector<std::vector<double>> basis;
    basis.reserve(max_iterations_ + 1);
    std::vector<double> v0(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        v0[i] = r[i] / beta1;
    }
    basis.push_back(v0);

    std::vector<double> alphas;
    std::vector<double> betas;
    alphas.reserve(max_iterations_);
    betas.reserve(max_iterations_);

    std::vector<double> v_prev(n, 0.0);
    double beta_prev = 0.0;

    for (std::size_t it = 0; it < max_iterations_; ++it) {
        const std::vector<double>& v = basis[it];
        std::vector<double> w(n, 0.0);
        op.apply(v, w);

        if (it > 0) {
            axpy(-beta_prev, v_prev, w);
        }

        const double alpha = dot(v, w);
        alphas.push_back(alpha);
        axpy(-alpha, v, w);

        const double beta = norm2(w);
        if (it + 1 < max_iterations_) {
            betas.push_back(beta);
        }

        if (beta > 0.0 && it + 1 < max_iterations_) {
            std::vector<double> v_next(n, 0.0);
            for (std::size_t i = 0; i < n; ++i) {
                v_next[i] = w[i] / beta;
            }
            basis.push_back(std::move(v_next));
        }

        const std::size_t k = it + 1;
        const std::vector<double> y = minres_projected_step(alphas, betas, beta1);

        result.solution = initial_guess.empty() ? std::vector<double>(n, 0.0) : initial_guess;
        for (std::size_t j = 0; j < k; ++j) {
            axpy(y[j], basis[j], result.solution);
        }

        op.apply(result.solution, ax);
        for (std::size_t i = 0; i < n; ++i) {
            r[i] = rhs[i] - ax[i];
        }

        const double rel = norm2(r) / rhs_norm;
        result.residual_history.push_back(rel);
        result.final_rel_residual = rel;
        result.iterations = k;

        if (rel <= tolerance_) {
            result.converged = true;
            return result;
        }

        if (beta <= std::numeric_limits<double>::epsilon()) {
            result.converged = rel <= tolerance_;
            return result;
        }

        v_prev = v;
        beta_prev = beta;
    }

    result.converged = false;
    return result;
}

}  // namespace permeability
