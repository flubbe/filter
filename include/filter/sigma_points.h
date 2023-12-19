/**
 * filter - some header-only C++ filter implementations.
 *
 * sigma point generation.
 *
 * Reference:
 *     FilterPy, https://github.com/rlabbe/filterpy,
 *               filterpy/kalman/sigma_points.py
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include <functional>
#include <exception>
#include <cmath>

#include <fmt/core.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "filter/types.h"
#include "filter/functions.h"

namespace filter::ukf
{

class MerweScaledSigmaPoints
{
    std::uint32_t n;
    float alpha;
    float beta;
    float kappa;

    VectorType Wc;
    VectorType Wm;

    std::function<MatrixType(const MatrixType&)> sqrt_fn;
    std::function<VectorType(const VectorType&, const VectorType&)> subtract_fn;

    void compute_weights()
    {
        float lambda = alpha * alpha * (n + kappa) - n;
        float c = .5 / (n + lambda);

        Wc = VectorType{2 * n + 1};
        Wc.fill(c);

        Wm = VectorType{2 * n + 1};
        Wm.fill(c);

        Wc[0] = lambda / (n + lambda) + (1 - alpha * alpha + beta);
        Wm[0] = lambda / (n + lambda);
    }

public:
    MerweScaledSigmaPoints(
      std::uint32_t in_n,
      float in_alpha,
      float in_beta,
      float in_kappa,
      std::function<MatrixType(const MatrixType&)> in_sqrt_fn = cholesky,
      std::function<VectorType(const VectorType&, const VectorType&)> in_subtract_fn = subtract)
    : n{in_n}
    , alpha{in_alpha}
    , beta{in_beta}
    , kappa{in_kappa}
    , sqrt_fn{in_sqrt_fn}
    , subtract_fn{in_subtract_fn}
    {
        compute_weights();
    }

    MatrixType sigma_points(const VectorType& x, const MatrixType& P) const
    {
        if(x.size() != n)
        {
            throw std::runtime_error(
              fmt::format("expected size(x) {}, but size is {}",
                          n, x.size()));
        }

        float lambda = alpha * alpha * (n + kappa) - n;
        MatrixType U = sqrt_fn((lambda + n) * P);

        MatrixType sigmas = MatrixType::Zero(2 * n + 1, n);
        sigmas.row(0) = x;

        for(std::uint32_t k = 0; k < n; ++k)
        {
            sigmas.row(k + 1) = subtract_fn(x, -U.row(k));
            sigmas.row(n + k + 1) = subtract_fn(x, U.row(k));
        }

        return sigmas;
    }

    VectorType get_Wm() const
    {
        return Wm;
    }

    VectorType get_Wc() const
    {
        return Wc;
    }

    std::uint32_t get_num_sigmas() const
    {
        return 2 * n + 1;
    }
};

class JulierSigmaPoints
{
    std::uint32_t n;
    float kappa;

    VectorType Wc;
    VectorType Wm;

    std::function<MatrixType(const MatrixType&)> sqrt_fn;
    std::function<VectorType(const VectorType&, const VectorType&)> subtract_fn;

    void compute_weights()
    {
        Wm = VectorType{2 * n + 1};
        Wm.fill(0.5 / (n + kappa));
        Wm[0] = kappa / (n + kappa);
        Wc = Wm;
    }

public:
    JulierSigmaPoints(
      std::uint32_t in_n,
      float in_kappa,
      std::function<MatrixType(const MatrixType&)> in_sqrt_fn = cholesky,
      std::function<VectorType(const VectorType&, const VectorType&)> in_subtract_fn = subtract)
    : n{in_n}
    , kappa{in_kappa}
    , sqrt_fn{in_sqrt_fn}
    , subtract_fn{in_subtract_fn}
    {
        compute_weights();
    }

    MatrixType sigma_points(const VectorType& x, const MatrixType& P) const
    {
        if(x.size() != n)
        {
            throw std::runtime_error(
              fmt::format("expected size(x) {}, but size is {}",
                          n, x.size()));
        }

        MatrixType sigmas = MatrixType::Zero(2 * n + 1, n);
        MatrixType U = sqrt_fn((n + kappa) * P);
        sigmas.row(0) = x;

        for(std::uint32_t k = 0; k < n; ++k)
        {
            sigmas.row(k + 1) = subtract_fn(x, -U.row(k));
            sigmas.row(n + k + 1) = subtract_fn(x, U.row(k));
        }

        return sigmas;
    }

    VectorType get_Wm() const
    {
        return Wm;
    }

    VectorType get_Wc() const
    {
        return Wc;
    }

    std::uint32_t get_num_sigmas() const
    {
        return 2 * n + 1;
    }
};

class SimplexSigmaPoints
{
    std::uint32_t n;
    float alpha;

    VectorType Wc;
    VectorType Wm;

    std::function<MatrixType(const MatrixType&)> sqrt_fn;
    std::function<VectorType(const VectorType&, const VectorType&)> subtract_fn;

    void compute_weights()
    {
        float c = 1.f / (n + 1);
        Wm = VectorType{n + 1};
        Wm.fill(c);
        Wc = Wm;
    }

public:
    SimplexSigmaPoints(
      std::uint32_t in_n,
      float in_alpha = 1,
      std::function<MatrixType(const MatrixType&)> in_sqrt_fn = cholesky,
      std::function<VectorType(const VectorType&, const VectorType&)> in_subtract_fn = subtract)
    : n{in_n}
    , alpha{in_alpha}
    , sqrt_fn{in_sqrt_fn}
    , subtract_fn{in_subtract_fn}
    {
        compute_weights();
    }

    MatrixType sigma_points(const VectorType& x, const MatrixType& P) const
    {
        if(x.size() != n)
        {
            throw std::runtime_error(
              fmt::format("expected size(x) {}, but size is {}",
                          n, x.size()));
        }

        MatrixType U = sqrt_fn(P);

        float lambda = static_cast<float>(n) / (n + 1);
        MatrixType Istar = (MatrixType{1, 2} << -1.f / std::sqrt(2 * lambda), 1.f / std::sqrt(2 * lambda)).finished();

        for(std::uint32_t d = 2; d < n + 1; ++d)
        {
            RowVectorType row = RowVectorType{Istar.cols() + 1};
            row.fill(1. / std::sqrt(lambda * d * (d + 1)));
            row(row.cols() - 1) = -static_cast<float>(d) / std::sqrt(lambda * d * (d + 1));

            Istar.conservativeResize(Istar.rows() + 1, Istar.cols() + 1);
            Istar.col(Istar.cols() - 1).setZero();
            Istar.row(Istar.rows() - 1) = row;
        }

        MatrixType I = std::sqrt(n) * Istar;
        MatrixType scaled_unitary = U * I;

        MatrixType sigmas = MatrixType::Zero(n, n + 1);
        for(std::uint32_t k = 0; k < n + 1; ++k)
        {
            sigmas.col(k) = subtract_fn(x, -scaled_unitary.col(k));
        }

        return sigmas.transpose();
    }

    VectorType get_Wm() const
    {
        return Wm;
    }

    VectorType get_Wc() const
    {
        return Wc;
    }

    std::uint32_t get_num_sigmas() const
    {
        return n + 1;
    }
};

}    // namespace filter::ukf