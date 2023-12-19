/**
 * filter - some header-only C++ filter implementations.
 *
 * Unscented transformation.
 *
 * Reference:
 *     FilterPy, https://github.com/rlabbe/filterpy,
 *               filterpy/kalman/unscented_transform.py
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include <utility>
#include <optional>
#include <functional>

#include <Eigen/Core>

#include "filter/types.h"

namespace filter::ukf
{

inline std::pair<VectorType, MatrixType> unscented_transform(
  const MatrixType& sigmas,
  const VectorType& Wm,
  const VectorType& Wc,
  const std::optional<MatrixType>& noise_cov = std::nullopt,
  const std::function<VectorType(const MatrixType&, const VectorType&)> mean_fn = nullptr,
  const std::function<VectorType(const VectorType&, const VectorType&)> residual_fn = nullptr)
{
    Eigen::Index kmax = sigmas.rows();
    Eigen::Index n = sigmas.cols();

    VectorType x = (!mean_fn) ? Wm.transpose() * sigmas : mean_fn(sigmas, Wm);

    MatrixType P = MatrixType::Zero(n, n);
    VectorType y;
    if(!residual_fn)
    {
        // use subtraction.
        for(Eigen::Index k = 0; k < kmax; ++k)
        {
            y = sigmas.row(k) - x.transpose();
            P += Wc(k) * (y * y.transpose());
        }
    }
    else
    {
        for(Eigen::Index k = 0; k < kmax; ++k)
        {
            y = residual_fn(sigmas.row(k), x);
            P += Wc(k) * (y * y.transpose());
        }
    }

    if(noise_cov)
    {
        P += noise_cov.value();
    }

    return std::make_pair(x, P);
}

}    // namespace filter::ukf