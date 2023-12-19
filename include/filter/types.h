/**
 * filter - some header-only C++ filter implementations.
 *
 * Linear algebra types used in the library.
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include <Eigen/Core>

namespace filter::ukf
{

#if !defined(FILTER_USE_FP32) && !defined(FILTER_USE_FP64)
#    error Configuration error: Neither FILTER_USE_FP32 nor FILTER_USE_FP64 are defined.
#elif defined(FILTER_USE_FP32) && defined(FILTER_USE_FP64)
#    error Configuration error: Only one of FILTER_USE_FP32 and FILTER_USE_FP64 can be defined.
#endif

#if defined(FILTER_USE_FP64)

using MatrixType = Eigen::MatrixXd;
using VectorType = Eigen::VectorXd;
using RowVectorType = Eigen::RowVectorXd;

#elif defined(FILTER_USE_FP32)

using MatrixType = Eigen::MatrixXf;
using VectorType = Eigen::VectorXf;
using RowVectorType = Eigen::RowVectorXf;

#endif

}    // namespace filter::ukf
