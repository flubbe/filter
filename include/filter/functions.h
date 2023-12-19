/**
 * filter - some header-only C++ filter implementations.
 *
 * math helper functions.
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include "filter/types.h"

namespace filter::ukf
{

inline MatrixType cholesky(const MatrixType& m)
{
    return m.llt().matrixL();
}

inline VectorType subtract(const VectorType& a, const VectorType& b)
{
    return a - b;
}

inline VectorType add(const VectorType& a, const VectorType& b)
{
    return a + b;
}

}    // namespace filter::ukf