/**
 * filter - some header-only C++ filter implementations.
 *
 * libfmt formatters for Eigen types.
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>

#define MAKE_FORMATTER(T)                                                              \
    inline std::ostream& operator<<(std::ostream& os, const T& v)                      \
    {                                                                                  \
        const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, "\t", " ", "", "", "", ""); \
        return os << v.format(fmt);                                                    \
    }                                                                                  \
    template<>                                                                         \
    struct fmt::formatter<T> : fmt::ostream_formatter                                  \
    {                                                                                  \
    };

MAKE_FORMATTER(Eigen::VectorXf);
MAKE_FORMATTER(Eigen::VectorXd);
MAKE_FORMATTER(Eigen::MatrixXf);
MAKE_FORMATTER(Eigen::MatrixXd);

#undef MAKE_FORMATTER
