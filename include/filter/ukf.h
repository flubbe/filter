/**
 * filter - some header-only C++ filter implementations.
 *
 * Unscented Kalman filter.
 *
 * Reference:
 *     FilterPy, https://github.com/rlabbe/filterpy,
 *               filterpy/kalman/UKF.py
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#pragma once

#include <functional>
#include <optional>
#include <vector>
#include <tuple>

#include <fmt/core.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>

#if !defined(FILTER_USE_FP32) && !defined(FILTER_USE_FP64)
#    define FILTER_USE_FP64
#elif defined(FILTER_USE_FP32) && defined(FILTER_USE_FP64)
#    error Configuration error: Only one of FILTER_USE_FP32 and FILTER_USE_FP64 can be defined.
#endif

#include "filter/types.h"
#include "filter/transform.h"
#include "filter/functions.h"

namespace filter::ukf
{

using hx_t = VectorType(const VectorType&);
using fx_t = VectorType(const VectorType&, float);
using sqrt_fn_t = MatrixType(const MatrixType&);
using mean_fn_t = MatrixType(const MatrixType&, const VectorType&);
using residual_fn_t = VectorType(const VectorType&, const VectorType&);

using unscented_transform_t =
  std::pair<VectorType, MatrixType>(
    const MatrixType& sigmas,
    const VectorType& Wm,
    const VectorType& Wc,
    const std::optional<MatrixType>& noise_cov,
    const std::function<mean_fn_t> mean_fn,
    const std::function<residual_fn_t> residual_fn);

template<typename PointCalculator>
class UnscentedKalmanFilter
{
protected:
    std::uint32_t dim_x, dim_z;
    VectorType x, x_prior, x_post;
    MatrixType P, P_prior, P_post;
    MatrixType Q, R;
    PointCalculator points;
    float dt;
    std::size_t num_sigmas;
    std::function<hx_t> hx;
    std::function<fx_t> fx;
    std::function<mean_fn_t> x_mean_fn;
    std::function<mean_fn_t> z_mean_fn;
    std::function<sqrt_fn_t> sqrt_fn;
    VectorType Wm, Wc;
    std::function<residual_fn_t> residual_x;
    std::function<residual_fn_t> residual_z;
    std::function<residual_fn_t> state_add;
    MatrixType sigmas_f, sigmas_h;
    MatrixType K;
    VectorType y;
    VectorType z;
    MatrixType S;
    MatrixType SI;

public:
    /** Constructor. */
    UnscentedKalmanFilter(
      std::uint32_t in_dim_x,
      std::uint32_t in_dim_z,
      float in_dt,
      std::function<hx_t> in_hx,
      std::function<fx_t> in_fx,
      PointCalculator in_points,
      std::function<sqrt_fn_t> in_sqrt_fn = cholesky,
      std::function<mean_fn_t> in_x_mean_fn = nullptr,
      std::function<mean_fn_t> in_z_mean_fn = nullptr,
      std::function<residual_fn_t> in_residual_x = subtract,
      std::function<residual_fn_t> in_residual_z = subtract,
      std::function<residual_fn_t> in_state_add = add)
    : dim_x{in_dim_x}
    , dim_z{in_dim_z}
    , x{VectorType::Zero(in_dim_x)}
    , P{MatrixType::Identity(in_dim_x, in_dim_x)}
    , x_prior{x}
    , P_prior{P}
    , x_post{x}
    , P_post{P}
    , Q{MatrixType::Identity(in_dim_x, in_dim_x)}
    , R{MatrixType::Identity(in_dim_z, in_dim_z)}
    , points{in_points}
    , dt{in_dt}
    , num_sigmas{points.get_num_sigmas()}
    , hx{in_hx}
    , fx{in_fx}
    , x_mean_fn{in_x_mean_fn}
    , z_mean_fn{in_z_mean_fn}
    , sqrt_fn{in_sqrt_fn}
    , Wm{points.get_Wm()}
    , Wc{points.get_Wc()}
    , residual_x{in_residual_x}
    , residual_z{in_residual_z}
    , state_add{in_state_add}
    , sigmas_f{MatrixType::Zero(num_sigmas, dim_x)}
    , sigmas_h{MatrixType::Zero(num_sigmas, dim_z)}
    , K{MatrixType::Zero(dim_x, dim_z)}
    , y{VectorType::Zero(dim_z)}
    , z{VectorType::Zero(dim_z)}
    , S{MatrixType::Zero(dim_z, dim_z)}
    , SI{MatrixType::Zero(dim_z, dim_z)}
    {
    }

    void predict(
      std::optional<float> in_dt = std::nullopt,
      std::function<unscented_transform_t> in_UT = nullptr,
      std::function<fx_t> in_fx = nullptr)
    {
        float dt = in_dt.has_value() ? in_dt.value() : this->dt;
        if(!in_UT)
        {
            in_UT = unscented_transform;
        }

        compute_process_sigmas(dt, in_fx);

        auto [x_ret, P_ret] = in_UT(sigmas_f, Wm, Wc, Q, x_mean_fn, residual_x);
        x = x_ret;
        P = P_ret;

        sigmas_f = points.sigma_points(x, P);

        x_prior = x;
        P_prior = P;
    }

    void update(
      const std::optional<VectorType>& in_z,
      std::optional<MatrixType> in_R = std::nullopt,
      std::function<unscented_transform_t> in_UT = nullptr,
      std::function<hx_t> in_hx = nullptr)
    {
        if(!in_z.has_value())
        {
            z = VectorType::Zero(dim_z);
            x_post = x;
            P_post = P;
            return;
        }

        if(!in_hx)
        {
            in_hx = hx;
        }

        if(!in_UT)
        {
            in_UT = unscented_transform;
        }

        if(!in_R.has_value())
        {
            in_R = R;
        }

        for(Eigen::Index i = 0; i < sigmas_f.rows(); ++i)
        {
            sigmas_h.row(i) = hx(sigmas_f.row(i));
        }

        auto ret = in_UT(sigmas_h, Wm, Wc, R, z_mean_fn, residual_z);
        filter::ukf::VectorType zp = std::get<0>(ret);
        S = std::get<1>(ret);
        SI = S.inverse();

        MatrixType Pxz = cross_variance(x, zp, sigmas_f, sigmas_h);

        K = Pxz * SI;                        // Kalman gain
        y = residual_z(in_z.value(), zp);    // residual

        // update Gaussian state estimate (x, P)
        x = state_add(x, K * y);
        P -= K * (S * K.transpose());

        // save measurement and posterior state
        z = in_z.value();
        x_post = x;
        P_post = P;
    }

    MatrixType cross_variance(
      const VectorType& x,
      const VectorType& z,
      const MatrixType& sigmas_f,
      const MatrixType& sigmas_h) const
    {
        MatrixType Pxz = MatrixType::Zero(sigmas_f.cols(), sigmas_h.cols());
        for(Eigen::Index i = 0; i < sigmas_f.rows(); ++i)
        {
            VectorType dx = residual_x(sigmas_f.row(i), x);
            VectorType dz = residual_z(sigmas_h.row(i), z);
            Pxz += Wc(i) * (dx * dz.transpose());
        }
        return Pxz;
    }

    void compute_process_sigmas(float dt, std::function<fx_t> in_fx = nullptr)
    {
        if(!in_fx)
        {
            in_fx = fx;
        }

        MatrixType sigmas = points.sigma_points(x, P);

        for(Eigen::Index i = 0; i < sigmas.rows(); ++i)
        {
            sigmas_f.row(i) = in_fx(sigmas.row(i), dt);
        }
    }

    std::pair<
      std::vector<VectorType>,
      std::vector<MatrixType>>
      batch_filter(
        const std::vector<std::optional<VectorType>>& zs,
        std::optional<std::vector<MatrixType>> Rs = std::nullopt,
        std::optional<std::vector<float>> dts = std::nullopt,
        std::function<unscented_transform_t> UT = nullptr)
    {
        if(zs.size() == 0)
        {
            return std::make_pair(std::vector<VectorType>{}, std::vector<MatrixType>{});
        }

        if(dim_z != zs[0].value().size())
        {
            throw std::runtime_error(fmt::format("each element in zs must be a 1D array of length {}", dim_z));
        }

        auto Rs_val = Rs.has_value() ? Rs.value() : std::vector<MatrixType>(zs.size(), R);
        auto dts_val = dts.has_value() ? dts.value() : std::vector<float>(zs.size(), dt);

        // mean estimates from Kalman Filter
        std::vector<VectorType> means = std::vector<VectorType>(zs.size(), VectorType::Zero(dim_x));

        // state covariances from Kalman Filter
        std::vector<MatrixType> covariances = std::vector<MatrixType>(zs.size(), MatrixType::Zero(dim_x, dim_x));

        for(std::size_t i = 0; i < zs.size(); ++i)
        {
            predict(dts_val[i], UT);
            update(zs[i], Rs_val[i], UT);
            means[i] = x;
            covariances[i] = P;
        }

        return std::make_pair(means, covariances);
    }

    std::tuple<
      std::vector<VectorType>,
      std::vector<MatrixType>,
      std::vector<MatrixType>>
      rts_smoother(
        const std::vector<VectorType>& Xs,
        const std::vector<MatrixType>& Ps,
        std::optional<std::vector<MatrixType>> Qs = std::nullopt,
        std::optional<std::vector<float>> dts = std::nullopt,
        std::function<unscented_transform_t> UT = nullptr)
    {
        if(Xs.size() != Ps.size())
        {
            throw std::runtime_error("Xs and Ps must have the same length");
        }

        std::size_t n = Xs.size();
        if(n == 0)
        {
            return {};
        }

        std::size_t dim_x = Xs[0].rows();

        if(!dts)
        {
            dts = std::vector<float>(n, dt);
        }
        auto dts_val = dts.value();

        if(!Qs)
        {
            Qs = std::vector<MatrixType>(n, Q);
        }

        if(!UT)
        {
            UT = unscented_transform;
        }

        // smoother gain
        std::vector<MatrixType> Ks(n, MatrixType::Zero(dim_x, dim_x));

        std::vector<VectorType> xs = Xs;
        std::vector<MatrixType> ps = Ps;

        if(n < 2)
        {
            return std::make_tuple(xs, ps, Ks);
        }

        MatrixType sigmas_f = MatrixType::Zero(num_sigmas, dim_x);

        for(std::size_t ctr = n - 1; ctr > 0; --ctr)
        {
            std::size_t k = ctr - 1;

            // create sigma points from state estimate, pass through state func
            MatrixType sigmas = points.sigma_points(xs[k], ps[k]);
            for(std::uint32_t i = 0; i < num_sigmas; ++i)
            {
                sigmas_f.row(i) = fx(sigmas.row(i), dts_val[k]);
            }

            auto [xb, Pb] = UT(
              sigmas_f, Wm, Wc, Q,
              x_mean_fn, residual_x);

            // compute cross variance
            MatrixType Pxb = MatrixType::Zero(sigmas_f.cols(), sigmas.cols());
            for(std::size_t i = 0; i < num_sigmas; ++i)
            {
                VectorType y = residual_x(sigmas_f.row(i), xb);
                VectorType z = residual_x(sigmas.row(i), Xs[k]);
                Pxb += Wc(i) * (y * z.transpose());
            }

            // compute gain
            MatrixType K = Pxb * Pb.inverse();

            // update the smoothed estimates
            xs[k] += K * residual_x(xs[k + 1], xb);
            ps[k] += K * (ps[k + 1] - Pb) * K.transpose();
            Ks[k] = K;
        }

        return std::make_tuple(xs, ps, Ks);
    }
};

}    // namespace filter::ukf