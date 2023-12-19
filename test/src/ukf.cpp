/**
 * filter - some header-only C++ filter implementations.
 *
 * Implements most tests found in filterpy's test_ukf.py.
 *
 * Reference:
 *     FilterPy, https://github.com/rlabbe/filterpy,
 *               filterpy/kalman/tests/test_ukf.py
 *
 * \author Felix Lubbe
 * \copyright Copyright (c) 2023
 * \license Distributed under the MIT software license (see accompanying LICENSE.txt).
 */

#include <vector>
#include <random>

#include <fmt/core.h>
#include <Eigen/Core>

#include <gtest/gtest.h>

#define FILTER_USE_FP64

#include "filter/ukf.h"
#include "filter/sigma_points.h"

#include "fmt_format.h"

namespace
{

/*
 * helper functions from numpy.
 */

template<typename DerivedA, typename DerivedB>
bool allclose(const Eigen::DenseBase<DerivedA>& a,
              const Eigen::DenseBase<DerivedB>& b,
              const typename DerivedA::RealScalar& rtol = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar& atol = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
{
    return ((a.derived() - b.derived()).array().abs()
            <= (atol + rtol * b.derived().array().abs()))
      .all();
}

std::random_device rd;
std::mt19937 gen(rd());

template<typename RealType = float, typename GeneratorType = std::mt19937>
RealType randn(GeneratorType& generator)
{
    std::normal_distribution<RealType> dist;
    return dist(generator);
}

TEST(filter, unscented_transform)
{
    filter::ukf::MatrixType sigmas = (filter::ukf::MatrixType{3, 1} << 1, 2, 3).finished();
    filter::ukf::VectorType Wm = (filter::ukf::VectorType{3} << 1, 1, 1).finished();
    filter::ukf::VectorType Wc = (filter::ukf::VectorType{3} << 1, 1, 1).finished();

    auto [x, P] = filter::ukf::unscented_transform(sigmas, Wm, Wc);

    EXPECT_EQ(x.rows(), 1);
    EXPECT_EQ(x.cols(), 1);
    EXPECT_EQ(P.rows(), 1);
    EXPECT_EQ(P.cols(), 1);

    EXPECT_EQ(x(0), 6);
    EXPECT_EQ(P(0), 50);
}

TEST(filter, sigma_points)
{
    auto mssp = filter::ukf::MerweScaledSigmaPoints(1, 1, 1, 1);

    filter::ukf::VectorType Wm = mssp.get_Wm();
    filter::ukf::VectorType Wc = mssp.get_Wc();

    EXPECT_EQ(Wm.rows(), 3);
    EXPECT_EQ(Wm.cols(), 1);
    EXPECT_EQ(Wc.rows(), 3);
    EXPECT_EQ(Wc.cols(), 1);

    EXPECT_EQ(Wm, (filter::ukf::VectorType{3} << 0.5, 0.25, 0.25).finished());
    EXPECT_EQ(Wc, (filter::ukf::VectorType{3} << 1.5, 0.25, 0.25).finished());

    filter::ukf::MatrixType sp = mssp.sigma_points(
      (filter::ukf::VectorType{1} << 1).finished(),
      (filter::ukf::MatrixType{1, 1} << 2).finished());

    EXPECT_EQ(sp, (filter::ukf::MatrixType{3, 1} << 1, 3, -1).finished());
}

TEST(filter, ukf_instantiation)
{
    auto ukf_instance = filter::ukf::UnscentedKalmanFilter<filter::ukf::MerweScaledSigmaPoints>(
      1, 1,                                           /* dim x, dim z */
      1,                                              /* dt */
      nullptr,                                        /* in_hx */
      nullptr,                                        /* in_fx */
      filter::ukf::MerweScaledSigmaPoints(1, 1, 1, 1) /* points */
    );
}

template<typename PointCalculator>
class TestUnscentedKalmanFilter : public filter::ukf::UnscentedKalmanFilter<PointCalculator>
{
    using super = filter::ukf::UnscentedKalmanFilter<PointCalculator>;

public:
    /** Constructor. */
    TestUnscentedKalmanFilter(
      std::uint32_t in_dim_x,
      std::uint32_t in_dim_z,
      float in_dt,
      std::function<filter::ukf::hx_t> in_hx,
      std::function<filter::ukf::fx_t> in_fx,
      PointCalculator in_points,
      std::function<filter::ukf::sqrt_fn_t> in_sqrt_fn = filter::ukf::cholesky,
      std::function<filter::ukf::mean_fn_t> in_x_mean_fn = nullptr,
      std::function<filter::ukf::mean_fn_t> in_z_mean_fn = nullptr,
      std::function<filter::ukf::residual_fn_t> in_residual_x = filter::ukf::subtract,
      std::function<filter::ukf::residual_fn_t> in_residual_z = filter::ukf::subtract,
      std::function<filter::ukf::residual_fn_t> in_state_add = filter::ukf::add)
    : filter::ukf::UnscentedKalmanFilter<PointCalculator>(in_dim_x, in_dim_z, in_dt, in_hx, in_fx, in_points, in_sqrt_fn, in_x_mean_fn, in_z_mean_fn, in_residual_x, in_residual_z, in_state_add)
    {
    }

    filter::ukf::VectorType& get_x()
    {
        return super::x;
    }
    filter::ukf::VectorType& get_x_prior()
    {
        return super::x_prior;
    };
    filter::ukf::MatrixType& get_P()
    {
        return super::P;
    }
    filter::ukf::MatrixType& get_P_prior()
    {
        return super::P_prior;
    };
    filter::ukf::MatrixType& get_Q()
    {
        return super::Q;
    }
    filter::ukf::MatrixType& get_R()
    {
        return super::R;
    }
    filter::ukf::MatrixType& get_K()
    {
        return super::K;
    }

    filter::ukf::VectorType get_Wm() const
    {
        return super::points.get_Wm();
    }
    filter::ukf::VectorType get_Wc() const
    {
        return super::points.get_Wc();
    }

    filter::ukf::MatrixType get_sigmas_f() const
    {
        return super::sigmas_f;
    }

    filter::ukf::VectorType get_y() const
    {
        return super::y;
    }
    filter::ukf::VectorType get_z() const
    {
        return super::z;
    }

    filter::ukf::MatrixType get_S() const
    {
        return super::S;
    }
    filter::ukf::MatrixType get_SI() const
    {
        return super::SI;
    }
};

/*
 * filterpy tests.
 */

TEST(filter, scaled_weights)
{
    for(int n = 1; n < 5; ++n)
    {
        for(float alpha = 0.99; alpha < 1.01; alpha += 0.0002)
        {
            for(float beta = 0; beta < 2; ++beta)
            {
                for(float kappa = 0; kappa < 2; ++kappa)
                {
                    auto sp = filter::ukf::MerweScaledSigmaPoints(n, alpha, 0, 3 - n);
                    EXPECT_LT(std::abs(sp.get_Wm().sum() - 1), 1e-1);
                    EXPECT_LT(std::abs(sp.get_Wc().sum() - 1), 1e-1);
                }
            }
        }
    }
}

TEST(filter, julier_sigma_points_1D)
{
    float kappa = 0;
    auto sp = filter::ukf::JulierSigmaPoints(1, kappa);

    EXPECT_TRUE(allclose(sp.get_Wm(), sp.get_Wc(), 1e-12));
    EXPECT_EQ(sp.get_Wm().size(), 3);

    float mean = 5;
    float cov = 9;

    filter::ukf::VectorType mean_vec = (filter::ukf::VectorType{1} << mean).finished();
    filter::ukf::MatrixType cov_mat = (filter::ukf::MatrixType{1, 1} << cov).finished();

    auto Xi = sp.sigma_points(mean_vec, cov_mat);
    auto [xm, ucov] = filter::ukf::unscented_transform(
      Xi, sp.get_Wm(), sp.get_Wc(),
      (filter::ukf::MatrixType{1, 1} << 0).finished());

    // sum of weights* sigma points should be the original mean
    EXPECT_EQ(Xi.size(), sp.get_Wm().size());
    float m = 0;
    for(int i = 0; i < Xi.rows(); ++i)
    {
        m += Xi(i) * sp.get_Wm()(i);
    }

    EXPECT_LT(std::abs(m - mean), 1e-12);
    EXPECT_LT(std::abs(xm(0) - mean), 1e-12);
    EXPECT_LT(std::abs(ucov(0, 0) - cov), 1e-12);

    EXPECT_EQ(Xi.rows(), 3);
    EXPECT_EQ(Xi.cols(), 1);
}

TEST(filter, simplex_sigma_points_1D)
{
    auto sp = filter::ukf::SimplexSigmaPoints(1);

    auto Wm = sp.get_Wm();
    auto Wc = sp.get_Wc();

    EXPECT_TRUE(allclose(Wm, Wc, 1e-12));
    EXPECT_EQ(Wm.rows(), 2);

    float mean = 5;
    float cov = 9;

    auto Xi = sp.sigma_points(
      (filter::ukf::VectorType{1} << mean).finished(),
      (filter::ukf::MatrixType{1, 1} << cov).finished());
    auto [xm, ucov] = filter::ukf::unscented_transform(
      Xi, Wm, Wc,
      (filter::ukf::MatrixType{1, 1} << 0).finished());

    // sum of weights*sigma points should be the original mean
    float m = 0.0;
    for(std::size_t i = 0; i < Xi.rows(); ++i)
    {
        m += Xi(i) * Wm(i);
    }

    EXPECT_LT(std::abs(m - mean), 1e-12);
    EXPECT_LT(std::abs(xm(0) - mean), 1e-12);
    EXPECT_LT(std::abs(ucov(0, 0) - cov), 1e-12);

    EXPECT_EQ(Xi.rows(), 2);
    EXPECT_EQ(Xi.cols(), 1);
}

TEST(filter, simplex_sigma_points_2D)
{
    auto sp = filter::ukf::SimplexSigmaPoints(4);

    auto Wm = sp.get_Wm();
    auto Wc = sp.get_Wc();

    EXPECT_TRUE(allclose(Wm, Wc, 1e-12));
    EXPECT_EQ(Wm.rows(), 5);

    // clang-format off
    filter::ukf::VectorType mean = (filter::ukf::VectorType{4} << 
        -1, 2, 0, 5
    ).finished();

    filter::ukf::MatrixType cov1 = (filter::ukf::MatrixType{2, 2} << 
        1, 0.5,
        0.5, 1
    ).finished();

    filter::ukf::MatrixType cov2 = (filter::ukf::MatrixType{2, 2} <<
        5, 0.5,
        0.5, 3
    ).finished();
    // clang-format on

    filter::ukf::MatrixType cov = filter::ukf::MatrixType::Zero(4, 4);
    cov.topLeftCorner(2, 2) = cov1;
    cov.bottomRightCorner(2, 2) = cov2;

    filter::ukf::MatrixType Xi = sp.sigma_points(mean, cov);
    auto [xm, ucov] = filter::ukf::unscented_transform(Xi, Wm, Wc);

    EXPECT_TRUE(allclose(xm, mean, 1e-5, 1e-8));
    EXPECT_TRUE(allclose(cov, ucov, 1e-5, 1e-8));
}

struct RadarSim
{
    float x;
    float dt;

    RadarSim(float in_dt)
    : x{0}
    , dt{in_dt}
    {
    }

    float get_range()
    {
        float vel = 100 + 5 * randn(gen);
        float alt = 1000 + 10 * randn(gen);

        x += vel * dt;
        float v = x * 0.05 * randn(gen);
        return std::sqrt(x * x + alt * alt) + v;
    }
};

TEST(filter, radar)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        filter::ukf::MatrixType A = (filter::ukf::MatrixType::Identity(3, 3)
                                     + dt * (filter::ukf::MatrixType{3, 3} << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished());
        return A * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{1} << std::sqrt(x(0) * x(0) + x(2) * x(2))).finished();
    };

    float dt = 0.05;

    auto sp = filter::ukf::JulierSigmaPoints(3, 0.);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::JulierSigmaPoints>(
      3, 1, dt, hx, fx, sp);
    EXPECT_TRUE(allclose(kf.get_x(), kf.get_x_prior()));
    EXPECT_TRUE(allclose(kf.get_P(), kf.get_P_prior()));

    kf.get_Q() *= 0.01;
    kf.get_R() = (filter::ukf::MatrixType{1, 1} << 10).finished();
    kf.get_x() = (filter::ukf::VectorType{3} << 0., 90., 1100.).finished();
    kf.get_P() *= 100.;
    auto radar = RadarSim(dt);

    std::size_t n = 401;    // = len(t) with t = np.arange(0, 20+dt, dt)
    filter::ukf::MatrixType xs = filter::ukf::MatrixType::Zero(n, 3);

    std::vector<float> rs;
    for(std::size_t i = 0; i < n; ++i)
    {
        float r = radar.get_range();

        kf.predict();
        kf.update((filter::ukf::VectorType{1} << r).finished());

        xs.row(i) = kf.get_x();
        rs.push_back(r);
    }
}

TEST(filter, linear_2d_merwe)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        // clang-format off
        filter::ukf::MatrixType F = (filter::ukf::MatrixType{4, 4} << 
            1, dt, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dt,
            0, 0, 0, 1
        ).finished();
        // clang-format on

        return F * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{2} << x(0), x(2)).finished();
    };

    float dt = 0.1;
    auto points = filter::ukf::MerweScaledSigmaPoints(4, .1, 2., -1);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::MerweScaledSigmaPoints>(
      4, 2, dt, hx, fx, points);

    kf.get_x() = (filter::ukf::VectorType{4} << -1, 1, -1, 1).finished();
    kf.get_P() *= 1.1;

    std::vector<std::optional<filter::ukf::VectorType>> zs(20);
    for(std::size_t i = 0; i < zs.size(); ++i)
    {
        zs[i] = (filter::ukf::VectorType{2} << i + randn(gen) * 0.1, i + randn(gen) * 0.1).finished();
    }

    auto [Ms, Ps] = kf.batch_filter(zs);

    std::vector<float> dts(Ps.size(), dt);
    auto [smooth_x, P, K] = kf.rts_smoother(Ms, Ps, std::nullopt, dts);
}

TEST(filter, linear_2d_simplex)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        // clang-format off
        filter::ukf::MatrixType F = (filter::ukf::MatrixType{4, 4} << 
            1, dt, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dt,
            0, 0, 0, 1
        ).finished();
        // clang-format on

        return F * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{2} << x(0), x(2)).finished();
    };

    float dt = 0.1;
    auto points = filter::ukf::SimplexSigmaPoints(4);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::SimplexSigmaPoints>(
      4, 2, dt, hx, fx, points);

    kf.get_x() = (filter::ukf::VectorType{4} << -1, 1, -1, 1).finished();
    kf.get_P() *= 0.0001;

    auto mss = filter::ukf::MerweScaledSigmaPoints(4, .1, 2., -1);
    auto ukf = TestUnscentedKalmanFilter<filter::ukf::MerweScaledSigmaPoints>(
      4, 2, dt, hx, fx, mss);

    kf.get_x() = (filter::ukf::VectorType{4} << -1, 1, -1, 1).finished();
    kf.get_P() *= 1;

    filter::ukf::VectorType x = kf.get_x();

    std::vector<std::optional<filter::ukf::VectorType>> zs(20);
    for(std::size_t i = 0; i < zs.size(); ++i)
    {
        x = fx(x, dt);
        filter::ukf::VectorType z = (filter::ukf::VectorType{2} << x(0) + randn(gen) * 0.1, x(2) + randn(gen) * 0.1).finished();

        zs[i] = z;
    }

    auto [Ms, Ps] = kf.batch_filter(zs);

    std::vector<float> dts(Ps.size(), dt);
    auto [smooth_x, P, K] = kf.rts_smoother(Ms, Ps, std::nullopt, dts);
}

TEST(filter, linear_1d)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        filter::ukf::MatrixType F = (filter::ukf::MatrixType{2, 2} << 1, dt, 0, 1).finished();
        return F * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{1} << x(0)).finished();
    };

    float dt = 0.1;
    auto points = filter::ukf::MerweScaledSigmaPoints(2, .1, 2., -1);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::MerweScaledSigmaPoints>(
      2, 1, dt, hx, fx, points);

    kf.get_x() = (filter::ukf::VectorType{2} << 1, 2).finished();
    kf.get_P() = (filter::ukf::MatrixType{2, 2} << 1, 1.1, 1.1, 3).finished();

    kf.get_R() *= 0.05;
    kf.get_Q() = (filter::ukf::MatrixType{2, 2} << 0, 0, 0, 0.001).finished();

    filter::ukf::VectorType z = (filter::ukf::VectorType{1} << 2).finished();

    kf.predict();
    kf.update(z);

    std::vector<filter::ukf::VectorType> zs;
    for(int i = 0; i < 50; ++i)
    {
        filter::ukf::VectorType z = (filter::ukf::VectorType{1} << i + randn(gen) * 0.1).finished();
        zs.push_back(z);

        kf.predict();
        kf.update(z);

        filter::ukf::MatrixType KT = kf.get_K().transpose();
        fmt::print("K {}\n", KT);
        fmt::print("x {}\n", kf.get_x());
    }
}

TEST(filter, test_batch_missing_data)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        // clang-format off
        filter::ukf::MatrixType F = (filter::ukf::MatrixType{4, 4} << 
            1, dt, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, dt,
            0, 0, 0, 1
        ).finished();
        // clang-format on

        return F * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{2} << x(0), x(2)).finished();
    };

    float dt = 0.1;
    auto points = filter::ukf::MerweScaledSigmaPoints(4, .1, 2., -1);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::MerweScaledSigmaPoints>(
      4, 2, dt, hx, fx, points);

    kf.get_x() = (filter::ukf::VectorType{4} << -1., 1., -1., 1).finished();
    kf.get_P() *= 0.0001;

    std::vector<std::optional<filter::ukf::VectorType>> zs(20);
    for(std::size_t i = 0; i < zs.size(); ++i)
    {
        filter::ukf::VectorType z = (filter::ukf::VectorType{2} << i + randn(gen) * 0.1, i + randn(gen) * 0.1).finished();
        zs[i] = z;
    }

    zs[2] = std::nullopt;
    auto [Ms, Ps] = kf.batch_filter(zs);
}

TEST(filter, rts)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        filter::ukf::MatrixType A = (filter::ukf::MatrixType::Identity(3, 3)
                                     + dt * (filter::ukf::MatrixType{3, 3} << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished());
        return A * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{1} << std::sqrt(x(0) * x(0) + x(2) * x(2))).finished();
    };

    float dt = 0.05;

    auto sp = filter::ukf::JulierSigmaPoints(3, 1.);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::JulierSigmaPoints>(
      3, 1, dt, hx, fx, sp);
    EXPECT_TRUE(allclose(kf.get_x(), kf.get_x_prior()));
    EXPECT_TRUE(allclose(kf.get_P(), kf.get_P_prior()));

    kf.get_Q() *= 0.01;
    kf.get_R() = (filter::ukf::MatrixType{1, 1} << 10).finished();
    kf.get_x() = (filter::ukf::VectorType{3} << 0., 90., 1100.).finished();
    kf.get_P() *= 100.;
    auto radar = RadarSim(dt);

    std::size_t n = 401;    // = len(t) with t = np.arange(0, 20+dt, dt)
    std::vector<filter::ukf::VectorType> xs;

    std::vector<std::optional<filter::ukf::VectorType>> rs;
    for(std::size_t i = 0; i < n; ++i)
    {
        filter::ukf::VectorType r = (filter::ukf::VectorType{1} << radar.get_range()).finished();
        kf.predict();
        kf.update(r);

        xs.push_back(kf.get_x());
        rs.push_back(r);
    }

    kf.get_x() = (filter::ukf::VectorType{3} << 0., 90., 1100.).finished();
    kf.get_P() = filter::ukf::MatrixType::Identity(3, 3) * 100;
    auto [M, P] = kf.batch_filter(rs);
    EXPECT_EQ(M, xs);
}

TEST(filter, fixed_lag)
{
    auto fx = [](const filter::ukf::VectorType& x, float dt) -> filter::ukf::VectorType
    {
        filter::ukf::MatrixType A = (filter::ukf::MatrixType::Identity(3, 3)
                                     + dt * (filter::ukf::MatrixType{3, 3} << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished());
        return A * x;
    };

    auto hx = [](const filter::ukf::VectorType& x) -> filter::ukf::VectorType
    {
        return (filter::ukf::VectorType{1} << std::sqrt(x(0) * x(0) + x(2) * x(2))).finished();
    };

    float dt = 0.05;

    auto sp = filter::ukf::JulierSigmaPoints(3, 0.);
    auto kf = TestUnscentedKalmanFilter<filter::ukf::JulierSigmaPoints>(
      3, 1, dt, hx, fx, sp);

    kf.get_Q() *= 0.01;
    kf.get_R() = (filter::ukf::MatrixType{1, 1} << 10).finished();
    kf.get_x() = (filter::ukf::VectorType{3} << 0., 90., 1100.).finished();
    auto radar = RadarSim(dt);

    std::size_t n = 401;    // = len(t) with t = np.arange(0, 20+dt, dt)
    std::vector<filter::ukf::VectorType> xs, flxs, M;
    std::vector<filter::ukf::MatrixType> P;
    std::vector<std::optional<filter::ukf::VectorType>> rs;
    std::size_t N = 10;
    for(std::size_t i = 0; i < n; ++i)
    {
        filter::ukf::VectorType r = (filter::ukf::VectorType{1} << radar.get_range()).finished();
        kf.predict();
        kf.update(r);

        xs.push_back(kf.get_x());
        flxs.push_back(kf.get_x());
        rs.push_back(r);

        M.push_back(kf.get_x());
        P.push_back(kf.get_P());

        fmt::print("{}\n", i);
        if(i == 20 && M.size() >= N)
        {
            try
            {

                auto [M2, P2, K] = kf.rts_smoother({M.end() - N, M.end()}, {P.end() - N, P.end()});
                std::copy(M2.begin(), M2.end(), flxs.end() - N);
            }
            catch(const std::exception&)
            {
                fmt::print("except: {}\n", i);
            }
        }
    }

    kf.get_x() = (filter::ukf::VectorType{3} << 0., 90., 1100.).finished();
    kf.get_P() = filter::ukf::MatrixType::Identity(3, 3) * 100;
    auto [M_batch, P_batch] = kf.batch_filter(rs);

    std::vector<filter::ukf::MatrixType> Qs{n, kf.get_Q()};
    auto [M2, P2, K] = kf.rts_smoother(M, P, Qs);

    fmt::print("({}, {})\n", xs[0].rows(), xs[0].cols());
}

}    // namespace