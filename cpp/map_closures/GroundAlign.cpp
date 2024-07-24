#include "GroundAlign.hpp"

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <algorithm>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

namespace {
void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &pointcloud) {
    std::transform(pointcloud.cbegin(), pointcloud.cend(), pointcloud.begin(),
                   [&](const auto &point) { return T * point; });
}

using LinearSystem = std::pair<Eigen::Matrix3d, Eigen::Vector3d>;
LinearSystem BuildLinearSystem(const std::vector<Eigen::Vector3d> &points,
                               const double projection_threshold) {
    auto compute_jacobian_and_residual = [](const auto &point) {
        const double residual = point.z();
        Eigen::Matrix<double, 1, 3> J;
        J(0, 0) = 1.0;
        J(0, 1) = point.y();
        J(0, 2) = -point.x();
        return std::make_pair(J, residual);
    };

    auto sum_linear_systems = [](LinearSystem a, const LinearSystem &b) {
        a.first += b.first;
        a.second += b.second;
        return a;
    };

    auto Weight = [&](const double residual, const double threshold) {
        return std::abs(residual) <= threshold ? 1.0 : 0.0;
    };

    const auto &[H, b] =
        std::transform_reduce(points.cbegin(), points.cend(),
                              LinearSystem(Eigen::Matrix3d::Zero(), Eigen::Vector3d::Zero()),
                              sum_linear_systems, [&](const auto &point) {
                                  const auto &[J, residual] = compute_jacobian_and_residual(point);
                                  const double w = Weight(residual, projection_threshold);
                                  return LinearSystem(J.transpose() * w * J,          // JTJ
                                                      J.transpose() * w * residual);  // JTr
                              });
    return {H, b};
}

static constexpr double convergence_threshold = 1e-3;
static constexpr int max_iterations = 20;
}  // namespace

namespace map_closures {

std::vector<Eigen::Vector3d> ComputeLowestPoints(const std::vector<Eigen::Vector3d> &pointcloud,
                                                 const float resolution) {
    std::vector<Eigen::Vector3d> lowest_points;
    lowest_points.reserve(pointcloud.size());
    tsl::robin_map<Eigen::Vector2i, Eigen::Vector3d, PixelHash> lowest_point_hash_map;
    auto PointToPixel = [&resolution](const Eigen::Vector3d &pt) -> Eigen::Vector2i {
        return Eigen::Vector2i(static_cast<int>(std::floor(pt.x() / resolution)),
                               static_cast<int>(std::floor(pt.y() / resolution)));
    };
    std::for_each(pointcloud.cbegin(), pointcloud.cend(), [&](const Eigen::Vector3d &point) {
        auto pixel = PointToPixel(point);
        if (lowest_point_hash_map.find(pixel) == lowest_point_hash_map.cend()) {
            lowest_point_hash_map.insert({pixel, point});
        } else if (point.z() < lowest_point_hash_map[pixel].z()) {
            lowest_point_hash_map[pixel] = point;
        }
    });
    std::for_each(lowest_point_hash_map.cbegin(), lowest_point_hash_map.cend(),
                  [&](const auto &entry) { lowest_points.emplace_back(entry.second); });
    lowest_points.shrink_to_fit();
    return lowest_points;
}

Eigen::Matrix4d GetGroundAlignment(const std::vector<Eigen::Vector3d> &pointcloud,
                                const double projection_threshold,
                                const float resolution) {
    auto lowest_points = ComputeLowestPoints(pointcloud, resolution);

    Sophus::SE3d T = Sophus::SE3d();
    for (int iters = 0; iters < max_iterations; iters++) {
        const auto &[H, b] = BuildLinearSystem(lowest_points, projection_threshold);
        const Eigen::Vector3d dx = H.ldlt().solve(-b);
        const Eigen::Matrix<double, 6, 1> se3 =
            Eigen::Matrix<double, 6, 1>(0, 0, dx.x(), dx.y(), dx.z(), 0.0);
        Sophus::SE3d estimation(Sophus::SE3d::exp(se3));
        TransformPoints(estimation, lowest_points);
        T = estimation * T;

        if (dx.norm() < convergence_threshold) break;
    }
    return T.matrix();
}
}  // namespace map_closures