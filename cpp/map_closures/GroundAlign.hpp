#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace map_closures {
struct PixelHash {
    size_t operator()(const Eigen::Vector2i &pixel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(pixel.data());
        return (vec[0] * 73856093 ^ vec[1] * 19349669);
    }
};

std::vector<Eigen::Vector3d> ComputeLowestPoints(const std::vector<Eigen::Vector3d> &pointcloud,
                                                 const float resolution);

Eigen::Matrix4d GetGroundAlignment(const std::vector<Eigen::Vector3d> &pointcloud,
                                const double projection_threshold,
                                const float resolution);
}  // namespace map_closures