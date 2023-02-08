#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <torch/torch.h>

#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(const torch::Tensor &rFeat,
                         const torch::Tensor &rIndices);

cv::Mat GetAffineTransform(const std::array<double, 2> &rCenter,
                           const double scale, const double rot,
                           const std::array<int, 2> &rOutputSize,
                           const std::array<double, 2> &rShift,
                           const bool inv = false);

std::vector<double> GetDir(const torch::ArrayRef<double> &rSrcPoint,
                           const double rotRad);

void GetThirdPoint(const cv::Point2f &rPoint1, const cv::Point2f &rPoint2,
                   cv::Point2f &rThirdPoint);

cv::Mat Letterbox(cv::Mat image, int targetHeight, int targetWidth);

void TransformCoords(torch::Tensor &rCoords,
                     const std::array<double, 2> &rCenter, const double scale,
                     const std::array<int, 2> &rOutputSize);

torch::Tensor TransposeAndGatherFeat(const torch::Tensor &rFeat,
                                     const torch::Tensor &rIndices);
}  // namespace util
}  // namespace fairmot

#endif  // SRC_UTILS_HPP_

