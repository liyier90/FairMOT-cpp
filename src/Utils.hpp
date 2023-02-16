#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include "DataType.hpp"

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(const torch::Tensor &rFeat,
                         const torch::Tensor &rIndices);

cv::Mat GetAffineTransform(const std::array<double, 2> &rCenter,
                           const double scale, const double rot,
                           const std::array<int, 2> &rOutputSize,
                           const std::array<double, 2> &rShift,
                           const bool inv = false);

cv::Scalar GetColor(int idx);

std::vector<double> GetDir(const torch::ArrayRef<double> &rSrcPoint,
                           const double rotRad);

void GetThirdPoint(const cv::Point2f &rPoint1, const cv::Point2f &rPoint2,
                   cv::Point2f &rThirdPoint);

double Lapjv(const std::vector<std::vector<float>> &rCost,
             std::vector<int> &rRowsol, std::vector<int> &rColsol,
             bool extendCost = false, float costLimit = FLT_MAX,
             bool returnCost = true);

cv::Mat Letterbox(cv::Mat image, int targetHeight, int targetWidth);

void TransformCoords(torch::Tensor &rCoords,
                     const std::array<double, 2> &rCenter, const double scale,
                     const std::array<int, 2> &rOutputSize);

torch::Tensor TransposeAndGatherFeat(const torch::Tensor &rFeat,
                                     const torch::Tensor &rIndices);

void Visualize(cv::Mat image, const std::vector<TrackOutput> &rResults,
               const int frameId);

template <typename Matrix, typename Row, typename Value>
void Map(Matrix &rTarget, const std::vector<std::vector<Value>> &rSource) {
  auto idx = 0;
  for (const auto &r_row : rSource) {
    rTarget.row(idx++) = Eigen::Map<const Row>(r_row.data(), r_row.size());
  }
}
}  // namespace util
}  // namespace fairmot

#endif  // SRC_UTILS_HPP_

