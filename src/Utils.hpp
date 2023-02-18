#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

#include "DataType.hpp"

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(const torch::Tensor &rFeat,
                         const torch::Tensor &rIndices);

cv::Mat GetAffineTransform(const Vec2D<double> &rCenter, const double scale,
                           const double rot, const Vec2D<int> &rOutputSize,
                           const Vec2D<double> &rShift, const bool inv = false);

cv::Scalar GetColor(int idx);

Vec2D<double> GetDir(const torch::ArrayRef<double> &rSrcPoint,
                     const double rotRad);

void GetThirdPoint(const cv::Point2f &rPoint1, const cv::Point2f &rPoint2,
                   cv::Point2f &rThirdPoint);

float Lap(const std::vector<float> &rCost, const int numRows, const int numCols,
          std::vector<int> &rRowsol, std::vector<int> &rColsol,
          bool extendCost = false, float costLimit = FLT_MAX);

cv::Mat Letterbox(cv::Mat image, int targetHeight, int targetWidth);

void TransformCoords(torch::Tensor &rCoords, const Vec2D<double> &rCenter,
                     const double scale, const Vec2D<int> &rOutputSize);

torch::Tensor TransposeAndGatherFeat(const torch::Tensor &rFeat,
                                     const torch::Tensor &rIndices);

void Visualize(cv::Mat image, const std::vector<TrackOutput> &rResults,
               const int frameId);

template <typename Matrix, typename Row, typename Value>
void Map(Matrix &rTarget, const std::vector<std::vector<Value>> &rSource);

template <typename Matrix, typename Row, typename Value>
void Map(Matrix &rTarget, const std::vector<Value> &rSource, const int numRows,
         const int numCols);
}  // namespace util
}  // namespace fairmot

#include "Utils.ipp"

#endif  // SRC_UTILS_HPP_

