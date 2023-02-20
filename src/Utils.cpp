#include "Utils.hpp"

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Cpuid.hpp"
#include "DataType.hpp"
#include "Lap.hpp"
#include "STrack.hpp"

static SIMDFlags simd_flags;

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(const torch::Tensor &rFeat,
                         const torch::Tensor &rIndices) {
  const auto dim = rFeat.size(2);
  const auto indices =
      rIndices.unsqueeze(2).expand({rIndices.size(0), rIndices.size(1), dim});
  const auto feat = rFeat.gather(1, indices);

  return feat;
}

cv::Mat GetAffineTransform(const Vec2D<double> &rCenter, const double scale,
                           const double rot, const Vec2D<int> &rOutputSize,
                           const Vec2D<double> &rShift, const bool inv) {
  const auto src_dir = GetDir({0.0, scale * -0.5}, M_PI * rot / 180.0);
  const Vec2D<double> dst_dir = {0.0, rOutputSize[0] * -0.5};

  cv::Point2f src[3];
  cv::Point2f dst[3];
  src[0] = cv::Point2f(rCenter[0] + scale * rShift[0],
                       rCenter[1] + scale * rShift[1]);
  src[1] = cv::Point2f(rCenter[0] + src_dir[0] + scale * rShift[0],
                       rCenter[1] + src_dir[1] + scale * rShift[1]);
  dst[0] = cv::Point2f(rOutputSize[0] * 0.5, rOutputSize[1] * 0.5);
  dst[1] = cv::Point2f(rOutputSize[0] * 0.5 + dst_dir[0],
                       rOutputSize[1] * 0.5 + dst_dir[1]);
  GetThirdPoint(src[0], src[1], src[2]);
  GetThirdPoint(dst[0], dst[1], dst[2]);

  cv::Mat matrix;
  if (inv) {
    matrix = cv::getAffineTransform(dst, src);
  } else {
    matrix = cv::getAffineTransform(src, dst);
  }

  return matrix;
}

cv::Scalar GetColor(int idx) {
  idx *= 3;
  return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}

Vec2D<double> GetDir(const torch::ArrayRef<double> &rSrcPoint,
                     const double rotRad) {
  const auto sin = std::sin(rotRad);
  const auto cos = std::cos(rotRad);

  const Vec2D<double> src_result = {rSrcPoint[0] * cos - rSrcPoint[1] * sin,
                                    rSrcPoint[0] * sin + rSrcPoint[1] * cos};

  return src_result;
}

void GetThirdPoint(const cv::Point2f &rPoint1, const cv::Point2f &rPoint2,
                   cv::Point2f &rThirdPoint) {
  rThirdPoint.x = rPoint2.x + rPoint2.y - rPoint1.y;
  rThirdPoint.y = rPoint2.y + rPoint2.x - rPoint1.x;
}

float Lap(const std::vector<float> &rCost, const int numRows, const int numCols,
          std::vector<int> &rRowsol, std::vector<int> &rColsol, bool extendCost,
          float costLimit) {
  ArrayR<float, -1> cost_f(numRows, numCols);
  Map<ArrayR<float, -1>, ArrayR<float, 1, -1>, float>(cost_f, rCost, numRows,
                                                      numCols);

  if (numRows != numCols && !extendCost) {
    throw std::logic_error(
        "Square cost array expected. Pass extendCost = true if non-square "
        "cost is intentional");
  }

  auto n = numRows;
  if (extendCost || costLimit < FLT_MAX) {
    n = numRows + numCols;
    cost_f.conservativeResize(n, n);
    const auto max_cost = cost_f.maxCoeff();
    cost_f.topRightCorner(numRows, n - numCols) =
        costLimit < FLT_MAX ? costLimit / 2.0 : max_cost;
    cost_f.bottomLeftCorner(n - numRows, numCols) =
        costLimit < FLT_MAX ? costLimit / 2.0 : max_cost;
    cost_f.bottomRightCorner(n - numRows, n - numCols) = 0.0;
  }

  auto *p_u = new float[sizeof(float) * n];
  auto *p_v = new float[sizeof(float) * n];

  ArrayR<int, 1, -1> p_x_c(n);
  ArrayR<int, 1, -1> p_y_c(n);

  auto opt = simd_flags.hasAVX2()
                 ? lap<true, int, float>(n, cost_f.data(), /*verbose=*/false,
                                         p_x_c.data(), p_y_c.data(), p_u, p_v)
                 : lap<false, int, float>(n, cost_f.data(), /*verbose=*/false,
                                          p_x_c.data(), p_y_c.data(), p_u, p_v);

  if (n != numRows) {
    p_x_c = (p_x_c >= static_cast<int>(numCols)).select(-1, p_x_c);
    p_y_c = (p_y_c >= static_cast<int>(numRows)).select(-1, p_y_c);
    rRowsol.assign(p_x_c.data(), p_x_c.data() + numRows);
    rColsol.assign(p_y_c.data(), p_y_c.data() + numCols);
  }

  delete[] p_u;
  delete[] p_v;

  return opt;
}

cv::Mat Letterbox(const cv::Mat &rImage, int targetHeight, int targetWidth) {
  auto shape = rImage.size();
  auto height = static_cast<double>(shape.height);
  auto width = static_cast<double>(shape.width);
  auto ratio = std::min(static_cast<double>(targetHeight) / height,
                        static_cast<double>(targetWidth) / width);
  auto new_shape =
      cv::Size(std::round(width * ratio), std::round(height * ratio));
  auto height_padding =
      static_cast<double>(targetHeight - new_shape.height) / 2.0;
  auto width_padding = static_cast<double>(targetWidth - new_shape.width) / 2.0;
  int top = std::round(height_padding - 0.1);
  int bottom = std::round(height_padding + 0.1);
  int left = std::round(width_padding - 0.1);
  int right = std::round(width_padding + 0.1);

  cv::Mat image;
  cv::resize(rImage, image, new_shape, 0.0, 0.0, cv::INTER_AREA);
  cv::copyMakeBorder(image, image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));

  return image;
}

void TransformCoords(torch::Tensor &rCoords, const Vec2D<double> &rCenter,
                     const double scale, const Vec2D<int> &rOutputSize) {
  auto matrix = GetAffineTransform(rCenter, scale, /*rot=*/0, rOutputSize,
                                   /*rShift=*/{0.0, 0.0}, /*inv=*/true);
  matrix = matrix.t();
  auto mat =
      torch::from_blob(matrix.data, {matrix.rows, matrix.cols}, torch::kDouble)
          .to(torch::kFloat32);
  auto xy1 = rCoords.slice(1, 0, 2).contiguous();
  auto xy2 = rCoords.slice(1, 2, 4).contiguous();
  rCoords.slice(1, 0, 2) =
      torch::matmul(xy1, mat.slice(0, 0, 2)) + mat.select(0, 2);
  rCoords.slice(1, 2, 4) =
      torch::matmul(xy2, mat.slice(0, 0, 2)) + mat.select(0, 2);
}

torch::Tensor TransposeAndGatherFeat(const torch::Tensor &rFeat,
                                     const torch::Tensor &rIndices) {
  auto feat = rFeat.permute({0, 2, 3, 1}).contiguous();
  feat = feat.view({feat.size(0), -1, feat.size(3)});

  return GatherFeat(feat, rIndices);
}

void Visualize(cv::Mat image, const std::vector<TrackOutput> &rResults,
               const int frameId) {
  std::stringstream text_stream;
  text_stream << "frame: " << frameId << " num: " << rResults.size();
  cv::putText(image, text_stream.str(), cv::Point(0, 20),
              cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
  for (const auto &r_result : rResults) {
    const auto tlwh = r_result.tlwh;
    const auto tid = r_result.track_id;
    const auto color = GetColor(tid);
    cv::rectangle(image, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), color,
                  2);
    cv::putText(image, std::to_string(tid), cv::Point(tlwh[0], tlwh[1]), 0, 0.6,
                color, 2);
  }
}
}  // namespace util
}  // namespace fairmot
