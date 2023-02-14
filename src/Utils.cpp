#include "Utils.hpp"

#include <torch/torch.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

#include "Lapjv.hpp"

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

cv::Mat GetAffineTransform(const std::array<double, 2> &rCenter,
                           const double scale, const double rot,
                           const std::array<int, 2> &rOutputSize,
                           const std::array<double, 2> &rShift,
                           const bool inv) {
  const auto src_dir = GetDir({0.0, scale * -0.5}, M_PI * rot / 180.0);
  const std::vector<double> dst_dir = {0.0, rOutputSize[0] * -0.5};

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

std::vector<double> GetDir(const torch::ArrayRef<double> &rSrcPoint,
                           const double rotRad) {
  const auto sin = std::sin(rotRad);
  const auto cos = std::cos(rotRad);

  std::vector<double> src_result = {rSrcPoint[0] * cos - rSrcPoint[1] * sin,
                                    rSrcPoint[0] * sin + rSrcPoint[1] * cos};

  return src_result;
}

void GetThirdPoint(const cv::Point2f &rPoint1, const cv::Point2f &rPoint2,
                   cv::Point2f &rThirdPoint) {
  rThirdPoint.x = rPoint2.x + rPoint2.y - rPoint1.y;
  rThirdPoint.y = rPoint2.y + rPoint2.x - rPoint1.x;
}

double Lapjv(const std::vector<std::vector<float>> &rCost,
             std::vector<int> &rRowsol, std::vector<int> &rColsol,
             bool extendCost, float costLimit, bool returnCost) {
  auto cost_c = rCost;
  const auto n_rows = rCost.size();
  const auto n_cols = rCost[0].size();
  Eigen::ArrayXXf cost_mat(n_rows, n_cols);
  Map<Eigen::ArrayXXf, Eigen::ArrayXf, float>(cost_mat, cost_c);

  rRowsol.reserve(n_rows);
  rColsol.reserve(n_cols);

  if (n_rows != n_cols && !extendCost) {
    throw std::logic_error(
        "Square cost array expected. Pass extendCost = true if non-square cost "
        "is intentional");
  }

  auto n = n_rows;
  if (extendCost || costLimit < FLT_MAX) {
    std::vector<std::vector<float>> cost_c_extended;
    n = n_rows + n_cols;
    Eigen::ArrayXXf cost_mat_extended(n, n);
    if (costLimit < FLT_MAX) {
      cost_mat_extended = costLimit / 2.0;
      cost_c_extended.assign(n, std::vector<float>(n, costLimit / 2.0));
    } else {
      cost_mat_extended = cost_mat.maxCoeff();
      float max_cost = 0.0;
      for (const auto &r_row : cost_c) {
        for (const auto &r_cost : r_row) {
          if (r_cost > max_cost) {
            max_cost = r_cost;
          }
        }
      }
      cost_c_extended.assign(n, std::vector<float>(n, max_cost + 1.0));
    }
    for (auto i = n_rows; i < n; ++i) {
      for (auto j = n_cols; j < n; ++j) {
        cost_c_extended[i][j] = 0.0;
      }
    }
    cost_mat_extended.bottomRightCorner(n - n_rows, n - n_cols) = 0.0;
    cost_mat_extended.topLeftCorner(n_rows, n_cols) = cost_mat;
    for (auto i = 0u; i < n_rows; ++i) {
      for (auto j = 0u; j < n_cols; ++j) {
        cost_c_extended[i][j] = cost_c[i][j];
      }
    }
    cost_c = cost_c_extended;
    cost_mat = cost_mat_extended;
  }

  auto **p_cost = new double *[sizeof(double *) * n];
  for (auto i = 0u; i < n; ++i) {
    p_cost[i] = new double[sizeof(double) * n];
  }
  for (auto i = 0u; i < n; ++i) {
    for (auto j = 0u; j < n; ++j) {
      p_cost[i][j] = cost_c[i][j];
    }
  }

  auto *p_x_c = new int[sizeof(int) * n];
  auto *p_y_c = new int[sizeof(int) * n];
  auto ret = lapjv_internal(n, p_cost, p_x_c, p_y_c);
  if (ret != 0) {
    throw std::logic_error("Calculate wrong!");
  }

  auto opt = 0.0;
  if (n != n_rows) {
    for (auto i = 0u; i < n; ++i) {
      if (p_x_c[i] >= static_cast<int>(n_cols)) {
        p_x_c[i] = -1;
      }
      if (p_y_c[i] >= static_cast<int>(n_rows)) {
        p_y_c[i] = -1;
      }
    }
    for (auto i = 0u; i < n_rows; ++i) {
      rRowsol.push_back(p_x_c[i]);
    }
    for (auto i = 0u; i < n_cols; ++i) {
      rColsol.push_back(p_y_c[i]);
    }

    if (returnCost) {
      for (auto i = 0u; i < n_rows; ++i) {
        if (rRowsol[i] != -1) {
          opt += p_cost[i][rRowsol[i]];
        }
      }
    }
  } else if (returnCost) {
    for (auto i = 0u; i < n_rows; ++i) {
      opt += p_cost[i][rRowsol[i]];
    }
  }

  for (auto i = 0u; i < n; ++i) {
    delete[] p_cost[i];
  }
  delete[] p_cost;
  delete[] p_x_c;
  delete[] p_y_c;

  return opt;
}

cv::Mat Letterbox(cv::Mat image, int targetHeight, int targetWidth) {
  auto shape = image.size();
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

  cv::resize(image, image, new_shape, cv::INTER_AREA);
  cv::copyMakeBorder(image, image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));

  return image;
}

void TransformCoords(torch::Tensor &rCoords,
                     const std::array<double, 2> &rCenter, const double scale,
                     const std::array<int, 2> &rOutputSize) {
  const auto matrix = GetAffineTransform(rCenter, scale, /*rot=*/0, rOutputSize,
                                         /*rShift=*/{0.0, 0.0}, /*inv=*/true);
  auto x1 = rCoords.select(1, 0).contiguous();
  auto y1 = rCoords.select(1, 1).contiguous();
  auto x2 = rCoords.select(1, 2).contiguous();
  auto y2 = rCoords.select(1, 3).contiguous();

  rCoords.select(1, 0) = matrix.at<double>(0, 0) * x1 +
                         matrix.at<double>(0, 1) * y1 + matrix.at<double>(0, 2);
  rCoords.select(1, 1) = matrix.at<double>(1, 0) * x1 +
                         matrix.at<double>(1, 1) * y1 + matrix.at<double>(1, 2);
  rCoords.select(1, 2) = matrix.at<double>(0, 0) * x2 +
                         matrix.at<double>(0, 1) * y2 + matrix.at<double>(0, 2);
  rCoords.select(1, 3) = matrix.at<double>(1, 0) * x2 +
                         matrix.at<double>(1, 1) * y2 + matrix.at<double>(1, 2);
}

torch::Tensor TransposeAndGatherFeat(const torch::Tensor &rFeat,
                                     const torch::Tensor &rIndices) {
  auto feat = rFeat.permute({0, 2, 3, 1}).contiguous();
  feat = feat.view({feat.size(0), -1, feat.size(3)});

  return GatherFeat(feat, rIndices);
}
}  // namespace util
}  // namespace fairmot
