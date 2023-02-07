#include "Utils.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(
    const torch::Tensor &rFeat,
    const torch::Tensor &rIndices) {
  const auto dim = rFeat.size(2);
  const auto indices = rIndices.unsqueeze(2).expand(
      {rIndices.size(0), rIndices.size(1), dim});
  const auto feat = rFeat.gather(1, indices);

  return feat;
}

cv::Mat Letterbox(
    cv::Mat image,
    int targetHeight,
    int targetWidth) {
  auto shape = image.size();
  auto height = static_cast<double>(shape.height);
  auto width = static_cast<double>(shape.width);
  auto ratio = std::min(static_cast<double>(targetHeight) / height,
      static_cast<double>(targetWidth) / width);
  auto new_shape = cv::Size(std::round(width * ratio),
      std::round(height * ratio));
  auto height_padding = static_cast<double>(targetHeight - new_shape.height) /
      2.0;
  auto width_padding = static_cast<double>(targetWidth - new_shape.width) /
      2.0;
  int top = std::round(height_padding - 0.1);
  int bottom = std::round(height_padding + 0.1);
  int left = std::round(width_padding - 0.1);
  int right = std::round(width_padding + 0.1);

  cv::resize(image, image, new_shape, cv::INTER_AREA);
  cv::copyMakeBorder(image, image, top, bottom, left, right,
      cv::BORDER_CONSTANT, cv::Scalar(127.5, 127.5, 127.5));
  return image;
}

torch::Tensor TransposeAndGatherFeat(
    const torch::Tensor &rFeat,
    const torch::Tensor &rIndices) {
  auto feat = rFeat.permute({0, 2, 3, 1}).contiguous();
  feat = feat.view({feat.size(0), -1, feat.size(3)});

  return GatherFeat(feat, rIndices);
}
}  // namespace util
}  // namespace fairmot
