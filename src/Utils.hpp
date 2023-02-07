#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace fairmot {
namespace util {
torch::Tensor GatherFeat(
    const torch::Tensor &rFeat,
    const torch::Tensor &rIndices);

cv::Mat Letterbox(
    cv::Mat image,
    int targetHeight,
    int targetWidth);

torch::Tensor TransposeAndGatherFeat(
    const torch::Tensor &rFeat,
    const torch::Tensor &rIndices);
}  // namespace util
}  // namespace fairmot

#endif  // SRC_UTILS_HPP_

