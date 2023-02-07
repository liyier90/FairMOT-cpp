#ifndef SRC_DECODER_HPP_
#define SRC_DECODER_HPP_

#include <tuple>
#include <utility>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace fairmot {
class Decoder {
 public:
  Decoder(
      int maxPerImage,
      int downRatio);

  std::pair<torch::Tensor, torch::Tensor> operator()(
      const torch::Tensor &rHeatmap,
      const torch::Tensor &rSize,
      const torch::Tensor &rOffset,
      const cv::Size& rOrigShape,
      const torch::IntArrayRef &rInputShape);


 private:
  torch::Tensor Nms(
      const torch::Tensor &rHeatmap,
      int kernel = 3);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
      torch::Tensor>
  TopK(const torch::Tensor &rScores);

  const int mMaxPerImage;
  const int mDownRatio;
};
}  // namespace fairmot

#endif  // SRC_DECODER_HPP_

