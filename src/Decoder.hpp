#ifndef SRC_DECODER_HPP_
#define SRC_DECODER_HPP_

#include <torch/torch.h>

#include <array>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <utility>

namespace fairmot {
class Decoder {
 public:
  Decoder(int maxPerImage, int downRatio);

  std::pair<torch::Tensor, torch::Tensor> operator()(
      const torch::Tensor &rHeatmap, const torch::Tensor &rSize,
      const torch::Tensor &rOffset, const cv::Size &rOrigShape,
      const torch::IntArrayRef &rInputShape);

 private:
  void CtdetPostProcess(torch::Tensor &rDetections,
                        const std::array<double, 2> &rCenter,
                        const double scale,
                        const std::array<int, 2> &rOutputSize);
  torch::Tensor Nms(const torch::Tensor &rHeatmap, int kernel = 3);

  torch::Tensor PostProcess(const torch::Tensor &rDetections,
                            const cv::Size &rOrigShape,
                            const torch::IntArrayRef &rInputShape);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor>
  TopK(const torch::Tensor &rScores);

  const int mMaxPerImage;
  const int mDownRatio;
};
}  // namespace fairmot

#endif  // SRC_DECODER_HPP_

