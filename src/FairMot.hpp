#ifndef SRC_FAIRMOT_HPP_
#define SRC_FAIRMOT_HPP_

#include <torch/script.h>
#include <torchvision/vision.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "Decoder.hpp"

namespace fairmot {
class FairMot {
 public:
  FairMot(const std::string &rModelPath, int maxPerImage);

  ~FairMot();

  void Track(const cv::Mat &rImage);

  std::pair<torch::Tensor, torch::Tensor> Predict(const cv::Mat &rPaddedImage,
                                                  const cv::Mat &rImage);

  void Update(const torch::Tensor &rDetections,
              const torch::Tensor &rEmbeddings);

 private:
  cv::Mat Preprocess(cv::Mat image);

  torch::jit::script::Module mModel;
  torch::Device *mpDevice;

  Decoder mDecoder;

  const int mInputHeight;
  const int mInputWidth;
  const double mScoreThreshold;

  int mFrameId;
};
}  // namespace fairmot

#endif  // SRC_FAIRMOT_HPP_

