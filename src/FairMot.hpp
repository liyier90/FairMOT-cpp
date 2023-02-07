#ifndef SRC_FAIRMOT_HPP_
#define SRC_FAIRMOT_HPP_

#include <string>

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torchvision/vision.h>

#include "Decoder.hpp"

namespace fairmot {
class FairMot {
 public:
  FairMot(
      const std::string &rModelPath,
      int maxPerImage);

  ~FairMot();

  void Track(const cv::Mat &rImage);

  void Update(
      const cv::Mat &rPaddedImage,
      const cv::Mat &rImage);

 private:
  cv::Mat Preprocess(cv::Mat image);

  torch::jit::script::Module mModel;
  torch::Device *mpDevice;

  Decoder mDecoder;

  const int mInputHeight;
  const int mInputWidth;
};
}  // namespace fairmot

#endif  // SRC_FAIRMOT_HPP_

