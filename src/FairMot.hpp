#ifndef SRC_FAIRMOT_HPP_
#define SRC_FAIRMOT_HPP_

#include <torch/script.h>
#include <torchvision/vision.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "Decoder.hpp"
#include "STrack.hpp"

namespace fairmot {
class FairMot {
 public:
  FairMot(const std::string &rModelPath, const double frameRate,
          const int maxPerImage, const int trackBuffer);

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
  const int mMaxTimeLost;

  int mFrameId;

  std::vector<STrack> mLostStracks;
  std::vector<STrack> mTrackedStracks;
  std::vector<STrack> mRemoveStracks;
};
}  // namespace fairmot

#endif  // SRC_FAIRMOT_HPP_

