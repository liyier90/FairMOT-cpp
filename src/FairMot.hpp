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

  FairMot(const FairMot &) = delete;

  ~FairMot();

  FairMot operator=(const FairMot &) = delete;

  std::pair<torch::Tensor, torch::Tensor> Predict(const cv::Mat &rPaddedImage,
                                                  const cv::Mat &rImage);

  std::vector<TrackOutput> Track(const cv::Mat &rImage);

  std::vector<STrack> Update(const torch::Tensor &rDetections,
                             const torch::Tensor &rEmbeddings);

 private:
  std::vector<TrackOutput> Postprocess(
      const std::vector<STrack> &rOnlineTargets);
  cv::Mat Preprocess(cv::Mat image);

  torch::jit::script::Module mModel;
  torch::Device *mpDevice = nullptr;

  Decoder mDecoder;

  const int mInputHeight;
  const int mInputWidth;
  const float mMinBoxArea;
  const double mScoreThreshold;
  const int mMaxTimeLost;

  int mFrameId;

  std::vector<STrackPtr> mLostStracks;
  std::vector<STrackPtr> mTrackedStracks;
  std::vector<STrackPtr> mRemoveStracks;
};
}  // namespace fairmot

#endif  // SRC_FAIRMOT_HPP_

