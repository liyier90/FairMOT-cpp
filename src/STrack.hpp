#ifndef SRC_STRACK_HPP_
#define SRC_STRACK_HPP_

#include <vector>

#include "DataType.hpp"

namespace fairmot {
enum class TrackState { kNEW = 0, kTRACKED, kLOST, kREMOVED };

class STrack {
 public:
  STrack(const std::vector<float> &rTlwh, const float score,
         const std::vector<float> &rFeat, const int bufferSize = 30);

  void UpdateFeatures(const std::vector<float> &rFeat);

  void UpdateTlwh();
  void UpdateXyxy();

  int mTrackId = 0;
  bool mIsActivated = false;
  TrackState mState = TrackState::kNEW;

  std::vector<float> mCurrFeat;
  std::vector<float> mSmoothFeat;
  double mAlpha = 0.9;

  std::vector<float> mTlwh;
  std::vector<float> mXyxy;

  int mFrameId = 0;
  int mStartFrame = 0;
  int mTrackletLen = 0;

  RowVector8fR mMean;
  Matrix8fR mCovariance;

 private:
  std::vector<float> mTlwhCache;
};
}  // namespace fairmot

#endif  // SRC_STRACK_HPP_

