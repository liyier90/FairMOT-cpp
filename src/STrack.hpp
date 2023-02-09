#ifndef SRC_STRACK_HPP_
#define SRC_STRACK_HPP_

#include <vector>

#include "DataType.hpp"
#include "KalmanFilter.hpp"

namespace fairmot {
enum class TrackState { kNEW = 0, kTRACKED, kLOST, kREMOVED };

class STrack {
 public:
  static const KalmanFilter kSharedKalman;

  STrack(const std::vector<float> &rTlwh, const float score,
         const std::vector<float> &rFeat, const int bufferSize = 30);

  static void MultiPredict(const std::vector<STrack *> &rStracks);

  static const std::vector<float> &rXyxyToTlwh(std::vector<float> &rXyxy);

  void Activate(const int frameId);

  void MarkLost() { mState = TrackState::kLOST; }
  void MarkRemoved() { mState = TrackState::kREMOVED; }
  const int &rEndFrame() const { return mFrameId; }

  int NextId();

  RowVector4fR TlwhToXyah(const std::vector<float> &rTlwh) const;

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
  const float mScore;

  int mFrameId = 0;
  int mStartFrame = 0;
  int mTrackletLen = 0;

  RowVector8fR mMean;
  Matrix8fR mCovariance;

 private:
  std::vector<float> mTlwhCache;
};

namespace strack_util {
std::vector<STrack *> CombineStracks(const std::vector<STrack *> &rStracks1,
                                     const std::vector<STrack> &rStracks2);

std::vector<STrack> CombineStracks(const std::vector<STrack> &rStracks1,
                                   const std::vector<STrack> &rStracks2);

std::vector<STrack> SubstractStracks(const std::vector<STrack> &rStracks1,
                                     const std::vector<STrack> &rStracks2);

void RemoveDuplicateStracks(const std::vector<STrack> &rStracks1,
                            const std::vector<STrack> &rStracks2,
                            std::vector<STrack> &rRes1,
                            std::vector<STrack> &rRes2);
}  // namespace strack_util
}  // namespace fairmot

#endif  // SRC_STRACK_HPP_

