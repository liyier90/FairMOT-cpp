#ifndef SRC_STRACK_HPP_
#define SRC_STRACK_HPP_

#include <iostream>
#include <vector>

#include "DataType.hpp"
#include "KalmanFilter.hpp"

namespace fairmot {
enum class TrackState { kNEW = 0, kTRACKED, kLOST, kREMOVED };

class STrack {
 public:
  static const KalmanFilter kSharedKalman;

  STrack(const BBox &rTlwh, const float score, const Embedding &rFeat);

  friend std::ostream &operator<<(std::ostream &rOStream,
                                  const STrack &rSTrack);

  friend std::ostream &operator<<(std::ostream &rOStream,
                                  const STrack *const pSTrack);

  static void MultiPredict(const std::vector<STrack *> &rStracks);

  static const BBox &rXyxyToTlwh(BBox &rXyxy);

  void Activate(const int frameId);

  void MarkLost() { mState = TrackState::kLOST; }
  void MarkRemoved() { mState = TrackState::kREMOVED; }
  const int &rEndFrame() const { return mFrameId; }

  int NextId();

  void ReActivate(const STrack *const pOther, const int frameId,
                  const bool newId = false);

  RowVecR<4> TlwhToXyah(const BBox &rTlwh) const;
  RowVecR<4> ToXyah() const;

  void Update(const STrack *const pOther, const int frameId,
              const bool updateFeature = true);

  void UpdateFeatures(const Embedding &rFeat);

  void UpdateTlwh();
  void UpdateXyxy();

  int mTrackId = 0;
  bool mIsActivated = false;
  TrackState mState = TrackState::kNEW;

  Embedding mCurrFeat;
  Embedding mSmoothFeat;
  double mAlpha = 0.9;

  BBox mTlwh;
  BBox mXyxy;
  float mScore;

  int mFrameId = 0;
  int mStartFrame = 0;
  int mTrackletLen = 0;

  RowVecRU<8> mMean;
  MatrixRU<8> mCovariance;

 private:
  BBox mTlwhCache;
  bool mEmptySmoothFeat = true;
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

