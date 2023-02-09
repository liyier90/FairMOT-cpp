#include "STrack.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataType.hpp"
#include "KalmanFilter.hpp"

namespace fairmot {
const KalmanFilter STrack::kSharedKalman;

STrack::STrack(const std::vector<float> &rTlwh, const float score,
               const std::vector<float> &rFeat, const int bufferSize)
    : mScore{score},
      mMean{RowVector8fR::Zero()},
      mCovariance{Matrix8fR::Zero()},
      mTlwhCache{rTlwh} {
  this->UpdateTlwh();
  this->UpdateXyxy();
  this->UpdateFeatures(rFeat);
}

void STrack::MultiPredict(const std::vector<STrack *> &rStracks) {}

void STrack::Activate(const int frameId) {
  mTrackId = this->NextId();
  auto xyah = this->TlwhToXyah(mTlwhCache);

  std::tie(mMean, mCovariance) = STrack::kSharedKalman.Initiate(xyah);

  this->UpdateTlwh();
  this->UpdateXyxy();

  mTrackletLen = 0;
  mState = TrackState::kTRACKED;
  if (frameId == 1) {
    mIsActivated = true;
  }
  mFrameId = frameId;
  mStartFrame = frameId;
}

int STrack::NextId() {
  static int count = 0;
  ++count;
  return count;
}

RowVector4fR STrack::TlwhToXyah(const std::vector<float> &rTlwh) const {
  RowVector4fR xyah =
      Eigen::Map<const RowVector4fR>(rTlwh.data(), rTlwh.size());
  xyah({0, 1}) += xyah({2, 3}) / 2.0;
  xyah[2] /= xyah[3];

  return xyah;
}

const std::vector<float> &STrack::rXyxyToTlwh(std::vector<float> &rXyxy) {
  rXyxy[2] -= rXyxy[0];
  rXyxy[3] -= rXyxy[1];
  return std::move(rXyxy);
}

void STrack::UpdateFeatures(const std::vector<float> &rFeat) {
  Eigen::VectorXf features =
      Eigen::Map<const Eigen::VectorXf>(rFeat.data(), rFeat.size());
  features = features.normalized();
  mCurrFeat.assign(&features[0], features.data() + features.size());
  if (!mSmoothFeat.empty()) {
    // TODO: Check if this is better than assigning through for loop
    Eigen::VectorXf smooth_feat =
        Eigen::Map<Eigen::VectorXf>(mSmoothFeat.data(), mSmoothFeat.size());
    features = mAlpha * smooth_feat + (1.0 - mAlpha) * features;
  }
  features = features.normalized();
  mSmoothFeat.assign(&features[0], features.data() + features.size());
}

void STrack::UpdateTlwh() {
  if (mMean.isZero()) {
    mTlwh = mTlwhCache;
    return;
  }
  mTlwh[0] = mMean[0];
  mTlwh[1] = mMean[1];
  mTlwh[2] = mMean[2];
  mTlwh[3] = mMean[3];

  mTlwh[2] *= mTlwh[3];
  mTlwh[0] -= mTlwh[2] / 2.0;
  mTlwh[1] -= mTlwh[3] / 2.0;
}

void STrack::UpdateXyxy() {
  mXyxy = mTlwh;
  mXyxy[2] += mXyxy[0];
  mXyxy[3] += mXyxy[1];
}

namespace strack_util {
std::vector<STrack *> CombineStracks(const std::vector<STrack *> &rStracks1,
                                     const std::vector<STrack> &rStracks2) {
  std::unordered_set<int> exists;
  auto res = rStracks1;
  for (const auto &r_strack : rStracks1) {
    exists.insert(r_strack->mTrackId);
  }
  for (const auto &r_strack : rStracks2) {
    const auto tid = r_strack.mTrackId;
    if (exists.count(tid) == 0) {
      exists.insert(tid);
      res.push_back(const_cast<STrack *>(&r_strack));
    }
  }

  return res;
}

std::vector<STrack> CombineStracks(const std::vector<STrack> &rStracks1,
                                   const std::vector<STrack> &rStracks2) {
  std::unordered_set<int> exists;
  auto res = rStracks1;
  for (const auto &r_strack : rStracks1) {
    exists.insert(r_strack.mTrackId);
  }
  for (const auto &r_strack : rStracks2) {
    const auto tid = r_strack.mTrackId;
    if (exists.count(tid) == 0) {
      res.push_back(r_strack);
    }
  }

  return res;
}

void RemoveDuplicateStracks(const std::vector<STrack> &rStracks1,
                            const std::vector<STrack> &rStracks2,
                            std::vector<STrack> &rRes1,
                            std::vector<STrack> &rRes2) {}

std::vector<STrack> SubstractStracks(const std::vector<STrack> &rStracks1,
                                     const std::vector<STrack> &rStracks2) {
  std::unordered_map<int, STrack> stracks;
  for (const auto &r_track : rStracks1) {
    stracks.insert(std::make_pair(r_track.mTrackId, r_track));
  }
  for (const auto &r_track : rStracks2) {
    const auto tid = r_track.mTrackId;
    if (stracks.count(tid) != 0) {
      stracks.erase(tid);
    }
  }
  std::vector<STrack> res(stracks.size());
  for (const auto &r_track : stracks) {
    res.push_back(r_track.second);
  }

  return res;
}
}  // namespace strack_util
}  // namespace fairmot

