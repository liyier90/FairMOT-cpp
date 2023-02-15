#include "STrack.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstring>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataType.hpp"
#include "KalmanFilter.hpp"
#include "Matching.hpp"

namespace fairmot {
const KalmanFilter STrack::kSharedKalman;

// STrack::STrack(const BBox &rTlwh, const float score,
//                const std::vector<float> &rFeat, const int bufferSize)
STrack::STrack(const BBox &rTlwh, const float score, const Embedding &rFeat,
               const int bufferSize)
    : mCurrFeat(),
      mSmoothFeat(),
      mTlwh(),
      mXyxy(),
      mScore{score},
      mMean{RowVecR<8>::Zero()},
      mCovariance{MatrixR<8>::Zero()},
      mTlwhCache{rTlwh} {
  this->UpdateTlwh();
  this->UpdateXyxy();
  this->UpdateFeatures(rFeat);
}

std::ostream &operator<<(std::ostream &rOStream, const STrack &rSTrack) {
  rOStream << "OT_" << rSTrack.mTrackId << "_(" << rSTrack.mStartFrame << "-"
           << rSTrack.rEndFrame() << ")";

  return rOStream;
}

std::ostream &operator<<(std::ostream &rOStream, const STrack *const pSTrack) {
  rOStream << "OT_" << pSTrack->mTrackId << "_(" << pSTrack->mStartFrame << "-"
           << pSTrack->rEndFrame() << ")";

  return rOStream;
}

void STrack::MultiPredict(const std::vector<STrack *> &rStracks) {
  for (const auto &r_track : rStracks) {
    if (r_track->mState != TrackState::kTRACKED) {
      r_track->mMean[7] = 0.0;
    }
    STrack::kSharedKalman.Predict(r_track->mMean, r_track->mCovariance);
    r_track->UpdateTlwh();
    r_track->UpdateXyxy();
  }
}

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

void STrack::ReActivate(const STrack *const pOther, const int frameId,
                        const bool newId) {
  mFrameId = frameId;
  mTrackletLen = 0;

  const auto xyah = this->TlwhToXyah(pOther->mTlwh);
  std::tie(mMean, mCovariance) = kSharedKalman.Update(mMean, mCovariance, xyah);

  this->UpdateTlwh();
  this->UpdateXyxy();

  mState = TrackState::kTRACKED;
  mIsActivated = true;
  if (newId) {
    mTrackId = this->NextId();
  }

  this->UpdateFeatures(pOther->mCurrFeat);
}

RowVecR<4> STrack::TlwhToXyah(const BBox &rTlwh) const {
  RowVecR<4> xyah = Eigen::Map<const RowVecR<4>>(rTlwh.data(), rTlwh.size());
  xyah({0, 1}) += xyah({2, 3}) / 2.0;
  xyah[2] /= xyah[3];

  return xyah;
}

RowVecR<4> STrack::ToXyah() const { return this->TlwhToXyah(mTlwh); }

const BBox &STrack::rXyxyToTlwh(BBox &rXyxy) {
  rXyxy[2] -= rXyxy[0];
  rXyxy[3] -= rXyxy[1];
  return std::move(rXyxy);
}

void STrack::Update(const STrack *const pOther, const int frameId,
                    const bool updateFeature) {
  mFrameId = frameId;
  ++mTrackletLen;

  const auto xyah = this->TlwhToXyah(pOther->mTlwh);
  std::tie(mMean, mCovariance) = kSharedKalman.Update(mMean, mCovariance, xyah);

  this->UpdateTlwh();
  this->UpdateXyxy();

  mState = TrackState::kTRACKED;
  mIsActivated = true;
  mScore = pOther->mScore;
  if (updateFeature) {
    this->UpdateFeatures(pOther->mCurrFeat);
  }
}

void STrack::UpdateFeatures(const Embedding &rFeat) {
  RowVecR<128> features =
      Eigen::Map<const RowVecR<128>>(rFeat.data(), rFeat.size());
  features = features.normalized();
  memcpy(mCurrFeat.data(), features.data(), sizeof(float) * features.size());
  if (mEmptySmoothFeat) {
    mEmptySmoothFeat = false;
  } else {
    RowVecR<128> smooth_feat =
        Eigen::Map<const RowVecR<128>>(mSmoothFeat.data(), mSmoothFeat.size());
    features = mAlpha * smooth_feat + (1.0 - mAlpha) * features;
  }
  features = features.normalized();
  // TODO: Check if memmove can be used
  memcpy(mSmoothFeat.data(), features.data(), sizeof(float) * features.size());
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
                            std::vector<STrack> &rRes2) {
  const auto distances = matching::IouDistance(rStracks1, rStracks2);
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0u; i < distances.size(); ++i) {
    for (auto j = 0u; j < distances[i].size(); ++j) {
      if (distances[i][j] < 0.15) {
        pairs.push_back(std::make_pair(i, j));
      }
    }
  }

  std::vector<int> duplicates_1;
  std::vector<int> duplicates_2;
  for (const auto &r_pair : pairs) {
    const auto idx_1 = r_pair.first;
    const auto idx_2 = r_pair.second;
    const auto age_1 = rStracks1[idx_1].mFrameId - rStracks1[idx_1].mStartFrame;
    const auto age_2 = rStracks2[idx_2].mFrameId - rStracks2[idx_2].mStartFrame;
    if (age_1 > age_2) {
      duplicates_2.push_back(idx_2);
    } else {
      duplicates_1.push_back(idx_1);
    }
  }

  for (auto i = 0u; i < rStracks1.size(); ++i) {
    auto iter = std::find(duplicates_1.begin(), duplicates_1.end(), i);
    if (iter == duplicates_1.end()) {
      rRes1.push_back(rStracks1[i]);
    }
  }

  for (auto i = 0u; i < rStracks2.size(); ++i) {
    auto iter = std::find(duplicates_2.begin(), duplicates_2.end(), i);
    if (iter == duplicates_2.end()) {
      rRes2.push_back(rStracks2[i]);
    }
  }
}

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
  std::vector<STrack> res;
  res.reserve(stracks.size());
  for (const auto &r_track : stracks) {
    res.push_back(r_track.second);
  }

  return res;
}
}  // namespace strack_util
}  // namespace fairmot

