#include "STrack.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

#include "DataType.hpp"

namespace fairmot {

STrack::STrack(const std::vector<float> &rTlwh, const float score,
               const std::vector<float> &rFeat, const int bufferSize)
    : mMean{RowVector8fR::Zero()},
      mCovariance{Matrix8fR::Zero()},
      mTlwhCache{rTlwh} {
  this->UpdateTlwh();
  this->UpdateXyxy();
  this->UpdateFeatures(rFeat);
}

void STrack::UpdateFeatures(const std::vector<float> &rFeat) {
  Eigen::VectorXf features =
      Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned>(rFeat.data(),
                                                          rFeat.size());
  features = features.normalized();
  mCurrFeat.assign(&features[0], features.data() + features.size());
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
}  // namespace fairmot

