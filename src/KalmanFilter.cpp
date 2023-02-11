#include "KalmanFilter.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "DataType.hpp"

namespace fairmot {
const std::array<double, 10> KalmanFilter::kChi2Inv95 = {
    0.0,    3.8415, 5.9915, 7.8147, 9.4877,
    11.070, 12.592, 14.067, 15.507, 16.919};

KalmanFilter::KalmanFilter()
    : mMotionMat{MatrixR<8>::Identity()},
      mUpdateMat{MatrixR<4, 8>::Identity()},
      mStdWeightPosition{1.0 / 20.0},
      mStdWeightVelocity{1.0 / 160.0} {}

Eigen::RowVectorXf KalmanFilter::GatingDistance(
    const RowVectorR<8> &rMean, const MatrixR<8> &rCovariance,
    const std::vector<RowVectorR<4>> &rMeasurements,
    const bool onlyPosition) const {
  if (onlyPosition) {
    throw std::logic_error("'onlyPosition = true' not implemented.");
  }
  RowVectorR<4> mean;
  MatrixR<4> covariance;
  std::tie(mean, covariance) = this->Project(rMean, rCovariance);
  MatrixR<-1, 4> distances(rMeasurements.size(), 4);
  auto pos = 0;
  for (const auto &r_box : rMeasurements) {
    distances.row(pos++) = r_box - mean;
  }
  MatrixR<-1, -1> factor = covariance.llt().matrixL();
  Eigen::ArrayXXf z = factor.triangularView<Eigen::Lower>()
                          .solve<Eigen::OnTheRight>(distances)
                          .transpose()
                          .array();
  auto squared_maha = (z * z).matrix().colwise().sum();

  return squared_maha;
}

std::pair<RowVectorR<8>, MatrixR<8>> KalmanFilter::Initiate(
    const RowVectorR<4> &rMeasurement) const {
  RowVectorR<8> mean;
  mean << rMeasurement, RowVectorR<4>::Zero();

  RowVectorA<8> std_dev;
  // clang-format off
  std_dev <<
      2.0 * mStdWeightPosition * rMeasurement[3],
      2.0 * mStdWeightPosition * rMeasurement[3],
      1e-2,
      2.0 * mStdWeightPosition * rMeasurement[3],
      10.0 * mStdWeightVelocity * rMeasurement[3],
      10.0 * mStdWeightVelocity * rMeasurement[3],
      1e-5,
      10.0 * mStdWeightVelocity * rMeasurement[3];
  // clang-format on
  MatrixR<8> covariance = std_dev.square().matrix().asDiagonal();

  return std::make_pair(mean, covariance);
}

void KalmanFilter::Predict(RowVectorR<8> &rMean,
                           MatrixR<8> &rCovariance) const {
  // TODO: Look into MultiPredict
  RowVectorA<8> std_pos_vel;
  // clang-format off
  std_pos_vel <<
      mStdWeightPosition * rMean[3],
      mStdWeightPosition * rMean[3],
      1e-2,
      mStdWeightPosition * rMean[3],
      mStdWeightVelocity * rMean[3],
      mStdWeightVelocity * rMean[3],
      1e-5,
      mStdWeightVelocity * rMean[3];
  // clang-format on
  RowVectorR<8> mean = mMotionMat * rMean.transpose();
  MatrixR<8> covariance = mMotionMat * rCovariance * mMotionMat.transpose();
  covariance += std_pos_vel.square().matrix().asDiagonal();

  rMean = std::move(mean);
  rCovariance = std::move(covariance);
}

std::pair<RowVectorR<4>, MatrixR<4>> KalmanFilter::Project(
    const RowVectorR<8> &rMean, const MatrixR<8> &rCovariance) const {
  RowVectorR<4> std_dev;
  // clang-format off
  std_dev <<
      mStdWeightPosition * rMean[3],
      mStdWeightPosition * rMean[3],
      1e-1,
      mStdWeightPosition * rMean[3];
  // clang-format on
  RowVectorR<4> mean = mUpdateMat * rMean.transpose();
  MatrixR<4> covariance = mUpdateMat * rCovariance * mUpdateMat.transpose();
  Eigen::Matrix4f diag = std_dev.asDiagonal();
  diag = diag.array().square().matrix();
  covariance += diag;
  return std::make_pair(mean, covariance);
}
}  // namespace fairmot
