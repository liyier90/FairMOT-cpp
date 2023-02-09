#include "KalmanFilter.hpp"

#include <iostream>

#include "DataType.hpp"

namespace fairmot {
KalmanFilter::KalmanFilter()
    : mStdWeightPosition{1.0 / 20.0}, mStdWeightVelocity{1.0 / 160.0} {}

std::pair<RowVector8fR, Matrix8fR> KalmanFilter::Initiate(
    const RowVector4fR &rMeasurement) const {
  RowVector8fR mean;
  mean << rMeasurement, RowVector4fR::Zero();

  RowVector8fA std_dev;
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
  Matrix8fR covariance = std_dev.square().matrix().asDiagonal();

  return std::make_pair(mean, covariance);
}
}  // namespace fairmot
