#ifndef SRC_KALMANFILTER_HPP_
#define SRC_KALMANFILTER_HPP_

#include <array>
#include <utility>
#include <vector>

#include "DataType.hpp"

namespace fairmot {
class KalmanFilter {
 public:
  static const std::array<double, 10> kChi2Inv95;

  KalmanFilter();

  Eigen::RowVectorXf GatingDistance(
      const RowVecR<8> &rMean, const MatrixR<8> &rCovariance,
      const std::vector<RowVecR<4>> &rMeasurements,
      const bool onlyPosition) const;

  std::pair<RowVecR<8>, MatrixR<8>> Initiate(
      const RowVecR<4> &rMeasurement) const;

  void Predict(RowVecR<8> &rMean, MatrixR<8> &rCovariance) const;

  std::pair<RowVecR<4>, MatrixR<4>> Project(
      const RowVecR<8> &rMean, const MatrixR<8> &rCovariance) const;

  std::pair<RowVecR<8>, MatrixR<8>> Update(
      const RowVecR<8> &rMean, const MatrixR<8> &rCovariance,
      const RowVecR<4> &rMeasurement) const;

 private:
  static constexpr int kNDim = 4;
  static constexpr double kDt = 1.0;
  MatrixR<8> mMotionMat;
  MatrixR<4, 8> mUpdateMat;
  const float mStdWeightPosition;
  const float mStdWeightVelocity;
};
}  // namespace fairmot

#endif  // SRC_KALMANFILTER_HPP_

