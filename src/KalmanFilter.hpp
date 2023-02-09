#ifndef SRC_KALMANFILTER_HPP_
#define SRC_KALMANFILTER_HPP_

#include <utility>

#include "DataType.hpp"

namespace fairmot {
class KalmanFilter {
 public:
  KalmanFilter();

  std::pair<RowVector8fR, Matrix8fR> Initiate(
      const RowVector4fR &rMeasurement) const;

 private:
  static constexpr int kNDim = 4;
  static constexpr double kDt = 1.0;
  const float mStdWeightPosition;
  const float mStdWeightVelocity;
};
}  // namespace fairmot

#endif  // SRC_KALMANFILTER_HPP_

