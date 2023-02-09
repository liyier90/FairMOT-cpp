#ifndef SRC_DATATYPE_HPP_
#define SRC_DATATYPE_HPP_

#include <Eigen/Dense>

namespace fairmot {
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> RowVector4fR;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> RowVector8fR;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> Matrix8fR;

typedef Eigen::Array<float, 1, 8> RowVector8fA;
}  // namespace fairmot

#endif  // SRC_DATATYPE_HPP_

