#ifndef SRC_DATATYPE_IPP_
#define SRC_DATATYPE_IPP_

#ifndef SRC_DATATYPE_HPP_
#error __FILE__ should only be include from DataType.hpp
#endif

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace fairmot {
// Trick to specialize alias templates
template <typename Scalar, int... Sizes>
struct ArrayRHelper {};

template <typename Scalar, int Rows, int Cols>
struct ArrayRHelper<Scalar, Rows, Cols> {
  using Type = Eigen::Array<Scalar, Rows, Cols, Eigen::RowMajor>;
};

template <typename Scalar, int Rows>
struct ArrayRHelper<Scalar, Rows> {
  using Type = Eigen::Array<Scalar, Rows, Rows, Eigen::RowMajor>;
};

template <int... Sizes>
struct MatrixHelper {};

template <int Rows, int Cols>
struct MatrixHelper<Rows, Cols> {
  using Type = Eigen::Matrix<float, Rows, Cols>;
};

template <int Rows>
struct MatrixHelper<Rows> {
  using Type = Eigen::Matrix<float, Rows, Rows>;
};

template <int... Sizes>
struct MatrixRHelper {};

template <int Rows, int Cols>
struct MatrixRHelper<Rows, Cols> {
  using Type = Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>;
};

template <int Rows>
struct MatrixRHelper<Rows> {
  using Type = Eigen::Matrix<float, Rows, Rows, Eigen::RowMajor>;
};
}  // namespace fairmot

#endif  // SRC_DATATYPE_IPP_

