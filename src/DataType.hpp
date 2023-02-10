#ifndef SRC_DATATYPE_HPP_
#define SRC_DATATYPE_HPP_

#include <Eigen/Dense>
#include <array>

namespace fairmot {
// Trick to specialize alias templates
template <int... Sizes>
struct MatrixRHelper {};

template <int Rows, int Cols>
struct MatrixRHelper<Rows, Cols> {
  using Type =
      Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor | Eigen::DontAlign>;
};

template <int Rows>
struct MatrixRHelper<Rows> {
  using Type =
      Eigen::Matrix<float, Rows, Rows, Eigen::RowMajor | Eigen::DontAlign>;
};

template <int... Sizes>
using MatrixR = typename MatrixRHelper<Sizes...>::Type;

template <int NumCols>
using RowVectorR =
    Eigen::Matrix<float, 1, NumCols, Eigen::RowMajor | Eigen::DontAlign>;

template <int NumCols>
using RowVectorA = Eigen::Array<float, 1, NumCols>;

typedef std::array<float, 4> BBox;
typedef std::array<float, 128> Embedding;
}  // namespace fairmot

#endif  // SRC_DATATYPE_HPP_

