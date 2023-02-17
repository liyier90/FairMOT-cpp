#ifndef SRC_DATATYPE_HPP_
#define SRC_DATATYPE_HPP_

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace fairmot {
static constexpr std::size_t kBBoxSize = 4;
static constexpr std::size_t kEmbeddingSize = 128;

typedef std::array<float, kBBoxSize> BBox;
typedef std::array<float, kEmbeddingSize> Embedding;

class STrack;
typedef std::shared_ptr<STrack> STrackPtr;

struct TrackOutput {
  BBox tlwh;
  int track_id;
  float score;
};

// Trick to specialize alias templates
template <typename Scalar, int... Sizes>
struct ArrayRHelper;

template <typename Scalar, int Rows, int Cols>
struct ArrayRHelper<Scalar, Rows, Cols>;

template <typename Scalar, int Rows>
struct ArrayRHelper<Scalar, Rows>;

template <int... Sizes>
struct MatrixHelper;

template <int Rows, int Cols>
struct MatrixHelper<Rows, Cols>;

template <int Rows>
struct MatrixHelper<Rows>;

template <int... Sizes>
struct MatrixRHelper;

template <int Rows, int Cols>
struct MatrixRHelper<Rows, Cols>;

template <int Rows>
struct MatrixRHelper<Rows>;

template <typename Scalar, int... Sizes>
using ArrayR = typename ArrayRHelper<Scalar, Sizes...>::Type;

template <int... Sizes>
using Matrix = typename MatrixHelper<Sizes...>::Type;

template <int... Sizes>
using MatrixR = typename MatrixRHelper<Sizes...>::Type;

template <int NumCols>
using RowVecA = Eigen::Array<float, 1, NumCols>;

template <int NumCols>
using RowVecR = Eigen::Matrix<float, 1, NumCols, Eigen::RowMajor>;

template <typename Scalar>
using Vec2D = std::array<Scalar, 2>;
}  // namespace fairmot

#include "DataType.ipp"

#endif  // SRC_DATATYPE_HPP_

