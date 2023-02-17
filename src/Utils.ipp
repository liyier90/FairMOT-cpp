#ifndef SRC_UTILS_IPP_
#define SRC_UTILS_IPP_

#ifndef SRC_UTILS_HPP_
#error __FILE__ should only be include from DataType.hpp
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace fairmot {
namespace util {
template <typename Matrix, typename Row, typename Value>
void Map(Matrix &rTarget, const std::vector<std::vector<Value>> &rSource) {
  auto idx = 0;
  for (const auto &r_row : rSource) {
    rTarget.row(idx++) = Eigen::Map<const Row>(r_row.data(), r_row.size());
  }
}

template <typename Matrix, typename Row, typename Value>
void Map(Matrix &rTarget, const std::vector<Value> &rSource, const int numRows,
         const int numCols) {
  for (auto i = 0; i < numRows; ++i) {
    rTarget.row(i) =
        Eigen::Map<const Row>(rSource.data() + i * numCols, numCols);
  }
}
}  // namespace util
}  // namespace fairmot

#endif  // SRC_UTILS_IPP_
