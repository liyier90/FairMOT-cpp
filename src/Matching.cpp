#include "Matching.hpp"

#include <cmath>
#include <numeric>

namespace fairmot {
namespace matching {
void EmbeddingDistance(const std::vector<STrack *> &rTracks,
                       const std::vector<STrack> &rDetections,
                       std::vector<std::vector<float>> &rCostMatrix,
                       int &rNumRows, int &rNumCols) {
  rNumRows = rTracks.size();
  rNumCols = rDetections.size();
  if (rNumRows * rNumCols == 0) {
    return;
  }

  for (int i = 0; i < rNumRows; ++i) {
    std::vector<float> cost_matrix_row(rNumCols);
    auto track_feature = rTracks[i]->mSmoothFeat;
    for (int j = 0; j < rNumCols; ++j) {
      auto det_feature = rDetections[j].mCurrFeat;
      float feat_square = 0.0;
      for (auto k = 0u; k < det_feature.size(); ++k) {
        feat_square += (track_feature[k] - det_feature[k]) *
                       (track_feature[k] - det_feature[k]);
      }
      cost_matrix_row[j] = std::sqrt(feat_square);
    }
    rCostMatrix.push_back(cost_matrix_row);
  }
}

void LinearAssignment(const std::vector<std::vector<float>> &rCostMatrix,
                      const int numRows, const int numCols,
                      const float threshold,
                      std::vector<std::vector<int>> &rMatches,
                      std::vector<int> &rUnmatched1,
                      std::vector<int> &rUnmatched2) {
  if (rCostMatrix.size() == 0) {
    rUnmatched1.resize(numRows);
    rUnmatched2.resize(numCols);
    std::iota(rUnmatched1.begin(), rUnmatched1.end(), 0);
    std::iota(rUnmatched2.begin(), rUnmatched2.end(), 0);
    return;
  }
}
}  // namespace matching
}  // namespace fairmot
