#ifndef SRC_MATCHING_HPP_
#define SRC_MATCHING_HPP_

#include "STrack.hpp"
namespace fairmot {
namespace matching {
void EmbeddingDistance(const std::vector<STrack *> &rTracks,
                       const std::vector<STrack> &rDetections,
                       std::vector<std::vector<float>> &rCostMatrix,
                       int &rNumRows, int &rNumCols);

void LinearAssignment(const std::vector<std::vector<float>> &rCostMatrix,
                      const int numRows, const int numCols,
                      const float threshold,
                      std::vector<std::vector<int>> &rMatches,
                      std::vector<int> &rUnmatched1,
                      std::vector<int> &rUnmatched2);
}  // namespace matching
}  // namespace fairmot

#endif  // SRC_MATCHING_HPP_

