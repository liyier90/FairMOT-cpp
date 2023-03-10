#ifndef SRC_MATCHING_HPP_
#define SRC_MATCHING_HPP_

#include <utility>

#include "DataType.hpp"
#include "KalmanFilter.hpp"
#include "STrack.hpp"

namespace fairmot {
namespace matching {
void EmbeddingDistance(const std::vector<STrackPtr> &rTracks,
                       const std::vector<STrackPtr> &rDetections,
                       std::vector<float> &rCostMatrix, int &rNumRows,
                       int &rNumCols);

void FuseMotion(const KalmanFilter &rKalmanFilter,
                const std::vector<STrackPtr> &rTracks,
                const std::vector<STrackPtr> rDetections,
                std::vector<float> &rCostMatrix, const int numCols,
                const bool onlyPosition = false, const float coeff = 0.98);

float GetArea(const BBox &rXyxy);

std::vector<float> IouDistance(const std::vector<STrackPtr> &rTracks1,
                               const std::vector<STrackPtr> &rTracks2);

std::vector<float> IouDistance(const std::vector<STrackPtr> &rTracks1,
                               const std::vector<STrackPtr> &rTracks2,
                               int &rNumRows, int &rNumCols);

std::vector<float> Ious(const std::vector<BBox> &rXyxys1,
                        const std::vector<BBox> &rXyxys2);

void LinearAssignment(const std::vector<float> &rCostMatrix, const int numRows,
                      const int numCols, const float threshold,
                      std::vector<std::pair<int, int>> &rMatches,
                      std::vector<int> &rUnmatched1,
                      std::vector<int> &rUnmatched2);
}  // namespace matching
}  // namespace fairmot

#endif  // SRC_MATCHING_HPP_

