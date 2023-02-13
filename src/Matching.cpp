#include "Matching.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <numeric>

#include "DataType.hpp"

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
    RowVectorR<128> track_feature = Eigen::Map<const RowVectorR<128>>(
        rTracks[i]->mSmoothFeat.data(), rTracks[i]->mSmoothFeat.size());

    for (int j = 0; j < rNumCols; ++j) {
      RowVectorR<128> det_feature = Eigen::Map<const RowVectorR<128>>(
          rDetections[j].mCurrFeat.data(), rDetections[j].mCurrFeat.size());
      cost_matrix_row[j] = 1.0 - track_feature.dot(det_feature) /
                                     track_feature.norm() / det_feature.norm();
    }
    rCostMatrix.push_back(cost_matrix_row);
  }
}

void FuseMotion(const KalmanFilter &rKalmanFilter,
                const std::vector<STrack *> &rTracks,
                const std::vector<STrack> rDetections,
                std::vector<std::vector<float>> &rCostMatrix,
                const bool onlyPosition, const float coeff) {
  if (rCostMatrix.size() == 0) {
    return;
  }

  const auto gating_threshold = KalmanFilter::kChi2Inv95[onlyPosition ? 2 : 4];
  std::vector<RowVectorR<4>> measurements;
  measurements.reserve(rDetections.size());
  for (const auto &r_det : rDetections) {
    measurements.push_back(r_det.ToXyah());
  }
  for (auto i = 0u; i < rTracks.size(); ++i) {
    const auto gating_distance = rKalmanFilter.GatingDistance(
        rTracks[i]->mMean, rTracks[i]->mCovariance, measurements, onlyPosition);
    for (auto j = 0u; j < rCostMatrix[i].size(); ++j) {
      if (gating_distance[j] > gating_threshold) {
        rCostMatrix[i][j] = FLT_MAX;
      } else {
        rCostMatrix[i][j] =
            coeff * rCostMatrix[i][j] + (1.0 - coeff) * gating_distance[j];
      }
    }
  }
}

inline float GetArea(const BBox &rXyxy) {
  return (rXyxy[2] - rXyxy[0] + 1.0) * (rXyxy[3] - rXyxy[1] + 1.0);
}

std::vector<std::vector<float>> IouDistance(
    const std::vector<STrack> rTracks1, const std::vector<STrack> rTracks2) {
  std::vector<BBox> xyxys_1;
  std::vector<BBox> xyxys_2;
  xyxys_1.reserve(rTracks1.size());
  xyxys_2.reserve(rTracks2.size());

  for (const auto &r_track : rTracks1) {
    xyxys_1.push_back(r_track.mXyxy);
  }
  for (const auto &r_track : rTracks2) {
    xyxys_2.push_back(r_track.mXyxy);
  }

  auto ious = Ious(xyxys_1, xyxys_2);
  std::vector<std::vector<float>> cost_matrix(ious.size());
  for (auto i = 0u; i < ious.size(); ++i) {
    cost_matrix[i].reserve(ious[i].size());
    for (auto j = 0u; j < ious[i].size(); ++j) {
      cost_matrix[i][j] = 1.0 - ious[i][j];
    }
  }

  return cost_matrix;
}

std::vector<std::vector<float>> Ious(const std::vector<BBox> &rXyxys1,
                                     const std::vector<BBox> &rXyxys2) {
  const auto num_boxes = rXyxys1.size();
  const auto num_queries = rXyxys2.size();
  std::vector<std::vector<float>> ious;
  if (num_boxes * num_queries == 0) {
    return ious;
  }

  ious.assign(num_boxes, std::vector<float>(num_queries, 0.0));

  auto k = 0;
  for (const auto &r_xyxy_2 : rXyxys2) {
    const auto box_a = GetArea(r_xyxy_2);
    auto n = 0;
    for (const auto &r_xyxy_1 : rXyxys1) {
      const auto intersect_w = std::min(r_xyxy_1[2], r_xyxy_2[2]) -
                               std::max(r_xyxy_1[0], r_xyxy_2[0]) + 1.0;
      if (intersect_w > 0.0) {
        const auto intersect_h = std::min(r_xyxy_1[3], r_xyxy_2[3]) -
                                 std::max(r_xyxy_1[1], r_xyxy_2[1]) + 1.0;
        if (intersect_h > 0.0) {
          const auto intersect_a = intersect_w * intersect_h;
          const auto union_a = GetArea(r_xyxy_1) + box_a - intersect_a;
          ious[n][k] = intersect_a / union_a;
        }
      }
      ++n;
    }
    ++k;
  }

  return ious;
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
