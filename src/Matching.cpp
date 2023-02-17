#include "Matching.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <numeric>
#include <utility>

#include "DataType.hpp"
#include "Utils.hpp"

namespace fairmot {
namespace matching {
void EmbeddingDistance(const std::vector<STrackPtr> &rTracks,
                       const std::vector<STrackPtr> &rDetections,
                       std::vector<float> &rCostMatrix, int &rNumRows,
                       int &rNumCols) {
  rNumRows = rTracks.size();
  rNumCols = rDetections.size();
  if (rNumRows * rNumCols == 0) {
    return;
  }

  for (int i = 0; i < rNumRows; ++i) {
    std::vector<float> cost_matrix_row(rNumCols);
    RowVecR<128> track_feature = Eigen::Map<const RowVecR<128>>(
        rTracks[i]->mSmoothFeat.data(), rTracks[i]->mSmoothFeat.size());

    for (int j = 0; j < rNumCols; ++j) {
      RowVecR<128> det_feature = Eigen::Map<const RowVecR<128>>(
          rDetections[j]->mCurrFeat.data(), rDetections[j]->mCurrFeat.size());
      rCostMatrix.push_back(1.0 - track_feature.dot(det_feature) /
                                      track_feature.norm() /
                                      det_feature.norm());
    }
  }
}

void FuseMotion(const KalmanFilter &rKalmanFilter,
                const std::vector<STrackPtr> &rTracks,
                const std::vector<STrackPtr> rDetections,
                std::vector<float> &rCostMatrix, const int numCols,
                const bool onlyPosition, const float coeff) {
  if (rCostMatrix.size() == 0) {
    return;
  }

  const auto gating_threshold = KalmanFilter::kChi2Inv95[onlyPosition ? 2 : 4];
  std::vector<RowVecR<4>, Eigen::aligned_allocator<RowVecR<4>>> measurements;
  measurements.reserve(rDetections.size());
  for (const auto &r_det : rDetections) {
    measurements.push_back(r_det->ToXyah());
  }
  for (auto i = 0u; i < rTracks.size(); ++i) {
    const auto gating_distance = rKalmanFilter.GatingDistance(
        rTracks[i]->mMean, rTracks[i]->mCovariance, measurements, onlyPosition);
    for (auto j = 0; j < numCols; ++j) {
      const auto distance = gating_distance[j];
      const auto idx = i * numCols + j;
      if (distance > gating_threshold) {
        rCostMatrix[idx] = FLT_MAX;
      } else {
        rCostMatrix[idx] = coeff * rCostMatrix[idx] + (1.0 - coeff) * distance;
      }
    }
  }
}

inline float GetArea(const BBox &rXyxy) {
  return (rXyxy[2] - rXyxy[0] + 1.0) * (rXyxy[3] - rXyxy[1] + 1.0);
}

std::vector<float> IouDistance(const std::vector<STrackPtr> &rTracks1,
                               const std::vector<STrackPtr> &rTracks2) {
  auto extract_xyxy = [](STrackPtr p_track) { return p_track->mXyxy; };
  std::vector<BBox> xyxys_1;
  std::vector<BBox> xyxys_2;

  std::transform(rTracks1.begin(), rTracks1.end(), std::back_inserter(xyxys_1),
                 extract_xyxy);
  std::transform(rTracks2.begin(), rTracks2.end(), std::back_inserter(xyxys_2),
                 extract_xyxy);

  auto ious = Ious(xyxys_1, xyxys_2);
  std::for_each(ious.begin(), ious.end(),
                [](float &r_val) { r_val = 1.0 - r_val; });

  return ious;
}

std::vector<float> IouDistance(const std::vector<STrackPtr> &rTracks1,
                               const std::vector<STrackPtr> &rTracks2,
                               int &rNumRows, int &rNumCols) {
  auto extract_xyxy = [](STrackPtr p_track) { return p_track->mXyxy; };
  rNumRows = rTracks1.size();
  rNumCols = rTracks2.size();

  std::vector<BBox> xyxys_1;
  std::vector<BBox> xyxys_2;

  std::transform(rTracks1.begin(), rTracks1.end(), std::back_inserter(xyxys_1),
                 extract_xyxy);
  std::transform(rTracks2.begin(), rTracks2.end(), std::back_inserter(xyxys_2),
                 extract_xyxy);

  auto ious = Ious(xyxys_1, xyxys_2);
  std::for_each(ious.begin(), ious.end(),
                [](float &r_val) { r_val = 1.0 - r_val; });

  return ious;
}

std::vector<float> Ious(const std::vector<BBox> &rXyxys1,
                        const std::vector<BBox> &rXyxys2) {
  const auto num_rows = rXyxys1.size();
  const auto num_cols = rXyxys2.size();
  std::vector<float> ious;
  if (num_rows * num_cols == 0) {
    return ious;
  }

  ious.assign(num_rows * num_cols, 0.0);

  auto j = 0;
  for (const auto &r_xyxy_2 : rXyxys2) {
    const auto box_a = GetArea(r_xyxy_2);
    auto i = 0;
    for (const auto &r_xyxy_1 : rXyxys1) {
      const auto intersect_w = std::min(r_xyxy_1[2], r_xyxy_2[2]) -
                               std::max(r_xyxy_1[0], r_xyxy_2[0]) + 1.0;
      if (intersect_w > 0.0) {
        const auto intersect_h = std::min(r_xyxy_1[3], r_xyxy_2[3]) -
                                 std::max(r_xyxy_1[1], r_xyxy_2[1]) + 1.0;
        if (intersect_h > 0.0) {
          const auto intersect_a = intersect_w * intersect_h;
          const auto union_a = GetArea(r_xyxy_1) + box_a - intersect_a;
          ious[i * num_cols + j] = intersect_a / union_a;
        }
      }
      ++i;
    }
    ++j;
  }

  return ious;
}

void LinearAssignment(const std::vector<float> &rCostMatrix, const int numRows,
                      const int numCols, const float threshold,
                      std::vector<std::pair<int, int>> &rMatches,
                      std::vector<int> &rUnmatched1,
                      std::vector<int> &rUnmatched2) {
  if (rCostMatrix.size() == 0) {
    rUnmatched1.resize(numRows);
    rUnmatched2.resize(numCols);
    std::iota(rUnmatched1.begin(), rUnmatched1.end(), 0);
    std::iota(rUnmatched2.begin(), rUnmatched2.end(), 0);
    return;
  }

  std::vector<int> rowsol;
  std::vector<int> colsol;
  const auto cost = util::Lapjv(rCostMatrix, numRows, numCols, rowsol, colsol,
                                /*extendCost=*/true, threshold);
  for (auto i = 0u; i < rowsol.size(); ++i) {
    int index = static_cast<int>(i);
    if (rowsol[i] >= 0) {
      rMatches.push_back(std::make_pair(index, rowsol[i]));
    } else {
      rUnmatched1.push_back(index);
    }
  }
  for (auto i = 0u; i < colsol.size(); ++i) {
    if (colsol[i] < 0) {
      rUnmatched2.push_back(static_cast<int>(i));
    }
  }
}
}  // namespace matching
}  // namespace fairmot
