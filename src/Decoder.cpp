#include "Decoder.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <tuple>
#include <utility>

#include "Utils.hpp"

namespace fairmot {
Decoder::Decoder(int maxPerImage, int downRatio)
    : mMaxPerImage{maxPerImage}, mDownRatio{downRatio} {}

std::pair<torch::Tensor, torch::Tensor> Decoder::operator()(
    const torch::Tensor &rHeatmap, const torch::Tensor &rSize,
    const torch::Tensor &rOffset, const cv::Size &rOrigShape,
    const torch::IntArrayRef &rInputShape) {
  const auto batch = rHeatmap.sizes()[0];
  const auto heatmap = this->Nms(rHeatmap);

  torch::Tensor scores;
  torch::Tensor indices;
  torch::Tensor classes;
  torch::Tensor y_coords;
  torch::Tensor x_coords;
  std::tie(scores, indices, classes, y_coords, x_coords) = this->TopK(heatmap);

  auto offset = util::TransposeAndGatherFeat(rOffset, indices);
  offset = offset.view({batch, mMaxPerImage, 2});
  x_coords = x_coords.view({batch, mMaxPerImage, 1}) + offset.slice(2, 0, 1);
  y_coords = y_coords.view({batch, mMaxPerImage, 1}) + offset.slice(2, 1, 2);

  auto size = util::TransposeAndGatherFeat(rSize, indices);
  size = size.view({batch, mMaxPerImage, 4});
  classes = classes.view({batch, mMaxPerImage, 1});
  scores = scores.view({batch, mMaxPerImage, 1});
  const auto bboxes = torch::cat(
      {x_coords - size.slice(2, 0, 1), y_coords - size.slice(2, 1, 2),
       x_coords + size.slice(2, 2, 3), y_coords + size.slice(2, 3, 4)},
      /*dim=*/2);
  auto detections = torch::cat({bboxes, scores, classes}, /*dim=*/2);
  detections = this->PostProcess(detections, rOrigShape, rInputShape);

  return std::make_pair(detections, indices);
}

void Decoder::CtdetPostProcess(torch::Tensor &rDetections,
                               const std::array<double, 2> &rCenter,
                               const double scale,
                               const std::array<int, 2> &rOutputSize) {
  auto coords = rDetections.slice(1, 0, 4);
  util::TransformCoords(coords, rCenter, scale, rOutputSize);
  // TODO: Check if masking by class is really necessary
  const auto mask =
      torch::nonzero(rDetections.select(1, 5).to(torch::kInt64) == 0).squeeze();
  rDetections = rDetections.slice(1, 0, 5).index_select(0, mask);
}

torch::Tensor Decoder::Nms(const torch::Tensor &rHeatmap, int kernel) {
  namespace F = torch::nn::functional;
  int pad = (kernel - 1) / 2;
  const auto hmax = F::max_pool2d(
      rHeatmap, F::MaxPool2dFuncOptions(3).stride(1).padding(pad));
  const auto keep = (hmax == rHeatmap).to(torch::kFloat);

  return rHeatmap * keep;
}

torch::Tensor Decoder::PostProcess(const torch::Tensor &rDetections,
                                   const cv::Size &rOrigShape,
                                   const torch::IntArrayRef &rInputShape) {
  const auto orig_h = static_cast<double>(rOrigShape.height);
  const auto orig_w = static_cast<double>(rOrigShape.width);
  const auto input_h = static_cast<double>(rInputShape[2]);
  const auto input_w = static_cast<double>(rInputShape[3]);
  auto detections_cpu =
      rDetections.to(torch::kCPU).view({1, -1, rDetections.size(2)}).squeeze(0);
  this->CtdetPostProcess(detections_cpu, {orig_w / 2.0, orig_h / 2.0},
                         std::max(input_w / input_h * orig_h, orig_w),
                         {static_cast<int>(input_w) / mDownRatio,
                          static_cast<int>(input_h) / mDownRatio});

  return detections_cpu;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
Decoder::TopK(const torch::Tensor &rScores) {
  const auto sizes = rScores.sizes();
  const auto batch = sizes[0];
  const auto cat = sizes[1];
  const auto height = sizes[2];
  const auto width = sizes[3];

  torch::Tensor scores;
  torch::Tensor indices;
  std::tie(scores, indices) =
      torch::topk(rScores.view({batch, cat, -1}), mMaxPerImage);
  indices = (indices % (height * width)).view({batch, -1, 1});
  auto y_coords = torch::div(indices, width, /*rounding_mode=*/"floor");
  auto x_coords = indices % width;

  torch::Tensor score;
  torch::Tensor index;
  std::tie(score, index) = torch::topk(scores.view({batch, -1}), mMaxPerImage);
  auto classes = torch::div(index, mMaxPerImage, /*rounding_mode=*/"trunc");

  indices = util::GatherFeat(indices, index).view({batch, mMaxPerImage});
  y_coords = util::GatherFeat(y_coords, index).view({batch, mMaxPerImage});
  x_coords = util::GatherFeat(x_coords, index).view({batch, mMaxPerImage});

  return std::make_tuple(score, indices, classes, y_coords, x_coords);
}
}  // namespace fairmot
