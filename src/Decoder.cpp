#include "Decoder.hpp"

#include <tuple>
#include <utility>

#include <torch/torch.h>

#include "Utils.hpp"

namespace fairmot {
Decoder::Decoder(
    int maxPerImage,
    int downRatio)
  : mMaxPerImage {maxPerImage},
    mDownRatio {downRatio}
{}


std::pair<torch::Tensor, torch::Tensor> Decoder::operator()(
    const torch::Tensor &rHeatmap,
    const torch::Tensor &rSize,
    const torch::Tensor &rOffset,
    const cv::Size& rOrigShape,
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
  // {
  //   std::ofstream outfile;
  //   outfile.open("/home/yier/code/PeekingDuck/results/cpp0.txt");
  //   outfile << tensors[1];
  //   outfile.close();
  // }
  // for (int i = 0; i < 3; ++i) {
  //   std::ofstream outfile;
  //   outfile.open("/home/yier/code/PeekingDuck/results/cpp" + std::to_string(i) + ".txt");
  //   outfile << image_tensor[0][i];
  //   outfile.close();
  // }
  assert(false);
  return std::make_pair(rHeatmap, rSize);
}

torch::Tensor Decoder::Nms(
    const torch::Tensor &rHeatmap,
    int kernel) {
  namespace F = torch::nn::functional;
  int pad = (kernel - 1) / 2;
  const auto hmax = F::max_pool2d(rHeatmap,
      F::MaxPool2dFuncOptions(3).stride(1).padding(pad));
  const auto keep = (hmax == rHeatmap).to(torch::kFloat);
  
  return rHeatmap * keep;
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
  std::tie(scores, indices) = torch::topk(rScores.view({batch, cat, -1}),
      mMaxPerImage);
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
