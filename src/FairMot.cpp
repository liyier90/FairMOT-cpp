#include "FairMot.hpp"

#include <torch/torch.h>
#include <torchvision/vision.h>

#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "STrack.hpp"
#include "Utils.hpp"

namespace fairmot {
FairMot::FairMot(const std::string &rModelPath, int maxPerImage)
    : mModel{torch::jit::load(rModelPath)},
      mDecoder(maxPerImage, 4),
      mInputHeight{480},
      mInputWidth{864},
      mScoreThreshold{0.4},
      mFrameId{0} {
  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "Using CUDA." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Using CPU." << std::endl;
    device_type = torch::kCPU;
  }

  mpDevice = new torch::Device(device_type);
  mModel.to(*mpDevice);
}

FairMot::~FairMot() { delete mpDevice; }

void FairMot::Track(const cv::Mat &rImage) {
  auto padded_image = this->Preprocess(rImage);
  torch::Tensor detections;
  torch::Tensor embeddings;
  std::tie(detections, embeddings) = this->Predict(padded_image, rImage);
  this->Update(detections.contiguous(), embeddings.contiguous());
}

std::pair<torch::Tensor, torch::Tensor> FairMot::Predict(
    const cv::Mat &rPaddedImage, const cv::Mat &rImage) {
  namespace F = torch::nn::functional;
  torch::NoGradGuard no_grad;

  auto image_tensor = torch::from_blob(
      rPaddedImage.data, {mInputHeight, mInputWidth, 3}, torch::kFloat32);
  image_tensor = torch::unsqueeze(image_tensor, 0).permute({0, 3, 1, 2});

  std::vector<torch::jit::IValue> inputs = {image_tensor.to(*mpDevice)};
  // TODO: Improve the method of extracting model output
  // Contains hm, wh, id, and reg
  const auto output = mModel.forward(inputs).toTupleRef().elements();
  const auto heatmap = torch::sigmoid(output[0].toTensor());
  auto id_feature =
      F::normalize(output[2].toTensor(), F::NormalizeFuncOptions().dim(1));

  torch::Tensor detections;
  torch::Tensor indices;
  std::tie(detections, indices) =
      mDecoder(heatmap, output[1].toTensor(), output[3].toTensor(),
               rImage.size(), image_tensor.sizes());
  id_feature = util::TransposeAndGatherFeat(id_feature, indices);
  const auto mask =
      torch::nonzero(detections.select(1, 4) > mScoreThreshold).squeeze();
  // TODO: Check if filtering by score here is fine
  detections = detections.index_select(0, mask);
  id_feature = id_feature.squeeze(0).to(torch::kCPU).index_select(0, mask);

  return std::make_pair(detections, id_feature);
}

void FairMot::Update(const torch::Tensor &rDetections,
                     const torch::Tensor &rEmbeddings) {
  ++mFrameId;

  std::vector<STrack> detections;

  if (rDetections.size(0) > 0 && rEmbeddings.size(0) > 0) {
    const auto embedding_size = rEmbeddings.size(1);
    for (int i = 0; i < rDetections.size(0); ++i) {
      std::vector<float> xyxy(rDetections[i].data_ptr<float>(),
                              rDetections[i].data_ptr<float>() + 4);
      const auto score = rDetections[i][4].item<float>();
      std::vector<float> embedding(
          rEmbeddings[i].data_ptr<float>(),
          rEmbeddings[i].data_ptr<float>() + embedding_size);
      STrack strack(xyxy, score, embedding);
    }
  }
}

cv::Mat FairMot::Preprocess(cv::Mat image) {
  auto padded_image = util::Letterbox(image, mInputHeight, mInputWidth);

  cv::cvtColor(padded_image, padded_image, cv::COLOR_BGR2RGB);
  padded_image.convertTo(padded_image, CV_32FC3);
  padded_image /= 255.0;

  return padded_image;
}
}  // namespace fairmot
