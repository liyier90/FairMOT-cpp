#include "FairMot.hpp"

#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "Utils.hpp"

namespace fairmot {
FairMot::FairMot(
    const std::string &rModelPath,
    int maxPerImage)
  : mModel {torch::jit::load(rModelPath)},
    mDecoder(maxPerImage, 4),
    mInputHeight {480},
    mInputWidth {864}
{
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

FairMot::~FairMot() {
  delete mpDevice;
}

void FairMot::Track(const cv::Mat &rImage) {
  auto padded_image = this->Preprocess(rImage);
  this->Update(padded_image, rImage);
}

void FairMot::Update(
    const cv::Mat &rPaddedImage,
    const cv::Mat &rImage) {
  namespace F = torch::nn::functional;

  auto image_tensor = torch::from_blob(rPaddedImage.data,
      {mInputHeight, mInputWidth, 3}, torch::kFloat32);
  image_tensor = torch::unsqueeze(image_tensor, 0).permute({0, 3, 1, 2});

  std::vector<torch::jit::IValue> inputs = {image_tensor.to(*mpDevice)};
  // TODO: Improve the method of extracting model output
  // Contains hm, wh, id, and reg
  const auto output = mModel.forward(inputs).toTupleRef().elements();
  const auto heatmap = torch::sigmoid(output[0].toTensor());

  const auto id_feature = F::normalize(output[2].toTensor(),
      F::NormalizeFuncOptions().dim(1));
  const auto decoded_output = mDecoder(heatmap, output[1].toTensor(),
      output[3].toTensor(), rImage.size(), image_tensor.sizes());
}

cv::Mat FairMot::Preprocess(cv::Mat image) {
  auto padded_image = util::Letterbox(image, mInputHeight, mInputWidth);

  cv::cvtColor(padded_image, padded_image, cv::COLOR_BGR2RGB);
  padded_image.convertTo(padded_image, CV_32FC3);
  padded_image /= 255.0;
  
  return padded_image;
}
}  // namespace fairmot
