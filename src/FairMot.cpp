#include "FairMot.hpp"

#include <torch/torch.h>
#include <torchvision/vision.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "DataType.hpp"
#include "Matching.hpp"
#include "STrack.hpp"
#include "Utils.hpp"

namespace fairmot {
FairMot::FairMot(const std::string &rModelPath, const double frameRate,
                 const int maxPerImage, const int trackBuffer)
    : mModel{torch::jit::load(rModelPath)},
      mDecoder(maxPerImage, 4),
      mInputHeight{480},
      mInputWidth{864},
      mScoreThreshold{0.4},
      mMaxTimeLost{static_cast<int>(frameRate / 30.0 * trackBuffer)},
      mFrameId{0},
      mLostStracks(),
      mTrackedStracks(),
      mRemoveStracks() {
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

void FairMot::Track(const cv::Mat &rImage) {
  auto padded_image = this->Preprocess(rImage);
  torch::Tensor detections;
  torch::Tensor embeddings;
  std::tie(detections, embeddings) = this->Predict(padded_image, rImage);
  const auto output_stracks =
      this->Update(detections.contiguous(), embeddings.contiguous());
}

std::vector<STrack> FairMot::Update(const torch::Tensor &rDetections,
                                    const torch::Tensor &rEmbeddings) {
  ++mFrameId;
  if (mFrameId == 3) {
    assert(false);
  }
  std::cout << "=== " << mFrameId << std::endl;
  std::vector<STrack> activated_stracks;
  std::vector<STrack> refind_stracks;
  std::vector<STrack> lost_stracks;
  std::vector<STrack> removed_stracks;

  const auto num_detections = rDetections.size(0);
  std::vector<STrack> detections;

  // Step 1: Network forward, get detections & embeddings
  if (num_detections > 0 && rEmbeddings.size(0) > 0) {
    detections.reserve(num_detections);
    const auto embedding_size = rEmbeddings.size(1);
    // Detections is a list of (x1, y1, x2, y2, score)
    for (int i = 0; i < num_detections; ++i) {
      BBox xyxy;
      memcpy(xyxy.data(), rDetections[i].data_ptr<float>(), sizeof(float) * 4);
      const auto score = rDetections[i][4].item<float>();
      Embedding embedding;
      memcpy(embedding.data(), rEmbeddings[i].data_ptr<float>(),
             sizeof(float) * 128);

      STrack strack(STrack::rXyxyToTlwh(xyxy), score, embedding);
      detections.push_back(strack);
    }
  }

  std::vector<STrack *> tracked_stracks;
  std::vector<STrack *> unconfirmed;

  std::vector<STrack *> strack_pool;
  // Add newly detected tracklets to tracked_stracks
  for (auto i = 0u; i < mTrackedStracks.size(); ++i) {
    if (mTrackedStracks[i].mIsActivated) {
      // Active tracks are added to the local list 'tracked_stracks'
      tracked_stracks.push_back(&mTrackedStracks[i]);
    } else {
      // previous tracks which are not active in the current frame are added in
      // unconfirmed list
      unconfirmed.push_back(&mTrackedStracks[i]);
    }
  }

  // Step 2: First association, with embedding
  // Combining tracked_stracks and mLostStracks
  strack_pool = strack_util::CombineStracks(tracked_stracks, mLostStracks);
  std::cout << "strack_pool " << strack_pool << std::endl;
  // Predict the current location with KF
  STrack::MultiPredict(strack_pool);
  std::cout << "MultiPredict" << std::endl;

  // The dists is a matrix of distances of the detection with the tracks
  // in strack_pool
  std::vector<std::vector<float>> dists;
  auto num_rows = 0;
  auto num_cols = 0;
  matching::EmbeddingDistance(strack_pool, detections, dists, num_rows,
                              num_cols);
  matching::FuseMotion(STrack::kSharedKalman, strack_pool, detections, dists);

  std::vector<std::vector<int>> matches;
  std::vector<int> u_track;
  std::vector<int> u_detection;
  matching::LinearAssignment(dists, num_rows, num_cols, /*threshold=*/0.4,
                             matches, u_track, u_detection);
  std::cout << "u_detection " << u_detection << std::endl;

  // After all these confirmation steps, if a new detection is found, it is
  // initialized for a new track
  // Step 4: Init new stracks
  for (const auto &r_u_det : u_detection) {
    auto *p_track = &detections[r_u_det];
    if (p_track->mScore < mScoreThreshold) {
      continue;
    }
    p_track->Activate(mFrameId);
    activated_stracks.push_back(*p_track);
  }
  std::cout << "activated_stracks ";
  std::cout << activated_stracks << std::endl;
  // Step 5: Update state
  // If the tracks are lost for more frames than the threshold number, the
  // tracks are removed.
  for (auto &r_track : mLostStracks) {
    if (mFrameId - r_track.rEndFrame() > mMaxTimeLost) {
      r_track.MarkRemoved();
      removed_stracks.push_back(r_track);
    }
  }
  // Update the mTrackedStracks and mLostStracks using the updates in this step.
  std::vector<STrack> tracked_stracks_swap;
  tracked_stracks_swap.reserve(mTrackedStracks.size());
  for (auto &r_track : mTrackedStracks) {
    if (r_track.mState == TrackState::kTRACKED) {
      tracked_stracks_swap.push_back(r_track);
    }
  }
  mTrackedStracks = std::move(tracked_stracks_swap);
  mTrackedStracks =
      strack_util::CombineStracks(mTrackedStracks, activated_stracks);
  mTrackedStracks =
      strack_util::CombineStracks(mTrackedStracks, refind_stracks);

  mLostStracks = strack_util::SubstractStracks(mLostStracks, mTrackedStracks);
  for (const auto &r_track : lost_stracks) {
    mLostStracks.push_back(r_track);
  }
  mLostStracks = strack_util::SubstractStracks(mLostStracks, mRemoveStracks);
  for (const auto &r_track : removed_stracks) {
    mRemoveStracks.push_back(r_track);
  }

  std::vector<STrack> res_1;
  std::vector<STrack> res_2;
  strack_util::RemoveDuplicateStracks(mTrackedStracks, mLostStracks, res_1,
                                      res_2);
  mTrackedStracks = std::move(res_1);
  mLostStracks = std::move(res_2);
  std::cout << "mTrackedStracks ";
  std::cout << mTrackedStracks << std::endl;

  std::vector<STrack> output_stracks;
  for (const auto &r_track : mTrackedStracks) {
    if (r_track.mIsActivated) {
      output_stracks.push_back(r_track);
    }
  }

  return output_stracks;
}

cv::Mat FairMot::Preprocess(cv::Mat image) {
  auto padded_image = util::Letterbox(image, mInputHeight, mInputWidth);

  cv::cvtColor(padded_image, padded_image, cv::COLOR_BGR2RGB);
  padded_image.convertTo(padded_image, CV_32FC3);
  padded_image /= 255.0;

  return padded_image;
}
}  // namespace fairmot
