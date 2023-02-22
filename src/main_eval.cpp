#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "FairMot.hpp"
#include "Filesystem.hpp"
#include "ini.h"

void EvaluateSequence(const std::string &rModelPath, const fs::path &rInputDir,
                      const fs::path &rOutputDir);

std::vector<fs::path> Glob(const fs::path &rDirectory,
                           const std::string &rExtension);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Missing benchmark data!" << std::endl;
    std::cerr << "Usage: ./FairMOT <path/to/MOT/dataset>" << std::endl;
    return EXIT_FAILURE;
  }

  auto input_dir = fs::path(argv[1]);
  if (!fs::exists(input_dir)) {
    std::cerr << "Could not open directory: " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }
  input_dir = fs::canonical(input_dir);

  std::string model_path = "../weights/fairmot_dla34_jit.pth";
  auto output_dir = fs::current_path().parent_path() / "results";
  fs::create_directory(output_dir);
  for (const auto &r_dir_entry : fs::directory_iterator(input_dir)) {
    EvaluateSequence(model_path, r_dir_entry.path(), output_dir);
  }

  return 0;
}

void EvaluateSequence(const std::string &rModelPath, const fs::path &rInputDir,
                      const fs::path &rOutputDir) {
  std::cout << rInputDir << std::endl;
  mINI::INIFile seq_file((rInputDir / "seqinfo.ini").string());
  mINI::INIStructure seq_config;
  seq_file.read(seq_config);
  const auto frame_rate =
      static_cast<double>(std::stoi(seq_config["Sequence"]["frameRate"]));

  fairmot::FairMot tracker(rModelPath, frame_rate,
                           /*maxPerImage=*/500,
                           /*trackBuffer=*/30);
  const auto output_path =
      (rOutputDir / rInputDir.filename()).replace_extension(".txt");
  std::ofstream outfile;
  outfile.open(output_path);

  for (const auto &r_path : Glob(rInputDir / "img1", ".jpg")) {
    cv::Mat image = cv::imread(r_path.string());
    const auto results = tracker.Track(image);
    for (const auto &r_track : results) {
      const auto tid = r_track.track_id;
      if (tid < 0) {
        continue;
      }
      outfile << std::stoi(r_path.stem()) << "," << std::to_string(tid) << ",";
      for (const auto &r_val : r_track.tlwh) {
        outfile << std::to_string(r_val) << ",";
      }
      outfile << "1,-1,-1,-1" << std::endl;
    }
  }
  outfile.close();
}

std::vector<fs::path> Glob(const fs::path &rDirectory,
                           const std::string &rExtension) {
  std::vector<fs::path> file_paths;
  for (const auto &r_file : fs::directory_iterator(rDirectory)) {
    const auto img_path = r_file.path();
    if (img_path.extension() != rExtension) {
      continue;
    }
    file_paths.push_back(img_path);
  }
  std::sort(file_paths.begin(), file_paths.end());

  return file_paths;
}

