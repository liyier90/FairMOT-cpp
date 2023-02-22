# FairMOT-cpp

An inference-only C++ implementation of [FairMOT](https://github.com/ifzhang/FairMOT).

## Pre-requisites

- CMake 3.18.4
- Python headers 3.8.10
- CUDA 11.4
- cuDNN 8.1.1
- LibTorch 1.12.1
- Torchvision 0.13.1
- OpenCV 4.7.0
- Eigen 3.4.0

_Other than Eigen, older versions for the dependencies may also work._

## Quick start

1. Clone the repo and its submodules

   ```
   git clone https://github.com/liyier90/FairMOT-cpp.git && cd FairMOT-cpp
   git submodule update --init --recursive
   ```

2. Download [fairmot_dla34.pth](https://drive.google.com/file/d/1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi/view) to the `weights/` folder.

3. Convert the PyTorch model to a JIT model.

   ```
   cd python
   pip install -r requirements.txt
   python convert_to_jit.py
   ```

   _Older versions of the required Python packages may work as well._

4. Set up third-party C++ dependencies.

   ```
   cd ../third_party
   ```

   1. Download and install [Eigen](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz).

      ```
      wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      tar -xvf eigen-3.4.0.tar.gz && cd eigen-3.4.0
      mkdir build && cd build
      cmake -DCMAKE_INSTALL_PREFIX=</path/to/third_party/eigen> ..
      make install
      cd ../..
      ```

   2. Download and install [OpenCV](https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz)

      ```
      wget https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz
      tar -xvf 4.7.0.tar.gz && cd opencv-4.7.0
      mkdir build && cd build
      cmake -DCMAKE_INSTALL_PREFIX=</path/to/third_party/opencv4> ..
      cmake --build .
      make install
      cd ../..
      ```

      _This OpenCV may use Eigen instead of LAPACK as the linear alegbra package._

   3. Download [LibTorch](https://pytorch.org/cppdocs/installing.html).

      ```
      wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip
      unzip libtorch-cxx11-abi-shared-with-deps-1.12.1+cu113.zip
      ```

   4. Clone and install [Torchvision](https://github.com/pytorch/vision.git)

      ```
      git clone https://github.com/pytorch/vision.git && cd vision
      git checkout tags/v0.13.1
      mkdir build && cd build
      cmake -DCMAKE_INSTALL_PREFIX=</path/to/third_party/torchvision> \
            -DCMAKE_PREFIX_PATH=</path/to/third_party/libtorch/share/cmake> \
            -DWITH_CUDA=ON \
            -DUSE_PYTHON=OFF ..
      cmake --build .
      make install
      cd ../..
      ```

5. Compile the source code

   ```
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build .
   ```

6. Run FairMOT

   ```
   ./FairMOT </path/to/video/file>
   ```
