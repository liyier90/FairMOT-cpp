cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(FairMOT)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE
    PATHS ./third_party/eigen
    NO_DEFAULT_PATH
)
find_package(OpenCV REQUIRED
    PATHS ./third_party/opencv4
    NO_DEFAULT_PATH
)
set(CMAKE_PREFIX_PATH
    "${PROJECT_SOURCE_DIR}/third_party/libtorch/share/cmake")
find_package(Torch REQUIRED)

find_package(Python3 COMPONENTS Development)
set(CMAKE_PREFIX_PATH
    "${PROJECT_SOURCE_DIR}/third_party/torchvision/share/cmake")
find_package(TorchVision REQUIRED)

set(FairMOT_CXX_WARNING_FLAGS "\
    -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
    -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op \
    -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept \
    -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow \
    -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default \
    -Wundef -Wno-unused"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FairMOT_CXX_WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

if (FairMOT_PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif ()

set(SOURCES
    src/main.cpp
    src/Decoder.cpp
    src/FairMot.cpp
    src/STrack.cpp
    src/Utils.cpp
)

add_executable(FairMOT ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(FairMOT
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(FairMOT
    stdc++fs
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Python3::Python
    ${TORCH_LIBRARIES}
    TorchVision::TorchVision
)
set_property(TARGET FairMOT PROPERTY CXX_STANDARD 14)

