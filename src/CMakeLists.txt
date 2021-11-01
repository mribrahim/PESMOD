cmake_minimum_required(VERSION 3.20)
project(BGS)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.1 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.1 )
set(CMAKE_PREFIX_PATH /home/ibrahim/Desktop/Programlar/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp utils.cpp utils.h SimpleBackground.h simpleBackground.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})