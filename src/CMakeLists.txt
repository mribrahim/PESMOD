cmake_minimum_required(VERSION 3.20)
project(BGS)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.1 )

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED )
include_directories (${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp utils.cpp utils.h SimpleBackground.h simpleBackground.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})