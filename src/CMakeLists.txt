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

set(GSLICR_LIB
        gSLICr_Lib/engines/gSLICr_core_engine.h
        gSLICr_Lib/engines/gSLICr_seg_engine.h
        gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
        gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
        gSLICr_Lib/engines/gSLICr_core_engine.cpp
        gSLICr_Lib/engines/gSLICr_seg_engine.cpp
        gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
        gSLICr_Lib/objects/gSLICr_settings.h
        gSLICr_Lib/objects/gSLICr_spixel_info.h
        gSLICr_Lib/gSLICr_defines.h
        gSLICr_Lib/gSLICr.h
        superpixel.cpp superpixel.h)

SOURCE_GROUP(engines FILES ${GSLICR_LIB})
add_subdirectory(ORUtils)

cuda_add_library(gSLICr_lib
        ${GSLICR_LIB}
        OPTIONS -gencode arch=compute_30,code=compute_30)

add_executable(${PROJECT_NAME} main.cpp utils.cpp utils.h SimpleBackground.h simpleBackground.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} gSLICr_lib)