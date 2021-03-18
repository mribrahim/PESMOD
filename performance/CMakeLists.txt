cmake_minimum_required(VERSION 3.17)
project(performance)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )

include_directories (${OpenCV_INCLUDE_DIRS})

file (GLOB rapid_xml_src rapidxml-1.13/*.hpp)


add_executable(${PROJECT_NAME} main.cpp ${rapid_xml_src} utils.cpp utils.h)

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})