cmake_minimum_required(VERSION 3.1)
project(ChangeEmotion)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -std=c++14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

SET(OpenCV_DIR libs/opencv)
SET(CMAKE_PREFIX_PATH libs/libtorch/share/cmake/Torch)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
add_executable(ChangeEmotion ChangeEmotion.cpp utils.cpp FaceHandler.cpp)
target_link_libraries(ChangeEmotion ${OpenCV_LIBS} "${TORCH_LIBRARIES}")