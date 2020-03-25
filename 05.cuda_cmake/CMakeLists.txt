cmake_minimum_required(VERSION 3.5)
project(empty)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

message(${CUDA_LIBRARIES})
message(${OpenCV_LIBRARIES})
