cmake_minimum_required(VERSION 3.17)
project(image-recognition)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)

add_executable(image-recognition image-recognition.cpp)
target_link_libraries(image-recognition "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
