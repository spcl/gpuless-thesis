cmake_minimum_required(VERSION 3.17)
project(image-recogntion)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(image-recognition image-recognition.cpp)
target_link_libraries(image-recognition "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
