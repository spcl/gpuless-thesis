cmake_minimum_required(VERSION 3.17)
project(image-recognition)

cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
if (${_hostname} MATCHES ".*\.cscs\.ch")
    set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch/share/cmake;$ENV{HOME}/opencv-4.5.4/build")

    set(CUDNN_ROOT $ENV{HOME}/cudnn/cuda)
    set(CUDNN_LIBRARY $ENV{HOME}/cudnn/cuda/lib64)
    set(CUDNN_INCLUDE_DIR $ENV{HOME}/cudnn/cuda/include)

    include_directories($ENV{HOME}/miniconda3/include)
    link_directories($ENV{HOME}/miniconda3/lib)
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

add_executable(image-recognition image-recognition.cpp)
target_link_libraries(image-recognition "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

add_executable(image-recognition-batched image-recognition-batched.cpp)
target_link_libraries(image-recognition-batched "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

add_executable(image-recognition-batched-total image-recognition-batched-total.cpp)
target_link_libraries(image-recognition-batched-total "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
