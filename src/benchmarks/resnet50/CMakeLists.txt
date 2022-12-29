cmake_minimum_required(VERSION 3.17)
project(image-recognition)

cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
if (${_hostname} MATCHES ".*\.cscs\.ch")
    set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch/share/cmake;$ENV{HOME}/opencv-4.x/build")

    set(CUDNN_ROOT $ENV{HOME}/cudnn)
    set(CUDNN_LIBRARY $ENV{HOME}/cudnn/lib64)
    set(CUDNN_INCLUDE_DIR $ENV{HOME}/cudnn/include)

    include_directories($ENV{HOME}/conda/include)
    #link_directories(/opt/intel/mkl/lib/intel64)
    link_directories($ENV{HOME}/conda/lib)
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
target_link_libraries(image-recognition "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" iomp5)

add_executable(image-recognition-batched image-recognition-batched.cpp)
target_link_libraries(image-recognition-batched "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" iomp5)

add_executable(image-recognition-batched-total image-recognition-batched-total.cpp)
target_link_libraries(image-recognition-batched-total "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" iomp5)
