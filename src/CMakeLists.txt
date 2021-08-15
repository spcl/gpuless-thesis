cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Wextra -g")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

# build fatbin test kernels with external Makefile
add_custom_target(
    fatbins
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
    )

add_executable(test test.cpp)
target_link_libraries(test CUDA::cuda_driver)
add_dependencies(test fatbins)

add_executable(benchmark-local benchmark-local.cpp)
target_link_libraries(benchmark-local CUDA::cudart CUDA::cuda_driver)

add_executable(benchmark-tcp benchmark-tcp.cpp)
target_link_libraries(benchmark-tcp CUDA::cudart CUDA::cuda_driver)

add_executable(server-tcp server-tcp.cpp)
target_link_libraries(server-tcp CUDA::cudart CUDA::cuda_driver)
