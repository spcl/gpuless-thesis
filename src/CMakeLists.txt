cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Wall -Wextra")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

# build fatbin test kernels with external Makefile
add_custom_target(
    fatbins
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
    )

add_executable(test test.cc)
target_link_libraries(test CUDA::cuda_driver)
add_dependencies(test fatbins)
