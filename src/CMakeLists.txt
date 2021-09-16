cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wextra -Werror -ggdb")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

# build fatbin test kernels with external Makefile
add_custom_target(
    fatbins
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
    )

# build flatbuffer schemas with external Makefile
add_custom_target(
    flatbuffer
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/schemas
    )

add_executable(test test.cpp)
target_link_libraries(test CUDA::cuda_driver)
add_dependencies(test fatbins)

add_executable(manager manager/manager.cpp manager/manager_device.cpp utils.cpp)
add_dependencies(manager flatbuffer)
target_link_libraries(manager CUDA::cuda_driver boost_serialization pthread)

add_executable(test_tcp test/test_tcp.cpp executors/executor_tcp.cpp utils.cpp)
add_dependencies(test_tcp flatbuffer)
target_link_libraries(test_tcp boost_serialization CUDA::cudart CUDA::cuda_driver)

add_library(cuadapter SHARED
            adapter/libcuadapter.cpp
            adapter/cubin_analysis.cpp
            executors/executor_tcp.cpp
            utils.cpp)
add_dependencies(cuadapter flatbuffer)
target_link_libraries(cuadapter CUDA::cuda_driver)
