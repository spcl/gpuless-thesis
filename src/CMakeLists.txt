cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)

find_package(CUDAToolkit)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -Wall -Wextra -Wno-unused-parameter -ggdb")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# build fatbin test kernels with external Makefile
# add_custom_target(
#     fatbins
#     COMMAND make
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/kernels
#     )

# build flatbuffer schemas with external Makefile
add_custom_target(
    flatbuffer
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/schemas
    )

# add_executable(test test.cpp)
# target_link_libraries(test CUDA::cuda_driver)
# add_dependencies(test fatbins)

# add_executable(manager manager/manager.cpp manager/manager_device.cpp utils.cpp)
# add_dependencies(manager flatbuffer)
# target_link_libraries(manager iberty CUDA::cuda_driver boost_serialization pthread)

add_executable(manager_trace
    trace/manager/manager.cpp
    trace/manager/manager_device.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/dlsym_util.cpp
    utils.cpp)
add_dependencies(manager_trace flatbuffer)
target_link_libraries(manager_trace
    spdlog::spdlog_header_only
    iberty
    CUDA::cudart
    CUDA::cuda_driver
    pthread)

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/benchmarks/resnet50
    ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)

add_library(analysis SHARED
    analysis/libcudaanalysis.cpp
    trace/cubin_analysis.cpp
    utils.cpp trace/trace_executor.hpp)
target_link_libraries(analysis spdlog::spdlog_header_only CUDA::cudart CUDA::cuda_driver)

add_library(gpuless SHARED
    trace/libgpuless.cpp
    trace/hijack_cudnn.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/dlsym_util.cpp
    utils.cpp
    trace/cubin_analysis.cpp trace/cudnn_api_calls.cpp)
target_link_libraries(gpuless
    spdlog::spdlog_header_only
    iberty
    CUDA::cuda_driver
    CUDA::cudart)

add_custom_command(
    TARGET gpuless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR})
