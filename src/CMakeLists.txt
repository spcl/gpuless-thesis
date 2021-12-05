cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit 11.4)

# set desired log level
# add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_OFF)
add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/lib)

# special options for compiling on the CSCS cluster
cmake_host_system_information(RESULT _hostname QUERY HOSTNAME)
if (${_hostname} MATCHES ".*\.cscs\.ch")
    include_directories($ENV{HOME}/cudnn/cuda/include)
    link_directories($ENV{HOME}/cudnn/cuda/lib64)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -ggdb")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# build flatbuffer schemas with external Makefile
add_custom_target(
    flatbuffer
    COMMAND make
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/schemas)

add_executable(manager_trace
    trace/manager/manager.cpp
    trace/manager/manager_device.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cudnn_api_calls.cpp
    trace/cublas_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/dlsym_util.cpp
    utils.cpp)
add_dependencies(manager_trace flatbuffer)
target_link_libraries(manager_trace
    iberty
    stdc++fs
    cudnn
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    pthread)

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/benchmarks/resnet50
    ${CMAKE_SOURCE_DIR}/benchmarks/resnet50)

add_library(analysis SHARED
    analysis/libcudaanalysis.cpp
    trace/cubin_analysis.cpp
    utils.cpp trace/trace_executor.hpp)
target_link_libraries(analysis
    CUDA::cudart
    CUDA::cuda_driver)

add_library(gpuless SHARED
    trace/libgpuless.cpp
    trace/hijack_cudnn.cpp
    trace/hijack_cublas.cpp
    trace/cuda_trace.cpp
    trace/cuda_trace_converter.cpp
    trace/cuda_api_calls.cpp
    trace/cudnn_api_calls.cpp
    trace/cublas_api_calls.cpp
    trace/cuda_virtual_device.cpp
    trace/cuda_virtual_device.cpp
    trace/trace_executor_local.cpp
    trace/trace_executor_tcp_client.cpp
    trace/dlsym_util.cpp
    utils.cpp
    trace/cubin_analysis.cpp)
target_link_libraries(gpuless
    iberty
    stdc++fs
    cudnn
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas)

add_custom_command(
    TARGET gpuless POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR})
