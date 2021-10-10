cmake_minimum_required(VERSION 3.17)
project(gpuless)

enable_language(CUDA)
find_package(CUDAToolkit)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -Wall -Wextra -Werror -Wno-unused-parameter -ggdb")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

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

# add_executable(test_tcp test/test_tcp.cpp executors/executor_tcp.cpp utils.cpp)
# add_dependencies(test_tcp flatbuffer)
# target_link_libraries(test_tcp boost_serialization CUDA::cudart CUDA::cuda_driver)

# add_library(cuadapter SHARED
#             adapter/libcuadapter.cpp
#             adapter/cubin_analysis.cpp
#             adapter/cuda_vdev.cpp
#             executors/executor_tcp.cpp
#             utils.cpp)
# add_dependencies(cuadapter flatbuffer)
# target_link_libraries(cuadapter CUDA::cuda_driver)

# # set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
# # set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
add_executable(image-recognition benchmarks/image-recognition.cpp)
target_link_libraries(image-recognition "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

# add_executable(image-recognition-exe benchmarks/image-recognition-exe.cpp)
# target_link_libraries(image-recognition-exe "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
# set_target_properties(image-recognition PROPERTIES LINK_SEARCH_START_STATIC 1)
# set_target_properties(image-recognition PROPERTIES LINK_SEARCH_END_STATIC 1)

add_library(analysis SHARED
    analysis/libcudaanalysis.cpp
    adapter/cubin_analysis.cpp
    utils.cpp)
target_link_libraries(analysis CUDA::cudart CUDA::cuda_driver)

add_library(gpuless SHARED
    trace/libgpuless.cpp
    trace/cuda_trace.cpp
    trace/cuda_api_calls.cpp
    trace/trace_executor.cpp
    trace/dlsym_util.cpp)
target_link_libraries(gpuless CUDA::cudart)

add_executable(trace_test trace/trace_test.cu)
target_link_libraries(gpuless CUDA::cudart)
