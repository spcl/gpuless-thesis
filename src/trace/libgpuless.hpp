#ifndef __LIBGPULESS_HPP__
#define __LIBGPULESS_HPP__

#include <iostream>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>

#define LINK_CU_FUNCTION(symbol, f)                                            \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            spdlog::debug("{}({}) [pid={}]", __func__, symbol, getpid());      \
            *pfn = (void *)&f;                                                 \
            return CUDA_SUCCESS;                                               \
        }                                                                      \
    } while (0)

#define HIJACK_FN_PROLOGUE()                                                   \
    do {                                                                       \
        spdlog::info("{}() [pid={}]", __func__, getpid());                     \
    } while (0)

#define EXIT_NOT_IMPLEMENTED(fn)                                               \
    do {                                                                       \
        std::cerr << "not implemented: " << fn << std::endl;                   \
        std::exit(EXIT_FAILURE);                                               \
    } while (0)

#define EXIT_UNRECOVERABLE(msg)                                                \
    do {                                                                       \
        std::cerr << msg << std::endl;                                         \
        std::cerr << "unrecoverable error, exiting" << std::endl;              \
        std::exit(EXIT_FAILURE);                                               \
    } while (0)

struct CudaCallConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem{};
    struct CUstream_st *stream{};
};

struct CudaRegisterState {
    uint64_t current_fatbin_handle;
    bool is_registering;
};

#endif // __LIBGPULESS_HPP__