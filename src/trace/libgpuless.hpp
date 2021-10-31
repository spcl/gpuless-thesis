#ifndef __LIBGPULESS_HPP__
#define __LIBGPULESS_HPP__

#include <iostream>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>

#define LINK_CU_FUNCTION(symbol, f)                                            \
    do {                                                                       \
        if (strcmp(symbol, #f) == 0) {                                         \
            *pfn = (void *)&f;                                                 \
            return CUDA_SUCCESS;                                               \
        }                                                                      \
    } while (0)

#define HIJACK_FN_PROLOGUE()                                                   \
    do {                                                                       \
        dbgprintf("%s() [pid=%d]\n", __func__, getpid());                      \
    } while (0)

#define dbgprintf(str, ...)                                                    \
    do {                                                                       \
        if (debug_print) {                                                     \
            printf("[libgpuless] ");                                           \
            printf(str, ##__VA_ARGS__);                                        \
        }                                                                      \
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
    size_t sharedMem;
    struct CUstream_st *stream;
};

#endif // __LIBGPULESS_HPP__