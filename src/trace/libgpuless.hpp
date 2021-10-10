#ifndef __LIBGPULESS_HPP__
#define __LIBGPULESS_HPP__

#include <iostream>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>

#define HIJACK_FN_PROLOGUE()                                                   \
    do {                                                                       \
        dbgprintf("%s()\n", __func__);                                         \
    } while (0)

#define dbgprintf(str, ...)                                                    \
    do {                                                                       \
        if (debug_print) {                                                     \
            printf("[libgpuless] ");                                           \
            printf(str, ##__VA_ARGS__);                                        \
        }                                                                      \
    } while (0)

#define EXIT_NOT_IMPLEMENTED(fn)                                               \
    std::cerr << "not implemented: " << fn << std::endl;                       \
    std::exit(EXIT_FAILURE);

struct CudaCallConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    struct CUstream_st *stream;
};

#endif // __LIBGPULESS_HPP__