#ifndef __LIBCUADAPTER_HPP__
#define __LIBCUADAPTER_HPP__

#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>

#define LINK_CU_FUNCTION(symbol, f)    \
    do {                               \
        if (strcmp(symbol, #f) == 0) { \
            *pfn = (void *)&f;         \
            return CUDA_SUCCESS;       \
        }                              \
    } while (0)

#define HIJACK_FN_PROLOGUE()           \
    do {                               \
        dbgprintf("%s()\n", __func__); \
        if (!vdev_initialized) {       \
            vdev_initialize();         \
        }                              \
    } while (0)

#define dbgprintf(str, ...)                              \
    do {                                                 \
        if (debug_print) {                               \
            char prefix[] = "[libcuadapter] ";           \
            size_t s = sizeof(prefix) + strlen(str);     \
            char *dst = (char *)calloc(sizeof(char), s); \
            strncpy(dst, prefix, sizeof(char) * s);      \
            strncat(dst, str, sizeof(char) * s);         \
            printf(dst, ##__VA_ARGS__);                  \
            free(dst);                                   \
        }                                                \
    } while (0)

struct CudaCallConfig {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    struct CUstream_st *stream;
};

std::stack<CudaCallConfig> cuda_call_config_stack;

// pointers types for the real dlsym function
typedef void *(*fnDlsym)(void *, const char *);

// pointers types for real CUDA driver functions
typedef CUresult CUDAAPI (*fnCuGetProcAddress)(const char *, void **, int,
                                               cuuint64_t);
typedef CUresult CUDAAPI (*fnCuModuleLoadData)(CUmodule *, const void *);
typedef CUresult CUDAAPI (*fnCuInit)(unsigned int);

extern "C" {
void *__libc_dlsym(void *map, const char *name);
}

extern "C" {
void *__libc_dlopen_mode(const char *name, int mode);
}

#endif  // __LIBCUADAPTER_HPP__