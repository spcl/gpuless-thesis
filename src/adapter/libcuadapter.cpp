#include <dlfcn.h>
#include <cstring>
#include <iostream>

#include <cuda.h>

#define LINK_CU_FUNCTION(symbol, f)     \
    do {                                \
        if (strcmp(symbol, #f) == 0) {  \
            *pfn = (void *) &f;         \
            return CUDA_SUCCESS;        \
        }                               \
    } while(0);

extern "C" { void *__libc_dlsym(void *map, const char *name); }
extern "C" { void *__libc_dlopen_mode(const char *name, int mode); }

typedef void* (*fnDlsym)(void *, const char *);
static void* real_dlsym(void *handle, const char *symbol) {
    static fnDlsym internal_dlsym = (fnDlsym) __libc_dlsym(
            __libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    std::cout << __func__ << "()" << std::endl;
    const char n[] = "libcuadapter";
    strncpy(name, "libcuadapter", sizeof(n));
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimxX,
                                unsigned int gridDimxY,
                                unsigned int gridDimxZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
    std::cout << __func__ << "()" << std::endl;
    return CUDA_SUCCESS;
}

typedef CUresult CUDAAPI (*fnCuGetProcAddress)(const char *, void **, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    static fnCuGetProcAddress real_func = (fnCuGetProcAddress) real_dlsym(RTLD_NEXT, __func__);

    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuInit);
    LINK_CU_FUNCTION(symbol, cuDeviceGet);
    LINK_CU_FUNCTION(symbol, cuDeviceGetName);
    LINK_CU_FUNCTION(symbol, cuCtxCreate_v2);
    LINK_CU_FUNCTION(symbol, cuModuleLoad);
    LINK_CU_FUNCTION(symbol, cuModuleGetFunction);
    LINK_CU_FUNCTION(symbol, cuMemAlloc_v2);
    LINK_CU_FUNCTION(symbol, cuMemcpyHtoD_v2);
    LINK_CU_FUNCTION(symbol, cuMemcpyDtoH_v2);
    LINK_CU_FUNCTION(symbol, cuLaunchKernel);
    LINK_CU_FUNCTION(symbol, cuCtxSynchronize);

    printf("cuGetProcAddress(%s)\n", symbol);
    return real_func(symbol, pfn, cudaVersion, flags);
}

void *dlsym(void *handle, const char *symbol) {
    // early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        return (void*) &cuGetProcAddress;
    }

    return (real_dlsym(handle, symbol));
}
