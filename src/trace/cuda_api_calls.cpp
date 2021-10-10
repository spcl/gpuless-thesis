#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "cuda_api_calls.hpp"
#include "dlsym_util.hpp"

/*
 * cudaMalloc
 */
gpuless::CudaMalloc::CudaMalloc(size_t size) : devPtr(nullptr), size(size) {}

cudaError_t gpuless::CudaMalloc::executeNative() {
    static auto realCudaMalloc =
        (decltype(&cudaMalloc<void>))real_dlsym(RTLD_NEXT, "cudaMalloc");
    auto r = realCudaMalloc(&this->devPtr, size);
    return r;
}

/*
 * cudaMemcpyH2D
 */
gpuless::CudaMemcpyH2D::CudaMemcpyH2D(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size), buffer(size) {
    std::memcpy(this->buffer.data(), src, size);
}

cudaError_t gpuless::CudaMemcpyH2D::executeNative() {
    static auto realCudaMemcpy =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return realCudaMemcpy(this->dst, this->buffer.data(), this->size,
                          cudaMemcpyHostToDevice);
}

/*
 * cudaMemcpyD2H
 */
gpuless::CudaMemcpyD2H::CudaMemcpyD2H(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size) {}

cudaError_t gpuless::CudaMemcpyD2H::executeNative() {
    static auto realCudaMemcpy =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return realCudaMemcpy(this->dst, this->src, this->size,
                          cudaMemcpyDeviceToHost);
}

/*
 * cudaMemcpyAsyncH2D
 */
gpuless::CudaMemcpyAsyncH2D::CudaMemcpyAsyncH2D(void *dst, const void *src,
                                                size_t size)
    : dst(dst), src(src), size(size) {}

cudaError_t gpuless::CudaMemcpyAsyncH2D::executeNative() {
    static auto realCudaMemcpyAsync =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return realCudaMemcpyAsync(this->dst, this->src, this->size,
                               cudaMemcpyHostToDevice, 0);
}

/*
 * cudaMemcpyAsyncD2H
 */
gpuless::CudaMemcpyAsyncD2H::CudaMemcpyAsyncD2H(void *dst, const void *src,
                                                size_t size)
    : dst(dst), src(src), size(size) {}

cudaError_t gpuless::CudaMemcpyAsyncD2H::executeNative() {
    static auto realCudaMemcpyAsync =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return realCudaMemcpyAsync(this->dst, this->src, this->size,
                               cudaMemcpyDeviceToHost, 0);
}

/*
 * cudaLaunchKernel
 */
gpuless::CudaLaunchKernel::CudaLaunchKernel(const void *fnPtr,
                                            const dim3 &gridDim,
                                            const dim3 &blockDim, void **args,
                                            size_t sharedMem,
                                            cudaStream_t stream)
    : fnPtr(fnPtr), gridDim(gridDim), blockDim(blockDim), args(args),
      sharedMem(sharedMem), stream(stream) {}

cudaError_t gpuless::CudaLaunchKernel::executeNative() {
    static auto realCudaLaunchKernel =
        (decltype(&cudaLaunchKernel<void>))real_dlsym(RTLD_NEXT,
                                                      "cudaLaunchKernel");
    return realCudaLaunchKernel(this->fnPtr, this->gridDim, this->blockDim,
                                this->args, this->sharedMem, this->stream);
}
