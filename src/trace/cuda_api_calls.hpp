#ifndef __CUDA_API_CALLS_H__
#define __CUDA_API_CALLS_H__

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <iostream>

#include "../adapter/cubin_analysis.hpp"

namespace gpuless {

class CudaApiCall {
  public:
    virtual cudaError_t executeNative() = 0;
    virtual ~CudaApiCall() = default;

    virtual std::string typeName() { return typeid(*this).name(); }
};

/*
 * Public CUDA API functions
 */

class CudaMalloc : public CudaApiCall {
  public:
    void *devPtr;
    size_t size;

    explicit CudaMalloc(size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyH2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    std::vector<uint8_t> buffer;

    CudaMemcpyH2D(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyD2H : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    std::vector<uint8_t> buffer;

    CudaMemcpyD2H(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyD2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;

    CudaMemcpyD2D(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyAsyncH2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    std::vector<uint8_t> buffer;

    CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative() override;
};

class CudaMemcpyAsyncD2H : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;
    std::vector<uint8_t> buffer;

    CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative() override;
};

class CudaMemcpyAsyncD2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;
    cudaStream_t stream;

    CudaMemcpyAsyncD2D(void *dst, const void *src, size_t size,
                       cudaStream_t stream);
    cudaError_t executeNative() override;
};

class CudaFree : public CudaApiCall {
  public:
    void *devPtr;

    CudaFree(void *devPtr);
    cudaError_t executeNative() override;
};

class CudaLaunchKernel : public CudaApiCall {
  public:
    const void *fnPtr;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;

    std::vector<std::vector<uint8_t>> paramBuffers;
    std::vector<KParamInfo> paramInfos;

    CudaLaunchKernel(const void *fnPtr, const dim3 &gridDim,
                     const dim3 &blockDim, size_t sharedMem,
                     cudaStream_t stream,
                     std::vector<std::vector<uint8_t>> &paramBuffers,
                     std::vector<KParamInfo> &paramInfos);
    cudaError_t executeNative() override;
};

class CudaStreamSynchronize : public CudaApiCall {
  public:
    cudaStream_t stream;

    CudaStreamSynchronize(cudaStream_t stream);
    cudaError_t executeNative() override;
};

/*
 * Private CUDA API functions
 */

class PrivCudaPushCallConfiguration : public CudaApiCall {
  public:
};

class PrivCudaPopCallConfiguration : public CudaApiCall {
  public:
};

class PrivCudaRegisterFatBinary : public CudaApiCall {
  public:
};

class PrivCudaUnregisterFatBinary : public CudaApiCall {
  public:
};

class PrivCudaRegisterFunction : public CudaApiCall {
  public:
};

} // namespace gpuless

#endif // __CUDA_API_CALLS_H__
