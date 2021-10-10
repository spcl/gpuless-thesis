#ifndef __CUDA_API_CALLS_H__
#define __CUDA_API_CALLS_H__

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <typeinfo>

namespace gpuless {

class CudaApiCall {
  public:
    virtual cudaError_t executeNative() = 0;
    virtual ~CudaApiCall() = default;

    virtual std::string typeName() {
        return typeid(*this).name();
    }
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

    CudaMemcpyD2H(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyAsyncH2D : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;

    CudaMemcpyAsyncH2D(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaMemcpyAsyncD2H : public CudaApiCall {
  public:
    void *dst;
    const void *src;
    size_t size;

    CudaMemcpyAsyncD2H(void *dst, const void *src, size_t size);
    cudaError_t executeNative() override;
};

class CudaLaunchKernel : public CudaApiCall {
  public:
    const void *fnPtr;
    dim3 gridDim;
    dim3 blockDim;
    void **args;
    size_t sharedMem;
    cudaStream_t stream;

    CudaLaunchKernel(const void *fnPtr, const dim3 &gridDim,
                     const dim3 &blockDim, void **args, size_t sharedMem,
                     cudaStream_t stream);
    cudaError_t executeNative() override;
};

/*
 * Private CUDA API functions
 */

class PrivCudaPushCallConfiguration : public CudaApiCall {
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
