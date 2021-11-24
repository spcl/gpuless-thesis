#ifndef GPULESS_CUBLAS_API_CALLS_HPP
#define GPULESS_CUBLAS_API_CALLS_HPP

#include "cuda_api_calls.hpp"

namespace gpuless {

class CublasApiCAll : public AbstractCudaApiCall {
  public:
    std::string nativeErrorToString(uint64_t err) override;
};

class CublasCreateV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;

    explicit CublasCreateV2(uint64_t virtualHandle);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CublasSetStreamV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cudaStream_t stream;

    CublasSetStreamV2(uint64_t virtualHandle, cudaStream_t stream);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CublasSetMathMode : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasMath_t mode;

    CublasSetMathMode(uint64_t virtualHandle, cublasMath_t mode);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CublasSgemmV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasOperation_t transa;
    cublasOperation_t transb;
    int m;
    int n;
    int k;
    float alpha;
    float beta;
    const float *A;
    const float *B;
    const float *C;
    int lda;
    int ldb;
    int ldc;

    CublasSgemmV2(uint64_t virtualHandle, cublasOperation_t transa,
                  cublasOperation_t transb, int m, int n, int k, float alpha,
                  float beta, const float *a, const float *b, const float *c,
                  int lda, int ldb, int ldc);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

} // namespace gpuless

#endif // GPULESS_CUBLAS_API_CALLS_HPP
