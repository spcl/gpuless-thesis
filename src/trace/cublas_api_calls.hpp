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
    explicit CublasCreateV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSetStreamV2 : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cudaStream_t stream;

    CublasSetStreamV2(uint64_t virtualHandle, cudaStream_t stream);
    explicit CublasSetStreamV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

class CublasSetMathMode : public CublasApiCAll {
  public:
    uint64_t virtual_handle;
    cublasMath_t mode;

    CublasSetMathMode(uint64_t virtualHandle, cublasMath_t mode);
    explicit CublasSetMathMode(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
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
    explicit CublasSgemmV2(const FBCudaApiCall *fb_cuda_api_call);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;

    flatbuffers::Offset<FBCudaApiCall>
    fbSerialize(flatbuffers::FlatBufferBuilder &builder) override;
};

} // namespace gpuless

#endif // GPULESS_CUBLAS_API_CALLS_HPP
