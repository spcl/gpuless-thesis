#include "cublas_api_calls.hpp"
#include "../schemas/cublas_calls_generated.h"
#include "../schemas/trace_execution_protocol_generated.h"
#include "libgpuless.hpp"

namespace gpuless {

std::string gpuless::CublasApiCAll::nativeErrorToString(uint64_t err) {
    auto status = static_cast<cublasStatus_t>(err);
    std::string str_err;

    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        str_err = "CUBLAS_STATUS_SUCCESS";
        break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
        str_err = "CUBLAS_STATUS_NOT_INITIALIZED";
        break;
    case CUBLAS_STATUS_ALLOC_FAILED:
        str_err = "CUBLAS_STATUS_ALLOC_FAILED";
        break;
    case CUBLAS_STATUS_INVALID_VALUE:
        str_err = "CUBLAS_STATUS_INVALID_VALUE";
        break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
        str_err = "CUBLAS_STATUS_ARCH_MISMATCH";
        break;
    case CUBLAS_STATUS_MAPPING_ERROR:
        str_err = "CUBLAS_STATUS_MAPPING_ERROR";
        break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
        str_err = "CUBLAS_STATUS_EXECUTION_FAILED";
        break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
        str_err = "CUBLAS_STATUS_INTERNAL_ERROR";
        break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
        str_err = "CUBLAS_STATUS_NOT_SUPPORTED";
        break;
    case CUBLAS_STATUS_LICENSE_ERROR:
        str_err = "CUBLAS_STATUS_LICENSE_ERROR";
        break;
    }

    return "[cublas] " + str_err;
}

/*
 * cublasCreate_v2
 */
gpuless::CublasCreateV2::CublasCreateV2(uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

uint64_t gpuless::CublasCreateV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasCreate_v2);
    if (vdev.cublas_handle_virtual_to_real.size() < this->virtual_handle + 1) {
        vdev.cublas_handle_virtual_to_real.resize(this->virtual_handle + 1);
    }
    return real(&vdev.cublas_handle_virtual_to_real[this->virtual_handle]);
}

flatbuffers::Offset<FBCudaApiCall>
gpuless::CublasCreateV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasCreateV2(builder, this->virtual_handle);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasCreateV2, api_call.Union());
    return api_call_union;
}

CublasCreateV2::CublasCreateV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasCreateV2();
    this->virtual_handle = c->virtual_handle();
}

/*
 * cublasSetStream_v2
 */
gpuless::CublasSetStreamV2::CublasSetStreamV2(uint64_t virtualHandle,
                                              cudaStream_t stream)
    : virtual_handle(virtualHandle), stream(stream) {}

uint64_t gpuless::CublasSetStreamV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSetStream_v2);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->stream);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSetStreamV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasSetStreamV2(builder, this->virtual_handle,
                                  reinterpret_cast<uint64_t>(this->stream));
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSetStreamV2, api_call.Union());
    return api_call_union;
}

CublasSetStreamV2::CublasSetStreamV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSetStreamV2();
    this->virtual_handle = c->virtual_handle();
    this->stream = reinterpret_cast<cudaStream_t>(c->stream());
}

/*
 * cublasSetMathMode
 */
gpuless::CublasSetMathMode::CublasSetMathMode(uint64_t virtualHandle,
                                              cublasMath_t mode)
    : virtual_handle(virtualHandle), mode(mode) {}

uint64_t gpuless::CublasSetMathMode::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSetMathMode);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->mode);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSetMathMode::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call =
        CreateFBCublasSetMathMode(builder, this->virtual_handle, this->mode);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSetMathMode, api_call.Union());
    return api_call_union;
}

CublasSetMathMode::CublasSetMathMode(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSetMathMode();
    this->virtual_handle = c->virtual_handle();
    this->mode = static_cast<cublasMath_t>(c->math_mode());
}

/*
 * cublasSgemm_v2
 */
gpuless::CublasSgemmV2::CublasSgemmV2(uint64_t virtualHandle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb, int m, int n,
                                      int k, float alpha, float beta,
                                      const float *a, const float *b,
                                      const float *c, int lda, int ldb, int ldc)
    : virtual_handle(virtualHandle), transa(transa), transb(transb), m(m), n(n),
      k(k), alpha(alpha), beta(beta), A(a), B(b), C(c), lda(lda), ldb(ldb),
      ldc(ldc) {}

uint64_t gpuless::CublasSgemmV2::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cublasSgemm_v2);
    return real(vdev.cublas_handle_virtual_to_real[this->virtual_handle],
                this->transa, this->transb, this->m, this->n, this->k,
                &this->alpha, this->A, this->lda, this->B, this->ldb,
                &this->beta, const_cast<float *>(this->C), this->ldc);
}

flatbuffers::Offset<FBCudaApiCall>
CublasSgemmV2::fbSerialize(flatbuffers::FlatBufferBuilder &builder) {
    auto api_call = CreateFBCublasSgemmV2(
        builder, this->virtual_handle, this->transa, this->transb, this->m,
        this->n, this->k, this->alpha, this->beta,
        reinterpret_cast<uint64_t>(this->A),
        reinterpret_cast<uint64_t>(this->B),
        reinterpret_cast<uint64_t>(this->C), this->lda, this->ldb, this->ldc);
    auto api_call_union = CreateFBCudaApiCall(
        builder, FBCudaApiCallUnion_FBCublasSgemmV2, api_call.Union());
    return api_call_union;
}

CublasSgemmV2::CublasSgemmV2(const FBCudaApiCall *fb_cuda_api_call) {
    auto c = fb_cuda_api_call->api_call_as_FBCublasSgemmV2();
    this->virtual_handle = c->virtual_handle();
    this->transa = static_cast<cublasOperation_t>(c->transa_op());
    this->transb = static_cast<cublasOperation_t>(c->transb_op());
    this->m = c->m();
    this->n = c->n();
    this->k = c->k();
    this->alpha = c->alpha();
    this->beta = c->alpha();
    this->A = reinterpret_cast<const float *>(c->a());
    this->B = reinterpret_cast<const float *>(c->b());
    this->C = reinterpret_cast<const float *>(c->c());
    this->lda = c->lda();
    this->ldb = c->ldb();
    this->ldc = c->ldc();
}

} // namespace gpuless
