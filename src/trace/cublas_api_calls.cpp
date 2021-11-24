#include "cublas_api_calls.hpp"
#include "libgpuless.hpp"

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

    return str_err;
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