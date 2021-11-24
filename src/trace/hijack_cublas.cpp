#include "cublas_api_calls.hpp"
#include "libgpuless.hpp"

#include <cublas.h>

namespace gpuless {

static uint64_t nextCublasHandle() {
    static uint64_t next = 1;
    return next++;
}

extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {
    HIJACK_FN_PROLOGUE();
    auto virtual_handle = nextCublasHandle();
    *handle = reinterpret_cast<cublasHandle_t>(virtual_handle);
    getCudaTrace().record(std::make_shared<CublasCreateV2>(virtual_handle));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle,
                                  cudaStream_t streamId) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSetStreamV2>(
        reinterpret_cast<uint64_t>(handle), streamId));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSetMathMode>(
        reinterpret_cast<uint64_t>(handle), mode));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float *alpha, const float *A, int lda,
                              const float *B, int ldb, const float *beta,
                              float *C, int ldc) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CublasSgemmV2>(
        reinterpret_cast<uint64_t>(handle), transa, transb, m, n, k, *alpha,
        *beta, A, B, C, lda, ldb, ldc));
    return CUBLAS_STATUS_SUCCESS;
}
}

} // namespace gpuless
