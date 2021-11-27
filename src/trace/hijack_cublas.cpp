#include "cublas_api_calls.hpp"
#include "libgpuless.hpp"

#include <cublas.h>
#include <cublasLt.h>

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

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
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

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t
cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                               cublasLtMatmulDescAttributes_t attr,
                               const void *buf, size_t sizeInBytes) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t
cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
               const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
               const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
               const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
               cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
               void *workspace, size_t workspaceSizeInBytes,
               cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type, uint64_t rows,
                                          uint64_t cols, int64_t ld) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}

cublasStatus_t
cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                 cublasLtMatrixLayoutAttribute_t attr,
                                 const void *buf, size_t sizeInBytes) {
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
}
}

} // namespace gpuless
