#include <cstdio>
#include <cstring>
#include <memory>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>
#include <fatbinary_section.h>

#include "libgpuless.hpp"
#include "cuda_trace.hpp"
#include "trace_executor.hpp"

bool debug_print = true;

using namespace gpuless::executor;
using namespace gpuless;

static std::stack<CudaCallConfig> cudaCallConfigStack;

static CudaTrace &getCudaTrace() {
    static TraceExecutor traceExecutor;
    static CudaTrace cudaTrace(traceExecutor);
    return cudaTrace;
}

extern "C" {

/*
 * CUDA runtime API
 */

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaMalloc>(size));
    getCudaTrace().synchronize();
    *devPtr = std::static_pointer_cast<CudaMalloc>(getCudaTrace().historyTop())->devPtr;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind) {
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        dbgprintf("  cudaMemcpyHostToDevice\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyH2D>(dst, src, count));
    } else if (kind == cudaMemcpyDeviceToHost) {
        dbgprintf("  cudaMemcpyDeviceToHost\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyD2H>(dst, src, count));
        getCudaTrace().synchronize();
    } else {
        EXIT_NOT_IMPLEMENTED("cudaMemcpyKind");
    }
    return cudaSuccess;
}

// TODO
cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                      enum cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    return cudaSuccess;
}

// TODO
cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim,
                                       dim3 blockDim, void **args,
                                       size_t sharedMem, cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaLaunchKernel>(func, gridDim, blockDim,
                                                    args, sharedMem, stream));
    getCudaTrace().synchronize();
    return cudaSuccess;
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
    HIJACK_FN_PROLOGUE();
    cudaCallConfigStack.push({gridDim, blockDim, sharedMem, stream});
    return cudaSuccess;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
    HIJACK_FN_PROLOGUE();
    CudaCallConfig config = cudaCallConfigStack.top();
    cudaCallConfigStack.pop();
    *gridDim = config.gridDim;
    *blockDim = config.blockDim;
    *sharedMem = config.sharedMem;
    *((CUstream_st **)stream) = config.stream;
    return cudaSuccess;
}

// TODO
//void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
//    HIJACK_FN_PROLOGUE();
//
//    return nullptr;
//}
//
//// TODO
//void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
//    HIJACK_FN_PROLOGUE();
//}
//
//// TODO
//void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
//                                      const char *hostFun, char *deviceFun,
//                                      const char *deviceName, int thread_limit,
//                                      uint3 *tid, uint3 *bid, dim3 *bDim,
//                                      dim3 *gDim, int *wSize) {
//    HIJACK_FN_PROLOGUE();
//}

/*
 * CUDA driver API
 */

}
