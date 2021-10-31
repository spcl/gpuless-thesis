
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../adapter/cubin_analysis.hpp"
#include "../utils.hpp"
#include "cuda_trace.hpp"
#include "dlsym_util.hpp"
#include "libgpuless.hpp"
#include "trace_executor_local.hpp"

const int CUDA_MAJOR_VERSION = 8;
const int CUDA_MINOR_VERSION = 6;

bool debug_print = true;

using namespace gpuless::executor;
using namespace gpuless;

static std::stack<CudaCallConfig> cuda_call_config_stack;

static std::map<const void *, std::string> &getSymbolMap() {
    static std::map<const void *, std::string> fnptr_to_symbol;
    return fnptr_to_symbol;
}

static void exitHandler();

static CudaTrace &getCudaTrace() {
    static auto traceExecutor = std::make_shared<TraceExecutorLocal>();
    static CubinAnalyzer cubinAnalyzer;
    static CudaTrace cudaTrace(traceExecutor, cubinAnalyzer);

    static bool exit_handler_registerd = false;
    if (!exit_handler_registerd) {
        exit_handler_registerd = true;
        std::atexit([]() {
            exitHandler();
        });
    }

    if (!cubinAnalyzer.isInitialized()) {
        char *cuda_binary = std::getenv("CUDA_BINARY");
        if (cuda_binary == nullptr) {
            std::cerr << "please set CUDA_BINARY environment variable"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<std::string> binaries;
        string_split(std::string(cuda_binary), ',', binaries);
        dbgprintf("analyzing CUDA binaries (%s)\n", cuda_binary);
        cubinAnalyzer.analyze(binaries, CUDA_MAJOR_VERSION, CUDA_MINOR_VERSION);
    }

    return cudaTrace;
}

static void exitHandler() {
    dbgprintf("std::atexit()\n");
    // this does not work, because CUDA exit handler have already destructed
    // the drive runtime at std::atexit time
//    getCudaTrace().synchronize();
}

extern "C" {

/*
 * CUDA runtime API
 */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaMalloc>(size));
    getCudaTrace().synchronize();
    *devPtr = std::static_pointer_cast<CudaMalloc>(getCudaTrace().historyTop())
                  ->devPtr;
    printf("  %p\n", *devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        dbgprintf("  cudaMemcpyHostToDevice\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyH2D>(dst, src, count));
        dbgprintf("  %p <- %p\n", dst, src);
    } else if (kind == cudaMemcpyDeviceToHost) {
        dbgprintf("  cudaMemcpyDeviceToHost\n");
        auto rec = std::make_shared<CudaMemcpyD2H>(dst, src, count);
        getCudaTrace().record(rec);
        dbgprintf("  %p <- %p\n", dst, src);
        getCudaTrace().synchronize();
        std::memcpy(dst, rec->buffer.data(), count);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        dbgprintf("  cudaMemcpyDeviceToDevice\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyD2D>(dst, src, count));
    } else {
        EXIT_NOT_IMPLEMENTED("cudaMemcpyKind");
    }
    return cudaSuccess;
}

// TODO
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        dbgprintf("  cudaMemcpyHostToDevice\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyAsyncH2D>(dst, src, count, stream));
        dbgprintf("  %p <- %p\n", dst, src);
    } else if (kind == cudaMemcpyDeviceToHost) {
        dbgprintf("  cudaMemcpyDeviceToHost\n");
        auto rec = std::make_shared<CudaMemcpyAsyncD2H>(dst, src, count, stream);
        getCudaTrace().record(rec);
        dbgprintf("  %p <- %p\n", dst, src);
        getCudaTrace().synchronize();
        std::memcpy(dst, rec->buffer.data(), count);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        dbgprintf("  cudaMemcpyDeviceToDevice\n");
        getCudaTrace().record(std::make_shared<CudaMemcpyAsyncD2D>(dst, src, count, stream));
    } else {
        EXIT_NOT_IMPLEMENTED("cudaMemcpyKind");
    }
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();

    auto it = getSymbolMap().find(func);
    if (it == getSymbolMap().end()) {
        EXIT_UNRECOVERABLE("unknown function");
    }
    std::string &symbol = it->second;
    dbgprintf("  %s\n", cpp_demangle(symbol).c_str());

    std::vector<KParamInfo> paramInfos;
    const auto &analyzer = getCudaTrace().cubinAnalyzer();
    if (!analyzer.kernel_parameters(symbol, paramInfos)) {
        EXIT_UNRECOVERABLE("unable to look up kernel parameter data");
    }

    std::vector<std::vector<uint8_t>> paramBuffers(paramInfos.size());
    for (unsigned i = 0; i < paramInfos.size(); i++) {
        const auto &p = paramInfos[i];
        paramBuffers[i].resize(p.size * p.typeSize);
        std::memcpy(paramBuffers[i].data(), args[i], p.size * p.typeSize);
    }

    getCudaTrace().record(std::make_shared<CudaLaunchKernel>(
        func, gridDim, blockDim, sharedMem, stream, paramBuffers, paramInfos));
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaFree>(devPtr));
    // have to synchronize here until i find a way to hook the cuda exit handler
    getCudaTrace().synchronize();
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaStreamSynchronize>(stream));
    return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void) {
    HIJACK_FN_PROLOGUE();
    // TODO
    return cudaSuccess;
}

// TODO
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
    HIJACK_FN_PROLOGUE();
    static auto real_func =
        (decltype(&cudaStreamCreateWithFlags))real_dlsym(RTLD_NEXT, __func__);
    return real_func(pStream, flags);
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0,
                                     struct CUstream_st *stream = 0) {
    HIJACK_FN_PROLOGUE();
    cuda_call_config_stack.push({gridDim, blockDim, sharedMem, stream});
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    HIJACK_FN_PROLOGUE();
    CudaCallConfig config = cuda_call_config_stack.top();
    cuda_call_config_stack.pop();
    *gridDim = config.gridDim;
    *blockDim = config.blockDim;
    *sharedMem = config.sharedMem;
    *((CUstream_st **)stream) = config.stream;
    return cudaSuccess;
}

// TODO
void **__cudaRegisterFatBinary(void *fatCubin) {
    HIJACK_FN_PROLOGUE();

    //    std::vector<uint8_t> fatBinCWrapperCpy(sizeof(__fatBinC_Wrapper_t));
    //    std::memcpy(fatBinCWrapperCpy.data(), fatCubin,
    //    sizeof(__fatBinC_Wrapper_t));

    //    auto fatBinCWrapper = static_cast<__fatBinC_Wrapper_t *>(fatCubin);
    //    const long long unsigned *data = fatBinCWrapper->data;
    //    uint64_t imgSize = data[1];

    //    std::cout << "sndGW=" << imgSize << std::endl;
    //    std::cout << "xx=" << fatBinCWrapper->filename_or_fatbins <<
    //    std::endl;

    static auto real_func =
        (decltype(&__cudaRegisterFatBinary))real_dlsym(RTLD_NEXT, __func__);
    return real_func(fatCubin);
}

// TODO
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    HIJACK_FN_PROLOGUE();
    static auto real_func =
        (decltype(&__cudaRegisterFatBinaryEnd))real_dlsym(RTLD_NEXT, __func__);
    return real_func(fatCubinHandle);
}

// TODO
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
    //    HIJACK_FN_PROLOGUE();
    dbgprintf("%s(%s)\n", __func__, cpp_demangle(deviceName).c_str());
    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(hostFun), deviceName));
    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(deviceFun), deviceName));
    static auto real_func =
        (decltype(&__cudaRegisterFunction))real_dlsym(RTLD_NEXT, __func__);
    real_func(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
              bid, bDim, gDim, wSize);
}

//void __cudaUnregisterFatBinary(void **fatCubinHandle) {
//    HIJACK_FN_PROLOGUE();
//    (void)fatCubinHandle;
//    getCudaTrace().synchronize();
//}

/*
 * CUDA driver API
 */

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    HIJACK_FN_PROLOGUE();
    (void)dev;
    getCudaTrace().synchronize();
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
    static auto real =
        (decltype(&cuGetProcAddress))real_dlsym(RTLD_NEXT, __func__);
//    dbgprintf("%s(%s)\n", __func__, symbol);

    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuDevicePrimaryCtxRelease_v2);

//    if (strncmp(symbol, "cu", 2) == 0) {
//        dbgprintf("cuGetProcAddress(%s): symbol not implemented\n", symbol);
//    }

    return real(symbol, pfn, cudaVersion, flags);
}

void *dlsym(void *handle, const char *symbol) {
    dbgprintf("dlsym(%s) [pid=%d]\n", symbol, getpid());

    // early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        return (void *)&cuGetProcAddress;
    }

    return real_dlsym(handle, symbol);
}

}
