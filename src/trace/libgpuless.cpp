#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stack>

#include <cuda.h>
#include <cuda_runtime.h>
#include <fatbinary_section.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "../utils.hpp"
#include "cubin_analysis.hpp"
#include "cuda_trace.hpp"
#include "dlsym_util.hpp"
#include "libgpuless.hpp"
#include "trace_executor_local.hpp"

using namespace gpuless;

const int CUDA_MAJOR_VERSION = 8;
const int CUDA_MINOR_VERSION = 6;

const bool debug_register_func = false;

static void hijackInit() {
    static bool hijack_initialized = false;
    if (!hijack_initialized) {
        hijack_initialized = true;
        spdlog::debug("hijackInit()");

        // load log level from env variable SPDLOG_LEVEL
        spdlog::cfg::load_env_levels();
    }
}

static std::stack<CudaCallConfig> &getCudaCallConfigStack() {
    static std::stack<CudaCallConfig> stack;
    return stack;
}

static std::map<const void *, std::string> &getSymbolMap() {
    static std::map<const void *, std::string> fnptr_to_symbol;
    return fnptr_to_symbol;
}

static CudaRegisterState &getCudaRegisterState() {
    static CudaRegisterState state{0, false};
    return state;
}

static uint64_t incrementFatbinCount() {
    static std::atomic<uint64_t> ctr = 1;
    return ctr++;
}

static void exitHandler();

static std::shared_ptr<TraceExecutor> getTraceExecutor() {
    static auto trace_executor = std::make_shared<TraceExecutorLocal>();
    return trace_executor;
}

static CudaTrace &getCudaTrace() {
    static auto cuda_virtual_device = std::make_shared<CudaVirtualDevice>();
    static CubinAnalyzer cubin_analyzer;
    static CudaTrace cuda_trace(cubin_analyzer, cuda_virtual_device);

    static bool exit_handler_registered = false;
    if (!exit_handler_registered) {
        exit_handler_registered = true;
        std::atexit([]() { exitHandler(); });
    }

    if (!cubin_analyzer.isInitialized()) {
        char *cuda_binary = std::getenv("CUDA_BINARY");
        if (cuda_binary == nullptr) {
            std::cerr << "please set CUDA_BINARY environment variable"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<std::string> binaries;
        string_split(std::string(cuda_binary), ',', binaries);
        spdlog::info("Analyzing CUDA binaries ({})", cuda_binary);
        cubin_analyzer.analyze(binaries, CUDA_MAJOR_VERSION,
                               CUDA_MINOR_VERSION);
    }

    return cuda_trace;
}

static void exitHandler() {
    spdlog::debug("std::atexit()");
    // this does not work, because CUDA exit handler have already destructed
    // the drive runtime at std::atexit time
    //    getCudaTrace().synchronize();
}

extern "C" {

/*
 * CUDA runtime API
 */

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaMalloc>(size));
    getTraceExecutor()->synchronize(getCudaTrace());
    *devPtr = std::static_pointer_cast<CudaMalloc>(getCudaTrace().historyTop())
                  ->devPtr;
    //    printf("  %p\n", *devPtr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        spdlog::info("{}() [cudaMemcpyHostToDevice, {} <- {}, pid={}]",
                     __func__, dst, src, getpid());
        getCudaTrace().record(std::make_shared<CudaMemcpyH2D>(dst, src, count));
    } else if (kind == cudaMemcpyDeviceToHost) {
        spdlog::info("{}() [cudaMemcpyDeviceToHost, {} <- {}, pid={}]",
                     __func__, dst, src, getpid());
        auto rec = std::make_shared<CudaMemcpyD2H>(dst, src, count);
        getCudaTrace().record(rec);
        getTraceExecutor()->synchronize(getCudaTrace());
        std::memcpy(dst, rec->buffer.data(), count);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        spdlog::info("{}() [cudaMemcpyDeviceToDevice, {} <- {}, pid={}]",
                     __func__, dst, src, getpid());
        getCudaTrace().record(std::make_shared<CudaMemcpyD2D>(dst, src, count));
    } else {
        EXIT_NOT_IMPLEMENTED("cudaMemcpyKind");
    }
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    hijackInit();
    //    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        spdlog::info(
            "{}() [cudaMemcpyHostToDevice, {} <- {}, stream={}, pid={}]",
            __func__, dst, src, reinterpret_cast<uint64_t>(stream), getpid());
        getCudaTrace().record(
            std::make_shared<CudaMemcpyAsyncH2D>(dst, src, count, stream));
    } else if (kind == cudaMemcpyDeviceToHost) {
        spdlog::info(
            "{}() [cudaMemcpyDeviceToHost, {} <- {}, stream={}, pid={}]",
            __func__, dst, src, reinterpret_cast<uint64_t>(stream), getpid());
        auto rec =
            std::make_shared<CudaMemcpyAsyncD2H>(dst, src, count, stream);
        getCudaTrace().record(rec);
        getTraceExecutor()->synchronize(getCudaTrace());
        std::memcpy(dst, rec->buffer.data(), count);
    } else if (kind == cudaMemcpyDeviceToDevice) {
        spdlog::info(
            "{}() [cudaMemcpyDeviceToDevice, {} <- {}, stream={}, pid={}]",
            __func__, dst, src, reinterpret_cast<uint64_t>(stream), getpid());
        getCudaTrace().record(
            std::make_shared<CudaMemcpyAsyncD2D>(dst, src, count, stream));
    } else {
        EXIT_NOT_IMPLEMENTED("cudaMemcpyKind");
    }
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    hijackInit();
    HIJACK_FN_PROLOGUE();

    auto it = getSymbolMap().find(func);
    if (it == getSymbolMap().end()) {
        EXIT_UNRECOVERABLE("unknown function");
    }
    std::string &symbol = it->second;
    spdlog::info("cudaLaunchKernel({})", cpp_demangle(symbol).c_str());
    // spdlog::debug("")

    std::vector<KParamInfo> paramInfos;
    const auto &analyzer = getCudaTrace().cubinAnalyzer();
    if (!analyzer.kernel_parameters(symbol, paramInfos)) {
        EXIT_UNRECOVERABLE("unable to look up kernel parameter data");
    }

    // debug information
    std::stringstream ss;
    ss << "parameters: [";
    for (const auto &p : paramInfos) {
        std::string type = getPtxParameterTypeToStr()[p.type];
        ss << type << "[" << p.size << "], ";
    }
    ss << "]" << std::endl;
    spdlog::debug(ss.str());

    std::vector<std::vector<uint8_t>> paramBuffers(paramInfos.size());
    for (unsigned i = 0; i < paramInfos.size(); i++) {
        const auto &p = paramInfos[i];
        paramBuffers[i].resize(p.size * p.typeSize);
        std::memcpy(paramBuffers[i].data(), args[i], p.size * p.typeSize);
    }

    auto &cuda_trace = getCudaTrace();
    auto &vdev = cuda_trace.cudaVirtualDevice();
    auto mod_id_it = vdev.symbol_to_module_id_map.find(symbol);
    if (mod_id_it == vdev.symbol_to_module_id_map.end()) {
        spdlog::error("function in unknown module");
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint64_t> required_cuda_modules{mod_id_it->second};
    std::vector<std::string> required_function_symbols{symbol};

    getCudaTrace().record(std::make_shared<CudaLaunchKernel>(
        symbol, required_cuda_modules, required_function_symbols, func, gridDim,
        blockDim, sharedMem, stream, paramBuffers, paramInfos));
    getTraceExecutor()->synchronize(getCudaTrace());
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaFree>(devPtr));
    // have to synchronize here until I find a way to hook the cuda exit handler
    getTraceExecutor()->synchronize(getCudaTrace());
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    getCudaTrace().record(std::make_shared<CudaStreamSynchronize>(stream));
    return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    // TODO
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    EXIT_NOT_IMPLEMENTED(__func__);
    // static auto real_func =
    //     (decltype(&cudaStreamCreateWithFlags))real_dlsym(RTLD_NEXT,
    //     __func__);
    // return real_func(pStream, flags);
}

cudaError_t
cudaStreamIsCapturing(cudaStream_t stream,
                      enum cudaStreamCaptureStatus *pCaptureStatus) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    *pCaptureStatus = cudaStreamCaptureStatusNone;
    //    *pCaptureStatus = cudaStreamCaptureStatusActive;
    //    getCudaTrace().record(std::make_shared<CudaStreamIsCapturing>(stream));
    //    getCudaTrace().synchronize();
    //    auto t = std::static_pointer_cast<CudaStreamIsCapturing>(
    //        getCudaTrace().historyTop());
    //    *pCaptureStatus = t->cudaStreamCaptureStatus;
    return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem = 0,
                                     struct CUstream_st *stream = 0) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    getCudaCallConfigStack().push({gridDim, blockDim, sharedMem, stream});
    return cudaSuccess;
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    CudaCallConfig config = getCudaCallConfigStack().top();
    getCudaCallConfigStack().pop();
    *gridDim = config.gridDim;
    *blockDim = config.blockDim;
    *sharedMem = config.sharedMem;
    *((CUstream_st **)stream) = config.stream;
    return cudaSuccess;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
    hijackInit();
    HIJACK_FN_PROLOGUE();

    auto &state = getCudaRegisterState();
    auto &vdev = getCudaTrace().cudaVirtualDevice();

    uint64_t fatbin_id = incrementFatbinCount();
    state.is_registering = true;
    state.current_fatbin_handle = fatbin_id;

    vdev.module_id_to_fatbin_wrapper_map.emplace(
        fatbin_id, *static_cast<__fatBinC_Wrapper_t *>(fatCubin));

    if (debug_register_func) {
        static auto real =
            (decltype(&__cudaRegisterFatBinary))real_dlsym(RTLD_NEXT, __func__);
        return real(fatCubin);
    } else {
        getCudaTrace().record(
            std::make_shared<PrivCudaRegisterFatBinary>(fatbin_id));
        return reinterpret_cast<void **>(fatbin_id);
    }
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    hijackInit();
    HIJACK_FN_PROLOGUE();

    auto &state = getCudaRegisterState();
    if (!state.is_registering) {
        EXIT_UNRECOVERABLE("__cudaRegisterFatBinaryEnd called without a "
                           "previous call to __cudaRegisterFatBinary");
    }
    state.is_registering = false;

    if (debug_register_func) {
        static auto real = (decltype(&__cudaRegisterFatBinaryEnd))real_dlsym(
            RTLD_NEXT, __func__);
        real(fatCubinHandle);
    } else {
        getCudaTrace().record(std::make_shared<PrivCudaRegisterFatBinaryEnd>(
            state.current_fatbin_handle));
    }
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
    hijackInit();
    spdlog::debug("{}({})", __func__, cpp_demangle(deviceName).c_str());

    auto &state = getCudaRegisterState();
    auto &vdev = getCudaTrace().cudaVirtualDevice();
    if (!state.is_registering) {
        EXIT_UNRECOVERABLE("__cudaRegisterFunction called without a "
                           "previous call to __cudaRegisterFatBinary");
    }

    vdev.symbol_to_module_id_map.emplace(deviceName,
                                         state.current_fatbin_handle);

    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(hostFun), deviceName));
    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(deviceFun), deviceName));

    // execute locally, not by trace recording for debug purposes
    if (debug_register_func) {
        static auto real =
            (decltype(&__cudaRegisterFunction))real_dlsym(RTLD_NEXT, __func__);
        real(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid,
             bid, bDim, gDim, wSize);
    } else {
        getCudaTrace().record(std::make_shared<PrivCudaRegisterFunction>(
            reinterpret_cast<uint64_t>(fatCubinHandle),
            const_cast<void *>(reinterpret_cast<const void *>(hostFun)),
            deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize));
    }
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
    hijackInit();
    HIJACK_FN_PROLOGUE();

    auto &state = getCudaRegisterState();
    auto &vdev = getCudaTrace().cudaVirtualDevice();
    if (!state.is_registering) {
        EXIT_UNRECOVERABLE("__cudaRegisterVar called without a "
                           "previous call to __cudaRegisterFatBinary");
    }

    vdev.global_var_to_module_id_map.emplace(deviceName,
                                             state.current_fatbin_handle);

    std::vector<uint64_t> required_cuda_modules{state.current_fatbin_handle};

    if (debug_register_func) {
        static auto real =
            (decltype(&__cudaRegisterVar))real_dlsym(RTLD_NEXT, __func__);
        real(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
             constant, global);
    } else {
        getCudaTrace().record(std::make_shared<PrivCudaRegisterVar>(
            required_cuda_modules, state.current_fatbin_handle, hostVar,
            deviceAddress, deviceName, ext, size, constant, global));
    }
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    hijackInit();
    // HIJACK_FN_PROLOGUE();
    (void)fatCubinHandle;
    // TODO
}

/*
 * CUDA driver API
 */

CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    (void)dev;
    getTraceExecutor()->synchronize(getCudaTrace());
    return CUDA_SUCCESS;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
    hijackInit();

    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuDevicePrimaryCtxRelease_v2);

//    if (strncmp(symbol, "cu", 2) == 0) {
//        spdlog::debug("cuGetProcAddress({}): symbol not implemented", symbol);
//        *pfn = nullptr;
//    }

//    return CUDA_SUCCESS;

    static auto real =
        (decltype(&cuGetProcAddress))real_dlsym(RTLD_NEXT, __func__);
    return real(symbol, pfn, cudaVersion, flags);
}

void *dlsym(void *handle, const char *symbol) {
    spdlog::debug("{}({}) [pid={}]", __func__, symbol, getpid());

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
