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

#include "cubin_analysis.hpp"
#include "cuda_trace.hpp"
#include "dlsym_util.hpp"
#include "libgpuless.hpp"
#include "trace_executor_local.hpp"
#include "trace_executor_tcp_client.hpp"

using namespace gpuless;

const int CUDA_MAJOR_VERSION = 8;
const int CUDA_MINOR_VERSION = 0;

short manager_port = 8002;
std::string manager_ip = "127.0.0.1";

static void hijackInit() {
    static bool hijack_initialized = false;
    if (!hijack_initialized) {
        hijack_initialized = true;
        spdlog::debug("hijackInit()");

        // load log level from env variable SPDLOG_LEVEL
        spdlog::cfg::load_env_levels();

        char *manager_port_env = std::getenv("MANAGER_PORT");
        if (manager_port_env) {
            manager_port = std::stoi(manager_port_env);
        }

        char *manager_ip_env = std::getenv("MANAGER_IP");
        if (manager_ip_env) {
            manager_ip = manager_ip_env;
        }
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

std::shared_ptr<TraceExecutor> getTraceExecutor() {
    static std::shared_ptr<TraceExecutor> trace_executor;
    static bool te_initialized = false;
    if (!te_initialized) {
        spdlog::info("Initializing trace executor");
        te_initialized = true;
        bool useTcp = true;
        char *executor_type = std::getenv("EXECUTOR_TYPE");
        if (executor_type != nullptr) {
            std::string executor_type_str(executor_type);
            if (executor_type_str == "local") {
                useTcp = false;
            } else if (executor_type_str == "tcp") {
                useTcp = true;
            } else {
                useTcp = false;
            }
        }

        if (useTcp) {
            trace_executor = std::make_shared<TraceExecutorTcp>();
            bool r = trace_executor->init(manager_ip.c_str(), manager_port,
                                          manager::instance_profile::NO_MIG);
            if (!r) {
                spdlog::error("Failed to initialize TCP trace executor");
                std::exit(EXIT_FAILURE);
            }
        } else {
            trace_executor = std::make_shared<TraceExecutorLocal>();
        }
    }

    return trace_executor;
}

static CubinAnalyzer &getCubinAnalyzer() {
    static CubinAnalyzer cubin_analyzer;
    return cubin_analyzer;
}

CudaTrace &getCudaTrace() {
    static CudaTrace cuda_trace;

    static bool exit_handler_registered = false;
    if (!exit_handler_registered) {
        exit_handler_registered = true;
        std::atexit([]() { exitHandler(); });
    }

    if (!getCubinAnalyzer().isInitialized()) {
        char *cuda_binary = std::getenv("CUDA_BINARY");
        if (cuda_binary == nullptr) {
            std::cerr << "please set CUDA_BINARY environment variable"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<std::string> binaries;
        string_split(std::string(cuda_binary), ',', binaries);
        spdlog::info("Analyzing CUDA binaries ({})", cuda_binary);
        getCubinAnalyzer().analyze(binaries, CUDA_MAJOR_VERSION,
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
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **devPtr, size_t size) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
    hijackInit();
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        spdlog::info("{}() [cudaMemcpyHostToDevice, {} <- {}, pid={}]",
                     __func__, dst, src, getpid());
        auto rec = std::make_shared<CudaMemcpyH2D>(dst, src, count);
        std::memcpy(rec->buffer.data(), src, count);
        getCudaTrace().record(rec);
    } else if (kind == cudaMemcpyDeviceToHost) {
        spdlog::info("{}() [cudaMemcpyDeviceToHost, {} <- {}, pid={}]",
                     __func__, dst, src, getpid());
        auto rec = std::make_shared<CudaMemcpyD2H>(dst, src, count);
        getCudaTrace().record(rec);
        getTraceExecutor()->synchronize(getCudaTrace());

        std::shared_ptr<CudaMemcpyD2H> top =
            (const std::shared_ptr<CudaMemcpyD2H> &)getCudaTrace().historyTop();
        std::memcpy(dst, top->buffer.data(), count);
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
        auto rec =
            std::make_shared<CudaMemcpyAsyncH2D>(dst, src, count, stream);
        std::memcpy(rec->buffer.data(), src, count);
        getCudaTrace().record(rec);
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
    const auto &analyzer = getCubinAnalyzer();
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
    ss << "]";
    spdlog::debug(ss.str());

    std::vector<std::vector<uint8_t>> paramBuffers(paramInfos.size());
    for (unsigned i = 0; i < paramInfos.size(); i++) {
        const auto &p = paramInfos[i];
        paramBuffers[i].resize(p.size * p.typeSize);
        std::memcpy(paramBuffers[i].data(), args[i], p.size * p.typeSize);
    }

    auto &cuda_trace = getCudaTrace();
    auto &symbol_to_module_id_map = cuda_trace.getSymbolToModuleId();
    auto mod_id_it = symbol_to_module_id_map.find(symbol);
    if (mod_id_it == symbol_to_module_id_map.end()) {
        spdlog::error("function in unknown module");
        std::exit(EXIT_FAILURE);
    }

    std::vector<uint64_t> required_cuda_modules{mod_id_it->second.first};
    std::vector<std::string> required_function_symbols{symbol};

    getCudaTrace().record(std::make_shared<CudaLaunchKernel>(
        symbol, required_cuda_modules, required_function_symbols, func, gridDim,
        blockDim, sharedMem, stream, paramBuffers, paramInfos));
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

    uint64_t fatbin_id = incrementFatbinCount();
    state.is_registering = true;
    state.current_fatbin_handle = fatbin_id;

    auto wrapper = static_cast<__fatBinC_Wrapper_t *>(fatCubin);
    const unsigned long long *data_ull = wrapper->data;

    // this seems to work. no idea why, this reverse engineering result first
    // appears in dscuda (as far as i know)
    size_t data_len = ((data_ull[1] - 1) / 8 + 1) * 8 + 16;

    spdlog::debug("Recording Fatbin data [id={}, size={}]", fatbin_id,
                  data_len);

    void *resource_ptr =
        reinterpret_cast<void *>(const_cast<unsigned long long *>(data_ull));
    getCudaTrace().recordFatbinData(resource_ptr, data_len, fatbin_id);
    return reinterpret_cast<void **>(fatbin_id);
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

    //    getCudaTrace().record(std::make_shared<PrivCudaRegisterFatBinaryEnd>(
    //        state.current_fatbin_handle));
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
    hijackInit();
    spdlog::trace("{}({})", __func__, cpp_demangle(deviceName).c_str());

    auto &state = getCudaRegisterState();
    if (!state.is_registering) {
        EXIT_UNRECOVERABLE("__cudaRegisterFunction called without a "
                           "previous call to __cudaRegisterFatBinary");
    }

    std::string symbol(deviceName);
    getCudaTrace().recordSymbolMapEntry(symbol, state.current_fatbin_handle);

    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(hostFun), deviceName));
    getSymbolMap().emplace(
        std::make_pair(static_cast<const void *>(deviceFun), deviceName));
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
    hijackInit();
    HIJACK_FN_PROLOGUE();

    auto &state = getCudaRegisterState();
    if (!state.is_registering) {
        EXIT_UNRECOVERABLE("__cudaRegisterVar called without a "
                           "previous call to __cudaRegisterFatBinary");
    }

    std::string symbol(deviceName);
    getCudaTrace().recordGlobalVarMapEntry(symbol, state.current_fatbin_handle);

    std::vector<uint64_t> required_cuda_modules{state.current_fatbin_handle};
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
    spdlog::trace("{}({}) [pid={}]", __func__, symbol, getpid());

    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuDevicePrimaryCtxRelease_v2);

    //    if (strncmp(symbol, "cu", 2) == 0) {
    //        spdlog::debug("cuGetProcAddress({}): symbol not implemented",
    //        symbol); *pfn = nullptr;
    //    }

    //    return CUDA_SUCCESS;

    static auto real =
        (decltype(&cuGetProcAddress))real_dlsym(RTLD_NEXT, __func__);
    return real(symbol, pfn, cudaVersion, flags);
}

void *dlsym(void *handle, const char *symbol) {
    spdlog::trace("{}({}) [pid={}]", __func__, symbol, getpid());

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
