#include "libcuadapter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstring>
#include <iostream>
#include <map>

#include "../executors/executor_tcp.hpp"
#include "../utils.hpp"
#include "cubin_analysis.hpp"
#include "cuda_vdev.hpp"

static bool vdev_initialized = false;
static bool debug_print = false;

extern "C" {

static void *real_dlsym(void *handle, const char *symbol) {
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(
        __libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

void vdev_remote_init() {
    CudaVDev &vdev = CudaVDev::getInstance();
    if (!vdev.executor.init(vdev.manager_ip, vdev.manager_port,
                   gpuless::manager::instance_profile::NO_MIG)) {
        dbgprintf("failed to initialize remote device\n");
    }
}

void vdev_initialize() {
    CudaVDev &vdev = CudaVDev::getInstance();

    // load environment variables
    char *cuda_binary = std::getenv("CUDA_BINARY");
    if (cuda_binary == nullptr) {
        std::cerr << "please set CUDA_BINARY environment variable" << std::endl;
    }

    char *manager_ip = std::getenv("MANAGER_IP");
    if (manager_ip == nullptr) {
        std::cerr << "please set MANAGER_IP environment variable" << std::endl;
    }

    char *manager_port_str = std::getenv("MANAGER_PORT");
    if (manager_port_str == nullptr) {
        std::cerr << "please set MANAGER_PORT environment variable"
                  << std::endl;
    }
    short manager_port = std::stoi(manager_port_str);

    char *debug_print_str = std::getenv("DEBUG");
    if (debug_print_str != nullptr) {
        debug_print = true;
    }

    std::vector<std::string> cuda_binaries;
    string_split(std::string(cuda_binary), ',', cuda_binaries);
    vdev.initialize(cuda_binaries, manager_ip, manager_port);
    vdev_remote_init();

    // load module code
    int compute_capability_major;
    int compute_capability_minor;
    vdev.executor.get_device_attribute(
        &compute_capability_major,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    vdev.executor.get_device_attribute(
        &compute_capability_minor,
        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);

    vdev.cubin_analyzer.analyze(cuda_binaries, compute_capability_major,
                                compute_capability_minor);

    // auto binary_versions = cubin_analyzer.arch_modules();
    // bool suitable_code_found = false;
    // for (const auto &v : binary_versions) {
    //     int major_minor = v.first;
    //     int minor = major_minor % 10;
    //     int major = major_minor / 10;
    //     if (major == compute_capability_major &&
    //         minor <= compute_capability_minor) {
    //         exec.set_cuda_code(v.second.data(), v.second.size());
    //         suitable_code_found = true;
    //         break;
    //     }
    // }

    // if (!suitable_code_found) {
    //     std::cerr << "no suitable code module found in CUDA_BINARY"
    //               << std::endl;
    // }

    vdev_initialized = true;
}

CUfunction vdev_get_function(const char *name) {
    CudaVDev &vdev = CudaVDev::getInstance();
    CUfunction f = vdev.next_cufunction_ptr();
    vdev.function_name.emplace(std::make_pair(f, name));
    return f;
}

// CUdeviceptr vdev_alloc_buf(size_t size) {
//     CudaVDev &vdev = CudaVDev::getInstance();
//     CUdeviceptr p = vdev.next_cudevice_ptr();
//     vdev.device_buffers.emplace(std::make_pair(p, std::vector<uint8_t>(size)));
//     return p;
// }

void vdev_free_buf(CUdeviceptr ptr) {
    CudaVDev &vdev = CudaVDev::getInstance();
    auto it = vdev.device_buffers.find(ptr);
    if (it != vdev.device_buffers.end()) {
        vdev.device_buffers.erase(it);
    }
}

void vdev_memcpy_htod(CUdeviceptr dst, const void *src, size_t size) {
    CudaVDev &vdev = CudaVDev::getInstance();
    auto it = vdev.device_buffers.find(dst);
    if (it != vdev.device_buffers.end() && it->second.size() >= size) {
        void *dst_raw = (void *)it->second.data();
        memcpy(dst_raw, src, size);
    }
}

void vdev_memcpy_dtoh(void *dst, CUdeviceptr src, size_t size) {
    CudaVDev &vdev = CudaVDev::getInstance();
    auto it = vdev.device_buffers.find(src);
    if (it != vdev.device_buffers.end()) {
        void *src_raw = (void *)it->second.data();
        memcpy(dst, src_raw, size);
    }
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
    HIJACK_FN_PROLOGUE();
    (void)Flags;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    HIJACK_FN_PROLOGUE();
    (void)device;
    (void)ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
    HIJACK_FN_PROLOGUE();
    (void)dev;
    CudaVDev &vdev = CudaVDev::getInstance();
    vdev.executor.get_device_attribute(pi, attrib);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev) {
    HIJACK_FN_PROLOGUE();
    (void)dev;
    CudaVDev &vdev = CudaVDev::getInstance();
    vdev.executor.deallocate();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetCount(int *count) {
    HIJACK_FN_PROLOGUE();
    *count = 1;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    HIJACK_FN_PROLOGUE();
    (void)dev;
    const char n[] = "libcuadapter";
    if ((size_t)len >= sizeof(n)) {
        size_t l = sizeof(n);
        strncpy(name, n, l);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                CUdevice dev) {
    HIJACK_FN_PROLOGUE();
    (void)pctx;
    (void)flags;
    (void)dev;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    HIJACK_FN_PROLOGUE();
    (void)ctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    HIJACK_FN_PROLOGUE();
    (void)pctx;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuGetExportTable(const void **ppExportTable,
                                  const CUuuid *pExportTableId) {
    HIJACK_FN_PROLOGUE();
    (void)ppExportTable;
    (void)pExportTableId;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    HIJACK_FN_PROLOGUE();
    (void)module;
    CudaVDev &vdev = CudaVDev::getInstance();
    if (!vdev.executor.set_cuda_code_file(fname)) {
        dbgprintf("failed to load ptx file: %s\n", fname);
    }
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    HIJACK_FN_PROLOGUE();
    (void)hmod;
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    HIJACK_FN_PROLOGUE();
    (void)module;
    (void)image;

    return CUDA_SUCCESS;
}

// TODO: support more than one module
CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                     const char *name) {
    HIJACK_FN_PROLOGUE();
    (void)hmod;
    *hfunc = vdev_get_function(name);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    HIJACK_FN_PROLOGUE();
    *dptr = (CUdeviceptr) CudaVDev::getInstance().memAlloc(bytesize);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
    HIJACK_FN_PROLOGUE();
    CudaVDev::getInstance().memCpyHtoD(dstDevice, (void *)srcHost, ByteCount);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
    HIJACK_FN_PROLOGUE();
    // vdev_memcpy_dtoh(dstHost, srcDevice, ByteCount);
    CudaVDev::getInstance().memCpyDtoH((void *)dstHost, srcDevice, ByteCount);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                                unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void **kernelParams, void **extra) {
    HIJACK_FN_PROLOGUE();
    (void)hStream;
    (void)extra;
    (void)sharedMemBytes;
    CudaVDev &vdev = CudaVDev::getInstance();

    std::string kernel = vdev.function_name.find(f)->second;
    dim3 dimGrid(gridDimX, gridDimY, gridDimZ);
    dim3 dimBlock(blockDimX, blockDimY, blockDimZ);

    // TODO: support arguments passed in **extra
    if (kernelParams != nullptr) {
        // look up function name
        auto fname_it = vdev.function_name.find(f);
        if (fname_it == vdev.function_name.end()) {
            std::cerr << "unknown function" << std::endl;
            return CUDA_SUCCESS;
        }
        std::string fname = fname_it->second;
        dbgprintf("kernel: %s\n", fname.c_str());

        // load the module code
        std::vector<uint8_t> module_data;
        if (!vdev.cubin_analyzer.kernel_module(fname, module_data)) {
            std::cerr << "failed to load module" << std::endl;
            return CUDA_SUCCESS;
        }
        vdev.executor.set_cuda_code(module_data.data(), module_data.size());

        // get the number and sizes of parameters from the analysis
        std::vector<KParamInfo> params;
        if (!vdev.cubin_analyzer.kernel_parameters(kernel, params)) {
            std::cerr << "failed to look up kernel parameters" << std::endl;
            return CUDA_SUCCESS;
        }

        // build kernel argument vector for remote execution
        std::vector<kernel_argument> args(params.size());
        for (unsigned i = 0; i < params.size(); i++) {
            KParamInfo p = params[i];
            void *input_param = kernelParams[i];

            // check if parameter is an allocated device pointer
            if (p.size == sizeof(CUdeviceptr)) {
                auto device_ptr = *((CUdeviceptr *)input_param);
                auto mem = vdev.buffer_from_ptr(device_ptr);
                if (mem != nullptr) {
                    std::string id = "buf" + i;
                    args[i] = kernel_argument(id,
                                              KERNEL_ARG_POINTER |
                                                  KERNEL_ARG_COPY_TO_DEVICE |
                                                  KERNEL_ARG_COPY_TO_HOST,
                                              *mem);

                }
            } else { // otherwise it has to be a value
                std::vector<uint8_t> value_data(p.size);
                memcpy(value_data.data(), input_param, p.size);
                std::string id = "buf" + i;
                args[i] = kernel_argument(id, KERNEL_ARG_VALUE, value_data);
            }
        }

        if (!vdev.executor.execute(kernel.c_str(), dimGrid, dimBlock, args)) {
            std::cerr << "execution of " << kernel << " failed" << std::endl;
        }

        // store updated memory back to virtual device buffers
        for (unsigned i = 0; i < params.size(); i++) {
            KParamInfo p = params[i];
            void *input_param = kernelParams[i];

            // check if parameter is an allocated device pointer
            if (p.size == sizeof(CUdeviceptr)) {
                auto device_ptr = *((CUdeviceptr *)input_param);
                auto mem = vdev.buffer_from_ptr(device_ptr);
                if (mem != nullptr) {
                    std::string id = "buf" + i;
                    for (const auto &a : args) {
                        if (a.id == id) {
                            mem->insert(mem->begin(), a.buffer.begin(),
                                        a.buffer.end());
                        }
                    }
                }
            }
        }
    }

    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxSynchronize(void) {
    HIJACK_FN_PROLOGUE();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy_v2(CUcontext ctx) {
    HIJACK_FN_PROLOGUE();
    (void)ctx;
    CudaVDev::getInstance().executor.deallocate();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemFree_v2(CUdeviceptr dptr) {
    HIJACK_FN_PROLOGUE();
    (void)dptr;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    HIJACK_FN_PROLOGUE();
    *driverVersion = 1140;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cu_default_function() {
    HIJACK_FN_PROLOGUE();
    return CUDA_SUCCESS;
}

cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    HIJACK_FN_PROLOGUE();
    cuMemAlloc_v2((CUdeviceptr *)devPtr, size);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    HIJACK_FN_PROLOGUE();
    (void)devPtr;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind) {
    HIJACK_FN_PROLOGUE();
    if (kind == cudaMemcpyHostToDevice) {
        cuMemcpyHtoD_v2((CUdeviceptr)dst, src, count);
    } else if (kind == cudaMemcpyDeviceToHost) {
        cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, count);
    } else {
        std::cerr << "cudaMemcpyKind not implemented" << std::endl;
    }
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                      enum cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    (void)stream;
    cudaMemcpy(dst, src, count, kind);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim,
                                       dim3 blockDim, void **args,
                                       size_t sharedMem, cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    dbgprintf("grid(%ld,%ld,%ld), block(%ld,%ld,%ld)\n", gridDim.x, gridDim.y,
              gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    CudaVDev &vdev = CudaVDev::getInstance();

    auto it = vdev.fn_ptr_to_cufunction.find((void *)func);
    if (it == vdev.fn_ptr_to_cufunction.end()) {
        dbgprintf("no funtion at %p registered\n", func);
        return cudaSuccess;
    }

    cuLaunchKernel(it->second, gridDim.x, gridDim.y, gridDim.z, blockDim.x,
                   blockDim.y, blockDim.z, sharedMem, stream, args, nullptr);
    return cudaSuccess;
}

unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                               size_t sharedMem = 0,
                                               struct CUstream_st *stream = 0) {
    HIJACK_FN_PROLOGUE();
    cuda_call_config_stack.push({gridDim, blockDim, sharedMem, stream});
    return cudaSuccess;
}

cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                 size_t *sharedMem,
                                                 void *stream) {
    HIJACK_FN_PROLOGUE();
    CudaCallConfig config = cuda_call_config_stack.top();
    cuda_call_config_stack.pop();
    *gridDim = config.gridDim;
    *blockDim = config.blockDim;
    *sharedMem = config.sharedMem;
    *((CUstream_st **)stream) = config.stream;
    return cudaSuccess;
}

void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin) {
    HIJACK_FN_PROLOGUE();
    (void)fatCubin;
    return nullptr;
}

void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    HIJACK_FN_PROLOGUE();
    (void)fatCubinHandle;
}

void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle) {
    HIJACK_FN_PROLOGUE();
    (void)fatCubinHandle;
    CudaVDev::getInstance().executor.deallocate(); // TODO: call this at a more suitable place
}

cudaError_t CUDARTAPI cudaGetLastError(void) {
    HIJACK_FN_PROLOGUE();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    HIJACK_FN_PROLOGUE();
    *device = 0;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device) {
    (void) device;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    HIJACK_FN_PROLOGUE();
    *count = 1;
    return cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(void **fatCubinHandle,
                                      const char *hostFun, char *deviceFun,
                                      const char *deviceName, int thread_limit,
                                      uint3 *tid, uint3 *bid, dim3 *bDim,
                                      dim3 *gDim, int *wSize) {
    HIJACK_FN_PROLOGUE();
    (void)fatCubinHandle;
    (void)thread_limit;
    (void)tid;
    (void)bid;
    (void)bDim;
    (void)gDim;
    (void)wSize;
    CUfunction f = vdev_get_function(deviceName);
    std::cout << deviceName << std::endl;
    CudaVDev &vdev = CudaVDev::getInstance();
    vdev.fn_ptr_to_cufunction.emplace(std::make_pair((void *)deviceFun, f));
    vdev.fn_ptr_to_cufunction.emplace(std::make_pair((void *)hostFun, f));
}

cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    HIJACK_FN_PROLOGUE();
    (void) device;
    CudaVDev &vdev = CudaVDev::getInstance();
    *prop = vdev.getDeviceProperties();
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamIsCapturing(
    cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
    HIJACK_FN_PROLOGUE();
    (void)stream;
    *pCaptureStatus = cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
    // *pCaptureStatus = cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    HIJACK_FN_PROLOGUE();
    (void) stream;
    return cudaSuccess;
}


cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode) {
    HIJACK_FN_PROLOGUE();
    (void) mode;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(
    cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus,
    unsigned long long *pId) {
    (void) stream;
    *pCaptureStatus = cudaStreamCaptureStatusNone;
    *pId = 1;
    return cudaSuccess;
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn,
                                  int cudaVersion, cuuint64_t flags) {
    static fnCuGetProcAddress real_func =
        (fnCuGetProcAddress)real_dlsym(RTLD_NEXT, __func__);
    dbgprintf("%s(%s)\n", __func__, symbol);

    LINK_CU_FUNCTION(symbol, cuGetProcAddress);
    LINK_CU_FUNCTION(symbol, cuInit);
    LINK_CU_FUNCTION(symbol, cuDeviceGet);
    LINK_CU_FUNCTION(symbol, cuDeviceGetCount);
    LINK_CU_FUNCTION(symbol, cuDeviceGetName);
    LINK_CU_FUNCTION(symbol, cuDeviceGetAttribute);
    LINK_CU_FUNCTION(symbol, cuDevicePrimaryCtxRelease_v2);
    LINK_CU_FUNCTION(symbol, cuDriverGetVersion);
    LINK_CU_FUNCTION(symbol, cuCtxCreate_v2);
    LINK_CU_FUNCTION(symbol, cuCtxSetCurrent);
    LINK_CU_FUNCTION(symbol, cuCtxGetCurrent);
    LINK_CU_FUNCTION(symbol, cuGetExportTable);
    LINK_CU_FUNCTION(symbol, cuModuleLoad);
    LINK_CU_FUNCTION(symbol, cuModuleUnload);
    LINK_CU_FUNCTION(symbol, cuModuleLoadData);
    LINK_CU_FUNCTION(symbol, cuModuleGetFunction);
    LINK_CU_FUNCTION(symbol, cuMemAlloc_v2);
    LINK_CU_FUNCTION(symbol, cuMemcpyHtoD_v2);
    LINK_CU_FUNCTION(symbol, cuMemcpyDtoH_v2);
    LINK_CU_FUNCTION(symbol, cuLaunchKernel);
    LINK_CU_FUNCTION(symbol, cuCtxSynchronize);
    LINK_CU_FUNCTION(symbol, cuCtxDestroy_v2);
    LINK_CU_FUNCTION(symbol, cuMemFree_v2);

    if (strncmp(symbol, "cu", 2) == 0) {
        dbgprintf("cuGetProcAddress(%s): symbol not implemented\n", symbol);
        *pfn = (void *)&cu_default_function;
        return CUDA_SUCCESS;
    }

    return real_func(symbol, pfn, cudaVersion, flags);
}

void *dlsym(void *handle, const char *symbol) {
    dbgprintf("dlsym(%s)\n", symbol);

    // early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    // cuGetProcAddress is looked up by CUDA runtime binaries
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        return (void *)&cuGetProcAddress;
    }

    return (real_dlsym(handle, symbol));
}

} // extern "C"
