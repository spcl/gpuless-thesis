#include "iostream"
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include "cuda_api_calls.hpp"
#include "dlsym_util.hpp"

/*
 * cudaMalloc
 */
gpuless::CudaMalloc::CudaMalloc(size_t size) : devPtr(nullptr), size(size) {}

cudaError_t gpuless::CudaMalloc::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMalloc<void>))real_dlsym(RTLD_NEXT, "cudaMalloc");
    return real(&this->devPtr, size);
}

/*
 * cudaMemcpyH2D
 */
gpuless::CudaMemcpyH2D::CudaMemcpyH2D(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size), buffer(size) {
    std::memcpy(this->buffer.data(), src, size);
}

cudaError_t gpuless::CudaMemcpyH2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return real(this->dst, this->buffer.data(), this->size,
                cudaMemcpyHostToDevice);
}

/*
 * cudaMemcpyD2H
 */
gpuless::CudaMemcpyD2H::CudaMemcpyD2H(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size), buffer(size) {}

cudaError_t gpuless::CudaMemcpyD2H::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return real(this->buffer.data(), this->src, this->size,
                cudaMemcpyDeviceToHost);
}

/*
 * cudaMemcpyD2D
 */
gpuless::CudaMemcpyD2D::CudaMemcpyD2D(void *dst, const void *src, size_t size)
    : dst(dst), src(src), size(size) {}

cudaError_t gpuless::CudaMemcpyD2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return real(this->dst, this->src, this->size, cudaMemcpyDeviceToDevice);
}

/*
 * cudaMemcpyAsyncH2D
 */
gpuless::CudaMemcpyAsyncH2D::CudaMemcpyAsyncH2D(void *dst, const void *src,
                                                size_t size,
                                                cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream), buffer(size) {
    std::memcpy(this->buffer.data(), src, size);
}

cudaError_t
gpuless::CudaMemcpyAsyncH2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->buffer.data(), this->size,
                cudaMemcpyHostToDevice, this->stream);
}

/*
 * cudaMemcpyAsyncD2H
 */
gpuless::CudaMemcpyAsyncD2H::CudaMemcpyAsyncD2H(void *dst, const void *src,
                                                size_t size,
                                                cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream), buffer(size) {}

cudaError_t
gpuless::CudaMemcpyAsyncD2H::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->buffer.data(), this->src, this->size,
                cudaMemcpyDeviceToHost, this->stream);
}

/*
 * cudaMemcpyAsyncD2D
 */
gpuless::CudaMemcpyAsyncD2D::CudaMemcpyAsyncD2D(void *dst, const void *src,
                                                size_t size,
                                                cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream) {}

cudaError_t
gpuless::CudaMemcpyAsyncD2D::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->src, this->size, cudaMemcpyDeviceToDevice,
                this->stream);
}

/*
 * cudaFree
 */
gpuless::CudaFree::CudaFree(void *devPtr) : devPtr(devPtr) {}

cudaError_t gpuless::CudaFree::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudaFree))real_dlsym(RTLD_NEXT, "cudaFree");
    return real(this->devPtr);
}

/*
 * cudaLaunchKernel
 */
gpuless::CudaLaunchKernel::CudaLaunchKernel(
    std::string symbol, std::vector<uint64_t> required_cuda_modules,
    std::vector<std::string> required_function_symbols, const void *fnPtr,
    const dim3 &gridDim, const dim3 &blockDim, size_t sharedMem,
    cudaStream_t stream, std::vector<std::vector<uint8_t>> &paramBuffers,
    std::vector<KParamInfo> &paramInfos)
    : symbol(symbol), required_cuda_modules_(required_cuda_modules),
      required_function_symbols_(required_function_symbols), fnPtr(fnPtr),
      gridDim(gridDim), blockDim(blockDim), sharedMem(sharedMem),
      stream(stream), paramBuffers(paramBuffers), paramInfos(paramInfos) {}

cudaError_t gpuless::CudaLaunchKernel::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cuLaunchKernel))real_dlsym(RTLD_NEXT, "cuLaunchKernel");

    auto fn_reg_it = vdev.function_registry_.find(this->symbol);
    if (fn_reg_it == vdev.function_registry_.end()) {
        std::cerr << "function not registerd: " << this->symbol << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<void *> args;
    for (unsigned i = 0; i < paramBuffers.size(); i++) {
        auto &b = this->paramBuffers[i];
        args.push_back(b.data());
    }

    auto ret = real(fn_reg_it->second, this->gridDim.x, this->gridDim.y,
                    this->gridDim.z, this->blockDim.x, this->blockDim.y,
                    this->blockDim.z, this->sharedMem, this->stream,
                    args.data(), nullptr);

    if (ret != CUDA_SUCCESS) {
        std::cerr << "cuLaunchKernel() failed" << std::endl;
    }
    return cudaSuccess;
}
std::vector<uint64_t> gpuless::CudaLaunchKernel::requiredCudaModuleIds() {
    return this->required_cuda_modules_;
}

std::vector<std::string> gpuless::CudaLaunchKernel::requiredFunctionSymbols() {
    return this->required_function_symbols_;
}

/*
 * cudaStreamSynchronize
 */
gpuless::CudaStreamSynchronize::CudaStreamSynchronize(cudaStream_t stream)
    : stream(stream) {}

cudaError_t
gpuless::CudaStreamSynchronize::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudaStreamSynchronize))real_dlsym(
        RTLD_NEXT, "cudaStreamSynchronize");
    return real(this->stream);
}

/*
 * cudaStreamIsCapturing
 */
gpuless::CudaStreamIsCapturing::CudaStreamIsCapturing(cudaStream_t stream)
    : stream(stream), cudaStreamCaptureStatus(cudaStreamCaptureStatusNone) {}

cudaError_t
gpuless::CudaStreamIsCapturing::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudaStreamIsCapturing))real_dlsym(
        RTLD_NEXT, "cudaStreamIsCapturing");
    return real(this->stream, &this->cudaStreamCaptureStatus);
}

/*
 * __cudaRegisterFatBinary
 */
gpuless::PrivCudaRegisterFatBinary::PrivCudaRegisterFatBinary(
    uint64_t client_handle_id)
    : client_handle_id(client_handle_id) {}

cudaError_t
gpuless::PrivCudaRegisterFatBinary::executeNative(CudaVirtualDevice &vdev) {
    return cudaSuccess;
}

/*
 * __cudaRegisterFatBinaryEnd
 */
gpuless::PrivCudaRegisterFatBinaryEnd::PrivCudaRegisterFatBinaryEnd(
    uint64_t client_handle_id)
    : client_handle_id(client_handle_id) {}

cudaError_t
gpuless::PrivCudaRegisterFatBinaryEnd::executeNative(CudaVirtualDevice &vdev) {
    return cudaSuccess;
}

/*
 * __cudaRegisterFunction
 */
gpuless::PrivCudaRegisterFunction::PrivCudaRegisterFunction(
    uint64_t client_handle_id, void *hostFnPtr, void *deviceFnPtr,
    const std::string &fnName, int threadLimit, uint3 *tid, uint3 *bid,
    dim3 *blockDim, dim3 *gridDim, int *wSize)
    : client_handle_id(client_handle_id), host_fn_ptr(hostFnPtr),
      device_fn_ptr(deviceFnPtr), fn_name(fnName), thread_limit(threadLimit),
      tid(tid), bid(bid), block_dim(blockDim), grid_dim(gridDim), wSize(wSize) {
}

cudaError_t
gpuless::PrivCudaRegisterFunction::executeNative(CudaVirtualDevice &vdev) {
    return cudaSuccess;
}

/*
 * __cudaRegisterVar
 */
gpuless::PrivCudaRegisterVar::PrivCudaRegisterVar(
    std::vector<uint64_t> required_cuda_modules, uint64_t clientHandleId,
    void *hostVar, void *deviceAddress, std::string deviceName, int ext,
    size_t size, int constant, int global)
    : required_cuda_modules_(std::move(required_cuda_modules)),
      client_handle_id(clientHandleId), host_var(hostVar),
      device_address(deviceAddress), device_name(std::move(deviceName)),
      ext(ext), size(size), constant(constant), global(global) {}

cudaError_t
gpuless::PrivCudaRegisterVar::executeNative(CudaVirtualDevice &vdev) {
    auto globvar_mod_id_it =
        vdev.global_var_to_module_id_map.find(this->device_name);
    if (globvar_mod_id_it == vdev.global_var_to_module_id_map.end()) {
        std::cerr << "global var in unknown module: " << this->device_name
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }

    uint64_t mod_id = globvar_mod_id_it->second;
    auto mod_id_it = vdev.module_registry_.find(globvar_mod_id_it->second);
    if (mod_id_it == vdev.module_registry_.end()) {
        std::cerr << "module not registered: " << mod_id << std::endl;
        std::exit(EXIT_FAILURE);
    }
    CUmodule mod = mod_id_it->second;
    CUdeviceptr device_ptr;
    checkCudaErrors(cuModuleGetGlobal(&device_ptr, &this->size, mod, this->device_name.c_str()));
    return cudaSuccess;
}

std::vector<uint64_t> gpuless::PrivCudaRegisterVar::requiredCudaModuleIds() {
    return this->required_cuda_modules_;
}