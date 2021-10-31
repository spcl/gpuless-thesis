#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include "iostream"

#include "cuda_api_calls.hpp"
#include "dlsym_util.hpp"

/*
 * cudaMalloc
 */
gpuless::CudaMalloc::CudaMalloc(size_t size) : devPtr(nullptr), size(size) {}

cudaError_t gpuless::CudaMalloc::executeNative() {
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

cudaError_t gpuless::CudaMemcpyH2D::executeNative() {
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

cudaError_t gpuless::CudaMemcpyD2H::executeNative() {
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

cudaError_t gpuless::CudaMemcpyD2D::executeNative() {
    static auto real =
        (decltype(&cudaMemcpy))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    return real(this->dst, this->src, this->size,
                cudaMemcpyDeviceToDevice);
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

cudaError_t gpuless::CudaMemcpyAsyncH2D::executeNative() {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->buffer.data(), this->size,
                cudaMemcpyHostToDevice, this->stream);
}

/*
 * cudaMemcpyAsyncD2H
 */
gpuless::CudaMemcpyAsyncD2H::CudaMemcpyAsyncD2H(void *dst, const void *src,
                                                size_t size, cudaStream_t stream)
    : dst(dst), src(src), size(size), stream(stream), buffer(size) {}

cudaError_t gpuless::CudaMemcpyAsyncD2H::executeNative() {
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

cudaError_t gpuless::CudaMemcpyAsyncD2D::executeNative() {
    static auto real =
        (decltype(&cudaMemcpyAsync))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return real(this->dst, this->src, this->size, cudaMemcpyDeviceToDevice,
                this->stream);
}

/*
 * cudaFree
 */
gpuless::CudaFree::CudaFree(void *devPtr) : devPtr(devPtr) {}

cudaError_t gpuless::CudaFree::executeNative() {
    static auto real = (decltype(&cudaFree))real_dlsym(RTLD_NEXT, "cudaFree");
    return real(this->devPtr);
}

/*
 * cudaLaunchKernel
 */
gpuless::CudaLaunchKernel::CudaLaunchKernel(
    const void *fnPtr, const dim3 &gridDim, const dim3 &blockDim,
    size_t sharedMem, cudaStream_t stream,
    std::vector<std::vector<uint8_t>> &paramBuffers,
    std::vector<KParamInfo> &paramInfos)
    : fnPtr(fnPtr), gridDim(gridDim), blockDim(blockDim), sharedMem(sharedMem),
      stream(stream), paramBuffers(paramBuffers), paramInfos(paramInfos) {}

cudaError_t gpuless::CudaLaunchKernel::executeNative() {
    static auto real = (decltype(&cudaLaunchKernel<void>))real_dlsym(
        RTLD_NEXT, "cudaLaunchKernel");
    std::vector<void *> args;
    for (unsigned i = 0; i < paramBuffers.size(); i++) {
        auto &b = this->paramBuffers[i];

        args.push_back(b.data());

        // dbg
//        std::stringstream ss;
//        ss << "  ptx param: ";
//        ss << this->paramInfos[i].type;
//        ss << " 0x";
//        ss << std::hex << std::setfill('0');
//        for (int j = b.size() - 1; j >= 0; j--) {
//            ss << std::setw(2) << static_cast<unsigned>(b[j]);
//        }
//        ss << std::endl;
//        std::cout << ss.str();
    }
    return real(this->fnPtr, this->gridDim, this->blockDim, args.data(),
                this->sharedMem, this->stream);
}

/*
 * cudaStreamSynchronize
 */
gpuless::CudaStreamSynchronize::CudaStreamSynchronize(cudaStream_t stream)
    : stream(stream) {}

cudaError_t gpuless::CudaStreamSynchronize::executeNative() {
    static auto real = (decltype(&cudaStreamSynchronize))real_dlsym(
        RTLD_NEXT, "cudaStreamSynchronize");
    return real(this->stream);
}
