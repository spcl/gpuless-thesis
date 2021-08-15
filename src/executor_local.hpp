#ifndef __EXECUTOR_LOCAL_HPP__
#define __EXECUTOR_LOCAL_HPP__

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "utils.hpp"

namespace gpuless {

class buffer_local {
public:
    CUdeviceptr device;
    void *host = nullptr;
    size_t size;

    bool copy_to_device;
    bool copy_to_host;

    buffer_local(size_t size, bool copy_to_device, bool copy_to_host)
        : size(size), copy_to_device(copy_to_device), copy_to_host(copy_to_host) {
        this->host = malloc(this->size);
        checkCudaErrors(cuMemAlloc(&this->device, this->size));
    }
};

class executor_local {
private:
    CUdevice device;
    CUcontext context;
    CUmodule module;

    std::vector<buffer_local> buffers;

public:
    executor_local() {};
    ~executor_local() {
        cuCtxDestroy(this->context);
    };

    bool allocate(const char *fname) {
        checkCudaErrors(cuInit(0));
        checkCudaErrors(cuDeviceGet(&this->device, 0));
        checkCudaErrors(cuCtxCreate(&this->context, 0, device));
        checkCudaErrors(cuModuleLoad(&this->module, fname));
        return true;
    }

    bool register_buffer(buffer_local buffer) {
        this->buffers.push_back(buffer);
        return true;
    }

    template<typename... Ts>
    bool execute(const char *kernel, dim3 dimGrid, dim3 dimBlock, Ts... args) {
        CUfunction function;
        checkCudaErrors(cuModuleGetFunction(&function, this->module, kernel));

        for (auto& b : buffers) {
            if (b.copy_to_device) {
                cuMemcpyHtoD(b.device, b.host, b.size);
            }
        }

        void *kernel_args[] = { &args... };
        checkCudaErrors(cuLaunchKernel(function,
                dimGrid.x, dimGrid.y, dimGrid.z,
                dimBlock.x, dimBlock.y, dimBlock.z,
                0, 0, kernel_args, 0));

        checkCudaErrors(cuCtxSynchronize());

        for (auto& b : buffers) {
            if (b.copy_to_host) {
                cuMemcpyDtoH(b.host, b.device, b.size);
            }
        }

        return true;
    }
};

} // namespace gpuless

#endif // __EXECUTOR_LCOAL_HPP__

