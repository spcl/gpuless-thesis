#ifndef __GPULESS_HPP__
#define __GPULESS_HPP__

#include <iostream>
#include <vector>
#include <any>

#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult r, const char *file, const int line) {
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(r, &msg);
        std::cout << "cuda error in " << file << "(" << line << "):"
            << std::endl << msg << std::endl;
    }
}

namespace gpuless {

class local_buffer {
public:
    // T *device = nullptr;
    CUdeviceptr device;
    void *host = nullptr;
    size_t size;

    bool copy_to_device;
    bool copy_to_host;

    local_buffer(size_t size, bool copy_to_device, bool copy_to_host)
        : size(size), copy_to_device(copy_to_device), copy_to_host(copy_to_host) {
        this->host = malloc(this->size);
        checkCudaErrors(cuMemAlloc(&this->device, this->size));
    }
};

class local_executor {
private:
    CUdevice device;
    CUcontext context;
    CUmodule module;

    std::vector<local_buffer> buffers;

public:
    local_executor() {};
    ~local_executor() {
        cuCtxDestroy(this->context);
    };

    bool allocate(const char *fname) {
        checkCudaErrors(cuInit(0));
        checkCudaErrors(cuDeviceGet(&this->device, 0));
        // char name[256];
        // checkCudaErrors(cuDeviceGetName(name, 256, device));
        // std::cout << "Using device: " << name << std::endl;
        checkCudaErrors(cuCtxCreate(&this->context, 0, device));
        checkCudaErrors(cuModuleLoad(&this->module, fname));
        return true;
    }

    bool register_buffer(local_buffer buffer) {
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

#endif // __GPULESS_HPP__
