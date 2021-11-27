#include "cuda_virtual_device.hpp"
#include <spdlog/spdlog.h>

void CudaVirtualDevice::initRealDevice() {
    if (this->initialized) {
        return;
    }

    spdlog::info("CudaVirtualDevice: initializing real device");
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&this->device, 0));
    checkCudaErrors(cuCtxCreate(&this->context, 0, device));
    checkCudaErrors(cuCtxSetCurrent(this->context));
    checkCudaErrors(cuDevicePrimaryCtxRetain(&this->context, this->device));
    this->initialized = true;
}