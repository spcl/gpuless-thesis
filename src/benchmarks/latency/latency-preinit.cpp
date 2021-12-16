#include <iostream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

int main() {

    CUdevice device;
    CUcontext context;

    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);
    cuCtxSetCurrent(context);
    cuDevicePrimaryCtxRetain(&context, device);

    float x = 1.0;
    float *d_ptr;

    auto s = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_ptr, sizeof(float));
    cudaMemcpy(d_ptr, &x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&x, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;

    printf("%.10f\n", d);
    return 0;
}
