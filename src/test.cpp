#include <cuda.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#define N 1024
#define VADD_FATBIN "../kernels/build/common.fatbin"
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

void __checkCudaErrors(CUresult r, const char *file, const int line) {
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(r, &msg);
        std::cout << "cuda error in " << file << "(" << line
                  << "):" << std::endl
                  << msg << std::endl;
    }
}

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction vadd_f;

    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[256];
    checkCudaErrors(cuDeviceGetName(name, 256, device));
    std::cout << "Using device: " << name << std::endl;

    checkCudaErrors(cuCtxCreate(&context, 0, device));
    checkCudaErrors(cuModuleLoad(&module, VADD_FATBIN));
    checkCudaErrors(cuModuleGetFunction(&vadd_f, module, "vadd"));

    CUdeviceptr d_a, d_b, d_c;
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    checkCudaErrors(cuMemAlloc(&d_a, sizeof(float) * N));
    checkCudaErrors(cuMemAlloc(&d_b, sizeof(float) * N));
    checkCudaErrors(cuMemAlloc(&d_c, sizeof(float) * N));

    checkCudaErrors(cuMemcpyHtoD(d_a, h_a, sizeof(float) * N));
    checkCudaErrors(cuMemcpyHtoD(d_b, h_b, sizeof(float) * N));

    int n = N;
    void *args[4] = {&d_a, &d_b, &d_c, &n};
    checkCudaErrors(cuLaunchKernel(vadd_f, N, 1, 1, 1, 1, 1, 0, 0, args, 0));
    checkCudaErrors(cuCtxSynchronize());
    checkCudaErrors(cuMemcpyDtoH(h_c, d_c, sizeof(float) * N));

    assert(abs(h_c[0] - 1.0) < 1e-6);
    assert(abs(h_c[1] - 3.0) < 1e-6);
    assert(abs(h_c[2] - 5.0) < 1e-6);
    assert(abs(h_c[3] - 7.0) < 1e-6);
    assert(abs(h_c[4] - 9.0) < 1e-6);

    checkCudaErrors(cuMemFree(d_a));
    checkCudaErrors(cuMemFree(d_b));
    checkCudaErrors(cuMemFree(d_c));

    cuCtxDestroy(context);

    return 0;
}
