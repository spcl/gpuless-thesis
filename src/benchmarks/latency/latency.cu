#include <chrono>
#include <iostream>

#define N 4096

int main() {
    auto s = std::chrono::high_resolution_clock::now();
    float x = 1.0;
    float *d_ptr;
    cudaMalloc(&d_ptr, sizeof(float));
    cudaMemcpy(d_ptr, &x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&x, d_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("%.8f\n", d);
    return 0;
}
