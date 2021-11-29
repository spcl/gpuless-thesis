#include <chrono>
#include <iostream>

#define N 4096

__global__ void noop() {
}

int main() {
    auto s = std::chrono::high_resolution_clock::now();
    noop<<<N, 1>>>();
    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("%.8f\n", d);
    return 0;
}
