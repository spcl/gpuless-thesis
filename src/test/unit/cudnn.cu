#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cudnn.h>
#include <cstdint>

struct T {
    uint64_t a;
    int* b;
};

__global__ void class_kernel(T test, uint64_t* c) {
    *c = test.a + *test.b;
}

int main() {
    int *d_b;
    uint64_t *d_c;

    int b = 2;
    cudaMalloc(&d_b, sizeof(int));
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    T test_c{1UL, d_b};

    cudaMalloc(&d_c, sizeof(uint64_t));
    class_kernel<<<1, 1>>>(test_c, d_c);
    uint64_t result;
    cudaMemcpy(&result, d_c, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("%lu\n", result);

    assert(result == 3);

    return 0;
}
