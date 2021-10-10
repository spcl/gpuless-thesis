#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define N 4096

__global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 2 * N * sizeof(float));
    cudaMalloc(&d_b, 2 * N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a + N, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b + N, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    //    const int n = N;
    //    void *args[] = {&d_a, &d_b, &d_c, (void *)&n};
    //    cudaLaunchKernel((const void *)vadd, dim3(N), dim3(1), args, 0, 0);
    vadd<<<N, 1>>>(d_a + N, d_b + N, d_c, N);
    vadd<<<N, 1>>>(d_c, d_a + N, d_c, N);
    vadd<<<N, 1>>>(d_c, d_a + N, d_c, N);
    vadd<<<N, 1>>>(d_c, d_a + N, d_c, N);

    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\n", h_c[0]);
    printf("%f\n", h_c[1]);
    printf("%f\n", h_c[2]);
    printf("%f\n", h_c[3]);
    printf("%f\n", h_c[4]);

    assert(abs(h_c[0] - 0.0) < 1e-6);
    assert(abs(h_c[1] - 5.0) < 1e-6);
    assert(abs(h_c[2] - 10.0) < 1e-6);
    assert(abs(h_c[3] - 15.0) < 1e-6);
    assert(abs(h_c[4] - 20.0) < 1e-6);

    return 0;
}
