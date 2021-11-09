#include <cstdio>
#include <cstdlib>
#include <cassert>

#define N 4096

__device__ float dev_const = 420.0;

__global__ void vadd(const float *a, const float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i] + dev_const;
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
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    vadd<<<N, 1>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    assert(abs(h_c[0] - (420.0 + 0.0)) < 1e-6);
    assert(abs(h_c[1] - (420.0 + 2.0)) < 1e-6);
    assert(abs(h_c[2] - (420.0 + 4.0)) < 1e-6);
    assert(abs(h_c[3] - (420.0 + 6.0)) < 1e-6);
    assert(abs(h_c[4] - (420.0 + 8.0)) < 1e-6);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
