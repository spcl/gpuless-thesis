#include <cstdio>
#include <cstdlib>
#include <cassert>

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
    cudaMallocAsync(&d_a, N * sizeof(float), 0);
    cudaMallocAsync(&d_b, N * sizeof(float), 0);
    cudaMallocAsync(&d_c, N * sizeof(float), 0);

    cudaMemcpyAsync(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice, 0);
    vadd<<<N, 1>>>(d_a, d_b, d_c, N);
    cudaMemcpyAsync(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost, 0);

    assert(abs(h_c[0] - 0.0) < 1e-6);
    assert(abs(h_c[1] - 2.0) < 1e-6);
    assert(abs(h_c[2] - 4.0) < 1e-6);
    assert(abs(h_c[3] - 6.0) < 1e-6);
    assert(abs(h_c[4] - 8.0) < 1e-6);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
