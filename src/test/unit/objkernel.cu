#include <cstdio>
#include <cstdlib>
#include <cassert>

#define N 4096

struct X {
    float fs[16];
    float x1;
    float x2;
    float x3;
};

__global__ void vadd_obj(const float *a, const float *b, float *c, int n, X x) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i] + x.x3;
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
    X x{{0.0}, 100.0, 200.0, 300.0};
    vadd_obj<<<N, 1>>>(d_a, d_b, d_c, N, x);
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    assert(abs(h_c[0] - 300.0) < 1e-6);
    assert(abs(h_c[1] - 302.0) < 1e-6);
    assert(abs(h_c[2] - 304.0) < 1e-6);
    assert(abs(h_c[3] - 306.0) < 1e-6);
    assert(abs(h_c[4] - 308.0) < 1e-6);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
