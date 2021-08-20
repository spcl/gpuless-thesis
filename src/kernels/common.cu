extern "C" __global__ void
vadd(const float *a, const float *b, float *c, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void
noop_arg(float *b)
{
}

extern "C" __global__ void
noop()
{
}
