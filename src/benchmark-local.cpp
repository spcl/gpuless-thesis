#include <chrono>
#include <cassert>
#include <iostream>

#include "executor_local.hpp"

void bench_rtt(size_t size, int n_runs) {
    for (int i = 0; i < n_runs; i++) {
        gpuless::executor_local exec;
        exec.allocate("../kernels/build/common.fatbin");
        gpuless::buffer_local b(sizeof(float) * size, true, true);
        exec.register_buffer(&b);

        ((float *) b.host)[0] = 0.1;
        ((float *) b.host)[1] = 0.2;
        ((float *) b.host)[2] = 0.3;
        ((float *) b.host)[size-1] = 0.1;
        ((float *) b.host)[size-2] = 0.2;
        ((float *) b.host)[size-3] = 0.3;

        auto s = std::chrono::high_resolution_clock::now();
        exec.execute("noop_arg", dim3(1), dim3(1), b);
        auto e = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000.0;
        std::cout << d << std::endl;

        assert(((float *) b.host)[0] - 0.1 < 1e-6);
        assert(((float *) b.host)[1] - 0.2 < 1e-6);
        assert(((float *) b.host)[2] - 0.3 < 1e-6);
        assert(((float *) b.host)[size-1] - 0.1 < 1e-6);
        assert(((float *) b.host)[size-2] - 0.2 < 1e-6);
        assert(((float *) b.host)[size-3] - 0.3 < 1e-6);
    }
}

int main(int argc, const char **argv) {
    if (argc != 3) {
        std::cout << "wrong number of arguments" << std::endl;
        return 1;
    }

    int n_runs = atoi(argv[1]);
    size_t size = atol(argv[2]);
    bench_rtt(size, n_runs);
    return 0;
}
