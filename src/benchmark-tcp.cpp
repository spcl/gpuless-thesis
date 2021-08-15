#include <chrono>
#include <vector>
#include <cassert>
#include <iostream>

#include "executor_tcp.hpp"

const int N = 1024;
const int RUNS = 10;

void test_vadd_tcp() {
        gpuless::executor_tcp exec;


        std::vector<gpuless::buffer_tcp> buffers {
            gpuless::buffer_tcp(sizeof(float) * N), // a
            gpuless::buffer_tcp(sizeof(float) * N), // b
            gpuless::buffer_tcp(sizeof(float) * N), // c
        };

        // allocate memory on remote host
        exec.allocate("127.0.0.1", 8001, "../kernels/build/common.fatbin", buffers);

        for (int i = 0; i < N; i++) {
            ((float *) buffers[0].data)[i] = i;
            ((float *) buffers[1].data)[i] = i+1;
            ((float *) buffers[2].data)[i] = 0;
        }

        std::vector<gpuless::kernel_arg> args {
            gpuless::executor_tcp::pointer_argument(buffers[0],
                    gpuless::KERNEL_ARG_POINTER | gpuless::KERNEL_ARG_COPY_TO_DEVICE),
            gpuless::executor_tcp::pointer_argument(buffers[1],
                    gpuless::KERNEL_ARG_POINTER | gpuless::KERNEL_ARG_COPY_TO_DEVICE),
            gpuless::executor_tcp::pointer_argument(buffers[2],
                    gpuless::KERNEL_ARG_POINTER | gpuless::KERNEL_ARG_COPY_TO_HOST),
            gpuless::executor_tcp::value_argument(&N, gpuless::KERNEL_ARG_VALUE),
        };

        auto s = std::chrono::high_resolution_clock::now();

        // execute kernel on remote host
        exec.execute("vadd", dim3(N), dim3(1), args);

        auto e = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000.0;
        std::cout << d << std::endl;

        // free memory on remote host
        exec.deallocate();
}

void bench_latency_tcp() {
    for (int i = 0; i < RUNS; i++) {
        // gpuless::executor_tcp exec;

        // std::vector<size_t> buffers;
        // exec.allocate(buffers);

        // std::vector<gpuless::kernel_arg> args;
        // auto s = std::chrono::high_resolution_clock::now();
        // exec.execute("../kernels/build/common.fatbin", "noop", dim3(1), dim3(1), args);
        // auto e = std::chrono::high_resolution_clock::now();
        // auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000.0;
        // std::cout << d << std::endl;
    }
}

int main() {
    test_vadd_tcp();
    // bench_latency_tcp();
    return 0;
}
