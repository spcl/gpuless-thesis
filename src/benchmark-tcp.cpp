#include <chrono>
#include <vector>
#include <cassert>
#include <iostream>

#include "executor_tcp.hpp"

const int RUNS = 10;

void test_vadd_tcp(char *ip) {
    const int N = 1024;

    gpuless::executor_tcp exec;
    gpuless::buffer_tcp a(sizeof(float) * N);
    gpuless::buffer_tcp b(sizeof(float) * N);
    gpuless::buffer_tcp c(sizeof(float) * N);

    std::vector<gpuless::buffer_tcp*> buffers {
        &a, &b, &c
    };

    // allocate memory on remote host
    exec.allocate(ip, 8001, "../kernels/build/common.fatbin", buffers);

    for (int i = 0; i < N; i++) {
        ((float *) a.data)[i] = i;
        ((float *) b.data)[i] = i+1;
        ((float *) c.data)[i] = 0;
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
    std::cout << d << " ms" << std::endl;

    float *c_ = (float *) c.data;
    assert(c_[0] == 1);
    assert(c_[1] == 3);
    assert(c_[2] == 5);
    assert(c_[3] == 7);
    assert(c_[4] == 9);

    // free memory on remote host
    exec.deallocate();
}

void bench_rtt(char *ip, size_t size) {
    for (int i = 0; i < RUNS; i++) {
        gpuless::executor_tcp exec;
        gpuless::buffer_tcp b(size);
        std::vector<gpuless::buffer_tcp*> buffers { &b };
        exec.allocate(ip, 8001, "../kernels/build/common.fatbin", buffers);

        std::vector<gpuless::kernel_arg> args {
            gpuless::executor_tcp::pointer_argument(buffers[0],
                    gpuless::KERNEL_ARG_POINTER
                        | gpuless::KERNEL_ARG_COPY_TO_DEVICE
                        | gpuless::KERNEL_ARG_COPY_TO_HOST),
        };

        auto s = std::chrono::high_resolution_clock::now();
        exec.execute("noop_arg", dim3(1), dim3(1), args);
        auto e = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000.0;
        std::cout << d << " ms" << std::endl;

        exec.deallocate();
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "wrong number of arguments" << std::endl;
        return 1;
    }

    size_t size = atol(argv[1]);
    char *ip = argv[2];
    bench_rtt(ip, size);

    // test_vadd_tcp();

    return 0;
}
