#include <cassert>
#include <chrono>

#include "../executors/executor_tcp.hpp"

#define CUDA_BIN "../kernels/build/common.fatbin"
#define KERNEL "vadd"
#define IP "127.0.0.1"

const int N_RUNS = 10;
const short PORT = 8002;

int main(int argc, char **argv) {
    int size = 4096;
    if (argc == 2) {
        size = atoi(argv[1]);
    }

    gpuless::executor::executor_tcp exec;
    exec.init(IP, PORT, gpuless::manager::instance_profile::NO_MIG);
    exec.set_cuda_code_file(CUDA_BIN);

    std::vector<kernel_argument> args{
        gpuless::executor::pointer_arg("a", sizeof(float) * size,
                                       KERNEL_ARG_COPY_TO_DEVICE),
        gpuless::executor::pointer_arg("b", sizeof(float) * size,
                                       KERNEL_ARG_COPY_TO_DEVICE),
        gpuless::executor::pointer_arg("c", sizeof(float) * size,
                                       KERNEL_ARG_COPY_TO_HOST),
        gpuless::executor::value_arg("n", &size, 0),
    };

    float *a = (float *)args[0].buffer.data();
    float *b = (float *)args[1].buffer.data();
    float *c = (float *)args[2].buffer.data();

    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i + 1;
        c[i] = 0;
    }

    auto s = std::chrono::high_resolution_clock::now();
    exec.execute(KERNEL, dim3(size), dim3(1), args);
    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000.0;
    std::cout << d << " ms" << std::endl;

    assert(abs(c[0] - 1.0) < 1e-6);
    assert(abs(c[1] - 3.0) < 1e-6);
    assert(abs(c[2] - 5.0) < 1e-6);
    assert(abs(c[3] - 7.0) < 1e-6);
    assert(abs(c[4] - 9.0) < 1e-6);

    exec.deallocate();
    // std::cout << "executor_tcp deallocated" << std::endl;
    return EXIT_SUCCESS;
}
