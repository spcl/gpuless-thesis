#include <chrono>
#include <cassert>

#include "gpuless.hpp"

const int N = 1024;
const int RUNS = 10;

void bench_latency_local() {
    for (int i = 0; i < RUNS; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        gpuless::local_executor exec;
        exec.allocate("../kernels/build/common.fatbin");
        exec.execute("noop", dim3(1), dim3(1));
        auto e = std::chrono::high_resolution_clock::now();
        auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000.0;
        std::cout << d << std::endl;
    }
}

int main() {
    bench_latency_local();
    return 0;
}
