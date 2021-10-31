#include "trace_executor_local.hpp"
#include <iostream>

namespace gpuless::executor {

TraceExecutorLocal::TraceExecutorLocal() = default;
TraceExecutorLocal::~TraceExecutorLocal() = default;

bool TraceExecutorLocal::init(const char *ip, const short port,
                              manager::instance_profile profile) {
    return true;
}

bool TraceExecutorLocal::synchronize(
    std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack) {
    std::cout << "TraceExecutorLocal::synchronize()" << std::endl;
    for (auto &apiCall : callStack) {
        std::cout << "  " << apiCall->typeName() << std::endl;
        cudaError_t err;
        if ((err = apiCall->executeNative()) != cudaSuccess) {
            std::cerr << "    failed to execute call trace: "
                      << cudaGetErrorString(err) << " (" << err << ")"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    return true;
}

bool TraceExecutorLocal::deallocate() { return false; }

} // namespace gpuless::executor