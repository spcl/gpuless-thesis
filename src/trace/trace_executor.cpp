#include "trace_executor.hpp"
#include <iostream>

namespace gpuless::executor {

TraceExecutor::TraceExecutor() = default;
TraceExecutor::~TraceExecutor() = default;

// TODO
bool TraceExecutor::negotiateSession(manager::instance_profile profile) {
    return true;
}

// TODO
bool TraceExecutor::init(const char *ip, const short port,
                         manager::instance_profile profile) {
    return true;
}

// TODO

bool TraceExecutor::synchronize(std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack) {
    std::cout << "TraceExecutor::synchronize" << std::endl;
    for (auto &apiCall : callStack) {
        std::cout << "  " << apiCall->typeName() << std::endl;
        cudaError_t err;
        if ((err = apiCall->executeNative()) != cudaSuccess) {
            std::cerr << "    failed to execute call trace: " << err << std::endl;
        }
    }

    return true;
}

bool TraceExecutor::deallocate() { return false; }

} // namespace gpuless::executor