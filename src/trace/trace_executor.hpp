#ifndef GPULESS_TRACEEXECUTOR_H
#define GPULESS_TRACEEXECUTOR_H

#include "../manager/manager.hpp"
#include "../manager/manager_device.hpp"
#include "cuda_api_calls.hpp"

namespace gpuless {
namespace executor {

class TraceExecutor {
//  private:
//    sockaddr_in managerAddr{};
//    sockaddr_in execAddr{};
//
//    virtual bool
//    negotiateSession(gpuless::manager::instance_profile profile) = 0;

  public:
//    TraceExecutor() = default;
//    ~TraceExecutor() = default;

    virtual bool init(const char *ip, const short port,
                      gpuless::manager::instance_profile profile) = 0;
    virtual bool synchronize(
        std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack) = 0;
    virtual bool deallocate() = 0;
};

} // namespace executor
} // namespace gpuless

#endif // GPULESS_TRACEEXECUTOR_H
