#ifndef __TRACE_EXECUTOR_HPP__
#define __TRACE_EXECUTOR_HPP__

#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../manager/manager.hpp"
#include "../manager/manager_device.hpp"
#include "cuda_api_calls.hpp"
#include "trace_executor.hpp"

namespace gpuless {
namespace executor {

class TraceExecutorLocal final : public TraceExecutor {
  public:
    TraceExecutorLocal();
    ~TraceExecutorLocal();

    bool init(const char *ip, const short port,
              manager::instance_profile profile);
    bool
    synchronize(std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack);
    bool deallocate();
};

} // namespace executor
} // namespace gpuless

#endif // __TRACE_EXECUTOR_HPP__