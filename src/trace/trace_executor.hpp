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

namespace gpuless {
namespace executor {

class TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};

    bool negotiateSession(manager::instance_profile profile);

  public:
    TraceExecutor();
    ~TraceExecutor();

    bool init(const char *ip, const short port,
              manager::instance_profile profile);
    bool
    synchronize(std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack);
    bool deallocate();
};

} // namespace executor
} // namespace gpuless

#endif // __TRACE_EXECUTOR_HPP__