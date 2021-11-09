#ifndef GPULESS_TRACE_EXECUTOR_TCP_H
#define GPULESS_TRACE_EXECUTOR_TCP_H

#include "trace_executor.hpp"

class TraceExecutorTcp : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};

    bool negotiateSession(manager::instance_profile profile);

  public:
    TraceExecutorTcp();
    ~TraceExecutorTcp();

    bool init(const char *ip, const short port,
              manager::instance_profile profile);
    bool
    synchronize(std::vector<std::shared_ptr<gpuless::CudaApiCall>> &callStack);
    bool deallocate();
};

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
