#ifndef GPULESS_TRACE_EXECUTOR_TCP_H
#define GPULESS_TRACE_EXECUTOR_TCP_H

#include "trace_executor.hpp"

namespace gpuless {

class TraceExecutorTcp : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};
    uint64_t synchronize_counter_ = 0;
    int32_t session_id_ = -1;

    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

  public:
    TraceExecutorTcp();
    ~TraceExecutorTcp();

    bool init(const char *ip, short port,
              manager::instance_profile profile) override;
    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
    bool deallocate() override;
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
