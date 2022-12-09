#ifndef GPULESS_TRACE_EXECUTOR_TCP_H
#define GPULESS_TRACE_EXECUTOR_TCP_H

#include "trace_executor.hpp"

namespace gpuless {

class TraceExecutorTcp : public TraceExecutor {
  private:
    sockaddr_in manager_addr{};
    sockaddr_in exec_addr{};
    int32_t session_id_ = -1;

    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

  private:
    bool negotiateSession(manager::instance_profile profile);
    bool getDeviceAttributes();

    bool init(const char *ip, const short port,
              manager::instance_profile profile);
    bool deallocate();

  public:
    TraceExecutorTcp(const char *ip, short port,
                     manager::instance_profile profile);
    ~TraceExecutorTcp();

    bool synchronize(gpuless::CudaTrace &cuda_trace) override;

    double getSynchronizeTotalTime() const override;
};

} // namespace gpuless

#endif // GPULESS_TRACE_EXECUTOR_TCP_H
