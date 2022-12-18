#ifndef GPULESS_TRACE_EXECUTOR_TCP_H
#define GPULESS_TRACE_EXECUTOR_TCP_H

#include "../TcpClient.hpp"
#include "tcp_gpu_session.hpp"
#include "trace_executor.hpp"

namespace gpuless {

class TraceExecutorTcp : public TraceExecutor {
  private:
    TcpGpuSession m_gpusession;
    sockaddr_in m_manager_addr;

    uint64_t synchronize_counter_ = 0;
    double synchronize_total_time_ = 0;

  private:
    TcpGpuSession negotiateSession(const char *ip, const short port,
                                   manager::instance_profile profile);
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
