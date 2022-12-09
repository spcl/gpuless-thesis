#ifndef __TRACE_EXECUTOR_HPP__
#define __TRACE_EXECUTOR_HPP__

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "cuda_api_calls.hpp"
#include "cuda_virtual_device.hpp"
#include "manager/manager.hpp"
#include "manager/manager_device.hpp"
#include "trace_executor.hpp"

namespace gpuless {

class TraceExecutorLocal final : public TraceExecutor {
  private:
    CudaVirtualDevice cuda_virtual_device_;
    uint64_t synchronize_counter_ = 0;

  public:
    TraceExecutorLocal();
    ~TraceExecutorLocal();

    bool synchronize(gpuless::CudaTrace &cuda_trace) override;
};

} // namespace gpuless

#endif // __TRACE_EXECUTOR_HPP__
