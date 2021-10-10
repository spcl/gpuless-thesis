#ifndef __TRACE_HPP__
#define __TRACE_HPP__

#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "libgpuless.hpp"
#include "cuda_api_calls.hpp"
#include "trace_executor.hpp"

namespace gpuless {

class CudaTrace {
  private:
    std::vector<std::shared_ptr<CudaApiCall>> synchronized_history_;
    std::vector<std::shared_ptr<CudaApiCall>> call_stack_;
    executor::TraceExecutor trace_executor_;
    void markSynchronized();

  public:
    const std::shared_ptr<CudaApiCall> &historyTop();
    CudaTrace(executor::TraceExecutor &traceExecutor);
    void record(const std::shared_ptr<CudaApiCall>& cudaApiCall);
    void synchronize();
};

} // namespace gpuless
#endif //  __TRACE_HPP__
