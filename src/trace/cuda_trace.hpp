#ifndef __TRACE_HPP__
#define __TRACE_HPP__

#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "../adapter/cubin_analysis.hpp"
#include "cuda_api_calls.hpp"
#include "libgpuless.hpp"
#include "trace_executor.hpp"
#include "trace_executor_local.hpp"

namespace gpuless {

class CudaTrace {
  private:
    std::shared_ptr<executor::TraceExecutor> trace_executor_;
    CubinAnalyzer &cubin_analyzer_;

    std::vector<std::shared_ptr<CudaApiCall>> synchronized_history_;
    std::vector<std::shared_ptr<CudaApiCall>> call_stack_;
    void markSynchronized();

  public:
    CudaTrace(std::shared_ptr<executor::TraceExecutor> trace_executor,
              CubinAnalyzer &cubin_analyzer);

    const std::shared_ptr<CudaApiCall> &historyTop();
    const CubinAnalyzer &cubinAnalyzer();

    void record(const std::shared_ptr<CudaApiCall>& cudaApiCall);
    void synchronize();
};

} // namespace gpuless
#endif //  __TRACE_HPP__
