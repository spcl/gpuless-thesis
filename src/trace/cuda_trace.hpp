#ifndef __TRACE_HPP__
#define __TRACE_HPP__

#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "cubin_analysis.hpp"
#include "cuda_api_calls.hpp"
#include "libgpuless.hpp"

namespace gpuless {

class CudaTrace {
  private:
    CubinAnalyzer &cubin_analyzer_;
    std::shared_ptr<CudaVirtualDevice> cuda_virtual_device_;

    std::vector<std::shared_ptr<CudaApiCall>> synchronized_history_;
    std::vector<std::shared_ptr<CudaApiCall>> call_stack_;

  public:
    CudaTrace(CubinAnalyzer &cubin_analyzer,
              std::shared_ptr<CudaVirtualDevice> cuda_virtual_device);

    const std::shared_ptr<CudaApiCall> &historyTop();
    const CubinAnalyzer &cubinAnalyzer();
    CudaVirtualDevice &cudaVirtualDevice();
    std::vector<std::shared_ptr<CudaApiCall>> callStack();

    void record(const std::shared_ptr<CudaApiCall> &cudaApiCall);
    void markSynchronized();
//    void synchronize();
};

} // namespace gpuless

#endif //  __TRACE_HPP__
