#ifndef __CUDA_TRACE_HPP__
#define __CUDA_TRACE_HPP__

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
    std::vector<std::shared_ptr<CudaApiCall>> synchronized_history_;
    std::vector<std::shared_ptr<CudaApiCall>> call_stack_;

    // map of all registered symbols
    // symbol -> (module_id, is_loaded)
    std::map<std::string, std::pair<uint64_t, bool>> symbol_to_module_id_;

    // map module ids to storage location and size of a fatbin module
    // module_id -> (resource_ptr, size, is_loaded)
    std::map<uint64_t, std::tuple<void *, uint64_t, bool>>
        module_id_to_fatbin_resource_;

  public:
    CudaTrace();

    const std::shared_ptr<CudaApiCall> &historyTop();
    void setHistoryTop(std::shared_ptr<CudaApiCall> top);
    std::vector<std::shared_ptr<CudaApiCall>> callStack();

    std::map<std::string, std::pair<uint64_t, bool>> &getSymbolToModuleId();
    std::map<uint64_t, std::tuple<void *, uint64_t, bool>> &
    getModuleIdToFatbinResource();

    void
    setCallStack(const std::vector<std::shared_ptr<CudaApiCall>> &callStack);

    void recordFatbinData(void *data, uint64_t size, uint64_t module_id);
    void recordSymbolMapEntry(std::string &symbol, uint64_t module_id);
    void recordGlobalVarMapEntry(std::string &symbol, uint64_t module_id);

    void record(const std::shared_ptr<CudaApiCall> &cudaApiCall);
    void markSynchronized();
};

} // namespace gpuless

#endif //  __CUDA_TRACE_HPP__
