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

    // complete map of all registered symbols
    std::map<std::string, uint64_t> symbol_to_module_id_;

    // newly recorded mappings that need to be synchronized with the remote
    // executor
    std::vector<std::pair<std::string, uint64_t>> new_symbol_to_module_id_;
    std::vector<std::pair<std::string, uint64_t>> new_global_var_to_module_id_;
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>>
        new_module_id_to_fatbin_data_map_;

  public:
    CudaTrace();

    const std::shared_ptr<CudaApiCall> &historyTop();
    std::vector<std::shared_ptr<CudaApiCall>> callStack();

    std::vector<std::pair<std::string, uint64_t>> &
    getNewSymbolToModuleId();
    std::vector<std::pair<std::string, uint64_t>> &
    getNewGlobalVarToModuleId();
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> &
    getNewModuleIdToFatbinDataMap();
    std::map<std::string, uint64_t> &symbolToModuleIdMap();

    void recordFatbinData(std::vector<uint8_t> &data, uint64_t module_id);
    void recordSymbolMapEntry(std::string &symbol, uint64_t module_id);
    void recordGlobalVarMapEntry(std::string &symbol, uint64_t module_id);

    void record(const std::shared_ptr<CudaApiCall> &cudaApiCall);
    void markSynchronized();
};

} // namespace gpuless

#endif //  __CUDA_TRACE_HPP__
