#include "cuda_trace.hpp"

#include <utility>

namespace gpuless {

CudaTrace::CudaTrace() {}

void CudaTrace::record(const std::shared_ptr<CudaApiCall> &cudaApiCall) {
    this->call_stack_.push_back(cudaApiCall);
}

void CudaTrace::markSynchronized() {
    // move current trace to history
    std::move(std::begin(this->call_stack_), std::end(this->call_stack_),
              std::back_inserter(this->synchronized_history_));

    // clear the current trace
    this->call_stack_.clear();
}

const std::shared_ptr<CudaApiCall> &CudaTrace::historyTop() {
    return this->synchronized_history_.back();
}

std::vector<std::shared_ptr<CudaApiCall>> CudaTrace::callStack() {
    return this->call_stack_;
}

void CudaTrace::recordFatbinData(std::vector<uint8_t> &data,
                                 uint64_t module_id) {
    this->new_module_id_to_fatbin_data_map_.emplace_back(module_id, data);
}

void CudaTrace::recordSymbolMapEntry(std::string &symbol, uint64_t module_id) {
    this->new_symbol_to_module_id_.emplace_back(symbol, module_id);
    this->symbol_to_module_id_.emplace(symbol, module_id);
}

void CudaTrace::recordGlobalVarMapEntry(std::string &symbol,
                                        uint64_t module_id) {
    this->new_global_var_to_module_id_.emplace_back(symbol, module_id);
}
std::map<std::string, uint64_t> &CudaTrace::symbolToModuleIdMap() {
    return this->symbol_to_module_id_;
}

std::vector<std::pair<std::string, uint64_t>> &
CudaTrace::getNewSymbolToModuleId() {
    return new_symbol_to_module_id_;
}

std::vector<std::pair<std::string, uint64_t>> &
CudaTrace::getNewGlobalVarToModuleId() {
    return new_global_var_to_module_id_;
}

std::vector<std::pair<uint64_t, std::vector<uint8_t>>> &
CudaTrace::getNewModuleIdToFatbinDataMap() {
    return new_module_id_to_fatbin_data_map_;
}

} // namespace gpuless
