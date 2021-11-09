#include "cuda_trace.hpp"

#include <utility>

namespace gpuless {

CudaTrace::CudaTrace(CubinAnalyzer &cubin_analyzer,
                     std::shared_ptr<CudaVirtualDevice> cuda_virtual_device)
    : cubin_analyzer_(cubin_analyzer),
      cuda_virtual_device_(std::move(cuda_virtual_device)) {}

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

const CubinAnalyzer &CudaTrace::cubinAnalyzer() {
    return this->cubin_analyzer_;
}

CudaVirtualDevice &CudaTrace::cudaVirtualDevice() {
    return *this->cuda_virtual_device_;
}
std::vector<std::shared_ptr<CudaApiCall>> CudaTrace::callStack() {
    return this->call_stack_;
}

//void CudaTrace::synchronize() {
//    this->trace_executor_->synchronize(this->call_stack_);
//    this->markSynchronized();
//}

} // namespace gpuless
