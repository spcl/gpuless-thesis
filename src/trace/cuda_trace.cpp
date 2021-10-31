#include "cuda_trace.hpp"

namespace gpuless {

CudaTrace::CudaTrace(std::shared_ptr<executor::TraceExecutor> trace_executor,
                     CubinAnalyzer &cubin_analyzer)
    : trace_executor_(trace_executor), cubin_analyzer_(cubin_analyzer) {}

void CudaTrace::record(const std::shared_ptr<CudaApiCall>& cudaApiCall) {
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

void CudaTrace::synchronize() {
    this->trace_executor_->synchronize(this->call_stack_);
    this->markSynchronized();
}

} // namespace gpuless
