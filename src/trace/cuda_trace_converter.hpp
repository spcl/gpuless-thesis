#ifndef GPULESS_CUDA_TRACE_CONVERTER_HPP
#define GPULESS_CUDA_TRACE_CONVERTER_HPP

#include "cuda_trace.hpp"
#include "flatbuffers/flatbuffers.h"
#include <vector>

namespace gpuless {

class CudaTraceConverter {
  public:
    void traceToExecRequest(CudaTrace &trace, std::vector<uint8_t> &buffer);
    void execResponseToTrace(CudaTrace &trace, std::vector<uint8_t> &buffer);
};

} // namespace gpuless

#endif // GPULESS_CUDA_TRACE_CONVERTER_HPP