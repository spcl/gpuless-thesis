#ifndef GPULESS_TRACEEXECUTOR_H
#define GPULESS_TRACEEXECUTOR_H

#include "../manager/manager.hpp"
#include "../manager/manager_device.hpp"
#include "cuda_api_calls.hpp"
#include "cuda_trace.hpp"

namespace gpuless {

class CudaTrace;

class TraceExecutor {
  public:
    virtual bool init(const char *ip, short port,
                      gpuless::manager::instance_profile profile) = 0;
    virtual bool synchronize(gpuless::CudaTrace &cuda_trace) = 0;
    virtual bool deallocate() = 0;
};

} // namespace gpuless

#endif // GPULESS_TRACEEXECUTOR_H
