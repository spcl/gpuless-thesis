#ifndef __CUDA_VDEV_HPP__
#define __CUDA_VDEV_HPP__

#include <cuda.h>
#include <cuda_runtime.h>

#include "../executors/executor_tcp.hpp"
#include "cubin_analysis.hpp"

/*
 * Virtual CUDA device implemented as a singleton.
 */

class CudaVDev {
  private:
    CudaVDev() {}

    bool initialized = false;
    CUdeviceptr next_cudevice_ptr = 1ULL;
    CUfunction next_function_ptr = (CUfunction)1;

  public:
    CudaVDev(CudaVDev const &) = delete;
    void operator=(CudaVDev const &) = delete;

    static CudaVDev &getInstance() {
        static CudaVDev instance;
        return instance;
    }

    gpuless::executor::executor_tcp executor;
    CubinAnalyzer cubin_analyzer;

    std::map<CUdeviceptr, std::vector<uint8_t>> device_buffers;
    std::map<CUfunction, std::string> function_name;
    std::map<void *, CUfunction> fn_ptr_to_cufunction;

    cudaDeviceProp getDeviceProperties();
    void *memAlloc(size_t size);
    void memCpyHtoD(CUdeviceptr dst, void *src, size_t size);
    void memCpyDtoH(void *dst, CUdeviceptr src, size_t size);
    std::vector<uint8_t> *buffer_from_ptr(CUdeviceptr ptr);

    std::vector<std::string> cuda_binaries;
    char *manager_ip = nullptr;
    short manager_port = -1;

    void initialize(std::vector<std::string> cuda_bin_fname, char *manager_ip, short manager_port);
    CUfunction next_cufunction_ptr();
    // CUdeviceptr next_cudevice_ptr();
};

#endif // __CUDA_VDEV_HPP__