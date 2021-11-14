#ifndef GPULESS_CUDA_VDEV_H
#define GPULESS_CUDA_VDEV_H

#include "../utils.hpp"
#include <cstdint>
#include <cuda.h>
#include <fatbinary_section.h>
#include <iostream>
#include <map>
#include <string>

class CudaVirtualDevice {
  private:
    bool initialized = false;

  public:
    std::map<std::string, uint64_t> symbol_to_module_id_map;
    std::map<std::string, uint64_t> global_var_to_module_id_map;
    std::map<uint64_t, std::vector<uint8_t>> module_id_to_fatbin_data_map;

    std::map<uint64_t, CUmodule> module_registry_;
    std::map<std::string, CUfunction> function_registry_;

    // execution side state
    CUdevice device;
    CUcontext context;

    void initRealDevice();
};

#endif // GPULESS_CUDA_VDEV_H
