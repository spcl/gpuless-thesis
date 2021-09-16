#ifndef __EXECUTOR_TCP_HPP__
#define __EXECUTOR_TCP_HPP__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cerrno>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "../manager/manager.hpp"
#include "../manager/manager_device.hpp"

namespace gpuless {
namespace executor {

kernel_argument pointer_arg(std::string &&id, size_t size, int32_t flags);

template<typename T>
kernel_argument value_arg(std::string &&id, T *value, int32_t flags) {
    std::vector<uint8_t> buf(sizeof(*value));
    memcpy(buf.data(), (void *) value, sizeof(*value));
    return kernel_argument {
        id,
        flags | KERNEL_ARG_VALUE,
        buf,
    };
}

class executor_tcp {
private:
    std::vector<uint8_t> cuda_bin;

    sockaddr_in manager_addr;
    sockaddr_in exec_addr;
    int session_id = -1;

    std::vector<int32_t> device_attributes;

    bool load_cuda_bin(const char *fname);
    bool negotiate_session(gpuless::manager::instance_profile profile);
    bool query_device_attributes();

public:
    executor_tcp()
        : device_attributes(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX) {}

    bool set_cuda_code(const void *data, size_t size);
    bool set_cuda_code_file(const char *fname);

    bool get_device_attribute(int *value, CUdevice_attribute attribute);
    bool init(const char *ip, const short port, gpuless::manager::instance_profile profile);
    bool execute(const char *kernel, dim3 dim_grid, dim3 dim_block, std::vector<kernel_argument> &args);
    bool deallocate();
};

} // namespace executor
} // namespace gpuless

#endif // __EXECUTOR_TCP_HPP__

