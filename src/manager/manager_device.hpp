#ifndef __MANAGER_DEVICE_HPP__
#define __MANAGER_DEVICE_HPP__

#include <cstdint>
#include <string>
#include <vector>

extern const int BACKLOG;
extern const bool DEBUG;

const int32_t KERNEL_ARG_POINTER        = 1 << 0;
const int32_t KERNEL_ARG_VALUE          = 1 << 1;
const int32_t KERNEL_ARG_COPY_TO_DEVICE = 1 << 2;
const int32_t KERNEL_ARG_COPY_TO_HOST   = 1 << 3;

struct kernel_argument {
    std::string          id;
    int32_t              flags;
    std::vector<uint8_t> buffer;

    kernel_argument() {}
    kernel_argument(std::string id, int32_t flags, std::vector<uint8_t> buffer)
        : id(id), flags(flags), buffer(buffer) {}
};

void manage_device(int device, uint16_t port);

#endif // __MANAGER_DEVICE_HPP__
