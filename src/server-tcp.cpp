#include <vector>
#include <iostream>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "executor_tcp.hpp"
#include "utils.hpp"

using namespace gpuless;

const uint16_t PORT = 8001;
const int BACKLOG = 5;

std::vector<CUdeviceptr> cu_device_buffers;
CUdevice                 cu_device;
CUcontext                cu_context;
CUmodule                 cu_module;
CUfunction               cu_function;

void handle_allocate(std::vector<uint8_t> &buf, std::vector<uint8_t> &answer) {
    size_t *next_object = (size_t *) buf.data();

    read_value<uint8_t>(&next_object); // type
    size_t n_buffers = read_value<size_t>(&next_object);

    std::vector<size_t> buffer_sizes;
    for (size_t i = 0; i < n_buffers; i++) {
        size_t s = read_value<size_t>(&next_object);
        buffer_sizes.push_back(s);
    }

    std::cout << "allocating buffers: ";
    for (const auto &s : buffer_sizes) {
        std::cout << s << " ";
    }
    std::cout << std::endl;

    // init cuda
    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGet(&cu_device, 0));
    char name[256];
    checkCudaErrors(cuDeviceGetName(name, 256, cu_device));
    std::cout << "Using device: " << name << std::endl;
    checkCudaErrors(cuCtxCreate(&cu_context, 0, cu_device));

    // cuda malloc
    cu_device_buffers.resize(buffer_sizes.size());
    for (size_t i = 0; i < n_buffers; i++) {
        checkCudaErrors(cuMemAlloc(&cu_device_buffers[i], buffer_sizes[i]));
    }

    gpuless::append_buffer_value(answer, &gpuless::REQ_ANS_OK);
}

void handle_deallocate(std::vector<uint8_t> &answer) {
    for (const auto &b : cu_device_buffers) {
        checkCudaErrors(cuMemFree(b));
    }
    cuCtxDestroy(cu_context);

    gpuless::append_buffer_value(answer, &gpuless::REQ_ANS_OK);
}

void handle_execute(std::vector<uint8_t> &buf, std::vector<uint8_t> &answer) {
    size_t *next_object = (size_t *) buf.data();

    read_value<uint8_t>(&next_object); // type
    dim3 dimGrid = read_value<dim3>(&next_object);
    dim3 dimBlock = read_value<dim3>(&next_object);

    // kernel name string
    size_t kernel_name_len = *next_object;
    char kernel[kernel_name_len] = {0};
    strncpy(kernel, (const char *) (next_object+1), kernel_name_len);
    next_object = (size_t *) ((uint8_t *) next_object + sizeof(size_t) + kernel_name_len);

    // cuda binary code
    size_t cuda_bin_len = *next_object;
    uint8_t cuda_bin[cuda_bin_len] = {0};
    memcpy(cuda_bin, next_object+1, cuda_bin_len);
    next_object = (size_t *) ((uint8_t *) next_object + sizeof(size_t) + cuda_bin_len);
    checkCudaErrors(cuModuleLoadData(&cu_module, cuda_bin));
    checkCudaErrors(cuModuleGetFunction(&cu_function, cu_module, kernel));

    // input data
    size_t n_args = read_value<size_t>(&next_object);
    gpuless::kernel_arg kernel_args[n_args];
    for (size_t i = 0; i < n_args; i++) {
        int32_t flags = read_value<int32_t>(&next_object);
        size_t data_len = *next_object;
        kernel_args[i] = gpuless::kernel_arg {
            flags,
            data_len,
            (void *) (next_object + 1),
        };
        next_object = (size_t *) ((uint8_t *) next_object + sizeof(size_t) + data_len);
    }

    // copy data to device
    int cu_buf_ctr = 0;
    for (size_t i = 0; i < n_args; i++) {
        if (kernel_args[i].flags & gpuless::KERNEL_ARG_COPY_TO_DEVICE) {
            checkCudaErrors(cuMemcpyHtoD(cu_device_buffers[cu_buf_ctr],
                                         kernel_args[i].data,
                                         kernel_args[i].length));
            cu_buf_ctr++;
        }
    }

    void *args[n_args];
    cu_buf_ctr = 0;
    for (size_t i = 0; i < n_args; i++) {
        if (kernel_args[i].flags & gpuless::KERNEL_ARG_POINTER) {
            args[i] = &cu_device_buffers[cu_buf_ctr];
            cu_buf_ctr++;
        } else {
            args[i] = kernel_args[i].data;
        }
    }

    printf("launching kernel: %s on grid(%d,%d,%d), block(%d,%d,%d)\n",
            kernel, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    checkCudaErrors(cuLaunchKernel(cu_function,
                                   dimGrid.x, dimGrid.y, dimGrid.z,
                                   dimBlock.x, dimBlock.y, dimBlock.z,
                                   0, 0, args, 0));
    checkCudaErrors(cuCtxSynchronize());

    // copy data back to host
    cu_buf_ctr = 0;
    size_t n_copy_back = 0;
    for (size_t i = 0; i < n_args; i++) {
        if (kernel_args[i].flags & gpuless::KERNEL_ARG_POINTER) {
            if (kernel_args[i].flags & gpuless::KERNEL_ARG_COPY_TO_HOST) {
                checkCudaErrors(cuMemcpyDtoH(kernel_args[i].data,
                            cu_device_buffers[cu_buf_ctr],
                            kernel_args[i].length));
                n_copy_back++;
            }
            cu_buf_ctr++;
        }
    }

    gpuless::append_buffer_value(answer, &gpuless::REQ_ANS_OK);
    gpuless::append_buffer_value(answer, &n_copy_back);
    for (size_t i = 0; i < n_args; i++) {
        int flags = kernel_args[i].flags;
        if (flags & gpuless::KERNEL_ARG_POINTER &&
            flags & gpuless::KERNEL_ARG_COPY_TO_HOST) {
            gpuless::append_buffer_raw(answer,
                                       kernel_args[i].data,
                                       kernel_args[i].length);
        }
    }
}

void handle_request(int socket_fd) {
    size_t req_len;
    recv(socket_fd, &req_len, sizeof(req_len), 0);

    std::vector<uint8_t> buf(req_len);
    recv(socket_fd, buf.data(), req_len, 0);

    std::vector<uint8_t> answer;

    size_t *next_object = (size_t *) buf.data();
    uint8_t type = read_value<uint8_t>(&next_object);

    if (type == gpuless::REQ_TYPE_ALLOCATE) {
        std::cout << "allocate request" << std::endl;
        handle_allocate(buf, answer);
    } else if (type == gpuless::REQ_TYPE_DEALLOCATE) {
        std::cout << "deallocate request" << std::endl;
        handle_deallocate(answer);
    } else if (type == gpuless::REQ_TYPE_EXECUTE) {
        std::cout << "execute request" << std::endl;
        handle_execute(buf, answer);
    } else {
        std::cout << "invalid request" << std::endl;
    }

    size_t ans_len = answer.size();
    send(socket_fd, &ans_len, sizeof(ans_len), 0);
    send(socket_fd, answer.data(), answer.size(), 0);
    close(socket_fd);
}

int main() {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        std::cerr << "failed to open socket" << std::endl;
        exit(EXIT_FAILURE);
    }

    sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(PORT);

    if (bind(s, (sockaddr *) &sa, sizeof(sa)) < 0) {
        std::cerr << "failed to bind socket" << std::endl;
        close(s);
        exit(EXIT_FAILURE);
    }

    if (listen(s, BACKLOG) < 0) {
        std::cerr << "failed to listen on socket" << std::endl;
        close(s);
        exit(EXIT_FAILURE);
    }

    int s_new;
    sockaddr remote_addr;
    socklen_t remote_addrlen = sizeof(remote_addr);
    while ((s_new = accept(s, &remote_addr, &remote_addrlen))) {
        const char *ip = inet_ntoa(((sockaddr_in *) &remote_addr)->sin_addr);
        std::cout << "connection from " << ip << std::endl;
        handle_request(s_new);
        close(s_new);
    }

    close(s);
    return 0;
}
