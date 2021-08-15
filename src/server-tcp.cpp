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

const uint16_t PORT = 8001;
const int BACKLOG = 5;

std::vector<CUdeviceptr> cu_device_buffers;
CUdevice                 cu_device;
CUcontext                cu_context;
CUmodule                 cu_module;
CUfunction               cu_function;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult r, const char *file, const int line) {
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(r, &msg);
        std::cout
            << "cuda error in "
            << file
            << "(" << line << "):"
            << std::endl
            << msg
            << std::endl;
    }
}

template<typename T>
static T read_value(size_t **next_object) {
    size_t l = **next_object;
    T v = *((T *) (*next_object + 1));
    *next_object = (size_t *) ((uint8_t *) *next_object + sizeof(size_t) + l);
    return v;
}

void handle_allocate(std::vector<uint8_t> &buf) {
    size_t *next_object = (size_t *) buf.data();

    uint8_t type = read_value<uint8_t>(&next_object);
    size_t n_buffers = read_value<size_t>(&next_object);

    std::vector<size_t> buffer_sizes;
    for (int i = 0; i < n_buffers; i++) {
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
    for (int i = 0; i < n_buffers; i++) {
        checkCudaErrors(cuMemAlloc(&cu_device_buffers[i], buffer_sizes[i]));
    }
}

void handle_deallocate() {
    for (const auto &b : cu_device_buffers) {
        checkCudaErrors(cuMemFree(b));
    }
    cuCtxDestroy(cu_context);
}

void handle_execute(std::vector<uint8_t> &buf) {
    size_t *next_object = (size_t *) buf.data();

    uint8_t type = read_value<uint8_t>(&next_object);
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
    for (int i = 0; i < n_args; i++) {
        int32_t flags = read_value<int32_t>(&next_object);
        size_t data_len = *next_object;
        kernel_args[i] = gpuless::kernel_arg {
            flags,
            data_len,
            (void *) (next_object + 1),
        };
        // args[i] = (void *) (next_object + 1);
        next_object = (size_t *) ((uint8_t *) next_object + sizeof(size_t) + data_len);
    }


    // copy data to device
    int cu_buf_ctr = 0;
    for (int i = 0; i < n_args; i++) {
        if (kernel_args[i].flags & gpuless::KERNEL_ARG_COPY_TO_DEVICE) {
            checkCudaErrors(cuMemcpyHtoD(cu_device_buffers[cu_buf_ctr],
                                         kernel_args[i].data,
                                         kernel_args[i].length));
            cu_buf_ctr++;
        }
    }

    void *args[n_args];
    cu_buf_ctr = 0;
    for (int i = 0; i < n_args; i++) {
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
    for (int i = 0; i < n_args; i++) {
        if (kernel_args[i].flags & gpuless::KERNEL_ARG_POINTER) {
            if (kernel_args[i].flags & gpuless::KERNEL_ARG_COPY_TO_HOST) {
                checkCudaErrors(cuMemcpyDtoH(kernel_args[i].data,
                            cu_device_buffers[cu_buf_ctr],
                            kernel_args[i].length));
            }
            cu_buf_ctr++;
        }
    }
}

void handle_request(int socket_fd) {
    std::vector<uint8_t> buf(256);
    ssize_t n;
    ssize_t bytes_read = 0;
    while ((n = read(socket_fd, buf.data()+bytes_read, 256)) > 0) {
        bytes_read += n;
        auto d = buf.size() - bytes_read;
        if (d < 256) {
            buf.resize(buf.size() + 256 - d);
        }
    }

    // type
    size_t *next_object = (size_t *) buf.data();
    uint8_t type = *((uint8_t *) (next_object + 1));

    if (type == gpuless::REQ_TYPE_ALLOCATE) {
        std::cout << "allocate request" << std::endl;
        handle_allocate(buf);
    } else if (type == gpuless::REQ_TYPE_DEALLOCATE) {
        std::cout << "deallocate request" << std::endl;
        handle_deallocate();
    } else if (type == gpuless::REQ_TYPE_EXECUTE) {
        std::cout << "execute request" << std::endl;
        handle_execute(buf);
    } else {
        std::cout << "invalid request" << std::endl;
    }
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
