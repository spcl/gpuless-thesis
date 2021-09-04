#include <iostream>
#include <vector>
#include <sstream>

#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "manager_device.hpp"
#include "../utils.hpp"
#include "../schemas/execution_protocol_generated.h"

extern const int BACKLOG;
extern const bool DEBUG;

CUdevice cu_device;
CUcontext cu_context;

void handle_request(int socket_fd) {
    using namespace gpuless::execution;
    flatbuffers::FlatBufferBuilder builder;

    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto execution_request_msg = GetProtocolMessage(buffer.data());
    auto fb_args = execution_request_msg->message_as_ExecutionRequest()->arguments();
    auto cuda_bin = execution_request_msg->message_as_ExecutionRequest()->cuda_bin()->data();
    auto kernel = execution_request_msg->message_as_ExecutionRequest()->kernel()->data();
    auto dim_grid = execution_request_msg->message_as_ExecutionRequest()->dim_grid();
    auto dim_block = execution_request_msg->message_as_ExecutionRequest()->dim_block();

    if (DEBUG) {
        printf("executing \"%s\"\n", kernel);
    }

    std::vector<CUdeviceptr> cu_device_buffers(fb_args->size());
    CUmodule cu_module;
    CUfunction cu_function;

    // load cuda binary
    checkCudaErrors(cuModuleLoadData(&cu_module, cuda_bin));
    checkCudaErrors(cuModuleGetFunction(&cu_function, cu_module, kernel));

    std::vector<void*> args(fb_args->size(), nullptr);
    for (size_t i = 0; i < fb_args->size(); i++) {
        auto a = fb_args->Get(i);

        // allocate memory on the device
        if (a->flags() & KERNEL_ARG_POINTER) {
            checkCudaErrors(cuMemAlloc(&cu_device_buffers[i], a->buffer()->size()));
        }

        // copy memory to device
        if (a->flags() & KERNEL_ARG_COPY_TO_DEVICE) {
            checkCudaErrors(cuMemcpyHtoD(cu_device_buffers[i],
                                         a->buffer()->data(),
                                         a->buffer()->size()));
        }

        // construct kernel argument array
        if (a->flags() & KERNEL_ARG_POINTER) {
            args[i] = (void *) &cu_device_buffers[i];
        } else {
            args[i] = (void *) a->buffer()->data();
        }
    }

    // execute kernel
    checkCudaErrors(cuLaunchKernel(cu_function,
                                   dim_grid->x(), dim_grid->y(), dim_grid->z(),
                                   dim_block->x(), dim_block->y(), dim_block->z(),
                                   0, 0, args.data(), 0));
    checkCudaErrors(cuCtxSynchronize());

    std::vector<flatbuffers::Offset<ReturnBuffer>> return_buffers;
    for (size_t i = 0; i < fb_args->size(); i++) {
        auto a = fb_args->Get(i);

        // copy back memory from device
        if (a->flags() & KERNEL_ARG_COPY_TO_HOST) {
            checkCudaErrors(cuMemcpyDtoH((void*) a->buffer()->data(),
                                         cu_device_buffers[i],
                                         a->buffer()->size()));
            auto v = builder.CreateVector(a->buffer()->data(), a->buffer()->size());
            auto s = builder.CreateString(a->id()->data());
            auto rb = CreateReturnBuffer(builder, s, v);
            return_buffers.push_back(rb);
        }

        // free device memory
        if (a->flags() & KERNEL_ARG_POINTER) {
            checkCudaErrors(cuMemFree(cu_device_buffers[i]));
        }
    }

    auto execution_answer_msg = CreateProtocolMessage(
        builder,
        Message_ExecutionAnswer,
        CreateExecutionAnswer(builder, Status_OK, builder.CreateVector(return_buffers)).Union());
    builder.Finish(execution_answer_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void manage_device(int device, uint16_t port) {
    // initialize gpu device
    checkCudaErrors(cuInit(device));
    checkCudaErrors(cuDeviceGet(&cu_device, 0));
    char name[256];
    checkCudaErrors(cuDeviceGetName(name, 256, cu_device));
    std::cout << "initialized device: " << name << std::endl;
    checkCudaErrors(cuCtxCreate(&cu_context, 0, cu_device));

    // start server
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) {
        std::cerr << "failed to open socket" << std::endl;
        exit(EXIT_FAILURE);
    }

    int opt = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (void *) &opt, sizeof(opt));

    sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(port);

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
        if (DEBUG) {
            const char *ip = inet_ntoa(((sockaddr_in *)&remote_addr)->sin_addr);
            std::cout << "manager_device: connection from " << ip << std::endl;
        }
        // synchronous request handler
        handle_request(s_new);
        close(s_new);
    }

    close(s);
    exit(EXIT_SUCCESS);
}
