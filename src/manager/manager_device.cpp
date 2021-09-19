#include <iostream>
#include <sstream>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../schemas/execution_protocol_generated.h"
#include "../utils.hpp"
#include "manager_device.hpp"

extern const int BACKLOG;
extern const bool DEBUG;

CUdevice cu_device;
CUcontext cu_context;

using namespace gpuless::execution;

void handle_attribute_request(int socket_fd,
                              const gpuless::execution::ProtocolMessage *msg) {
    (void)msg;
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<CUdeviceAttributeValue>> attribute_values;

    for (int i = 1; i < CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX; i++) {
        int32_t result;
        checkCudaErrors(cuDeviceGetAttribute(
            &result, static_cast<CUdevice_attribute>(i), cu_device));
        auto av = CreateCUdeviceAttributeValue(
            builder, static_cast<CUdeviceAttribute>(i), result);
        attribute_values.push_back(av);
    }

    auto attributes_answer_msg = CreateProtocolMessage(
        builder, Message_AttributesAnswer,
        CreateAttributesAnswer(builder, Status_OK,
                               builder.CreateVector(attribute_values))
            .Union());
    builder.Finish(attributes_answer_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());
}

void handle_execute_request(int socket_fd,
                            const gpuless::execution::ProtocolMessage *msg) {
    flatbuffers::FlatBufferBuilder builder;
    auto fb_args = msg->message_as_ExecutionRequest()->arguments();
    auto cuda_bin = msg->message_as_ExecutionRequest()->cuda_bin()->data();
    auto kernel = msg->message_as_ExecutionRequest()->kernel()->data();
    auto dim_grid = msg->message_as_ExecutionRequest()->dim_grid();
    auto dim_block = msg->message_as_ExecutionRequest()->dim_block();

    if (DEBUG) {
        printf("executing \"%s\"\n", kernel);
        printf("grid(%ld,%ld,%ld), block(%ld,%ld,%ld)\n", dim_grid->x(),
               dim_grid->y(), dim_grid->z(), dim_block->x(), dim_block->y(),
               dim_block->z());
    }

    std::vector<CUdeviceptr> cu_device_buffers(fb_args->size());
    CUmodule cu_module;
    CUfunction cu_function;

    // load cuda binary
    checkCudaErrors(cuModuleLoadData(&cu_module, cuda_bin));
    checkCudaErrors(cuModuleGetFunction(&cu_function, cu_module, kernel));

    std::vector<void *> args(fb_args->size(), nullptr);
    for (size_t i = 0; i < fb_args->size(); i++) {
        auto a = fb_args->Get(i);

        // allocate memory on the device
        if (a->flags() & KERNEL_ARG_POINTER) {
            checkCudaErrors(
                cuMemAlloc(&cu_device_buffers[i], a->buffer()->size()));
        }

        // copy memory to device
        if (a->flags() & KERNEL_ARG_COPY_TO_DEVICE) {
            checkCudaErrors(cuMemcpyHtoD(cu_device_buffers[i],
                                         a->buffer()->data(),
                                         a->buffer()->size()));
        }

        // construct kernel argument array
        if (a->flags() & KERNEL_ARG_POINTER) {
            args[i] = (void *)&cu_device_buffers[i];
        } else {
            args[i] = (void *)a->buffer()->data();
        }
    }

    // execute kernel
    checkCudaErrors(cuLaunchKernel(
        cu_function, dim_grid->x(), dim_grid->y(), dim_grid->z(),
        dim_block->x(), dim_block->y(), dim_block->z(), 0, 0, args.data(), 0));
    checkCudaErrors(cuCtxSynchronize());

    std::vector<flatbuffers::Offset<ReturnBuffer>> return_buffers;
    for (size_t i = 0; i < fb_args->size(); i++) {
        auto a = fb_args->Get(i);

        // copy back memory from device
        if (a->flags() & KERNEL_ARG_COPY_TO_HOST) {
            checkCudaErrors(cuMemcpyDtoH((void *)a->buffer()->data(),
                                         cu_device_buffers[i],
                                         a->buffer()->size()));
            auto v =
                builder.CreateVector(a->buffer()->data(), a->buffer()->size());
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
        builder, Message_ExecutionAnswer,
        CreateExecutionAnswer(builder, Status_OK,
                              builder.CreateVector(return_buffers))
            .Union());
    builder.Finish(execution_answer_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    if (DEBUG) {
        printf("finished executing \"%s\"\n", kernel);
    }
}

void handle_request(int socket_fd) {
    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto msg = GetProtocolMessage(buffer.data());
    if (msg->message_type() == Message_ExecutionRequest) {
        handle_execute_request(socket_fd, msg);
    } else if (msg->message_type() == Message_AttributesRequest) {
        handle_attribute_request(socket_fd, msg);
    } else {
        std::cerr << "invalid request" << std::endl;
    }
}

void manage_device(int device, uint16_t port) {
    // initialize gpu device
    checkCudaErrors(cuInit(device));
    checkCudaErrors(cuDeviceGet(&cu_device, device));
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
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (void *)&opt, sizeof(opt));

    sockaddr_in sa;
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(port);

    if (bind(s, (sockaddr *)&sa, sizeof(sa)) < 0) {
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
