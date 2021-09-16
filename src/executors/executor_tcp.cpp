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

#include "executor_tcp.hpp"
#include "../manager/manager_device.hpp"
#include "../utils.hpp"
#include "../manager/manager.hpp"
#include "../schemas/allocation_protocol_generated.h"
#include "../schemas/execution_protocol_generated.h"

namespace gpuless {
namespace executor {

kernel_argument pointer_arg(std::string &&id, size_t size, int32_t flags) {
    return kernel_argument {
        id,
        flags | KERNEL_ARG_POINTER,
        std::vector<uint8_t>(size),
    };
}

bool executor_tcp::load_cuda_bin(const char *fname) {
    std::ifstream input(fname, std::ios::binary);
    this->cuda_bin = std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
    return true;
}

bool executor_tcp::negotiate_session(gpuless::manager::instance_profile profile) {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "failed to open socket" << std::endl;
        return false;
    }

    if (connect(socket_fd, (sockaddr *) &this->manager_addr, sizeof(manager_addr)) < 0) {
        std::cerr << "failed to connect" << std::endl;
        return false;
    }

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;

    // make initial request
    auto allocate_request_msg = CreateProtocolMessage(
        builder,
        Message_AllocateRequest,
        CreateAllocateRequest(builder, profile, -1).Union());
    builder.Finish(allocate_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // manager offers some sessions
    std::vector<uint8_t> buffer_offer = recv_buffer(socket_fd);
    auto allocate_offer_msg = GetProtocolMessage(buffer_offer.data());
    auto offered_profiles = allocate_offer_msg->message_as_AllocateOffer()->available_profiles();
    int32_t selected_profile = offered_profiles->Get(0);
    int32_t session_id = allocate_offer_msg->message_as_AllocateOffer()->session_id();

    // choose a profile and send finalize request
    builder.Reset();
    auto allocate_select_msg = CreateProtocolMessage(
        builder,
        Message_AllocateSelect,
        CreateAllocateSelect(builder, Status_OK, session_id, selected_profile).Union());
    builder.Finish(allocate_select_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    // get server confirmation
    std::vector<uint8_t> buffer_confirm = recv_buffer(socket_fd);
    auto allocate_confirm_msg = GetProtocolMessage(buffer_confirm.data());
    bool ret = false;
    if (allocate_confirm_msg->message_as_AllocateConfirm()->status() == Status_OK) {
        auto port = allocate_confirm_msg->message_as_AllocateConfirm()->port();
        auto ip = allocate_confirm_msg->message_as_AllocateConfirm()->ip();
        this->session_id = session_id;
        this->exec_addr.sin_family = AF_INET;
        this->exec_addr.sin_port = htons(port);
        this->exec_addr.sin_addr = *((struct in_addr *) &ip);
        ret = true;
    }

    close(socket_fd);
    return ret;
}


bool executor_tcp::query_device_attributes() {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "failed to open socket" << std::endl;
        return false;
    }

    if (connect(socket_fd, (sockaddr *) &exec_addr, sizeof(exec_addr)) < 0) {
        std::cerr << "failed to connect" << std::endl;
        return false;
    }

    using namespace gpuless::execution;
    flatbuffers::FlatBufferBuilder builder;

    auto attributes_request_msg = CreateProtocolMessage(
        builder,
        Message_AttributesRequest,
        CreateAttributesRequest(builder).Union());
    builder.Finish(attributes_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto execution_answer_msg = GetProtocolMessage(buffer.data());
    auto attributes = execution_answer_msg->message_as_AttributesAnswer()->attributes();
    for (unsigned i = 0; i < attributes->size(); i++) {
        auto av = attributes->Get(i);
        int32_t attr_int = static_cast<int32_t>(av->attribute());
        int32_t value = av->value();
        this->device_attributes[attr_int] = value;
    }

    return true;
}

// load kernel code from memory
bool executor_tcp::set_cuda_code(const void *data, size_t size) {
    this->cuda_bin.resize(size);
    memcpy(this->cuda_bin.data(), data, size);
    return true;
}

// load kernel code from file
bool executor_tcp::set_cuda_code_file(const char *fname) {
    if (!load_cuda_bin(fname)) {
        std::cerr << "failed to load cuda binary" << std::endl;
        return false;
    }
    return true;
}

bool executor_tcp::init(const char *ip,
            const short port,
            gpuless::manager::instance_profile profile) {
    // store and check server address/port
    manager_addr.sin_family = AF_INET;
    manager_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &manager_addr.sin_addr) < 0) {
        std::cerr << "invalid ip address" << std::endl;
        return false;
    }

    // allocate an execution instance with the manager
    if (!negotiate_session(profile)) {
        std::cerr << "failed to negotiate session" << std::endl;
        return false;
    }

    // query all device attributes
    if (!query_device_attributes()) {
        std::cerr << "failed to query device attributes" << std::endl;
        return false;
    }
    
    return true;
}

bool executor_tcp::get_device_attribute(int *value, CUdevice_attribute attribute) {
    *value = this->device_attributes[static_cast<int32_t>(attribute)];
    return true;
}

bool executor_tcp::execute(const char *kernel,
                dim3 dim_grid,
                dim3 dim_block,
                std::vector<kernel_argument> &args) {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "failed to open socket" << std::endl;
        return false;
    }

    if (connect(socket_fd, (sockaddr *) &exec_addr, sizeof(exec_addr)) < 0) {
        std::cerr << "failed to connect" << std::endl;
        return false;
    }

    using namespace gpuless::execution;
    flatbuffers::FlatBufferBuilder builder;
    auto fb_dim_grid = Dim3 {dim_grid.x, dim_grid.y, dim_grid.z};
    auto fb_dim_block = Dim3 {dim_block.x, dim_block.y, dim_block.z};
    std::vector<flatbuffers::Offset<KernelArgument>> fb_args;
    for (const auto &a : args) {
        auto fba = CreateKernelArgument(builder,
                                        builder.CreateString(a.id),
                                        a.flags,
                                        builder.CreateVector(a.buffer));
        fb_args.push_back(fba);
    }
    auto execution_request_msg = CreateProtocolMessage(
        builder,
        Message_ExecutionRequest,
        CreateExecutionRequest(builder,
                                builder.CreateString(kernel),
                                builder.CreateVector(this->cuda_bin),
                                builder.CreateVector(fb_args),
                                &fb_dim_grid,
                                &fb_dim_block).Union());
    builder.Finish(execution_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    std::vector<uint8_t> buffer = recv_buffer(socket_fd);
    auto execution_answer_msg = GetProtocolMessage(buffer.data());
    auto status = execution_answer_msg->message_as_ExecutionAnswer()->status();
    auto return_buffers = execution_answer_msg->message_as_ExecutionAnswer()->return_buffers();
    for (size_t i = 0; i < return_buffers->size(); i++) {
        auto rb = return_buffers->Get(i);
        for (auto &a : args) {
            if (a.id == rb->id()->str()) {
                const uint8_t *data = rb->buffer()->data();
                a.buffer.assign(data, data + rb->buffer()->size());
            }
        }
    }
    return status == Status_OK;
}

bool executor_tcp::deallocate() {
    int socket_fd;
    if ((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cerr << "failed to open socket" << std::endl;
        return false;
    }

    if (connect(socket_fd, (sockaddr *) &this->manager_addr, sizeof(manager_addr)) < 0) {
        std::cerr << "failed to connect" << std::endl;
        return false;
    }

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;
    auto deallocate_request_msg = CreateProtocolMessage(
        builder,
        Message_DeallocateRequest,
        CreateDeallocateRequest(builder, this->session_id).Union());
    builder.Finish(deallocate_request_msg);
    send_buffer(socket_fd, builder.GetBufferPointer(), builder.GetSize());

    std::vector<uint8_t> buffer = recv_buffer(socket_fd);;
    auto deallocate_confirm_msg = GetProtocolMessage(buffer.data());
    auto status = deallocate_confirm_msg->message_as_DeallocateConfirm()->status();
    this->session_id = -1;
    return status == Status_OK;

}

} // namespace executor
} // namespace gpuless
