#include "trace_executor_tcp_client.hpp"
#include "../TcpClient.hpp"
#include "../schemas/allocation_protocol_generated.h"
#include "cuda_trace_converter.hpp"
#include <spdlog/spdlog.h>

namespace gpuless {

static sockaddr_in iptoaddr(const char *ip, const short port) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) < 0) {
        SPDLOG_ERROR("Invalid IP address: {}", ip);
        throw "Invalid IP address";
    }
    return addr;
}

TraceExecutorTcp::TraceExecutorTcp(const char *ip, const short port,
                                   manager::instance_profile profile)
    : m_gpusession(negotiateSession(ip, port, profile)),
      m_manager_addr(iptoaddr(ip, port)) {
    getDeviceAttributes();
};

TraceExecutorTcp::~TraceExecutorTcp() {
    bool success = deallocate();
    if (!success) {
        SPDLOG_ERROR("Failed to deallocate session");
    } else {
        SPDLOG_INFO("Deallocated session");
    }
};

TcpGpuSession
TraceExecutorTcp::negotiateSession(const char *ip, const short port,
                                   gpuless::manager::instance_profile profile) {

    TcpClient tcp = TcpClient(ip, port);

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;

    // make initial request
    auto allocate_request_msg = CreateProtocolMessage(
        builder, Message_AllocateRequest,
        CreateAllocateRequest(builder, profile, -1).Union());
    builder.Finish(allocate_request_msg);
    tcp.send(builder.GetBufferPointer(), builder.GetSize());

    // manager offers some sessions
    std::vector<uint8_t> buffer_offer = tcp.recv();
    auto allocate_offer_msg = GetProtocolMessage(buffer_offer.data());
    auto offered_profiles =
        allocate_offer_msg->message_as_AllocateOffer()->available_profiles();
    int32_t selected_profile = offered_profiles->Get(0);
    int32_t session_id =
        allocate_offer_msg->message_as_AllocateOffer()->session_id();

    // choose a profile and send finalize request
    builder.Reset();
    auto allocate_select_msg = CreateProtocolMessage(
        builder, Message_AllocateSelect,
        CreateAllocateSelect(builder, Status_OK, session_id, selected_profile)
            .Union());
    builder.Finish(allocate_select_msg);
    tcp.send(builder.GetBufferPointer(), builder.GetSize());

    // get server confirmation
    std::vector<uint8_t> buffer_confirm = tcp.recv();
    auto allocate_confirm_msg = GetProtocolMessage(buffer_confirm.data());
    sockaddr_in addr = {};
    if (allocate_confirm_msg->message_as_AllocateConfirm()->status() ==
        Status_OK) {
        auto port = allocate_confirm_msg->message_as_AllocateConfirm()->port();
        auto ip = allocate_confirm_msg->message_as_AllocateConfirm()->ip();

        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr = *((struct in_addr *)&ip);
    } else {
        throw "Failed to negotiate Session";
    }

    // this->getDeviceAttributes();
    return TcpGpuSession(addr, session_id);
}

bool TraceExecutorTcp::deallocate() {

    TcpClient tcp = TcpClient(m_manager_addr);

    using namespace gpuless::manager;
    flatbuffers::FlatBufferBuilder builder;
    auto deallocate_request_msg = CreateProtocolMessage(
        builder, Message_DeallocateRequest,
        CreateDeallocateRequest(builder, m_gpusession.getSessionId()).Union());
    builder.Finish(deallocate_request_msg);
    tcp.send(builder.GetBufferPointer(), builder.GetSize());

    SPDLOG_DEBUG("Deallocate request sent");

    std::vector<uint8_t> buffer = tcp.recv();
    auto deallocate_confirm_msg = GetProtocolMessage(buffer.data());
    auto status =
        deallocate_confirm_msg->message_as_DeallocateConfirm()->status();
    return status == Status_OK;
}

bool TraceExecutorTcp::synchronize(CudaTrace &cuda_trace) {
    auto s = std::chrono::high_resolution_clock::now();

    // collect statistics on synchronizations
    this->synchronize_counter_++;
    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() [synchronize_counter={}, size={}]",
        this->synchronize_counter_, cuda_trace.callStack().size());

    // send trace execution request
    flatbuffers::FlatBufferBuilder builder;
    CudaTraceConverter::traceToExecRequest(cuda_trace, builder);
    m_gpusession.send(builder.GetBufferPointer(), builder.GetSize());
    SPDLOG_INFO("Trace execution request sent");

    // receive trace execution response
    std::vector<uint8_t> response_buffer = m_gpusession.recv();
    SPDLOG_INFO("Trace execution response received");
    auto fb_protocol_message_response =
        GetFBProtocolMessage(response_buffer.data());
    auto fb_trace_exec_response =
        fb_protocol_message_response->message_as_FBTraceExecResponse();
    auto cuda_api_call =
        CudaTraceConverter::execResponseToTopApiCall(fb_trace_exec_response);

    cuda_trace.markSynchronized();
    cuda_trace.setHistoryTop(cuda_api_call);

    auto e = std::chrono::high_resolution_clock::now();
    auto d =
        std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
        1000000.0;
    this->synchronize_total_time_ += d;

    SPDLOG_INFO(
        "TraceExecutorTcp::synchronize() successful [t={}s, total_time={}s]", d,
        this->synchronize_total_time_);
    return true;
}

bool TraceExecutorTcp::getDeviceAttributes() {
    SPDLOG_INFO("TraceExecutorTcp::getDeviceAttributes()");

    flatbuffers::FlatBufferBuilder builder;
    auto attr_request =
        CreateFBProtocolMessage(builder, FBMessage_FBTraceAttributeRequest,
                                CreateFBTraceAttributeRequest(builder).Union());
    builder.Finish(attr_request);
    m_gpusession.send(builder.GetBufferPointer(), builder.GetSize());
    SPDLOG_DEBUG("FBTraceAttributeRequest sent");

    std::vector<uint8_t> response_buffer = m_gpusession.recv();
    SPDLOG_DEBUG("FBTraceAttributeResponse received");

    auto fb_protocol_message_response =
        GetFBProtocolMessage(response_buffer.data());
    auto fb_trace_attribute_response =
        fb_protocol_message_response->message_as_FBTraceAttributeResponse();

    this->device_total_mem = fb_trace_attribute_response->total_mem();
    this->device_attributes.resize(CU_DEVICE_ATTRIBUTE_MAX);
    for (const auto &a : *fb_trace_attribute_response->device_attributes()) {
        int32_t value = a->value();
        auto dev_attr = static_cast<CUdevice_attribute>(a->device_attribute());
        this->device_attributes[dev_attr] = value;
    }

    return true;
}

double TraceExecutorTcp::getSynchronizeTotalTime() const {
    return synchronize_total_time_;
}

} // namespace gpuless