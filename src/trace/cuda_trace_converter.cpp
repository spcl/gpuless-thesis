#include "cuda_trace_converter.hpp"
#include "cuda_trace.hpp"

namespace gpuless {

void CudaTraceConverter::traceToExecRequest(CudaTrace &trace,
                                  std::vector<uint8_t> &buffer) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<FBCudaApiCall>> fb_call_trace;
    for (auto &c : trace.callStack()) {
        c->appendToFBCudaApiCallList(builder, fb_call_trace);
    }

    auto fb_protocol_message = CreateFBProtocolMessage(
        builder, FBMessage_FBTraceExecRequest,
        CreateFBTraceExecRequest(builder, builder.CreateVector(fb_call_trace))
            .Union());
    builder.Finish(fb_protocol_message);
}

void CudaTraceConverter::execResponseToTrace(CudaTrace &trace,
                                    std::vector<uint8_t> &buffer) {
    auto fb_protocol_message = GetFBProtocolMessage(buffer.data());
    // TODO
}

} // namespace gpuless
