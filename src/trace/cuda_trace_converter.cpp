#include "cuda_trace_converter.hpp"
#include "cuda_trace.hpp"
#include <spdlog/spdlog.h>

namespace gpuless {

void CudaTraceConverter::traceToExecRequest(
    CudaTrace &cuda_trace, flatbuffers::FlatBufferBuilder &builder) {
    std::vector<flatbuffers::Offset<FBCudaApiCall>> fb_call_trace;
    for (auto &c : cuda_trace.callStack()) {
        spdlog::debug("Serializing api call: {}", c->typeName());
        c->appendToFBCudaApiCallList(builder, fb_call_trace);
    }

    std::vector<flatbuffers::Offset<FBNewModule>> fb_new_modules;
    std::vector<flatbuffers::Offset<FBNewFunction>> fb_new_functions;

    std::set<uint64_t> required_modules;
    std::set<std::string> required_functions;
    for (auto &apiCall : cuda_trace.callStack()) {
        auto rmod_vec = apiCall->requiredCudaModuleIds();
        required_modules.insert(rmod_vec.begin(), rmod_vec.end());
        auto rfunc_vec = apiCall->requiredFunctionSymbols();
        required_functions.insert(rfunc_vec.begin(), rfunc_vec.end());
    }

    for (const auto &rmod_id : required_modules) {
        auto it = cuda_trace.getModuleIdToFatbinResource().find(rmod_id);
        if (it == cuda_trace.getModuleIdToFatbinResource().end()) {
            spdlog::error("Required module {} unknown");
        }

        const void *resource_ptr = std::get<0>(it->second);
        uint64_t size = std::get<1>(it->second);
        bool is_loaded = std::get<2>(it->second);

        if (!is_loaded) {
            std::vector<uint8_t> buffer(size);
            std::memcpy(buffer.data(), resource_ptr, size);
            fb_new_modules.push_back(CreateFBNewModule(
                builder, builder.CreateVector(buffer), rmod_id));
            std::get<2>(it->second) = true;
        }
    }

    for (const auto &rfunc : required_functions) {
        auto it = cuda_trace.getSymbolToModuleId().find(rfunc);
        if (it == cuda_trace.getSymbolToModuleId().end()) {
            spdlog::error("Required function {} unknown");
        }

        uint64_t module_id = std::get<0>(it->second);
        bool fn_is_loaded = std::get<1>(it->second);

        if (!fn_is_loaded) {
            auto mod_it =
                cuda_trace.getModuleIdToFatbinResource().find(module_id);
            if (mod_it == cuda_trace.getModuleIdToFatbinResource().end()) {
                spdlog::error("Unknown module {} for function", module_id,
                              rfunc);
            }

            bool module_is_loaded = std::get<2>(mod_it->second);
            if (!module_is_loaded) {
                spdlog::error("Module {} not previously loaded", module_id);
            }

            fb_new_functions.push_back(CreateFBNewFunction(
                builder, builder.CreateString(rfunc), module_id));
            std::get<1>(it->second) = true;
        }
    }

    auto fb_exec_request =
        CreateFBTraceExecRequest(builder, builder.CreateVector(fb_call_trace),
                                 builder.CreateVector(fb_new_modules),
                                 builder.CreateVector(fb_new_functions));

    auto fb_protocol_message = CreateFBProtocolMessage(
        builder, FBMessage_FBTraceExecRequest, fb_exec_request.Union());
    builder.Finish(fb_protocol_message);
}

std::shared_ptr<CudaApiCall> CudaTraceConverter::fbCudaApiCallDeserialize(
    const FBCudaApiCall *fb_cuda_api_call) {
    std::shared_ptr<CudaApiCall> cuda_api_call;

    switch (fb_cuda_api_call->api_call_type()) {
    case FBCudaApiCallUnion_FBCudaMalloc: {
        auto c = fb_cuda_api_call->api_call_as_FBCudaMalloc();
        auto p = std::make_shared<CudaMalloc>(c->size());
        p->devPtr = reinterpret_cast<void *>(c->dev_ptr());
        cuda_api_call = p;
        break;
    }
    case FBCudaApiCallUnion_FBCudaMemcpyH2D: {
        auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyH2D();
        //        spdlog::info("{:x} {:x} {:x}", c->dst(), c->src(), c->size());
        auto p = std::make_shared<CudaMemcpyH2D>(
            reinterpret_cast<void *>(c->dst()),
            reinterpret_cast<void *>(c->src()), c->size());

        std::memcpy(p->buffer.data(), c->buffer()->data(), c->buffer()->size());
        cuda_api_call = p;
        break;
    }
    case FBCudaApiCallUnion_FBCudaMemcpyD2H: {
        auto c = fb_cuda_api_call->api_call_as_FBCudaMemcpyD2H();
        auto p = std::make_shared<CudaMemcpyD2H>(
            reinterpret_cast<void *>(c->dst()),
            reinterpret_cast<void *>(c->src()), c->size());
        std::memcpy(p->buffer.data(), c->buffer()->data(), c->buffer()->size());
        cuda_api_call = p;
        break;
    }
    case FBCudaApiCallUnion_FBCudaLaunchKernel: {
        auto c = fb_cuda_api_call->api_call_as_FBCudaLaunchKernel();
        const FBDim3 *fb_grid_dim = c->grid_dim();
        const FBDim3 *fb_block_dim = c->block_dim();
        const dim3 grid_dim{static_cast<unsigned int>(fb_grid_dim->x()),
                            static_cast<unsigned int>(fb_grid_dim->y()),
                            static_cast<unsigned int>(fb_grid_dim->z())};
        const dim3 block_dim{static_cast<unsigned int>(fb_block_dim->x()),
                             static_cast<unsigned int>(fb_block_dim->y()),
                             static_cast<unsigned int>(fb_block_dim->z())};
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(c->stream());

        std::vector<std::vector<uint8_t>> pb;
        for (const auto &b : *c->param_buffers()) {
            pb.emplace_back(b->buffer()->size());
            std::memcpy(pb.back().data(), b->buffer()->data(),
                        b->buffer()->size());
        }

        std::vector<KParamInfo> kpi;
        for (const auto &i : *c->param_infos()) {
            KParamInfo info{i->name()->str(),
                            static_cast<PtxParameterType>(i->ptx_param_type()),
                            static_cast<int>(i->type_size()),
                            static_cast<int>(i->align()),
                            static_cast<int>(i->size())};
            kpi.push_back(info);
        }

        auto p = std::make_shared<CudaLaunchKernel>(
            c->symbol()->str(), std::vector<uint64_t>(),
            std::vector<std::string>(), nullptr, grid_dim, block_dim,
            c->shared_mem(), stream, pb, kpi);
        cuda_api_call = p;
        break;
    }
    case FBCudaApiCallUnion_FBCudaFree: {
        auto c = fb_cuda_api_call->api_call_as_FBCudaFree();
        cuda_api_call =
            std::make_shared<CudaFree>(reinterpret_cast<void *>(c->dev_ptr()));
        break;
    }
    case FBCudaApiCallUnion_NONE:
        spdlog::error("FBCudaApiCallUnion should not be of type NONE");
        break;
    }

    return cuda_api_call;
}

std::shared_ptr<CudaApiCall> CudaTraceConverter::execResponseToTopApiCall(
    const FBTraceExecResponse *fb_trace_exec_response) {
    return CudaTraceConverter::fbCudaApiCallDeserialize(
        fb_trace_exec_response->trace_top());
}

std::vector<std::shared_ptr<CudaApiCall>>
CudaTraceConverter::execRequestToTrace(
    const FBTraceExecRequest *fb_trace_exec_request) {
    std::vector<std::shared_ptr<CudaApiCall>> cuda_api_calls;

    for (const auto &c : *fb_trace_exec_request->trace()) {
        cuda_api_calls.push_back(
            CudaTraceConverter::fbCudaApiCallDeserialize(c));
    }

    return cuda_api_calls;
}

} // namespace gpuless
