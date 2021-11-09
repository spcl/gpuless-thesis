#include "trace_executor_local.hpp"
#include "dlsym_util.hpp"
#include <dlfcn.h>
#include <iostream>
#include <spdlog/spdlog.h>

namespace gpuless {

TraceExecutorLocal::TraceExecutorLocal() {}

TraceExecutorLocal::~TraceExecutorLocal() = default;

bool TraceExecutorLocal::init(const char *ip, const short port,
                              manager::instance_profile profile) {
    return true;
}

bool TraceExecutorLocal::synchronize(gpuless::CudaTrace &cuda_trace) {
    spdlog::info("TraceExecutorLocal::synchronize() [size={}]",
                 cuda_trace.callStack().size());

    auto &vdev = cuda_trace.cudaVirtualDevice();
    vdev.initRealDevice();

    // load modules and functions that are not loaded yet but required by the
    // given trace
    for (auto &apiCall : cuda_trace.callStack()) {
        std::vector<uint64_t> required_modules =
            apiCall->requiredCudaModuleIds();
        std::vector<std::string> required_functions =
            apiCall->requiredFunctionSymbols();

        for (auto id : required_modules) {
            spdlog::debug("Required module: {}", id);
            auto mod_reg_it = vdev.module_registry_.find(id);
            if (mod_reg_it == vdev.module_registry_.end()) {
                auto fatbin_wrapper_it =
                    vdev.module_id_to_fatbin_wrapper_map.find(id);
                if (fatbin_wrapper_it ==
                    vdev.module_id_to_fatbin_wrapper_map.end()) {
                    std::cerr << "fatbin wrapper missing for module: " << id
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                CUmodule mod;
                checkCudaErrors(
                    cuModuleLoadData(&mod, fatbin_wrapper_it->second.data));
                vdev.module_registry_.emplace(id, mod);
                spdlog::debug("Module loaded: {}", id);
            }
        }

        for (auto &fn : required_functions) {
            spdlog::debug("Required function: {}", fn);
            auto fn_reg_it = vdev.function_registry_.find(fn);
            if (fn_reg_it == vdev.function_registry_.end()) {
                auto mod_for_sym_it = vdev.symbol_to_module_id_map.find(fn);
                if (mod_for_sym_it == vdev.symbol_to_module_id_map.end()) {
                    std::cerr << "unknown module for symbol: " << fn
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                uint64_t mod_id = mod_for_sym_it->second;

                auto mod_reg_it = vdev.module_registry_.find(mod_id);
                if (mod_reg_it == vdev.module_registry_.end()) {
                    std::cerr << "module not previously registered: " << mod_id
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                CUfunction func;
                checkCudaErrors(
                    cuModuleGetFunction(&func, mod_reg_it->second, fn.c_str()));
                vdev.function_registry_.emplace(fn, func);
                spdlog::debug("Function loaded: {}", fn);
            }
        }
    }

    for (auto &apiCall : cuda_trace.callStack()) {
        spdlog::debug("Executing: {}", apiCall->typeName());
        cudaError_t err;
        if ((err = apiCall->executeNative(cuda_trace.cudaVirtualDevice())) !=
            cudaSuccess) {
            spdlog::error("Failed to execute call trace: {} ({})",
                          cudaGetErrorString(err), err);
            std::exit(EXIT_FAILURE);
        }

        //        // tmp
        //        static auto sync =
        //        (decltype(&cudaStreamSynchronize))real_dlsym(
        //            RTLD_NEXT, "cudaStreamSynchronize");
        //        if ((err = sync(0)) != cudaSuccess) {
        //            std::cerr << "    failed to execute call trace: "
        //                      << cudaGetErrorString(err) << " (" << err << ")"
        //                      << std::endl;
        //            std::exit(EXIT_FAILURE);
        //        }
    }

    cuda_trace.markSynchronized();
    return true;
}

bool TraceExecutorLocal::deallocate() { return false; }

} // namespace gpuless