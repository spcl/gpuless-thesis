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
    this->synchronize_counter_++;
    spdlog::info(
        "TraceExecutorLocal::synchronize() [synchronize_counter={}, size={}]",
        this->synchronize_counter_, cuda_trace.callStack().size());

    auto &vdev = this->cuda_virtual_device_;
    this->cuda_virtual_device_.initRealDevice();

    // synchronize the execution side virtual device with new mappings
    // from the client
    vdev.symbol_to_module_id_map.insert(
        cuda_trace.getNewSymbolToModuleId().begin(),
        cuda_trace.getNewSymbolToModuleId().end());
    cuda_trace.getNewSymbolToModuleId().clear();

    vdev.global_var_to_module_id_map.insert(
        cuda_trace.getNewGlobalVarToModuleId().begin(),
        cuda_trace.getNewGlobalVarToModuleId().end());
    cuda_trace.getNewGlobalVarToModuleId().clear();

    vdev.module_id_to_fatbin_data_map.insert(
        cuda_trace.getNewModuleIdToFatbinDataMap().begin(),
        cuda_trace.getNewModuleIdToFatbinDataMap().end());
    cuda_trace.getNewModuleIdToFatbinDataMap().clear();

    // load modules and functions that are not loaded yet but required by
    // the given trace
    for (auto &apiCall : cuda_trace.callStack()) {
        std::vector<uint64_t> required_modules =
            apiCall->requiredCudaModuleIds();
        std::vector<std::string> required_functions =
            apiCall->requiredFunctionSymbols();

        for (auto id : required_modules) {
            spdlog::debug("Required module: {}", id);
            auto mod_reg_it = vdev.module_registry_.find(id);
            if (mod_reg_it == vdev.module_registry_.end()) {
                auto mod_data_it = vdev.module_id_to_fatbin_data_map.find(id);
                if (mod_data_it == vdev.module_id_to_fatbin_data_map.end()) {
                    spdlog::error("fatbin data missing for module: {}", id);
                    std::exit(EXIT_FAILURE);
                }

                spdlog::debug("Loading module: {} [size={}]", id,
                              mod_data_it->second.size());

                void *fatbin_data_ptr = mod_data_it->second.data();
                CUmodule mod;
                checkCudaErrors(cuModuleLoadData(&mod, fatbin_data_ptr));
//                checkCudaErrors(cuModuleLoadFatBinary(&mod, fatbin_data_ptr));
                vdev.module_registry_.emplace(id, mod);
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
        if ((err = apiCall->executeNative(vdev)) != cudaSuccess) {
            spdlog::error("Failed to execute call trace: {} ({})",
                          cudaGetErrorString(err), err);
            std::exit(EXIT_FAILURE);
        }

        // tmp
        // static auto sync =
        //     (decltype(&cudaStreamSynchronize))real_dlsym(
        //             RTLD_NEXT, "cudaStreamSynchronize");
        // if ((err = sync(0)) != cudaSuccess) {
        //     std::cerr << "    failed to execute call trace: "
        //         << cudaGetErrorString(err) << " (" << err << ")"
        //         << std::endl;
        //     std::exit(EXIT_FAILURE);
        // }
    }

    cuda_trace.markSynchronized();
    return true;
}

bool TraceExecutorLocal::deallocate() { return false; }

} // namespace gpuless
