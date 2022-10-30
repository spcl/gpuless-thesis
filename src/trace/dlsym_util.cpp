#include "dlsym_util.hpp"
#include <dlfcn.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

void *real_dlsym(void *handle, const char *symbol) {
    static auto internal_dlsym = (decltype(&dlsym))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    return (*internal_dlsym)(handle, symbol);
}
