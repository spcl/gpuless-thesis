#include "dlsym_util.hpp"
#include <dlfcn.h>

void *real_dlsym(void *handle, const char *symbol) {
    static auto internal_dlsym = (decltype(&dlsym))__libc_dlsym(
        __libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}
