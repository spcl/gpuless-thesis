#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cuda.h>

namespace gpuless {

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
void __checkCudaErrors(CUresult r, const char *file, const int line) {
    if (r != CUDA_SUCCESS) {
        const char *msg;
        cuGetErrorName(r, &msg);
        std::cout << "cuda error in " << file << "(" << line << "):"
            << std::endl << msg << std::endl;
    }
}

} // namespace gpuless

#endif // __UTILS_HPP__
