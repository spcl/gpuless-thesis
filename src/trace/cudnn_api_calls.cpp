#include "cudnn_api_calls.hpp"
#include "dlsym_util.hpp"
#include <cudnn.h>
#include <dlfcn.h>

#include "libgpuless.hpp"
#include <utility>

std::string gpuless::CudaCudnnApiCall::nativeErrorToString(uint64_t err) {
    auto str =
        "[cudnn] " +
        std::string(cudnnGetErrorString(static_cast<cudnnStatus_t>(err)));
    return str;
}

/*
 * cudnnCreate
 */
uint64_t gpuless::CudnnCreate::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnCreate))real_dlsym(RTLD_NEXT, "cudnnCreate");
    if (vdev.cudnn_handles_virtual_to_real.size() < this->virtual_handle + 1) {
        vdev.cudnn_handles_virtual_to_real.resize(this->virtual_handle + 1);
    }
    return real(&vdev.cudnn_handles_virtual_to_real[this->virtual_handle]);
}

gpuless::CudnnCreate::CudnnCreate(uint64_t virtualHandle)
    : virtual_handle(virtualHandle) {}

/*
 * cudnnSetStream
 */
uint64_t gpuless::CudnnSetStream::executeNative(CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnSetStream))real_dlsym(RTLD_NEXT, "cudnnSetStream");
    return real(vdev.cudnn_handles_virtual_to_real[this->virtual_handle],
                this->stream);
}

gpuless::CudnnSetStream::CudnnSetStream(uint64_t virtualHandle,
                                        cudaStream_t stream)
    : virtual_handle(virtualHandle), stream(stream) {}

/*
 * cudnnCreateTensorDescriptor
 */
uint64_t
gpuless::CudnnCreateTensorDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateTensorDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateTensorDescriptor");
    if (vdev.cudnn_tensor_descriptor_virtual_to_real.size() <
        this->virtual_td + 1) {
        vdev.cudnn_tensor_descriptor_virtual_to_real.resize(this->virtual_td +
                                                            1);
    }
    return real(
        &vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td]);
}

gpuless::CudnnCreateTensorDescriptor::CudnnCreateTensorDescriptor(
    uint64_t virtualTd)
    : virtual_td(virtualTd) {}

/*
 * cudnnSetTensorNdDescriptor
 */
uint64_t
gpuless::CudnnSetTensorNdDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetTensorNdDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnSetTensorNdDescriptor");
    return real(vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td],
                this->data_type, this->nb_dims, this->dim_a.data(),
                this->stride_a.data());
}

gpuless::CudnnSetTensorNdDescriptor::CudnnSetTensorNdDescriptor(
    uint64_t virtualTd, cudnnDataType_t dataType, int nbDims,
    std::vector<int> dimA, std::vector<int> strideA)
    : virtual_td(virtualTd), data_type(dataType), nb_dims(nbDims),
      dim_a(std::move(dimA)), stride_a(std::move(strideA)) {}

/*
 * cudnnCreateFilterDescriptor
 */
uint64_t
gpuless::CudnnCreateFilterDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateFilterDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateFilterDescriptor");
    if (vdev.cudnn_filter_descriptor_virtual_to_real.size() <
        this->virtual_fd + 1) {
        vdev.cudnn_filter_descriptor_virtual_to_real.resize(this->virtual_fd +
                                                            1);
    }
    return real(
        &vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd]);
}

gpuless::CudnnCreateFilterDescriptor::CudnnCreateFilterDescriptor(
    uint64_t virtualFd)
    : virtual_fd(virtualFd) {}

/*
 * cudnnSetFilterNdDescriptor
 */
uint64_t
gpuless::CudnnSetFilterNdDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetFilterNdDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnSetFilterNdDescriptor");
    return real(vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd],
                this->data_type, this->format, this->nb_dims,
                this->filter_dim_a.data());
}

gpuless::CudnnSetFilterNdDescriptor::CudnnSetFilterNdDescriptor(
    uint64_t virtualFd, cudnnDataType_t dataType, cudnnTensorFormat_t format,
    int nbDims, const std::vector<int> &filterDimA)
    : virtual_fd(virtualFd), data_type(dataType), format(format),
      nb_dims(nbDims), filter_dim_a(filterDimA) {}

/*
 * cudnnCreateConvolutionDescriptor
 */
uint64_t gpuless::CudnnCreateConvolutionDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnCreateConvolutionDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnCreateConvolutionDescriptor");
    if (vdev.cudnn_convolution_descriptor_virtual_to_real.size() <
        this->virtual_cd + 1) {
        vdev.cudnn_convolution_descriptor_virtual_to_real.resize(
            this->virtual_cd + 1);
    }
    return real(
        &vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd]);
}

gpuless::CudnnCreateConvolutionDescriptor::CudnnCreateConvolutionDescriptor(
    uint64_t virtualCd)
    : virtual_cd(virtualCd) {}

/*
 * cudnnSetConvolutionGroupCount
 */
uint64_t
gpuless::CudnnSetConvolutionGroupCount::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetConvolutionGroupCount))real_dlsym(
        RTLD_NEXT, "cudnnSetConvolutionGroupCount");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->group_count);
}

gpuless::CudnnSetConvolutionGroupCount::CudnnSetConvolutionGroupCount(
    uint64_t virtualCd, int groupCount)
    : virtual_cd(virtualCd), group_count(groupCount) {}

/*
 * cudnnSetConvolutionMathType
 */
uint64_t
gpuless::CudnnSetConvolutionMathType::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetConvolutionMathType))real_dlsym(
        RTLD_NEXT, "cudnnSetConvolutionMathType");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->math_type);
}

gpuless::CudnnSetConvolutionMathType::CudnnSetConvolutionMathType(
    uint64_t virtualCd, cudnnMathType_t mathType)
    : virtual_cd(virtualCd), math_type(mathType) {}

/*
 * cudnnSetConvolutionNdDescriptor
 */
uint64_t gpuless::CudnnSetConvolutionNdDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnSetConvolutionNdDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnSetConvolutionNdDescriptor");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->array_length, this->pad_a.data(), this->filter_stride_a.data(),
        this->dilation.data(), this->convolution_mode, this->cudnn_data_type);
}

gpuless::CudnnSetConvolutionNdDescriptor::CudnnSetConvolutionNdDescriptor(
    uint64_t virtualCd, int arrayLength, std::vector<int> padA,
    std::vector<int> filterStrideA, std::vector<int> dilation,
    cudnnConvolutionMode_t convolutionMode, cudnnDataType_t cudnnDataType)
    : virtual_cd(virtualCd), array_length(arrayLength), pad_a(std::move(padA)),
      filter_stride_a(std::move(filterStrideA)), dilation(std::move(dilation)),
      convolution_mode(convolutionMode), cudnn_data_type(cudnnDataType) {}

/*
 * cudnnGetConvolutionForwardAlgorithm_v7
 */
uint64_t gpuless::CudnnGetConvolutionForwardAlgorithm_v7::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnGetConvolutionForwardAlgorithm_v7))real_dlsym(
            RTLD_NEXT, "cudnnGetConvolutionForwardAlgorithm_v7");
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle],
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd],
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc],
        this->requested_algo_count, &this->returned_algo_count,
        this->perf_results.data());
}

gpuless::CudnnGetConvolutionForwardAlgorithm_v7::
    CudnnGetConvolutionForwardAlgorithm_v7(uint64_t virtualHandle,
                                           uint64_t virtualTdXdesc,
                                           uint64_t virtualTdYdesc,
                                           uint64_t virtualFd,
                                           uint64_t virtualCd,
                                           int requestedAlgoCount)
    : virtual_handle(virtualHandle), virtual_td_xdesc(virtualTdXdesc),
      virtual_td_ydesc(virtualTdYdesc), virtual_fd(virtualFd),
      virtual_cd(virtualCd), requested_algo_count(requestedAlgoCount),
      perf_results(requestedAlgoCount) {}

/*
 * cudnnConvolutionForward
 */
uint64_t
gpuless::CudnnConvolutionForward::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnConvolutionForward))real_dlsym(
        RTLD_NEXT, "cudnnConvolutionForward");
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle],
        this->alpha.data(),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        this->x,
        vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd_wdesc],
        this->w,
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd],
        this->algo, this->workspace, this->workspace_size_in_bytes,
        this->beta.data(),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc],
        this->y);
}

gpuless::CudnnConvolutionForward::CudnnConvolutionForward(
    uint64_t virtualHandle, size_t scaling_size, const void *alpha_ptr,
    const void *beta_ptr, void *workspace, size_t workspaceSizeInBytes,
    uint64_t virtualCd, cudnnConvolutionFwdAlgo_t algo, uint64_t virtualFdWdesc,
    const void *w, uint64_t virtualTdXdesc, const void *x,
    uint64_t virtualTdYdesc, void *y)
    : virtual_handle(virtualHandle), alpha(scaling_size), beta(scaling_size),
      workspace(workspace), workspace_size_in_bytes(workspaceSizeInBytes),
      virtual_cd(virtualCd), algo(algo), virtual_fd_wdesc(virtualFdWdesc), w(w),
      virtual_td_xdesc(virtualTdXdesc), x(x), virtual_td_ydesc(virtualTdYdesc),
      y(y) {
    std::memcpy(this->alpha.data(), alpha_ptr, scaling_size);
    std::memcpy(this->beta.data(), beta_ptr, scaling_size);
}

/*
 * cudnnBatchNormalizationForwardInference
 */
uint64_t gpuless::CudnnBatchNormalizationForwardInference::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real =
        (decltype(&cudnnBatchNormalizationForwardInference))real_dlsym(
            RTLD_NEXT, "cudnnBatchNormalizationForwardInference");
    return real(
        vdev.cudnn_handles_virtual_to_real[this->virtual_handle], this->mode,
        this->alpha.data(), this->beta.data(),
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_xdesc],
        this->x,
        vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td_ydesc],
        this->y,
        vdev.cudnn_tensor_descriptor_virtual_to_real
            [this->virtual_td_bs_scale_bias_mean_var_desc],
        this->bn_scale, this->bn_bias, this->estimated_mean,
        this->estimated_variance, this->epsilon);
}

gpuless::CudnnBatchNormalizationForwardInference::
    CudnnBatchNormalizationForwardInference(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode, size_t scaling_size,
        const void *alpha_ptr, const void *beta_ptr, uint64_t virtualTdXdesc,
        const void *x, uint64_t virtualTdYdesc, void *y,
        uint64_t virtualTdBsScaleBiasMeanVarDesc, const void *bnScale,
        const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon)
    : virtual_handle(virtualHandle), mode(mode), alpha(scaling_size), beta(scaling_size),
      virtual_td_xdesc(virtualTdXdesc), x(x), virtual_td_ydesc(virtualTdYdesc),
      y(y),
      virtual_td_bs_scale_bias_mean_var_desc(virtualTdBsScaleBiasMeanVarDesc),
      bn_scale(bnScale), bn_bias(bnBias), estimated_mean(estimatedMean),
      estimated_variance(estimatedVariance), epsilon(epsilon) {
    std::memcpy(this->alpha.data(), alpha_ptr, scaling_size);
    std::memcpy(this->beta.data(), beta_ptr, scaling_size);
}

/*
 * cudnnDestroyConvolutionDescriptor
 */
gpuless::CudnnDestroyConvolutionDescriptor::CudnnDestroyConvolutionDescriptor(
    uint64_t virtualCd)
    : virtual_cd(virtualCd) {}

uint64_t gpuless::CudnnDestroyConvolutionDescriptor::executeNative(
    CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnDestroyConvolutionDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnDestroyConvolutionDescriptor");
    return real(
        vdev.cudnn_convolution_descriptor_virtual_to_real[this->virtual_cd]);
}

/*
 * cudnnDestroyFilterDescriptor
 */
gpuless::CudnnDestroyFilterDescriptor::CudnnDestroyFilterDescriptor(
    uint64_t virtualFd)
    : virtual_fd(virtualFd) {}

uint64_t
gpuless::CudnnDestroyFilterDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = (decltype(&cudnnDestroyFilterDescriptor))real_dlsym(
        RTLD_NEXT, "cudnnDestroyFilterDescriptor");
    return real(vdev.cudnn_filter_descriptor_virtual_to_real[this->virtual_fd]);
}

/*
 * cudnnDestroyTensorDescriptor
 */
gpuless::CudnnDestroyTensorDescriptor::CudnnDestroyTensorDescriptor(
    uint64_t virtualTd)
    : virtual_td(virtualTd) {}

uint64_t
gpuless::CudnnDestroyTensorDescriptor::executeNative(CudaVirtualDevice &vdev) {
    static auto real = GET_REAL_FUNCTION(cudnnDestroyTensorDescriptor);
    return real(vdev.cudnn_tensor_descriptor_virtual_to_real[this->virtual_td]);
}
