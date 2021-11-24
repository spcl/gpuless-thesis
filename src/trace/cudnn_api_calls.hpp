#ifndef GPULESS_CUDNN_API_CALLS_HPP
#define GPULESS_CUDNN_API_CALLS_HPP

#include "cuda_api_calls.hpp"

namespace gpuless {

class CudaCudnnApiCall : public AbstractCudaApiCall {
  public:
    std::string nativeErrorToString(uint64_t err) override;
};

class CudnnCreate : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;

    explicit CudnnCreate(uint64_t virtualHandle);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetStream : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudaStream_t stream;

    CudnnSetStream(uint64_t virtualHandle, cudaStream_t stream);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnCreateTensorDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;

    explicit CudnnCreateTensorDescriptor(uint64_t virtualTd);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetTensorNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;
    cudnnDataType_t data_type;
    int nb_dims;
    std::vector<int> dim_a;
    std::vector<int> stride_a;

    CudnnSetTensorNdDescriptor(uint64_t virtualTd, cudnnDataType_t dataType,
                               int nbDims, std::vector<int> dimA,
                               std::vector<int> strideA);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnCreateFilterDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;

    explicit CudnnCreateFilterDescriptor(uint64_t virtualFd);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetFilterNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;
    cudnnDataType_t data_type;
    cudnnTensorFormat_t format;
    int nb_dims;
    std::vector<int> filter_dim_a;

    CudnnSetFilterNdDescriptor(uint64_t virtualFd, cudnnDataType_t dataType,
                               cudnnTensorFormat_t format, int nbDims,
                               const std::vector<int> &filterDimA);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnCreateConvolutionDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;

    explicit CudnnCreateConvolutionDescriptor(uint64_t virtualCd);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetConvolutionGroupCount : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    int group_count;

    CudnnSetConvolutionGroupCount(uint64_t virtualCd, int groupCount);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetConvolutionMathType : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    cudnnMathType_t math_type;

    CudnnSetConvolutionMathType(uint64_t virtualCd, cudnnMathType_t mathType);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnSetConvolutionNdDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    int array_length;
    std::vector<int> pad_a;
    std::vector<int> filter_stride_a;
    std::vector<int> dilation;
    cudnnConvolutionMode_t convolution_mode;
    cudnnDataType_t cudnn_data_type;

    CudnnSetConvolutionNdDescriptor(uint64_t virtualCd, int arrayLength,
                                    std::vector<int> padA,
                                    std::vector<int> filterStrideA,
                                    std::vector<int> dilation,
                                    cudnnConvolutionMode_t convolutionMode,
                                    cudnnDataType_t cudnnDataType);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnGetConvolutionForwardAlgorithm_v7 : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    uint64_t virtual_td_xdesc;
    uint64_t virtual_td_ydesc;
    uint64_t virtual_fd;
    uint64_t virtual_cd;
    int requested_algo_count;

    // outputs
    int returned_algo_count{};
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results;

    CudnnGetConvolutionForwardAlgorithm_v7(uint64_t virtualHandle,
                                           uint64_t virtualTdXdesc,
                                           uint64_t virtualTdYdesc,
                                           uint64_t virtualFd,
                                           uint64_t virtualCd,
                                           int requestedAlgoCount);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnConvolutionForward : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    void *workspace;
    size_t workspace_size_in_bytes;
    uint64_t virtual_cd;
    cudnnConvolutionFwdAlgo_t algo;
    uint64_t virtual_fd_wdesc;
    const void *w;
    uint64_t virtual_td_xdesc;
    const void *x;
    uint64_t virtual_td_ydesc;
    void *y;

    CudnnConvolutionForward(uint64_t virtualHandle, size_t scaling_size,
                            const void *alpha_ptr, const void *beta_ptr,
                            void *workspace, size_t workspaceSizeInBytes,
                            uint64_t virtualCd, cudnnConvolutionFwdAlgo_t algo,
                            uint64_t virtualFdWdesc, const void *w,
                            uint64_t virtualTdXdesc, const void *x,
                            uint64_t virtualTdYdesc, void *y);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnBatchNormalizationForwardInference : public CudaCudnnApiCall {
  public:
    uint64_t virtual_handle;
    cudnnBatchNormMode_t mode;
    std::vector<uint8_t> alpha;
    std::vector<uint8_t> beta;
    uint64_t virtual_td_xdesc;
    const void *x;
    uint64_t virtual_td_ydesc;
    void *y;

    uint64_t virtual_td_bs_scale_bias_mean_var_desc;
    const void *bn_scale;
    const void *bn_bias;
    const void *estimated_mean;
    const void *estimated_variance;
    double epsilon;

    CudnnBatchNormalizationForwardInference(
        uint64_t virtualHandle, cudnnBatchNormMode_t mode, size_t scaling_size,
        const void *alpha_ptr, const void *beta_ptr, uint64_t virtualTdXdesc,
        const void *x, uint64_t virtualTdYdesc, void *y,
        uint64_t virtualTdBsScaleBiasMeanVarDesc, const void *bnScale,
        const void *bnBias, const void *estimatedMean,
        const void *estimatedVariance, double epsilon);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnDestroyConvolutionDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_cd;
    explicit CudnnDestroyConvolutionDescriptor(uint64_t virtualCd);

    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnDestroyFilterDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_fd;

    explicit CudnnDestroyFilterDescriptor(uint64_t virtualFd);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

class CudnnDestroyTensorDescriptor : public CudaCudnnApiCall {
  public:
    uint64_t virtual_td;
    CudnnDestroyTensorDescriptor(uint64_t virtualTd);
    uint64_t executeNative(CudaVirtualDevice &vdev) override;
};

} // namespace gpuless

#endif // GPULESS_CUDNN_API_CALLS_HPP
