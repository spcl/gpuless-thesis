#include "cuda_vdev.hpp"

// CudaVDev::CudaVDev() : memory(3) {}

void CudaVDev::initialize(std::vector<std::string> cuda_binaries, char *manager_ip,
                          short manager_port) {
    this->initialized = true;
    this->cuda_binaries = cuda_binaries;
    this->manager_ip = manager_ip;
    this->manager_port = manager_port;
}

CUfunction CudaVDev::next_cufunction_ptr() {
    CUfunction f = this->next_function_ptr;
    this->next_function_ptr = (CUfunction)(((uint64_t)f) + 1);
    return f;
}

// CUdeviceptr CudaVDev::next_cudevice_ptr() { return this->next_device_ptr++; }

cudaDeviceProp CudaVDev::getDeviceProperties() {
    int major = 0;
    this->executor.get_device_attribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int minor = 0;
    this->executor.get_device_attribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    cudaDeviceProp props {
        "libcuadapter", // char         name[256];                  /**< ASCII string identifying device */
        {0}, // cudaUUID_t   uuid;                       /**< 16-byte unique identifier */
        {0}, // char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
        0, // unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
        0, // size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
        0, // size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
        0, // int          regsPerBlock;               /**< 32-bit registers available per block */
        0, // int          warpSize;                   /**< Warp size in threads */
        0, // size_t       memPitch;                   /**< Maximum pitch in bytes allowed by memory copies */
        0, // int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
        {0}, // int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
        {0}, // int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
        0, // int          clockRate;                  /**< Clock frequency in kilohertz */
        0, // size_t       totalConstMem;              /**< Constant memory available on device in bytes */
        major, // int          major;                      /**< Major compute capability */
        minor, // int          minor;                      /**< Minor compute capability */
        0, // size_t       textureAlignment;           /**< Alignment requirement for textures */
        0, // size_t       texturePitchAlignment;      /**< Pitch alignment requirement for texture references bound to pitched memory */
        0, // int          deviceOverlap;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
        0, // int          multiProcessorCount;        /**< Number of multiprocessors on device */
        0, // int          kernelExecTimeoutEnabled;   /**< Specified whether there is a run time limit on kernels */
        0, // int          integrated;                 /**< Device is integrated as opposed to discrete */
        0, // int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
        0, // int          computeMode;                /**< Compute mode (See ::cudaComputeMode) */
        0, // int          maxTexture1D;               /**< Maximum 1D texture size */
        0, // int          maxTexture1DMipmap;         /**< Maximum 1D mipmapped texture size */
        0, // int          maxTexture1DLinear;         /**< Deprecated, do not use. Use cudaDeviceGetTexture1DLinearMaxWidth() or cuDeviceGetTexture1DLinearMaxWidth() instead. */
        {0}, // int          maxTexture2D[2];            /**< Maximum 2D texture dimensions */
        {0}, // int          maxTexture2DMipmap[2];      /**< Maximum 2D mipmapped texture dimensions */
        {0}, // int          maxTexture2DLinear[3];      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
        {0}, // int          maxTexture2DGather[2];      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
        {0}, // int          maxTexture3D[3];            /**< Maximum 3D texture dimensions */
        {0}, // int          maxTexture3DAlt[3];         /**< Maximum alternate 3D texture dimensions */
        0, // int          maxTextureCubemap;          /**< Maximum Cubemap texture dimensions */
        {0}, // int          maxTexture1DLayered[2];     /**< Maximum 1D layered texture dimensions */
        {0}, // int          maxTexture2DLayered[3];     /**< Maximum 2D layered texture dimensions */
        {0}, // int          maxTextureCubemapLayered[2];/**< Maximum Cubemap layered texture dimensions */
        0, // int          maxSurface1D;               /**< Maximum 1D surface size */
        {0}, // int          maxSurface2D[2];            /**< Maximum 2D surface dimensions */
        {0}, // int          maxSurface3D[3];            /**< Maximum 3D surface dimensions */
        {0}, // int          maxSurface1DLayered[2];     /**< Maximum 1D layered surface dimensions */
        {0}, // int          maxSurface2DLayered[3];     /**< Maximum 2D layered surface dimensions */
        0, // int          maxSurfaceCubemap;          /**< Maximum Cubemap surface dimensions */
        {0}, // int          maxSurfaceCubemapLayered[2];/**< Maximum Cubemap layered surface dimensions */
        0, // size_t       surfaceAlignment;           /**< Alignment requirements for surfaces */
        0, // int          concurrentKernels;          /**< Device can possibly execute multiple kernels concurrently */
        0, // int          ECCEnabled;                 /**< Device has ECC support enabled */
        0, // int          pciBusID;                   /**< PCI bus ID of the device */
        0, // int          pciDeviceID;                /**< PCI device ID of the device */
        0, // int          pciDomainID;                /**< PCI domain ID of the device */
        0, // int          tccDriver;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
        0, // int          asyncEngineCount;           /**< Number of asynchronous engines */
        0, // int          unifiedAddressing;          /**< Device shares a unified address space with the host */
        0, // int          memoryClockRate;            /**< Peak memory clock frequency in kilohertz */
        0, // int          memoryBusWidth;             /**< Global memory bus width in bits */
        0, // int          l2CacheSize;                /**< Size of L2 cache in bytes */
        0, // int          persistingL2CacheMaxSize;   /**< Device's maximum l2 persisting lines capacity setting in bytes */
        0, // int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
        0, // int          streamPrioritiesSupported;  /**< Device supports stream priorities */
        0, // int          globalL1CacheSupported;     /**< Device supports caching globals in L1 */
        0, // int          localL1CacheSupported;      /**< Device supports caching locals in L1 */
        0, // size_t       sharedMemPerMultiprocessor; /**< Shared memory available per multiprocessor in bytes */
        0, // int          regsPerMultiprocessor;      /**< 32-bit registers available per multiprocessor */
        0, // int          managedMemory;              /**< Device supports allocating managed memory on this system */
        0, // int          isMultiGpuBoard;            /**< Device is on a multi-GPU board */
        0, // int          multiGpuBoardGroupID;       /**< Unique identifier for a group of devices on the same multi-GPU board */
        0, // int          hostNativeAtomicSupported;  /**< Link between the device and the host supports native atomic operations */
        0, // int          singleToDoublePrecisionPerfRatio; /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
        0, // int          pageableMemoryAccess;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
        0, // int          concurrentManagedAccess;    /**< Device can coherently access managed memory concurrently with the CPU */
        0, // int          computePreemptionSupported; /**< Device supports Compute Preemption */
        0, // int          canUseHostPointerForRegisteredMem; /**< Device can access host registered memory at the same virtual address as the CPU */
        0, // int          cooperativeLaunch;          /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
        0, // int          cooperativeMultiDeviceLaunch; /**< Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated. */
        0, // size_t       sharedMemPerBlockOptin;     /**< Per device maximum shared memory per block usable by special opt in */
        0, // int          pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
        0, // int          directManagedMemAccessFromHost; /**< Host can directly access managed memory on the device without migration. */
        0, // int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
        0, // int          accessPolicyMaxWindowSize;  /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
        0, // size_t       reservedSharedMemPerBlock;  /**< Shared memory reserved by CUDA driver per block in bytes */
    };
    return props;
}

std::vector<uint8_t> *CudaVDev::buffer_from_ptr(CUdeviceptr ptr) {
    auto vdev_memory_it = this->device_buffers.find(ptr);
    if (vdev_memory_it != this->device_buffers.end()) {
        return &(vdev_memory_it->second);
    }
    std::cerr << "buffer not found" << std::endl;
    return nullptr;
}

void *CudaVDev::memAlloc(size_t size) {
    this->device_buffers.emplace(
        std::make_pair(this->next_cudevice_ptr, std::vector<uint8_t>(size)));
    return (void *)(this->next_cudevice_ptr++);
}

void CudaVDev::memCpyHtoD(CUdeviceptr dst, void *src, size_t size) {
    auto vdev_memory_it = this->device_buffers.find(dst);
    if (vdev_memory_it != this->device_buffers.end()) {
        std::memcpy(vdev_memory_it->second.data(), src, size);
    }
}

void CudaVDev::memCpyDtoH(void *dst, CUdeviceptr src, size_t size) {
    auto vdev_memory_it = this->device_buffers.find(src);
    if (vdev_memory_it != this->device_buffers.end()) {
        std::memcpy(dst, vdev_memory_it->second.data(), size);
    }
}
