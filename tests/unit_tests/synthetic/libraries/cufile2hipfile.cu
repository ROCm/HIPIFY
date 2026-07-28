// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipfile.h"
#include "cufile.h"
// CHECK-NOT: #include "hipfile.h"

int main() {
  printf("27. cuFile API to hipFile API synthetic test\n");

  // CHECK: hipFileOpError_t fileOpError;
  CUfileOpError fileOpError;

  // CHECK: hipFileError fileError;
  // CHECK-NEXT: hipFileError_t fileError_t;
  CUfileError fileError;
  CUfileError_t fileError_t;

  // CHECK: hipFileDriverStatusFlags driverStatusFlags;
  // CHECK-NEXT: hipFileDriverStatusFlags_t driverStatusFlags_t;
  CUfileDriverStatusFlags driverStatusFlags;
  CUfileDriverStatusFlags_t driverStatusFlags_t;

  // CHECK: hipFileDriverControlFlags driverControlFlags;
  // CHECK-NEXT: hipFileDriverControlFlags_t driverControlFlags_t;
  CUfileDriverControlFlags driverControlFlags;
  CUfileDriverControlFlags_t driverControlFlags_t;

  // CHECK: hipFileFeatureFlags featureFlags;
  // CHECK-NEXT: hipFileFeatureFlags_t featureFlags_t;
  CUfileFeatureFlags featureFlags;
  CUfileFeatureFlags_t featureFlags_t;

  // CHECK: hipFileFileHandleType fileHandleType;
  CUfileFileHandleType fileHandleType;

  // CHECK: hipFileDriverProps driverProps;
  // CHECK-NEXT: hipFileDriverProps_t driverProps_t;
  CUfileDrvProps driverProps;
  CUfileDrvProps_t driverProps_t;

  // CHECK: hipFileRDMAInfo rdmaInfo;
  // CHECK-NEXT: hipFileRDMAInfo_t rdmaInfo_t;
  cufileRDMAInfo rdmaInfo;
  cufileRDMAInfo_t rdmaInfo_t;

  // CHECK: hipFileFSOps fsOps;
  // CHECK-NEXT: hipFileFSOps_t fsOps_t;
  CUfileFSOps fsOps;
  CUfileFSOps_t fsOps_t;

  // CHECK: sockaddr sockAddr;
  sockaddr_t sockAddr;

  // CHECK: hipFileDescr_t fileDescr;
  CUfileDescr_t fileDescr;

  // CHECK: hipFileHandle_t fileHandle;
  CUfileHandle_t fileHandle;

#if CUDA_VERSION >= 11060
  // CHECK: hipFileOpcode opcode;
  // CHECK-NEXT: hipFileOpcode_t opcode_t;
  CUfileOpcode opcode;
  CUfileOpcode_t opcode_t;

  // CHECK: hipFileStatus fileStatus;
  // CHECK: hipFileStatus_t fileStatus_t;
  CUFILEStatus_enum fileStatus;
  CUfileStatus_t fileStatus_t;

  // CHECK: hipFileBatchHandle_t batchHandle;
  CUfileBatchHandle_t batchHandle;

  // CHECK: hipFileBatchMode batchMode;
  // CHECK-NEXT: hipFileBatchMode_t batchMode_t;
  cufileBatchMode batchMode;
  CUfileBatchMode_t batchMode_t;

  // CHECK: hipFileIOParams ioParams;
  // CHECK-NEXT: hipFileIOParams_t ioParams_t;
  CUfileIOParams ioParams;
  CUfileIOParams_t ioParams_t;

  // CHECK: hipFileIOEvents ioEvents;
  // CHECK: hipFileIOEvents_t ioEvents_t;
  CUfileIOEvents ioEvents;
  CUfileIOEvents_t ioEvents_t;
#endif

  // CHECK: hipFileOpError_t FILE_SUCCESS = hipFileSuccess;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_NOT_INITIALIZED = hipFileDriverNotInitialized;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_INVALID_PROPS = hipFileDriverInvalidProps;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_UNSUPPORTED_LIMIT = hipFileDriverUnsupportedLimit;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_VERSION_MISMATCH = hipFileDriverVersionMismatch;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_VERSION_READ_ERROR = hipFileDriverVersionReadError;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_CLOSING = hipFileDriverClosing;
  CUfileOpError FILE_SUCCESS = CU_FILE_SUCCESS;
  CUfileOpError FILE_DRIVER_NOT_INITIALIZED = CU_FILE_DRIVER_NOT_INITIALIZED;
  CUfileOpError FILE_DRIVER_INVALID_PROPS = CU_FILE_DRIVER_INVALID_PROPS;
  CUfileOpError FILE_DRIVER_UNSUPPORTED_LIMIT = CU_FILE_DRIVER_UNSUPPORTED_LIMIT;
  CUfileOpError FILE_DRIVER_VERSION_MISMATCH = CU_FILE_DRIVER_VERSION_MISMATCH;
  CUfileOpError FILE_DRIVER_VERSION_READ_ERROR = CU_FILE_DRIVER_VERSION_READ_ERROR;
  CUfileOpError FILE_DRIVER_CLOSING = CU_FILE_DRIVER_CLOSING;

  // CHECK: hipFileOpError_t FILE_PLATFORM_NOT_SUPPORTED = hipFilePlatformNotSupported;
  // CHECK-NEXT: hipFileOpError_t FILE_IO_NOT_SUPPORTED = hipFileIONotSupported;
  // CHECK-NEXT: hipFileOpError_t FILE_DEVICE_NOT_SUPPORTED = hipFileDeviceNotSupported;
  CUfileOpError FILE_PLATFORM_NOT_SUPPORTED = CU_FILE_PLATFORM_NOT_SUPPORTED;
  CUfileOpError FILE_IO_NOT_SUPPORTED = CU_FILE_IO_NOT_SUPPORTED;
  CUfileOpError FILE_DEVICE_NOT_SUPPORTED = CU_FILE_DEVICE_NOT_SUPPORTED;

  // CHECK: hipFileOpError_t FILE_NVFS_DRIVER_ERROR = hipFileDriverError;
  // CHECK-NEXT: hipFileOpError_t FILE_CUDA_DRIVER_ERROR = hipFileHipDriverError;
  // CHECK-NEXT: hipFileOpError_t FILE_CUDA_POINTER_INVALID = hipFileHipPointerInvalid;
  // CHECK-NEXT: hipFileOpError_t FILE_CUDA_MEMORY_TYPE_INVALID = hipFileHipMemoryTypeInvalid;
  // CHECK-NEXT: hipFileOpError_t FILE_CUDA_POINTER_RANGE_ERROR = hipFileHipPointerRangeError;
  // CHECK-NEXT: hipFileOpError_t FILE_CUDA_CONTEXT_MISMATCH = hipFileHipContextMismatch;
  CUfileOpError FILE_NVFS_DRIVER_ERROR = CU_FILE_NVFS_DRIVER_ERROR;
  CUfileOpError FILE_CUDA_DRIVER_ERROR = CU_FILE_CUDA_DRIVER_ERROR;
  CUfileOpError FILE_CUDA_POINTER_INVALID = CU_FILE_CUDA_POINTER_INVALID;
  CUfileOpError FILE_CUDA_MEMORY_TYPE_INVALID = CU_FILE_CUDA_MEMORY_TYPE_INVALID;
  CUfileOpError FILE_CUDA_POINTER_RANGE_ERROR = CU_FILE_CUDA_POINTER_RANGE_ERROR;
  CUfileOpError FILE_CUDA_CONTEXT_MISMATCH = CU_FILE_CUDA_CONTEXT_MISMATCH;

  // CHECK: hipFileOpError_t FILE_INVALID_MAPPING_SIZE = hipFileInvalidMappingSize;
  // CHECK-NEXT: hipFileOpError_t FILE_INVALID_MAPPING_RANGE = hipFileInvalidMappingRange;
  // CHECK-NEXT: hipFileOpError_t FILE_INVALID_FILE_TYPE = hipFileInvalidFileType;
  // CHECK-NEXT: hipFileOpError_t FILE_INVALID_FILE_OPEN_FLAG = hipFileInvalidFileOpenFlag;
  // CHECK-NEXT: hipFileOpError_t FILE_DIO_NOT_SET = hipFileDIONotSet;
  // CHECK-NEXT: hipFileOpError_t FILE_INVALID_VALUE = hipFileInvalidValue;
  CUfileOpError FILE_INVALID_MAPPING_SIZE = CU_FILE_INVALID_MAPPING_SIZE;
  CUfileOpError FILE_INVALID_MAPPING_RANGE = CU_FILE_INVALID_MAPPING_RANGE;
  CUfileOpError FILE_INVALID_FILE_TYPE = CU_FILE_INVALID_FILE_TYPE;
  CUfileOpError FILE_INVALID_FILE_OPEN_FLAG = CU_FILE_INVALID_FILE_OPEN_FLAG;
  CUfileOpError FILE_DIO_NOT_SET = CU_FILE_DIO_NOT_SET;
  CUfileOpError FILE_INVALID_VALUE = CU_FILE_INVALID_VALUE;

  // CHECK: hipFileOpError_t FILE_MEMORY_ALREADY_REGISTERED = hipFileMemoryAlreadyRegistered;
  // CHECK-NEXT: hipFileOpError_t FILE_MEMORY_NOT_REGISTERED = hipFileMemoryNotRegistered;
  // CHECK-NEXT: hipFileOpError_t FILE_PERMISSION_DENIED = hipFilePermissionDenied;
  // CHECK-NEXT: hipFileOpError_t FILE_DRIVER_ALREADY_OPEN = hipFileDriverAlreadyOpen;
  // CHECK-NEXT: hipFileOpError_t FILE_HANDLE_NOT_REGISTERED = hipFileHandleNotRegistered;
  // CHECK-NEXT: hipFileOpError_t FILE_HANDLE_ALREADY_REGISTERED = hipFileHandleAlreadyRegistered;
  CUfileOpError FILE_MEMORY_ALREADY_REGISTERED = CU_FILE_MEMORY_ALREADY_REGISTERED;
  CUfileOpError FILE_MEMORY_NOT_REGISTERED = CU_FILE_MEMORY_NOT_REGISTERED;
  CUfileOpError FILE_PERMISSION_DENIED = CU_FILE_PERMISSION_DENIED;
  CUfileOpError FILE_DRIVER_ALREADY_OPEN = CU_FILE_DRIVER_ALREADY_OPEN;
  CUfileOpError FILE_HANDLE_NOT_REGISTERED = CU_FILE_HANDLE_NOT_REGISTERED;
  CUfileOpError FILE_HANDLE_ALREADY_REGISTERED = CU_FILE_HANDLE_ALREADY_REGISTERED;

  // CHECK: hipFileOpError_t FILE_DEVICE_NOT_FOUND = hipFileDeviceNotFound;
  // CHECK-NEXT: hipFileOpError_t FILE_INTERNAL_ERROR = hipFileInternalError;
  // CHECK-NEXT: hipFileOpError_t FILE_GETNEWFD_FAILED = hipFileGetNewFDFailed;
  // CHECK-NEXT: hipFileOpError_t FILE_NVFS_SETUP_ERROR = hipFileDriverSetupError;
  // CHECK-NEXT: hipFileOpError_t FILE_IO_DISABLED = hipFileIODisabled;
  CUfileOpError FILE_DEVICE_NOT_FOUND = CU_FILE_DEVICE_NOT_FOUND;
  CUfileOpError FILE_INTERNAL_ERROR = CU_FILE_INTERNAL_ERROR;
  CUfileOpError FILE_GETNEWFD_FAILED = CU_FILE_GETNEWFD_FAILED;
  CUfileOpError FILE_NVFS_SETUP_ERROR = CU_FILE_NVFS_SETUP_ERROR;
  CUfileOpError FILE_IO_DISABLED = CU_FILE_IO_DISABLED;

#if CUDA_VERSION >= 11060
  // CHECK: hipFileOpError_t FILE_BATCH_SUBMIT_FAILED = hipFileBatchSubmitFailed;
  CUfileOpError FILE_BATCH_SUBMIT_FAILED = CU_FILE_BATCH_SUBMIT_FAILED;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: hipFileOpError_t FILE_GPU_MEMORY_PINNING_FAILED = hipFileGPUMemoryPinningFailed;
  CUfileOpError FILE_GPU_MEMORY_PINNING_FAILED = CU_FILE_GPU_MEMORY_PINNING_FAILED;
#endif

#if CUDA_VERSION >= 12010
  // CHECK: hipFileOpError_t FILE_BATCH_FULL = hipFileBatchFull;
  CUfileOpError FILE_BATCH_FULL = CU_FILE_BATCH_FULL;
#endif

#if CUDA_VERSION >= 12020
  // CHECK: hipFileOpError_t FILE_ASYNC_NOT_SUPPORTED = hipFileAsyncNotSupported;
  CUfileOpError FILE_ASYNC_NOT_SUPPORTED = CU_FILE_ASYNC_NOT_SUPPORTED;
#endif

#if CUDA_VERSION >= 11050
  // CHECK: hipFileOpError_t FILE_IO_MAX_ERROR = hipFileIOMaxError;
  CUfileOpError FILE_IO_MAX_ERROR = CU_FILE_IO_MAX_ERROR;
#endif

  // CHECK: hipFileDriverStatusFlags_t FILE_LUSTRE_SUPPORTED = hipFileLustreSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_WEKAFS_SUPPORTED = hipFileWekaFSSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_NFS_SUPPORTED = hipFileNFSSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_GPFS_SUPPORTED = hipFileGPFSSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_NVME_SUPPORTED = hipFileNVMeSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_NVMEOF_SUPPORTED = hipFileNVMeoFSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_SCSI_SUPPORTED = hipFileSCSISupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_SCALEFLUX_CSD_SUPPORTED = hipFileScaleFluxCSDSupported;
  // CHECK-NEXT: hipFileDriverStatusFlags_t FILE_NVMESH_SUPPORTED = hipFileNVMeshSupported;
  CUfileDriverStatusFlags_t FILE_LUSTRE_SUPPORTED = CU_FILE_LUSTRE_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_WEKAFS_SUPPORTED = CU_FILE_WEKAFS_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_NFS_SUPPORTED = CU_FILE_NFS_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_GPFS_SUPPORTED = CU_FILE_GPFS_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_NVME_SUPPORTED = CU_FILE_NVME_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_NVMEOF_SUPPORTED = CU_FILE_NVMEOF_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_SCSI_SUPPORTED = CU_FILE_SCSI_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_SCALEFLUX_CSD_SUPPORTED = CU_FILE_SCALEFLUX_CSD_SUPPORTED;
  CUfileDriverStatusFlags_t FILE_NVMESH_SUPPORTED = CU_FILE_NVMESH_SUPPORTED;

#if CUDA_VERSION >= 11060
  // CHECK: hipFileDriverStatusFlags_t FILE_BEEGFS_SUPPORTED = hipFileBEEGFSSupported;
  CUfileDriverStatusFlags_t FILE_BEEGFS_SUPPORTED = CU_FILE_BEEGFS_SUPPORTED;
#endif

#if CUDA_VERSION >= 12080
  // CHECK: hipFileDriverStatusFlags_t FILE_NVME_P2P_SUPPORTED = hipFileNVMeP2PSupported;
  CUfileDriverStatusFlags_t FILE_NVME_P2P_SUPPORTED = CU_FILE_NVME_P2P_SUPPORTED;
#endif

  // CHECK: hipFileDriverControlFlags_t FILE_USE_POLL_MODE = hipFileUsePollMode;
  // CHECK-NEXT: hipFileDriverControlFlags_t FILE_ALLOW_COMPAT_MODE = hipFileAllowCompatMode;
  CUfileDriverControlFlags_t FILE_USE_POLL_MODE = CU_FILE_USE_POLL_MODE;
  CUfileDriverControlFlags_t FILE_ALLOW_COMPAT_MODE = CU_FILE_ALLOW_COMPAT_MODE;

  // CHECK: hipFileFeatureFlags_t FILE_DYN_ROUTING_SUPPORTED = hipFileDynRoutingSupported;
  // CHECK-NEXT: hipFileFeatureFlags_t FILE_BATCH_IO_SUPPORTED = hipFileBatchIOSupported;
  // CHECK-NEXT: hipFileFeatureFlags_t FILE_STREAMS_SUPPORTED = hipFileStreamsSupported;
  CUfileFeatureFlags_t FILE_DYN_ROUTING_SUPPORTED = CU_FILE_DYN_ROUTING_SUPPORTED;
  CUfileFeatureFlags_t FILE_BATCH_IO_SUPPORTED = CU_FILE_BATCH_IO_SUPPORTED;
  CUfileFeatureFlags_t FILE_STREAMS_SUPPORTED = CU_FILE_STREAMS_SUPPORTED;

#if CUDA_VERSION >= 12030
  // CHECK: hipFileFeatureFlags_t FILE_PARALLEL_IO_SUPPORTED = hipFileParallelIOSupported;
  CUfileFeatureFlags_t FILE_PARALLEL_IO_SUPPORTED = CU_FILE_PARALLEL_IO_SUPPORTED;
#endif

  // CHECK: int RDMA_REGISTER = HIPFILE_RDMA_REGISTER;
  // CHECK-NEXT: int RDMA_RELAXED_ORDERING = HIPFILE_RDMA_RELAXED_ORDERING;
  int RDMA_REGISTER = CU_FILE_RDMA_REGISTER;
  int RDMA_RELAXED_ORDERING = CU_FILE_RDMA_RELAXED_ORDERING;

  // CHECK: hipFileFileHandleType HANDLE_TYPE_OPAQUE_FD = hipFileHandleTypeOpaqueFD;
  // CHECK-NEXT: hipFileFileHandleType HANDLE_TYPE_OPAQUE_WIN32 = hipFileHandleTypeOpaqueWin32;
  // CHECK-NEXT: hipFileFileHandleType HANDLE_TYPE_USERSPACE_FS = hipFileHandleTypeUserspaceFS;
  CUfileFileHandleType HANDLE_TYPE_OPAQUE_FD = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileFileHandleType HANDLE_TYPE_OPAQUE_WIN32 = CU_FILE_HANDLE_TYPE_OPAQUE_WIN32;
  CUfileFileHandleType HANDLE_TYPE_USERSPACE_FS = CU_FILE_HANDLE_TYPE_USERSPACE_FS;

#if CUDA_VERSION >= 11060
  // CHECK: hipFileOpcode_t FILE_OPCODE_READ = hipFileBatchRead;
  // CHECK-NEXT: hipFileOpcode_t FILE_OPCODE_WRITE = hipFileBatchWrite;
  CUfileOpcode_t FILE_OPCODE_READ = CUFILE_READ;
  CUfileOpcode_t FILE_OPCODE_WRITE = CUFILE_WRITE;

  // CHECK: hipFileStatus_t STATUS_WAITING = hipFileWaiting;
  // CHECK-NEXT: hipFileStatus_t STATUS_PENDING_ = hipFilePending;
  // CHECK-NEXT: hipFileStatus_t STATUS_INVALID = hipFileInvalid;
  // CHECK-NEXT: hipFileStatus_t STATUS_CANCELED = hipFileCanceled;
  // CHECK-NEXT: hipFileStatus_t STATUS_COMPLETE = hipFileComplete;
  // CHECK-NEXT: hipFileStatus_t STATUS_TIMEOUT_ = hipFileTimeout;
  // CHECK-NEXT: hipFileStatus_t STATUS_FAILED = hipFileFailed;
  CUfileStatus_t STATUS_WAITING = CUFILE_WAITING;
  CUfileStatus_t STATUS_PENDING_ = CUFILE_PENDING;
  CUfileStatus_t STATUS_INVALID = CUFILE_INVALID;
  CUfileStatus_t STATUS_CANCELED = CUFILE_CANCELED;
  CUfileStatus_t STATUS_COMPLETE = CUFILE_COMPLETE;
  CUfileStatus_t STATUS_TIMEOUT_ = CUFILE_TIMEOUT;
  CUfileStatus_t STATUS_FAILED = CUFILE_FAILED;

  // CHECK: hipFileBatchMode_t BATCH_MODE = hipFileBatch;
  CUfileBatchMode_t BATCH_MODE = CUFILE_BATCH;
#endif

#if CUDA_VERSION >= 12020
  // CHECK: int STREAM_FIXED_BUF_OFFSET = HIPFILE_STREAM_FIXED_BUF_OFFSET;
  // CHECK-NEXT: int STREAM_FIXED_FILE_OFFSET = HIPFILE_STREAM_FIXED_FILE_OFFSET;
  // CHECK-NEXT: int STREAM_FIXED_FILE_SIZE = HIPFILE_STREAM_FIXED_FILE_SIZE;
  // CHECK-NEXT: int STREAM_PAGE_ALIGNED_INPUTS = HIPFILE_STREAM_PAGE_ALIGNED_INPUTS;
  int STREAM_FIXED_BUF_OFFSET = CU_FILE_STREAM_FIXED_BUF_OFFSET;
  int STREAM_FIXED_FILE_OFFSET = CU_FILE_STREAM_FIXED_FILE_OFFSET;
  int STREAM_FIXED_FILE_SIZE = CU_FILE_STREAM_FIXED_FILE_SIZE;
  int STREAM_PAGE_ALIGNED_INPUTS = CU_FILE_STREAM_PAGE_ALIGNED_INPUTS;
#endif

#if CUDA_VERSION >= 12090
  // CHECK: hipFileSizeTConfigParameter_t PARAM_PROFILE_STATS = hipFileParamProfileStats;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_IO_QUEUE_DEPTH = hipFileParamExecutionMaxIOQueueDepth;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_IO_THREADS = hipFileParamExecutionMaxIOThreads;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MIN_IO_THRESHOLD = hipFileParamExecutionMinIOThresholdSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_REQ_PARALLELISM = hipFileParamExecutionMaxRequestParallelism;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_DIRECT_IO_SIZE = hipFileParamPropertiesMaxDirectIOSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_CACHE_SIZE = hipFileParamPropertiesMaxDeviceCacheSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_PER_BUF_CACHE_SIZE = hipFileParamPropertiesPerBufferCacheSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_MAX_PINNED_MEM_SIZE = hipFileParamPropertiesMaxDevicePinnedMemSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_IO_BATCHSIZE = hipFileParamPropertiesIOBatchsize;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_POLLTHRESHOLD_SIZE = hipFileParamPollthresholdSizeKB;
  // CHECK-NEXT: hipFileSizeTConfigParameter_t PARAM_BATCH_IO_TIMEOUT = hipFileParamPropertiesBatchIOTimeoutMs;
  CUFileSizeTConfigParameter_t PARAM_PROFILE_STATS = CUFILE_PARAM_PROFILE_STATS;
  CUFileSizeTConfigParameter_t PARAM_MAX_IO_QUEUE_DEPTH = CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH;
  CUFileSizeTConfigParameter_t PARAM_MAX_IO_THREADS = CUFILE_PARAM_EXECUTION_MAX_IO_THREADS;
  CUFileSizeTConfigParameter_t PARAM_MIN_IO_THRESHOLD = CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_MAX_REQ_PARALLELISM = CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM;
  CUFileSizeTConfigParameter_t PARAM_MAX_DIRECT_IO_SIZE = CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_MAX_CACHE_SIZE = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_PER_BUF_CACHE_SIZE = CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_MAX_PINNED_MEM_SIZE = CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_IO_BATCHSIZE = CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE;
  CUFileSizeTConfigParameter_t PARAM_POLLTHRESHOLD_SIZE = CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB;
  CUFileSizeTConfigParameter_t PARAM_BATCH_IO_TIMEOUT = CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS;

  // CHECK: hipFileBoolConfigParameter_t PARAM_USE_POLL_MODE = hipFileParamPropertiesUsePollMode;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_ALLOW_COMPAT_MODE = hipFileParamPropertiesAllowCompatMode;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_FORCE_COMPAT = hipFileParamForceCompatMode;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_FS_API_CHECK = hipFileParamFsMiscApiCheckAggressive;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_PARALLEL_IO = hipFileParamExecutionParallelIO;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_PROFILE_NVTX = hipFileParamProfileNvtx;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_ALLOW_SYS_MEM = hipFileParamPropertiesAllowSystemMemory;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_USE_PCIP2PDMA = hipFileParamUsePcip2pdma;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_PREFER_IO_URING = hipFileParamPreferIOUring;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_FORCE_ODIRECT = hipFileParamForceOdirectMode;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_SKIP_TOPOLOGY = hipFileParamSkipTopologyDetection;
  // CHECK-NEXT: hipFileBoolConfigParameter_t PARAM_STREAM_MEMOPS_BYPASS = hipFileParamStreamMemopsBypass;
  CUFileBoolConfigParameter_t PARAM_USE_POLL_MODE = CUFILE_PARAM_PROPERTIES_USE_POLL_MODE;
  CUFileBoolConfigParameter_t PARAM_ALLOW_COMPAT_MODE = CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE;
  CUFileBoolConfigParameter_t PARAM_FORCE_COMPAT = CUFILE_PARAM_FORCE_COMPAT_MODE;
  CUFileBoolConfigParameter_t PARAM_FS_API_CHECK = CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE;
  CUFileBoolConfigParameter_t PARAM_PARALLEL_IO = CUFILE_PARAM_EXECUTION_PARALLEL_IO;
  CUFileBoolConfigParameter_t PARAM_PROFILE_NVTX = CUFILE_PARAM_PROFILE_NVTX;
  CUFileBoolConfigParameter_t PARAM_ALLOW_SYS_MEM = CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY;
  CUFileBoolConfigParameter_t PARAM_USE_PCIP2PDMA = CUFILE_PARAM_USE_PCIP2PDMA;
  CUFileBoolConfigParameter_t PARAM_PREFER_IO_URING = CUFILE_PARAM_PREFER_IO_URING;
  CUFileBoolConfigParameter_t PARAM_FORCE_ODIRECT = CUFILE_PARAM_FORCE_ODIRECT_MODE;
  CUFileBoolConfigParameter_t PARAM_SKIP_TOPOLOGY = CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION;
  CUFileBoolConfigParameter_t PARAM_STREAM_MEMOPS_BYPASS = CUFILE_PARAM_STREAM_MEMOPS_BYPASS;

  // CHECK: hipFileStringConfigParameter_t PARAM_LOGGING_LEVEL = hipFileParamLoggingLevel;
  // CHECK-NEXT: hipFileStringConfigParameter_t PARAM_ENV_LOGFILE_PATH = hipFileParamEnvLogfilePath;
  // CHECK-NEXT: hipFileStringConfigParameter_t PARAM_LOG_DIR = hipFileParamLogDir;
  CUFileStringConfigParameter_t PARAM_LOGGING_LEVEL = CUFILE_PARAM_LOGGING_LEVEL;
  CUFileStringConfigParameter_t PARAM_ENV_LOGFILE_PATH = CUFILE_PARAM_ENV_LOGFILE_PATH;
  CUFileStringConfigParameter_t PARAM_LOG_DIR = CUFILE_PARAM_LOG_DIR;
#endif

  // Test error checking macros
  CUfileError_t testErr;
  testErr.err = CU_FILE_SUCCESS;

  // CHECK: bool isFileErr = IS_HIPFILE_ERR(testErr.err);
  bool isFileErr = IS_CUFILE_ERR(testErr.err);

  // CHECK: const char* errStr = HIPFILE_ERRSTR(testErr.err);
  const char* errStr = CUFILE_ERRSTR(testErr.err);

  // CHECK: bool isCudaErr = IS_HIP_DRV_ERR(testErr);
  bool isCudaErr = IS_CUDA_ERR(testErr);

  // CHECK: hipError_t cudaErr = HIP_DRV_ERR(testErr);
  CUresult cudaErr = CU_FILE_CUDA_ERR(testErr);

  return 0;
}
