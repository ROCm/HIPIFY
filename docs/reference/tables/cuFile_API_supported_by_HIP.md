<head>
    <meta charset="UTF-8">
    <meta name="description" content="CUDA APIs supported by HIPIFY">
    <meta name="keywords" content="HIPIFY, HIP, ROCm, CUDA, CUDA2HIP, hipification, hipify-clang, hipify-perl, hipFile, cuFile">
</head>

# cuFile API supported by HIP


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, `U`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **U** - Unsupported for CUDA version(s); **E** - Experimental

## **1. cuFile Types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`CUDA_DRV_ERR`|12.9| | | |`HIP_DRV_ERR`|7.2.0| | | | | |
|`CUFILE_BATCH`|1.2.0| | | |`hipFileBatch`|7.2.0| | | | | |
|`CUFILE_CANCELED`|1.2.0| | | |`hipFileCanceled`|7.2.0| | | | | |
|`CUFILE_COMPLETE`|1.2.0| | | |`hipFileComplete`|7.2.0| | | | | |
|`CUFILE_ERRSTR`|1.0.0| | | |`HIPFILE_ERRSTR`|7.2.0| | | | | |
|`CUFILE_FAILED`|1.2.0| | | |`hipFileFailed`|7.2.0| | | | | |
|`CUFILE_INVALID`|1.2.0| | | |`hipFileInvalid`|7.2.0| | | | | |
|`CUFILE_PARAM_ENV_LOGFILE_PATH`|1.14.0| | | |`hipFileParamEnvLogfilePath`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH`|1.14.0| | | |`hipFileParamExecutionMaxIOQueueDepth`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_IO_THREADS`|1.14.0| | | |`hipFileParamExecutionMaxIOThreads`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM`|1.14.0| | | |`hipFileParamExecutionMaxRequestParallelism`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB`|1.14.0| | | |`hipFileParamExecutionMinIOThresholdSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_PARALLEL_IO`|1.14.0| | | |`hipFileParamExecutionParallelIO`|7.2.0| | | | | |
|`CUFILE_PARAM_FORCE_COMPAT_MODE`|1.14.0| | | |`hipFileParamForceCompatMode`|7.2.0| | | | | |
|`CUFILE_PARAM_FORCE_ODIRECT_MODE`|1.14.0| | | |`hipFileParamForceOdirectMode`|7.2.0| | | | | |
|`CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE`|1.14.0| | | |`hipFileParamFsMiscApiCheckAggressive`|7.2.0| | | | | |
|`CUFILE_PARAM_LOGGING_LEVEL`|1.14.0| | | |`hipFileParamLoggingLevel`|7.2.0| | | | | |
|`CUFILE_PARAM_LOG_DIR`|1.14.0| | | |`hipFileParamLogDir`|7.2.0| | | | | |
|`CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB`|1.14.0| | | |`hipFileParamPollthresholdSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PREFER_IO_URING`|1.14.0| | | |`hipFileParamPreferIOUring`|7.2.0| | | | | |
|`CUFILE_PARAM_PROFILE_NVTX`|1.14.0| | | |`hipFileParamProfileNvtx`|7.2.0| | | | | |
|`CUFILE_PARAM_PROFILE_STATS`|1.14.0| | | |`hipFileParamProfileStats`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE`|1.14.0| | | |`hipFileParamPropertiesAllowCompatMode`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY`|1.14.0| | | |`hipFileParamPropertiesAllowSystemMemory`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS`|1.14.0| | | |`hipFileParamPropertiesBatchIOTimeoutMs`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE`|1.14.0| | | |`hipFileParamPropertiesIOBatchsize`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB`|1.14.0| | | |`hipFileParamPropertiesMaxDeviceCacheSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB`|1.14.0| | | |`hipFileParamPropertiesMaxDevicePinnedMemSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB`|1.14.0| | | |`hipFileParamPropertiesMaxDirectIOSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB`|1.14.0| | | |`hipFileParamPropertiesPerBufferCacheSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_USE_POLL_MODE`|1.14.0| | | |`hipFileParamPropertiesUsePollMode`|7.2.0| | | | | |
|`CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION`|1.14.0| | | |`hipFileParamSkipTopologyDetection`|7.2.0| | | | | |
|`CUFILE_PARAM_STREAM_MEMOPS_BYPASS`|1.14.0| | | |`hipFileParamStreamMemopsBypass`|7.2.0| | | | | |
|`CUFILE_PARAM_USE_PCIP2PDMA`|1.14.0| | | |`hipFileParamUsePcip2pdma`|7.2.0| | | | | |
|`CUFILE_PENDING`|1.2.0| | | |`hipFilePending`|7.2.0| | | | | |
|`CUFILE_READ`|1.2.0| | | |`hipFileBatchRead`|7.2.0| | | | | |
|`CUFILE_TIMEOUT`|1.2.0| | | |`hipFileTimeout`|7.2.0| | | | | |
|`CUFILE_WAITING`|1.2.0| | | |`hipFileWaiting`|7.2.0| | | | | |
|`CUFILE_WRITE`|1.2.0| | | |`hipFileBatchWrite`|7.2.0| | | | | |
|`CUFileBoolConfigParameter_t`|1.14.0| | | |`hipFileBoolConfigParameter_t`|7.2.0| | | | | |
|`CUFileSizeTConfigParameter_t`|1.14.0| | | |`hipFileSizeTConfigParameter_t`|7.2.0| | | | | |
|`CUFileStringConfigParameter_t`|1.14.0| | | |`hipFileStringConfigParameter_t`|7.2.0| | | | | |
|`CU_FILE_ALLOW_COMPAT_MODE`|1.0.0| | | |`hipFileAllowCompatMode`|7.2.0| | | | | |
|`CU_FILE_ASYNC_NOT_SUPPORTED`|1.7.0| | | |`hipFileAsyncNotSupported`|7.2.0| | | | | |
|`CU_FILE_BATCH_FULL`|1.5.1| | | |`hipFileBatchFull`|7.2.0| | | | | |
|`CU_FILE_BATCH_IO_SUPPORTED`|1.0.0| | | |`hipFileBatchIOSupported`|7.2.0| | | | | |
|`CU_FILE_BATCH_SUBMIT_FAILED`|1.2.0| | | |`hipFileBatchSubmitFailed`|7.2.0| | | | | |
|`CU_FILE_BEEGFS_SUPPORTED`|1.1.1| | | |`hipFileBEEGFSSupported`|7.2.0| | | | | |
|`CU_FILE_CUDA_CONTEXT_MISMATCH`|1.0.0| | | |`hipFileHipContextMismatch`|7.2.0| | | | | |
|`CU_FILE_CUDA_DRIVER_ERROR`|1.0.0| | | |`hipFileHipDriverError`|7.2.0| | | | | |
|`CU_FILE_CUDA_MEMORY_TYPE_INVALID`|1.0.0| | | |`hipFileHipMemoryTypeInvalid`|7.2.0| | | | | |
|`CU_FILE_CUDA_POINTER_INVALID`|1.0.0| | | |`hipFileHipPointerInvalid`|7.2.0| | | | | |
|`CU_FILE_CUDA_POINTER_RANGE_ERROR`|1.0.0| | | |`hipFileHipPointerRangeError`|7.2.0| | | | | |
|`CU_FILE_DEVICE_NOT_FOUND`|1.0.0| | | |`hipFileDeviceNotFound`|7.2.0| | | | | |
|`CU_FILE_DEVICE_NOT_SUPPORTED`|1.0.0| | | |`hipFileDeviceNotSupported`|7.2.0| | | | | |
|`CU_FILE_DIO_NOT_SET`|1.0.0| | | |`hipFileDIONotSet`|7.2.0| | | | | |
|`CU_FILE_DRIVER_ALREADY_OPEN`|1.0.0| | | |`hipFileDriverAlreadyOpen`|7.2.0| | | | | |
|`CU_FILE_DRIVER_CLOSING`|1.0.0| | | |`hipFileDriverClosing`|7.2.0| | | | | |
|`CU_FILE_DRIVER_INVALID_PROPS`|1.0.0| | | |`hipFileDriverInvalidProps`|7.2.0| | | | | |
|`CU_FILE_DRIVER_NOT_INITIALIZED`|1.0.0| | | |`hipFileDriverNotInitialized`|7.2.0| | | | | |
|`CU_FILE_DRIVER_UNSUPPORTED_LIMIT`|1.0.0| | | |`hipFileDriverUnsupportedLimit`|7.2.0| | | | | |
|`CU_FILE_DRIVER_VERSION_MISMATCH`|1.0.0| | | |`hipFileDriverVersionMismatch`|7.2.0| | | | | |
|`CU_FILE_DRIVER_VERSION_READ_ERROR`|1.0.0| | | |`hipFileDriverVersionReadError`|7.2.0| | | | | |
|`CU_FILE_DYN_ROUTING_SUPPORTED`|1.0.0| | | |`hipFileDynRoutingSupported`|7.2.0| | | | | |
|`CU_FILE_GETNEWFD_FAILED`|1.0.0| | | |`hipFileGetNewFDFailed`|7.2.0| | | | | |
|`CU_FILE_GPFS_SUPPORTED`|1.0.0| | | |`hipFileGPFSSupported`|7.2.0| | | | | |
|`CU_FILE_GPU_MEMORY_PINNING_FAILED`|1.2.0| | | |`hipFileGPUMemoryPinningFailed`|7.2.0| | | | | |
|`CU_FILE_HANDLE_ALREADY_REGISTERED`|1.0.0| | | |`hipFileHandleAlreadyRegistered`|7.2.0| | | | | |
|`CU_FILE_HANDLE_NOT_REGISTERED`|1.0.0| | | |`hipFileHandleNotRegistered`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_OPAQUE_FD`|1.0.0| | | |`hipFileHandleTypeOpaqueFD`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_OPAQUE_WIN32`|1.0.0| | | |`hipFileHandleTypeOpaqueWin32`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_USERSPACE_FS`|1.0.0| | | |`hipFileHandleTypeUserspaceFS`|7.2.0| | | | | |
|`CU_FILE_INTERNAL_ERROR`|1.0.0| | | |`hipFileInternalError`|7.2.0| | | | | |
|`CU_FILE_INVALID_FILE_OPEN_FLAG`|1.0.0| | | |`hipFileInvalidFileOpenFlag`|7.2.0| | | | | |
|`CU_FILE_INVALID_FILE_TYPE`|1.0.0| | | |`hipFileInvalidFileType`|7.2.0| | | | | |
|`CU_FILE_INVALID_MAPPING_RANGE`|1.0.0| | | |`hipFileInvalidMappingRange`|7.2.0| | | | | |
|`CU_FILE_INVALID_MAPPING_SIZE`|1.0.0| | | |`hipFileInvalidMappingSize`|7.2.0| | | | | |
|`CU_FILE_INVALID_VALUE`|1.0.0| | | |`hipFileInvalidValue`|7.2.0| | | | | |
|`CU_FILE_IO_DISABLED`|1.0.0| | | |`hipFileIODisabled`|7.2.0| | | | | |
|`CU_FILE_IO_MAX_ERROR`|1.1.0| | | |`hipFileIOMaxError`|7.2.0| | | | | |
|`CU_FILE_IO_NOT_SUPPORTED`|1.0.0| | | |`hipFileIONotSupported`|7.2.0| | | | | |
|`CU_FILE_LUSTRE_SUPPORTED`|1.0.0| | | |`hipFileLustreSupported`|7.2.0| | | | | |
|`CU_FILE_MEMORY_ALREADY_REGISTERED`|1.0.0| | | |`hipFileMemoryAlreadyRegistered`|7.2.0| | | | | |
|`CU_FILE_MEMORY_NOT_REGISTERED`|1.0.0| | | |`hipFileMemoryNotRegistered`|7.2.0| | | | | |
|`CU_FILE_NFS_SUPPORTED`|1.0.0| | | |`hipFileNFSSupported`|7.2.0| | | | | |
|`CU_FILE_NVFS_DRIVER_ERROR`|1.0.0| | | |`hipFileDriverError`|7.2.0| | | | | |
|`CU_FILE_NVFS_SETUP_ERROR`|1.0.0| | | |`hipFileDriverSetupError`|7.2.0| | | | | |
|`CU_FILE_NVMEOF_SUPPORTED`|1.0.0| | | |`hipFileNVMeoFSupported`|7.2.0| | | | | |
|`CU_FILE_NVMESH_SUPPORTED`|1.0.0| | | |`hipFileNVMeshSupported`|7.2.0| | | | | |
|`CU_FILE_NVME_P2P_SUPPORTED`|1.13.0| | | |`hipFileNVMeP2PSsupported`|7.2.0| | | | | |
|`CU_FILE_NVME_SUPPORTED`|1.0.0| | | |`hipFileNVMeSupported`|7.2.0| | | | | |
|`CU_FILE_PARALLEL_IO_SUPPORTED`|1.8.0| | | |`hipFileParallelIOSupported`|7.2.0| | | | | |
|`CU_FILE_PERMISSION_DENIED`|1.0.0| | | |`hipFilePermissionDenied`|7.2.0| | | | | |
|`CU_FILE_PLATFORM_NOT_SUPPORTED`|1.0.0| | | |`hipFilePlatformNotSupported`|7.2.0| | | | | |
|`CU_FILE_RDMA_REGISTER`|1.0.0| | | |`HIPFILE_RDMA_REGISTER`|7.2.0| | | | | |
|`CU_FILE_RDMA_RELAXED_ORDERING`|1.0.0| | | |`HIPFILE_RDMA_RELAXED_ORDERING`|7.2.0| | | | | |
|`CU_FILE_SCALEFLUX_CSD_SUPPORTED`|1.0.0| | | |`hipFileScaleFluxCSDSupported`|7.2.0| | | | | |
|`CU_FILE_SCSI_SUPPORTED`|1.0.0| | | |`hipFileSCSISupported`|7.2.0| | | | | |
|`CU_FILE_STREAMS_SUPPORTED`|1.0.0| | | |`hipFileStreamsSupported`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_BUF_OFFSET`|1.7.0| | | |`HIPFILE_STREAM_FIXED_BUF_OFFSET`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_FILE_OFFSET`|1.7.0| | | |`HIPFILE_STREAM_FIXED_FILE_OFFSET`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_FILE_SIZE`|1.7.0| | | |`HIPFILE_STREAM_FIXED_FILE_SIZE`|7.2.0| | | | | |
|`CU_FILE_STREAM_PAGE_ALIGNED_INPUTS`|1.7.0| | | |`HIPFILE_STREAM_PAGE_ALIGNED_INPUTS`|7.2.0| | | | | |
|`CU_FILE_SUCCESS`|1.0.0| | | |`hipFileSuccess`|7.2.0| | | | | |
|`CU_FILE_USE_POLL_MODE`|1.0.0| | | |`hipFileUsePollMode`|7.2.0| | | | | |
|`CU_FILE_WEKAFS_SUPPORTED`|1.0.0| | | |`hipFileWekaFSSupported`|7.2.0| | | | | |
|`CUfileBatchHandle_t`|1.2.0| | | |`hipFileBatchHandle_t`|7.2.0| | | | | |
|`CUfileBatchMode_t`|1.2.0| | | |`hipFileBatchMode_t`|7.2.0| | | | | |
|`CUfileDescr_t`|1.0.0| | | |`hipFileDescr_t`|7.2.0| | | | | |
|`CUfileDriverControlFlags_t`|1.0.0| | | |`hipFileDriverControlFlags_t`|7.2.0| | | | | |
|`CUfileDriverStatusFlags_t`|1.0.0| | | |`hipFileDriverStatusFlags_t`|7.2.0| | | | | |
|`CUfileDrvProps_t`|1.0.0| | | |`hipFileDriverProps_t`|7.2.0| | | | | |
|`CUfileError_t`|1.0.0| | | |`hipFileError_t`|7.2.0| | | | | |
|`CUfileFSOps_t`|1.0.0| | | |`hipFileFSOps_t`|7.2.0| | | | | |
|`CUfileFeatureFlags_t`|1.0.0| | | |`hipFileFeatureFlags_t`|7.2.0| | | | | |
|`CUfileFileHandleType`|1.0.0| | | |`hipFileFileHandleType`|7.2.0| | | | | |
|`CUfileHandle_t`|1.0.0| | | |`hipFileHandle_t`|7.2.0| | | | | |
|`CUfileIOEvents_t`|1.2.0| | | |`hipFileIOEvents_t`|7.2.0| | | | | |
|`CUfileIOParams_t`|1.2.0| | | |`hipFileIOParams_t`|7.2.0| | | | | |
|`CUfileOpError`|1.0.0| | | |`hipFileOpError_t`|7.2.0| | | | | |
|`CUfileOpcode_t`|1.2.0| | | |`hipFileOpcode_t`|7.2.0| | | | | |
|`CUfileStatus_t`|1.2.0| | | |`hipFileStatus_t`|7.2.0| | | | | |
|`IS_CUDA_ERR`|1.0.0| | | |`IS_HIP_DRV_ERR`|7.2.0| | | | | |
|`IS_CUFILE_ERR`|1.0.0| | | |`IS_HIPFILE_ERR`|7.2.0| | | | | |
|`cufileRDMAInfo_t`|1.0.0| | | |`hipFileRDMAInfo_t`|7.2.0| | | | | |

## **2. cuFile Functions**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cuFileBatchIOCancel`|12.9| | | |`hipFileBatchIOCancel`|7.2.0| | | | | |
|`cuFileBatchIODestroy`|12.9| | | |`hipFileBatchIODestroy`|7.2.0| | | | | |
|`cuFileBatchIOGetStatus`|12.9| | | |`hipFileBatchIOGetStatus`|7.2.0| | | | | |
|`cuFileBatchIOSetUp`|12.9| | | |`hipFileBatchIOSetUp`|7.2.0| | | | | |
|`cuFileBatchIOSubmit`|12.9| | | |`hipFileBatchIOSubmit`|7.2.0| | | | | |
|`cuFileBufDeregister`|12.9| | | |`hipFileBufDeregister`|7.2.0| | | | | |
|`cuFileBufRegister`|12.9| | | |`hipFileBufRegister`|7.2.0| | | | | |
|`cuFileDriverClose`|12.9| | | |`hipFileDriverClose`|7.2.0| | | | | |
|`cuFileDriverClose_v2`|12.9| | | |`hipFileDriverClose`|7.2.0| | | | | |
|`cuFileDriverGetProperties`|12.9| | | |`hipFileDriverGetProperties`|7.2.0| | | | | |
|`cuFileDriverOpen`|12.9| | | |`hipFileDriverOpen`|7.2.0| | | | | |
|`cuFileDriverSetMaxCacheSize`|12.9| | | |`hipFileDriverSetMaxCacheSize`|7.2.0| | | | | |
|`cuFileDriverSetMaxDirectIOSize`|12.9| | | |`hipFileDriverSetMaxDirectIOSize`|7.2.0| | | | | |
|`cuFileDriverSetMaxPinnedMemSize`|12.9| | | |`hipFileDriverSetMaxPinnedMemSize`|7.2.0| | | | | |
|`cuFileDriverSetPollMode`|12.9| | | |`hipFileDriverSetPollMode`|7.2.0| | | | | |
|`cuFileGetParameterBool`|12.9| | | |`hipFileGetParameterBool`|7.2.0| | | | | |
|`cuFileGetParameterSizeT`|12.9| | | |`hipFileGetParameterSizeT`|7.2.0| | | | | |
|`cuFileGetParameterString`|12.9| | | |`hipFileGetParameterString`|7.2.0| | | | | |
|`cuFileHandleDeregister`|12.9| | | |`hipFileHandleDeregister`|7.2.0| | | | | |
|`cuFileHandleRegister`|12.9| | | |`hipFileHandleRegister`|7.2.0| | | | | |
|`cuFileRead`|12.9| | | |`hipFileRead`|7.2.0| | | | | |
|`cuFileReadAsync`|12.9| | | |`hipFileReadAsync`|7.2.0| | | | | |
|`cuFileSetParameterBool`|12.9| | | |`hipFileSetParameterBool`|7.2.0| | | | | |
|`cuFileSetParameterSizeT`|12.9| | | |`hipFileSetParameterSizeT`|7.2.0| | | | | |
|`cuFileSetParameterString`|12.9| | | |`hipFileSetParameterString`|7.2.0| | | | | |
|`cuFileStreamDeregister`|12.9| | | |`hipFileStreamDeregister`|7.2.0| | | | | |
|`cuFileStreamRegister`|12.9| | | |`hipFileStreamRegister`|7.2.0| | | | | |
|`cuFileUseCount`|12.9| | | |`hipFileUseCount`|7.2.0| | | | | |
|`cuFileWrite`|12.9| | | |`hipFileWrite`|7.2.0| | | | | |
|`cuFileWriteAsync`|12.9| | | |`hipFileWriteAsync`|7.2.0| | | | | |
|`cufileop_status_error`|12.9| | | |`hipFileOpStatusError`|7.2.0| | | | | |

