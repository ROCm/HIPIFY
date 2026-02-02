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
|`CUFILE_BATCH`|12.9| | | |`hipFileBatch`|7.2.0| | | | | |
|`CUFILE_CANCELED`|12.9| | | |`hipFileCanceled`|7.2.0| | | | | |
|`CUFILE_COMPLETE`|12.9| | | |`hipFileComplete`|7.2.0| | | | | |
|`CUFILE_ERRSTR`|12.9| | | |`HIPFILE_ERRSTR`|7.2.0| | | | | |
|`CUFILE_FAILED`|12.9| | | |`hipFileFailed`|7.2.0| | | | | |
|`CUFILE_INVALID`|12.9| | | |`hipFileInvalid`|7.2.0| | | | | |
|`CUFILE_PARAM_ENV_LOGFILE_PATH`|12.9| | | |`hipFileParamEnvLogfilePath`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH`|12.9| | | |`hipFileParamExecutionMaxIOQueueDepth`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_IO_THREADS`|12.9| | | |`hipFileParamExecutionMaxIOThreads`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM`|12.9| | | |`hipFileParamExecutionMaxRequestParallelism`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB`|12.9| | | |`hipFileParamExecutionMinIOThresholdSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_EXECUTION_PARALLEL_IO`|12.9| | | |`hipFileParamExecutionParallelIO`|7.2.0| | | | | |
|`CUFILE_PARAM_FORCE_COMPAT_MODE`|12.9| | | |`hipFileParamForceCompatMode`|7.2.0| | | | | |
|`CUFILE_PARAM_FORCE_ODIRECT_MODE`|12.9| | | |`hipFileParamForceOdirectMode`|7.2.0| | | | | |
|`CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE`|12.9| | | |`hipFileParamFsMiscApiCheckAggressive`|7.2.0| | | | | |
|`CUFILE_PARAM_LOGGING_LEVEL`|12.9| | | |`hipFileParamLoggingLevel`|7.2.0| | | | | |
|`CUFILE_PARAM_LOG_DIR`|12.9| | | |`hipFileParamLogDir`|7.2.0| | | | | |
|`CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB`|12.9| | | |`hipFileParamPollthresholdSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PREFER_IO_URING`|12.9| | | |`hipFileParamPreferIOUring`|7.2.0| | | | | |
|`CUFILE_PARAM_PROFILE_NVTX`|12.9| | | |`hipFileParamProfileNvtx`|7.2.0| | | | | |
|`CUFILE_PARAM_PROFILE_STATS`|12.9| | | |`hipFileParamProfileStats`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE`|12.9| | | |`hipFileParamPropertiesAllowCompatMode`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY`|12.9| | | |`hipFileParamPropertiesAllowSystemMemory`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS`|12.9| | | |`hipFileParamPropertiesBatchIOTimeoutMs`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE`|12.9| | | |`hipFileParamPropertiesIOBatchsize`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB`|12.9| | | |`hipFileParamPropertiesMaxDeviceCacheSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB`|12.9| | | |`hipFileParamPropertiesMaxDevicePinnedMemSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB`|12.9| | | |`hipFileParamPropertiesMaxDirectIOSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB`|12.9| | | |`hipFileParamPropertiesPerBufferCacheSizeKB`|7.2.0| | | | | |
|`CUFILE_PARAM_PROPERTIES_USE_POLL_MODE`|12.9| | | |`hipFileParamPropertiesUsePollMode`|7.2.0| | | | | |
|`CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION`|12.9| | | |`hipFileParamSkipTopologyDetection`|7.2.0| | | | | |
|`CUFILE_PARAM_STREAM_MEMOPS_BYPASS`|12.9| | | |`hipFileParamStreamMemopsBypass`|7.2.0| | | | | |
|`CUFILE_PARAM_USE_PCIP2PDMA`|12.9| | | |`hipFileParamUsePcip2pdma`|7.2.0| | | | | |
|`CUFILE_PENDING`|12.9| | | |`hipFilePending`|7.2.0| | | | | |
|`CUFILE_READ`|12.9| | | |`hipFileBatchRead`|7.2.0| | | | | |
|`CUFILE_TIMEOUT`|12.9| | | |`hipFileTimeout`|7.2.0| | | | | |
|`CUFILE_WAITING`|12.9| | | |`hipFileWaiting`|7.2.0| | | | | |
|`CUFILE_WRITE`|12.9| | | |`hipFileBatchWrite`|7.2.0| | | | | |
|`CUFileBoolConfigParameter_t`|12.9| | | |`hipFileBoolConfigParameter_t`|7.2.0| | | | | |
|`CUFileSizeTConfigParameter_t`|12.9| | | |`hipFileSizeTConfigParameter_t`|7.2.0| | | | | |
|`CUFileStringConfigParameter_t`|12.9| | | |`hipFileStringConfigParameter_t`|7.2.0| | | | | |
|`CU_FILE_ALLOW_COMPAT_MODE`|12.9| | | |`hipFileAllowCompatMode`|7.2.0| | | | | |
|`CU_FILE_ASYNC_NOT_SUPPORTED`|12.9| | | |`hipFileAsyncNotSupported`|7.2.0| | | | | |
|`CU_FILE_BATCH_FULL`|12.9| | | |`hipFileBatchFull`|7.2.0| | | | | |
|`CU_FILE_BATCH_IO_SUPPORTED`|12.9| | | |`hipFileBatchIOSupported`|7.2.0| | | | | |
|`CU_FILE_BATCH_SUBMIT_FAILED`|12.9| | | |`hipFileBatchSubmitFailed`|7.2.0| | | | | |
|`CU_FILE_BEEGFS_SUPPORTED`|12.9| | | |`hipFileBEEGFSSupported`|7.2.0| | | | | |
|`CU_FILE_CUDA_CONTEXT_MISMATCH`|12.9| | | |`hipFileHipContextMismatch`|7.2.0| | | | | |
|`CU_FILE_CUDA_DRIVER_ERROR`|12.9| | | |`hipFileHipDriverError`|7.2.0| | | | | |
|`CU_FILE_CUDA_MEMORY_TYPE_INVALID`|12.9| | | |`hipFileHipMemoryTypeInvalid`|7.2.0| | | | | |
|`CU_FILE_CUDA_POINTER_INVALID`|12.9| | | |`hipFileHipPointerInvalid`|7.2.0| | | | | |
|`CU_FILE_CUDA_POINTER_RANGE_ERROR`|12.9| | | |`hipFileHipPointerRangeError`|7.2.0| | | | | |
|`CU_FILE_DEVICE_NOT_FOUND`|12.9| | | |`hipFileDeviceNotFound`|7.2.0| | | | | |
|`CU_FILE_DEVICE_NOT_SUPPORTED`|12.9| | | |`hipFileDeviceNotSupported`|7.2.0| | | | | |
|`CU_FILE_DIO_NOT_SET`|12.9| | | |`hipFileDIONotSet`|7.2.0| | | | | |
|`CU_FILE_DRIVER_ALREADY_OPEN`|12.9| | | |`hipFileDriverAlreadyOpen`|7.2.0| | | | | |
|`CU_FILE_DRIVER_CLOSING`|12.9| | | |`hipFileDriverClosing`|7.2.0| | | | | |
|`CU_FILE_DRIVER_INVALID_PROPS`|12.9| | | |`hipFileDriverInvalidProps`|7.2.0| | | | | |
|`CU_FILE_DRIVER_NOT_INITIALIZED`|12.9| | | |`hipFileDriverNotInitialized`|7.2.0| | | | | |
|`CU_FILE_DRIVER_UNSUPPORTED_LIMIT`|12.9| | | |`hipFileDriverUnsupportedLimit`|7.2.0| | | | | |
|`CU_FILE_DRIVER_VERSION_MISMATCH`|12.9| | | |`hipFileDriverVersionMismatch`|7.2.0| | | | | |
|`CU_FILE_DRIVER_VERSION_READ_ERROR`|12.9| | | |`hipFileDriverVersionReadError`|7.2.0| | | | | |
|`CU_FILE_DYN_ROUTING_SUPPORTED`|12.9| | | |`hipFileDynRoutingSupported`|7.2.0| | | | | |
|`CU_FILE_GETNEWFD_FAILED`|12.9| | | |`hipFileGetNewFDFailed`|7.2.0| | | | | |
|`CU_FILE_GPFS_SUPPORTED`|12.9| | | |`hipFileGPFSSupported`|7.2.0| | | | | |
|`CU_FILE_GPU_MEMORY_PINNING_FAILED`|12.9| | | |`hipFileGPUMemoryPinningFailed`|7.2.0| | | | | |
|`CU_FILE_HANDLE_ALREADY_REGISTERED`|12.9| | | |`hipFileHandleAlreadyRegistered`|7.2.0| | | | | |
|`CU_FILE_HANDLE_NOT_REGISTERED`|12.9| | | |`hipFileHandleNotRegistered`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_OPAQUE_FD`|12.9| | | |`hipFileHandleTypeOpaqueFD`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_OPAQUE_WIN32`|12.9| | | |`hipFileHandleTypeOpaqueWin32`|7.2.0| | | | | |
|`CU_FILE_HANDLE_TYPE_USERSPACE_FS`|12.9| | | |`hipFileHandleTypeUserspaceFS`|7.2.0| | | | | |
|`CU_FILE_INTERNAL_ERROR`|12.9| | | |`hipFileInternalError`|7.2.0| | | | | |
|`CU_FILE_INVALID_FILE_OPEN_FLAG`|12.9| | | |`hipFileInvalidFileOpenFlag`|7.2.0| | | | | |
|`CU_FILE_INVALID_FILE_TYPE`|12.9| | | |`hipFileInvalidFileType`|7.2.0| | | | | |
|`CU_FILE_INVALID_MAPPING_RANGE`|12.9| | | |`hipFileInvalidMappingRange`|7.2.0| | | | | |
|`CU_FILE_INVALID_MAPPING_SIZE`|12.9| | | |`hipFileInvalidMappingSize`|7.2.0| | | | | |
|`CU_FILE_INVALID_VALUE`|12.9| | | |`hipFileInvalidValue`|7.2.0| | | | | |
|`CU_FILE_IO_DISABLED`|12.9| | | |`hipFileIODisabled`|7.2.0| | | | | |
|`CU_FILE_IO_MAX_ERROR`|12.9| | | |`hipFileIOMaxError`|7.2.0| | | | | |
|`CU_FILE_IO_NOT_SUPPORTED`|12.9| | | |`hipFileIONotSupported`|7.2.0| | | | | |
|`CU_FILE_LUSTRE_SUPPORTED`|12.9| | | |`hipFileLustreSupported`|7.2.0| | | | | |
|`CU_FILE_MEMORY_ALREADY_REGISTERED`|12.9| | | |`hipFileMemoryAlreadyRegistered`|7.2.0| | | | | |
|`CU_FILE_MEMORY_NOT_REGISTERED`|12.9| | | |`hipFileMemoryNotRegistered`|7.2.0| | | | | |
|`CU_FILE_NFS_SUPPORTED`|12.9| | | |`hipFileNFSSupported`|7.2.0| | | | | |
|`CU_FILE_NVFS_DRIVER_ERROR`|12.9| | | |`hipFileDriverError`|7.2.0| | | | | |
|`CU_FILE_NVFS_SETUP_ERROR`|12.9| | | |`hipFileDriverSetupError`|7.2.0| | | | | |
|`CU_FILE_NVMEOF_SUPPORTED`|12.9| | | |`hipFileNVMeoFSupported`|7.2.0| | | | | |
|`CU_FILE_NVMESH_SUPPORTED`|12.9| | | |`hipFileNVMeshSupported`|7.2.0| | | | | |
|`CU_FILE_NVME_P2P_SUPPORTED`|12.9| | | |`hipFileNVMeP2PSsupported`|7.2.0| | | | | |
|`CU_FILE_NVME_SUPPORTED`|12.9| | | |`hipFileNVMeSupported`|7.2.0| | | | | |
|`CU_FILE_PARALLEL_IO_SUPPORTED`|12.9| | | |`hipFileParallelIOSupported`|7.2.0| | | | | |
|`CU_FILE_PERMISSION_DENIED`|12.9| | | |`hipFilePermissionDenied`|7.2.0| | | | | |
|`CU_FILE_PLATFORM_NOT_SUPPORTED`|12.9| | | |`hipFilePlatformNotSupported`|7.2.0| | | | | |
|`CU_FILE_RDMA_REGISTER`|12.9| | | |`HIPFILE_RDMA_REGISTER`|7.2.0| | | | | |
|`CU_FILE_RDMA_RELAXED_ORDERING`|12.9| | | |`HIPFILE_RDMA_RELAXED_ORDERING`|7.2.0| | | | | |
|`CU_FILE_SCALEFLUX_CSD_SUPPORTED`|12.9| | | |`hipFileScaleFluxCSDSupported`|7.2.0| | | | | |
|`CU_FILE_SCSI_SUPPORTED`|12.9| | | |`hipFileSCSISupported`|7.2.0| | | | | |
|`CU_FILE_STREAMS_SUPPORTED`|12.9| | | |`hipFileStreamsSupported`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_BUF_OFFSET`|12.9| | | |`HIPFILE_STREAM_FIXED_BUF_OFFSET`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_FILE_OFFSET`|12.9| | | |`HIPFILE_STREAM_FIXED_FILE_OFFSET`|7.2.0| | | | | |
|`CU_FILE_STREAM_FIXED_FILE_SIZE`|12.9| | | |`HIPFILE_STREAM_FIXED_FILE_SIZE`|7.2.0| | | | | |
|`CU_FILE_STREAM_PAGE_ALIGNED_INPUTS`|12.9| | | |`HIPFILE_STREAM_PAGE_ALIGNED_INPUTS`|7.2.0| | | | | |
|`CU_FILE_SUCCESS`|12.9| | | |`hipFileSuccess`|7.2.0| | | | | |
|`CU_FILE_USE_POLL_MODE`|12.9| | | |`hipFileUsePollMode`|7.2.0| | | | | |
|`CU_FILE_WEKAFS_SUPPORTED`|12.9| | | |`hipFileWekaFSSupported`|7.2.0| | | | | |
|`CUfileBatchHandle_t`|12.9| | | |`hipFileBatchHandle_t`|7.2.0| | | | | |
|`CUfileBatchMode_t`|12.9| | | |`hipFileBatchMode_t`|7.2.0| | | | | |
|`CUfileDescr_t`|12.9| | | |`hipFileDescr_t`|7.2.0| | | | | |
|`CUfileDriverControlFlags_t`|12.9| | | |`hipFileDriverControlFlags_t`|7.2.0| | | | | |
|`CUfileDriverStatusFlags_t`|12.9| | | |`hipFileDriverStatusFlags_t`|7.2.0| | | | | |
|`CUfileDrvProps_t`|12.9| | | |`hipFileDriverProps_t`|7.2.0| | | | | |
|`CUfileError_t`|12.9| | | |`hipFileError_t`|7.2.0| | | | | |
|`CUfileFSOps_t`| | | | |`hipFileFSOps_t`|7.2.0| | | | | |
|`CUfileFeatureFlags_t`|12.9| | | |`hipFileFeatureFlags_t`|7.2.0| | | | | |
|`CUfileFileHandleType`|12.9| | | |`hipFileFileHandleType`|7.2.0| | | | | |
|`CUfileHandle_t`|12.9| | | |`hipFileHandle_t`|7.2.0| | | | | |
|`CUfileIOEvents_t`|12.9| | | |`hipFileIOEvents_t`|7.2.0| | | | | |
|`CUfileIOParams_t`|12.9| | | |`hipFileIOParams_t`|7.2.0| | | | | |
|`CUfileOpError`|12.9| | | |`hipFileOpError_t`|7.2.0| | | | | |
|`CUfileOpcode_t`| | | | |`hipFileOpcode_t`|7.2.0| | | | | |
|`CUfileStatus_t`| | | | |`hipFileStatus_t`|7.2.0| | | | | |
|`IS_CUDA_ERR`|12.9| | | |`IS_HIP_DRV_ERR`|7.2.0| | | | | |
|`IS_CUFILE_ERR`|12.9| | | |`IS_HIPFILE_ERR`|7.2.0| | | | | |
|`cufileRDMAInfo_t`|12.9| | | |`hipFileRDMAInfo_t`|7.2.0| | | | | |

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

