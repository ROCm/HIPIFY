# CUDA Driver API supported by HIP

## **1. CUDA Driver Data Types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUDA_ARRAY3D_2DARRAY`| |5.0| | | | | | | | |
|`CUDA_ARRAY3D_COLOR_ATTACHMENT`|10.0| | | | | | | | | |
|`CUDA_ARRAY3D_CUBEMAP`| | | | |`hipArrayCubemap`|1.7.0| | | | |
|`CUDA_ARRAY3D_DEFERRED_MAPPING`|11.6| | | | | | | | | |
|`CUDA_ARRAY3D_DEPTH_TEXTURE`| | | | | | | | | | |
|`CUDA_ARRAY3D_DESCRIPTOR`| | | | |`HIP_ARRAY3D_DESCRIPTOR`|2.7.0| | | | |
|`CUDA_ARRAY3D_DESCRIPTOR_st`| | | | |`HIP_ARRAY3D_DESCRIPTOR`|2.7.0| | | | |
|`CUDA_ARRAY3D_DESCRIPTOR_v2`|11.3| | | |`HIP_ARRAY3D_DESCRIPTOR`|2.7.0| | | | |
|`CUDA_ARRAY3D_LAYERED`| | | | |`hipArrayLayered`|1.7.0| | | | |
|`CUDA_ARRAY3D_SPARSE`|11.1| | | | | | | | | |
|`CUDA_ARRAY3D_SURFACE_LDST`| | | | |`hipArraySurfaceLoadStore`|1.7.0| | | | |
|`CUDA_ARRAY3D_TEXTURE_GATHER`| | | | |`hipArrayTextureGather`|1.7.0| | | | |
|`CUDA_ARRAY_DESCRIPTOR`| | | | |`HIP_ARRAY_DESCRIPTOR`|1.7.0| | | | |
|`CUDA_ARRAY_DESCRIPTOR_st`| | | | |`HIP_ARRAY_DESCRIPTOR`|1.7.0| | | | |
|`CUDA_ARRAY_DESCRIPTOR_v1`| | | | |`HIP_ARRAY_DESCRIPTOR`|1.7.0| | | | |
|`CUDA_ARRAY_DESCRIPTOR_v1_st`| | | | |`HIP_ARRAY_DESCRIPTOR`|1.7.0| | | | |
|`CUDA_ARRAY_DESCRIPTOR_v2`|11.3| | | |`HIP_ARRAY_DESCRIPTOR`|1.7.0| | | | |
|`CUDA_ARRAY_MEMORY_REQUIREMENTS`|11.6| | | | | | | | | |
|`CUDA_ARRAY_MEMORY_REQUIREMENTS_st`|11.6| | | | | | | | | |
|`CUDA_ARRAY_MEMORY_REQUIREMENTS_v1`|11.6| | | | | | | | | |
|`CUDA_ARRAY_SPARSE_PROPERTIES`|11.1| | | | | | | | | |
|`CUDA_ARRAY_SPARSE_PROPERTIES_st`|11.1| | | | | | | | | |
|`CUDA_ARRAY_SPARSE_PROPERTIES_v1`|11.3| | | | | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS`|11.7| | | | | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS_st`|11.7| | |12.2| | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS_v1`|12.2| | | | | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st`|12.2| | | | | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS_v2`|12.2| | | | | | | | | |
|`CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st`|12.2| | | | | | | | | |
|`CUDA_CB`| | | | | | | | | | |
|`CUDA_CHILD_GRAPH_NODE_PARAMS`|12.2| | | |`hipChildGraphNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_CHILD_GRAPH_NODE_PARAMS_st`|12.2| | | |`hipChildGraphNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_CONDITIONAL_NODE_PARAMS`|12.3| | | | | | | | | |
|`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC`|9.0| | | |`hipCooperativeLaunchMultiDeviceNoPostSync`|3.2.0| | | | |
|`CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC`|9.0| | | |`hipCooperativeLaunchMultiDeviceNoPreSync`|3.2.0| | | | |
|`CUDA_ERROR_ALREADY_ACQUIRED`| | | | |`hipErrorAlreadyAcquired`|1.6.0| | | | |
|`CUDA_ERROR_ALREADY_MAPPED`| | | | |`hipErrorAlreadyMapped`|1.6.0| | | | |
|`CUDA_ERROR_ARRAY_IS_MAPPED`| | | | |`hipErrorArrayIsMapped`|1.6.0| | | | |
|`CUDA_ERROR_ASSERT`| | | | |`hipErrorAssert`|1.9.0| | | | |
|`CUDA_ERROR_CAPTURED_EVENT`|10.0| | | |`hipErrorCapturedEvent`|4.3.0| | | | |
|`CUDA_ERROR_CDP_NOT_SUPPORTED`|12.0| | | | | | | | | |
|`CUDA_ERROR_CDP_VERSION_MISMATCH`|12.0| | | | | | | | | |
|`CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE`|10.1| | | | | | | | | |
|`CUDA_ERROR_CONTEXT_ALREADY_CURRENT`| |3.2| | |`hipErrorContextAlreadyCurrent`|1.6.0| | | | |
|`CUDA_ERROR_CONTEXT_ALREADY_IN_USE`| | | | |`hipErrorContextAlreadyInUse`|1.6.0| | | | |
|`CUDA_ERROR_CONTEXT_IS_DESTROYED`| | | | |`hipErrorContextIsDestroyed`|4.3.0| | | | |
|`CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`|9.0| | | |`hipErrorCooperativeLaunchTooLarge`|3.2.0| | | | |
|`CUDA_ERROR_DEINITIALIZED`| | | | |`hipErrorDeinitialized`|1.6.0| | | | |
|`CUDA_ERROR_DEVICE_NOT_LICENSED`|11.1| | | | | | | | | |
|`CUDA_ERROR_DEVICE_UNAVAILABLE`|11.7| | | | | | | | | |
|`CUDA_ERROR_ECC_UNCORRECTABLE`| | | | |`hipErrorECCNotCorrectable`|1.6.0| | | | |
|`CUDA_ERROR_EXTERNAL_DEVICE`|11.4| | | | | | | | | |
|`CUDA_ERROR_FILE_NOT_FOUND`| | | | |`hipErrorFileNotFound`|1.6.0| | | | |
|`CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE`|10.2| | | |`hipErrorGraphExecUpdateFailure`|5.0.0| | | | |
|`CUDA_ERROR_HARDWARE_STACK_ERROR`| | | | | | | | | | |
|`CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`| | | | |`hipErrorHostMemoryAlreadyRegistered`|1.6.0| | | | |
|`CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED`| | | | |`hipErrorHostMemoryNotRegistered`|1.6.0| | | | |
|`CUDA_ERROR_ILLEGAL_ADDRESS`| | | | |`hipErrorIllegalAddress`|1.6.0| | | | |
|`CUDA_ERROR_ILLEGAL_INSTRUCTION`| | | | | | | | | | |
|`CUDA_ERROR_ILLEGAL_STATE`|10.0| | | |`hipErrorIllegalState`|5.0.0| | | | |
|`CUDA_ERROR_INVALID_ADDRESS_SPACE`| | | | | | | | | | |
|`CUDA_ERROR_INVALID_CLUSTER_SIZE`|11.8| | | | | | | | | |
|`CUDA_ERROR_INVALID_CONTEXT`| | | | |`hipErrorInvalidContext`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_DEVICE`| | | | |`hipErrorInvalidDevice`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_GRAPHICS_CONTEXT`| | | | |`hipErrorInvalidGraphicsContext`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_HANDLE`| | | | |`hipErrorInvalidHandle`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_IMAGE`| | | | |`hipErrorInvalidImage`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_PC`| | | | | | | | | | |
|`CUDA_ERROR_INVALID_PTX`| | | | |`hipErrorInvalidKernelFile`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_SOURCE`| | | | |`hipErrorInvalidSource`|1.6.0| | | | |
|`CUDA_ERROR_INVALID_VALUE`| | | | |`hipErrorInvalidValue`|1.6.0| | | | |
|`CUDA_ERROR_JIT_COMPILATION_DISABLED`|11.2| | | | | | | | | |
|`CUDA_ERROR_JIT_COMPILER_NOT_FOUND`|9.0| | | | | | | | | |
|`CUDA_ERROR_LAUNCH_FAILED`| | | | |`hipErrorLaunchFailure`|1.6.0| | | | |
|`CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING`| | | | | | | | | | |
|`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`| | | | |`hipErrorLaunchOutOfResources`|1.6.0| | | | |
|`CUDA_ERROR_LAUNCH_TIMEOUT`| | | | |`hipErrorLaunchTimeOut`|1.6.0| | | | |
|`CUDA_ERROR_LOSSY_QUERY`| | | | | | | | | | |
|`CUDA_ERROR_MAP_FAILED`| | | | |`hipErrorMapFailed`|1.6.0| | | | |
|`CUDA_ERROR_MISALIGNED_ADDRESS`| | | | | | | | | | |
|`CUDA_ERROR_MPS_CLIENT_TERMINATED`|11.8| | | | | | | | | |
|`CUDA_ERROR_MPS_CONNECTION_FAILED`|11.4| | | | | | | | | |
|`CUDA_ERROR_MPS_MAX_CLIENTS_REACHED`|11.4| | | | | | | | | |
|`CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED`|11.4| | | | | | | | | |
|`CUDA_ERROR_MPS_RPC_FAILURE`|11.4| | | | | | | | | |
|`CUDA_ERROR_MPS_SERVER_NOT_READY`|11.4| | | | | | | | | |
|`CUDA_ERROR_NOT_FOUND`| | | | |`hipErrorNotFound`|1.6.0| | | | |
|`CUDA_ERROR_NOT_INITIALIZED`| | | | |`hipErrorNotInitialized`|1.6.0| | | | |
|`CUDA_ERROR_NOT_MAPPED`| | | | |`hipErrorNotMapped`|1.6.0| | | | |
|`CUDA_ERROR_NOT_MAPPED_AS_ARRAY`| | | | |`hipErrorNotMappedAsArray`|1.6.0| | | | |
|`CUDA_ERROR_NOT_MAPPED_AS_POINTER`| | | | |`hipErrorNotMappedAsPointer`|1.6.0| | | | |
|`CUDA_ERROR_NOT_PERMITTED`| | | | | | | | | | |
|`CUDA_ERROR_NOT_READY`| | | | |`hipErrorNotReady`|1.6.0| | | | |
|`CUDA_ERROR_NOT_SUPPORTED`| | | | |`hipErrorNotSupported`|1.6.0| | | | |
|`CUDA_ERROR_NO_BINARY_FOR_GPU`| | | | |`hipErrorNoBinaryForGpu`|1.6.0| | | | |
|`CUDA_ERROR_NO_DEVICE`| | | | |`hipErrorNoDevice`|1.6.0| | | | |
|`CUDA_ERROR_NVLINK_UNCORRECTABLE`|8.0| | | | | | | | | |
|`CUDA_ERROR_OPERATING_SYSTEM`| | | | |`hipErrorOperatingSystem`|1.6.0| | | | |
|`CUDA_ERROR_OUT_OF_MEMORY`| | | | |`hipErrorOutOfMemory`|1.6.0| | | | |
|`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED`| | | | |`hipErrorPeerAccessAlreadyEnabled`|1.6.0| | | | |
|`CUDA_ERROR_PEER_ACCESS_NOT_ENABLED`| | | | |`hipErrorPeerAccessNotEnabled`|1.6.0| | | | |
|`CUDA_ERROR_PEER_ACCESS_UNSUPPORTED`| | | | |`hipErrorPeerAccessUnsupported`|1.6.0| | | | |
|`CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE`| | | | |`hipErrorSetOnActiveProcess`|1.6.0| | | | |
|`CUDA_ERROR_PROFILER_ALREADY_STARTED`| |5.0| | |`hipErrorProfilerAlreadyStarted`|1.6.0| | | | |
|`CUDA_ERROR_PROFILER_ALREADY_STOPPED`| |5.0| | |`hipErrorProfilerAlreadyStopped`|1.6.0| | | | |
|`CUDA_ERROR_PROFILER_DISABLED`| | | | |`hipErrorProfilerDisabled`|1.6.0| | | | |
|`CUDA_ERROR_PROFILER_NOT_INITIALIZED`| |5.0| | |`hipErrorProfilerNotInitialized`|1.6.0| | | | |
|`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`| | | | |`hipErrorSharedObjectInitFailed`|1.6.0| | | | |
|`CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`| | | | |`hipErrorSharedObjectSymbolNotFound`|1.6.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_IMPLICIT`|10.0| | | |`hipErrorStreamCaptureImplicit`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`|10.0| | | |`hipErrorStreamCaptureInvalidated`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_ISOLATION`|10.0| | | |`hipErrorStreamCaptureIsolation`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_MERGE`|10.0| | | |`hipErrorStreamCaptureMerge`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_UNJOINED`|10.0| | | |`hipErrorStreamCaptureUnjoined`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_UNMATCHED`|10.0| | | |`hipErrorStreamCaptureUnmatched`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`|10.0| | | |`hipErrorStreamCaptureUnsupported`|4.3.0| | | | |
|`CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD`|10.1| | | |`hipErrorStreamCaptureWrongThread`|4.3.0| | | | |
|`CUDA_ERROR_STUB_LIBRARY`|11.1| | | | | | | | | |
|`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`|10.1| | | | | | | | | |
|`CUDA_ERROR_SYSTEM_NOT_READY`|10.0| | | | | | | | | |
|`CUDA_ERROR_TIMEOUT`|10.2| | | | | | | | | |
|`CUDA_ERROR_TOO_MANY_PEERS`| | | | | | | | | | |
|`CUDA_ERROR_UNKNOWN`| | | | |`hipErrorUnknown`|1.6.0| | | | |
|`CUDA_ERROR_UNMAP_FAILED`| | | | |`hipErrorUnmapFailed`|1.6.0| | | | |
|`CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC`|12.1| | | | | | | | | |
|`CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY`|11.4| | | | | | | | | |
|`CUDA_ERROR_UNSUPPORTED_LIMIT`| | | | |`hipErrorUnsupportedLimit`|1.6.0| | | | |
|`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`|11.1| | | | | | | | | |
|`CUDA_EVENT_RECORD_NODE_PARAMS`|12.2| | | |`hipEventRecordNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_EVENT_RECORD_NODE_PARAMS_st`|12.2| | | |`hipEventRecordNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_EVENT_WAIT_NODE_PARAMS`|12.2| | | |`hipEventWaitNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_EVENT_WAIT_NODE_PARAMS_st`|12.2| | | |`hipEventWaitNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_EXTERNAL_MEMORY_BUFFER_DESC`|10.0| | | |`hipExternalMemoryBufferDesc`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st`|10.0| | | |`hipExternalMemoryBufferDesc_st`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1`|11.3| | | |`hipExternalMemoryBufferDesc`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_DEDICATED`|10.0| | | |`hipExternalMemoryDedicated`|5.5.0| | | | |
|`CUDA_EXTERNAL_MEMORY_HANDLE_DESC`|10.0| | | |`hipExternalMemoryHandleDesc`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st`|10.0| | | |`hipExternalMemoryHandleDesc_st`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1`|11.3| | | |`hipExternalMemoryHandleDesc`|4.3.0| | | | |
|`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`|10.0| | | | | | | | | |
|`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st`|10.0| | | | | | | | | |
|`CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1`|11.3| | | | | | | | | |
|`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC`|10.0| | | |`hipExternalSemaphoreHandleDesc`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st`|10.0| | | |`hipExternalSemaphoreHandleDesc_st`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1`|11.3| | | |`hipExternalSemaphoreHandleDesc`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS`|10.0| | | |`hipExternalSemaphoreSignalParams`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st`|10.0| | | |`hipExternalSemaphoreSignalParams_st`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1`|11.3| | | |`hipExternalSemaphoreSignalParams`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC`|10.2| | | | | | | | | |
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS`|10.0| | | |`hipExternalSemaphoreWaitParams`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st`|10.0| | | |`hipExternalSemaphoreWaitParams_st`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1`|11.3| | | |`hipExternalSemaphoreWaitParams`|4.4.0| | | | |
|`CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC`|10.2| | | | | | | | | |
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS`|11.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st`|11.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1`|11.3| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2`|12.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st`|12.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS`|11.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS_st`|11.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1`|11.3| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2`|12.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st`|12.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`CUDA_GRAPH_INSTANTIATE_ERROR`|12.0| | | |`hipGraphInstantiateError`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`|11.4| | | |`hipGraphInstantiateFlagAutoFreeOnLaunch`|5.2.0| | | | |
|`CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH`|12.0| | | |`hipGraphInstantiateFlagDeviceLaunch`|5.6.0| | | | |
|`CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD`|12.0| | | |`hipGraphInstantiateFlagUpload`|5.6.0| | | | |
|`CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY`|11.7| | | |`hipGraphInstantiateFlagUseNodePriority`|5.6.0| | | | |
|`CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE`|12.0| | | |`hipGraphInstantiateInvalidStructure`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED`|12.0| | | |`hipGraphInstantiateMultipleDevicesNotSupported`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED`|12.0| | | |`hipGraphInstantiateNodeOperationNotSupported`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_PARAMS`|12.0| | | |`hipGraphInstantiateParams`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_PARAMS_st`|12.0| | | |`hipGraphInstantiateParams`|6.1.0| | | |6.1.0|
|`CUDA_GRAPH_INSTANTIATE_SUCCESS`|12.0| | | |`hipGraphInstantiateSuccess`|6.1.0| | | |6.1.0|
|`CUDA_HOST_NODE_PARAMS`|10.0| | | |`hipHostNodeParams`|4.3.0| | | | |
|`CUDA_HOST_NODE_PARAMS_st`|10.0| | | |`hipHostNodeParams`|4.3.0| | | | |
|`CUDA_HOST_NODE_PARAMS_v1`|11.3| | | |`hipHostNodeParams`|4.3.0| | | | |
|`CUDA_HOST_NODE_PARAMS_v2`|12.2| | | | | | | | | |
|`CUDA_HOST_NODE_PARAMS_v2_st`|12.2| | | | | | | | | |
|`CUDA_KERNEL_NODE_PARAMS`|10.0| | | |`hipKernelNodeParams`|4.3.0| | | | |
|`CUDA_KERNEL_NODE_PARAMS_st`|10.0| | | |`hipKernelNodeParams`|4.3.0| | | | |
|`CUDA_KERNEL_NODE_PARAMS_v1`|11.3| | | |`hipKernelNodeParams`|4.3.0| | | | |
|`CUDA_KERNEL_NODE_PARAMS_v2`|12.0| | | | | | | | | |
|`CUDA_KERNEL_NODE_PARAMS_v2_st`|12.0| | | | | | | | | |
|`CUDA_KERNEL_NODE_PARAMS_v3`|12.2| | | | | | | | | |
|`CUDA_KERNEL_NODE_PARAMS_v3_st`|12.2| | | | | | | | | |
|`CUDA_LAUNCH_PARAMS`|9.0| | | |`hipFunctionLaunchParams`|5.5.0| | | | |
|`CUDA_LAUNCH_PARAMS_st`|9.0| | | |`hipFunctionLaunchParams_t`|5.5.0| | | | |
|`CUDA_LAUNCH_PARAMS_v1`|11.3| | | |`hipFunctionLaunchParams`|5.5.0| | | | |
|`CUDA_MEMCPY2D`| | | | |`hip_Memcpy2D`|1.7.0| | | | |
|`CUDA_MEMCPY2D_st`| | | | |`hip_Memcpy2D`|1.7.0| | | | |
|`CUDA_MEMCPY2D_v1`| | | | |`hip_Memcpy2D`|1.7.0| | | | |
|`CUDA_MEMCPY2D_v1_st`| | | | |`hip_Memcpy2D`|1.7.0| | | | |
|`CUDA_MEMCPY2D_v2`|11.3| | | |`hip_Memcpy2D`|1.7.0| | | | |
|`CUDA_MEMCPY3D`| | | | |`HIP_MEMCPY3D`|3.2.0| | | | |
|`CUDA_MEMCPY3D_PEER`| | | | | | | | | | |
|`CUDA_MEMCPY3D_PEER_st`| | | | | | | | | | |
|`CUDA_MEMCPY3D_PEER_v1`|11.3| | | | | | | | | |
|`CUDA_MEMCPY3D_st`| | | | |`HIP_MEMCPY3D`|3.2.0| | | | |
|`CUDA_MEMCPY3D_v1`| | | | |`HIP_MEMCPY3D`|3.2.0| | | | |
|`CUDA_MEMCPY3D_v1_st`| | | | |`HIP_MEMCPY3D`|3.2.0| | | | |
|`CUDA_MEMCPY3D_v2`|11.3| | | |`HIP_MEMCPY3D`|3.2.0| | | | |
|`CUDA_MEMCPY_NODE_PARAMS`|12.2| | | |`hipMemcpyNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_MEMCPY_NODE_PARAMS_st`|12.2| | | |`hipMemcpyNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_MEMSET_NODE_PARAMS`|10.0| | | |`hipMemsetParams`|4.3.0| | | | |
|`CUDA_MEMSET_NODE_PARAMS_st`|10.0| | | |`hipMemsetParams`|4.3.0| | | | |
|`CUDA_MEMSET_NODE_PARAMS_v1`|11.3| | | |`hipMemsetParams`|4.3.0| | | | |
|`CUDA_MEMSET_NODE_PARAMS_v2`|12.2| | | | | | | | | |
|`CUDA_MEMSET_NODE_PARAMS_v2_st`|12.2| | | | | | | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS`|11.4| | | |`hipMemAllocNodeParams`|5.5.0| | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS_st`|11.4| | |12.2|`hipMemAllocNodeParams`|5.5.0| | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS_v1`|12.2| | | |`hipMemAllocNodeParams`|5.5.0| | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS_v1_st`|12.2| | | |`hipMemAllocNodeParams`|5.5.0| | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS_v2`|12.2| | | | | | | | | |
|`CUDA_MEM_ALLOC_NODE_PARAMS_v2_st`|12.2| | | | | | | | | |
|`CUDA_MEM_FREE_NODE_PARAMS`|12.2| | | |`hipMemFreeNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_MEM_FREE_NODE_PARAMS_st`|12.2| | | |`hipMemFreeNodeParams`|6.1.0| | | |6.1.0|
|`CUDA_NVSCISYNC_ATTR_SIGNAL`|10.2| | | | | | | | | |
|`CUDA_NVSCISYNC_ATTR_WAIT`|10.2| | | | | | | | | |
|`CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS`|11.1| | | | | | | | | |
|`CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum`|11.1| | | | | | | | | |
|`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS`| | | | | | | | | | |
|`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st`| | | | | | | | | | |
|`CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1`|11.3| | | | | | | | | |
|`CUDA_RESOURCE_DESC`| | | | |`HIP_RESOURCE_DESC`|3.5.0| | | | |
|`CUDA_RESOURCE_DESC_st`| | | | |`HIP_RESOURCE_DESC_st`|3.5.0| | | | |
|`CUDA_RESOURCE_DESC_v1`|11.3| | | |`HIP_RESOURCE_DESC`|3.5.0| | | | |
|`CUDA_RESOURCE_VIEW_DESC`| | | | |`HIP_RESOURCE_VIEW_DESC`|3.5.0| | | | |
|`CUDA_RESOURCE_VIEW_DESC_st`| | | | |`HIP_RESOURCE_VIEW_DESC_st`|3.5.0| | | | |
|`CUDA_RESOURCE_VIEW_DESC_v1`|11.3| | | |`HIP_RESOURCE_VIEW_DESC`|3.5.0| | | | |
|`CUDA_SUCCESS`| | | | |`hipSuccess`|1.5.0| | | | |
|`CUDA_TEXTURE_DESC`| | | | |`HIP_TEXTURE_DESC`|3.5.0| | | | |
|`CUDA_TEXTURE_DESC_st`| | | | |`HIP_TEXTURE_DESC_st`|3.5.0| | | | |
|`CUDA_TEXTURE_DESC_v1`|11.3| | | |`HIP_TEXTURE_DESC`|3.5.0| | | | |
|`CUGLDeviceList`| | | | |`hipGLDeviceList`|4.4.0| | | | |
|`CUGLDeviceList_enum`| | | | |`hipGLDeviceList`|4.4.0| | | | |
|`CUGLmap_flags`| | | | | | | | | | |
|`CUGLmap_flags_enum`| | | | | | | | | | |
|`CUGPUDirectRDMAWritesOrdering`|11.3| | | |`hipGPUDirectRDMAWritesOrdering`|6.1.0| | | |6.1.0|
|`CUGPUDirectRDMAWritesOrdering_enum`|11.3| | | |`hipGPUDirectRDMAWritesOrdering`|6.1.0| | | |6.1.0|
|`CU_ACCESS_PROPERTY_NORMAL`|11.0| | | |`hipAccessPropertyNormal`|5.2.0| | | | |
|`CU_ACCESS_PROPERTY_PERSISTING`|11.0| | | |`hipAccessPropertyPersisting`|5.2.0| | | | |
|`CU_ACCESS_PROPERTY_STREAMING`|11.0| | | |`hipAccessPropertyStreaming`|5.2.0| | | | |
|`CU_AD_FORMAT_BC1_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC1_UNORM_SRGB`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC2_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC2_UNORM_SRGB`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC3_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC3_UNORM_SRGB`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC4_SNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC4_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC5_SNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC5_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC6H_SF16`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC6H_UF16`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC7_UNORM`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_BC7_UNORM_SRGB`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_FLOAT`| | | | |`HIP_AD_FORMAT_FLOAT`|1.7.0| | | | |
|`CU_AD_FORMAT_HALF`| | | | |`HIP_AD_FORMAT_HALF`|1.7.0| | | | |
|`CU_AD_FORMAT_NV12`|11.2| | | | | | | | | |
|`CU_AD_FORMAT_SIGNED_INT16`| | | | |`HIP_AD_FORMAT_SIGNED_INT16`|1.7.0| | | | |
|`CU_AD_FORMAT_SIGNED_INT32`| | | | |`HIP_AD_FORMAT_SIGNED_INT32`|1.7.0| | | | |
|`CU_AD_FORMAT_SIGNED_INT8`| | | | |`HIP_AD_FORMAT_SIGNED_INT8`|1.7.0| | | | |
|`CU_AD_FORMAT_SNORM_INT16X1`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_SNORM_INT16X2`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_SNORM_INT16X4`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_SNORM_INT8X1`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_SNORM_INT8X2`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_SNORM_INT8X4`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT16X1`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT16X2`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT16X4`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT8X1`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT8X2`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNORM_INT8X4`|11.5| | | | | | | | | |
|`CU_AD_FORMAT_UNSIGNED_INT16`| | | | |`HIP_AD_FORMAT_UNSIGNED_INT16`|1.7.0| | | | |
|`CU_AD_FORMAT_UNSIGNED_INT32`| | | | |`HIP_AD_FORMAT_UNSIGNED_INT32`|1.7.0| | | | |
|`CU_AD_FORMAT_UNSIGNED_INT8`| | | | |`HIP_AD_FORMAT_UNSIGNED_INT8`|1.7.0| | | | |
|`CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL`|11.1| | | | | | | | | |
|`CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL`|11.1| | | |`hipArraySparseSubresourceTypeMiptail`|5.2.0| | | | |
|`CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL`|11.1| | | |`hipArraySparseSubresourceTypeSparseLevel`|5.2.0| | | | |
|`CU_CLUSTER_SCHEDULING_POLICY_DEFAULT`|11.8| | | | | | | | | |
|`CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING`|11.8| | | | | | | | | |
|`CU_CLUSTER_SCHEDULING_POLICY_SPREAD`|11.8| | | | | | | | | |
|`CU_COMPUTEMODE_DEFAULT`| | | | |`hipComputeModeDefault`|1.9.0| | | | |
|`CU_COMPUTEMODE_EXCLUSIVE`| | | |8.0|`hipComputeModeExclusive`|1.9.0| | | | |
|`CU_COMPUTEMODE_EXCLUSIVE_PROCESS`| | | | |`hipComputeModeExclusiveProcess`|2.0.0| | | | |
|`CU_COMPUTEMODE_PROHIBITED`| | | | |`hipComputeModeProhibited`|1.9.0| | | | |
|`CU_COMPUTE_ACCELERATED_TARGET_BASE`|12.0| | | | | | | | | |
|`CU_COREDUMP_ENABLE_ON_EXCEPTION`|12.1| | | | | | | | | |
|`CU_COREDUMP_ENABLE_USER_TRIGGER`|12.1| | | | | | | | | |
|`CU_COREDUMP_FILE`|12.1| | | | | | | | | |
|`CU_COREDUMP_LIGHTWEIGHT`|12.1| | | | | | | | | |
|`CU_COREDUMP_MAX`|12.1| | | | | | | | | |
|`CU_COREDUMP_PIPE`|12.1| | | | | | | | | |
|`CU_COREDUMP_TRIGGER_HOST`|12.1| | | | | | | | | |
|`CU_CTX_BLOCKING_SYNC`| |4.0| | |`hipDeviceScheduleBlockingSync`|1.6.0| | | | |
|`CU_CTX_COREDUMP_ENABLE`|12.1| | | | | | | | | |
|`CU_CTX_FLAGS_MASK`| | | | | | | | | | |
|`CU_CTX_LMEM_RESIZE_TO_MAX`| | | | |`hipDeviceLmemResizeToMax`|1.6.0| | | | |
|`CU_CTX_MAP_HOST`| | | | |`hipDeviceMapHost`|1.6.0| | | | |
|`CU_CTX_SCHED_AUTO`| | | | |`hipDeviceScheduleAuto`|1.6.0| | | | |
|`CU_CTX_SCHED_BLOCKING_SYNC`| | | | |`hipDeviceScheduleBlockingSync`|1.6.0| | | | |
|`CU_CTX_SCHED_MASK`| | | | |`hipDeviceScheduleMask`|1.6.0| | | | |
|`CU_CTX_SCHED_SPIN`| | | | |`hipDeviceScheduleSpin`|1.6.0| | | | |
|`CU_CTX_SCHED_YIELD`| | | | |`hipDeviceScheduleYield`|1.6.0| | | | |
|`CU_CTX_SYNC_MEMOPS`|12.1| | | | | | | | | |
|`CU_CTX_USER_COREDUMP_ENABLE`|12.1| | | | | | | | | |
|`CU_CUBEMAP_FACE_NEGATIVE_X`| | | | | | | | | | |
|`CU_CUBEMAP_FACE_NEGATIVE_Y`| | | | | | | | | | |
|`CU_CUBEMAP_FACE_NEGATIVE_Z`| | | | | | | | | | |
|`CU_CUBEMAP_FACE_POSITIVE_X`| | | | | | | | | | |
|`CU_CUBEMAP_FACE_POSITIVE_Y`| | | | | | | | | | |
|`CU_CUBEMAP_FACE_POSITIVE_Z`| | | | | | | | | | |
|`CU_D3D10_DEVICE_LIST_ALL`| | | | | | | | | | |
|`CU_D3D10_DEVICE_LIST_CURRENT_FRAME`| | | | | | | | | | |
|`CU_D3D10_DEVICE_LIST_NEXT_FRAME`| | | | | | | | | | |
|`CU_D3D10_MAPRESOURCE_FLAGS_NONE`| | | | | | | | | | |
|`CU_D3D10_MAPRESOURCE_FLAGS_READONLY`| | | | | | | | | | |
|`CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD`| | | | | | | | | | |
|`CU_D3D10_REGISTER_FLAGS_ARRAY`| | | | | | | | | | |
|`CU_D3D10_REGISTER_FLAGS_NONE`| | | | | | | | | | |
|`CU_D3D11_DEVICE_LIST_ALL`| | | | | | | | | | |
|`CU_D3D11_DEVICE_LIST_CURRENT_FRAME`| | | | | | | | | | |
|`CU_D3D11_DEVICE_LIST_NEXT_FRAME`| | | | | | | | | | |
|`CU_D3D9_DEVICE_LIST_ALL`| | | | | | | | | | |
|`CU_D3D9_DEVICE_LIST_CURRENT_FRAME`| | | | | | | | | | |
|`CU_D3D9_DEVICE_LIST_NEXT_FRAME`| | | | | | | | | | |
|`CU_D3D9_MAPRESOURCE_FLAGS_NONE`| | | | | | | | | | |
|`CU_D3D9_MAPRESOURCE_FLAGS_READONLY`| | | | | | | | | | |
|`CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD`| | | | | | | | | | |
|`CU_D3D9_REGISTER_FLAGS_ARRAY`| | | | | | | | | | |
|`CU_D3D9_REGISTER_FLAGS_NONE`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT`| | | | |`hipDeviceAttributeAsyncEngineCount`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`|9.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY`| | | | |`hipDeviceAttributeCanMapHostMemory`|2.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER`| |5.0| | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS`|9.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1`|12.0|12.0| | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2`|11.7| | |12.0| | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`|9.0| | | |`hipDeviceAttributeCanUseHostPointerForRegisteredMem`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`|9.0| | |12.0| | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1`|12.0|12.0| | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR`|9.0| | | |`hipDeviceAttributeCanUseStreamWaitValue`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1`|12.0|12.0| | |`hipDeviceAttributeCanUseStreamWaitValue`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2`|11.7| | |12.0| | | | | | |
|`CU_DEVICE_ATTRIBUTE_CLOCK_RATE`| | | | |`hipDeviceAttributeClockRate`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH`|11.8| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`| | | | |`hipDeviceAttributeComputeCapabilityMajor`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`| | | | |`hipDeviceAttributeComputeCapabilityMinor`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_COMPUTE_MODE`| | | | |`hipDeviceAttributeComputeMode`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED`|8.0| | | |`hipDeviceAttributeComputePreemptionSupported`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`| | | | |`hipDeviceAttributeConcurrentKernels`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS`|8.0| | | |`hipDeviceAttributeConcurrentManagedAccess`|3.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`|9.0| | | |`hipDeviceAttributeCooperativeLaunch`|2.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH`|9.0| | | |`hipDeviceAttributeCooperativeMultiDeviceLaunch`|2.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED`|11.6| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST`|9.2| | | |`hipDeviceAttributeDirectManagedMemAccessFromHost`|3.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED`|11.7| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_ECC_ENABLED`| | | | |`hipDeviceAttributeEccEnabled`|2.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`|11.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED`| | | | |`hipDeviceAttributeGlobalL1CacheSupported`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH`| | | | |`hipDeviceAttributeMemoryBusWidth`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS`|11.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED`|11.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`|11.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING`|11.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_GPU_OVERLAP`| |5.0| | |`hipDeviceAttributeAsyncEngineCount`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`|12.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED`|10.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED`|10.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED`|10.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED`|8.0| | | |`hipDeviceAttributeHostNativeAtomicSupported`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`|12.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`|9.2| | | |`hipDeviceAttributeHostRegisterSupported`|6.0.0| | | | |
|`CU_DEVICE_ATTRIBUTE_INTEGRATED`| | | | |`hipDeviceAttributeIntegrated`|1.9.0| | | | |
|`CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED`|12.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT`| | | | |`hipDeviceAttributeKernelExecTimeout`|2.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`| | | | |`hipDeviceAttributeL2CacheSize`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED`| | | | |`hipDeviceAttributeLocalL1CacheSupported`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`| | | | |`hipDeviceAttributeManagedMemory`|3.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxSurface1DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH`| | | | |`hipDeviceAttributeMaxSurface1D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT`| | | | |`hipDeviceAttributeMaxSurface2D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT`| | | | |`hipDeviceAttributeMaxSurface2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxSurface2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH`| | | | |`hipDeviceAttributeMaxSurface2D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxSurfaceCubemapLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH`| | | | |`hipDeviceAttributeMaxSurfaceCubemap`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxTexture1DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH`| |11.2| | |`hipDeviceAttributeMaxTexture1DLinear`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH`| | | | |`hipDeviceAttributeMaxTexture1DMipmap`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH`| | | | |`hipDeviceAttributeMaxTexture1DWidth`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT`| |5.0| | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES`| |5.0| | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH`| |5.0| | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture2DGather`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH`| | | | |`hipDeviceAttributeMaxTexture2DGather`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture2DHeight`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture2DMipmap`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH`| | | | |`hipDeviceAttributeMaxTexture2DMipmap`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH`| | | | |`hipDeviceAttributeMaxTexture2DWidth`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH`| | | | |`hipDeviceAttributeMaxTexture3DDepth`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT`| | | | |`hipDeviceAttributeMaxTexture3DHeight`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH`| | | | |`hipDeviceAttributeMaxTexture3DWidth`|2.7.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS`| | | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH`| | | | |`hipDeviceAttributeMaxTextureCubemapLayered`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH`| | | | |`hipDeviceAttributeMaxTextureCubemap`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE`|11.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`|11.0| | | |`hipDeviceAttributeMaxBlocksPerMultiprocessor`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X`| | | | |`hipDeviceAttributeMaxBlockDimX`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y`| | | | |`hipDeviceAttributeMaxBlockDimY`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z`| | | | |`hipDeviceAttributeMaxBlockDimZ`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`| | | | |`hipDeviceAttributeMaxGridDimX`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y`| | | | |`hipDeviceAttributeMaxGridDimY`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z`| | | | |`hipDeviceAttributeMaxGridDimZ`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE`|11.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_PITCH`| | | | |`hipDeviceAttributeMaxPitch`|2.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK`| | | | |`hipDeviceAttributeMaxRegistersPerBlock`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR`| | | | |`hipDeviceAttributeMaxRegistersPerMultiprocessor`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`| | | | |`hipDeviceAttributeMaxSharedMemoryPerBlock`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`|9.0| | | |`hipDeviceAttributeSharedMemPerBlockOptin`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`| | | | |`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`| | | | |`hipDeviceAttributeMaxThreadsPerBlock`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR`| | | | |`hipDeviceAttributeMaxThreadsPerMultiProcessor`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE`| | | | |`hipDeviceAttributeMemoryClockRate`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`|11.2| | | |`hipDeviceAttributeMemoryPoolsSupported`|5.2.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES`|11.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT`|12.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MPS_ENABLED`|12.3| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`|12.1| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`| | | | |`hipDeviceAttributeMultiprocessorCount`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD`| | | | |`hipDeviceAttributeIsMultiGpuBoard`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID`| | | | |`hipDeviceAttributeMultiGpuBoardGroupId`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_NUMA_CONFIG`|12.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_NUMA_ID`|12.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`|8.0| | | |`hipDeviceAttributePageableMemoryAccess`|3.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`|9.2| | | |`hipDeviceAttributePageableMemoryAccessUsesHostPageTables`|3.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_PCI_BUS_ID`| | | | |`hipDeviceAttributePciBusId`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID`| | | | |`hipDeviceAttributePciDeviceId`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID`| | | | |`hipDeviceAttributePciDomainID`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`|11.1| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK`| |5.0| | |`hipDeviceAttributeMaxRegistersPerBlock`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK`|11.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK`| |5.0| | |`hipDeviceAttributeMaxSharedMemoryPerBlock`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`|8.0| | | |`hipDeviceAttributeSingleToDoublePrecisionPerfRatio`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED`|11.1| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED`| | | | |`hipDeviceAttributeStreamPrioritiesSupported`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT`| | | | |`hipDeviceAttributeSurfaceAlignment`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_TCC_DRIVER`| | | | |`hipDeviceAttributeTccDriver`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED`|12.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT`| | | | |`hipDeviceAttributeTextureAlignment`|2.10.0| | | | |
|`CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT`| | | | |`hipDeviceAttributeTexturePitchAlignment`|3.2.0| | | | |
|`CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED`|11.2| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY`| | | | |`hipDeviceAttributeTotalConstantMemory`|1.6.0| | | | |
|`CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING`| | | | |`hipDeviceAttributeUnifiedAddressing`|4.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS`|12.0| | | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED`|10.2|11.2| | | | | | | | |
|`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`|11.2| | | |`hipDeviceAttributeVirtualMemoryManagementSupported`|5.3.0| | | | |
|`CU_DEVICE_ATTRIBUTE_WARP_SIZE`| | | | |`hipDeviceAttributeWarpSize`|1.6.0| | | | |
|`CU_DEVICE_CPU`|8.0| | | |`hipCpuDeviceId`|3.7.0| | | | |
|`CU_DEVICE_INVALID`|8.0| | | |`hipInvalidDeviceId`|3.7.0| | | | |
|`CU_DEVICE_NUMA_CONFIG_NONE`|12.2| | | | | | | | | |
|`CU_DEVICE_NUMA_CONFIG_NUMA_NODE`|12.2| | | | | | | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED`|10.1|10.1| | |`hipDevP2PAttrHipArrayAccessSupported`|3.8.0| | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED`|8.0| | | |`hipDevP2PAttrAccessSupported`|3.8.0| | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_ARRAY_ACCESS_ACCESS_SUPPORTED`|9.2|10.0| |10.1|`hipDevP2PAttrHipArrayAccessSupported`|3.8.0| | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED`|10.0| | | |`hipDevP2PAttrHipArrayAccessSupported`|3.8.0| | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED`|8.0| | | |`hipDevP2PAttrNativeAtomicSupported`|3.8.0| | | | |
|`CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK`|8.0| | | |`hipDevP2PAttrPerformanceRank`|3.8.0| | | | |
|`CU_EGL_COLOR_FORMAT_A`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_ABGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_ARGB`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_AYUV`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_AYUV_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER10_BGGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER10_GBRG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER10_GRBG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER10_RGGB`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER12_BGGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER12_GBRG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER12_GRBG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER12_RGGB`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER14_BGGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER14_GBRG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER14_GRBG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER14_RGGB`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER20_BGGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER20_GBRG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER20_GRBG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER20_RGGB`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_BGGR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_GBRG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_GRBG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR`|9.2| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG`|9.2| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG`|9.2| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB`|9.2| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BAYER_RGGB`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BGR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_BGRA`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_L`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_MAX`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_R`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_RG`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_RGB`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_RGBA`| | | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_UYVY_422`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_UYVY_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_VYUY_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV420_PLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV422_PLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV444_PLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUVA_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUV_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUYV_422`|9.0| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YUYV_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU420_PLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU422_PLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU444_PLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER`|9.1| | | | | | | | | |
|`CU_EGL_COLOR_FORMAT_YVYU_ER`|9.1| | | | | | | | | |
|`CU_EGL_FRAME_TYPE_ARRAY`|9.0| | | | | | | | | |
|`CU_EGL_FRAME_TYPE_PITCH`|9.0| | | | | | | | | |
|`CU_EGL_RESOURCE_LOCATION_SYSMEM`|9.0| | | | | | | | | |
|`CU_EGL_RESOURCE_LOCATION_VIDMEM`|9.0| | | | | | | | | |
|`CU_EVENT_BLOCKING_SYNC`| | | | |`hipEventBlockingSync`|1.6.0| | | | |
|`CU_EVENT_DEFAULT`| | | | |`hipEventDefault`|1.6.0| | | | |
|`CU_EVENT_DISABLE_TIMING`| | | | |`hipEventDisableTiming`|1.6.0| | | | |
|`CU_EVENT_INTERPROCESS`| | | | |`hipEventInterprocess`|1.6.0| | | | |
|`CU_EVENT_RECORD_DEFAULT`|11.1| | | | | | | | | |
|`CU_EVENT_RECORD_EXTERNAL`|11.1| | | | | | | | | |
|`CU_EVENT_SCHED_AUTO`|11.8| | | | | | | | | |
|`CU_EVENT_SCHED_BLOCKING_SYNC`|11.8| | | | | | | | | |
|`CU_EVENT_SCHED_SPIN`|11.8| | | | | | | | | |
|`CU_EVENT_SCHED_YIELD`|11.8| | | | | | | | | |
|`CU_EVENT_WAIT_DEFAULT`|11.1| | | | | | | | | |
|`CU_EVENT_WAIT_EXTERNAL`|11.1| | | | | | | | | |
|`CU_EXEC_AFFINITY_TYPE_MAX`|11.4| | | | | | | | | |
|`CU_EXEC_AFFINITY_TYPE_SM_COUNT`|11.4| | | | | | | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE`|10.2| | | |`hipExternalMemoryHandleTypeD3D11Resource`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT`|10.2| | | |`hipExternalMemoryHandleTypeD3D11ResourceKmt`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`|10.0| | | |`hipExternalMemoryHandleTypeD3D12Heap`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`|10.0| | | |`hipExternalMemoryHandleTypeD3D12Resource`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF`|10.2| | | | | | | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueFd`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueWin32`|4.3.0| | | | |
|`CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueWin32Kmt`|4.3.0| | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE`|10.2| | | | | | | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX`|10.2| | | | | | | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT`|10.2| | | | | | | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`|10.0| | | |`hipExternalSemaphoreHandleTypeD3D12Fence`|4.4.0| | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC`|10.2| | | | | | | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueFd`|4.4.0| | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueWin32`|4.4.0| | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueWin32Kmt`|4.4.0| | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD`|11.2| | | | | | | | | |
|`CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32`|11.2| | | | | | | | | |
|`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptionHost`|6.1.0| | | |6.1.0|
|`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptionMemOps`|6.1.0| | | |6.1.0|
|`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX`|11.3| | | | | | | | | |
|`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES`|11.3| | | | | | | | | |
|`CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER`|11.3| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_BINARY_VERSION`| | | | |`HIP_FUNC_ATTRIBUTE_BINARY_VERSION`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_CACHE_MODE_CA`| | | | |`HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`| | | | |`HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`| | | | |`HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_MAX`| | | | |`HIP_FUNC_ATTRIBUTE_MAX`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`|9.0| | | |`HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`| | | | |`HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_NUM_REGS`| | | | |`HIP_FUNC_ATTRIBUTE_NUM_REGS`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`|9.0| | | |`HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_PTX_VERSION`| | | | |`HIP_FUNC_ATTRIBUTE_PTX_VERSION`|2.8.0| | | | |
|`CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH`|11.8| | | | | | | | | |
|`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`| | | | |`HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`|2.8.0| | | | |
|`CU_FUNC_CACHE_PREFER_EQUAL`| | | | |`hipFuncCachePreferEqual`|1.6.0| | | | |
|`CU_FUNC_CACHE_PREFER_L1`| | | | |`hipFuncCachePreferL1`|1.6.0| | | | |
|`CU_FUNC_CACHE_PREFER_NONE`| | | | |`hipFuncCachePreferNone`|1.6.0| | | | |
|`CU_FUNC_CACHE_PREFER_SHARED`| | | | |`hipFuncCachePreferShared`|1.6.0| | | | |
|`CU_GET_PROC_ADDRESS_DEFAULT`|11.3| | | | | | | | | |
|`CU_GET_PROC_ADDRESS_LEGACY_STREAM`|11.3| | | | | | | | | |
|`CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`|11.3| | | | | | | | | |
|`CU_GET_PROC_ADDRESS_SUCCESS`|12.0| | | |`HIP_GET_PROC_ADDRESS_SUCCESS`|6.1.0| | | |6.1.0|
|`CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`|12.0| | | |`HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`|6.1.0| | | |6.1.0|
|`CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`|12.0| | | |`HIP_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`|6.1.0| | | |6.1.0|
|`CU_GL_DEVICE_LIST_ALL`| | | | |`hipGLDeviceListAll`|4.4.0| | | | |
|`CU_GL_DEVICE_LIST_CURRENT_FRAME`| | | | |`hipGLDeviceListCurrentFrame`|4.4.0| | | | |
|`CU_GL_DEVICE_LIST_NEXT_FRAME`| | | | |`hipGLDeviceListNextFrame`|4.4.0| | | | |
|`CU_GL_MAP_RESOURCE_FLAGS_NONE`| | | | | | | | | | |
|`CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY`| | | | | | | | | | |
|`CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD`| | | | | | | | | | |
|`CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES`|11.3| | | |`hipGPUDirectRDMAWritesOrderingAllDevices`|6.1.0| | | |6.1.0|
|`CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE`|11.3| | | |`hipGPUDirectRDMAWritesOrderingNone`|6.1.0| | | |6.1.0|
|`CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER`|11.3| | | |`hipGPUDirectRDMAWritesOrderingOwner`|6.1.0| | | |6.1.0|
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE`| | | | | | | | | | |
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY`| | | | | | | | | | |
|`CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD`| | | | | | | | | | |
|`CU_GRAPHICS_REGISTER_FLAGS_NONE`| | | | |`hipGraphicsRegisterFlagsNone`|4.4.0| | | | |
|`CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY`| | | | |`hipGraphicsRegisterFlagsReadOnly`|4.4.0| | | | |
|`CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST`| | | | |`hipGraphicsRegisterFlagsSurfaceLoadStore`|4.4.0| | | | |
|`CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER`| | | | |`hipGraphicsRegisterFlagsTextureGather`|4.4.0| | | | |
|`CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD`| | | | |`hipGraphicsRegisterFlagsWriteDiscard`|4.4.0| | | | |
|`CU_GRAPH_COND_ASSIGN_DEFAULT`|12.3| | | | | | | | | |
|`CU_GRAPH_COND_TYPE_IF`|12.3| | | | | | | | | |
|`CU_GRAPH_COND_TYPE_WHILE`|12.3| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS`|11.7| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS`|12.3| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsEventNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO`|12.0| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsExtSemasSignalNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsExtSemasWaitNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES`|11.3| | | |`hipGraphDebugDotFlagsHandles`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsHostNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES`|11.3| | | |`hipGraphDebugDotFlagsKernelNodeAttributes`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsKernelNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsMemcpyNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS`|11.3| | | |`hipGraphDebugDotFlagsMemsetNodeParams`|5.5.0| | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS`|11.4| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS`|11.4| | | | | | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES`|11.3| | | |`hipGraphDebugDotFlagsRuntimeTypes`| | | | | |
|`CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE`|11.3| | | |`hipGraphDebugDotFlagsVerbose`|5.5.0| | | | |
|`CU_GRAPH_DEPENDENCY_TYPE_DEFAULT`|12.3| | | | | | | | | |
|`CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`|12.3| | | | | | | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR`|10.2| | | |`hipGraphExecUpdateError`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED`|11.6| | | | | | | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED`|10.2| | | |`hipGraphExecUpdateErrorFunctionChanged`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED`|10.2| | | |`hipGraphExecUpdateErrorNodeTypeChanged`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED`|10.2| | | |`hipGraphExecUpdateErrorNotSupported`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED`|10.2| | | |`hipGraphExecUpdateErrorParametersChanged`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED`|10.2| | | |`hipGraphExecUpdateErrorTopologyChanged`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE`|11.2| | | |`hipGraphExecUpdateErrorUnsupportedFunctionChange`|4.3.0| | | | |
|`CU_GRAPH_EXEC_UPDATE_SUCCESS`|10.2| | | |`hipGraphExecUpdateSuccess`|4.3.0| | | | |
|`CU_GRAPH_KERNEL_NODE_PORT_DEFAULT`|12.3| | | | | | | | | |
|`CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER`|12.3| | | | | | | | | |
|`CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC`|12.3| | | | | | | | | |
|`CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT`|11.4| | | |`hipGraphMemAttrReservedMemCurrent`|5.3.0| | | | |
|`CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH`|11.4| | | |`hipGraphMemAttrReservedMemHigh`|5.3.0| | | | |
|`CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT`|11.4| | | |`hipGraphMemAttrUsedMemCurrent`|5.3.0| | | | |
|`CU_GRAPH_MEM_ATTR_USED_MEM_HIGH`|11.4| | | |`hipGraphMemAttrUsedMemHigh`|5.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_BATCH_MEM_OP`|11.7| | | | | | | | | |
|`CU_GRAPH_NODE_TYPE_CONDITIONAL`|12.3| | | | | | | | | |
|`CU_GRAPH_NODE_TYPE_COUNT`|10.0| | |11.0|`hipGraphNodeTypeCount`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_EMPTY`|10.0| | | |`hipGraphNodeTypeEmpty`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_EVENT_RECORD`|11.1| | | |`hipGraphNodeTypeEventRecord`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL`|11.2| | | |`hipGraphNodeTypeExtSemaphoreSignal`|5.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT`|11.2| | | |`hipGraphNodeTypeExtSemaphoreWait`|5.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_GRAPH`|10.0| | | |`hipGraphNodeTypeGraph`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_HOST`|10.0| | | |`hipGraphNodeTypeHost`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_KERNEL`|10.0| | | |`hipGraphNodeTypeKernel`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_MEMCPY`|10.0| | | |`hipGraphNodeTypeMemcpy`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_MEMSET`|10.0| | | |`hipGraphNodeTypeMemset`|4.3.0| | | | |
|`CU_GRAPH_NODE_TYPE_MEM_ALLOC`|11.4| | | |`hipGraphNodeTypeMemAlloc`|5.5.0| | | | |
|`CU_GRAPH_NODE_TYPE_MEM_FREE`|11.4| | | |`hipGraphNodeTypeMemFree`|5.5.0| | | | |
|`CU_GRAPH_NODE_TYPE_WAIT_EVENT`|11.1| | | |`hipGraphNodeTypeWaitEvent`|4.3.0| | | | |
|`CU_GRAPH_USER_OBJECT_MOVE`|11.3| | | |`hipGraphUserObjectMove`|5.3.0| | | | |
|`CU_IPC_HANDLE_SIZE`| | | | |`HIP_IPC_HANDLE_SIZE`|1.6.0| | | | |
|`CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`| | | | |`hipIpcMemLazyEnablePeerAccess`|1.6.0| | | | |
|`CU_JIT_CACHE_MODE`| | | | |`HIPRTC_JIT_CACHE_MODE`|1.6.0| | | | |
|`CU_JIT_CACHE_OPTION_CA`| | | | | | | | | | |
|`CU_JIT_CACHE_OPTION_CG`| | | | | | | | | | |
|`CU_JIT_CACHE_OPTION_NONE`| | | | | | | | | | |
|`CU_JIT_ERROR_LOG_BUFFER`| | | | |`HIPRTC_JIT_ERROR_LOG_BUFFER`|1.6.0| | | | |
|`CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`| | | | |`HIPRTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`|1.6.0| | | | |
|`CU_JIT_FALLBACK_STRATEGY`| | | | |`HIPRTC_JIT_FALLBACK_STRATEGY`|1.6.0| | | | |
|`CU_JIT_FAST_COMPILE`| | | | |`HIPRTC_JIT_FAST_COMPILE`|1.6.0| | | | |
|`CU_JIT_FMA`|11.4|12.0| | | | | | | | |
|`CU_JIT_FTZ`|11.4|12.0| | | | | | | | |
|`CU_JIT_GENERATE_DEBUG_INFO`| | | | |`HIPRTC_JIT_GENERATE_DEBUG_INFO`|1.6.0| | | | |
|`CU_JIT_GENERATE_LINE_INFO`| | | | |`HIPRTC_JIT_GENERATE_LINE_INFO`|1.6.0| | | | |
|`CU_JIT_GLOBAL_SYMBOL_ADDRESSES`| | | | | | | | | | |
|`CU_JIT_GLOBAL_SYMBOL_COUNT`| | | | | | | | | | |
|`CU_JIT_GLOBAL_SYMBOL_NAMES`| | | | | | | | | | |
|`CU_JIT_INFO_LOG_BUFFER`| | | | |`HIPRTC_JIT_INFO_LOG_BUFFER`|1.6.0| | | | |
|`CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES`| | | | |`HIPRTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES`|1.6.0| | | | |
|`CU_JIT_INPUT_CUBIN`| | | | |`HIPRTC_JIT_INPUT_CUBIN`|5.3.0| | | | |
|`CU_JIT_INPUT_FATBINARY`| | | | |`HIPRTC_JIT_INPUT_FATBINARY`|5.3.0| | | | |
|`CU_JIT_INPUT_LIBRARY`| | | | |`HIPRTC_JIT_INPUT_LIBRARY`|5.3.0| | | | |
|`CU_JIT_INPUT_NVVM`|11.4|12.0| | |`HIPRTC_JIT_INPUT_NVVM`|5.3.0| | | | |
|`CU_JIT_INPUT_OBJECT`| | | | |`HIPRTC_JIT_INPUT_OBJECT`|5.3.0| | | | |
|`CU_JIT_INPUT_PTX`| | | | |`HIPRTC_JIT_INPUT_PTX`|5.3.0| | | | |
|`CU_JIT_LOG_VERBOSE`| | | | |`HIPRTC_JIT_LOG_VERBOSE`|1.6.0| | | | |
|`CU_JIT_LTO`|11.4|12.0| | | | | | | | |
|`CU_JIT_MAX_REGISTERS`| | | | |`HIPRTC_JIT_MAX_REGISTERS`|1.6.0| | | | |
|`CU_JIT_MIN_CTA_PER_SM`|12.3| | | | | | | | | |
|`CU_JIT_NEW_SM3X_OPT`| | | | |`HIPRTC_JIT_NEW_SM3X_OPT`|1.6.0| | | | |
|`CU_JIT_NUM_INPUT_TYPES`| | | | |`HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES`|5.3.0| | | | |
|`CU_JIT_NUM_OPTIONS`| | | | |`HIPRTC_JIT_NUM_OPTIONS`|1.6.0| | | | |
|`CU_JIT_OPTIMIZATION_LEVEL`| | | | |`HIPRTC_JIT_OPTIMIZATION_LEVEL`|1.6.0| | | | |
|`CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES`|11.7|12.0| | | | | | | | |
|`CU_JIT_POSITION_INDEPENDENT_CODE`|12.0| | | | | | | | | |
|`CU_JIT_PREC_DIV`|11.4|12.0| | | | | | | | |
|`CU_JIT_PREC_SQRT`|11.4|12.0| | | | | | | | |
|`CU_JIT_REFERENCED_KERNEL_COUNT`|11.7|12.0| | | | | | | | |
|`CU_JIT_REFERENCED_KERNEL_NAMES`|11.7|12.0| | | | | | | | |
|`CU_JIT_REFERENCED_VARIABLE_COUNT`|11.7|12.0| | | | | | | | |
|`CU_JIT_REFERENCED_VARIABLE_NAMES`|11.7|12.0| | | | | | | | |
|`CU_JIT_TARGET`| | | | |`HIPRTC_JIT_TARGET`|1.6.0| | | | |
|`CU_JIT_TARGET_FROM_CUCONTEXT`| | | | |`HIPRTC_JIT_TARGET_FROM_HIPCONTEXT`|1.6.0| | | | |
|`CU_JIT_THREADS_PER_BLOCK`| | | | |`HIPRTC_JIT_THREADS_PER_BLOCK`|1.6.0| | | | |
|`CU_JIT_WALL_TIME`| | | | |`HIPRTC_JIT_WALL_TIME`|1.6.0| | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW`|11.0| | | |`hipKernelNodeAttributeAccessPolicyWindow`|5.2.0| | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION`|11.8| | | | | | | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`|11.8| | | | | | | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE`|11.0| | | |`hipKernelNodeAttributeCooperative`|5.2.0| | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN`|12.0| | | | | | | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP`|12.0| | | | | | | | | |
|`CU_KERNEL_NODE_ATTRIBUTE_PRIORITY`|11.7| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_COOPERATIVE`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_IGNORE`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`|12.3| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_MAX`|12.1| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`|12.0| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP`|12.0| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_PRIORITY`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`|11.8| | | | | | | | | |
|`CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY`|11.8| | | | | | | | | |
|`CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT`|12.0| | | | | | | | | |
|`CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE`|12.0| | | | | | | | | |
|`CU_LAUNCH_PARAM_BUFFER_POINTER`| | | | |`HIP_LAUNCH_PARAM_BUFFER_POINTER`|1.6.0| | | | |
|`CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT`|11.7| | | | | | | | | |
|`CU_LAUNCH_PARAM_BUFFER_SIZE`| | | | |`HIP_LAUNCH_PARAM_BUFFER_SIZE`|1.6.0| | | | |
|`CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT`|11.7| | | | | | | | | |
|`CU_LAUNCH_PARAM_END`| | | | |`HIP_LAUNCH_PARAM_END`|1.6.0| | | | |
|`CU_LAUNCH_PARAM_END_AS_INT`|11.7| | | | | | | | | |
|`CU_LIBRARY_BINARY_IS_PRESERVED`|12.0| | | | | | | | | |
|`CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE`|12.0| | | | | | | | | |
|`CU_LIBRARY_NUM_OPTIONS`|12.0| | | | | | | | | |
|`CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT`| | | | | | | | | | |
|`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`| | | | | | | | | | |
|`CU_LIMIT_MALLOC_HEAP_SIZE`| | | | |`hipLimitMallocHeapSize`|1.6.0| | | | |
|`CU_LIMIT_MAX`| | | | | | | | | | |
|`CU_LIMIT_MAX_L2_FETCH_GRANULARITY`|10.0| | | | | | | | | |
|`CU_LIMIT_PERSISTING_L2_CACHE_SIZE`|11.0| | | | | | | | | |
|`CU_LIMIT_PRINTF_FIFO_SIZE`| | | | |`hipLimitPrintfFifoSize`|4.5.0| | | | |
|`CU_LIMIT_STACK_SIZE`| | | | |`hipLimitStackSize`|5.3.0| | | | |
|`CU_MEMHOSTALLOC_DEVICEMAP`| | | | |`hipHostMallocMapped`|1.6.0| | | | |
|`CU_MEMHOSTALLOC_PORTABLE`| | | | |`hipHostMallocPortable`|1.6.0| | | | |
|`CU_MEMHOSTALLOC_WRITECOMBINED`| | | | |`hipHostMallocWriteCombined`|1.6.0| | | | |
|`CU_MEMHOSTREGISTER_DEVICEMAP`| | | | |`hipHostRegisterMapped`|1.6.0| | | | |
|`CU_MEMHOSTREGISTER_IOMEMORY`|7.5| | | |`hipHostRegisterIoMemory`|1.6.0| | | | |
|`CU_MEMHOSTREGISTER_PORTABLE`| | | | |`hipHostRegisterPortable`|1.6.0| | | | |
|`CU_MEMHOSTREGISTER_READ_ONLY`|11.1| | | |`hipHostRegisterReadOnly`|5.6.0| | | | |
|`CU_MEMORYTYPE_ARRAY`| | | | |`hipMemoryTypeArray`|1.7.0| | | | |
|`CU_MEMORYTYPE_DEVICE`| | | | |`hipMemoryTypeDevice`|1.6.0| | | | |
|`CU_MEMORYTYPE_HOST`| | | | |`hipMemoryTypeHost`|1.6.0| | | | |
|`CU_MEMORYTYPE_UNIFIED`| | | | |`hipMemoryTypeUnified`|1.6.0| | | | |
|`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`|11.2| | | |`hipMemPoolAttrReleaseThreshold`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT`|11.3| | | |`hipMemPoolAttrReservedMemCurrent`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH`|11.3| | | |`hipMemPoolAttrReservedMemHigh`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES`|11.2| | | |`hipMemPoolReuseAllowInternalDependencies`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC`|11.2| | | |`hipMemPoolReuseAllowOpportunistic`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES`|11.2| | | |`hipMemPoolReuseFollowEventDependencies`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_USED_MEM_CURRENT`|11.3| | | |`hipMemPoolAttrUsedMemCurrent`|5.2.0| | | | |
|`CU_MEMPOOL_ATTR_USED_MEM_HIGH`|11.3| | | |`hipMemPoolAttrUsedMemHigh`|5.2.0| | | | |
|`CU_MEM_ACCESS_FLAGS_PROT_MAX`|10.2| | | | | | | | | |
|`CU_MEM_ACCESS_FLAGS_PROT_NONE`|10.2| | | |`hipMemAccessFlagsProtNone`|5.2.0| | | | |
|`CU_MEM_ACCESS_FLAGS_PROT_READ`|10.2| | | |`hipMemAccessFlagsProtRead`|5.2.0| | | | |
|`CU_MEM_ACCESS_FLAGS_PROT_READWRITE`|10.2| | | |`hipMemAccessFlagsProtReadWrite`|5.2.0| | | | |
|`CU_MEM_ADVISE_SET_ACCESSED_BY`|8.0| | | |`hipMemAdviseSetAccessedBy`|3.7.0| | | | |
|`CU_MEM_ADVISE_SET_PREFERRED_LOCATION`|8.0| | | |`hipMemAdviseSetPreferredLocation`|3.7.0| | | | |
|`CU_MEM_ADVISE_SET_READ_MOSTLY`|8.0| | | |`hipMemAdviseSetReadMostly`|3.7.0| | | | |
|`CU_MEM_ADVISE_UNSET_ACCESSED_BY`|8.0| | | |`hipMemAdviseUnsetAccessedBy`|3.7.0| | | | |
|`CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION`|8.0| | | |`hipMemAdviseUnsetPreferredLocation`|3.7.0| | | | |
|`CU_MEM_ADVISE_UNSET_READ_MOSTLY`|8.0| | | |`hipMemAdviseUnsetReadMostly`|3.7.0| | | | |
|`CU_MEM_ALLOCATION_TYPE_INVALID`|10.2| | | |`hipMemAllocationTypeInvalid`|5.2.0| | | | |
|`CU_MEM_ALLOCATION_TYPE_MAX`|10.2| | | |`hipMemAllocationTypeMax`|5.2.0| | | | |
|`CU_MEM_ALLOCATION_TYPE_PINNED`|10.2| | | |`hipMemAllocationTypePinned`|5.2.0| | | | |
|`CU_MEM_ALLOC_GRANULARITY_MINIMUM`|10.2| | | |`hipMemAllocationGranularityMinimum`|5.2.0| | | | |
|`CU_MEM_ALLOC_GRANULARITY_RECOMMENDED`|10.2| | | |`hipMemAllocationGranularityRecommended`|5.2.0| | | | |
|`CU_MEM_ATTACH_GLOBAL`| | | | |`hipMemAttachGlobal`|2.5.0| | | | |
|`CU_MEM_ATTACH_HOST`| | | | |`hipMemAttachHost`|2.5.0| | | | |
|`CU_MEM_ATTACH_SINGLE`| | | | |`hipMemAttachSingle`|3.7.0| | | | |
|`CU_MEM_CREATE_USAGE_TILE_POOL`|11.1| | | | | | | | | |
|`CU_MEM_HANDLE_TYPE_FABRIC`|12.3| | | | | | | | | |
|`CU_MEM_HANDLE_TYPE_GENERIC`|11.1| | | |`hipMemHandleTypeGeneric`|5.2.0| | | | |
|`CU_MEM_HANDLE_TYPE_MAX`|10.2| | | | | | | | | |
|`CU_MEM_HANDLE_TYPE_NONE`|11.2| | | |`hipMemHandleTypeNone`|5.2.0| | | | |
|`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`|10.2| | | |`hipMemHandleTypePosixFileDescriptor`|5.2.0| | | | |
|`CU_MEM_HANDLE_TYPE_WIN32`|10.2| | | |`hipMemHandleTypeWin32`|5.2.0| | | | |
|`CU_MEM_HANDLE_TYPE_WIN32_KMT`|10.2| | | |`hipMemHandleTypeWin32Kmt`|5.2.0| | | | |
|`CU_MEM_LOCATION_TYPE_DEVICE`|10.2| | | |`hipMemLocationTypeDevice`|5.2.0| | | | |
|`CU_MEM_LOCATION_TYPE_HOST`|12.2| | | | | | | | | |
|`CU_MEM_LOCATION_TYPE_HOST_NUMA`|12.2| | | | | | | | | |
|`CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT`|12.2| | | | | | | | | |
|`CU_MEM_LOCATION_TYPE_INVALID`|10.2| | | |`hipMemLocationTypeInvalid`|5.2.0| | | | |
|`CU_MEM_LOCATION_TYPE_MAX`|10.2| | | | | | | | | |
|`CU_MEM_OPERATION_TYPE_MAP`|11.1| | | |`hipMemOperationTypeMap`|5.2.0| | | | |
|`CU_MEM_OPERATION_TYPE_UNMAP`|11.1| | | |`hipMemOperationTypeUnmap`|5.2.0| | | | |
|`CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY`|8.0| | | |`hipMemRangeAttributeAccessedBy`|3.7.0| | | | |
|`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION`|8.0| | | |`hipMemRangeAttributeLastPrefetchLocation`|3.7.0| | | | |
|`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID`|12.2| | | | | | | | | |
|`CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE`|12.2| | | | | | | | | |
|`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION`|8.0| | | |`hipMemRangeAttributePreferredLocation`|3.7.0| | | | |
|`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID`|12.2| | | | | | | | | |
|`CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE`|12.2| | | | | | | | | |
|`CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY`|8.0| | | |`hipMemRangeAttributeReadMostly`|3.7.0| | | | |
|`CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`|11.7| | | | | | | | | |
|`CU_MEM_RANGE_HANDLE_TYPE_MAX`|11.7| | | | | | | | | |
|`CU_MODULE_EAGER_LOADING`|11.7| | | | | | | | | |
|`CU_MODULE_LAZY_LOADING`|11.7| | | | | | | | | |
|`CU_MULTICAST_GRANULARITY_MINIMUM`|12.1| | | | | | | | | |
|`CU_MULTICAST_GRANULARITY_RECOMMENDED`|12.1| | | | | | | | | |
|`CU_OCCUPANCY_DEFAULT`| | | | |`hipOccupancyDefault`|3.2.0| | | | |
|`CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE`| | | | |`hipOccupancyDisableCachingOverride`|5.5.0| | | | |
|`CU_PARAM_TR_DEFAULT`| | | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAGS`|11.1| | | |`HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE`|11.1| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ`|11.1| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE`|11.1| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES`|10.2| | | |`HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_BUFFER_ID`| | | | |`HIP_POINTER_ATTRIBUTE_BUFFER_ID`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_CONTEXT`| | | | |`HIP_POINTER_ATTRIBUTE_CONTEXT`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`|9.2| | | |`HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_DEVICE_POINTER`| | | | |`HIP_POINTER_ATTRIBUTE_DEVICE_POINTER`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_HOST_POINTER`| | | | |`HIP_POINTER_ATTRIBUTE_HOST_POINTER`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`|11.0| | | |`HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE`|10.2| | | |`HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_IS_MANAGED`| | | | |`HIP_POINTER_ATTRIBUTE_IS_MANAGED`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_MAPPED`|10.2| | | |`HIP_POINTER_ATTRIBUTE_MAPPED`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR`|11.7| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_MAPPING_SIZE`|11.7| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID`|11.7| | | | | | | | | |
|`CU_POINTER_ATTRIBUTE_MEMORY_TYPE`| | | | |`HIP_POINTER_ATTRIBUTE_MEMORY_TYPE`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`|11.3| | | |`HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_P2P_TOKENS`| | | | |`HIP_POINTER_ATTRIBUTE_P2P_TOKENS`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_RANGE_SIZE`|10.2| | | |`HIP_POINTER_ATTRIBUTE_RANGE_SIZE`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_RANGE_START_ADDR`|10.2| | | |`HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR`|5.0.0| | | | |
|`CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`| | | | |`HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS`|5.0.0| | | | |
|`CU_PREFER_BINARY`| | | | | | | | | | |
|`CU_PREFER_PTX`| | | | | | | | | | |
|`CU_RESOURCE_TYPE_ARRAY`| | | | |`HIP_RESOURCE_TYPE_ARRAY`|3.5.0| | | | |
|`CU_RESOURCE_TYPE_LINEAR`| | | | |`HIP_RESOURCE_TYPE_LINEAR`|3.5.0| | | | |
|`CU_RESOURCE_TYPE_MIPMAPPED_ARRAY`| | | | |`HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY`|3.5.0| | | | |
|`CU_RESOURCE_TYPE_PITCH2D`| | | | |`HIP_RESOURCE_TYPE_PITCH2D`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_1X16`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_1X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_1X32`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_1X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_2X16`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_2X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_2X32`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_2X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_4X16`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_4X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_FLOAT_4X32`| | | | |`HIP_RES_VIEW_FORMAT_FLOAT_4X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_NONE`| | | | |`HIP_RES_VIEW_FORMAT_NONE`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SIGNED_BC4`| | | | |`HIP_RES_VIEW_FORMAT_SIGNED_BC4`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SIGNED_BC5`| | | | |`HIP_RES_VIEW_FORMAT_SIGNED_BC5`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SIGNED_BC6H`| | | | |`HIP_RES_VIEW_FORMAT_SIGNED_BC6H`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_1X16`| | | | |`HIP_RES_VIEW_FORMAT_SINT_1X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_1X32`| | | | |`HIP_RES_VIEW_FORMAT_SINT_1X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_1X8`| | | | |`HIP_RES_VIEW_FORMAT_SINT_1X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_2X16`| | | | |`HIP_RES_VIEW_FORMAT_SINT_2X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_2X32`| | | | |`HIP_RES_VIEW_FORMAT_SINT_2X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_2X8`| | | | |`HIP_RES_VIEW_FORMAT_SINT_2X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_4X16`| | | | |`HIP_RES_VIEW_FORMAT_SINT_4X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_4X32`| | | | |`HIP_RES_VIEW_FORMAT_SINT_4X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_SINT_4X8`| | | | |`HIP_RES_VIEW_FORMAT_SINT_4X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_1X16`| | | | |`HIP_RES_VIEW_FORMAT_UINT_1X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_1X32`| | | | |`HIP_RES_VIEW_FORMAT_UINT_1X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_1X8`| | | | |`HIP_RES_VIEW_FORMAT_UINT_1X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_2X16`| | | | |`HIP_RES_VIEW_FORMAT_UINT_2X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_2X32`| | | | |`HIP_RES_VIEW_FORMAT_UINT_2X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_2X8`| | | | |`HIP_RES_VIEW_FORMAT_UINT_2X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_4X16`| | | | |`HIP_RES_VIEW_FORMAT_UINT_4X16`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_4X32`| | | | |`HIP_RES_VIEW_FORMAT_UINT_4X32`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UINT_4X8`| | | | |`HIP_RES_VIEW_FORMAT_UINT_4X8`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC1`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC1`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC2`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC2`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC3`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC3`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC4`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC4`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC5`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC5`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC6H`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H`|3.5.0| | | | |
|`CU_RES_VIEW_FORMAT_UNSIGNED_BC7`| | | | |`HIP_RES_VIEW_FORMAT_UNSIGNED_BC7`|3.5.0| | | | |
|`CU_SHAREDMEM_CARVEOUT_DEFAULT`|9.0| | | | | | | | | |
|`CU_SHAREDMEM_CARVEOUT_MAX_L1`|9.0| | | | | | | | | |
|`CU_SHAREDMEM_CARVEOUT_MAX_SHARED`|9.0| | | | | | | | | |
|`CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE`| | | | |`hipSharedMemBankSizeDefault`|1.6.0| | | | |
|`CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE`| | | | |`hipSharedMemBankSizeEightByte`|1.6.0| | | | |
|`CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE`| | | | |`hipSharedMemBankSizeFourByte`|1.6.0| | | | |
|`CU_STREAM_ADD_CAPTURE_DEPENDENCIES`|11.3| | | |`hipStreamAddCaptureDependencies`|5.0.0| | | | |
|`CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW`|11.0| | | | | | | | | |
|`CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN`|12.0| | | | | | | | | |
|`CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP`|12.0| | | | | | | | | |
|`CU_STREAM_ATTRIBUTE_PRIORITY`|12.0| | | | | | | | | |
|`CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY`|11.0| | | | | | | | | |
|`CU_STREAM_CAPTURE_MODE_GLOBAL`|10.1| | | |`hipStreamCaptureModeGlobal`|4.3.0| | | | |
|`CU_STREAM_CAPTURE_MODE_RELAXED`|10.1| | | |`hipStreamCaptureModeRelaxed`|4.3.0| | | | |
|`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`|10.1| | | |`hipStreamCaptureModeThreadLocal`|4.3.0| | | | |
|`CU_STREAM_CAPTURE_STATUS_ACTIVE`|10.0| | | |`hipStreamCaptureStatusActive`|4.3.0| | | | |
|`CU_STREAM_CAPTURE_STATUS_INVALIDATED`|10.0| | | |`hipStreamCaptureStatusInvalidated`|4.3.0| | | | |
|`CU_STREAM_CAPTURE_STATUS_NONE`|10.0| | | |`hipStreamCaptureStatusNone`|4.3.0| | | | |
|`CU_STREAM_DEFAULT`| | | | |`hipStreamDefault`|1.6.0| | | | |
|`CU_STREAM_LEGACY`| | | | | | | | | | |
|`CU_STREAM_MEMORY_BARRIER_TYPE_GPU`|11.7| | | | | | | | | |
|`CU_STREAM_MEMORY_BARRIER_TYPE_SYS`|11.7| | | | | | | | | |
|`CU_STREAM_MEM_OP_BARRIER`|11.7| | | | | | | | | |
|`CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES`|8.0| | | | | | | | | |
|`CU_STREAM_MEM_OP_WAIT_VALUE_32`|8.0| | | | | | | | | |
|`CU_STREAM_MEM_OP_WAIT_VALUE_64`|9.0| | | | | | | | | |
|`CU_STREAM_MEM_OP_WRITE_VALUE_32`|8.0| | | | | | | | | |
|`CU_STREAM_MEM_OP_WRITE_VALUE_64`|9.0| | | | | | | | | |
|`CU_STREAM_NON_BLOCKING`| | | | |`hipStreamNonBlocking`|1.6.0| | | | |
|`CU_STREAM_PER_THREAD`| | | | |`hipStreamPerThread`|4.5.0| | | | |
|`CU_STREAM_SET_CAPTURE_DEPENDENCIES`|11.3| | | |`hipStreamSetCaptureDependencies`|5.0.0| | | | |
|`CU_STREAM_WAIT_VALUE_AND`|8.0| | | |`hipStreamWaitValueAnd`|4.2.0| | | | |
|`CU_STREAM_WAIT_VALUE_EQ`|8.0| | | |`hipStreamWaitValueEq`|4.2.0| | | | |
|`CU_STREAM_WAIT_VALUE_FLUSH`|8.0| | | | | | | | | |
|`CU_STREAM_WAIT_VALUE_GEQ`|8.0| | | |`hipStreamWaitValueGte`|4.2.0| | | | |
|`CU_STREAM_WAIT_VALUE_NOR`|9.0| | | |`hipStreamWaitValueNor`|4.2.0| | | | |
|`CU_STREAM_WRITE_VALUE_DEFAULT`|8.0| | | | | | | | | |
|`CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER`|8.0| | | | | | | | | |
|`CU_SYNC_POLICY_AUTO`|11.0| | | | | | | | | |
|`CU_SYNC_POLICY_BLOCKING_SYNC`|11.0| | | | | | | | | |
|`CU_SYNC_POLICY_SPIN`|11.0| | | | | | | | | |
|`CU_SYNC_POLICY_YIELD`|11.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_10`| | | |9.0| | | | | | |
|`CU_TARGET_COMPUTE_11`| | | |9.0| | | | | | |
|`CU_TARGET_COMPUTE_12`| | | |9.0| | | | | | |
|`CU_TARGET_COMPUTE_13`| | | |9.0| | | | | | |
|`CU_TARGET_COMPUTE_20`| | | |12.0| | | | | | |
|`CU_TARGET_COMPUTE_21`| | | |12.0| | | | | | |
|`CU_TARGET_COMPUTE_30`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_32`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_35`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_37`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_50`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_52`| | | | | | | | | | |
|`CU_TARGET_COMPUTE_53`|8.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_60`|8.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_61`|8.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_62`|8.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_70`|9.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_72`|10.1| | | | | | | | | |
|`CU_TARGET_COMPUTE_73`|9.1| | |10.0| | | | | | |
|`CU_TARGET_COMPUTE_75`|9.1| | | | | | | | | |
|`CU_TARGET_COMPUTE_80`|11.0| | | | | | | | | |
|`CU_TARGET_COMPUTE_86`|11.1| | | | | | | | | |
|`CU_TARGET_COMPUTE_87`|11.7| | | | | | | | | |
|`CU_TARGET_COMPUTE_89`|11.8| | | | | | | | | |
|`CU_TARGET_COMPUTE_90`|11.8| | | | | | | | | |
|`CU_TARGET_COMPUTE_90A`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_BFLOAT16`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_FLOAT16`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_FLOAT32`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_FLOAT64`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_INT32`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_INT64`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_TFLOAT32`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_UINT16`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_UINT32`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_UINT64`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_DATA_TYPE_UINT8`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_INTERLEAVE_16B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_INTERLEAVE_32B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_INTERLEAVE_NONE`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_L2_PROMOTION_L2_128B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_L2_PROMOTION_L2_256B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_L2_PROMOTION_L2_64B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_L2_PROMOTION_NONE`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_NUM_QWORDS`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_SWIZZLE_128B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_SWIZZLE_32B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_SWIZZLE_64B`|12.0| | | | | | | | | |
|`CU_TENSOR_MAP_SWIZZLE_NONE`|12.0| | | | | | | | | |
|`CU_TRSA_OVERRIDE_FORMAT`| | | | |`HIP_TRSA_OVERRIDE_FORMAT`|1.7.0| | | | |
|`CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION`|11.0| | | | | | | | | |
|`CU_TRSF_NORMALIZED_COORDINATES`| | | | |`HIP_TRSF_NORMALIZED_COORDINATES`|1.7.0| | | | |
|`CU_TRSF_READ_AS_INTEGER`| | | | |`HIP_TRSF_READ_AS_INTEGER`|1.7.0| | | | |
|`CU_TRSF_SEAMLESS_CUBEMAP`|11.6| | | | | | | | | |
|`CU_TRSF_SRGB`| | | | |`HIP_TRSF_SRGB`|3.2.0| | | | |
|`CU_TR_ADDRESS_MODE_BORDER`| | | | |`HIP_TR_ADDRESS_MODE_BORDER`|3.5.0| | | | |
|`CU_TR_ADDRESS_MODE_CLAMP`| | | | |`HIP_TR_ADDRESS_MODE_CLAMP`|3.5.0| | | | |
|`CU_TR_ADDRESS_MODE_MIRROR`| | | | |`HIP_TR_ADDRESS_MODE_MIRROR`|3.5.0| | | | |
|`CU_TR_ADDRESS_MODE_WRAP`| | | | |`HIP_TR_ADDRESS_MODE_WRAP`|3.5.0| | | | |
|`CU_TR_FILTER_MODE_LINEAR`| | | | |`HIP_TR_FILTER_MODE_LINEAR`|3.5.0| | | | |
|`CU_TR_FILTER_MODE_POINT`| | | | |`HIP_TR_FILTER_MODE_POINT`|3.5.0| | | | |
|`CU_USER_OBJECT_NO_DESTRUCTOR_SYNC`|11.3| | | |`hipUserObjectNoDestructorSync`|5.3.0| | | | |
|`CUaccessPolicyWindow`|11.0| | | |`hipAccessPolicyWindow`|5.2.0| | | | |
|`CUaccessPolicyWindow_st`|11.0| | | |`hipAccessPolicyWindow`|5.2.0| | | | |
|`CUaccessProperty`|11.0| | | |`hipAccessProperty`|5.2.0| | | | |
|`CUaccessProperty_enum`|11.0| | | |`hipAccessProperty`|5.2.0| | | | |
|`CUaddress_mode`| | | | |`HIPaddress_mode`|3.5.0| | | | |
|`CUaddress_mode_enum`| | | | |`HIPaddress_mode_enum`|3.5.0| | | | |
|`CUarray`| | | | |`hipArray_t`|1.7.0| | | | |
|`CUarrayMapInfo`|11.1| | | |`hipArrayMapInfo`|5.2.0| | | | |
|`CUarrayMapInfo_st`|11.1| | | |`hipArrayMapInfo`|5.2.0| | | | |
|`CUarrayMapInfo_v1`|11.3| | | |`hipArrayMapInfo`|5.2.0| | | | |
|`CUarraySparseSubresourceType`|11.1| | | |`hipArraySparseSubresourceType`|5.2.0| | | | |
|`CUarraySparseSubresourceType_enum`|11.1| | | |`hipArraySparseSubresourceType`|5.2.0| | | | |
|`CUarray_cubemap_face`| | | | | | | | | | |
|`CUarray_cubemap_face_enum`| | | | | | | | | | |
|`CUarray_format`| | | | |`hipArray_Format`|1.7.0| | | | |
|`CUarray_format_enum`| | | | |`hipArray_Format`|1.7.0| | | | |
|`CUarray_st`| | | | |`hipArray`|1.7.0| | | | |
|`CUclusterSchedulingPolicy`|11.8| | | | | | | | | |
|`CUclusterSchedulingPolicy_enum`|11.8| | | | | | | | | |
|`CUcomputemode`| | | | |`hipComputeMode`|1.9.0| | | | |
|`CUcomputemode_enum`| | | | |`hipComputeMode`|1.9.0| | | | |
|`CUcontext`| | | | |`hipCtx_t`|1.6.0| | | | |
|`CUcoredumpSettings`|12.1| | | | | | | | | |
|`CUcoredumpSettings_enum`|12.1| | | | | | | | | |
|`CUctx_flags`| | | | | | | | | | |
|`CUctx_flags_enum`| | | | | | | | | | |
|`CUctx_st`| | | | |`ihipCtx_t`|1.6.0| | | | |
|`CUd3d10DeviceList`| | | | | | | | | | |
|`CUd3d10DeviceList_enum`| | | | | | | | | | |
|`CUd3d10map_flags`| | | | | | | | | | |
|`CUd3d10map_flags_enum`| | | | | | | | | | |
|`CUd3d10register_flags`| | | | | | | | | | |
|`CUd3d10register_flags_enum`| | | | | | | | | | |
|`CUd3d11DeviceList`| | | | | | | | | | |
|`CUd3d11DeviceList_enum`| | | | | | | | | | |
|`CUd3d9DeviceList`| | | | | | | | | | |
|`CUd3d9DeviceList_enum`| | | | | | | | | | |
|`CUd3d9map_flags`| | | | | | | | | | |
|`CUd3d9map_flags_enum`| | | | | | | | | | |
|`CUd3d9register_flags`| | | | | | | | | | |
|`CUd3d9register_flags_enum`| | | | | | | | | | |
|`CUdevice`| | | | |`hipDevice_t`|1.6.0| | | | |
|`CUdeviceNumaConfig`|12.2| | | | | | | | | |
|`CUdeviceNumaConfig_enum`|12.2| | | | | | | | | |
|`CUdevice_P2PAttribute`|8.0| | | |`hipDeviceP2PAttr`|3.8.0| | | | |
|`CUdevice_P2PAttribute_enum`|8.0| | | |`hipDeviceP2PAttr`|3.8.0| | | | |
|`CUdevice_attribute`| | | | |`hipDeviceAttribute_t`|1.6.0| | | | |
|`CUdevice_attribute_enum`| | | | |`hipDeviceAttribute_t`|1.6.0| | | | |
|`CUdevice_v1`|11.3| | | |`hipDevice_t`|1.6.0| | | | |
|`CUdeviceptr`| | | | |`hipDeviceptr_t`|1.7.0| | | | |
|`CUdeviceptr_v1`| | | | |`hipDeviceptr_t`|1.7.0| | | | |
|`CUdeviceptr_v2`|11.3| | | |`hipDeviceptr_t`|1.7.0| | | | |
|`CUdevprop`| | | | | | | | | | |
|`CUdevprop_st`| | | | | | | | | | |
|`CUdevprop_v1`|11.3| | | | | | | | | |
|`CUdriverProcAddressQueryResult`|12.0| | | |`hipDriverProcAddressQueryResult`|6.1.0| | | |6.1.0|
|`CUdriverProcAddressQueryResult_enum`|12.0| | | |`hipDriverProcAddressQueryResult`|6.1.0| | | |6.1.0|
|`CUdriverProcAddress_flags`|11.3| | | | | | | | | |
|`CUdriverProcAddress_flags_enum`|11.3| | | | | | | | | |
|`CUeglColorFormat`|9.0| | | | | | | | | |
|`CUeglColorFormate_enum`|9.0| | | | | | | | | |
|`CUeglFrameType`|9.0| | | | | | | | | |
|`CUeglFrameType_enum`|9.0| | | | | | | | | |
|`CUeglResourceLocationFlags`|9.0| | | | | | | | | |
|`CUeglResourceLocationFlags_enum`|9.0| | | | | | | | | |
|`CUeglStreamConnection`|9.0| | | | | | | | | |
|`CUeglStreamConnection_st`|9.0| | | | | | | | | |
|`CUevent`| | | | |`hipEvent_t`|1.6.0| | | | |
|`CUevent_flags`| | | | | | | | | | |
|`CUevent_flags_enum`| | | | | | | | | | |
|`CUevent_record_flags`|11.1| | | | | | | | | |
|`CUevent_record_flags_enum`|11.1| | | | | | | | | |
|`CUevent_sched_flags`|11.8| | | | | | | | | |
|`CUevent_sched_flags_enum`|11.8| | | | | | | | | |
|`CUevent_st`| | | | |`ihipEvent_t`|1.6.0| | | | |
|`CUevent_wait_flags`|11.1| | | | | | | | | |
|`CUevent_wait_flags_enum`| | | | | | | | | | |
|`CUexecAffinityParam`|11.4| | | | | | | | | |
|`CUexecAffinityParam_st`|11.4| | | | | | | | | |
|`CUexecAffinityParam_v1`|11.4| | | | | | | | | |
|`CUexecAffinitySmCount`|11.4| | | | | | | | | |
|`CUexecAffinitySmCount_st`|11.4| | | | | | | | | |
|`CUexecAffinitySmCount_v1`|11.4| | | | | | | | | |
|`CUexecAffinityType`|11.4| | | | | | | | | |
|`CUexecAffinityType_enum`|11.4| | | | | | | | | |
|`CUextMemory_st`|10.0| | | | | | | | | |
|`CUextSemaphore_st`|10.0| | | | | | | | | |
|`CUexternalMemory`|10.0| | | |`hipExternalMemory_t`|4.3.0| | | | |
|`CUexternalMemoryHandleType`|10.0| | | |`hipExternalMemoryHandleType`|4.3.0| | | | |
|`CUexternalMemoryHandleType_enum`|10.0| | | |`hipExternalMemoryHandleType_enum`|4.3.0| | | | |
|`CUexternalSemaphore`|10.0| | | |`hipExternalSemaphore_t`|4.4.0| | | | |
|`CUexternalSemaphoreHandleType`|10.0| | | |`hipExternalSemaphoreHandleType`|4.4.0| | | | |
|`CUexternalSemaphoreHandleType_enum`|10.0| | | |`hipExternalSemaphoreHandleType_enum`|4.4.0| | | | |
|`CUfilter_mode`| | | | |`HIPfilter_mode`|3.5.0| | | | |
|`CUfilter_mode_enum`| | | | |`HIPfilter_mode_enum`|3.5.0| | | | |
|`CUflushGPUDirectRDMAWritesOptions`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptions`|6.1.0| | | |6.1.0|
|`CUflushGPUDirectRDMAWritesOptions_enum`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptions`|6.1.0| | | |6.1.0|
|`CUflushGPUDirectRDMAWritesScope`|11.3| | | | | | | | | |
|`CUflushGPUDirectRDMAWritesScope_enum`|11.3| | | | | | | | | |
|`CUflushGPUDirectRDMAWritesTarget`|11.3| | | | | | | | | |
|`CUflushGPUDirectRDMAWritesTarget_enum`|11.3| | | | | | | | | |
|`CUfunc_cache`| | | | |`hipFuncCache_t`|1.6.0| | | | |
|`CUfunc_cache_enum`| | | | |`hipFuncCache_t`|1.6.0| | | | |
|`CUfunc_st`| | | | |`ihipModuleSymbol_t`|1.6.0| | | | |
|`CUfunction`| | | | |`hipFunction_t`|1.6.0| | | | |
|`CUfunction_attribute`| | | | |`hipFunction_attribute`|2.8.0| | | | |
|`CUfunction_attribute_enum`| | | | |`hipFunction_attribute`|2.8.0| | | | |
|`CUgraph`|10.0| | | |`hipGraph_t`|4.3.0| | | | |
|`CUgraphConditionalHandle`|12.3| | | | | | | | | |
|`CUgraphConditionalNodeType`|12.3| | | | | | | | | |
|`CUgraphConditionalNodeType_enum`|12.3| | | | | | | | | |
|`CUgraphDebugDot_flags`|11.3| | | |`hipGraphDebugDotFlags`|5.5.0| | | | |
|`CUgraphDebugDot_flags_enum`|11.3| | | |`hipGraphDebugDotFlags`|5.5.0| | | | |
|`CUgraphDependencyType`|12.3| | | | | | | | | |
|`CUgraphDependencyType_enum`|12.3| | | | | | | | | |
|`CUgraphEdgeData`|12.3| | | | | | | | | |
|`CUgraphEdgeData_st`|12.3| | | | | | | | | |
|`CUgraphExec`|10.0| | | |`hipGraphExec_t`|4.3.0| | | | |
|`CUgraphExecUpdateResult`|10.2| | | |`hipGraphExecUpdateResult`|4.3.0| | | | |
|`CUgraphExecUpdateResultInfo`|12.0| | | | | | | | | |
|`CUgraphExecUpdateResultInfo_st`|12.0| | | | | | | | | |
|`CUgraphExecUpdateResultInfo_v1`|12.0| | | | | | | | | |
|`CUgraphExecUpdateResult_enum`|10.2| | | |`hipGraphExecUpdateResult`|4.3.0| | | | |
|`CUgraphExec_st`|10.0| | | |`hipGraphExec`|4.3.0| | | | |
|`CUgraphInstantiateResult`|12.0| | | |`hipGraphInstantiateResult`|6.1.0| | | |6.1.0|
|`CUgraphInstantiateResult_enum`|12.0| | | |`hipGraphInstantiateResult`|6.1.0| | | |6.1.0|
|`CUgraphInstantiate_flags`|11.4| | | |`hipGraphInstantiateFlags`|5.2.0| | | | |
|`CUgraphInstantiate_flags_enum`|11.4| | | |`hipGraphInstantiateFlags`|5.2.0| | | | |
|`CUgraphMem_attribute`|11.4| | | |`hipGraphMemAttributeType`|5.3.0| | | | |
|`CUgraphMem_attribute_enum`|11.4| | | |`hipGraphMemAttributeType`|5.3.0| | | | |
|`CUgraphNode`|10.0| | | |`hipGraphNode_t`|4.3.0| | | | |
|`CUgraphNodeParams`|12.2| | | |`hipGraphNodeParams`|6.1.0| | | |6.1.0|
|`CUgraphNodeParams_st`|12.2| | | |`hipGraphNodeParams`|6.1.0| | | |6.1.0|
|`CUgraphNodeType`|10.0| | | |`hipGraphNodeType`|4.3.0| | | | |
|`CUgraphNodeType_enum`|10.0| | | |`hipGraphNodeType`|4.3.0| | | | |
|`CUgraphNode_st`|10.0| | | |`hipGraphNode`|4.3.0| | | | |
|`CUgraph_st`|10.0| | | |`ihipGraph`|4.3.0| | | | |
|`CUgraphicsMapResourceFlags`| | | | | | | | | | |
|`CUgraphicsMapResourceFlags_enum`| | | | | | | | | | |
|`CUgraphicsRegisterFlags`| | | | |`hipGraphicsRegisterFlags`|4.4.0| | | | |
|`CUgraphicsRegisterFlags_enum`| | | | |`hipGraphicsRegisterFlags`|4.4.0| | | | |
|`CUgraphicsResource`| | | | |`hipGraphicsResource_t`|4.4.0| | | | |
|`CUgraphicsResource_st`| | | | |`hipGraphicsResource`|4.4.0| | | | |
|`CUhostFn`|10.0| | | |`hipHostFn_t`|4.3.0| | | | |
|`CUipcEventHandle`| | | | |`hipIpcEventHandle_t`|1.6.0| | | | |
|`CUipcEventHandle_st`| | | | |`hipIpcEventHandle_st`|3.5.0| | | | |
|`CUipcEventHandle_v1`|11.3| | | |`hipIpcEventHandle_t`|1.6.0| | | | |
|`CUipcMemHandle`| | | | |`hipIpcMemHandle_t`|1.6.0| | | | |
|`CUipcMemHandle_st`| | | | |`hipIpcMemHandle_st`|1.6.0| | | | |
|`CUipcMemHandle_v1`|11.3| | | |`hipIpcMemHandle_t`|1.6.0| | | | |
|`CUipcMem_flags`| | | | | | | | | | |
|`CUipcMem_flags_enum`| | | | | | | | | | |
|`CUjitInputType`| | | | |`hiprtcJITInputType`|5.3.0| | | | |
|`CUjitInputType_enum`| | | | |`hiprtcJITInputType`|5.3.0| | | | |
|`CUjit_cacheMode`| | | | | | | | | | |
|`CUjit_cacheMode_enum`| | | | | | | | | | |
|`CUjit_fallback`| | | | | | | | | | |
|`CUjit_fallback_enum`| | | | | | | | | | |
|`CUjit_option`| | | | |`hipJitOption`|1.6.0| | | | |
|`CUjit_option_enum`| | | | |`hipJitOption`|1.6.0| | | | |
|`CUjit_target`| | | | | | | | | | |
|`CUjit_target_enum`| | | | | | | | | | |
|`CUkern_st`|12.0| | | | | | | | | |
|`CUkernel`|12.0| | | | | | | | | |
|`CUkernelNodeAttrID`|11.0| | | |`hipKernelNodeAttrID`|5.2.0| | | | |
|`CUkernelNodeAttrID_enum`|11.0| | |11.8|`hipKernelNodeAttrID`|5.2.0| | | | |
|`CUkernelNodeAttrValue`|11.0| | | |`hipKernelNodeAttrValue`|5.2.0| | | | |
|`CUkernelNodeAttrValue_union`|11.0| | |11.8|`hipKernelNodeAttrValue`|5.2.0| | | | |
|`CUkernelNodeAttrValue_v1`|11.3| | | |`hipKernelNodeAttrValue`|5.2.0| | | | |
|`CUlaunchAttribute`|11.8| | | | | | | | | |
|`CUlaunchAttributeID`|11.8| | | | | | | | | |
|`CUlaunchAttributeID_enum`|11.8| | | | | | | | | |
|`CUlaunchAttributeValue`|11.8| | | | | | | | | |
|`CUlaunchAttributeValue_union`|11.8| | | | | | | | | |
|`CUlaunchAttribute_st`|11.8| | | | | | | | | |
|`CUlaunchConfig`|11.8| | | | | | | | | |
|`CUlaunchConfig_st`|11.8| | | | | | | | | |
|`CUlaunchMemSyncDomain`|12.0| | | | | | | | | |
|`CUlaunchMemSyncDomainMap`|12.0| | | | | | | | | |
|`CUlaunchMemSyncDomainMap_st`|12.0| | | | | | | | | |
|`CUlaunchMemSyncDomain_enum`|12.0| | | | | | | | | |
|`CUlib_st`|12.0| | | | | | | | | |
|`CUlibrary`|12.0| | | | | | | | | |
|`CUlibraryHostUniversalFunctionAndDataTable`|12.0| | | | | | | | | |
|`CUlibraryHostUniversalFunctionAndDataTable_st`|12.0| | | | | | | | | |
|`CUlibraryOption`|12.0| | | | | | | | | |
|`CUlibraryOption_enum`|12.0| | | | | | | | | |
|`CUlimit`| | | | |`hipLimit_t`|1.6.0| | | | |
|`CUlimit_enum`| | | | |`hipLimit_t`|1.6.0| | | | |
|`CUlinkState`| | | | |`hiprtcLinkState`|5.3.0| | | | |
|`CUlinkState_st`| | | | |`ihiprtcLinkState`|5.3.0| | | | |
|`CUmemAccessDesc`|10.2| | | |`hipMemAccessDesc`|5.2.0| | | | |
|`CUmemAccessDesc_st`|10.2| | | |`hipMemAccessDesc`|5.2.0| | | | |
|`CUmemAccessDesc_v1`|11.3| | | |`hipMemAccessDesc`|5.2.0| | | | |
|`CUmemAccess_flags`|10.2| | | |`hipMemAccessFlags`|5.2.0| | | | |
|`CUmemAccess_flags_enum`|10.2| | | |`hipMemAccessFlags`|5.2.0| | | | |
|`CUmemAllocationGranularity_flags`|10.2| | | |`hipMemAllocationGranularity_flags`|5.2.0| | | | |
|`CUmemAllocationGranularity_flags_enum`|10.2| | | |`hipMemAllocationGranularity_flags`|5.2.0| | | | |
|`CUmemAllocationHandleType`|10.2| | | |`hipMemAllocationHandleType`|5.2.0| | | | |
|`CUmemAllocationHandleType_enum`|10.2| | | |`hipMemAllocationHandleType`|5.2.0| | | | |
|`CUmemAllocationProp`|10.2| | | |`hipMemAllocationProp`|5.2.0| | | | |
|`CUmemAllocationProp_st`|10.2| | | |`hipMemAllocationProp`|5.2.0| | | | |
|`CUmemAllocationProp_v1`|11.3| | | |`hipMemAllocationProp`|5.2.0| | | | |
|`CUmemAllocationType`|10.2| | | |`hipMemAllocationType`|5.2.0| | | | |
|`CUmemAllocationType_enum`|10.2| | | |`hipMemAllocationType`|5.2.0| | | | |
|`CUmemAttach_flags`| | | | | | | | | | |
|`CUmemAttach_flags_enum`| | | | | | | | | | |
|`CUmemFabricHandle`|12.3| | | | | | | | | |
|`CUmemFabricHandle_st`|12.3| | | | | | | | | |
|`CUmemFabricHandle_v1`|12.3| | | | | | | | | |
|`CUmemGenericAllocationHandle`|10.2| | | |`hipMemGenericAllocationHandle_t`|5.2.0| | | | |
|`CUmemGenericAllocationHandle_v1`|11.3| | | |`hipMemGenericAllocationHandle_t`|5.2.0| | | | |
|`CUmemHandleType`|11.1| | | |`hipMemHandleType`|5.2.0| | | | |
|`CUmemHandleType_enum`|11.1| | | |`hipMemHandleType`|5.2.0| | | | |
|`CUmemLocation`|10.2| | | |`hipMemLocation`|5.2.0| | | | |
|`CUmemLocationType`|10.2| | | |`hipMemLocationType`|5.2.0| | | | |
|`CUmemLocationType_enum`|10.2| | | |`hipMemLocationType`|5.2.0| | | | |
|`CUmemLocation_st`|10.2| | | |`hipMemLocation`|5.2.0| | | | |
|`CUmemLocation_v1`|11.3| | | |`hipMemLocation`|5.2.0| | | | |
|`CUmemOperationType`|11.1| | | |`hipMemOperationType`|5.2.0| | | | |
|`CUmemOperationType_enum`|11.1| | | |`hipMemOperationType`|5.2.0| | | | |
|`CUmemPoolHandle_st`|11.2| | | |`ihipMemPoolHandle_t`|5.2.0| | | | |
|`CUmemPoolProps`|11.2| | | |`hipMemPoolProps`|5.2.0| | | | |
|`CUmemPoolProps_st`|11.2| | | |`hipMemPoolProps`|5.2.0| | | | |
|`CUmemPoolProps_v1`|11.3| | | |`hipMemPoolProps`|5.2.0| | | | |
|`CUmemPoolPtrExportData`|11.2| | | |`hipMemPoolPtrExportData`|5.2.0| | | | |
|`CUmemPoolPtrExportData_st`|11.2| | | |`hipMemPoolPtrExportData`|5.2.0| | | | |
|`CUmemPoolPtrExportData_v1`|11.3| | | |`hipMemPoolPtrExportData`|5.2.0| | | | |
|`CUmemPool_attribute`|11.2| | | |`hipMemPoolAttr`|5.2.0| | | | |
|`CUmemPool_attribute_enum`|11.2| | | |`hipMemPoolAttr`|5.2.0| | | | |
|`CUmemRangeHandleType`|11.7| | | | | | | | | |
|`CUmemRangeHandleType_enum`|11.7| | | | | | | | | |
|`CUmem_advise`|8.0| | | |`hipMemoryAdvise`|3.7.0| | | | |
|`CUmem_advise_enum`|8.0| | | |`hipMemoryAdvise`|3.7.0| | | | |
|`CUmem_range_attribute`|8.0| | | |`hipMemRangeAttribute`|3.7.0| | | | |
|`CUmem_range_attribute_enum`|8.0| | | |`hipMemRangeAttribute`|3.7.0| | | | |
|`CUmemoryPool`|11.2| | | |`hipMemPool_t`|5.2.0| | | | |
|`CUmemorytype`| | | | |`hipMemoryType`|1.6.0| | | | |
|`CUmemorytype_enum`| | | | |`hipMemoryType`|1.6.0| | | | |
|`CUmipmappedArray`| | | | |`hipMipmappedArray_t`|1.7.0| | | | |
|`CUmipmappedArray_st`| | | | |`hipMipmappedArray`|1.7.0| | | | |
|`CUmod_st`| | | | |`ihipModule_t`|1.6.0| | | | |
|`CUmodule`| | | | |`hipModule_t`|1.6.0| | | | |
|`CUmoduleLoadingMode`|11.7| | | | | | | | | |
|`CUmoduleLoadingMode_enum`|11.7| | | | | | | | | |
|`CUmulticastGranularity_flags`|12.1| | | | | | | | | |
|`CUmulticastGranularity_flags_enum`|12.1| | | | | | | | | |
|`CUmulticastObjectProp`|12.1| | | | | | | | | |
|`CUmulticastObjectProp_st`|12.1| | | | | | | | | |
|`CUmulticastObjectProp_v1`|12.1| | | | | | | | | |
|`CUoccupancyB2DSize`| | | | |`void*`| | | | | |
|`CUoccupancy_flags`| | | | | | | | | | |
|`CUoccupancy_flags_enum`| | | | | | | | | | |
|`CUpointer_attribute`| | | | |`hipPointer_attribute`|5.0.0| | | | |
|`CUpointer_attribute_enum`| | | | |`hipPointer_attribute`|5.0.0| | | | |
|`CUresourceViewFormat`| | | | |`HIPresourceViewFormat`|3.5.0| | | | |
|`CUresourceViewFormat_enum`| | | | |`HIPresourceViewFormat_enum`|3.5.0| | | | |
|`CUresourcetype`| | | | |`HIPresourcetype`|3.5.0| | | | |
|`CUresourcetype_enum`| | | | |`HIPresourcetype_enum`|3.5.0| | | | |
|`CUresult`| | | | |`hipError_t`|1.5.0| | | | |
|`CUshared_carveout`|9.0| | | | | | | | | |
|`CUshared_carveout_enum`|9.0| | | | | | | | | |
|`CUsharedconfig`| | | | |`hipSharedMemConfig`|1.6.0| | | | |
|`CUsharedconfig_enum`| | | | |`hipSharedMemConfig`|1.6.0| | | | |
|`CUstream`| | | | |`hipStream_t`|1.5.0| | | | |
|`CUstreamAttrID`|11.0| | | | | | | | | |
|`CUstreamAttrID_enum`|11.0| | |11.8| | | | | | |
|`CUstreamAttrValue`|11.0| | | | | | | | | |
|`CUstreamAttrValue_union`|11.0| | | | | | | | | |
|`CUstreamAttrValue_v1`|11.3| | | | | | | | | |
|`CUstreamBatchMemOpParams`|8.0| | | | | | | | | |
|`CUstreamBatchMemOpParams_union`|8.0| | | | | | | | | |
|`CUstreamBatchMemOpParams_v1`|11.3| | | | | | | | | |
|`CUstreamBatchMemOpType`|8.0| | | | | | | | | |
|`CUstreamBatchMemOpType_enum`|8.0| | | | | | | | | |
|`CUstreamCallback`| | | | |`hipStreamCallback_t`|1.6.0| | | | |
|`CUstreamCaptureMode`|10.1| | | |`hipStreamCaptureMode`|4.3.0| | | | |
|`CUstreamCaptureMode_enum`|10.1| | | |`hipStreamCaptureMode`|4.3.0| | | | |
|`CUstreamCaptureStatus`|10.0| | | |`hipStreamCaptureStatus`|4.3.0| | | | |
|`CUstreamCaptureStatus_enum`|10.0| | | |`hipStreamCaptureStatus`|4.3.0| | | | |
|`CUstreamMemOpMemoryBarrierParams_st`|11.7| | | | | | | | | |
|`CUstreamMemoryBarrier_flags`|11.7| | | | | | | | | |
|`CUstreamMemoryBarrier_flags_enum`|11.7| | | | | | | | | |
|`CUstreamUpdateCaptureDependencies_flags`|11.3| | | |`hipStreamUpdateCaptureDependenciesFlags`|5.0.0| | | | |
|`CUstreamUpdateCaptureDependencies_flags_enum`|11.3| | | |`hipStreamUpdateCaptureDependenciesFlags`|5.0.0| | | | |
|`CUstreamWaitValue_flags`|8.0| | | | | | | | | |
|`CUstreamWaitValue_flags_enum`|8.0| | | | | | | | | |
|`CUstreamWriteValue_flags`|8.0| | | | | | | | | |
|`CUstreamWriteValue_flags_enum`|8.0| | | | | | | | | |
|`CUstream_flags`| | | | | | | | | | |
|`CUstream_flags_enum`| | | | | | | | | | |
|`CUstream_st`| | | | |`ihipStream_t`|1.5.0| | | | |
|`CUsurfObject`| | | | |`hipSurfaceObject_t`|1.9.0| | | | |
|`CUsurfObject_v1`|11.3| | | |`hipSurfaceObject_t`|1.9.0| | | | |
|`CUsurfref`| | | | | | | | | | |
|`CUsurfref_st`| | | | | | | | | | |
|`CUsynchronizationPolicy`|11.0| | | | | | | | | |
|`CUsynchronizationPolicy_enum`|11.0| | | | | | | | | |
|`CUtensorMap`|12.0| | | | | | | | | |
|`CUtensorMapDataType`|12.0| | | | | | | | | |
|`CUtensorMapDataType_enum`|12.0| | | | | | | | | |
|`CUtensorMapFloatOOBfill`|12.0| | | | | | | | | |
|`CUtensorMapFloatOOBfill_enum`|12.0| | | | | | | | | |
|`CUtensorMapInterleave`|12.0| | | | | | | | | |
|`CUtensorMapInterleave_enum`|12.0| | | | | | | | | |
|`CUtensorMapL2promotion`|12.0| | | | | | | | | |
|`CUtensorMapL2promotion_enum`|12.0| | | | | | | | | |
|`CUtensorMapSwizzle`|12.0| | | | | | | | | |
|`CUtensorMapSwizzle_enum`|12.0| | | | | | | | | |
|`CUtensorMap_st`|12.0| | | | | | | | | |
|`CUtexObject`| | | | |`hipTextureObject_t`|1.7.0| | | | |
|`CUtexObject_v1`|11.3| | | |`hipTextureObject_t`|1.7.0| | | | |
|`CUtexref`| | | | |`hipTexRef`|3.10.0| | | | |
|`CUtexref_st`| | | | |`textureReference`|1.6.0| | | | |
|`CUuserObject`|11.3| | | |`hipUserObject_t`|5.3.0| | | | |
|`CUuserObjectRetain_flags`|11.3| | | |`hipUserObjectRetainFlags`|5.3.0| | | | |
|`CUuserObjectRetain_flags_enum`|11.3| | | |`hipUserObjectRetainFlags`|5.3.0| | | | |
|`CUuserObject_flags`|11.3| | | |`hipUserObjectFlags`|5.3.0| | | | |
|`CUuserObject_flags_enum`|11.3| | | |`hipUserObjectFlags`|5.3.0| | | | |
|`CUuserObject_st`|11.3| | | |`hipUserObject`|5.3.0| | | | |
|`CUuuid`| | | | |`hipUUID`|5.2.0| | | | |
|`CUuuid_st`| | | | |`hipUUID_t`|5.2.0| | | | |
|`GLenum`| | | | |`GLenum`|5.1.0| | | | |
|`GLuint`| | | | |`GLuint`|5.1.0| | | | |
|`NVCL_CTX_SCHED_AUTO`|11.8| | | | | | | | | |
|`NVCL_CTX_SCHED_BLOCKING_SYNC`|11.8| | | | | | | | | |
|`NVCL_CTX_SCHED_SPIN`|11.8| | | | | | | | | |
|`NVCL_CTX_SCHED_YIELD`|11.8| | | | | | | | | |
|`NVCL_EVENT_SCHED_AUTO`|11.8| | | | | | | | | |
|`NVCL_EVENT_SCHED_BLOCKING_SYNC`|11.8| | | | | | | | | |
|`NVCL_EVENT_SCHED_SPIN`|11.8| | | | | | | | | |
|`NVCL_EVENT_SCHED_YIELD`|11.8| | | | | | | | | |
|`__CUDACC__`| | | | |`__HIPCC__`|1.6.0| | | | |
|`cl_context_flags`|11.8| | | | | | | | | |
|`cl_context_flags_enum`|11.8| | | | | | | | | |
|`cl_event_flags`|11.8| | | | | | | | | |
|`cl_event_flags_enum`|11.8| | | | | | | | | |
|`cudaError_enum`| | | | |`hipError_t`|1.5.0| | | | |
|`memoryBarrier`|11.7| | | | | | | | | |

## **2. Error Handling**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuGetErrorName`| | | | |`hipDrvGetErrorName`|5.4.0| | | | |
|`cuGetErrorString`| | | | |`hipDrvGetErrorString`|5.4.0| | | | |

## **3. Initialization**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuInit`| | | | |`hipInit`|1.6.0| | | | |

## **4. Version Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDriverGetVersion`| | | | |`hipDriverGetVersion`|1.6.0| | | | |

## **5. Device Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDeviceGet`| | | | |`hipDeviceGet`|1.6.0| | | | |
|`cuDeviceGetAttribute`| | | | |`hipDeviceGetAttribute`|1.6.0| | | | |
|`cuDeviceGetCount`| | | | |`hipGetDeviceCount`|1.6.0| | | | |
|`cuDeviceGetDefaultMemPool`|11.2| | | |`hipDeviceGetDefaultMemPool`|5.2.0| | | | |
|`cuDeviceGetExecAffinitySupport`|11.4| | | | | | | | | |
|`cuDeviceGetLuid`|10.0| | | | | | | | | |
|`cuDeviceGetMemPool`|11.2| | | |`hipDeviceGetMemPool`|5.2.0| | | | |
|`cuDeviceGetName`| | | | |`hipDeviceGetName`|1.6.0| | | | |
|`cuDeviceGetNvSciSyncAttributes`|10.2| | | | | | | | | |
|`cuDeviceGetTexture1DLinearMaxWidth`|11.1| | | | | | | | | |
|`cuDeviceGetUuid`|9.2| | | |`hipDeviceGetUuid`|5.2.0| | | | |
|`cuDeviceGetUuid_v2`|11.4| | | |`hipDeviceGetUuid`|5.2.0| | | | |
|`cuDeviceSetMemPool`|11.2| | | |`hipDeviceSetMemPool`|5.2.0| | | | |
|`cuDeviceTotalMem`| | | | |`hipDeviceTotalMem`|1.6.0| | | | |
|`cuDeviceTotalMem_v2`| | | | |`hipDeviceTotalMem`|1.6.0| | | | |
|`cuFlushGPUDirectRDMAWrites`|11.3| | | | | | | | | |

## **6. Device Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDeviceComputeCapability`| |9.2| | |`hipDeviceComputeCapability`|1.6.0| | | | |
|`cuDeviceGetProperties`| |9.2| | | | | | | | |

## **7. Primary Context Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDevicePrimaryCtxGetState`| | | | |`hipDevicePrimaryCtxGetState`|1.9.0| | | | |
|`cuDevicePrimaryCtxRelease`| | | | |`hipDevicePrimaryCtxRelease`|1.9.0| | | | |
|`cuDevicePrimaryCtxRelease_v2`|11.0| | | |`hipDevicePrimaryCtxRelease`|1.9.0| | | | |
|`cuDevicePrimaryCtxReset`| | | | |`hipDevicePrimaryCtxReset`|1.9.0| | | | |
|`cuDevicePrimaryCtxReset_v2`|11.0| | | |`hipDevicePrimaryCtxReset`|1.9.0| | | | |
|`cuDevicePrimaryCtxRetain`| | | | |`hipDevicePrimaryCtxRetain`|1.9.0| | | | |
|`cuDevicePrimaryCtxSetFlags`| | | | |`hipDevicePrimaryCtxSetFlags`|1.9.0| | | | |
|`cuDevicePrimaryCtxSetFlags_v2`|11.0| | | |`hipDevicePrimaryCtxSetFlags`|1.9.0| | | | |

## **8. Context Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuCtxCreate`| | | | |`hipCtxCreate`|1.6.0|1.9.0| | | |
|`cuCtxCreate_v2`| | | | |`hipCtxCreate`|1.6.0|1.9.0| | | |
|`cuCtxCreate_v3`|11.4| | | | | | | | | |
|`cuCtxDestroy`| | | | |`hipCtxDestroy`|1.6.0|1.9.0| | | |
|`cuCtxDestroy_v2`| | | | |`hipCtxDestroy`|1.6.0|1.9.0| | | |
|`cuCtxGetApiVersion`| | | | |`hipCtxGetApiVersion`|1.9.0|1.9.0| | | |
|`cuCtxGetCacheConfig`| | | | |`hipCtxGetCacheConfig`|1.9.0|1.9.0| | | |
|`cuCtxGetCurrent`| | | | |`hipCtxGetCurrent`|1.6.0|1.9.0| | | |
|`cuCtxGetDevice`| | | | |`hipCtxGetDevice`|1.6.0|1.9.0| | | |
|`cuCtxGetExecAffinity`|11.4| | | | | | | | | |
|`cuCtxGetFlags`| | | | |`hipCtxGetFlags`|1.9.0|1.9.0| | | |
|`cuCtxGetId`|12.0| | | | | | | | | |
|`cuCtxGetLimit`| | | | |`hipDeviceGetLimit`|1.6.0| | | | |
|`cuCtxGetSharedMemConfig`| | | | |`hipCtxGetSharedMemConfig`|1.9.0|1.9.0| | | |
|`cuCtxGetStreamPriorityRange`| | | | |`hipDeviceGetStreamPriorityRange`|2.0.0| | | | |
|`cuCtxPopCurrent`| | | | |`hipCtxPopCurrent`|1.6.0|1.9.0| | | |
|`cuCtxPopCurrent_v2`| | | | |`hipCtxPopCurrent`|1.6.0|1.9.0| | | |
|`cuCtxPushCurrent`| | | | |`hipCtxPushCurrent`|1.6.0|1.9.0| | | |
|`cuCtxPushCurrent_v2`| | | | |`hipCtxPushCurrent`|1.6.0|1.9.0| | | |
|`cuCtxResetPersistingL2Cache`|11.0| | | | | | | | | |
|`cuCtxSetCacheConfig`| | | | |`hipCtxSetCacheConfig`|1.9.0|1.9.0| | | |
|`cuCtxSetCurrent`| | | | |`hipCtxSetCurrent`|1.6.0|1.9.0| | | |
|`cuCtxSetFlags`|12.1| | | | | | | | | |
|`cuCtxSetLimit`| | | | |`hipDeviceSetLimit`|5.3.0| | | | |
|`cuCtxSetSharedMemConfig`| | | | |`hipCtxSetSharedMemConfig`|1.9.0|1.9.0| | | |
|`cuCtxSynchronize`| | | | |`hipCtxSynchronize`|1.9.0|1.9.0| | | |

## **9. Context Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuCtxAttach`| | | | | | | | | | |
|`cuCtxDetach`| | | | | | | | | | |

## **10. Module Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuLinkAddData`| | | | |`hiprtcLinkAddData`|5.3.0| | | | |
|`cuLinkAddData_v2`| | | | |`hiprtcLinkAddData`|5.3.0| | | | |
|`cuLinkAddFile`| | | | |`hiprtcLinkAddFile`|5.3.0| | | | |
|`cuLinkAddFile_v2`| | | | |`hiprtcLinkAddFile`|5.3.0| | | | |
|`cuLinkComplete`| | | | |`hiprtcLinkComplete`|5.3.0| | | | |
|`cuLinkCreate`| | | | |`hiprtcLinkCreate`|5.3.0| | | | |
|`cuLinkCreate_v2`| | | | |`hiprtcLinkCreate`|5.3.0| | | | |
|`cuLinkDestroy`| | | | |`hiprtcLinkDestroy`|5.3.0| | | | |
|`cuModuleGetFunction`| | | | |`hipModuleGetFunction`|1.6.0| | | | |
|`cuModuleGetGlobal`| | | | |`hipModuleGetGlobal`|1.6.0| | | | |
|`cuModuleGetGlobal_v2`| | | | |`hipModuleGetGlobal`|1.6.0| | | | |
|`cuModuleGetLoadingMode`|11.7| | | | | | | | | |
|`cuModuleLoad`| | | | |`hipModuleLoad`|1.6.0| | | | |
|`cuModuleLoadData`| | | | |`hipModuleLoadData`|1.6.0| | | | |
|`cuModuleLoadDataEx`| | | | |`hipModuleLoadDataEx`|1.6.0| | | | |
|`cuModuleLoadFatBinary`| | | | | | | | | | |
|`cuModuleUnload`| | | | |`hipModuleUnload`|1.6.0| | | | |

## **11. Module Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuModuleGetSurfRef`| |12.0| | | | | | | | |
|`cuModuleGetTexRef`| |12.0| | |`hipModuleGetTexRef`|1.7.0| | | | |

## **12. Library Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuKernelGetAttribute`|12.0| | | | | | | | | |
|`cuKernelGetFunction`|12.0| | | | | | | | | |
|`cuKernelGetName`|12.3| | | | | | | | | |
|`cuKernelSetAttribute`|12.0| | | | | | | | | |
|`cuKernelSetCacheConfig`|12.0| | | | | | | | | |
|`cuLibraryGetGlobal`|12.0| | | | | | | | | |
|`cuLibraryGetKernel`|12.0| | | | | | | | | |
|`cuLibraryGetManaged`|12.0| | | | | | | | | |
|`cuLibraryGetModule`|12.0| | | | | | | | | |
|`cuLibraryGetUnifiedFunction`|12.0| | | | | | | | | |
|`cuLibraryLoadData`|12.0| | | | | | | | | |
|`cuLibraryLoadFromFile`|12.0| | | | | | | | | |
|`cuLibraryUnload`|12.0| | | | | | | | | |

## **13. Memory Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuArray3DCreate`| | | | |`hipArray3DCreate`|1.7.1| | | | |
|`cuArray3DCreate_v2`| | | | |`hipArray3DCreate`|1.7.1| | | | |
|`cuArray3DGetDescriptor`| | | | |`hipArray3DGetDescriptor`|5.6.0| | | | |
|`cuArray3DGetDescriptor_v2`| | | | |`hipArray3DGetDescriptor`|5.6.0| | | | |
|`cuArrayCreate`| | | | |`hipArrayCreate`|1.9.0| | | | |
|`cuArrayCreate_v2`| | | | |`hipArrayCreate`|1.9.0| | | | |
|`cuArrayDestroy`| | | | |`hipArrayDestroy`|4.2.0| | | | |
|`cuArrayGetDescriptor`| | | | |`hipArrayGetDescriptor`|5.6.0| | | | |
|`cuArrayGetDescriptor_v2`| | | | |`hipArrayGetDescriptor`|5.6.0| | | | |
|`cuArrayGetMemoryRequirements`|11.6| | | | | | | | | |
|`cuArrayGetPlane`|11.2| | | | | | | | | |
|`cuArrayGetSparseProperties`|11.1| | | | | | | | | |
|`cuDeviceGetByPCIBusId`| | | | |`hipDeviceGetByPCIBusId`|1.6.0| | | | |
|`cuDeviceGetPCIBusId`| | | | |`hipDeviceGetPCIBusId`|1.6.0| | | | |
|`cuIpcCloseMemHandle`| | | | |`hipIpcCloseMemHandle`|1.6.0| | | | |
|`cuIpcGetEventHandle`| | | | |`hipIpcGetEventHandle`|1.6.0| | | | |
|`cuIpcGetMemHandle`| | | | |`hipIpcGetMemHandle`|1.6.0| | | | |
|`cuIpcOpenEventHandle`| | | | |`hipIpcOpenEventHandle`|1.6.0| | | | |
|`cuIpcOpenMemHandle`| | | | |`hipIpcOpenMemHandle`|1.6.0| | | | |
|`cuMemAlloc`| | | | |`hipMalloc`|1.5.0| | | | |
|`cuMemAllocHost`| | | | |`hipMemAllocHost`|3.0.0|3.0.0| | | |
|`cuMemAllocHost_v2`| | | | |`hipMemAllocHost`|3.0.0|3.0.0| | | |
|`cuMemAllocManaged`| | | | |`hipMallocManaged`|2.5.0| | | | |
|`cuMemAllocPitch`| | | | |`hipMemAllocPitch`|3.0.0| | | | |
|`cuMemAllocPitch_v2`| | | | |`hipMemAllocPitch`|3.0.0| | | | |
|`cuMemAlloc_v2`| | | | |`hipMalloc`|1.5.0| | | | |
|`cuMemFree`| | | | |`hipFree`|1.5.0| | | | |
|`cuMemFreeHost`| | | | |`hipHostFree`|1.6.0| | | | |
|`cuMemFree_v2`| | | | |`hipFree`|1.5.0| | | | |
|`cuMemGetAddressRange`| | | | |`hipMemGetAddressRange`|1.9.0| | | | |
|`cuMemGetAddressRange_v2`| | | | |`hipMemGetAddressRange`|1.9.0| | | | |
|`cuMemGetHandleForAddressRange`|11.7| | | | | | | | | |
|`cuMemGetInfo`| | | | |`hipMemGetInfo`|1.6.0| | | | |
|`cuMemGetInfo_v2`| | | | |`hipMemGetInfo`|1.6.0| | | | |
|`cuMemHostAlloc`| | | | |`hipHostAlloc`|1.6.0| | | | |
|`cuMemHostGetDevicePointer`| | | | |`hipHostGetDevicePointer`|1.6.0| | | | |
|`cuMemHostGetDevicePointer_v2`| | | | |`hipHostGetDevicePointer`|1.6.0| | | | |
|`cuMemHostGetFlags`| | | | |`hipHostGetFlags`|1.6.0| | | | |
|`cuMemHostRegister`| | | | |`hipHostRegister`|1.6.0| | | | |
|`cuMemHostRegister_v2`| | | | |`hipHostRegister`|1.6.0| | | | |
|`cuMemHostUnregister`| | | | |`hipHostUnregister`|1.6.0| | | | |
|`cuMemcpy`| | | | | | | | | | |
|`cuMemcpy2D`| | | | |`hipMemcpyParam2D`|1.7.0| | | | |
|`cuMemcpy2DAsync`| | | | |`hipMemcpyParam2DAsync`|2.8.0| | | | |
|`cuMemcpy2DAsync_v2`| | | | |`hipMemcpyParam2DAsync`|2.8.0| | | | |
|`cuMemcpy2DUnaligned`| | | | |`hipDrvMemcpy2DUnaligned`|4.2.0| | | | |
|`cuMemcpy2DUnaligned_v2`| | | | |`hipDrvMemcpy2DUnaligned`|4.2.0| | | | |
|`cuMemcpy2D_v2`| | | | |`hipMemcpyParam2D`|1.7.0| | | | |
|`cuMemcpy3D`| | | | |`hipDrvMemcpy3D`|3.5.0| | | | |
|`cuMemcpy3DAsync`| | | | |`hipDrvMemcpy3DAsync`|3.5.0| | | | |
|`cuMemcpy3DAsync_v2`| | | | |`hipDrvMemcpy3DAsync`|3.5.0| | | | |
|`cuMemcpy3DPeer`| | | | | | | | | | |
|`cuMemcpy3DPeerAsync`| | | | | | | | | | |
|`cuMemcpy3D_v2`| | | | |`hipDrvMemcpy3D`|3.5.0| | | | |
|`cuMemcpyAsync`| | | | | | | | | | |
|`cuMemcpyAtoA`| | | | | | | | | | |
|`cuMemcpyAtoA_v2`| | | | | | | | | | |
|`cuMemcpyAtoD`| | | | | | | | | | |
|`cuMemcpyAtoD_v2`| | | | | | | | | | |
|`cuMemcpyAtoH`| | | | |`hipMemcpyAtoH`|1.9.0| | | | |
|`cuMemcpyAtoHAsync`| | | | | | | | | | |
|`cuMemcpyAtoHAsync_v2`| | | | | | | | | | |
|`cuMemcpyAtoH_v2`| | | | |`hipMemcpyAtoH`|1.9.0| | | | |
|`cuMemcpyDtoA`| | | | | | | | | | |
|`cuMemcpyDtoA_v2`| | | | | | | | | | |
|`cuMemcpyDtoD`| | | | |`hipMemcpyDtoD`|1.6.0| | | | |
|`cuMemcpyDtoDAsync`| | | | |`hipMemcpyDtoDAsync`|1.6.0| | | | |
|`cuMemcpyDtoDAsync_v2`| | | | |`hipMemcpyDtoDAsync`|1.6.0| | | | |
|`cuMemcpyDtoD_v2`| | | | |`hipMemcpyDtoD`|1.6.0| | | | |
|`cuMemcpyDtoH`| | | | |`hipMemcpyDtoH`|1.6.0| | | | |
|`cuMemcpyDtoHAsync`| | | | |`hipMemcpyDtoHAsync`|1.6.0| | | | |
|`cuMemcpyDtoHAsync_v2`| | | | |`hipMemcpyDtoHAsync`|1.6.0| | | | |
|`cuMemcpyDtoH_v2`| | | | |`hipMemcpyDtoH`|1.6.0| | | | |
|`cuMemcpyHtoA`| | | | |`hipMemcpyHtoA`|1.9.0| | | | |
|`cuMemcpyHtoAAsync`| | | | | | | | | | |
|`cuMemcpyHtoAAsync_v2`| | | | | | | | | | |
|`cuMemcpyHtoA_v2`| | | | |`hipMemcpyHtoA`|1.9.0| | | | |
|`cuMemcpyHtoD`| | | | |`hipMemcpyHtoD`|1.6.0| | | | |
|`cuMemcpyHtoDAsync`| | | | |`hipMemcpyHtoDAsync`|1.6.0| | | | |
|`cuMemcpyHtoDAsync_v2`| | | | |`hipMemcpyHtoDAsync`|1.6.0| | | | |
|`cuMemcpyHtoD_v2`| | | | |`hipMemcpyHtoD`|1.6.0| | | | |
|`cuMemcpyPeer`| | | | | | | | | | |
|`cuMemcpyPeerAsync`| | | | | | | | | | |
|`cuMemsetD16`| | | | |`hipMemsetD16`|3.0.0| | | | |
|`cuMemsetD16Async`| | | | |`hipMemsetD16Async`|3.0.0| | | | |
|`cuMemsetD16_v2`| | | | |`hipMemsetD16`|3.0.0| | | | |
|`cuMemsetD2D16`| | | | | | | | | | |
|`cuMemsetD2D16Async`| | | | | | | | | | |
|`cuMemsetD2D16_v2`| | | | | | | | | | |
|`cuMemsetD2D32`| | | | | | | | | | |
|`cuMemsetD2D32Async`| | | | | | | | | | |
|`cuMemsetD2D32_v2`| | | | | | | | | | |
|`cuMemsetD2D8`| | | | | | | | | | |
|`cuMemsetD2D8Async`| | | | | | | | | | |
|`cuMemsetD2D8_v2`| | | | | | | | | | |
|`cuMemsetD32`| | | | |`hipMemsetD32`|2.3.0| | | | |
|`cuMemsetD32Async`| | | | |`hipMemsetD32Async`|2.3.0| | | | |
|`cuMemsetD32_v2`| | | | |`hipMemsetD32`|2.3.0| | | | |
|`cuMemsetD8`| | | | |`hipMemsetD8`|1.6.0| | | | |
|`cuMemsetD8Async`| | | | |`hipMemsetD8Async`|3.0.0| | | | |
|`cuMemsetD8_v2`| | | | |`hipMemsetD8`|1.6.0| | | | |
|`cuMipmappedArrayCreate`| | | | |`hipMipmappedArrayCreate`|3.5.0|5.7.0| | | |
|`cuMipmappedArrayDestroy`| | | | |`hipMipmappedArrayDestroy`|3.5.0|5.7.0| | | |
|`cuMipmappedArrayGetLevel`| | | | |`hipMipmappedArrayGetLevel`|3.5.0|5.7.0| | | |
|`cuMipmappedArrayGetMemoryRequirements`|11.6| | | | | | | | | |

## **14. Virtual Memory Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuMemAddressFree`|10.2| | | |`hipMemAddressFree`|5.2.0| | | | |
|`cuMemAddressReserve`|10.2| | | |`hipMemAddressReserve`|5.2.0| | | | |
|`cuMemCreate`|10.2| | | |`hipMemCreate`|5.2.0| | | | |
|`cuMemExportToShareableHandle`|10.2| | | |`hipMemExportToShareableHandle`|5.2.0| | | | |
|`cuMemGetAccess`|10.2| | | |`hipMemGetAccess`|5.2.0| | | | |
|`cuMemGetAllocationGranularity`|10.2| | | |`hipMemGetAllocationGranularity`|5.2.0| | | | |
|`cuMemGetAllocationPropertiesFromHandle`|10.2| | | |`hipMemGetAllocationPropertiesFromHandle`|5.2.0| | | | |
|`cuMemImportFromShareableHandle`|10.2| | | |`hipMemImportFromShareableHandle`|5.2.0| | | | |
|`cuMemMap`|10.2| | | |`hipMemMap`|5.2.0| | | | |
|`cuMemMapArrayAsync`|11.1| | | |`hipMemMapArrayAsync`|5.2.0| | | | |
|`cuMemRelease`|10.2| | | |`hipMemRelease`|5.2.0| | | | |
|`cuMemRetainAllocationHandle`|11.0| | | |`hipMemRetainAllocationHandle`|5.2.0| | | | |
|`cuMemSetAccess`|10.2| | | |`hipMemSetAccess`|5.2.0| | | | |
|`cuMemUnmap`|10.2| | | |`hipMemUnmap`|5.2.0| | | | |

## **15. Stream Ordered Memory Allocator**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuMemAllocAsync`|11.2| | | |`hipMallocAsync`|5.2.0| | | | |
|`cuMemAllocFromPoolAsync`|11.2| | | |`hipMallocFromPoolAsync`|5.2.0| | | | |
|`cuMemFreeAsync`|11.2| | | |`hipFreeAsync`|5.2.0| | | | |
|`cuMemPoolCreate`|11.2| | | |`hipMemPoolCreate`|5.2.0| | | | |
|`cuMemPoolDestroy`|11.2| | | |`hipMemPoolDestroy`|5.2.0| | | | |
|`cuMemPoolExportPointer`|11.2| | | |`hipMemPoolExportPointer`|5.2.0| | | | |
|`cuMemPoolExportToShareableHandle`|11.2| | | |`hipMemPoolExportToShareableHandle`|5.2.0| | | | |
|`cuMemPoolGetAccess`|11.2| | | |`hipMemPoolGetAccess`|5.2.0| | | | |
|`cuMemPoolGetAttribute`|11.2| | | |`hipMemPoolGetAttribute`|5.2.0| | | | |
|`cuMemPoolImportFromShareableHandle`|11.2| | | |`hipMemPoolImportFromShareableHandle`|5.2.0| | | | |
|`cuMemPoolImportPointer`|11.2| | | |`hipMemPoolImportPointer`|5.2.0| | | | |
|`cuMemPoolSetAccess`|11.2| | | |`hipMemPoolSetAccess`|5.2.0| | | | |
|`cuMemPoolSetAttribute`|11.2| | | |`hipMemPoolSetAttribute`|5.2.0| | | | |
|`cuMemPoolTrimTo`|11.2| | | |`hipMemPoolTrimTo`|5.2.0| | | | |

## **16. Multicast Object Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuMulticastAddDevice`|12.1| | | | | | | | | |
|`cuMulticastBindAddr`|12.1| | | | | | | | | |
|`cuMulticastBindMem`|12.1| | | | | | | | | |
|`cuMulticastCreate`|12.1| | | | | | | | | |
|`cuMulticastGetGranularity`|12.1| | | | | | | | | |
|`cuMulticastUnbind`|12.1| | | | | | | | | |

## **17. Unified Addressing**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuMemAdvise`|8.0| | | |`hipMemAdvise`|3.7.0| | | | |
|`cuMemAdvise_v2`|12.2| | | | | | | | | |
|`cuMemPrefetchAsync`|8.0| | | |`hipMemPrefetchAsync`|3.7.0| | | | |
|`cuMemPrefetchAsync_v2`|12.2| | | | | | | | | |
|`cuMemRangeGetAttribute`|8.0| | | |`hipMemRangeGetAttribute`|3.7.0| | | | |
|`cuMemRangeGetAttributes`|8.0| | | |`hipMemRangeGetAttributes`|3.7.0| | | | |
|`cuPointerGetAttribute`| | | | |`hipPointerGetAttribute`|5.0.0| | | | |
|`cuPointerGetAttributes`| | | | |`hipDrvPointerGetAttributes`|5.0.0| | | | |
|`cuPointerSetAttribute`| | | | |`hipPointerSetAttribute`|5.5.0| | | | |

## **18. Stream Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuStreamAddCallback`| | | | |`hipStreamAddCallback`|1.6.0| | | | |
|`cuStreamAttachMemAsync`| | | | |`hipStreamAttachMemAsync`|3.7.0| | | | |
|`cuStreamBeginCapture`|10.0| | | |`hipStreamBeginCapture`|4.3.0| | | | |
|`cuStreamBeginCaptureToGraph`|12.3| | | | | | | | | |
|`cuStreamBeginCapture_ptsz`|10.1| | | | | | | | | |
|`cuStreamBeginCapture_v2`|10.1| | | |`hipStreamBeginCapture`|4.3.0| | | | |
|`cuStreamCopyAttributes`|11.0| | | | | | | | | |
|`cuStreamCreate`| | | | |`hipStreamCreateWithFlags`|1.6.0| | | | |
|`cuStreamCreateWithPriority`| | | | |`hipStreamCreateWithPriority`|2.0.0| | | | |
|`cuStreamDestroy`| | | | |`hipStreamDestroy`|1.6.0| | | | |
|`cuStreamDestroy_v2`| | | | |`hipStreamDestroy`|1.6.0| | | | |
|`cuStreamEndCapture`|10.0| | | |`hipStreamEndCapture`|4.3.0| | | | |
|`cuStreamGetAttribute`|11.0| | | | | | | | | |
|`cuStreamGetCaptureInfo`|10.1| | | |`hipStreamGetCaptureInfo`|5.0.0| | | | |
|`cuStreamGetCaptureInfo_v2`|11.3| | | |`hipStreamGetCaptureInfo_v2`|5.0.0| | | | |
|`cuStreamGetCaptureInfo_v3`|12.3| | | | | | | | | |
|`cuStreamGetCtx`|9.2| | | | | | | | | |
|`cuStreamGetFlags`| | | | |`hipStreamGetFlags`|1.6.0| | | | |
|`cuStreamGetId`|12.0| | | | | | | | | |
|`cuStreamGetPriority`| | | | |`hipStreamGetPriority`|2.0.0| | | | |
|`cuStreamIsCapturing`|10.0| | | |`hipStreamIsCapturing`|5.0.0| | | | |
|`cuStreamQuery`| | | | |`hipStreamQuery`|1.6.0| | | | |
|`cuStreamSetAttribute`|11.0| | | | | | | | | |
|`cuStreamSynchronize`| | | | |`hipStreamSynchronize`|1.6.0| | | | |
|`cuStreamUpdateCaptureDependencies`|11.3| | | |`hipStreamUpdateCaptureDependencies`|5.0.0| | | | |
|`cuStreamUpdateCaptureDependencies_v2`|12.3| | | | | | | | | |
|`cuStreamWaitEvent`| | | | |`hipStreamWaitEvent`|1.6.0| | | | |
|`cuThreadExchangeStreamCaptureMode`|10.1| | | |`hipThreadExchangeStreamCaptureMode`|5.2.0| | | | |

## **19. Event Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuEventCreate`| | | | |`hipEventCreateWithFlags`|1.6.0| | | | |
|`cuEventDestroy`| | | | |`hipEventDestroy`|1.6.0| | | | |
|`cuEventDestroy_v2`| | | | |`hipEventDestroy`|1.6.0| | | | |
|`cuEventElapsedTime`| | | | |`hipEventElapsedTime`|1.6.0| | | | |
|`cuEventQuery`| | | | |`hipEventQuery`|1.6.0| | | | |
|`cuEventRecord`| | | | |`hipEventRecord`|1.6.0| | | | |
|`cuEventRecordWithFlags`|11.1| | | | | | | | | |
|`cuEventSynchronize`| | | | |`hipEventSynchronize`|1.6.0| | | | |

## **20. External Resource Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDestroyExternalMemory`|10.0| | | |`hipDestroyExternalMemory`|4.3.0| | | | |
|`cuDestroyExternalSemaphore`|10.0| | | |`hipDestroyExternalSemaphore`|4.4.0| | | | |
|`cuExternalMemoryGetMappedBuffer`|10.0| | | |`hipExternalMemoryGetMappedBuffer`|4.3.0| | | | |
|`cuExternalMemoryGetMappedMipmappedArray`|10.0| | | | | | | | | |
|`cuImportExternalMemory`|10.0| | | |`hipImportExternalMemory`|4.3.0| | | | |
|`cuImportExternalSemaphore`|10.0| | | |`hipImportExternalSemaphore`|4.4.0| | | | |
|`cuSignalExternalSemaphoresAsync`|10.0| | | |`hipSignalExternalSemaphoresAsync`|4.4.0| | | | |
|`cuWaitExternalSemaphoresAsync`|10.0| | | |`hipWaitExternalSemaphoresAsync`|4.4.0| | | | |

## **21. Stream Memory Operations**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuStreamBatchMemOp`|8.0| | | | | | | | | |
|`cuStreamBatchMemOp_v2`|11.7| | | | | | | | | |
|`cuStreamWaitValue32`|8.0| | | |`hipStreamWaitValue32`|4.2.0| | | | |
|`cuStreamWaitValue32_v2`|11.7| | | |`hipStreamWaitValue32`|4.2.0| | | | |
|`cuStreamWaitValue64`|9.0| | | |`hipStreamWaitValue64`|4.2.0| | | | |
|`cuStreamWaitValue64_v2`|11.7| | | |`hipStreamWaitValue64`|4.2.0| | | | |
|`cuStreamWriteValue32`|8.0| | | |`hipStreamWriteValue32`|4.2.0| | | | |
|`cuStreamWriteValue32_v2`|11.7| | | |`hipStreamWriteValue32`|4.2.0| | | | |
|`cuStreamWriteValue64`|9.0| | | |`hipStreamWriteValue64`|4.2.0| | | | |
|`cuStreamWriteValue64_v2`|11.7| | | |`hipStreamWriteValue64`|4.2.0| | | | |

## **22. Execution Control**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuFuncGetAttribute`| | | | |`hipFuncGetAttribute`|2.8.0| | | | |
|`cuFuncGetModule`|11.0| | | | | | | | | |
|`cuFuncGetName`|12.3| | | | | | | | | |
|`cuFuncSetAttribute`|9.0| | | | | | | | | |
|`cuFuncSetCacheConfig`| | | | | | | | | | |
|`cuFuncSetSharedMemConfig`| | | | | | | | | | |
|`cuLaunchCooperativeKernel`|9.0| | | |`hipModuleLaunchCooperativeKernel`|5.5.0| | | | |
|`cuLaunchCooperativeKernelMultiDevice`|9.0|11.3| | |`hipModuleLaunchCooperativeKernelMultiDevice`|5.5.0| | | | |
|`cuLaunchHostFunc`|10.0| | | |`hipLaunchHostFunc`|5.2.0| | | | |
|`cuLaunchKernel`| | | | |`hipModuleLaunchKernel`|1.6.0| | | | |
|`cuLaunchKernelEx`|11.8| | | | | | | | | |

## **23. Execution Control [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuFuncSetBlockShape`| |9.2| | | | | | | | |
|`cuFuncSetSharedSize`| |9.2| | | | | | | | |
|`cuLaunch`| |9.2| | | | | | | | |
|`cuLaunchGrid`| |9.2| | | | | | | | |
|`cuLaunchGridAsync`| |9.2| | | | | | | | |
|`cuParamSetSize`| |9.2| | | | | | | | |
|`cuParamSetTexRef`| |9.2| | | | | | | | |
|`cuParamSetf`| |9.2| | | | | | | | |
|`cuParamSeti`| |9.2| | | | | | | | |
|`cuParamSetv`| |9.2| | | | | | | | |

## **24. Graph Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuDeviceGetGraphMemAttribute`|11.4| | | |`hipDeviceGetGraphMemAttribute`|5.3.0| | | | |
|`cuDeviceGraphMemTrim`|11.4| | | |`hipDeviceGraphMemTrim`|5.3.0| | | | |
|`cuDeviceSetGraphMemAttribute`|11.4| | | |`hipDeviceSetGraphMemAttribute`|5.3.0| | | | |
|`cuGraphAddBatchMemOpNode`|11.7| | | |`hipGraphAddBatchMemOpNode`| | | | | |
|`cuGraphAddChildGraphNode`|10.0| | | |`hipGraphAddChildGraphNode`|5.0.0| | | | |
|`cuGraphAddDependencies`|10.0| | | |`hipGraphAddDependencies`|4.5.0| | | | |
|`cuGraphAddDependencies_v2`|12.3| | | | | | | | | |
|`cuGraphAddEmptyNode`|10.0| | | |`hipGraphAddEmptyNode`|4.5.0| | | | |
|`cuGraphAddEventRecordNode`|11.1| | | |`hipGraphAddEventRecordNode`|5.0.0| | | | |
|`cuGraphAddEventWaitNode`|11.1| | | |`hipGraphAddEventWaitNode`|5.0.0| | | | |
|`cuGraphAddExternalSemaphoresSignalNode`|11.2| | | |`hipGraphAddExternalSemaphoresSignalNode`|6.1.0| | | |6.1.0|
|`cuGraphAddExternalSemaphoresWaitNode`|11.2| | | |`hipGraphAddExternalSemaphoresWaitNode`|6.1.0| | | |6.1.0|
|`cuGraphAddHostNode`|10.0| | | |`hipGraphAddHostNode`|5.0.0| | | | |
|`cuGraphAddKernelNode`|10.0| | | |`hipGraphAddKernelNode`|4.3.0| | | | |
|`cuGraphAddMemAllocNode`|11.4| | | |`hipGraphAddMemAllocNode`|5.5.0| | | | |
|`cuGraphAddMemFreeNode`|11.4| | | |`hipDrvGraphAddMemFreeNode`|6.1.0| | | |6.1.0|
|`cuGraphAddMemcpyNode`|10.0| | | |`hipDrvGraphAddMemcpyNode`|6.0.0| | | | |
|`cuGraphAddMemsetNode`|10.0| | | |`hipDrvGraphAddMemsetNode`|6.1.0| | | |6.1.0|
|`cuGraphAddNode`|12.2| | | |`hipGraphAddNode`|6.1.0| | | |6.1.0|
|`cuGraphAddNode_v2`|12.3| | | | | | | | | |
|`cuGraphBatchMemOpNodeGetParams`|11.7| | | |`hipGraphBatchMemOpNodeGetParams`| | | | | |
|`cuGraphBatchMemOpNodeSetParams`|11.7| | | |`hipGraphBatchMemOpNodeSetParams`| | | | | |
|`cuGraphChildGraphNodeGetGraph`|10.0| | | |`hipGraphChildGraphNodeGetGraph`|5.0.0| | | | |
|`cuGraphClone`|10.0| | | |`hipGraphClone`|5.0.0| | | | |
|`cuGraphConditionalHandleCreate`|12.3| | | | | | | | | |
|`cuGraphCreate`|10.0| | | |`hipGraphCreate`|4.3.0| | | | |
|`cuGraphDebugDotPrint`|11.3| | | |`hipGraphDebugDotPrint`|5.5.0| | | | |
|`cuGraphDestroy`|10.0| | | |`hipGraphDestroy`|4.3.0| | | | |
|`cuGraphDestroyNode`|10.0| | | |`hipGraphDestroyNode`|5.0.0| | | | |
|`cuGraphEventRecordNodeGetEvent`|11.1| | | |`hipGraphEventRecordNodeGetEvent`|5.0.0| | | | |
|`cuGraphEventRecordNodeSetEvent`|11.1| | | |`hipGraphEventRecordNodeSetEvent`|5.0.0| | | | |
|`cuGraphEventWaitNodeGetEvent`|11.1| | | |`hipGraphEventWaitNodeGetEvent`|5.0.0| | | | |
|`cuGraphEventWaitNodeSetEvent`|11.1| | | |`hipGraphEventWaitNodeSetEvent`|5.0.0| | | | |
|`cuGraphExecBatchMemOpNodeSetParams`|11.7| | | |`hipGraphExecBatchMemOpNodeSetParams`| | | | | |
|`cuGraphExecChildGraphNodeSetParams`|11.1| | | |`hipGraphExecChildGraphNodeSetParams`|5.0.0| | | | |
|`cuGraphExecDestroy`|10.0| | | |`hipGraphExecDestroy`|4.3.0| | | | |
|`cuGraphExecEventRecordNodeSetEvent`|11.1| | | |`hipGraphExecEventRecordNodeSetEvent`|5.0.0| | | | |
|`cuGraphExecEventWaitNodeSetEvent`|11.1| | | |`hipGraphExecEventWaitNodeSetEvent`|5.0.0| | | | |
|`cuGraphExecExternalSemaphoresSignalNodeSetParams`|11.2| | | |`hipGraphExecExternalSemaphoresSignalNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExecExternalSemaphoresWaitNodeSetParams`|11.2| | | |`hipGraphExecExternalSemaphoresWaitNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExecGetFlags`|12.0| | | |`hipGraphExecGetFlags`|6.1.0| | | |6.1.0|
|`cuGraphExecHostNodeSetParams`|10.2| | | |`hipGraphExecHostNodeSetParams`|5.0.0| | | | |
|`cuGraphExecKernelNodeSetParams`|10.1| | | |`hipGraphExecKernelNodeSetParams`|4.5.0| | | | |
|`cuGraphExecMemcpyNodeSetParams`|10.2| | | |`hipDrvGraphExecMemcpyNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExecMemsetNodeSetParams`|10.2| | | |`hipDrvGraphExecMemsetNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExecNodeSetParams`|12.2| | | |`hipGraphExecNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExecUpdate`|10.2| | | |`hipGraphExecUpdate`|5.0.0| | | | |
|`cuGraphExternalSemaphoresSignalNodeGetParams`|11.2| | | |`hipGraphExternalSemaphoresSignalNodeGetParams`|6.1.0| | | |6.1.0|
|`cuGraphExternalSemaphoresSignalNodeSetParams`|11.2| | | |`hipGraphExternalSemaphoresSignalNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphExternalSemaphoresWaitNodeGetParams`|11.2| | | |`hipGraphExternalSemaphoresWaitNodeGetParams`|6.1.0| | | |6.1.0|
|`cuGraphExternalSemaphoresWaitNodeSetParams`|11.2| | | |`hipGraphExternalSemaphoresWaitNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphGetEdges`|10.0| | | |`hipGraphGetEdges`|5.0.0| | | | |
|`cuGraphGetEdges_v2`|12.3| | | | | | | | | |
|`cuGraphGetNodes`|10.0| | | |`hipGraphGetNodes`|4.5.0| | | | |
|`cuGraphGetRootNodes`|10.0| | | |`hipGraphGetRootNodes`|4.5.0| | | | |
|`cuGraphHostNodeGetParams`|10.0| | | |`hipGraphHostNodeGetParams`|5.0.0| | | | |
|`cuGraphHostNodeSetParams`|10.0| | | |`hipGraphHostNodeSetParams`|5.0.0| | | | |
|`cuGraphInstantiate`|10.0| | | |`hipGraphInstantiate`|4.3.0| | | | |
|`cuGraphInstantiateWithFlags`|11.4| | | |`hipGraphInstantiateWithFlags`|5.0.0| | | | |
|`cuGraphInstantiateWithParams`|12.0| | | |`hipGraphInstantiateWithParams`|6.1.0| | | |6.1.0|
|`cuGraphInstantiate_v2`|11.0| | | |`hipGraphInstantiate`|4.3.0| | | | |
|`cuGraphKernelNodeCopyAttributes`|11.0| | | |`hipGraphKernelNodeCopyAttributes`|5.5.0| | | | |
|`cuGraphKernelNodeGetAttribute`|11.0| | | |`hipGraphKernelNodeGetAttribute`|5.2.0| | | | |
|`cuGraphKernelNodeGetParams`|10.0| | | |`hipGraphKernelNodeGetParams`|4.5.0| | | | |
|`cuGraphKernelNodeSetAttribute`|11.0| | | |`hipGraphKernelNodeSetAttribute`|5.2.0| | | | |
|`cuGraphKernelNodeSetParams`|10.0| | | |`hipGraphKernelNodeSetParams`|4.5.0| | | | |
|`cuGraphLaunch`|10.0| | | |`hipGraphLaunch`|4.3.0| | | | |
|`cuGraphMemAllocNodeGetParams`|11.4| | | |`hipGraphMemAllocNodeGetParams`|5.5.0| | | | |
|`cuGraphMemFreeNodeGetParams`|11.4| | | |`hipGraphMemFreeNodeGetParams`|5.5.0| | | | |
|`cuGraphMemcpyNodeGetParams`|10.0| | | |`hipDrvGraphMemcpyNodeGetParams`|6.1.0| | | |6.1.0|
|`cuGraphMemcpyNodeSetParams`|10.0| | | |`hipDrvGraphMemcpyNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphMemsetNodeGetParams`|10.0| | | |`hipGraphMemsetNodeGetParams`|4.5.0| | | | |
|`cuGraphMemsetNodeSetParams`|10.0| | | |`hipGraphMemsetNodeSetParams`|4.5.0| | | | |
|`cuGraphNodeFindInClone`|10.0| | | |`hipGraphNodeFindInClone`|5.0.0| | | | |
|`cuGraphNodeGetDependencies`|10.0| | | |`hipGraphNodeGetDependencies`|5.0.0| | | | |
|`cuGraphNodeGetDependencies_v2`|12.3| | | | | | | | | |
|`cuGraphNodeGetDependentNodes`|10.0| | | |`hipGraphNodeGetDependentNodes`|5.0.0| | | | |
|`cuGraphNodeGetDependentNodes_v2`|12.3| | | | | | | | | |
|`cuGraphNodeGetEnabled`|11.6| | | |`hipGraphNodeGetEnabled`|5.5.0| | | | |
|`cuGraphNodeGetType`|10.0| | | |`hipGraphNodeGetType`|5.0.0| | | | |
|`cuGraphNodeSetEnabled`|11.6| | | |`hipGraphNodeSetEnabled`|5.5.0| | | | |
|`cuGraphNodeSetParams`|12.2| | | |`hipGraphNodeSetParams`|6.1.0| | | |6.1.0|
|`cuGraphReleaseUserObject`|11.3| | | |`hipGraphReleaseUserObject`|5.3.0| | | | |
|`cuGraphRemoveDependencies`|10.0| | | |`hipGraphRemoveDependencies`|5.0.0| | | | |
|`cuGraphRemoveDependencies_v2`|12.3| | | | | | | | | |
|`cuGraphRetainUserObject`|11.3| | | |`hipGraphRetainUserObject`|5.3.0| | | | |
|`cuGraphUpload`|11.1| | | |`hipGraphUpload`|5.3.0| | | | |
|`cuUserObjectCreate`|11.3| | | |`hipUserObjectCreate`|5.3.0| | | | |
|`cuUserObjectRelease`|11.3| | | |`hipUserObjectRelease`|5.3.0| | | | |
|`cuUserObjectRetain`|11.3| | | |`hipUserObjectRetain`|5.3.0| | | | |

## **25. Occupancy**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuOccupancyAvailableDynamicSMemPerBlock`|11.0| | | | | | | | | |
|`cuOccupancyMaxActiveBlocksPerMultiprocessor`| | | | |`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`|3.5.0| | | | |
|`cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`| | | | |`hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|3.5.0| | | | |
|`cuOccupancyMaxActiveClusters`|11.8| | | | | | | | | |
|`cuOccupancyMaxPotentialBlockSize`| | | | |`hipModuleOccupancyMaxPotentialBlockSize`|3.5.0| | | | |
|`cuOccupancyMaxPotentialBlockSizeWithFlags`| | | | |`hipModuleOccupancyMaxPotentialBlockSizeWithFlags`|3.5.0| | | | |
|`cuOccupancyMaxPotentialClusterSize`|11.8| | | | | | | | | |

## **26. Texture Reference Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuTexRefCreate`| |11.0| | | | | | | | |
|`cuTexRefDestroy`| |11.0| | | | | | | | |
|`cuTexRefGetAddress`| |11.0| | |`hipTexRefGetAddress`|3.0.0|4.3.0| | | |
|`cuTexRefGetAddressMode`| |11.0| | |`hipTexRefGetAddressMode`|3.0.0|4.3.0| | | |
|`cuTexRefGetAddress_v2`| |11.0| | |`hipTexRefGetAddress`|3.0.0|4.3.0| | | |
|`cuTexRefGetArray`| |11.0| | |`hipTexRefGetArray`|3.0.0| | |4.2.0| |
|`cuTexRefGetBorderColor`|8.0|11.0| | | | | | | | |
|`cuTexRefGetFilterMode`| |11.0| | |`hipTexRefGetFilterMode`|3.5.0|4.3.0| | | |
|`cuTexRefGetFlags`| |11.0| | |`hipTexRefGetFlags`|3.5.0|4.3.0| | | |
|`cuTexRefGetFormat`| |11.0| | |`hipTexRefGetFormat`|3.5.0|4.3.0| | | |
|`cuTexRefGetMaxAnisotropy`| |11.0| | |`hipTexRefGetMaxAnisotropy`|3.5.0|4.3.0| | | |
|`cuTexRefGetMipmapFilterMode`| |11.0| | |`hipTexRefGetMipmapFilterMode`|3.5.0|4.3.0| | | |
|`cuTexRefGetMipmapLevelBias`| |11.0| | |`hipTexRefGetMipmapLevelBias`|3.5.0|4.3.0| | | |
|`cuTexRefGetMipmapLevelClamp`| |11.0| | |`hipTexRefGetMipmapLevelClamp`|3.5.0|4.3.0| | | |
|`cuTexRefGetMipmappedArray`| |11.0| | |`hipTexRefGetMipMappedArray`|3.5.0|4.3.0| | | |
|`cuTexRefSetAddress`| |11.0| | |`hipTexRefSetAddress`|1.7.0|4.3.0| | | |
|`cuTexRefSetAddress2D`| |11.0| | |`hipTexRefSetAddress2D`|1.7.0|4.3.0| | | |
|`cuTexRefSetAddress2D_v2`| | | | |`hipTexRefSetAddress2D`|1.7.0|4.3.0| | | |
|`cuTexRefSetAddress2D_v3`| | | | |`hipTexRefSetAddress2D`|1.7.0|4.3.0| | | |
|`cuTexRefSetAddressMode`| |11.0| | |`hipTexRefSetAddressMode`|1.9.0|5.3.0| | | |
|`cuTexRefSetAddress_v2`| |11.0| | |`hipTexRefSetAddress`|1.7.0|4.3.0| | | |
|`cuTexRefSetArray`| |11.0| | |`hipTexRefSetArray`|1.9.0|5.3.0| | | |
|`cuTexRefSetBorderColor`|8.0|11.0| | |`hipTexRefSetBorderColor`|3.5.0|4.3.0| | | |
|`cuTexRefSetFilterMode`| |11.0| | |`hipTexRefSetFilterMode`|1.9.0|5.3.0| | | |
|`cuTexRefSetFlags`| |11.0| | |`hipTexRefSetFlags`|1.9.0|5.3.0| | | |
|`cuTexRefSetFormat`| |11.0| | |`hipTexRefSetFormat`|1.9.0|5.3.0| | | |
|`cuTexRefSetMaxAnisotropy`| |11.0| | |`hipTexRefSetMaxAnisotropy`|3.5.0|4.3.0| | | |
|`cuTexRefSetMipmapFilterMode`| |11.0| | |`hipTexRefSetMipmapFilterMode`|3.5.0|5.3.0| | | |
|`cuTexRefSetMipmapLevelBias`| |11.0| | |`hipTexRefSetMipmapLevelBias`|3.5.0|5.3.0| | | |
|`cuTexRefSetMipmapLevelClamp`| |11.0| | |`hipTexRefSetMipmapLevelClamp`|3.5.0|5.3.0| | | |
|`cuTexRefSetMipmappedArray`| |11.0| | |`hipTexRefSetMipmappedArray`|3.5.0|5.3.0| | | |

## **27. Surface Reference Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuSurfRefGetArray`| |11.0| | | | | | | | |
|`cuSurfRefSetArray`| |11.0| | | | | | | | |

## **28. Texture Object Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuTexObjectCreate`| | | | |`hipTexObjectCreate`|3.5.0| | | | |
|`cuTexObjectDestroy`| | | | |`hipTexObjectDestroy`|3.5.0| | | | |
|`cuTexObjectGetResourceDesc`| | | | |`hipTexObjectGetResourceDesc`|3.5.0| | | | |
|`cuTexObjectGetResourceViewDesc`| | | | |`hipTexObjectGetResourceViewDesc`|3.5.0| | | | |
|`cuTexObjectGetTextureDesc`| | | | |`hipTexObjectGetTextureDesc`|3.5.0| | | | |

## **29. Surface Object Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuSurfObjectCreate`| | | | | | | | | | |
|`cuSurfObjectDestroy`| | | | | | | | | | |
|`cuSurfObjectGetResourceDesc`| | | | | | | | | | |

## **30. Tensor Core Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuTensorMapEncodeIm2col`|12.0| | | | | | | | | |
|`cuTensorMapEncodeTiled`|12.0| | | | | | | | | |
|`cuTensorMapReplaceAddress`|12.0| | | | | | | | | |

## **31. Peer Context Memory Access**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuCtxDisablePeerAccess`| | | | |`hipCtxDisablePeerAccess`|1.6.0|1.9.0| | | |
|`cuCtxEnablePeerAccess`| | | | |`hipCtxEnablePeerAccess`|1.6.0|1.9.0| | | |
|`cuDeviceCanAccessPeer`| | | | |`hipDeviceCanAccessPeer`|1.9.0| | | | |
|`cuDeviceGetP2PAttribute`|8.0| | | |`hipDeviceGetP2PAttribute`|3.8.0| | | | |

## **32. Graphics Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuGraphicsMapResources`| | | | |`hipGraphicsMapResources`|4.5.0| | | | |
|`cuGraphicsResourceGetMappedMipmappedArray`| | | | | | | | | | |
|`cuGraphicsResourceGetMappedPointer`| | | | |`hipGraphicsResourceGetMappedPointer`|4.5.0| | | | |
|`cuGraphicsResourceGetMappedPointer_v2`| | | | |`hipGraphicsResourceGetMappedPointer`|4.5.0| | | | |
|`cuGraphicsResourceSetMapFlags`| | | | | | | | | | |
|`cuGraphicsResourceSetMapFlags_v2`| | | | | | | | | | |
|`cuGraphicsSubResourceGetMappedArray`| | | | |`hipGraphicsSubResourceGetMappedArray`|5.1.0| | | | |
|`cuGraphicsUnmapResources`| | | | |`hipGraphicsUnmapResources`|4.5.0| | | | |
|`cuGraphicsUnregisterResource`| | | | |`hipGraphicsUnregisterResource`|4.5.0| | | | |

## **33. Driver Entry Point Access**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuGetProcAddress`|11.3| | | |`hipGetProcAddress`|6.1.0| | | |6.1.0|

## **34. Coredump Attributes Control API**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuCoredumpGetAttribute`|12.1| | | | | | | | | |
|`cuCoredumpGetAttributeGlobal`|12.1| | | | | | | | | |
|`cuCoredumpSetAttribute`|12.1| | | | | | | | | |
|`cuCoredumpSetAttributeGlobal`|12.1| | | | | | | | | |

## **35. Profiler Control [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuProfilerInitialize`| |11.0| | | | | | | | |

## **36. Profiler Control**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuProfilerStart`| | | | |`hipProfilerStart`|1.6.0|3.0.0| | | |
|`cuProfilerStop`| | | | |`hipProfilerStop`|1.6.0|3.0.0| | | |

## **37. OpenGL Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuGLCtxCreate`| |9.2| | | | | | | | |
|`cuGLGetDevices`| | | | |`hipGLGetDevices`|4.5.0| | | | |
|`cuGLInit`| |9.2| | | | | | | | |
|`cuGLMapBufferObject`| |9.2| | | | | | | | |
|`cuGLMapBufferObjectAsync`| |9.2| | | | | | | | |
|`cuGLRegisterBufferObject`| |9.2| | | | | | | | |
|`cuGLSetBufferObjectMapFlags`| |9.2| | | | | | | | |
|`cuGLUnmapBufferObject`| |9.2| | | | | | | | |
|`cuGLUnmapBufferObjectAsync`| |9.2| | | | | | | | |
|`cuGLUnregisterBufferObject`| |9.2| | | | | | | | |
|`cuGraphicsGLRegisterBuffer`| | | | |`hipGraphicsGLRegisterBuffer`|4.5.0| | | | |
|`cuGraphicsGLRegisterImage`| | | | |`hipGraphicsGLRegisterImage`|5.1.0| | | | |
|`cuWGLGetDevice`| | | | | | | | | | |

## **38. Direct3D 9 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuD3D9CtxCreate`| | | | | | | | | | |
|`cuD3D9CtxCreateOnDevice`| | | | | | | | | | |
|`cuD3D9GetDevice`| | | | | | | | | | |
|`cuD3D9GetDevices`| | | | | | | | | | |
|`cuD3D9GetDirect3DDevice`| | | | | | | | | | |
|`cuD3D9MapResources`| |9.2| | | | | | | | |
|`cuD3D9RegisterResource`| |9.2| | | | | | | | |
|`cuD3D9ResourceGetMappedArray`| |9.2| | | | | | | | |
|`cuD3D9ResourceGetMappedPitch`| |9.2| | | | | | | | |
|`cuD3D9ResourceGetMappedPointer`| |9.2| | | | | | | | |
|`cuD3D9ResourceGetMappedSize`| |9.2| | | | | | | | |
|`cuD3D9ResourceGetSurfaceDimensions`| |9.2| | | | | | | | |
|`cuD3D9ResourceSetMapFlags`| |9.2| | | | | | | | |
|`cuD3D9UnmapResources`| |9.2| | | | | | | | |
|`cuD3D9UnregisterResource`| |9.2| | | | | | | | |
|`cuGraphicsD3D9RegisterResource`| | | | | | | | | | |

## **39. Direct3D 10 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuD3D10CtxCreate`| |9.2| | | | | | | | |
|`cuD3D10CtxCreateOnDevice`| |9.2| | | | | | | | |
|`cuD3D10GetDevice`| | | | | | | | | | |
|`cuD3D10GetDevices`| | | | | | | | | | |
|`cuD3D10GetDirect3DDevice`| |9.2| | | | | | | | |
|`cuD3D10MapResources`| |9.2| | | | | | | | |
|`cuD3D10RegisterResource`| |9.2| | | | | | | | |
|`cuD3D10ResourceGetMappedArray`| |9.2| | | | | | | | |
|`cuD3D10ResourceGetMappedPitch`| |9.2| | | | | | | | |
|`cuD3D10ResourceGetMappedPointer`| |9.2| | | | | | | | |
|`cuD3D10ResourceGetMappedSize`| |9.2| | | | | | | | |
|`cuD3D10ResourceGetSurfaceDimensions`| |9.2| | | | | | | | |
|`cuD3D10ResourceSetMapFlags`| |9.2| | | | | | | | |
|`cuD3D10UnmapResources`| |9.2| | | | | | | | |
|`cuD3D10UnregisterResource`| |9.2| | | | | | | | |
|`cuGraphicsD3D10RegisterResource`| | | | | | | | | | |

## **40. Direct3D 11 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuD3D11CtxCreate`| |9.2| | | | | | | | |
|`cuD3D11CtxCreateOnDevice`| |9.2| | | | | | | | |
|`cuD3D11GetDevice`| | | | | | | | | | |
|`cuD3D11GetDevices`| | | | | | | | | | |
|`cuD3D11GetDirect3DDevice`| |9.2| | | | | | | | |
|`cuGraphicsD3D11RegisterResource`| | | | | | | | | | |

## **41. VDPAU Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuGraphicsVDPAURegisterOutputSurface`| | | | | | | | | | |
|`cuGraphicsVDPAURegisterVideoSurface`| | | | | | | | | | |
|`cuVDPAUCtxCreate`| | | | | | | | | | |
|`cuVDPAUGetDevice`| | | | | | | | | | |

## **42. EGL Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cuEGLStreamConsumerAcquireFrame`|9.1| | | | | | | | | |
|`cuEGLStreamConsumerConnect`|9.1| | | | | | | | | |
|`cuEGLStreamConsumerConnectWithFlags`|9.1| | | | | | | | | |
|`cuEGLStreamConsumerDisconnect`|9.1| | | | | | | | | |
|`cuEGLStreamConsumerReleaseFrame`|9.1| | | | | | | | | |
|`cuEGLStreamProducerConnect`|9.1| | | | | | | | | |
|`cuEGLStreamProducerDisconnect`|9.1| | | | | | | | | |
|`cuEGLStreamProducerPresentFrame`|9.1| | | | | | | | | |
|`cuEGLStreamProducerReturnFrame`|9.1| | | | | | | | | |
|`cuEventCreateFromEGLSync`|9.1| | | | | | | | | |
|`cuGraphicsEGLRegisterImage`|9.1| | | | | | | | | |
|`cuGraphicsResourceGetMappedEglFrame`|9.1| | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental