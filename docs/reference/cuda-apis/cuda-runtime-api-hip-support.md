# CUDA Runtime API supported by HIP

## **1. Device Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaChooseDevice`| | | | |`hipChooseDevice`|1.6.0| | | | |
|`cudaDeviceFlushGPUDirectRDMAWrites`|11.3| | | | | | | | | |
|`cudaDeviceGetAttribute`| | | | |`hipDeviceGetAttribute`|1.6.0| | | | |
|`cudaDeviceGetByPCIBusId`| | | | |`hipDeviceGetByPCIBusId`|1.6.0| | | | |
|`cudaDeviceGetCacheConfig`| | | | |`hipDeviceGetCacheConfig`|1.6.0| | | | |
|`cudaDeviceGetDefaultMemPool`|11.2| | | |`hipDeviceGetDefaultMemPool`|5.2.0| | | | |
|`cudaDeviceGetLimit`| | | | |`hipDeviceGetLimit`|1.6.0| | | | |
|`cudaDeviceGetMemPool`|11.2| | | |`hipDeviceGetMemPool`|5.2.0| | | | |
|`cudaDeviceGetNvSciSyncAttributes`|10.2| | | | | | | | | |
|`cudaDeviceGetP2PAttribute`|8.0| | | |`hipDeviceGetP2PAttribute`|3.8.0| | | | |
|`cudaDeviceGetPCIBusId`| | | | |`hipDeviceGetPCIBusId`|1.6.0| | | | |
|`cudaDeviceGetSharedMemConfig`| | | | |`hipDeviceGetSharedMemConfig`|1.6.0| | | | |
|`cudaDeviceGetStreamPriorityRange`| | | | |`hipDeviceGetStreamPriorityRange`|2.0.0| | | | |
|`cudaDeviceGetTexture1DLinearMaxWidth`|11.1| | | | | | | | | |
|`cudaDeviceReset`| | | | |`hipDeviceReset`|1.6.0| | | | |
|`cudaDeviceSetCacheConfig`| | | | |`hipDeviceSetCacheConfig`|1.6.0| | | | |
|`cudaDeviceSetLimit`| | | | |`hipDeviceSetLimit`|5.3.0| | | | |
|`cudaDeviceSetMemPool`|11.2| | | |`hipDeviceSetMemPool`|5.2.0| | | | |
|`cudaDeviceSetSharedMemConfig`| | | | |`hipDeviceSetSharedMemConfig`|1.6.0| | | | |
|`cudaDeviceSynchronize`| | | | |`hipDeviceSynchronize`|1.6.0| | | | |
|`cudaGetDevice`| | | | |`hipGetDevice`|1.6.0| | | | |
|`cudaGetDeviceCount`| | | | |`hipGetDeviceCount`|1.6.0| | | | |
|`cudaGetDeviceFlags`| | | | |`hipGetDeviceFlags`|3.6.0| | | | |
|`cudaGetDeviceProperties`| | | | |`hipGetDeviceProperties`|1.6.0| | | | |
|`cudaInitDevice`|12.0| | | | | | | | | |
|`cudaIpcCloseMemHandle`| | | | |`hipIpcCloseMemHandle`|1.6.0| | | | |
|`cudaIpcGetEventHandle`| | | | |`hipIpcGetEventHandle`|1.6.0| | | | |
|`cudaIpcGetMemHandle`| | | | |`hipIpcGetMemHandle`|1.6.0| | | | |
|`cudaIpcOpenEventHandle`| | | | |`hipIpcOpenEventHandle`|1.6.0| | | | |
|`cudaIpcOpenMemHandle`| | | | |`hipIpcOpenMemHandle`|1.6.0| | | | |
|`cudaSetDevice`| | | | |`hipSetDevice`|1.6.0| | | | |
|`cudaSetDeviceFlags`| | | | |`hipSetDeviceFlags`|1.6.0| | | | |
|`cudaSetValidDevices`| | | | | | | | | | |

## **2. Thread Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaThreadExit`| |10.0| | |`hipDeviceReset`|1.6.0| | | | |
|`cudaThreadGetCacheConfig`| |10.0| | |`hipDeviceGetCacheConfig`|1.6.0| | | | |
|`cudaThreadGetLimit`| |10.0| | | | | | | | |
|`cudaThreadSetCacheConfig`| |10.0| | |`hipDeviceSetCacheConfig`|1.6.0| | | | |
|`cudaThreadSetLimit`| |10.0| | | | | | | | |
|`cudaThreadSynchronize`| |10.0| | |`hipDeviceSynchronize`|1.6.0| | | | |

## **3. Error Handling**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGetErrorName`| | | | |`hipGetErrorName`|1.6.0| | | | |
|`cudaGetErrorString`| | | | |`hipGetErrorString`|1.6.0| | | | |
|`cudaGetLastError`| | | | |`hipGetLastError`|1.6.0| | | | |
|`cudaPeekAtLastError`| | | | |`hipPeekAtLastError`|1.6.0| | | | |

## **4. Stream Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaCtxResetPersistingL2Cache`|11.0| | | | | | | | | |
|`cudaStreamAddCallback`| | | | |`hipStreamAddCallback`|1.6.0| | | | |
|`cudaStreamAttachMemAsync`| | | | |`hipStreamAttachMemAsync`|3.7.0| | | | |
|`cudaStreamBeginCapture`|10.0| | | |`hipStreamBeginCapture`|4.3.0| | | | |
|`cudaStreamBeginCaptureToGraph`|12.3| | | | | | | | | |
|`cudaStreamCopyAttributes`|11.0| | | | | | | | | |
|`cudaStreamCreate`| | | | |`hipStreamCreate`|1.6.0| | | | |
|`cudaStreamCreateWithFlags`| | | | |`hipStreamCreateWithFlags`|1.6.0| | | | |
|`cudaStreamCreateWithPriority`| | | | |`hipStreamCreateWithPriority`|2.0.0| | | | |
|`cudaStreamDestroy`| | | | |`hipStreamDestroy`|1.6.0| | | | |
|`cudaStreamEndCapture`|10.0| | | |`hipStreamEndCapture`|4.3.0| | | | |
|`cudaStreamGetAttribute`|11.0| | | | | | | | | |
|`cudaStreamGetCaptureInfo`|10.1| | | |`hipStreamGetCaptureInfo`|5.0.0| | | | |
|`cudaStreamGetCaptureInfo_v3`|12.3| | | | | | | | | |
|`cudaStreamGetFlags`| | | | |`hipStreamGetFlags`|1.6.0| | | | |
|`cudaStreamGetId`|12.0| | | | | | | | | |
|`cudaStreamGetPriority`| | | | |`hipStreamGetPriority`|2.0.0| | | | |
|`cudaStreamIsCapturing`|10.0| | | |`hipStreamIsCapturing`|5.0.0| | | | |
|`cudaStreamQuery`| | | | |`hipStreamQuery`|1.6.0| | | | |
|`cudaStreamSetAttribute`|11.0| | | | | | | | | |
|`cudaStreamSynchronize`| | | | |`hipStreamSynchronize`|1.6.0| | | | |
|`cudaStreamUpdateCaptureDependencies`|11.3| | | |`hipStreamUpdateCaptureDependencies`|5.0.0| | | | |
|`cudaStreamUpdateCaptureDependencies_v2`|12.3| | | | | | | | | |
|`cudaStreamWaitEvent`| | | | |`hipStreamWaitEvent`|1.6.0| | | | |
|`cudaThreadExchangeStreamCaptureMode`|10.1| | | |`hipThreadExchangeStreamCaptureMode`|5.2.0| | | | |

## **5. Event Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaEventCreate`| | | | |`hipEventCreate`|1.6.0| | | | |
|`cudaEventCreateWithFlags`| | | | |`hipEventCreateWithFlags`|1.6.0| | | | |
|`cudaEventDestroy`| | | | |`hipEventDestroy`|1.6.0| | | | |
|`cudaEventElapsedTime`| | | | |`hipEventElapsedTime`|1.6.0| | | | |
|`cudaEventQuery`| | | | |`hipEventQuery`|1.6.0| | | | |
|`cudaEventRecord`| | | | |`hipEventRecord`|1.6.0| | | | |
|`cudaEventRecordWithFlags`|11.1| | | | | | | | | |
|`cudaEventSynchronize`| | | | |`hipEventSynchronize`|1.6.0| | | | |

## **6. External Resource Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaDestroyExternalMemory`|10.0| | | |`hipDestroyExternalMemory`|4.3.0| | | | |
|`cudaDestroyExternalSemaphore`|10.0| | | |`hipDestroyExternalSemaphore`|4.4.0| | | | |
|`cudaExternalMemoryGetMappedBuffer`|10.0| | | |`hipExternalMemoryGetMappedBuffer`|4.3.0| | | | |
|`cudaExternalMemoryGetMappedMipmappedArray`|10.0| | | | | | | | | |
|`cudaImportExternalMemory`|10.0| | | |`hipImportExternalMemory`|4.3.0| | | | |
|`cudaImportExternalSemaphore`|10.0| | | |`hipImportExternalSemaphore`|4.4.0| | | | |
|`cudaSignalExternalSemaphoresAsync`|10.0| | | |`hipSignalExternalSemaphoresAsync`|4.4.0| | | | |
|`cudaWaitExternalSemaphoresAsync`|10.0| | | |`hipWaitExternalSemaphoresAsync`|4.4.0| | | | |

## **7. Execution Control**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaFuncGetAttributes`| | | | |`hipFuncGetAttributes`|1.9.0| | | | |
|`cudaFuncGetName`|12.3| | | | | | | | | |
|`cudaFuncSetAttribute`|9.0| | | |`hipFuncSetAttribute`|3.9.0| | | | |
|`cudaFuncSetCacheConfig`| | | | |`hipFuncSetCacheConfig`|1.6.0| | | | |
|`cudaFuncSetSharedMemConfig`| | | | |`hipFuncSetSharedMemConfig`|3.9.0| | | | |
|`cudaGetParameterBuffer`| | | | | | | | | | |
|`cudaGetParameterBufferV2`| | | | | | | | | | |
|`cudaLaunchCooperativeKernel`|9.0| | | |`hipLaunchCooperativeKernel`|2.6.0| | | | |
|`cudaLaunchCooperativeKernelMultiDevice`|9.0|11.3| | |`hipLaunchCooperativeKernelMultiDevice`|2.6.0| | | | |
|`cudaLaunchHostFunc`|10.0| | | |`hipLaunchHostFunc`|5.2.0| | | | |
|`cudaLaunchKernel`| | | | |`hipLaunchKernel`|1.6.0| | | | |
|`cudaLaunchKernelExC`|11.8| | | | | | | | | |
|`cudaSetDoubleForDevice`| |10.0| | | | | | | | |
|`cudaSetDoubleForHost`| |10.0| | | | | | | | |

## **8. Occupancy**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaOccupancyAvailableDynamicSMemPerBlock`|11.0| | | | | | | | | |
|`cudaOccupancyMaxActiveBlocksPerMultiprocessor`| | | | |`hipOccupancyMaxActiveBlocksPerMultiprocessor`|1.6.0| | | | |
|`cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`| | | | |`hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags`|2.6.0| | | | |
|`cudaOccupancyMaxActiveClusters`|11.8| | | | | | | | | |
|`cudaOccupancyMaxPotentialBlockSize`| | | | |`hipOccupancyMaxPotentialBlockSize`|1.6.0| | | | |
|`cudaOccupancyMaxPotentialBlockSizeVariableSMem`| | | | |`hipOccupancyMaxPotentialBlockSizeVariableSMem`|5.5.0| | | | |
|`cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags`| | | | |`hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags`|5.5.0| | | | |
|`cudaOccupancyMaxPotentialBlockSizeWithFlags`| | | | |`hipOccupancyMaxPotentialBlockSizeWithFlags`|3.5.0| | | | |
|`cudaOccupancyMaxPotentialClusterSize`|11.8| | | | | | | | | |

## **9. Memory Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaArrayGetInfo`| | | | |`hipArrayGetInfo`|5.6.0| | | | |
|`cudaArrayGetMemoryRequirements`|11.6| | | | | | | | | |
|`cudaArrayGetPlane`|11.2| | | | | | | | | |
|`cudaArrayGetSparseProperties`|11.1| | | | | | | | | |
|`cudaFree`| | | | |`hipFree`|1.5.0| | | | |
|`cudaFreeArray`| | | | |`hipFreeArray`|1.6.0| | | | |
|`cudaFreeHost`| | | | |`hipHostFree`|1.6.0| | | | |
|`cudaFreeMipmappedArray`| | | | |`hipFreeMipmappedArray`|3.5.0| | | | |
|`cudaGetMipmappedArrayLevel`| | | | |`hipGetMipmappedArrayLevel`|3.5.0| | | | |
|`cudaGetSymbolAddress`| | | | |`hipGetSymbolAddress`|2.0.0| | | | |
|`cudaGetSymbolSize`| | | | |`hipGetSymbolSize`|2.0.0| | | | |
|`cudaHostAlloc`| | | | |`hipHostAlloc`|1.6.0| | | | |
|`cudaHostGetDevicePointer`| | | | |`hipHostGetDevicePointer`|1.6.0| | | | |
|`cudaHostGetFlags`| | | | |`hipHostGetFlags`|1.6.0| | | | |
|`cudaHostRegister`| | | | |`hipHostRegister`|1.6.0| | | | |
|`cudaHostUnregister`| | | | |`hipHostUnregister`|1.6.0| | | | |
|`cudaMalloc`| | | | |`hipMalloc`|1.5.0| | | | |
|`cudaMalloc3D`| | | | |`hipMalloc3D`|1.9.0| | | | |
|`cudaMalloc3DArray`| | | | |`hipMalloc3DArray`|1.7.0| | | | |
|`cudaMallocArray`| | | | |`hipMallocArray`|1.6.0| | | | |
|`cudaMallocHost`| | | | |`hipHostMalloc`|1.6.0| | | | |
|`cudaMallocManaged`| | | | |`hipMallocManaged`|2.5.0| | | | |
|`cudaMallocMipmappedArray`| | | | |`hipMallocMipmappedArray`|3.5.0| | | | |
|`cudaMallocPitch`| | | | |`hipMallocPitch`|1.6.0| | | | |
|`cudaMemAdvise`|8.0| | | |`hipMemAdvise`|3.7.0| | | | |
|`cudaMemAdvise_v2`|12.2| | | | | | | | | |
|`cudaMemGetInfo`| | | | |`hipMemGetInfo`|1.6.0| | | | |
|`cudaMemPrefetchAsync`|8.0| | | |`hipMemPrefetchAsync`|3.7.0| | | | |
|`cudaMemPrefetchAsync_v2`|12.2| | | | | | | | | |
|`cudaMemRangeGetAttribute`|8.0| | | |`hipMemRangeGetAttribute`|3.7.0| | | | |
|`cudaMemRangeGetAttributes`|8.0| | | |`hipMemRangeGetAttributes`|3.7.0| | | | |
|`cudaMemcpy`| | | | |`hipMemcpy`|1.5.0| | | | |
|`cudaMemcpy2D`| | | | |`hipMemcpy2D`|1.6.0| | | | |
|`cudaMemcpy2DArrayToArray`| | | | | | | | | | |
|`cudaMemcpy2DAsync`| | | | |`hipMemcpy2DAsync`|1.6.0| | | | |
|`cudaMemcpy2DFromArray`| | | | |`hipMemcpy2DFromArray`|3.0.0| | | | |
|`cudaMemcpy2DFromArrayAsync`| | | | |`hipMemcpy2DFromArrayAsync`|3.0.0| | | | |
|`cudaMemcpy2DToArray`| | | | |`hipMemcpy2DToArray`|1.6.0| | | | |
|`cudaMemcpy2DToArrayAsync`| | | | |`hipMemcpy2DToArrayAsync`|4.3.0| | | | |
|`cudaMemcpy3D`| | | | |`hipMemcpy3D`|1.6.0| | | | |
|`cudaMemcpy3DAsync`| | | | |`hipMemcpy3DAsync`|2.8.0| | | | |
|`cudaMemcpy3DPeer`| | | | | | | | | | |
|`cudaMemcpy3DPeerAsync`| | | | | | | | | | |
|`cudaMemcpyAsync`| | | | |`hipMemcpyAsync`|1.6.0| | | | |
|`cudaMemcpyFromSymbol`| | | | |`hipMemcpyFromSymbol`|1.6.0| | | | |
|`cudaMemcpyFromSymbolAsync`| | | | |`hipMemcpyFromSymbolAsync`|1.6.0| | | | |
|`cudaMemcpyPeer`| | | | |`hipMemcpyPeer`|1.6.0| | | | |
|`cudaMemcpyPeerAsync`| | | | |`hipMemcpyPeerAsync`|1.6.0| | | | |
|`cudaMemcpyToSymbol`| | | | |`hipMemcpyToSymbol`|1.6.0| | | | |
|`cudaMemcpyToSymbolAsync`| | | | |`hipMemcpyToSymbolAsync`|1.6.0| | | | |
|`cudaMemset`| | | | |`hipMemset`|1.6.0| | | | |
|`cudaMemset2D`| | | | |`hipMemset2D`|1.7.0| | | | |
|`cudaMemset2DAsync`| | | | |`hipMemset2DAsync`|1.9.0| | | | |
|`cudaMemset3D`| | | | |`hipMemset3D`|1.9.0| | | | |
|`cudaMemset3DAsync`| | | | |`hipMemset3DAsync`|1.9.0| | | | |
|`cudaMemsetAsync`| | | | |`hipMemsetAsync`|1.6.0| | | | |
|`make_cudaExtent`| | | | |`make_hipExtent`|1.7.0| | | | |
|`make_cudaPitchedPtr`| | | | |`make_hipPitchedPtr`|1.7.0| | | | |
|`make_cudaPos`| | | | |`make_hipPos`|1.7.0| | | | |

## **10. Memory Management [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaMemcpyArrayToArray`| |10.1| | | | | | | | |
|`cudaMemcpyFromArray`| |10.1| | |`hipMemcpyFromArray`|1.9.0|3.8.0| | | |
|`cudaMemcpyFromArrayAsync`| |10.1| | | | | | | | |
|`cudaMemcpyToArray`| |10.1| | |`hipMemcpyToArray`|1.6.0|3.8.0| | | |
|`cudaMemcpyToArrayAsync`| |10.1| | | | | | | | |

## **11. Stream Ordered Memory Allocator**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaFreeAsync`|11.2| | | |`hipFreeAsync`|5.2.0| | | | |
|`cudaMallocAsync`|11.2| | | |`hipMallocAsync`|5.2.0| | | | |
|`cudaMallocFromPoolAsync`|11.2| | | |`hipMallocFromPoolAsync`|5.2.0| | | | |
|`cudaMemPoolCreate`|11.2| | | |`hipMemPoolCreate`|5.2.0| | | | |
|`cudaMemPoolDestroy`|11.2| | | |`hipMemPoolDestroy`|5.2.0| | | | |
|`cudaMemPoolExportPointer`|11.2| | | |`hipMemPoolExportPointer`|5.2.0| | | | |
|`cudaMemPoolExportToShareableHandle`|11.2| | | |`hipMemPoolExportToShareableHandle`|5.2.0| | | | |
|`cudaMemPoolGetAccess`|11.2| | | |`hipMemPoolGetAccess`|5.2.0| | | | |
|`cudaMemPoolGetAttribute`|11.2| | | |`hipMemPoolGetAttribute`|5.2.0| | | | |
|`cudaMemPoolImportFromShareableHandle`|11.2| | | |`hipMemPoolImportFromShareableHandle`|5.2.0| | | | |
|`cudaMemPoolImportPointer`|11.2| | | |`hipMemPoolImportPointer`|5.2.0| | | | |
|`cudaMemPoolSetAccess`|11.2| | | |`hipMemPoolSetAccess`|5.2.0| | | | |
|`cudaMemPoolSetAttribute`|11.2| | | |`hipMemPoolSetAttribute`|5.2.0| | | | |
|`cudaMemPoolTrimTo`|11.2| | | |`hipMemPoolTrimTo`|5.2.0| | | | |

## **12. Unified Addressing**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaPointerGetAttributes`| | | | |`hipPointerGetAttributes`|1.6.0| | | | |

## **13. Peer Device Memory Access**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaDeviceCanAccessPeer`| | | | |`hipDeviceCanAccessPeer`|1.9.0| | | | |
|`cudaDeviceDisablePeerAccess`| | | | |`hipDeviceDisablePeerAccess`|1.9.0| | | | |
|`cudaDeviceEnablePeerAccess`| | | | |`hipDeviceEnablePeerAccess`|1.9.0| | | | |

## **14. OpenGL Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGLGetDevices`| | | | |`hipGLGetDevices`|4.5.0| | | | |
|`cudaGraphicsGLRegisterBuffer`| | | | |`hipGraphicsGLRegisterBuffer`|4.5.0| | | | |
|`cudaGraphicsGLRegisterImage`| | | | |`hipGraphicsGLRegisterImage`|5.1.0| | | | |
|`cudaWGLGetDevice`| | | | | | | | | | |

## **15. OpenGL Interoperability [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGLMapBufferObject`| |10.0| | | | | | | | |
|`cudaGLMapBufferObjectAsync`| |10.0| | | | | | | | |
|`cudaGLRegisterBufferObject`| |10.0| | | | | | | | |
|`cudaGLSetBufferObjectMapFlags`| |10.0| | | | | | | | |
|`cudaGLSetGLDevice`| |10.0| | | | | | | | |
|`cudaGLUnmapBufferObject`| |10.0| | | | | | | | |
|`cudaGLUnmapBufferObjectAsync`| |10.0| | | | | | | | |
|`cudaGLUnregisterBufferObject`| |10.0| | | | | | | | |

## **16. Direct3D 9 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D9GetDevice`| | | | | | | | | | |
|`cudaD3D9GetDevices`| | | | | | | | | | |
|`cudaD3D9GetDirect3DDevice`| | | | | | | | | | |
|`cudaD3D9SetDirect3DDevice`| | | | | | | | | | |
|`cudaGraphicsD3D9RegisterResource`| | | | | | | | | | |

## **17. Direct3D 9 Interoperability [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D9MapResources`| |10.0| | | | | | | | |
|`cudaD3D9RegisterResource`| | | | | | | | | | |
|`cudaD3D9ResourceGetMappedArray`| |10.0| | | | | | | | |
|`cudaD3D9ResourceGetMappedPitch`| |10.0| | | | | | | | |
|`cudaD3D9ResourceGetMappedPointer`| |10.0| | | | | | | | |
|`cudaD3D9ResourceGetMappedSize`| |10.0| | | | | | | | |
|`cudaD3D9ResourceGetSurfaceDimensions`| |10.0| | | | | | | | |
|`cudaD3D9ResourceSetMapFlags`| |10.0| | | | | | | | |
|`cudaD3D9UnmapResources`| |10.0| | | | | | | | |
|`cudaD3D9UnregisterResource`| |10.0| | | | | | | | |

## **18. Direct3D 10 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D10GetDevice`| | | | | | | | | | |
|`cudaD3D10GetDevices`| | | | | | | | | | |
|`cudaGraphicsD3D10RegisterResource`| | | | | | | | | | |

## **19. Direct3D 10 Interoperability [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D10GetDirect3DDevice`| |10.0| | | | | | | | |
|`cudaD3D10MapResources`| |10.0| | | | | | | | |
|`cudaD3D10RegisterResource`| |10.0| | | | | | | | |
|`cudaD3D10ResourceGetMappedArray`| |10.0| | | | | | | | |
|`cudaD3D10ResourceGetMappedPitch`| |10.0| | | | | | | | |
|`cudaD3D10ResourceGetMappedPointer`| |10.0| | | | | | | | |
|`cudaD3D10ResourceGetMappedSize`| |10.0| | | | | | | | |
|`cudaD3D10ResourceGetSurfaceDimensions`| |10.0| | | | | | | | |
|`cudaD3D10ResourceSetMapFlags`| |10.0| | | | | | | | |
|`cudaD3D10SetDirect3DDevice`| |10.0| | | | | | | | |
|`cudaD3D10UnmapResources`| |10.0| | | | | | | | |
|`cudaD3D10UnregisterResource`| |10.0| | | | | | | | |

## **20. Direct3D 11 Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D11GetDevice`| | | | | | | | | | |
|`cudaD3D11GetDevices`| | | | | | | | | | |
|`cudaGraphicsD3D11RegisterResource`| | | | | | | | | | |

## **21. Direct3D 11 Interoperability [DEPRECATED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaD3D11GetDirect3DDevice`| |10.0| | | | | | | | |
|`cudaD3D11SetDirect3DDevice`| |10.0| | | | | | | | |

## **22. VDPAU Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGraphicsVDPAURegisterOutputSurface`| | | | | | | | | | |
|`cudaGraphicsVDPAURegisterVideoSurface`| | | | | | | | | | |
|`cudaVDPAUGetDevice`| | | | | | | | | | |
|`cudaVDPAUSetVDPAUDevice`| | | | | | | | | | |

## **23. EGL Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaEGLStreamConsumerAcquireFrame`|9.1| | | | | | | | | |
|`cudaEGLStreamConsumerConnect`|9.1| | | | | | | | | |
|`cudaEGLStreamConsumerConnectWithFlags`|9.1| | | | | | | | | |
|`cudaEGLStreamConsumerDisconnect`|9.1| | | | | | | | | |
|`cudaEGLStreamConsumerReleaseFrame`|9.1| | | | | | | | | |
|`cudaEGLStreamProducerConnect`|9.1| | | | | | | | | |
|`cudaEGLStreamProducerDisconnect`|9.1| | | | | | | | | |
|`cudaEGLStreamProducerPresentFrame`|9.1| | | | | | | | | |
|`cudaEGLStreamProducerReturnFrame`|9.1| | | | | | | | | |
|`cudaEventCreateFromEGLSync`|9.1| | | | | | | | | |
|`cudaGraphicsEGLRegisterImage`|9.1| | | | | | | | | |
|`cudaGraphicsResourceGetMappedEglFrame`|9.1| | | | | | | | | |

## **24. Graphics Interoperability**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGraphicsMapResources`| | | | |`hipGraphicsMapResources`|4.5.0| | | | |
|`cudaGraphicsResourceGetMappedMipmappedArray`| | | | | | | | | | |
|`cudaGraphicsResourceGetMappedPointer`| | | | |`hipGraphicsResourceGetMappedPointer`|4.5.0| | | | |
|`cudaGraphicsResourceSetMapFlags`| | | | | | | | | | |
|`cudaGraphicsSubResourceGetMappedArray`| | | | |`hipGraphicsSubResourceGetMappedArray`|5.1.0| | | | |
|`cudaGraphicsUnmapResources`| | | | |`hipGraphicsUnmapResources`|4.5.0| | | | |
|`cudaGraphicsUnregisterResource`| | | | |`hipGraphicsUnregisterResource`|4.5.0| | | | |

## **25. Texture Object Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaCreateChannelDesc`| | | | |`hipCreateChannelDesc`|1.6.0| | | | |
|`cudaCreateTextureObject`| | | | |`hipCreateTextureObject`|1.7.0| | | | |
|`cudaCreateTextureObject_v2`|11.8| | |12.0| | | | | | |
|`cudaDestroyTextureObject`| | | | |`hipDestroyTextureObject`|1.7.0| | | | |
|`cudaGetChannelDesc`| | | | |`hipGetChannelDesc`|1.7.0| | | | |
|`cudaGetTextureObjectResourceDesc`| | | | |`hipGetTextureObjectResourceDesc`|1.7.0| | | | |
|`cudaGetTextureObjectResourceViewDesc`| | | | |`hipGetTextureObjectResourceViewDesc`|1.7.0| | | | |
|`cudaGetTextureObjectTextureDesc`| | | | |`hipGetTextureObjectTextureDesc`|1.7.0| | | | |
|`cudaGetTextureObjectTextureDesc_v2`|11.8| | |12.0| | | | | | |

## **26. Surface Object Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaCreateSurfaceObject`| | | | |`hipCreateSurfaceObject`|1.9.0| | | | |
|`cudaDestroySurfaceObject`| | | | |`hipDestroySurfaceObject`|1.9.0| | | | |
|`cudaGetSurfaceObjectResourceDesc`| | | | | | | | | | |

## **27. Version Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaDriverGetVersion`| | | | |`hipDriverGetVersion`|1.6.0| | | | |
|`cudaRuntimeGetVersion`| | | | |`hipRuntimeGetVersion`|1.6.0| | | | |

## **28. Graph Management**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaDeviceGetGraphMemAttribute`|11.4| | | |`hipDeviceGetGraphMemAttribute`|5.3.0| | | | |
|`cudaDeviceGraphMemTrim`|11.4| | | |`hipDeviceGraphMemTrim`|5.3.0| | | | |
|`cudaDeviceSetGraphMemAttribute`|11.4| | | |`hipDeviceSetGraphMemAttribute`|5.3.0| | | | |
|`cudaGraphAddChildGraphNode`|10.0| | | |`hipGraphAddChildGraphNode`|5.0.0| | | | |
|`cudaGraphAddDependencies`|10.0| | | |`hipGraphAddDependencies`|4.5.0| | | | |
|`cudaGraphAddDependencies_v2`|12.3| | | | | | | | | |
|`cudaGraphAddEmptyNode`|10.0| | | |`hipGraphAddEmptyNode`|4.5.0| | | | |
|`cudaGraphAddEventRecordNode`|11.1| | | |`hipGraphAddEventRecordNode`|5.0.0| | | | |
|`cudaGraphAddEventWaitNode`|11.1| | | |`hipGraphAddEventWaitNode`|5.0.0| | | | |
|`cudaGraphAddExternalSemaphoresSignalNode`|11.2| | | |`hipGraphAddExternalSemaphoresSignalNode`|6.1.0| | | |6.1.0|
|`cudaGraphAddExternalSemaphoresWaitNode`|11.2| | | |`hipGraphAddExternalSemaphoresWaitNode`|6.1.0| | | |6.1.0|
|`cudaGraphAddHostNode`|10.0| | | |`hipGraphAddHostNode`|5.0.0| | | | |
|`cudaGraphAddKernelNode`|10.0| | | |`hipGraphAddKernelNode`|4.3.0| | | | |
|`cudaGraphAddMemAllocNode`|11.4| | | |`hipGraphAddMemAllocNode`|5.5.0| | | | |
|`cudaGraphAddMemFreeNode`|11.4| | | |`hipGraphAddMemFreeNode`|5.5.0| | | | |
|`cudaGraphAddMemcpyNode`|10.0| | | |`hipGraphAddMemcpyNode`|4.3.0| | | | |
|`cudaGraphAddMemcpyNode1D`|11.1| | | |`hipGraphAddMemcpyNode1D`|4.5.0| | | | |
|`cudaGraphAddMemcpyNodeFromSymbol`|11.1| | | |`hipGraphAddMemcpyNodeFromSymbol`|5.0.0| | | | |
|`cudaGraphAddMemcpyNodeToSymbol`|11.1| | | |`hipGraphAddMemcpyNodeToSymbol`|5.0.0| | | | |
|`cudaGraphAddMemsetNode`|10.0| | | |`hipGraphAddMemsetNode`|4.3.0| | | | |
|`cudaGraphAddNode`|12.2| | | |`hipGraphAddNode`|6.1.0| | | |6.1.0|
|`cudaGraphAddNode_v2`|12.3| | | | | | | | | |
|`cudaGraphChildGraphNodeGetGraph`|10.0| | | |`hipGraphChildGraphNodeGetGraph`|5.0.0| | | | |
|`cudaGraphClone`|10.0| | | |`hipGraphClone`|5.0.0| | | | |
|`cudaGraphConditionalHandleCreate`|12.3| | | | | | | | | |
|`cudaGraphCreate`|10.0| | | |`hipGraphCreate`|4.3.0| | | | |
|`cudaGraphDebugDotPrint`|11.3| | | |`hipGraphDebugDotPrint`|5.5.0| | | | |
|`cudaGraphDestroy`|10.0| | | |`hipGraphDestroy`|4.3.0| | | | |
|`cudaGraphDestroyNode`|10.0| | | |`hipGraphDestroyNode`|5.0.0| | | | |
|`cudaGraphEventRecordNodeGetEvent`|11.1| | | |`hipGraphEventRecordNodeGetEvent`|5.0.0| | | | |
|`cudaGraphEventRecordNodeSetEvent`|11.1| | | |`hipGraphEventRecordNodeSetEvent`|5.0.0| | | | |
|`cudaGraphEventWaitNodeGetEvent`|11.1| | | |`hipGraphEventWaitNodeGetEvent`|5.0.0| | | | |
|`cudaGraphEventWaitNodeSetEvent`|11.1| | | |`hipGraphEventWaitNodeSetEvent`|5.0.0| | | | |
|`cudaGraphExecChildGraphNodeSetParams`|11.1| | | |`hipGraphExecChildGraphNodeSetParams`|5.0.0| | | | |
|`cudaGraphExecDestroy`|10.0| | | |`hipGraphExecDestroy`|4.3.0| | | | |
|`cudaGraphExecEventRecordNodeSetEvent`|11.1| | | |`hipGraphExecEventRecordNodeSetEvent`|5.0.0| | | | |
|`cudaGraphExecEventWaitNodeSetEvent`|11.1| | | |`hipGraphExecEventWaitNodeSetEvent`|5.0.0| | | | |
|`cudaGraphExecExternalSemaphoresSignalNodeSetParams`|11.2| | | |`hipGraphExecExternalSemaphoresSignalNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExecExternalSemaphoresWaitNodeSetParams`|11.2| | | |`hipGraphExecExternalSemaphoresWaitNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExecGetFlags`|12.0| | | |`hipGraphExecGetFlags`|6.1.0| | | |6.1.0|
|`cudaGraphExecHostNodeSetParams`|11.0| | | |`hipGraphExecHostNodeSetParams`|5.0.0| | | | |
|`cudaGraphExecKernelNodeSetParams`|11.0| | | |`hipGraphExecKernelNodeSetParams`|4.5.0| | | | |
|`cudaGraphExecMemcpyNodeSetParams`|11.0| | | |`hipGraphExecMemcpyNodeSetParams`|5.0.0| | | | |
|`cudaGraphExecMemcpyNodeSetParams1D`|11.1| | | |`hipGraphExecMemcpyNodeSetParams1D`|5.0.0| | | | |
|`cudaGraphExecMemcpyNodeSetParamsFromSymbol`|11.1| | | |`hipGraphExecMemcpyNodeSetParamsFromSymbol`|5.0.0| | | | |
|`cudaGraphExecMemcpyNodeSetParamsToSymbol`|11.1| | | |`hipGraphExecMemcpyNodeSetParamsToSymbol`|5.0.0| | | | |
|`cudaGraphExecMemsetNodeSetParams`|11.0| | | |`hipGraphExecMemsetNodeSetParams`|5.0.0| | | | |
|`cudaGraphExecNodeSetParams`|12.2| | | |`hipGraphExecNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExecUpdate`|11.0| | | |`hipGraphExecUpdate`|5.0.0| | | | |
|`cudaGraphExternalSemaphoresSignalNodeGetParams`|11.2| | | |`hipGraphExternalSemaphoresSignalNodeGetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExternalSemaphoresSignalNodeSetParams`|11.2| | | |`hipGraphExternalSemaphoresSignalNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExternalSemaphoresWaitNodeGetParams`|11.2| | | |`hipGraphExternalSemaphoresWaitNodeGetParams`|6.1.0| | | |6.1.0|
|`cudaGraphExternalSemaphoresWaitNodeSetParams`|11.2| | | |`hipGraphExternalSemaphoresWaitNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphGetEdges`|10.0| | | |`hipGraphGetEdges`|5.0.0| | | | |
|`cudaGraphGetEdges_v2`|12.3| | | | | | | | | |
|`cudaGraphGetNodes`|10.0| | | |`hipGraphGetNodes`|4.5.0| | | | |
|`cudaGraphGetRootNodes`|10.0| | | |`hipGraphGetRootNodes`|4.5.0| | | | |
|`cudaGraphHostNodeGetParams`|10.0| | | |`hipGraphHostNodeGetParams`|5.0.0| | | | |
|`cudaGraphHostNodeSetParams`|10.0| | | |`hipGraphHostNodeSetParams`|5.0.0| | | | |
|`cudaGraphInstantiate`|10.0| | | |`hipGraphInstantiate`|4.3.0| | | | |
|`cudaGraphInstantiateWithFlags`|11.4| | | |`hipGraphInstantiateWithFlags`|5.0.0| | | | |
|`cudaGraphInstantiateWithParams`|12.0| | | |`hipGraphInstantiateWithParams`|6.1.0| | | |6.1.0|
|`cudaGraphKernelNodeCopyAttributes`|11.0| | | |`hipGraphKernelNodeCopyAttributes`|5.5.0| | | | |
|`cudaGraphKernelNodeGetAttribute`|11.0| | | |`hipGraphKernelNodeGetAttribute`|5.2.0| | | | |
|`cudaGraphKernelNodeGetParams`|11.0| | | |`hipGraphKernelNodeGetParams`|4.5.0| | | | |
|`cudaGraphKernelNodeSetAttribute`|11.0| | | |`hipGraphKernelNodeSetAttribute`|5.2.0| | | | |
|`cudaGraphKernelNodeSetParams`|11.0| | | |`hipGraphKernelNodeSetParams`|4.5.0| | | | |
|`cudaGraphLaunch`|11.0| | | |`hipGraphLaunch`|4.3.0| | | | |
|`cudaGraphMemAllocNodeGetParams`|11.4| | | |`hipGraphMemAllocNodeGetParams`|5.5.0| | | | |
|`cudaGraphMemFreeNodeGetParams`|11.4| | | |`hipGraphMemFreeNodeGetParams`|5.5.0| | | | |
|`cudaGraphMemcpyNodeGetParams`|11.0| | | |`hipGraphMemcpyNodeGetParams`|4.5.0| | | | |
|`cudaGraphMemcpyNodeSetParams`|11.0| | | |`hipGraphMemcpyNodeSetParams`|4.5.0| | | | |
|`cudaGraphMemcpyNodeSetParams1D`|11.1| | | |`hipGraphMemcpyNodeSetParams1D`|5.0.0| | | | |
|`cudaGraphMemcpyNodeSetParamsFromSymbol`|11.1| | | |`hipGraphMemcpyNodeSetParamsFromSymbol`|5.0.0| | | | |
|`cudaGraphMemcpyNodeSetParamsToSymbol`|11.1| | | |`hipGraphMemcpyNodeSetParamsToSymbol`|5.0.0| | | | |
|`cudaGraphMemsetNodeGetParams`|11.0| | | |`hipGraphMemsetNodeGetParams`|4.5.0| | | | |
|`cudaGraphMemsetNodeSetParams`|11.0| | | |`hipGraphMemsetNodeSetParams`|4.5.0| | | | |
|`cudaGraphNodeFindInClone`|11.0| | | |`hipGraphNodeFindInClone`|5.0.0| | | | |
|`cudaGraphNodeGetDependencies`|11.0| | | |`hipGraphNodeGetDependencies`|5.0.0| | | | |
|`cudaGraphNodeGetDependencies_v2`|12.3| | | | | | | | | |
|`cudaGraphNodeGetDependentNodes`|11.0| | | |`hipGraphNodeGetDependentNodes`|5.0.0| | | | |
|`cudaGraphNodeGetDependentNodes_v2`|12.3| | | | | | | | | |
|`cudaGraphNodeGetEnabled`|11.6| | | |`hipGraphNodeGetEnabled`|5.5.0| | | | |
|`cudaGraphNodeGetType`|11.0| | | |`hipGraphNodeGetType`|5.0.0| | | | |
|`cudaGraphNodeSetEnabled`|11.6| | | |`hipGraphNodeSetEnabled`|5.5.0| | | | |
|`cudaGraphNodeSetParams`|12.2| | | |`hipGraphNodeSetParams`|6.1.0| | | |6.1.0|
|`cudaGraphReleaseUserObject`|11.3| | | |`hipGraphReleaseUserObject`|5.3.0| | | | |
|`cudaGraphRemoveDependencies`|11.0| | | |`hipGraphRemoveDependencies`|5.0.0| | | | |
|`cudaGraphRemoveDependencies_v2`|12.3| | | | | | | | | |
|`cudaGraphRetainUserObject`|11.3| | | |`hipGraphRetainUserObject`|5.3.0| | | | |
|`cudaGraphUpload`|11.1| | | |`hipGraphUpload`|5.3.0| | | | |
|`cudaUserObjectCreate`|11.3| | | |`hipUserObjectCreate`|5.3.0| | | | |
|`cudaUserObjectRelease`|11.3| | | |`hipUserObjectRelease`|5.3.0| | | | |
|`cudaUserObjectRetain`|11.3| | | |`hipUserObjectRetain`|5.3.0| | | | |

## **29. Driver Entry Point Access**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGetDriverEntryPoint`|11.3| | | |`hipGetProcAddress`|6.1.0| | | |6.1.0|

## **30. C++ API Routines**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGetKernel`|12.1| | | | | | | | | |

## **31. Interactions with the CUDA Driver API**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaGetFuncBySymbol`|11.0| | | | | | | | | |

## **32. Profiler Control**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaProfilerStart`| | | | |`hipProfilerStart`|1.6.0|3.0.0| | | |
|`cudaProfilerStop`| | | | |`hipProfilerStop`|1.6.0|3.0.0| | | |

## **33. Data types used by CUDA Runtime**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUDA_EGL_MAX_PLANES`|9.1| | | | | | | | | |
|`CUDA_IPC_HANDLE_SIZE`| | | | |`HIP_IPC_HANDLE_SIZE`|1.6.0| | | | |
|`CUeglStreamConnection_st`|9.1| | | | | | | | | |
|`CUevent_st`| | | | |`ihipEvent_t`|1.6.0| | | | |
|`CUexternalMemory_st`|10.0| | | | | | | | | |
|`CUexternalSemaphore_st`|10.0| | | | | | | | | |
|`CUgraphExec_st`|10.0| | | |`hipGraphExec`|4.3.0| | | | |
|`CUgraphNode_st`|10.0| | | |`hipGraphNode`|4.3.0| | | | |
|`CUgraph_st`|10.0| | | |`ihipGraph`|4.3.0| | | | |
|`CUkern_st`|12.1| | | | | | | | | |
|`CUstream_st`| | | | |`ihipStream_t`|1.5.0| | | | |
|`CUuuid_st`| | | | |`hipUUID_t`|5.2.0| | | | |
|`cudaAccessPolicyWindow`|11.0| | | |`hipAccessPolicyWindow`|5.2.0| | | | |
|`cudaAccessProperty`|11.0| | | |`hipAccessProperty`|5.2.0| | | | |
|`cudaAccessPropertyNormal`|11.0| | | |`hipAccessPropertyNormal`|5.2.0| | | | |
|`cudaAccessPropertyPersisting`|11.0| | | |`hipAccessPropertyPersisting`|5.2.0| | | | |
|`cudaAccessPropertyStreaming`|11.0| | | |`hipAccessPropertyStreaming`|5.2.0| | | | |
|`cudaAddressModeBorder`| | | | |`hipAddressModeBorder`|1.7.0| | | | |
|`cudaAddressModeClamp`| | | | |`hipAddressModeClamp`|1.7.0| | | | |
|`cudaAddressModeMirror`| | | | |`hipAddressModeMirror`|1.7.0| | | | |
|`cudaAddressModeWrap`| | | | |`hipAddressModeWrap`|1.7.0| | | | |
|`cudaArray`| | | | |`hipArray`|1.7.0| | | | |
|`cudaArrayColorAttachment`|10.0| | | | | | | | | |
|`cudaArrayCubemap`| | | | |`hipArrayCubemap`|1.7.0| | | | |
|`cudaArrayDefault`| | | | |`hipArrayDefault`|1.7.0| | | | |
|`cudaArrayDeferredMapping`|11.6| | | | | | | | | |
|`cudaArrayLayered`| | | | |`hipArrayLayered`|1.7.0| | | | |
|`cudaArrayMemoryRequirements`|11.6| | | | | | | | | |
|`cudaArraySparse`|11.1| | | | | | | | | |
|`cudaArraySparseProperties`|11.1| | | | | | | | | |
|`cudaArraySparsePropertiesSingleMipTail`|11.1| | | | | | | | | |
|`cudaArraySurfaceLoadStore`| | | | |`hipArraySurfaceLoadStore`|1.7.0| | | | |
|`cudaArrayTextureGather`| | | | |`hipArrayTextureGather`|1.7.0| | | | |
|`cudaArray_const_t`| | | | |`hipArray_const_t`|1.6.0| | | | |
|`cudaArray_t`| | | | |`hipArray_t`|1.7.0| | | | |
|`cudaBoundaryModeClamp`| | | | |`hipBoundaryModeClamp`|1.9.0| | | | |
|`cudaBoundaryModeTrap`| | | | |`hipBoundaryModeTrap`|1.9.0| | | | |
|`cudaBoundaryModeZero`| | | | |`hipBoundaryModeZero`|1.9.0| | | | |
|`cudaCGScope`|9.0| | | | | | | | | |
|`cudaCGScopeGrid`|9.0| | | | | | | | | |
|`cudaCGScopeInvalid`|9.0| | | | | | | | | |
|`cudaCGScopeMultiGrid`|9.0| | | | | | | | | |
|`cudaCSV`| | | |12.0| | | | | | |
|`cudaChannelFormatDesc`| | | | |`hipChannelFormatDesc`|1.6.0| | | | |
|`cudaChannelFormatKind`| | | | |`hipChannelFormatKind`|1.6.0| | | | |
|`cudaChannelFormatKindFloat`| | | | |`hipChannelFormatKindFloat`|1.6.0| | | | |
|`cudaChannelFormatKindNV12`|11.2| | | | | | | | | |
|`cudaChannelFormatKindNone`| | | | |`hipChannelFormatKindNone`|1.6.0| | | | |
|`cudaChannelFormatKindSigned`| | | | |`hipChannelFormatKindSigned`|1.6.0| | | | |
|`cudaChannelFormatKindSignedBlockCompressed4`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedBlockCompressed5`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedBlockCompressed6H`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized16X1`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized16X2`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized16X4`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized8X1`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized8X2`|11.5| | | | | | | | | |
|`cudaChannelFormatKindSignedNormalized8X4`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsigned`| | | | |`hipChannelFormatKindUnsigned`|1.6.0| | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed1`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed1SRGB`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed2`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed2SRGB`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed3`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed3SRGB`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed4`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed5`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed6H`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed7`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedBlockCompressed7SRGB`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized16X1`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized16X2`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized16X4`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized8X1`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized8X2`|11.5| | | | | | | | | |
|`cudaChannelFormatKindUnsignedNormalized8X4`|11.5| | | | | | | | | |
|`cudaChildGraphNodeParams`|12.2| | | |`hipChildGraphNodeParams`|6.1.0| | | |6.1.0|
|`cudaClusterSchedulingPolicy`|11.8| | | | | | | | | |
|`cudaClusterSchedulingPolicyDefault`|11.8| | | | | | | | | |
|`cudaClusterSchedulingPolicyLoadBalancing`|11.8| | | | | | | | | |
|`cudaClusterSchedulingPolicySpread`|11.8| | | | | | | | | |
|`cudaComputeMode`| | | | |`hipComputeMode`|1.9.0| | | | |
|`cudaComputeModeDefault`| | | | |`hipComputeModeDefault`|1.9.0| | | | |
|`cudaComputeModeExclusive`| | | | |`hipComputeModeExclusive`|1.9.0| | | | |
|`cudaComputeModeExclusiveProcess`| | | | |`hipComputeModeExclusiveProcess`|2.0.0| | | | |
|`cudaComputeModeProhibited`| | | | |`hipComputeModeProhibited`|1.9.0| | | | |
|`cudaConditionalNodeParams`|12.3| | | | | | | | | |
|`cudaCooperativeLaunchMultiDeviceNoPostSync`|9.0| | | |`hipCooperativeLaunchMultiDeviceNoPostSync`|3.2.0| | | | |
|`cudaCooperativeLaunchMultiDeviceNoPreSync`|9.0| | | |`hipCooperativeLaunchMultiDeviceNoPreSync`|3.2.0| | | | |
|`cudaCpuDeviceId`|8.0| | | |`hipCpuDeviceId`|3.7.0| | | | |
|`cudaD3D10DeviceList`| | | | | | | | | | |
|`cudaD3D10DeviceListAll`| | | | | | | | | | |
|`cudaD3D10DeviceListCurrentFrame`| | | | | | | | | | |
|`cudaD3D10DeviceListNextFrame`| | | | | | | | | | |
|`cudaD3D10MapFlags`| | | | | | | | | | |
|`cudaD3D10MapFlagsNone`| | | | | | | | | | |
|`cudaD3D10MapFlagsReadOnly`| | | | | | | | | | |
|`cudaD3D10MapFlagsWriteDiscard`| | | | | | | | | | |
|`cudaD3D10RegisterFlags`| | | | | | | | | | |
|`cudaD3D10RegisterFlagsArray`| | | | | | | | | | |
|`cudaD3D10RegisterFlagsNone`| | | | | | | | | | |
|`cudaD3D11DeviceList`| | | | | | | | | | |
|`cudaD3D11DeviceListAll`| | | | | | | | | | |
|`cudaD3D11DeviceListCurrentFrame`| | | | | | | | | | |
|`cudaD3D11DeviceListNextFrame`| | | | | | | | | | |
|`cudaD3D9DeviceList`| | | | | | | | | | |
|`cudaD3D9DeviceListAll`| | | | | | | | | | |
|`cudaD3D9DeviceListCurrentFrame`| | | | | | | | | | |
|`cudaD3D9DeviceListNextFrame`| | | | | | | | | | |
|`cudaD3D9MapFlags`| | | | | | | | | | |
|`cudaD3D9MapFlagsNone`| | | | | | | | | | |
|`cudaD3D9MapFlagsReadOnly`| | | | | | | | | | |
|`cudaD3D9MapFlagsWriteDiscard`| | | | | | | | | | |
|`cudaD3D9RegisterFlags`| | | | | | | | | | |
|`cudaD3D9RegisterFlagsArray`| | | | | | | | | | |
|`cudaD3D9RegisterFlagsNone`| | | | | | | | | | |
|`cudaDevAttrAsyncEngineCount`| | | | |`hipDeviceAttributeAsyncEngineCount`|4.3.0| | | | |
|`cudaDevAttrCanFlushRemoteWrites`|9.2| | | | | | | | | |
|`cudaDevAttrCanMapHostMemory`| | | | |`hipDeviceAttributeCanMapHostMemory`|2.10.0| | | | |
|`cudaDevAttrCanUseHostPointerForRegisteredMem`|8.0| | | |`hipDeviceAttributeCanUseHostPointerForRegisteredMem`|4.3.0| | | | |
|`cudaDevAttrClockRate`| | | | |`hipDeviceAttributeClockRate`|1.6.0| | | | |
|`cudaDevAttrClusterLaunch`|11.8| | | | | | | | | |
|`cudaDevAttrComputeCapabilityMajor`| | | | |`hipDeviceAttributeComputeCapabilityMajor`|1.6.0| | | | |
|`cudaDevAttrComputeCapabilityMinor`| | | | |`hipDeviceAttributeComputeCapabilityMinor`|1.6.0| | | | |
|`cudaDevAttrComputeMode`| | | | |`hipDeviceAttributeComputeMode`|1.6.0| | | | |
|`cudaDevAttrComputePreemptionSupported`|8.0| | | |`hipDeviceAttributeComputePreemptionSupported`|4.3.0| | | | |
|`cudaDevAttrConcurrentKernels`| | | | |`hipDeviceAttributeConcurrentKernels`|1.6.0| | | | |
|`cudaDevAttrConcurrentManagedAccess`|8.0| | | |`hipDeviceAttributeConcurrentManagedAccess`|3.10.0| | | | |
|`cudaDevAttrCooperativeLaunch`|9.0| | | |`hipDeviceAttributeCooperativeLaunch`|2.6.0| | | | |
|`cudaDevAttrCooperativeMultiDeviceLaunch`|9.0| | | |`hipDeviceAttributeCooperativeMultiDeviceLaunch`|2.6.0| | | | |
|`cudaDevAttrDeferredMappingCudaArraySupported`|11.6| | | | | | | | | |
|`cudaDevAttrDirectManagedMemAccessFromHost`|9.2| | | |`hipDeviceAttributeDirectManagedMemAccessFromHost`|3.10.0| | | | |
|`cudaDevAttrEccEnabled`| | | | |`hipDeviceAttributeEccEnabled`|2.10.0| | | | |
|`cudaDevAttrGPUDirectRDMAFlushWritesOptions`|11.3| | | | | | | | | |
|`cudaDevAttrGPUDirectRDMASupported`|11.3| | | | | | | | | |
|`cudaDevAttrGPUDirectRDMAWritesOrdering`|11.3| | | | | | | | | |
|`cudaDevAttrGlobalL1CacheSupported`| | | | |`hipDeviceAttributeGlobalL1CacheSupported`|4.3.0| | | | |
|`cudaDevAttrGlobalMemoryBusWidth`| | | | |`hipDeviceAttributeMemoryBusWidth`|1.6.0| | | | |
|`cudaDevAttrGpuOverlap`| | | | |`hipDeviceAttributeAsyncEngineCount`|4.3.0| | | | |
|`cudaDevAttrHostNativeAtomicSupported`|8.0| | | |`hipDeviceAttributeHostNativeAtomicSupported`|4.3.0| | | | |
|`cudaDevAttrHostNumaId`|12.2| | | | | | | | | |
|`cudaDevAttrHostRegisterReadOnlySupported`|11.1| | | | | | | | | |
|`cudaDevAttrHostRegisterSupported`|9.2| | | |`hipDeviceAttributeHostRegisterSupported`|6.0.0| | | | |
|`cudaDevAttrIntegrated`| | | | |`hipDeviceAttributeIntegrated`|1.9.0| | | | |
|`cudaDevAttrIpcEventSupport`|12.0| | | | | | | | | |
|`cudaDevAttrIsMultiGpuBoard`| | | | |`hipDeviceAttributeIsMultiGpuBoard`|1.6.0| | | | |
|`cudaDevAttrKernelExecTimeout`| | | | |`hipDeviceAttributeKernelExecTimeout`|2.10.0| | | | |
|`cudaDevAttrL2CacheSize`| | | | |`hipDeviceAttributeL2CacheSize`|1.6.0| | | | |
|`cudaDevAttrLocalL1CacheSupported`| | | | |`hipDeviceAttributeLocalL1CacheSupported`|4.3.0| | | | |
|`cudaDevAttrManagedMemory`| | | | |`hipDeviceAttributeManagedMemory`|3.10.0| | | | |
|`cudaDevAttrMax`|11.4| | | | | | | | | |
|`cudaDevAttrMaxAccessPolicyWindowSize`|11.3| | | | | | | | | |
|`cudaDevAttrMaxBlockDimX`| | | | |`hipDeviceAttributeMaxBlockDimX`|1.6.0| | | | |
|`cudaDevAttrMaxBlockDimY`| | | | |`hipDeviceAttributeMaxBlockDimY`|1.6.0| | | | |
|`cudaDevAttrMaxBlockDimZ`| | | | |`hipDeviceAttributeMaxBlockDimZ`|1.6.0| | | | |
|`cudaDevAttrMaxBlocksPerMultiprocessor`|11.0| | | |`hipDeviceAttributeMaxBlocksPerMultiprocessor`|4.3.0| | | | |
|`cudaDevAttrMaxGridDimX`| | | | |`hipDeviceAttributeMaxGridDimX`|1.6.0| | | | |
|`cudaDevAttrMaxGridDimY`| | | | |`hipDeviceAttributeMaxGridDimY`|1.6.0| | | | |
|`cudaDevAttrMaxGridDimZ`| | | | |`hipDeviceAttributeMaxGridDimZ`|1.6.0| | | | |
|`cudaDevAttrMaxPersistingL2CacheSize`|11.3| | | | | | | | | |
|`cudaDevAttrMaxPitch`| | | | |`hipDeviceAttributeMaxPitch`|2.10.0| | | | |
|`cudaDevAttrMaxRegistersPerBlock`| | | | |`hipDeviceAttributeMaxRegistersPerBlock`|1.6.0| | | | |
|`cudaDevAttrMaxRegistersPerMultiprocessor`| | | | |`hipDeviceAttributeMaxRegistersPerMultiprocessor`|4.3.0| | | | |
|`cudaDevAttrMaxSharedMemoryPerBlock`| | | | |`hipDeviceAttributeMaxSharedMemoryPerBlock`|1.6.0| | | | |
|`cudaDevAttrMaxSharedMemoryPerBlockOptin`|9.0| | | |`hipDeviceAttributeSharedMemPerBlockOptin`|4.3.0| | | | |
|`cudaDevAttrMaxSharedMemoryPerMultiprocessor`| | | | |`hipDeviceAttributeMaxSharedMemoryPerMultiprocessor`|1.6.0| | | | |
|`cudaDevAttrMaxSurface1DLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxSurface1DLayeredWidth`| | | | |`hipDeviceAttributeMaxSurface1DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxSurface1DWidth`| | | | |`hipDeviceAttributeMaxSurface1D`|4.3.0| | | | |
|`cudaDevAttrMaxSurface2DHeight`| | | | |`hipDeviceAttributeMaxSurface2D`|4.3.0| | | | |
|`cudaDevAttrMaxSurface2DLayeredHeight`| | | | |`hipDeviceAttributeMaxSurface2DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxSurface2DLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxSurface2DLayeredWidth`| | | | |`hipDeviceAttributeMaxSurface2DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxSurface2DWidth`| | | | |`hipDeviceAttributeMaxSurface2D`|4.3.0| | | | |
|`cudaDevAttrMaxSurface3DDepth`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`cudaDevAttrMaxSurface3DHeight`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`cudaDevAttrMaxSurface3DWidth`| | | | |`hipDeviceAttributeMaxSurface3D`|4.3.0| | | | |
|`cudaDevAttrMaxSurfaceCubemapLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxSurfaceCubemapLayeredWidth`| | | | |`hipDeviceAttributeMaxSurfaceCubemapLayered`|4.3.0| | | | |
|`cudaDevAttrMaxSurfaceCubemapWidth`| | | | |`hipDeviceAttributeMaxSurfaceCubemap`|4.3.0| | | | |
|`cudaDevAttrMaxTexture1DLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxTexture1DLayeredWidth`| | | | |`hipDeviceAttributeMaxTexture1DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxTexture1DLinearWidth`| | | | |`hipDeviceAttributeMaxTexture1DLinear`|4.3.0| | | | |
|`cudaDevAttrMaxTexture1DMipmappedWidth`| | | | |`hipDeviceAttributeMaxTexture1DMipmap`|4.3.0| | | | |
|`cudaDevAttrMaxTexture1DWidth`| | | | |`hipDeviceAttributeMaxTexture1DWidth`|2.7.0| | | | |
|`cudaDevAttrMaxTexture2DGatherHeight`| | | | |`hipDeviceAttributeMaxTexture2DGather`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DGatherWidth`| | | | |`hipDeviceAttributeMaxTexture2DGather`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DHeight`| | | | |`hipDeviceAttributeMaxTexture2DHeight`|2.7.0| | | | |
|`cudaDevAttrMaxTexture2DLayeredHeight`| | | | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxTexture2DLayeredWidth`| | | | |`hipDeviceAttributeMaxTexture2DLayered`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DLinearHeight`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DLinearPitch`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DLinearWidth`| | | | |`hipDeviceAttributeMaxTexture2DLinear`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DMipmappedHeight`| | | | |`hipDeviceAttributeMaxTexture2DMipmap`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DMipmappedWidth`| | | | |`hipDeviceAttributeMaxTexture2DMipmap`|4.3.0| | | | |
|`cudaDevAttrMaxTexture2DWidth`| | | | |`hipDeviceAttributeMaxTexture2DWidth`|2.7.0| | | | |
|`cudaDevAttrMaxTexture3DDepth`| | | | |`hipDeviceAttributeMaxTexture3DDepth`|2.7.0| | | | |
|`cudaDevAttrMaxTexture3DDepthAlt`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`cudaDevAttrMaxTexture3DHeight`| | | | |`hipDeviceAttributeMaxTexture3DHeight`|2.7.0| | | | |
|`cudaDevAttrMaxTexture3DHeightAlt`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`cudaDevAttrMaxTexture3DWidth`| | | | |`hipDeviceAttributeMaxTexture3DWidth`|2.7.0| | | | |
|`cudaDevAttrMaxTexture3DWidthAlt`| | | | |`hipDeviceAttributeMaxTexture3DAlt`|4.3.0| | | | |
|`cudaDevAttrMaxTextureCubemapLayeredLayers`| | | | | | | | | | |
|`cudaDevAttrMaxTextureCubemapLayeredWidth`| | | | |`hipDeviceAttributeMaxTextureCubemapLayered`|4.3.0| | | | |
|`cudaDevAttrMaxTextureCubemapWidth`| | | | |`hipDeviceAttributeMaxTextureCubemap`|4.3.0| | | | |
|`cudaDevAttrMaxThreadsPerBlock`| | | | |`hipDeviceAttributeMaxThreadsPerBlock`|1.6.0| | | | |
|`cudaDevAttrMaxThreadsPerMultiProcessor`| | | | |`hipDeviceAttributeMaxThreadsPerMultiProcessor`|1.6.0| | | | |
|`cudaDevAttrMaxTimelineSemaphoreInteropSupported`|11.2|11.5| | | | | | | | |
|`cudaDevAttrMemSyncDomainCount`|12.0| | | | | | | | | |
|`cudaDevAttrMemoryClockRate`| | | | |`hipDeviceAttributeMemoryClockRate`|1.6.0| | | | |
|`cudaDevAttrMemoryPoolSupportedHandleTypes`|11.3| | | | | | | | | |
|`cudaDevAttrMemoryPoolsSupported`|11.2| | | |`hipDeviceAttributeMemoryPoolsSupported`|5.2.0| | | | |
|`cudaDevAttrMpsEnabled`|12.3| | | | | | | | | |
|`cudaDevAttrMultiGpuBoardGroupID`| | | | |`hipDeviceAttributeMultiGpuBoardGroupID`|5.0.0| | | | |
|`cudaDevAttrMultiProcessorCount`| | | | |`hipDeviceAttributeMultiprocessorCount`|1.6.0| | | | |
|`cudaDevAttrNumaConfig`|12.2| | | | | | | | | |
|`cudaDevAttrNumaId`|12.2| | | | | | | | | |
|`cudaDevAttrPageableMemoryAccess`|8.0| | | |`hipDeviceAttributePageableMemoryAccess`|3.10.0| | | | |
|`cudaDevAttrPageableMemoryAccessUsesHostPageTables`|9.2| | | |`hipDeviceAttributePageableMemoryAccessUsesHostPageTables`|3.10.0| | | | |
|`cudaDevAttrPciBusId`| | | | |`hipDeviceAttributePciBusId`|1.6.0| | | | |
|`cudaDevAttrPciDeviceId`| | | | |`hipDeviceAttributePciDeviceId`|1.6.0| | | | |
|`cudaDevAttrPciDomainId`| | | | |`hipDeviceAttributePciDomainID`|4.3.0| | | | |
|`cudaDevAttrReserved122`|12.0| | | | | | | | | |
|`cudaDevAttrReserved123`|12.0| | | | | | | | | |
|`cudaDevAttrReserved124`|12.0| | | | | | | | | |
|`cudaDevAttrReserved127`|12.1| | | | | | | | | |
|`cudaDevAttrReserved128`|12.1| | | | | | | | | |
|`cudaDevAttrReserved129`|12.1| | | | | | | | | |
|`cudaDevAttrReserved132`|12.1| | | | | | | | | |
|`cudaDevAttrReserved92`|9.0| | | | | | | | | |
|`cudaDevAttrReserved93`|9.0| | | | | | | | | |
|`cudaDevAttrReserved94`|9.0| | | |`hipDeviceAttributeCanUseStreamWaitValue`|4.3.0| | | | |
|`cudaDevAttrReservedSharedMemoryPerBlock`|11.0| | | | | | | | | |
|`cudaDevAttrSingleToDoublePrecisionPerfRatio`|8.0| | | |`hipDeviceAttributeSingleToDoublePrecisionPerfRatio`|4.3.0| | | | |
|`cudaDevAttrSparseCudaArraySupported`|11.1| | | | | | | | | |
|`cudaDevAttrStreamPrioritiesSupported`| | | | |`hipDeviceAttributeStreamPrioritiesSupported`|4.3.0| | | | |
|`cudaDevAttrSurfaceAlignment`| | | | |`hipDeviceAttributeSurfaceAlignment`|4.3.0| | | | |
|`cudaDevAttrTccDriver`| | | | |`hipDeviceAttributeTccDriver`|4.3.0| | | | |
|`cudaDevAttrTextureAlignment`| | | | |`hipDeviceAttributeTextureAlignment`|2.10.0| | | | |
|`cudaDevAttrTexturePitchAlignment`| | | | |`hipDeviceAttributeTexturePitchAlignment`|3.2.0| | | | |
|`cudaDevAttrTimelineSemaphoreInteropSupported`|11.5| | | | | | | | | |
|`cudaDevAttrTotalConstantMemory`| | | | |`hipDeviceAttributeTotalConstantMemory`|1.6.0| | | | |
|`cudaDevAttrUnifiedAddressing`| | | | |`hipDeviceAttributeUnifiedAddressing`|4.3.0| | | | |
|`cudaDevAttrWarpSize`| | | | |`hipDeviceAttributeWarpSize`|1.6.0| | | | |
|`cudaDevP2PAttrAccessSupported`|8.0| | | |`hipDevP2PAttrAccessSupported`|3.8.0| | | | |
|`cudaDevP2PAttrCudaArrayAccessSupported`|9.2| | | |`hipDevP2PAttrHipArrayAccessSupported`|3.8.0| | | | |
|`cudaDevP2PAttrNativeAtomicSupported`|8.0| | | |`hipDevP2PAttrNativeAtomicSupported`|3.8.0| | | | |
|`cudaDevP2PAttrPerformanceRank`|8.0| | | |`hipDevP2PAttrPerformanceRank`|3.8.0| | | | |
|`cudaDeviceAttr`| | | | |`hipDeviceAttribute_t`|1.6.0| | | | |
|`cudaDeviceBlockingSync`| | | | |`hipDeviceScheduleBlockingSync`|1.6.0| | | | |
|`cudaDeviceLmemResizeToMax`| | | | |`hipDeviceLmemResizeToMax`|1.6.0| | | | |
|`cudaDeviceMapHost`| | | | |`hipDeviceMapHost`|1.6.0| | | | |
|`cudaDeviceMask`| | | | | | | | | | |
|`cudaDeviceNumaConfig`|12.2| | | | | | | | | |
|`cudaDeviceNumaConfigNone`|12.2| | | | | | | | | |
|`cudaDeviceNumaConfigNumaNode`|12.2| | | | | | | | | |
|`cudaDeviceP2PAttr`|8.0| | | |`hipDeviceP2PAttr`|3.8.0| | | | |
|`cudaDeviceProp`| | | | |`hipDeviceProp_t`|1.6.0| | | | |
|`cudaDevicePropDontCare`| | | |12.0| | | | | | |
|`cudaDeviceScheduleAuto`| | | | |`hipDeviceScheduleAuto`|1.6.0| | | | |
|`cudaDeviceScheduleBlockingSync`| | | | |`hipDeviceScheduleBlockingSync`|1.6.0| | | | |
|`cudaDeviceScheduleMask`| | | | |`hipDeviceScheduleMask`|1.6.0| | | | |
|`cudaDeviceScheduleSpin`| | | | |`hipDeviceScheduleSpin`|1.6.0| | | | |
|`cudaDeviceScheduleYield`| | | | |`hipDeviceScheduleYield`|1.6.0| | | | |
|`cudaDeviceSyncMemops`|12.1| | | | | | | | | |
|`cudaDriverEntryPointQueryResult`|12.0| | | | | | | | | |
|`cudaDriverEntryPointSuccess`|12.0| | | | | | | | | |
|`cudaDriverEntryPointSymbolNotFound`|12.0| | | | | | | | | |
|`cudaDriverEntryPointVersionNotSufficent`|12.0| | | | | | | | | |
|`cudaEglColorFormat`|9.1| | | | | | | | | |
|`cudaEglColorFormatA`|9.1| | | | | | | | | |
|`cudaEglColorFormatABGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatARGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatAYUV`|9.1| | | | | | | | | |
|`cudaEglColorFormatAYUV_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatBGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBGRA`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer10BGGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer10GBRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer10GRBG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer10RGGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer12BGGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer12GBRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer12GRBG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer12RGGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer14BGGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer14GBRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer14GRBG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer14RGGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer20BGGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer20GBRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer20GRBG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayer20RGGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayerBGGR`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayerGBRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayerGRBG`|9.1| | | | | | | | | |
|`cudaEglColorFormatBayerIspBGGR`|9.2| | | | | | | | | |
|`cudaEglColorFormatBayerIspGBRG`|9.2| | | | | | | | | |
|`cudaEglColorFormatBayerIspGRBG`|9.2| | | | | | | | | |
|`cudaEglColorFormatBayerIspRGGB`|9.2| | | | | | | | | |
|`cudaEglColorFormatBayerRGGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatL`|9.1| | | | | | | | | |
|`cudaEglColorFormatR`|9.1| | | | | | | | | |
|`cudaEglColorFormatRG`|9.1| | | | | | | | | |
|`cudaEglColorFormatRGB`|9.1| | | | | | | | | |
|`cudaEglColorFormatRGBA`|9.1| | | | | | | | | |
|`cudaEglColorFormatUYVY422`|9.1| | | | | | | | | |
|`cudaEglColorFormatUYVY_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatVYUY_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatY10V10U10_420SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatY10V10U10_444SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatY12V12U12_420SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatY12V12U12_444SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV420Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV420Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV420SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV420SemiPlanar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV422Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV422Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV422SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV422SemiPlanar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV444Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV444Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV444SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV444SemiPlanar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUVA_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUV_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUYV422`|9.1| | | | | | | | | |
|`cudaEglColorFormatYUYV_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU420Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU420Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU420SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU420SemiPlanar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU422Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU422Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU422SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU422SemiPlanar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU444Planar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU444Planar_ER`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU444SemiPlanar`|9.1| | | | | | | | | |
|`cudaEglColorFormatYVU444SemiPlanar_ER`| | | | | | | | | | |
|`cudaEglColorFormatYVYU_ER`|9.1| | | | | | | | | |
|`cudaEglFrame`|9.1| | | | | | | | | |
|`cudaEglFrameType`|9.1| | | | | | | | | |
|`cudaEglFrameTypeArray`|9.1| | | | | | | | | |
|`cudaEglFrameTypePitch`|9.1| | | | | | | | | |
|`cudaEglFrame_st`|9.1| | | | | | | | | |
|`cudaEglPlaneDesc`|9.1| | | | | | | | | |
|`cudaEglPlaneDesc_st`|9.1| | | | | | | | | |
|`cudaEglResourceLocationFlags`|9.1| | | | | | | | | |
|`cudaEglResourceLocationSysmem`|9.1| | | | | | | | | |
|`cudaEglResourceLocationVidmem`|9.1| | | | | | | | | |
|`cudaEglStreamConnection`|9.1| | | | | | | | | |
|`cudaEnableDefault`|11.3| | | | | | | | | |
|`cudaEnableLegacyStream`|11.3| | | | | | | | | |
|`cudaEnablePerThreadDefaultStream`|11.3| | | | | | | | | |
|`cudaError`| | | | |`hipError_t`|1.5.0| | | | |
|`cudaErrorAddressOfConstant`| |3.1| | | | | | | | |
|`cudaErrorAlreadyAcquired`|10.1| | | |`hipErrorAlreadyAcquired`|1.6.0| | | | |
|`cudaErrorAlreadyMapped`|10.1| | | |`hipErrorAlreadyMapped`|1.6.0| | | | |
|`cudaErrorApiFailureBase`| |4.1| | | | | | | | |
|`cudaErrorArrayIsMapped`|10.1| | | |`hipErrorArrayIsMapped`|1.6.0| | | | |
|`cudaErrorAssert`| | | | |`hipErrorAssert`|1.9.0| | | | |
|`cudaErrorCallRequiresNewerDriver`|11.1| | | | | | | | | |
|`cudaErrorCapturedEvent`|10.0| | | |`hipErrorCapturedEvent`|4.3.0| | | | |
|`cudaErrorCdpNotSupported`|12.0| | | | | | | | | |
|`cudaErrorCdpVersionMismatch`|12.0| | | | | | | | | |
|`cudaErrorCompatNotSupportedOnDevice`|10.1| | | | | | | | | |
|`cudaErrorContextIsDestroyed`|10.1| | | |`hipErrorContextIsDestroyed`|4.3.0| | | | |
|`cudaErrorCooperativeLaunchTooLarge`|9.0| | | |`hipErrorCooperativeLaunchTooLarge`|3.2.0| | | | |
|`cudaErrorCudartUnloading`| | | | |`hipErrorDeinitialized`|1.6.0| | | | |
|`cudaErrorDeviceAlreadyInUse`| | | | |`hipErrorContextAlreadyInUse`|1.6.0| | | | |
|`cudaErrorDeviceNotLicensed`|11.1| | | | | | | | | |
|`cudaErrorDeviceUninitialized`|10.2| | | |`hipErrorInvalidContext`|1.6.0| | | | |
|`cudaErrorDevicesUnavailable`| | | | | | | | | | |
|`cudaErrorDuplicateSurfaceName`| | | | | | | | | | |
|`cudaErrorDuplicateTextureName`| | | | | | | | | | |
|`cudaErrorDuplicateVariableName`| | | | | | | | | | |
|`cudaErrorECCUncorrectable`| | | | |`hipErrorECCNotCorrectable`|1.6.0| | | | |
|`cudaErrorExternalDevice`| | | | | | | | | | |
|`cudaErrorFileNotFound`|10.1| | | |`hipErrorFileNotFound`|1.6.0| | | | |
|`cudaErrorGraphExecUpdateFailure`|10.2| | | |`hipErrorGraphExecUpdateFailure`|5.0.0| | | | |
|`cudaErrorHardwareStackError`| | | | | | | | | | |
|`cudaErrorHostMemoryAlreadyRegistered`| | | | |`hipErrorHostMemoryAlreadyRegistered`|1.6.0| | | | |
|`cudaErrorHostMemoryNotRegistered`| | | | |`hipErrorHostMemoryNotRegistered`|1.6.0| | | | |
|`cudaErrorIllegalAddress`| | | | |`hipErrorIllegalAddress`|1.6.0| | | | |
|`cudaErrorIllegalInstruction`| | | | | | | | | | |
|`cudaErrorIllegalState`|10.0| | | |`hipErrorIllegalState`|5.0.0| | | | |
|`cudaErrorIncompatibleDriverContext`| | | | | | | | | | |
|`cudaErrorInitializationError`| | | | |`hipErrorNotInitialized`|1.6.0| | | | |
|`cudaErrorInsufficientDriver`| | | | |`hipErrorInsufficientDriver`|1.7.0| | | | |
|`cudaErrorInvalidAddressSpace`| | | | | | | | | | |
|`cudaErrorInvalidChannelDescriptor`| | | | | | | | | | |
|`cudaErrorInvalidClusterSize`|11.8| | | | | | | | | |
|`cudaErrorInvalidConfiguration`| | | | |`hipErrorInvalidConfiguration`|1.6.0| | | | |
|`cudaErrorInvalidDevice`| | | | |`hipErrorInvalidDevice`|1.6.0| | | | |
|`cudaErrorInvalidDeviceFunction`| | | | |`hipErrorInvalidDeviceFunction`|1.6.0| | | | |
|`cudaErrorInvalidDevicePointer`| |10.1| | |`hipErrorInvalidDevicePointer`|1.6.0| | | | |
|`cudaErrorInvalidFilterSetting`| | | | | | | | | | |
|`cudaErrorInvalidGraphicsContext`| | | | |`hipErrorInvalidGraphicsContext`|1.6.0| | | | |
|`cudaErrorInvalidHostPointer`| |10.1| | | | | | | | |
|`cudaErrorInvalidKernelImage`| | | | |`hipErrorInvalidImage`|1.6.0| | | | |
|`cudaErrorInvalidMemcpyDirection`| | | | |`hipErrorInvalidMemcpyDirection`|1.6.0| | | | |
|`cudaErrorInvalidNormSetting`| | | | | | | | | | |
|`cudaErrorInvalidPc`| | | | | | | | | | |
|`cudaErrorInvalidPitchValue`| | | | |`hipErrorInvalidPitchValue`|4.2.0| | | | |
|`cudaErrorInvalidPtx`| | | | |`hipErrorInvalidKernelFile`|1.6.0| | | | |
|`cudaErrorInvalidResourceHandle`| | | | |`hipErrorInvalidHandle`|1.6.0| | | | |
|`cudaErrorInvalidSource`|10.1| | | |`hipErrorInvalidSource`|1.6.0| | | | |
|`cudaErrorInvalidSurface`| | | | | | | | | | |
|`cudaErrorInvalidSymbol`| | | | |`hipErrorInvalidSymbol`|1.6.0| | | | |
|`cudaErrorInvalidTexture`| | | | | | | | | | |
|`cudaErrorInvalidTextureBinding`| | | | | | | | | | |
|`cudaErrorInvalidValue`| | | | |`hipErrorInvalidValue`|1.6.0| | | | |
|`cudaErrorJitCompilationDisabled`|11.2| | | | | | | | | |
|`cudaErrorJitCompilerNotFound`|9.0| | | | | | | | | |
|`cudaErrorLaunchFailure`| | | | |`hipErrorLaunchFailure`|1.6.0| | | | |
|`cudaErrorLaunchFileScopedSurf`| | | | | | | | | | |
|`cudaErrorLaunchFileScopedTex`| | | | | | | | | | |
|`cudaErrorLaunchIncompatibleTexturing`|10.1| | | | | | | | | |
|`cudaErrorLaunchMaxDepthExceeded`| | | | | | | | | | |
|`cudaErrorLaunchOutOfResources`| | | | |`hipErrorLaunchOutOfResources`|1.6.0| | | | |
|`cudaErrorLaunchPendingCountExceeded`| | | | | | | | | | |
|`cudaErrorLaunchTimeout`| | | | |`hipErrorLaunchTimeOut`|1.6.0| | | | |
|`cudaErrorLossyQuery`|12.3| | | |`hipErrorLossyQuery`| | | | | |
|`cudaErrorMapBufferObjectFailed`| | | | |`hipErrorMapFailed`|1.6.0| | | | |
|`cudaErrorMemoryAllocation`| | | | |`hipErrorOutOfMemory`|1.6.0| | | | |
|`cudaErrorMemoryValueTooLarge`| |3.1| | | | | | | | |
|`cudaErrorMisalignedAddress`| | | | | | | | | | |
|`cudaErrorMissingConfiguration`| | | | |`hipErrorMissingConfiguration`|1.6.0| | | | |
|`cudaErrorMixedDeviceExecution`| |3.1| | | | | | | | |
|`cudaErrorMpsClientTerminated`|11.8| | | | | | | | | |
|`cudaErrorMpsConnectionFailed`|11.4| | | | | | | | | |
|`cudaErrorMpsMaxClientsReached`|11.4| | | | | | | | | |
|`cudaErrorMpsMaxConnectionsReached`|11.4| | | | | | | | | |
|`cudaErrorMpsRpcFailure`|11.4| | | | | | | | | |
|`cudaErrorMpsServerNotReady`|11.4| | | | | | | | | |
|`cudaErrorNoDevice`| | | | |`hipErrorNoDevice`|1.6.0| | | | |
|`cudaErrorNoKernelImageForDevice`| | | | |`hipErrorNoBinaryForGpu`|1.6.0| | | | |
|`cudaErrorNotMapped`|10.1| | | |`hipErrorNotMapped`|1.6.0| | | | |
|`cudaErrorNotMappedAsArray`|10.1| | | |`hipErrorNotMappedAsArray`|1.6.0| | | | |
|`cudaErrorNotMappedAsPointer`|10.1| | | |`hipErrorNotMappedAsPointer`|1.6.0| | | | |
|`cudaErrorNotPermitted`| | | | | | | | | | |
|`cudaErrorNotReady`| | | | |`hipErrorNotReady`|1.6.0| | | | |
|`cudaErrorNotSupported`| | | | |`hipErrorNotSupported`|1.6.0| | | | |
|`cudaErrorNotYetImplemented`| |4.1| | | | | | | | |
|`cudaErrorNvlinkUncorrectable`|8.0| | | | | | | | | |
|`cudaErrorOperatingSystem`| | | | |`hipErrorOperatingSystem`|1.6.0| | | | |
|`cudaErrorPeerAccessAlreadyEnabled`| | | | |`hipErrorPeerAccessAlreadyEnabled`|1.6.0| | | | |
|`cudaErrorPeerAccessNotEnabled`| | | | |`hipErrorPeerAccessNotEnabled`|1.6.0| | | | |
|`cudaErrorPeerAccessUnsupported`| | | | |`hipErrorPeerAccessUnsupported`|1.6.0| | | | |
|`cudaErrorPriorLaunchFailure`| |3.1| | |`hipErrorPriorLaunchFailure`|1.6.0| | | | |
|`cudaErrorProfilerAlreadyStarted`| |5.0| | |`hipErrorProfilerAlreadyStarted`|1.6.0| | | | |
|`cudaErrorProfilerAlreadyStopped`| |5.0| | |`hipErrorProfilerAlreadyStopped`|1.6.0| | | | |
|`cudaErrorProfilerDisabled`| | | | |`hipErrorProfilerDisabled`|1.6.0| | | | |
|`cudaErrorProfilerNotInitialized`| |5.0| | |`hipErrorProfilerNotInitialized`|1.6.0| | | | |
|`cudaErrorSetOnActiveProcess`| | | | |`hipErrorSetOnActiveProcess`|1.6.0| | | | |
|`cudaErrorSharedObjectInitFailed`| | | | |`hipErrorSharedObjectInitFailed`|1.6.0| | | | |
|`cudaErrorSharedObjectSymbolNotFound`| | | | |`hipErrorSharedObjectSymbolNotFound`|1.6.0| | | | |
|`cudaErrorSoftwareValidityNotEstablished`|11.2| | | | | | | | | |
|`cudaErrorStartupFailure`| | | | | | | | | | |
|`cudaErrorStreamCaptureImplicit`|10.0| | | |`hipErrorStreamCaptureImplicit`|4.3.0| | | | |
|`cudaErrorStreamCaptureInvalidated`|10.0| | | |`hipErrorStreamCaptureInvalidated`|4.3.0| | | | |
|`cudaErrorStreamCaptureIsolation`|10.0| | | |`hipErrorStreamCaptureIsolation`|4.3.0| | | | |
|`cudaErrorStreamCaptureMerge`|10.0| | | |`hipErrorStreamCaptureMerge`|4.3.0| | | | |
|`cudaErrorStreamCaptureUnjoined`|10.0| | | |`hipErrorStreamCaptureUnjoined`|4.3.0| | | | |
|`cudaErrorStreamCaptureUnmatched`|10.0| | | |`hipErrorStreamCaptureUnmatched`|4.3.0| | | | |
|`cudaErrorStreamCaptureUnsupported`|10.0| | | |`hipErrorStreamCaptureUnsupported`|4.3.0| | | | |
|`cudaErrorStreamCaptureWrongThread`|10.1| | | |`hipErrorStreamCaptureWrongThread`|4.3.0| | | | |
|`cudaErrorStubLibrary`|11.1| | | | | | | | | |
|`cudaErrorSymbolNotFound`|10.1| | | |`hipErrorNotFound`|1.6.0| | | | |
|`cudaErrorSyncDepthExceeded`| | | | | | | | | | |
|`cudaErrorSynchronizationError`| |3.1| | | | | | | | |
|`cudaErrorSystemDriverMismatch`|10.1| | | | | | | | | |
|`cudaErrorSystemNotReady`|10.0| | | | | | | | | |
|`cudaErrorTextureFetchFailed`| |3.1| | | | | | | | |
|`cudaErrorTextureNotBound`| |3.1| | | | | | | | |
|`cudaErrorTimeout`|10.2| | | | | | | | | |
|`cudaErrorTooManyPeers`| | | | | | | | | | |
|`cudaErrorUnknown`| | | | |`hipErrorUnknown`|1.6.0| | | | |
|`cudaErrorUnmapBufferObjectFailed`| | | | |`hipErrorUnmapFailed`|1.6.0| | | | |
|`cudaErrorUnsupportedDevSideSync`|12.1| | | | | | | | | |
|`cudaErrorUnsupportedExecAffinity`|11.4| | | | | | | | | |
|`cudaErrorUnsupportedLimit`| | | | |`hipErrorUnsupportedLimit`|1.6.0| | | | |
|`cudaErrorUnsupportedPtxVersion`|11.1| | | | | | | | | |
|`cudaError_t`| | | | |`hipError_t`|1.5.0| | | | |
|`cudaEventBlockingSync`| | | | |`hipEventBlockingSync`|1.6.0| | | | |
|`cudaEventDefault`| | | | |`hipEventDefault`|1.6.0| | | | |
|`cudaEventDisableTiming`| | | | |`hipEventDisableTiming`|1.6.0| | | | |
|`cudaEventInterprocess`| | | | |`hipEventInterprocess`|1.6.0| | | | |
|`cudaEventRecordDefault`|11.1| | | | | | | | | |
|`cudaEventRecordExternal`|11.1| | | | | | | | | |
|`cudaEventRecordNodeParams`|12.2| | | |`hipEventRecordNodeParams`|6.1.0| | | |6.1.0|
|`cudaEventWaitDefault`|11.1| | | | | | | | | |
|`cudaEventWaitExternal`| | | | | | | | | | |
|`cudaEventWaitNodeParams`|12.2| | | |`hipEventWaitNodeParams`|6.1.0| | | |6.1.0|
|`cudaEvent_t`| | | | |`hipEvent_t`|1.6.0| | | | |
|`cudaExtent`| | | | |`hipExtent`|1.7.0| | | | |
|`cudaExternalMemoryBufferDesc`|10.0| | | |`hipExternalMemoryBufferDesc`|4.3.0| | | | |
|`cudaExternalMemoryDedicated`|10.0| | | |`hipExternalMemoryDedicated`|5.5.0| | | | |
|`cudaExternalMemoryHandleDesc`|10.0| | | |`hipExternalMemoryHandleDesc`|4.3.0| | | | |
|`cudaExternalMemoryHandleType`|10.0| | | |`hipExternalMemoryHandleType`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeD3D11Resource`|10.0| | | |`hipExternalMemoryHandleTypeD3D11Resource`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeD3D11ResourceKmt`|10.2| | | |`hipExternalMemoryHandleTypeD3D11ResourceKmt`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeD3D12Heap`|10.0| | | |`hipExternalMemoryHandleTypeD3D12Heap`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeD3D12Resource`|10.0| | | |`hipExternalMemoryHandleTypeD3D12Resource`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeNvSciBuf`|10.2| | | | | | | | | |
|`cudaExternalMemoryHandleTypeOpaqueFd`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueFd`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeOpaqueWin32`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueWin32`|4.3.0| | | | |
|`cudaExternalMemoryHandleTypeOpaqueWin32Kmt`|10.0| | | |`hipExternalMemoryHandleTypeOpaqueWin32Kmt`|4.3.0| | | | |
|`cudaExternalMemoryMipmappedArrayDesc`|10.0| | | | | | | | | |
|`cudaExternalMemory_t`|10.0| | | |`hipExternalMemory_t`|4.3.0| | | | |
|`cudaExternalSemaphoreHandleDesc`|10.0| | | |`hipExternalSemaphoreHandleDesc`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleType`|10.0| | | |`hipExternalSemaphoreHandleType`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleTypeD3D11Fence`|10.2| | | | | | | | | |
|`cudaExternalSemaphoreHandleTypeD3D12Fence`|10.0| | | |`hipExternalSemaphoreHandleTypeD3D12Fence`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleTypeKeyedMutex`|10.2| | | | | | | | | |
|`cudaExternalSemaphoreHandleTypeKeyedMutexKmt`|10.2| | | | | | | | | |
|`cudaExternalSemaphoreHandleTypeNvSciSync`|10.2| | | | | | | | | |
|`cudaExternalSemaphoreHandleTypeOpaqueFd`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueFd`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleTypeOpaqueWin32`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueWin32`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt`|10.0| | | |`hipExternalSemaphoreHandleTypeOpaqueWin32Kmt`|4.4.0| | | | |
|`cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd`|11.2| | | | | | | | | |
|`cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32`|11.2| | | | | | | | | |
|`cudaExternalSemaphoreSignalNodeParams`|11.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`cudaExternalSemaphoreSignalNodeParamsV2`|12.2| | | |`hipExternalSemaphoreSignalNodeParams`|6.0.0| | | | |
|`cudaExternalSemaphoreSignalParams`|10.0| | | |`hipExternalSemaphoreSignalParams`|4.4.0| | | | |
|`cudaExternalSemaphoreSignalParams_v1`|11.2| | | |`hipExternalSemaphoreSignalParams`|4.4.0| | | | |
|`cudaExternalSemaphoreSignalSkipNvSciBufMemSync`|10.2| | | | | | | | | |
|`cudaExternalSemaphoreWaitNodeParams`|11.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`cudaExternalSemaphoreWaitNodeParamsV2`|12.2| | | |`hipExternalSemaphoreWaitNodeParams`|6.0.0| | | | |
|`cudaExternalSemaphoreWaitParams`|10.0| | | |`hipExternalSemaphoreWaitParams`|4.4.0| | | | |
|`cudaExternalSemaphoreWaitParams_v1`|11.2| | | |`hipExternalSemaphoreWaitParams`|4.4.0| | | | |
|`cudaExternalSemaphoreWaitSkipNvSciBufMemSync`|10.2| | | | | | | | | |
|`cudaExternalSemaphore_t`|10.0| | | |`hipExternalSemaphore_t`|4.4.0| | | | |
|`cudaFilterModeLinear`| | | | |`hipFilterModeLinear`|1.7.0| | | | |
|`cudaFilterModePoint`| | | | |`hipFilterModePoint`|1.6.0| | | | |
|`cudaFlushGPUDirectRDMAWritesOptionHost`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptionHost`|6.1.0| | | |6.1.0|
|`cudaFlushGPUDirectRDMAWritesOptionMemOps`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptionMemOps`|6.1.0| | | |6.1.0|
|`cudaFlushGPUDirectRDMAWritesOptions`|11.3| | | |`hipFlushGPUDirectRDMAWritesOptions`|6.1.0| | | |6.1.0|
|`cudaFlushGPUDirectRDMAWritesScope`|11.3| | | | | | | | | |
|`cudaFlushGPUDirectRDMAWritesTarget`|11.3| | | | | | | | | |
|`cudaFlushGPUDirectRDMAWritesTargetCurrentDevice`|11.3| | | | | | | | | |
|`cudaFlushGPUDirectRDMAWritesToAllDevices`|11.3| | | | | | | | | |
|`cudaFlushGPUDirectRDMAWritesToOwner`|11.3| | | | | | | | | |
|`cudaFormatModeAuto`| | | | | | | | | | |
|`cudaFormatModeForced`| | | | | | | | | | |
|`cudaFuncAttribute`|9.0| | | |`hipFuncAttribute`|3.9.0| | | | |
|`cudaFuncAttributeClusterDimMustBeSet`|11.8| | | | | | | | | |
|`cudaFuncAttributeClusterSchedulingPolicyPreference`|11.8| | | | | | | | | |
|`cudaFuncAttributeMax`|9.0| | | |`hipFuncAttributeMax`|3.9.0| | | | |
|`cudaFuncAttributeMaxDynamicSharedMemorySize`|9.0| | | |`hipFuncAttributeMaxDynamicSharedMemorySize`|3.9.0| | | | |
|`cudaFuncAttributeNonPortableClusterSizeAllowed`|11.8| | | | | | | | | |
|`cudaFuncAttributePreferredSharedMemoryCarveout`|9.0| | | |`hipFuncAttributePreferredSharedMemoryCarveout`|3.9.0| | | | |
|`cudaFuncAttributeRequiredClusterDepth`|11.8| | | | | | | | | |
|`cudaFuncAttributeRequiredClusterHeight`|11.8| | | | | | | | | |
|`cudaFuncAttributeRequiredClusterWidth`|11.8| | | | | | | | | |
|`cudaFuncAttributes`| | | | |`hipFuncAttributes`|1.9.0| | | | |
|`cudaFuncCache`| | | | |`hipFuncCache_t`|1.6.0| | | | |
|`cudaFuncCachePreferEqual`| | | | |`hipFuncCachePreferEqual`|1.6.0| | | | |
|`cudaFuncCachePreferL1`| | | | |`hipFuncCachePreferL1`|1.6.0| | | | |
|`cudaFuncCachePreferNone`| | | | |`hipFuncCachePreferNone`|1.6.0| | | | |
|`cudaFuncCachePreferShared`| | | | |`hipFuncCachePreferShared`|1.6.0| | | | |
|`cudaFunction_t`|11.0| | | |`hipFunction_t`|1.6.0| | | | |
|`cudaGLDeviceList`| | | | |`hipGLDeviceList`|4.4.0| | | | |
|`cudaGLDeviceListAll`| | | | |`hipGLDeviceListAll`|4.4.0| | | | |
|`cudaGLDeviceListCurrentFrame`| | | | |`hipGLDeviceListCurrentFrame`|4.4.0| | | | |
|`cudaGLDeviceListNextFrame`| | | | |`hipGLDeviceListNextFrame`|4.4.0| | | | |
|`cudaGLMapFlags`| | | | | | | | | | |
|`cudaGLMapFlagsNone`| | | | | | | | | | |
|`cudaGLMapFlagsReadOnly`| | | | | | | | | | |
|`cudaGLMapFlagsWriteDiscard`| | | | | | | | | | |
|`cudaGPUDirectRDMAWritesOrdering`|11.3| | | |`hipGPUDirectRDMAWritesOrdering`|6.1.0| | | |6.1.0|
|`cudaGPUDirectRDMAWritesOrderingAllDevices`|11.3| | | |`hipGPUDirectRDMAWritesOrderingAllDevices`|6.1.0| | | |6.1.0|
|`cudaGPUDirectRDMAWritesOrderingNone`|11.3| | | |`hipGPUDirectRDMAWritesOrderingNone`|6.1.0| | | |6.1.0|
|`cudaGPUDirectRDMAWritesOrderingOwner`|11.3| | | |`hipGPUDirectRDMAWritesOrderingOwner`|6.1.0| | | |6.1.0|
|`cudaGetDriverEntryPointFlags`|11.3| | | | | | | | | |
|`cudaGraphCondAssignDefault`|12.3| | | | | | | | | |
|`cudaGraphCondTypeIf`|12.3| | | | | | | | | |
|`cudaGraphCondTypeWhile`|12.3| | | | | | | | | |
|`cudaGraphConditionalHandle`|12.3| | | | | | | | | |
|`cudaGraphConditionalHandleFlags`|12.3| | | | | | | | | |
|`cudaGraphConditionalNodeType`|12.3| | | | | | | | | |
|`cudaGraphDebugDotFlags`|11.3| | | |`hipGraphDebugDotFlags`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsConditionalNodeParams`|12.3| | | | | | | | | |
|`cudaGraphDebugDotFlagsEventNodeParams`|11.3| | | |`hipGraphDebugDotFlagsEventNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsExtSemasSignalNodeParams`|11.3| | | |`hipGraphDebugDotFlagsExtSemasSignalNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsExtSemasWaitNodeParams`|11.3| | | |`hipGraphDebugDotFlagsExtSemasWaitNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsHandles`|11.3| | | |`hipGraphDebugDotFlagsHandles`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsHostNodeParams`|11.3| | | |`hipGraphDebugDotFlagsHostNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsKernelNodeAttributes`|11.3| | | |`hipGraphDebugDotFlagsKernelNodeAttributes`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsKernelNodeParams`|11.3| | | |`hipGraphDebugDotFlagsKernelNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsMemcpyNodeParams`|11.3| | | |`hipGraphDebugDotFlagsMemcpyNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsMemsetNodeParams`|11.3| | | |`hipGraphDebugDotFlagsMemsetNodeParams`|5.5.0| | | | |
|`cudaGraphDebugDotFlagsVerbose`|11.3| | | |`hipGraphDebugDotFlagsVerbose`|5.5.0| | | | |
|`cudaGraphDependencyType`|12.3| | | | | | | | | |
|`cudaGraphDependencyTypeDefault`|12.3| | | | | | | | | |
|`cudaGraphDependencyTypeProgrammatic`|12.3| | | | | | | | | |
|`cudaGraphDependencyType_enum`|12.3| | | | | | | | | |
|`cudaGraphEdgeData`|12.3| | | | | | | | | |
|`cudaGraphEdgeData_st`|12.3| | | | | | | | | |
|`cudaGraphExecUpdateError`|10.2| | | |`hipGraphExecUpdateError`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorAttributesChanged`|11.6| | | | | | | | | |
|`cudaGraphExecUpdateErrorFunctionChanged`|10.2| | | |`hipGraphExecUpdateErrorFunctionChanged`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorNodeTypeChanged`|10.2| | | |`hipGraphExecUpdateErrorNodeTypeChanged`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorNotSupported`|10.2| | | |`hipGraphExecUpdateErrorNotSupported`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorParametersChanged`|10.2| | | |`hipGraphExecUpdateErrorParametersChanged`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorTopologyChanged`|10.2| | | |`hipGraphExecUpdateErrorTopologyChanged`|4.3.0| | | | |
|`cudaGraphExecUpdateErrorUnsupportedFunctionChange`|11.2| | | |`hipGraphExecUpdateErrorUnsupportedFunctionChange`|4.3.0| | | | |
|`cudaGraphExecUpdateResult`|10.2| | | |`hipGraphExecUpdateResult`|4.3.0| | | | |
|`cudaGraphExecUpdateResultInfo`|12.0| | | | | | | | | |
|`cudaGraphExecUpdateResultInfo_st`|12.0| | | | | | | | | |
|`cudaGraphExecUpdateSuccess`|10.2| | | |`hipGraphExecUpdateSuccess`|4.3.0| | | | |
|`cudaGraphExec_t`|10.0| | | |`hipGraphExec_t`|4.3.0| | | | |
|`cudaGraphInstantiateError`|12.0| | | |`hipGraphInstantiateError`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateFlagAutoFreeOnLaunch`|11.4| | | |`hipGraphInstantiateFlagAutoFreeOnLaunch`|5.2.0| | | | |
|`cudaGraphInstantiateFlagDeviceLaunch`|12.0| | | |`hipGraphInstantiateFlagDeviceLaunch`|5.6.0| | | | |
|`cudaGraphInstantiateFlagUpload`|12.0| | | |`hipGraphInstantiateFlagUpload`|5.6.0| | | | |
|`cudaGraphInstantiateFlagUseNodePriority`|11.7| | | |`hipGraphInstantiateFlagUseNodePriority`|5.6.0| | | | |
|`cudaGraphInstantiateFlags`|11.4| | | |`hipGraphInstantiateFlags`|5.2.0| | | | |
|`cudaGraphInstantiateInvalidStructure`|12.0| | | |`hipGraphInstantiateInvalidStructure`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateMultipleDevicesNotSupported`|12.0| | | |`hipGraphInstantiateMultipleDevicesNotSupported`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateNodeOperationNotSupported`|12.0| | | |`hipGraphInstantiateNodeOperationNotSupported`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateParams`|12.0| | | |`hipGraphInstantiateParams`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateParams_st`|12.0| | | |`hipGraphInstantiateParams`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateResult`|12.0| | | |`hipGraphInstantiateResult`|6.1.0| | | |6.1.0|
|`cudaGraphInstantiateSuccess`|12.0| | | |`hipGraphInstantiateSuccess`|6.1.0| | | |6.1.0|
|`cudaGraphKernelNodePortDefault`|12.3| | | | | | | | | |
|`cudaGraphKernelNodePortLaunchCompletion`|12.3| | | | | | | | | |
|`cudaGraphKernelNodePortProgrammatic`|12.3| | | | | | | | | |
|`cudaGraphMemAttrReservedMemCurrent`|11.4| | | |`hipGraphMemAttrReservedMemCurrent`|5.3.0| | | | |
|`cudaGraphMemAttrReservedMemHigh`|11.4| | | |`hipGraphMemAttrReservedMemHigh`|5.3.0| | | | |
|`cudaGraphMemAttrUsedMemCurrent`|11.4| | | |`hipGraphMemAttrUsedMemCurrent`|5.3.0| | | | |
|`cudaGraphMemAttrUsedMemHigh`|11.4| | | |`hipGraphMemAttrUsedMemHigh`|5.3.0| | | | |
|`cudaGraphMemAttributeType`|11.4| | | |`hipGraphMemAttributeType`|5.3.0| | | | |
|`cudaGraphNodeParams`|12.2| | | |`hipGraphNodeParams`|6.1.0| | | |6.1.0|
|`cudaGraphNodeType`|10.0| | | |`hipGraphNodeType`|4.3.0| | | | |
|`cudaGraphNodeTypeConditional`|12.3| | | |`hipGraphNodeTypeConditional`| | | | | |
|`cudaGraphNodeTypeCount`|10.0| | | |`hipGraphNodeTypeCount`|4.3.0| | | | |
|`cudaGraphNodeTypeEmpty`|10.0| | | |`hipGraphNodeTypeEmpty`|4.3.0| | | | |
|`cudaGraphNodeTypeEventRecord`|11.1| | | |`hipGraphNodeTypeEventRecord`|4.3.0| | | | |
|`cudaGraphNodeTypeExtSemaphoreSignal`|11.4| | | |`hipGraphNodeTypeExtSemaphoreSignal`|5.3.0| | | | |
|`cudaGraphNodeTypeExtSemaphoreWait`|11.4| | | |`hipGraphNodeTypeExtSemaphoreWait`|5.3.0| | | | |
|`cudaGraphNodeTypeGraph`|10.0| | | |`hipGraphNodeTypeGraph`|4.3.0| | | | |
|`cudaGraphNodeTypeHost`|10.0| | | |`hipGraphNodeTypeHost`|4.3.0| | | | |
|`cudaGraphNodeTypeKernel`|10.0| | | |`hipGraphNodeTypeKernel`|4.3.0| | | | |
|`cudaGraphNodeTypeMemAlloc`|11.4| | | |`hipGraphNodeTypeMemAlloc`|5.5.0| | | | |
|`cudaGraphNodeTypeMemFree`|11.4| | | |`hipGraphNodeTypeMemFree`|5.5.0| | | | |
|`cudaGraphNodeTypeMemcpy`|10.0| | | |`hipGraphNodeTypeMemcpy`|4.3.0| | | | |
|`cudaGraphNodeTypeMemset`|10.0| | | |`hipGraphNodeTypeMemset`|4.3.0| | | | |
|`cudaGraphNodeTypeWaitEvent`|11.1| | | |`hipGraphNodeTypeWaitEvent`|4.3.0| | | | |
|`cudaGraphNode_t`|10.0| | | |`hipGraphNode_t`|4.3.0| | | | |
|`cudaGraphUserObjectMove`|11.3| | | |`hipGraphUserObjectMove`|5.3.0| | | | |
|`cudaGraph_t`|10.0| | | |`hipGraph_t`|4.3.0| | | | |
|`cudaGraphicsCubeFace`| | | | | | | | | | |
|`cudaGraphicsCubeFaceNegativeX`| | | | | | | | | | |
|`cudaGraphicsCubeFaceNegativeY`| | | | | | | | | | |
|`cudaGraphicsCubeFaceNegativeZ`| | | | | | | | | | |
|`cudaGraphicsCubeFacePositiveX`| | | | | | | | | | |
|`cudaGraphicsCubeFacePositiveY`| | | | | | | | | | |
|`cudaGraphicsCubeFacePositiveZ`| | | | | | | | | | |
|`cudaGraphicsMapFlags`| | | | | | | | | | |
|`cudaGraphicsMapFlagsNone`| | | | | | | | | | |
|`cudaGraphicsMapFlagsReadOnly`| | | | | | | | | | |
|`cudaGraphicsMapFlagsWriteDiscard`| | | | | | | | | | |
|`cudaGraphicsRegisterFlags`| | | | |`hipGraphicsRegisterFlags`|4.4.0| | | | |
|`cudaGraphicsRegisterFlagsNone`| | | | |`hipGraphicsRegisterFlagsNone`|4.4.0| | | | |
|`cudaGraphicsRegisterFlagsReadOnly`| | | | |`hipGraphicsRegisterFlagsReadOnly`|4.4.0| | | | |
|`cudaGraphicsRegisterFlagsSurfaceLoadStore`| | | | |`hipGraphicsRegisterFlagsSurfaceLoadStore`|4.4.0| | | | |
|`cudaGraphicsRegisterFlagsTextureGather`| | | | |`hipGraphicsRegisterFlagsTextureGather`|4.4.0| | | | |
|`cudaGraphicsRegisterFlagsWriteDiscard`| | | | |`hipGraphicsRegisterFlagsWriteDiscard`|4.4.0| | | | |
|`cudaGraphicsResource`| | | | |`hipGraphicsResource`|4.4.0| | | | |
|`cudaGraphicsResource_t`| | | | |`hipGraphicsResource_t`|4.4.0| | | | |
|`cudaHostAllocDefault`| | | | |`hipHostMallocDefault`|1.6.0| | | | |
|`cudaHostAllocMapped`| | | | |`hipHostMallocMapped`|1.6.0| | | | |
|`cudaHostAllocPortable`| | | | |`hipHostMallocPortable`|1.6.0| | | | |
|`cudaHostAllocWriteCombined`| | | | |`hipHostMallocWriteCombined`|1.6.0| | | | |
|`cudaHostFn_t`|10.0| | | |`hipHostFn_t`|4.3.0| | | | |
|`cudaHostNodeParams`|10.0| | | |`hipHostNodeParams`|4.3.0| | | | |
|`cudaHostNodeParamsV2`|12.2| | | | | | | | | |
|`cudaHostRegisterDefault`| | | | |`hipHostRegisterDefault`|1.6.0| | | | |
|`cudaHostRegisterIoMemory`|7.5| | | |`hipHostRegisterIoMemory`|1.6.0| | | | |
|`cudaHostRegisterMapped`| | | | |`hipHostRegisterMapped`|1.6.0| | | | |
|`cudaHostRegisterPortable`| | | | |`hipHostRegisterPortable`|1.6.0| | | | |
|`cudaHostRegisterReadOnly`|11.1| | | |`hipHostRegisterReadOnly`|5.6.0| | | | |
|`cudaInitDeviceFlagsAreValid`|12.0| | | | | | | | | |
|`cudaInvalidDeviceId`|8.0| | | |`hipInvalidDeviceId`|3.7.0| | | | |
|`cudaIpcEventHandle_st`| | | | |`hipIpcEventHandle_st`|3.5.0| | | | |
|`cudaIpcEventHandle_t`| | | | |`hipIpcEventHandle_t`|1.6.0| | | | |
|`cudaIpcMemHandle_st`| | | | |`hipIpcMemHandle_st`|1.6.0| | | | |
|`cudaIpcMemHandle_t`| | | | |`hipIpcMemHandle_t`|1.6.0| | | | |
|`cudaIpcMemLazyEnablePeerAccess`| | | | |`hipIpcMemLazyEnablePeerAccess`|1.6.0| | | | |
|`cudaKernelNodeAttrID`|11.0| | | |`hipKernelNodeAttrID`|5.2.0| | | | |
|`cudaKernelNodeAttrValue`|11.0| | | |`hipKernelNodeAttrValue`|5.2.0| | | | |
|`cudaKernelNodeAttributeAccessPolicyWindow`|11.0| | | |`hipKernelNodeAttributeAccessPolicyWindow`|5.2.0| | | | |
|`cudaKernelNodeAttributeClusterDimension`|11.8| | | | | | | | | |
|`cudaKernelNodeAttributeClusterSchedulingPolicyPreference`|11.8| | | | | | | | | |
|`cudaKernelNodeAttributeCooperative`|11.0| | | |`hipKernelNodeAttributeCooperative`|5.2.0| | | | |
|`cudaKernelNodeAttributeMemSyncDomain`|12.0| | | | | | | | | |
|`cudaKernelNodeAttributeMemSyncDomainMap`|12.0| | | | | | | | | |
|`cudaKernelNodeAttributePriority`|11.7| | | | | | | | | |
|`cudaKernelNodeParams`|10.0| | | |`hipKernelNodeParams`|4.3.0| | | | |
|`cudaKernelNodeParamsV2`|12.2| | | | | | | | | |
|`cudaKernel_t`|12.1| | | | | | | | | |
|`cudaKeyValuePair`| | | |12.0| | | | | | |
|`cudaLaunchAttribute`|11.8| | | | | | | | | |
|`cudaLaunchAttributeAccessPolicyWindow`|11.8| | | | | | | | | |
|`cudaLaunchAttributeClusterDimension`|11.8| | | | | | | | | |
|`cudaLaunchAttributeClusterSchedulingPolicyPreference`|11.8| | | | | | | | | |
|`cudaLaunchAttributeCooperative`|11.8| | | | | | | | | |
|`cudaLaunchAttributeID`|11.8| | | | | | | | | |
|`cudaLaunchAttributeIgnore`|11.8| | | | | | | | | |
|`cudaLaunchAttributeLaunchCompletionEvent`|12.3| | | | | | | | | |
|`cudaLaunchAttributeMemSyncDomain`|12.0| | | | | | | | | |
|`cudaLaunchAttributeMemSyncDomainMap`|12.0| | | | | | | | | |
|`cudaLaunchAttributePriority`|11.8| | | | | | | | | |
|`cudaLaunchAttributeProgrammaticEvent`|11.8| | | | | | | | | |
|`cudaLaunchAttributeProgrammaticStreamSerialization`|11.8| | | | | | | | | |
|`cudaLaunchAttributeSynchronizationPolicy`|11.8| | | | | | | | | |
|`cudaLaunchAttributeValue`|11.8| | | | | | | | | |
|`cudaLaunchAttribute_st`|11.8| | | | | | | | | |
|`cudaLaunchConfig_st`|11.8| | | | | | | | | |
|`cudaLaunchConfig_t`|11.8| | | | | | | | | |
|`cudaLaunchMemSyncDomain`|12.0| | | | | | | | | |
|`cudaLaunchMemSyncDomainDefault`|12.0| | | | | | | | | |
|`cudaLaunchMemSyncDomainMap`|12.0| | | | | | | | | |
|`cudaLaunchMemSyncDomainMap_st`|12.0| | | | | | | | | |
|`cudaLaunchMemSyncDomainRemote`|12.0| | | | | | | | | |
|`cudaLaunchParams`|9.0| | | |`hipLaunchParams`|2.6.0| | | | |
|`cudaLimit`| | | | |`hipLimit_t`|1.6.0| | | | |
|`cudaLimitDevRuntimePendingLaunchCount`| | | | | | | | | | |
|`cudaLimitDevRuntimeSyncDepth`| | | | | | | | | | |
|`cudaLimitMallocHeapSize`| | | | |`hipLimitMallocHeapSize`|1.6.0| | | | |
|`cudaLimitMaxL2FetchGranularity`|10.0| | | | | | | | | |
|`cudaLimitPersistingL2CacheSize`|11.0| | | | | | | | | |
|`cudaLimitPrintfFifoSize`| | | | |`hipLimitPrintfFifoSize`|4.5.0| | | | |
|`cudaLimitStackSize`| | | | |`hipLimitStackSize`|5.3.0| | | | |
|`cudaMemAccessDesc`|11.2| | | |`hipMemAccessDesc`|5.2.0| | | | |
|`cudaMemAccessFlags`|11.2| | | |`hipMemAccessFlags`|5.2.0| | | | |
|`cudaMemAccessFlagsProtNone`|11.2| | | |`hipMemAccessFlagsProtNone`|5.2.0| | | | |
|`cudaMemAccessFlagsProtRead`|11.2| | | |`hipMemAccessFlagsProtRead`|5.2.0| | | | |
|`cudaMemAccessFlagsProtReadWrite`|11.2| | | |`hipMemAccessFlagsProtReadWrite`|5.2.0| | | | |
|`cudaMemAdviseSetAccessedBy`|8.0| | | |`hipMemAdviseSetAccessedBy`|3.7.0| | | | |
|`cudaMemAdviseSetPreferredLocation`|8.0| | | |`hipMemAdviseSetPreferredLocation`|3.7.0| | | | |
|`cudaMemAdviseSetReadMostly`|8.0| | | |`hipMemAdviseSetReadMostly`|3.7.0| | | | |
|`cudaMemAdviseUnsetAccessedBy`|8.0| | | |`hipMemAdviseUnsetAccessedBy`|3.7.0| | | | |
|`cudaMemAdviseUnsetPreferredLocation`|8.0| | | |`hipMemAdviseUnsetPreferredLocation`|3.7.0| | | | |
|`cudaMemAdviseUnsetReadMostly`|8.0| | | |`hipMemAdviseUnsetReadMostly`|3.7.0| | | | |
|`cudaMemAllocNodeParams`|11.4| | | |`hipMemAllocNodeParams`|5.5.0| | | | |
|`cudaMemAllocNodeParamsV2`|12.2| | | | | | | | | |
|`cudaMemAllocationHandleType`|11.2| | | |`hipMemAllocationHandleType`|5.2.0| | | | |
|`cudaMemAllocationType`|11.2| | | |`hipMemAllocationType`|5.2.0| | | | |
|`cudaMemAllocationTypeInvalid`|11.2| | | |`hipMemAllocationTypeInvalid`|5.2.0| | | | |
|`cudaMemAllocationTypeMax`|11.2| | | |`hipMemAllocationTypeMax`|5.2.0| | | | |
|`cudaMemAllocationTypePinned`|11.2| | | |`hipMemAllocationTypePinned`|5.2.0| | | | |
|`cudaMemAttachGlobal`| | | | |`hipMemAttachGlobal`|2.5.0| | | | |
|`cudaMemAttachHost`| | | | |`hipMemAttachHost`|2.5.0| | | | |
|`cudaMemAttachSingle`| | | | |`hipMemAttachSingle`|3.7.0| | | | |
|`cudaMemFabricHandle_st`|12.3| | | | | | | | | |
|`cudaMemFabricHandle_t`|12.3| | | | | | | | | |
|`cudaMemFreeNodeParams`|12.2| | | |`hipMemFreeNodeParams`|6.1.0| | | |6.1.0|
|`cudaMemHandleTypeNone`|11.2| | | |`hipMemHandleTypeNone`|5.2.0| | | | |
|`cudaMemHandleTypePosixFileDescriptor`|11.2| | | |`hipMemHandleTypePosixFileDescriptor`|5.2.0| | | | |
|`cudaMemHandleTypeWin32`|11.2| | | |`hipMemHandleTypeWin32`|5.2.0| | | | |
|`cudaMemHandleTypeWin32Kmt`|11.2| | | |`hipMemHandleTypeWin32Kmt`|5.2.0| | | | |
|`cudaMemLocation`|11.2| | | |`hipMemLocation`|5.2.0| | | | |
|`cudaMemLocationType`|11.2| | | |`hipMemLocationType`|5.2.0| | | | |
|`cudaMemLocationTypeDevice`|11.2| | | |`hipMemLocationTypeDevice`|5.2.0| | | | |
|`cudaMemLocationTypeHost`|12.2| | | | | | | | | |
|`cudaMemLocationTypeHostNuma`|12.2| | | | | | | | | |
|`cudaMemLocationTypeHostNumaCurrent`|12.2| | | | | | | | | |
|`cudaMemLocationTypeInvalid`|11.2| | | |`hipMemLocationTypeInvalid`|5.2.0| | | | |
|`cudaMemPoolAttr`|11.2| | | |`hipMemPoolAttr`|5.2.0| | | | |
|`cudaMemPoolAttrReleaseThreshold`|11.2| | | |`hipMemPoolAttrReleaseThreshold`|5.2.0| | | | |
|`cudaMemPoolAttrReservedMemCurrent`|11.3| | | |`hipMemPoolAttrReservedMemCurrent`|5.2.0| | | | |
|`cudaMemPoolAttrReservedMemHigh`|11.3| | | |`hipMemPoolAttrReservedMemHigh`|5.2.0| | | | |
|`cudaMemPoolAttrUsedMemCurrent`|11.3| | | |`hipMemPoolAttrUsedMemCurrent`|5.2.0| | | | |
|`cudaMemPoolAttrUsedMemHigh`|11.3| | | |`hipMemPoolAttrUsedMemHigh`|5.2.0| | | | |
|`cudaMemPoolProps`|11.2| | | |`hipMemPoolProps`|5.2.0| | | | |
|`cudaMemPoolPtrExportData`|11.2| | | |`hipMemPoolPtrExportData`|5.2.0| | | | |
|`cudaMemPoolReuseAllowInternalDependencies`|11.2| | | |`hipMemPoolReuseAllowInternalDependencies`|5.2.0| | | | |
|`cudaMemPoolReuseAllowOpportunistic`|11.2| | | |`hipMemPoolReuseAllowOpportunistic`|5.2.0| | | | |
|`cudaMemPoolReuseFollowEventDependencies`|11.2| | | |`hipMemPoolReuseFollowEventDependencies`|5.2.0| | | | |
|`cudaMemPool_t`|11.2| | | |`hipMemPool_t`|5.2.0| | | | |
|`cudaMemRangeAttribute`|8.0| | | |`hipMemRangeAttribute`|3.7.0| | | | |
|`cudaMemRangeAttributeAccessedBy`|8.0| | | |`hipMemRangeAttributeAccessedBy`|3.7.0| | | | |
|`cudaMemRangeAttributeLastPrefetchLocation`|8.0| | | |`hipMemRangeAttributeLastPrefetchLocation`|3.7.0| | | | |
|`cudaMemRangeAttributeLastPrefetchLocationId`|12.2| | | | | | | | | |
|`cudaMemRangeAttributeLastPrefetchLocationType`|12.2| | | | | | | | | |
|`cudaMemRangeAttributePreferredLocation`|8.0| | | |`hipMemRangeAttributePreferredLocation`|3.7.0| | | | |
|`cudaMemRangeAttributePreferredLocationId`|12.2| | | | | | | | | |
|`cudaMemRangeAttributePreferredLocationType`|12.2| | | | | | | | | |
|`cudaMemRangeAttributeReadMostly`|8.0| | | |`hipMemRangeAttributeReadMostly`|3.7.0| | | | |
|`cudaMemcpy3DParms`| | | | |`hipMemcpy3DParms`|1.7.0| | | | |
|`cudaMemcpy3DPeerParms`| | | | | | | | | | |
|`cudaMemcpyDefault`| | | | |`hipMemcpyDefault`|1.5.0| | | | |
|`cudaMemcpyDeviceToDevice`| | | | |`hipMemcpyDeviceToDevice`|1.5.0| | | | |
|`cudaMemcpyDeviceToHost`| | | | |`hipMemcpyDeviceToHost`|1.5.0| | | | |
|`cudaMemcpyHostToDevice`| | | | |`hipMemcpyHostToDevice`|1.5.0| | | | |
|`cudaMemcpyHostToHost`| | | | |`hipMemcpyHostToHost`|1.5.0| | | | |
|`cudaMemcpyKind`| | | | |`hipMemcpyKind`|1.5.0| | | | |
|`cudaMemcpyNodeParams`|12.2| | | |`hipMemcpyNodeParams`|6.1.0| | | |6.1.0|
|`cudaMemoryAdvise`|8.0| | | |`hipMemoryAdvise`|3.7.0| | | | |
|`cudaMemoryType`| | | | |`hipMemoryType`|1.6.0| | | | |
|`cudaMemoryTypeDevice`| | | | |`hipMemoryTypeDevice`|1.6.0| | | | |
|`cudaMemoryTypeHost`| | | | |`hipMemoryTypeHost`|1.6.0| | | | |
|`cudaMemoryTypeManaged`|10.0| | | |`hipMemoryTypeManaged`|5.3.0| | | | |
|`cudaMemoryTypeUnregistered`| | | | | | | | | | |
|`cudaMemsetParams`|10.0| | | |`hipMemsetParams`|4.3.0| | | | |
|`cudaMemsetParamsV2`|12.2| | | | | | | | | |
|`cudaMipmappedArray`| | | | |`hipMipmappedArray`|1.7.0| | | | |
|`cudaMipmappedArray_const_t`| | | | |`hipMipmappedArray_const_t`|1.6.0| | | | |
|`cudaMipmappedArray_t`| | | | |`hipMipmappedArray_t`|1.7.0| | | | |
|`cudaNvSciSyncAttrSignal`|10.2| | | | | | | | | |
|`cudaNvSciSyncAttrWait`|10.2| | | | | | | | | |
|`cudaOccupancyDefault`| | | | |`hipOccupancyDefault`|3.2.0| | | | |
|`cudaOccupancyDisableCachingOverride`| | | | |`hipOccupancyDisableCachingOverride`|5.5.0| | | | |
|`cudaOutputMode`| | | |12.0| | | | | | |
|`cudaOutputMode_t`| | | |12.0| | | | | | |
|`cudaPitchedPtr`| | | | |`hipPitchedPtr`|1.7.0| | | | |
|`cudaPointerAttributes`| | | | |`hipPointerAttribute_t`|1.6.0| | | | |
|`cudaPos`| | | | |`hipPos`|1.7.0| | | | |
|`cudaReadModeElementType`| | | | |`hipReadModeElementType`|1.6.0| | | | |
|`cudaReadModeNormalizedFloat`| | | | |`hipReadModeNormalizedFloat`|1.7.0| | | | |
|`cudaResViewFormatFloat1`| | | | |`hipResViewFormatFloat1`|1.7.0| | | | |
|`cudaResViewFormatFloat2`| | | | |`hipResViewFormatFloat2`|1.7.0| | | | |
|`cudaResViewFormatFloat4`| | | | |`hipResViewFormatFloat4`|1.7.0| | | | |
|`cudaResViewFormatHalf1`| | | | |`hipResViewFormatHalf1`|1.7.0| | | | |
|`cudaResViewFormatHalf2`| | | | |`hipResViewFormatHalf2`|1.7.0| | | | |
|`cudaResViewFormatHalf4`| | | | |`hipResViewFormatHalf4`|1.7.0| | | | |
|`cudaResViewFormatNone`| | | | |`hipResViewFormatNone`|1.7.0| | | | |
|`cudaResViewFormatSignedBlockCompressed4`| | | | |`hipResViewFormatSignedBlockCompressed4`|1.7.0| | | | |
|`cudaResViewFormatSignedBlockCompressed5`| | | | |`hipResViewFormatSignedBlockCompressed5`|1.7.0| | | | |
|`cudaResViewFormatSignedBlockCompressed6H`| | | | |`hipResViewFormatSignedBlockCompressed6H`|1.7.0| | | | |
|`cudaResViewFormatSignedChar1`| | | | |`hipResViewFormatSignedChar1`|1.7.0| | | | |
|`cudaResViewFormatSignedChar2`| | | | |`hipResViewFormatSignedChar2`|1.7.0| | | | |
|`cudaResViewFormatSignedChar4`| | | | |`hipResViewFormatSignedChar4`|1.7.0| | | | |
|`cudaResViewFormatSignedInt1`| | | | |`hipResViewFormatSignedInt1`|1.7.0| | | | |
|`cudaResViewFormatSignedInt2`| | | | |`hipResViewFormatSignedInt2`|1.7.0| | | | |
|`cudaResViewFormatSignedInt4`| | | | |`hipResViewFormatSignedInt4`|1.7.0| | | | |
|`cudaResViewFormatSignedShort1`| | | | |`hipResViewFormatSignedShort1`|1.7.0| | | | |
|`cudaResViewFormatSignedShort2`| | | | |`hipResViewFormatSignedShort2`|1.7.0| | | | |
|`cudaResViewFormatSignedShort4`| | | | |`hipResViewFormatSignedShort4`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed1`| | | | |`hipResViewFormatUnsignedBlockCompressed1`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed2`| | | | |`hipResViewFormatUnsignedBlockCompressed2`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed3`| | | | |`hipResViewFormatUnsignedBlockCompressed3`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed4`| | | | |`hipResViewFormatUnsignedBlockCompressed4`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed5`| | | | |`hipResViewFormatUnsignedBlockCompressed5`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed6H`| | | | |`hipResViewFormatUnsignedBlockCompressed6H`|1.7.0| | | | |
|`cudaResViewFormatUnsignedBlockCompressed7`| | | | |`hipResViewFormatUnsignedBlockCompressed7`|1.7.0| | | | |
|`cudaResViewFormatUnsignedChar1`| | | | |`hipResViewFormatUnsignedChar1`|1.7.0| | | | |
|`cudaResViewFormatUnsignedChar2`| | | | |`hipResViewFormatUnsignedChar2`|1.7.0| | | | |
|`cudaResViewFormatUnsignedChar4`| | | | |`hipResViewFormatUnsignedChar4`|1.7.0| | | | |
|`cudaResViewFormatUnsignedInt1`| | | | |`hipResViewFormatUnsignedInt1`|1.7.0| | | | |
|`cudaResViewFormatUnsignedInt2`| | | | |`hipResViewFormatUnsignedInt2`|1.7.0| | | | |
|`cudaResViewFormatUnsignedInt4`| | | | |`hipResViewFormatUnsignedInt4`|1.7.0| | | | |
|`cudaResViewFormatUnsignedShort1`| | | | |`hipResViewFormatUnsignedShort1`|1.7.0| | | | |
|`cudaResViewFormatUnsignedShort2`| | | | |`hipResViewFormatUnsignedShort2`|1.7.0| | | | |
|`cudaResViewFormatUnsignedShort4`| | | | |`hipResViewFormatUnsignedShort4`|1.7.0| | | | |
|`cudaResourceDesc`| | | | |`hipResourceDesc`|1.7.0| | | | |
|`cudaResourceType`| | | | |`hipResourceType`|1.7.0| | | | |
|`cudaResourceTypeArray`| | | | |`hipResourceTypeArray`|1.7.0| | | | |
|`cudaResourceTypeLinear`| | | | |`hipResourceTypeLinear`|1.7.0| | | | |
|`cudaResourceTypeMipmappedArray`| | | | |`hipResourceTypeMipmappedArray`|1.7.0| | | | |
|`cudaResourceTypePitch2D`| | | | |`hipResourceTypePitch2D`|1.7.0| | | | |
|`cudaResourceViewDesc`| | | | |`hipResourceViewDesc`|1.7.0| | | | |
|`cudaResourceViewFormat`| | | | |`hipResourceViewFormat`|1.7.0| | | | |
|`cudaSharedCarveout`|9.0| | | | | | | | | |
|`cudaSharedMemBankSizeDefault`| | | | |`hipSharedMemBankSizeDefault`|1.6.0| | | | |
|`cudaSharedMemBankSizeEightByte`| | | | |`hipSharedMemBankSizeEightByte`|1.6.0| | | | |
|`cudaSharedMemBankSizeFourByte`| | | | |`hipSharedMemBankSizeFourByte`|1.6.0| | | | |
|`cudaSharedMemConfig`| | | | |`hipSharedMemConfig`|1.6.0| | | | |
|`cudaSharedmemCarveoutDefault`|9.0| | | | | | | | | |
|`cudaSharedmemCarveoutMaxL1`|9.0| | | | | | | | | |
|`cudaSharedmemCarveoutMaxShared`|9.0| | | | | | | | | |
|`cudaStreamAddCaptureDependencies`|11.3| | | |`hipStreamAddCaptureDependencies`|5.0.0| | | | |
|`cudaStreamAttrID`|11.0| | | | | | | | | |
|`cudaStreamAttrValue`|11.0| | | | | | | | | |
|`cudaStreamAttributeAccessPolicyWindow`|11.0| | | | | | | | | |
|`cudaStreamAttributeMemSyncDomain`|12.0| | | | | | | | | |
|`cudaStreamAttributeMemSyncDomainMap`|12.0| | | | | | | | | |
|`cudaStreamAttributePriority`|12.0| | | | | | | | | |
|`cudaStreamAttributeSynchronizationPolicy`|11.0| | | | | | | | | |
|`cudaStreamCallback_t`| | | | |`hipStreamCallback_t`|1.6.0| | | | |
|`cudaStreamCaptureMode`|10.1| | | |`hipStreamCaptureMode`|4.3.0| | | | |
|`cudaStreamCaptureModeGlobal`|10.1| | | |`hipStreamCaptureModeGlobal`|4.3.0| | | | |
|`cudaStreamCaptureModeRelaxed`|10.1| | | |`hipStreamCaptureModeRelaxed`|4.3.0| | | | |
|`cudaStreamCaptureModeThreadLocal`|10.1| | | |`hipStreamCaptureModeThreadLocal`|4.3.0| | | | |
|`cudaStreamCaptureStatus`|10.0| | | |`hipStreamCaptureStatus`|4.3.0| | | | |
|`cudaStreamCaptureStatusActive`|10.0| | | |`hipStreamCaptureStatusActive`|4.3.0| | | | |
|`cudaStreamCaptureStatusInvalidated`|10.0| | | |`hipStreamCaptureStatusInvalidated`|4.3.0| | | | |
|`cudaStreamCaptureStatusNone`|10.0| | | |`hipStreamCaptureStatusNone`|4.3.0| | | | |
|`cudaStreamDefault`| | | | |`hipStreamDefault`|1.6.0| | | | |
|`cudaStreamLegacy`| | | | | | | | | | |
|`cudaStreamNonBlocking`| | | | |`hipStreamNonBlocking`|1.6.0| | | | |
|`cudaStreamPerThread`| | | | |`hipStreamPerThread`|4.5.0| | | | |
|`cudaStreamSetCaptureDependencies`|11.3| | | |`hipStreamSetCaptureDependencies`|5.0.0| | | | |
|`cudaStreamUpdateCaptureDependenciesFlags`|11.3| | | |`hipStreamUpdateCaptureDependenciesFlags`|5.0.0| | | | |
|`cudaStream_t`| | | | |`hipStream_t`|1.5.0| | | | |
|`cudaSuccess`| | | | |`hipSuccess`|1.5.0| | | | |
|`cudaSurfaceBoundaryMode`| | | | |`hipSurfaceBoundaryMode`|1.9.0| | | | |
|`cudaSurfaceFormatMode`| | | | | | | | | | |
|`cudaSurfaceObject_t`| | | | |`hipSurfaceObject_t`|1.9.0| | | | |
|`cudaSyncPolicyAuto`|11.0| | | | | | | | | |
|`cudaSyncPolicyBlockingSync`|11.0| | | | | | | | | |
|`cudaSyncPolicySpin`|11.0| | | | | | | | | |
|`cudaSyncPolicyYield`|11.0| | | | | | | | | |
|`cudaSynchronizationPolicy`|11.0| | | | | | | | | |
|`cudaTextureAddressMode`| | | | |`hipTextureAddressMode`|1.7.0| | | | |
|`cudaTextureDesc`| | | | |`hipTextureDesc`|1.7.0| | | | |
|`cudaTextureFilterMode`| | | | |`hipTextureFilterMode`|1.6.0| | | | |
|`cudaTextureObject_t`| | | | |`hipTextureObject_t`|1.7.0| | | | |
|`cudaTextureReadMode`| | | | |`hipTextureReadMode`|1.6.0| | | | |
|`cudaTextureType1D`| | | | |`hipTextureType1D`|1.6.0| | | | |
|`cudaTextureType1DLayered`| | | | |`hipTextureType1DLayered`|1.7.0| | | | |
|`cudaTextureType2D`| | | | |`hipTextureType2D`|1.7.0| | | | |
|`cudaTextureType2DLayered`| | | | |`hipTextureType2DLayered`|1.7.0| | | | |
|`cudaTextureType3D`| | | | |`hipTextureType3D`|1.7.0| | | | |
|`cudaTextureTypeCubemap`| | | | |`hipTextureTypeCubemap`|1.7.0| | | | |
|`cudaTextureTypeCubemapLayered`| | | | |`hipTextureTypeCubemapLayered`|1.7.0| | | | |
|`cudaUUID_t`| | | | |`hipUUID`|5.2.0| | | | |
|`cudaUserObjectFlags`|11.3| | | |`hipUserObjectFlags`|5.3.0| | | | |
|`cudaUserObjectNoDestructorSync`|11.3| | | |`hipUserObjectNoDestructorSync`|5.3.0| | | | |
|`cudaUserObjectRetainFlags`|11.3| | | |`hipUserObjectRetainFlags`|5.3.0| | | | |
|`cudaUserObject_t`|11.3| | | |`hipUserObject_t`|5.3.0| | | | |
|`libraryPropertyType`|8.0| | | | | | | | | |
|`libraryPropertyType_t`|8.0| | | | | | | | | |
|`surfaceReference`| | | |12.0|`surfaceReference`|1.9.0| | | | |
|`texture`| | | |12.0|`texture`| | | | | |
|`textureReference`| | | | |`textureReference`|1.6.0| | | | |

## **34. Execution Control [REMOVED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaConfigureCall`| | | |10.1|`hipConfigureCall`|1.9.0| | | | |
|`cudaLaunch`| | | |10.1|`hipLaunchByPtr`|1.9.0| | | | |
|`cudaSetupArgument`| | | |10.1|`hipSetupArgument`|1.9.0| | | | |

## **35. Texture Reference Management [REMOVED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaBindTexture`| |11.0| |12.0|`hipBindTexture`|1.6.0|3.8.0| | | |
|`cudaBindTexture2D`| |11.0| |12.0|`hipBindTexture2D`|1.7.0|3.8.0| | | |
|`cudaBindTextureToArray`| |11.0| |12.0|`hipBindTextureToArray`|1.6.0|3.8.0| | | |
|`cudaBindTextureToMipmappedArray`| |11.0| |12.0|`hipBindTextureToMipmappedArray`|1.7.0|5.7.0| | | |
|`cudaGetTextureAlignmentOffset`| |11.0| |12.0|`hipGetTextureAlignmentOffset`|1.9.0|3.8.0| | | |
|`cudaGetTextureReference`| |11.0| |12.0|`hipGetTextureReference`|1.7.0|5.3.0| | | |
|`cudaUnbindTexture`| |11.0| |12.0|`hipUnbindTexture`|1.6.0|3.8.0| | | |

## **36. Surface Reference Management [REMOVED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaBindSurfaceToArray`| |11.0| |12.0| | | | | | |
|`cudaGetSurfaceReference`| |11.0| |12.0| | | | | | |

## **37. Profiler Control [REMOVED]**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cudaProfilerInitialize`| |11.0| |12.0| | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental