/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

using SEC = runtime::CUDA_RUNTIME_API_SECTIONS;

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP = [] {
  std::map<llvm::StringRef,  hipCounter> m;

  // 1. Device Management
  // no analogue
  m["cudaChooseDevice"]                                        = {"hipChooseDevice",                                        "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuFlushGPUDirectRDMAWrites
  m["cudaDeviceFlushGPUDirectRDMAWrites"]                      = {"hipDeviceFlushGPUDirectRDMAWrites",                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE, HIP_UNSUPPORTED};
  // cuDeviceGetAttribute
  m["cudaDeviceGetAttribute"]                                  = {"hipDeviceGetAttribute",                                  "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetByPCIBusId
  m["cudaDeviceGetByPCIBusId"]                                 = {"hipDeviceGetByPCIBusId",                                 "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  m["cudaDeviceGetCacheConfig"]                                = {"hipDeviceGetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxGetLimit
  m["cudaDeviceGetLimit"]                                      = {"hipDeviceGetLimit",                                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetNvSciSyncAttributes
  m["cudaDeviceGetNvSciSyncAttributes"]                        = {"hipDeviceGetNvSciSyncAttributes",                        "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE, HIP_UNSUPPORTED};
  // cuDeviceGetP2PAttribute
  m["cudaDeviceGetP2PAttribute"]                               = {"hipDeviceGetP2PAttribute",                               "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetPCIBusId
  m["cudaDeviceGetPCIBusId"]                                   = {"hipDeviceGetPCIBusId",                                   "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxGetStreamPriorityRange
  m["cudaDeviceGetStreamPriorityRange"]                        = {"hipDeviceGetStreamPriorityRange",                        "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  m["cudaDeviceReset"]                                         = {"hipDeviceReset",                                         "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  m["cudaDeviceSetCacheConfig"]                                = {"hipDeviceSetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxSetLimit
  m["cudaDeviceSetLimit"]                                      = {"hipDeviceSetLimit",                                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxSynchronize
  m["cudaDeviceSynchronize"]                                   = {"hipDeviceSynchronize",                                   "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGet
  // NOTE: cuDeviceGet has no attr: int ordinal
  m["cudaGetDevice"]                                           = {"hipGetDevice",                                           "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetCount
  m["cudaGetDeviceCount"]                                      = {"hipGetDeviceCount",                                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxGetFlags
  m["cudaGetDeviceFlags"]                                      = {"hipGetDeviceFlags",                                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  // NOTE: Not equal to cuDeviceGetProperties due to different attributes: CUdevprop and cudaDeviceProp
  m["cudaGetDeviceProperties"]                                 = {"hipGetDeviceProperties",                                 "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuIpcCloseMemHandle
  m["cudaIpcCloseMemHandle"]                                   = {"hipIpcCloseMemHandle",                                   "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuIpcGetEventHandle
  m["cudaIpcGetEventHandle"]                                   = {"hipIpcGetEventHandle",                                   "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuIpcGetMemHandle
  m["cudaIpcGetMemHandle"]                                     = {"hipIpcGetMemHandle",                                     "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuIpcOpenEventHandle
  m["cudaIpcOpenEventHandle"]                                  = {"hipIpcOpenEventHandle",                                  "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuIpcOpenMemHandle
  m["cudaIpcOpenMemHandle"]                                    = {"hipIpcOpenMemHandle",                                    "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  m["cudaSetDevice"]                                           = {"hipSetDevice",                                           "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuCtxGetFlags
  m["cudaSetDeviceFlags"]                                      = {"hipSetDeviceFlags",                                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // no analogue
  m["cudaSetValidDevices"]                                     = {"hipSetValidDevices",                                     "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // NOTE: incompatible with cuDeviceGetTexture1DLinearMaxWidth
  m["cudaDeviceGetTexture1DLinearMaxWidth"]                    = {"hipDeviceGetTexture1DLinearMaxWidth",                    "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetDefaultMemPool
  m["cudaDeviceGetDefaultMemPool"]                             = {"hipDeviceGetDefaultMemPool",                             "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceSetMemPool
  m["cudaDeviceSetMemPool"]                                    = {"hipDeviceSetMemPool",                                    "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  // cuDeviceGetMemPool
  m["cudaDeviceGetMemPool"]                                    = {"hipDeviceGetMemPool",                                    "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE};
  //
  m["cudaInitDevice"]                                          = {"hipInitDevice",                                          "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE, HIP_UNSUPPORTED};
  // cuDeviceGetHostAtomicCapabilities
  m["cudaDeviceGetHostAtomicCapabilities"]                     = {"hipDeviceGetHostAtomicCapabilities",                     "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE, HIP_UNSUPPORTED};
  // cuDeviceGetP2PAtomicCapabilities
  m["cudaDeviceGetP2PAtomicCapabilities"]                      = {"hipDeviceGetP2PAtomicCapabilities",                      "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE, HIP_UNSUPPORTED};

  // 2. Device Management [DEPRECATED]
  // cuCtxGetSharedMemConfig -> hipCtxGetSharedMemConfig
  m["cudaDeviceGetSharedMemConfig"]                            = {"hipDeviceGetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_DEPRECATED, CUDA_DEPRECATED};
  // cuCtxSetSharedMemConfig -> hipCtxSetSharedMemConfig
  m["cudaDeviceSetSharedMemConfig"]                            = {"hipDeviceSetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME, SEC::DEVICE_DEPRECATED, CUDA_DEPRECATED};

  // 3. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  m["cudaGetErrorName"]                                        = {"hipGetErrorName",                                        "", CONV_ERROR, API_RUNTIME, SEC::ERROR};
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  m["cudaGetErrorString"]                                      = {"hipGetErrorString",                                      "", CONV_ERROR, API_RUNTIME, SEC::ERROR};
  // no analogue
  m["cudaGetLastError"]                                        = {"hipGetLastError",                                        "", CONV_ERROR, API_RUNTIME, SEC::ERROR};
  // no analogue
  m["cudaPeekAtLastError"]                                     = {"hipPeekAtLastError",                                     "", CONV_ERROR, API_RUNTIME, SEC::ERROR};

  // 4. Stream Management
  // cuStreamAddCallback
  m["cudaStreamAddCallback"]                                   = {"hipStreamAddCallback",                                   "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuCtxResetPersistingL2Cache
  m["cudaCtxResetPersistingL2Cache"]                           = {"hipCtxResetPersistingL2Cache",                           "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_UNSUPPORTED};
  // cuStreamAttachMemAsync
  m["cudaStreamAttachMemAsync"]                                = {"hipStreamAttachMemAsync",                                "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamBeginCapture
  m["cudaStreamBeginCapture"]                                  = {"hipStreamBeginCapture",                                  "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamBeginCaptureToGraph
  m["cudaStreamBeginCaptureToGraph"]                           = {"hipStreamBeginCaptureToGraph",                           "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamCopyAttributes
  m["cudaStreamCopyAttributes"]                                = {"hipStreamCopyAttributes",                                "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // no analogue
  // NOTE: Not equal to cuStreamCreate due to different signatures
  m["cudaStreamCreate"]                                        = {"hipStreamCreate",                                        "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamCreate
  m["cudaStreamCreateWithFlags"]                               = {"hipStreamCreateWithFlags",                               "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamCreateWithPriority
  m["cudaStreamCreateWithPriority"]                            = {"hipStreamCreateWithPriority",                            "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamDestroy
  m["cudaStreamDestroy"]                                       = {"hipStreamDestroy",                                       "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamEndCapture
  m["cudaStreamEndCapture"]                                    = {"hipStreamEndCapture",                                    "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetAttribute
  m["cudaStreamGetAttribute"]                                  = {"hipStreamGetAttribute",                                  "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamSetAttribute
  m["cudaStreamSetAttribute"]                                  = {"hipStreamSetAttribute",                                  "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetFlags
  m["cudaStreamGetFlags"]                                      = {"hipStreamGetFlags",                                      "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetPriority
  m["cudaStreamGetPriority"]                                   = {"hipStreamGetPriority",                                   "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamIsCapturing
  m["cudaStreamIsCapturing"]                                   = {"hipStreamIsCapturing",                                   "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetCaptureInfo
  m["cudaStreamGetCaptureInfo"]                                = {"hipStreamGetCaptureInfo",                                "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_PARTIALLY_SUPPORTED};
  // cuStreamGetCaptureInfo_v3
  m["cudaStreamGetCaptureInfo_v3"]                             = {"hipStreamGetCaptureInfo_v3",                             "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_UNSUPPORTED};
  // cuStreamUpdateCaptureDependencies
  m["cudaStreamUpdateCaptureDependencies"]                     = {"hipStreamUpdateCaptureDependencies",                     "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_PARTIALLY_SUPPORTED};
  // cuStreamUpdateCaptureDependencies_v2
  m["cudaStreamUpdateCaptureDependencies_v2"]                  = {"hipStreamUpdateCaptureDependencies_v2",                  "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_UNSUPPORTED};
  // cuStreamQuery
  m["cudaStreamQuery"]                                         = {"hipStreamQuery",                                         "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamSynchronize
  m["cudaStreamSynchronize"]                                   = {"hipStreamSynchronize",                                   "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamWaitEvent
  m["cudaStreamWaitEvent"]                                     = {"hipStreamWaitEvent",                                     "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuThreadExchangeStreamCaptureMode
  m["cudaThreadExchangeStreamCaptureMode"]                     = {"hipThreadExchangeStreamCaptureMode",                     "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetId
  m["cudaStreamGetId"]                                         = {"hipStreamGetId",                                         "", CONV_STREAM, API_RUNTIME, SEC::STREAM};
  // cuStreamGetDevice
  m["cudaStreamGetDevice"]                                     = {"hipStreamGetDevice",                                     "", CONV_STREAM, API_RUNTIME, SEC::STREAM, HIP_UNSUPPORTED};

  // 5. Event Management
  // no analogue
  // NOTE: Not equal to cuEventCreate due to different signatures
  m["cudaEventCreate"]                                         = {"hipEventCreate",                                         "", CONV_EVENT, API_RUNTIME, SEC::EVENT, CUDA_OVERLOADED};
  // cuEventCreate
  m["cudaEventCreateWithFlags"]                                = {"hipEventCreateWithFlags",                                "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // cuEventDestroy
  m["cudaEventDestroy"]                                        = {"hipEventDestroy",                                        "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // cuEventElapsedTime
  m["cudaEventElapsedTime"]                                    = {"hipEventElapsedTime",                                    "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  //
  m["cudaEventElapsedTime_v2"]                                 = {"hipEventElapsedTime_v2",                                 "", CONV_EVENT, API_RUNTIME, SEC::EVENT, HIP_UNSUPPORTED};
  // cuEventQuery
  m["cudaEventQuery"]                                          = {"hipEventQuery",                                          "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // cuEventRecord
  m["cudaEventRecord"]                                         = {"hipEventRecord",                                         "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // cuEventSynchronize
  m["cudaEventSynchronize"]                                    = {"hipEventSynchronize",                                    "", CONV_EVENT, API_RUNTIME, SEC::EVENT};
  // cuEventRecordWithFlags
  m["cudaEventRecordWithFlags"]                                = {"hipEventRecordWithFlags",                                "", CONV_EVENT, API_RUNTIME, SEC::EVENT};

  // 6. External Resource Interoperability
  // cuDestroyExternalMemory
  m["cudaDestroyExternalMemory"]                               = {"hipDestroyExternalMemory",                               "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuDestroyExternalSemaphore
  m["cudaDestroyExternalSemaphore"]                            = {"hipDestroyExternalSemaphore",                            "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuExternalMemoryGetMappedBuffer
  m["cudaExternalMemoryGetMappedBuffer"]                       = {"hipExternalMemoryGetMappedBuffer",                       "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuExternalMemoryGetMappedMipmappedArray
  m["cudaExternalMemoryGetMappedMipmappedArray"]               = {"hipExternalMemoryGetMappedMipmappedArray",               "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES, HIP_UNSUPPORTED};
  // cuImportExternalMemory
  m["cudaImportExternalMemory"]                                = {"hipImportExternalMemory",                                "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuImportExternalSemaphore
  m["cudaImportExternalSemaphore"]                             = {"hipImportExternalSemaphore",                             "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuSignalExternalSemaphoresAsync
  m["cudaSignalExternalSemaphoresAsync"]                       = {"hipSignalExternalSemaphoresAsync",                       "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};
  // cuWaitExternalSemaphoresAsync
  m["cudaWaitExternalSemaphoresAsync"]                         = {"hipWaitExternalSemaphoresAsync",                         "", CONV_EXTERNAL_RES, API_RUNTIME, SEC::EXTERNAL_RES};

  // 7. Execution Control
  // no analogue
  m["cudaFuncGetAttributes"]                                   = {"hipFuncGetAttributes",                                   "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cuFuncSetAttribute due to different signatures
  m["cudaFuncSetAttribute"]                                    = {"hipFuncSetAttribute",                                    "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cuFuncSetCacheConfig due to different signatures
  m["cudaFuncSetCacheConfig"]                                  = {"hipFuncSetCacheConfig",                                  "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // no analogue
  m["cudaGetParameterBuffer"]                                  = {"hipGetParameterBuffer",                                  "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  m["cudaGetParameterBufferV2"]                                = {"hipGetParameterBufferV2",                                "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernel due to different signatures
  m["cudaLaunchCooperativeKernel"]                             = {"hipLaunchCooperativeKernel",                             "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernelMultiDevice due to different signatures
  m["cudaLaunchCooperativeKernelMultiDevice"]                  = {"hipLaunchCooperativeKernelMultiDevice",                  "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, CUDA_DEPRECATED | CUDA_REMOVED};
  // cuLaunchHostFunc
  m["cudaLaunchHostFunc"]                                      = {"hipLaunchHostFunc",                                      "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // cuLaunchHostFunc_v2
  m["cudaLaunchHostFunc_v2"]                                   = {"hipLaunchHostFunc_v2",                                   "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cuLaunchKernel due to different signatures
  m["cudaLaunchKernel"]                                        = {"hipLaunchKernel",                                        "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // no analogue
  m["cudaSetDoubleForDevice"]                                  = {"hipSetDoubleForDevice",                                  "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaSetDoubleForHost"]                                    = {"hipSetDoubleForHost",                                    "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  // NOTE: Not equal to cuLaunchKernelEx due to different signatures
  m["cudaLaunchKernelExC"]                                     = {"hipLaunchKernelExC",                                     "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION};
  // cuFuncGetName
  m["cudaFuncGetName"]                                         = {"hipFuncGetName",                                         "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};
  // cuFuncGetParamInfo
  m["cudaFuncGetParamInfo"]                                    = {"hipFuncGetParamInfo",                                    "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};
  // cuFuncGetParamCount
  m["cudaFuncGetParamCount"]                                   = {"hipFuncGetParamCount",                                   "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION, HIP_UNSUPPORTED};

  // 8. Execution Control [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuFuncSetSharedMemConfig due to different signatures
  m["cudaFuncSetSharedMemConfig"]                              = { "hipFuncSetSharedMemConfig",                              "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION_DEPRECATED, CUDA_DEPRECATED };

  // 9. Occupancy
  // cuOccupancyAvailableDynamicSMemPerBlock
  m["cudaOccupancyAvailableDynamicSMemPerBlock"]               = {"hipOccupancyAvailableDynamicSMemPerBlock",               "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // cuOccupancyMaxActiveBlocksPerMultiprocessor
  m["cudaOccupancyMaxActiveBlocksPerMultiprocessor"]           = {"hipOccupancyMaxActiveBlocksPerMultiprocessor",           "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  m["cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]  = {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // cuOccupancyMaxPotentialBlockSize
  m["cudaOccupancyMaxPotentialBlockSize"]                      = {"hipOccupancyMaxPotentialBlockSize",                      "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // cuOccupancyMaxPotentialBlockSizeWithFlags
  m["cudaOccupancyMaxPotentialBlockSizeWithFlags"]             = {"hipOccupancyMaxPotentialBlockSizeWithFlags",             "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // no analogue
  m["cudaOccupancyMaxPotentialBlockSizeVariableSMem"]          = {"hipOccupancyMaxPotentialBlockSizeVariableSMem",          "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // no analogue
  m["cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags"] = {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY};
  // cuOccupancyMaxPotentialClusterSize
  m["cudaOccupancyMaxPotentialClusterSize"]                    = {"hipOccupancyMaxPotentialClusterSize",                    "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, HIP_UNSUPPORTED};
  // cuOccupancyMaxActiveClusters
  m["cudaOccupancyMaxActiveClusters"]                          = {"hipOccupancyMaxActiveClusters",                          "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, HIP_UNSUPPORTED};
  // cuGraphNodeGetParams
  m["cudaGraphNodeGetParams"]                                  = {"hipGraphNodeGetParams",                                  "", CONV_OCCUPANCY, API_RUNTIME, SEC::OCCUPANCY, HIP_UNSUPPORTED};

  // 10. Memory Management
  // no analogue
  m["cudaArrayGetInfo"]                                        = {"hipArrayGetInfo",                                        "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemFree
  m["cudaFree"]                                                = {"hipFree",                                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaFreeArray"]                                           = {"hipFreeArray",                                           "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemFreeHost
  m["cudaFreeHost"]                                            = {"hipHostFree",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayDestroy due to different signatures
  m["cudaFreeMipmappedArray"]                                  = {"hipFreeMipmappedArray",                                  "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayGetLevel due to different signatures
  m["cudaGetMipmappedArrayLevel"]                              = {"hipGetMipmappedArrayLevel",                              "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaGetSymbolAddress"]                                    = {"hipGetSymbolAddress",                                    "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaGetSymbolSize"]                                       = {"hipGetSymbolSize",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostAlloc
  m["cudaHostAlloc"]                                           = {"hipHostAlloc",                                           "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostGetDevicePointer
  m["cudaHostGetDevicePointer"]                                = {"hipHostGetDevicePointer",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostGetFlags
  m["cudaHostGetFlags"]                                        = {"hipHostGetFlags",                                        "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostRegister
  m["cudaHostRegister"]                                        = {"hipHostRegister",                                        "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostUnregister
  m["cudaHostUnregister"]                                      = {"hipHostUnregister",                                      "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemAlloc
  m["cudaMalloc"]                                              = {"hipMalloc",                                              "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMalloc3D"]                                            = {"hipMalloc3D",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMalloc3DArray"]                                       = {"hipMalloc3DArray",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMallocArray"]                                         = {"hipMallocArray",                                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemHostAlloc
  m["cudaMallocHost"]                                          = {"hipHostMalloc",                                          "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemAllocManaged
  m["cudaMallocManaged"]                                       = {"hipMallocManaged",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayCreate due to different signatures
  m["cudaMallocMipmappedArray"]                                = {"hipMallocMipmappedArray",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemAllocPitch due to different signatures
  m["cudaMallocPitch"]                                         = {"hipMallocPitch",                                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemAdvise
  m["cudaMemAdvise"]                                           = {"hipMemAdvise",                                           "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  // cuMemAdvise_v2
  m["cudaMemAdvise_v2"]                                        = {"hipMemAdvise_v2",                                        "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cuMemcpy due to different signatures
  m["cudaMemcpy"]                                              = {"hipMemcpy",                                              "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy2D due to different signatures
  m["cudaMemcpy2D"]                                            = {"hipMemcpy2D",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpy2DArrayToArray"]                                = {"hipMemcpy2DArrayToArray",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy2DAsync due to different signatures
  m["cudaMemcpy2DAsync"]                                       = {"hipMemcpy2DAsync",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpy2DFromArray"]                                   = {"hipMemcpy2DFromArray",                                   "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpy2DFromArrayAsync"]                              = {"hipMemcpy2DFromArrayAsync",                              "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpy2DToArray"]                                     = {"hipMemcpy2DToArray",                                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpy2DToArrayAsync"]                                = {"hipMemcpy2DToArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy3D due to different signatures
  m["cudaMemcpy3D"]                                            = {"hipMemcpy3D",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy3DAsync due to different signatures
  m["cudaMemcpy3DAsync"]                                       = {"hipMemcpy3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeer due to different signatures
  m["cudaMemcpy3DPeer"]                                        = {"hipMemcpy3DPeer",                                        "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeerAsync due to different signatures
  m["cudaMemcpy3DPeerAsync"]                                   = {"hipMemcpy3DPeerAsync",                                   "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpyAsync due to different signatures
  m["cudaMemcpyAsync"]                                         = {"hipMemcpyAsync",                                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpyFromSymbol"]                                    = {"hipMemcpyFromSymbol",                                    "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpyFromSymbolAsync"]                               = {"hipMemcpyFromSymbolAsync",                               "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpyPeer due to different signatures
  m["cudaMemcpyPeer"]                                          = {"hipMemcpyPeer",                                          "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cuMemcpyPeerAsync due to different signatures
  m["cudaMemcpyPeerAsync"]                                     = {"hipMemcpyPeerAsync",                                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpyToSymbol"]                                      = {"hipMemcpyToSymbol",                                      "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemcpyToSymbolAsync"]                                 = {"hipMemcpyToSymbolAsync",                                 "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemcpyBatchAsync
  m["cudaMemcpyBatchAsync"]                                    = {"hipMemcpyBatchAsync",                                    "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  // cuMemcpy3DBatchAsync
  m["cudaMemcpy3DBatchAsync"]                                  = {"hipMemcpy3DBatchAsync",                                  "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  // cuMemGetInfo
  m["cudaMemGetInfo"]                                          = {"hipMemGetInfo",                                          "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemPrefetchAsync
  m["cudaMemPrefetchAsync"]                                    = {"hipMemPrefetchAsync",                                    "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  // cuMemPrefetchAsync_v2
  m["cudaMemPrefetchAsync_v2"]                                 = {"hipMemPrefetchAsync_v2",                                 "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemRangeGetAttribute
  m["cudaMemRangeGetAttribute"]                                = {"hipMemRangeGetAttribute",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemRangeGetAttributes
  m["cudaMemRangeGetAttributes"]                               = {"hipMemRangeGetAttributes",                               "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemsetD32 - hipMemsetD32
  m["cudaMemset"]                                              = {"hipMemset",                                              "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemset2D"]                                            = {"hipMemset2D",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemset2DAsync"]                                       = {"hipMemset2DAsync",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemset3D"]                                            = {"hipMemset3D",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["cudaMemset3DAsync"]                                       = {"hipMemset3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuMemsetD32Async
  m["cudaMemsetAsync"]                                         = {"hipMemsetAsync",                                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["make_cudaExtent"]                                         = {"make_hipExtent",                                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["make_cudaPitchedPtr"]                                     = {"make_hipPitchedPtr",                                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // no analogue
  m["make_cudaPos"]                                            = {"make_hipPos",                                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY};
  // cuArrayGetSparseProperties
  m["cudaArrayGetSparseProperties"]                            = {"hipArrayGetSparseProperties",                            "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuArrayGetPlane
  m["cudaArrayGetPlane"]                                       = {"hipArrayGetPlane",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuArrayGetMemoryRequirements
  m["cudaArrayGetMemoryRequirements"]                          = {"hipArrayGetMemoryRequirements",                          "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuDeviceRegisterAsyncNotification
  m["cudaDeviceRegisterAsyncNotification"]                     = {"hipDeviceRegisterAsyncNotification",                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuDeviceUnregisterAsyncNotification
  m["cudaDeviceUnregisterAsyncNotification"]                   = {"hipDeviceUnregisterAsyncNotification",                   "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemDiscardBatchAsync
  m["cudaMemDiscardBatchAsync"]                                = {"hipMemDiscardBatchAsync",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemDiscardAndPrefetchBatchAsync
  m["cudaMemDiscardAndPrefetchBatchAsync"]                     = {"hipMemDiscardAndPrefetchBatchAsync",                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemPrefetchBatchAsync
  m["cudaMemPrefetchBatchAsync"]                               = {"hipMemPrefetchBatchAsync",                               "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemcpyWithAttributesAsync
  m["cudaMemcpyWithAttributesAsync"]                           = {"hipMemcpyWithAttributesAsync",                           "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};
  // cuMemcpy3DWithAttributesAsync
  m["cudaMemcpy3DWithAttributesAsync"]                         = {"hipMemcpy3DWithAttributesAsync",                         "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY, HIP_UNSUPPORTED};

  // 11. Memory Management [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuMemcpyAtoA due to different signatures
  m["cudaMemcpyArrayToArray"]                                  = {"hipMemcpyArrayToArray",                                  "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cudaMemcpyFromArray"]                                     = {"hipMemcpyFromArray",                                     "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY_DEPRECATED, DEPRECATED};
  // no analogue
  m["cudaMemcpyFromArrayAsync"]                                = {"hipMemcpyFromArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cudaMemcpyToArray"]                                       = {"hipMemcpyToArray",                                       "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY_DEPRECATED, DEPRECATED};
  // no analogue
  m["cudaMemcpyToArrayAsync"]                                  = {"hipMemcpyToArrayAsync",                                  "", CONV_MEMORY, API_RUNTIME, SEC::MEMORY_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 12. Stream Ordered Memory Allocator

  // cuMemAllocAsync
  m["cudaMallocAsync"]                                         = {"hipMallocAsync",                                         "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemFreeAsync
  m["cudaFreeAsync"]                                           = {"hipFreeAsync",                                           "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemAllocFromPoolAsync
  m["cudaMallocFromPoolAsync"]                                 = {"hipMallocFromPoolAsync",                                 "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolTrimTo
  m["cudaMemPoolTrimTo"]                                       = {"hipMemPoolTrimTo",                                       "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolSetAttribute
  m["cudaMemPoolSetAttribute"]                                 = {"hipMemPoolSetAttribute",                                 "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolGetAttribute
  m["cudaMemPoolGetAttribute"]                                 = {"hipMemPoolGetAttribute",                                 "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolSetAccess
  m["cudaMemPoolSetAccess"]                                    = {"hipMemPoolSetAccess",                                    "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolGetAccess
  m["cudaMemPoolGetAccess"]                                    = {"hipMemPoolGetAccess",                                    "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolCreate
  m["cudaMemPoolCreate"]                                       = {"hipMemPoolCreate",                                       "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolDestroy
  m["cudaMemPoolDestroy"]                                      = {"hipMemPoolDestroy",                                      "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolExportToShareableHandle
  m["cudaMemPoolExportToShareableHandle"]                      = {"hipMemPoolExportToShareableHandle",                      "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolImportFromShareableHandle
  m["cudaMemPoolImportFromShareableHandle"]                    = {"hipMemPoolImportFromShareableHandle",                    "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolExportPointer
  m["cudaMemPoolExportPointer"]                                = {"hipMemPoolExportPointer",                                "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemPoolImportPointer
  m["cudaMemPoolImportPointer"]                                = {"hipMemPoolImportPointer",                                "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY};
  // cuMemGetDefaultMemPool
  m["cudaMemGetDefaultMemPool"]                                = {"hipMemGetDefaultMemPool",                                "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};
  // cuMemGetMemPool
  m["cudaMemGetMemPool"]                                       = {"hipMemGetMemPool",                                       "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};
  // cuMemSetMemPool
  m["cudaMemSetMemPool"]                                       = {"hipMemSetMemPool",                                       "", CONV_MEMORY, API_RUNTIME, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};

  // 13. Unified Addressing
  // no analogue
  // NOTE: Not equal to cuPointerGetAttributes due to different signatures
  m["cudaPointerGetAttributes"]                                = {"hipPointerGetAttributes",                                "", CONV_UNIFIED, API_RUNTIME, SEC::UNIFIED};

  // 14. Peer Device Memory Access
  // cuDeviceCanAccessPeer
  m["cudaDeviceCanAccessPeer"]                                 = {"hipDeviceCanAccessPeer",                                 "", CONV_PEER, API_RUNTIME, SEC::PEER};
  // no analogue
  // NOTE: Not equal to cuCtxDisablePeerAccess due to different signatures
  m["cudaDeviceDisablePeerAccess"]                             = {"hipDeviceDisablePeerAccess",                             "", CONV_PEER, API_RUNTIME, SEC::PEER};
  // no analogue
  // NOTE: Not equal to cuCtxEnablePeerAccess due to different signatures
  m["cudaDeviceEnablePeerAccess"]                              = {"hipDeviceEnablePeerAccess",                              "", CONV_PEER, API_RUNTIME, SEC::PEER};

  // 15. OpenGL Interoperability
  // cuGLGetDevices
  m["cudaGLGetDevices"]                                        = {"hipGLGetDevices",                                        "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL};
  // cuGraphicsGLRegisterBuffer
  m["cudaGraphicsGLRegisterBuffer"]                            = {"hipGraphicsGLRegisterBuffer",                            "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL};
  // cuGraphicsGLRegisterImage
  m["cudaGraphicsGLRegisterImage"]                             = {"hipGraphicsGLRegisterImage",                             "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL};
  // cuWGLGetDevice
  m["cudaWGLGetDevice"]                                        = {"hipWGLGetDevice",                                        "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL, HIP_UNSUPPORTED};

  // 16. OpenGL Interoperability [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObject due to different signatures
  m["cudaGLMapBufferObject"]                                   = {"hipGLMapBufferObject",                                   "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObjectAsync due to different signatures
  m["cudaGLMapBufferObjectAsync"]                              = {"hipGLMapBufferObjectAsync",                              "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuGLRegisterBufferObject
  m["cudaGLRegisterBufferObject"]                              = {"hipGLRegisterBufferObject",                              "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuGLSetBufferObjectMapFlags
  m["cudaGLSetBufferObjectMapFlags"]                           = {"hipGLSetBufferObjectMapFlags",                           "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cudaGLSetGLDevice"]                                       = {"hipGLSetGLDevice",                                       "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuGLUnmapBufferObject
  m["cudaGLUnmapBufferObject"]                                 = {"hipGLUnmapBufferObject",                                 "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuGLUnmapBufferObjectAsync
  m["cudaGLUnmapBufferObjectAsync"]                            = {"hipGLUnmapBufferObjectAsync",                            "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuGLUnregisterBufferObject
  m["cudaGLUnregisterBufferObject"]                            = {"hipGLUnregisterBufferObject",                            "", CONV_OPENGL, API_RUNTIME, SEC::OPENGL_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 17. Direct3D 9 Interoperability
  // cuD3D9GetDevice
  m["cudaD3D9GetDevice"]                                       = {"hipD3D9GetDevice",                                       "", CONV_D3D9, API_RUNTIME, SEC::D3D9, HIP_UNSUPPORTED};
  // cuD3D9GetDevices
  m["cudaD3D9GetDevices"]                                      = {"hipD3D9GetDevices",                                      "", CONV_D3D9, API_RUNTIME, SEC::D3D9, HIP_UNSUPPORTED};
  // cuD3D9GetDirect3DDevice
  m["cudaD3D9GetDirect3DDevice"]                               = {"hipD3D9GetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, SEC::D3D9, HIP_UNSUPPORTED};
  // no analogue
  m["cudaD3D9SetDirect3DDevice"]                               = {"hipD3D9SetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, SEC::D3D9, HIP_UNSUPPORTED};
  // cuGraphicsD3D9RegisterResource
  m["cudaGraphicsD3D9RegisterResource"]                        = {"hipGraphicsD3D9RegisterResource",                        "", CONV_D3D9, API_RUNTIME, SEC::D3D9, HIP_UNSUPPORTED};

  // 18. Direct3D 9 Interoperability[DEPRECATED]
  // cuD3D9MapResources
  m["cudaD3D9MapResources"]                                    = {"hipD3D9MapResources",                                    "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9RegisterResource
  // NOTE: cudaD3D9RegisterResource is not marked as deprecated function even in CUDA 11.0
  m["cudaD3D9RegisterResource"]                                = {"hipD3D9RegisterResource",                                "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED};
  // cuD3D9ResourceGetMappedArray
  m["cudaD3D9ResourceGetMappedArray"]                          = {"hipD3D9ResourceGetMappedArray",                          "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9ResourceGetMappedPitch
  m["cudaD3D9ResourceGetMappedPitch"]                          = {"hipD3D9ResourceGetMappedPitch",                          "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9ResourceGetMappedPointer
  m["cudaD3D9ResourceGetMappedPointer"]                        = {"hipD3D9ResourceGetMappedPointer",                        "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9ResourceGetMappedSize
  m["cudaD3D9ResourceGetMappedSize"]                           = {"hipD3D9ResourceGetMappedSize",                           "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9ResourceGetSurfaceDimensions
  m["cudaD3D9ResourceGetSurfaceDimensions"]                    = {"hipD3D9ResourceGetSurfaceDimensions",                    "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9ResourceSetMapFlags
  m["cudaD3D9ResourceSetMapFlags"]                             = {"hipD3D9ResourceSetMapFlags",                             "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9UnmapResources
  m["cudaD3D9UnmapResources"]                                  = {"hipD3D9UnmapResources",                                  "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D9UnregisterResource
  m["cudaD3D9UnregisterResource"]                              = {"hipD3D9UnregisterResource",                              "", CONV_D3D9, API_RUNTIME, SEC::D3D9_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 19. Direct3D 10 Interoperability
  // cuD3D10GetDevice
  m["cudaD3D10GetDevice"]                                      = {"hipD3D10GetDevice",                                      "", CONV_D3D10, API_RUNTIME, SEC::D3D10, HIP_UNSUPPORTED};
  // cuD3D10GetDevices
  m["cudaD3D10GetDevices"]                                     = {"hipD3D10GetDevices",                                     "", CONV_D3D10, API_RUNTIME, SEC::D3D10, HIP_UNSUPPORTED};
  // cuGraphicsD3D10RegisterResource
  m["cudaGraphicsD3D10RegisterResource"]                       = {"hipGraphicsD3D10RegisterResource",                       "", CONV_D3D10, API_RUNTIME, SEC::D3D10, HIP_UNSUPPORTED};

  // 20. Direct3D 10 Interoperability [DEPRECATED]
  // cuD3D10GetDirect3DDevice
  m["cudaD3D10GetDirect3DDevice"]                              = {"hipD3D10GetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10MapResources
  m["cudaD3D10MapResources"]                                   = {"hipD3D10MapResources",                                   "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10RegisterResource
  m["cudaD3D10RegisterResource"]                               = {"hipD3D10RegisterResource",                               "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceGetMappedArray
  m["cudaD3D10ResourceGetMappedArray"]                         = {"hipD3D10ResourceGetMappedArray",                         "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceGetMappedPitch
  m["cudaD3D10ResourceGetMappedPitch"]                         = {"hipD3D10ResourceGetMappedPitch",                         "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceGetMappedPointer
  m["cudaD3D10ResourceGetMappedPointer"]                       = {"hipD3D10ResourceGetMappedPointer",                       "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceGetMappedSize
  m["cudaD3D10ResourceGetMappedSize"]                          = {"hipD3D10ResourceGetMappedSize",                          "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceGetSurfaceDimensions
  m["cudaD3D10ResourceGetSurfaceDimensions"]                   = {"hipD3D10ResourceGetSurfaceDimensions",                   "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10ResourceSetMapFlags
  m["cudaD3D10ResourceSetMapFlags"]                            = {"hipD3D10ResourceSetMapFlags",                            "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cudaD3D10SetDirect3DDevice"]                              = {"hipD3D10SetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10UnmapResources
  m["cudaD3D10UnmapResources"]                                 = {"hipD3D10UnmapResources",                                 "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cuD3D10UnregisterResource
  m["cudaD3D10UnregisterResource"]                             = {"hipD3D10UnregisterResource",                             "", CONV_D3D10, API_RUNTIME, SEC::D3D10_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 21. Direct3D 11 Interoperability
  // cuD3D11GetDevice
  m["cudaD3D11GetDevice"]                                      = {"hipD3D11GetDevice",                                      "", CONV_D3D11, API_RUNTIME, SEC::D3D11, HIP_UNSUPPORTED};
  // cuD3D11GetDevices
  m["cudaD3D11GetDevices"]                                     = {"hipD3D11GetDevices",                                     "", CONV_D3D11, API_RUNTIME, SEC::D3D11, HIP_UNSUPPORTED};
  // cuGraphicsD3D11RegisterResource
  m["cudaGraphicsD3D11RegisterResource"]                       = {"hipGraphicsD3D11RegisterResource",                       "", CONV_D3D11, API_RUNTIME, SEC::D3D11, HIP_UNSUPPORTED};

  // 22. Direct3D 11 Interoperability [DEPRECATED]
  // cuD3D11GetDirect3DDevice
  m["cudaD3D11GetDirect3DDevice"]                              = {"hipD3D11GetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, SEC::D3D11_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cudaD3D11SetDirect3DDevice"]                              = {"hipD3D11SetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, SEC::D3D11_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 23. VDPAU Interoperability
  // cuGraphicsVDPAURegisterOutputSurface
  m["cudaGraphicsVDPAURegisterOutputSurface"]                  = {"hipGraphicsVDPAURegisterOutputSurface",                  "", CONV_VDPAU, API_RUNTIME, SEC::VDPAU, HIP_UNSUPPORTED};
  // cuGraphicsVDPAURegisterVideoSurface
  m["cudaGraphicsVDPAURegisterVideoSurface"]                   = {"hipGraphicsVDPAURegisterVideoSurface",                   "", CONV_VDPAU, API_RUNTIME, SEC::VDPAU, HIP_UNSUPPORTED};
  // cuVDPAUGetDevice
  m["cudaVDPAUGetDevice"]                                      = {"hipVDPAUGetDevice",                                      "", CONV_VDPAU, API_RUNTIME, SEC::VDPAU, HIP_UNSUPPORTED};
  // no analogue
  m["cudaVDPAUSetVDPAUDevice"]                                 = {"hipVDPAUSetDevice",                                      "", CONV_VDPAU, API_RUNTIME, SEC::VDPAU, HIP_UNSUPPORTED};

  // 24. EGL Interoperability
  // cuEGLStreamConsumerAcquireFrame
  m["cudaEGLStreamConsumerAcquireFrame"]                       = {"hipEGLStreamConsumerAcquireFrame",                       "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamConsumerConnect
  m["cudaEGLStreamConsumerConnect"]                            = {"hipEGLStreamConsumerConnect",                            "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamConsumerConnectWithFlags
  m["cudaEGLStreamConsumerConnectWithFlags"]                   = {"hipEGLStreamConsumerConnectWithFlags",                   "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamConsumerDisconnect
  m["cudaEGLStreamConsumerDisconnect"]                         = {"hipEGLStreamConsumerDisconnect",                         "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamConsumerReleaseFrame
  m["cudaEGLStreamConsumerReleaseFrame"]                       = {"hipEGLStreamConsumerReleaseFrame",                       "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamProducerConnect
  m["cudaEGLStreamProducerConnect"]                            = {"hipEGLStreamProducerConnect",                            "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamProducerDisconnect
  m["cudaEGLStreamProducerDisconnect"]                         = {"hipEGLStreamProducerDisconnect",                         "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamProducerPresentFrame
  m["cudaEGLStreamProducerPresentFrame"]                       = {"hipEGLStreamProducerPresentFrame",                       "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEGLStreamProducerReturnFrame
  m["cudaEGLStreamProducerReturnFrame"]                        = {"hipEGLStreamProducerReturnFrame",                        "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuEventCreateFromEGLSync
  m["cudaEventCreateFromEGLSync"]                              = {"hipEventCreateFromEGLSync",                              "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuGraphicsEGLRegisterImage
  m["cudaGraphicsEGLRegisterImage"]                            = {"hipGraphicsEGLRegisterImage",                            "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};
  // cuGraphicsResourceGetMappedEglFrame
  m["cudaGraphicsResourceGetMappedEglFrame"]                   = {"hipGraphicsResourceGetMappedEglFrame",                   "", CONV_EGL, API_RUNTIME, SEC::EGL, HIP_UNSUPPORTED};

  // 25. Graphics Interoperability
  // cuGraphicsMapResources
  m["cudaGraphicsMapResources"]                                = {"hipGraphicsMapResources",                                "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS};
  // cuGraphicsResourceGetMappedMipmappedArray
  m["cudaGraphicsResourceGetMappedMipmappedArray"]             = {"hipGraphicsResourceGetMappedMipmappedArray",             "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS, HIP_UNSUPPORTED};
  // cuGraphicsResourceGetMappedPointer
  m["cudaGraphicsResourceGetMappedPointer"]                    = {"hipGraphicsResourceGetMappedPointer",                    "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS};
  // cuGraphicsResourceSetMapFlags
  m["cudaGraphicsResourceSetMapFlags"]                         = {"hipGraphicsResourceSetMapFlags",                         "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS, HIP_UNSUPPORTED};
  // cuGraphicsSubResourceGetMappedArray
  m["cudaGraphicsSubResourceGetMappedArray"]                   = {"hipGraphicsSubResourceGetMappedArray",                   "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS};
  // cuGraphicsUnmapResources
  m["cudaGraphicsUnmapResources"]                              = {"hipGraphicsUnmapResources",                              "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS};
  // cuGraphicsUnregisterResource
  m["cudaGraphicsUnregisterResource"]                          = {"hipGraphicsUnregisterResource",                          "", CONV_GRAPHICS, API_RUNTIME, SEC::GRAPHICS};

  // 26. Texture Object Management
  // no analogue
  // NOTE: Not equal to cuTexObjectCreate due to different signatures
  m["cudaCreateTextureObject"]                                 = {"hipCreateTextureObject",                                 "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  // cuTexObjectDestroy
  m["cudaDestroyTextureObject"]                                = {"hipDestroyTextureObject",                                "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  // no analogue
  // NOTE: Not equal to cuTexObjectGetResourceDesc due to different signatures
  m["cudaGetTextureObjectResourceDesc"]                        = {"hipGetTextureObjectResourceDesc",                        "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  // cuTexObjectGetResourceViewDesc
  m["cudaGetTextureObjectResourceViewDesc"]                    = {"hipGetTextureObjectResourceViewDesc",                    "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  // no analogue
  // NOTE: Not equal to cuTexObjectGetTextureDesc due to different signatures
  m["cudaGetTextureObjectTextureDesc"]                         = {"hipGetTextureObjectTextureDesc",                         "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  //
  m["cudaCreateTextureObject_v2"]                              = {"hipCreateTextureObject_v2",                              "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE, HIP_UNSUPPORTED | CUDA_REMOVED};
  //
  m["cudaGetTextureObjectTextureDesc_v2"]                      = {"hipGetTextureObjectTextureDesc_v2",                      "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE, HIP_UNSUPPORTED | CUDA_REMOVED};
  // no analogue
  m["cudaCreateChannelDesc"]                                   = {"hipCreateChannelDesc",                                   "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};
  // no analogue
  m["cudaGetChannelDesc"]                                      = {"hipGetChannelDesc",                                      "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE};

  // 27. Surface Object Management
  // no analogue
  // NOTE: Not equal to cuSurfObjectCreate due to different signatures
  m["cudaCreateSurfaceObject"]                                 = {"hipCreateSurfaceObject",                                 "", CONV_SURFACE, API_RUNTIME, SEC::SURFACE};
  // cuSurfObjectDestroy
  m["cudaDestroySurfaceObject"]                                = {"hipDestroySurfaceObject",                                "", CONV_SURFACE, API_RUNTIME, SEC::SURFACE};
  // no analogue
  // NOTE: Not equal to cuSurfObjectGetResourceDesc due to different signatures
  m["cudaGetSurfaceObjectResourceDesc"]                        = {"hipGetSurfaceObjectResourceDesc",                        "", CONV_SURFACE, API_RUNTIME, SEC::SURFACE, HIP_UNSUPPORTED};

  // 28. Version Management
  // cuDriverGetVersion
  m["cudaDriverGetVersion"]                                    = {"hipDriverGetVersion",                                    "", CONV_VERSION, API_RUNTIME, SEC::VERSION};
  // no analogue
  m["cudaRuntimeGetVersion"]                                   = {"hipRuntimeGetVersion",                                   "", CONV_VERSION, API_RUNTIME, SEC::VERSION};

  // 29. Log Management Functions
  // cuLogsRegisterCallback
  m["cudaLogsRegisterCallback"]                                = {"hipLogsRegisterCallback",                                "", CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cuLogsUnregisterCallback
  m["cudaLogsUnregisterCallback"]                              = {"hipLogsUnregisterCallback",                              "", CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cuLogsCurrent
  m["cudaLogsCurrent"]                                         = {"hipLogsCurrent",                                         "", CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cuLogsDumpToFile
  m["cudaLogsDumpToFile"]                                      = {"hipLogsDumpToFile",                                      "", CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cuLogsDumpToMemory
  m["cudaLogsDumpToMemory"]                                    = {"hipLogsDumpToMemory",                                    "", CONV_ERROR_LOG, API_RUNTIME, SEC::ERROR_LOG, HIP_UNSUPPORTED};

  // 30. Graph Management
  // cuGraphAddChildGraphNode
  m["cudaGraphAddChildGraphNode"]                              = {"hipGraphAddChildGraphNode",                              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddDependencies
  m["cudaGraphAddDependencies"]                                = {"hipGraphAddDependencies",                                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphAddDependencies_v2
  m["cudaGraphAddDependencies_v2"]                             = {"hipGraphAddDependencies_v2",                             "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphAddEmptyNode
  m["cudaGraphAddEmptyNode"]                                   = {"hipGraphAddEmptyNode",                                   "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddHostNode
  m["cudaGraphAddHostNode"]                                    = {"hipGraphAddHostNode",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddKernelNode
  m["cudaGraphAddKernelNode"]                                  = {"hipGraphAddKernelNode",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  // NOTE: Not equal to cuGraphAddMemcpyNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  m["cudaGraphAddMemcpyNode"]                                  = {"hipGraphAddMemcpyNode",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  // NOTE: Not equal to cuGraphAddMemsetNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);
  m["cudaGraphAddMemsetNode"]                                  = {"hipGraphAddMemsetNode",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphChildGraphNodeGetGraph
  m["cudaGraphChildGraphNodeGetGraph"]                         = {"hipGraphChildGraphNodeGetGraph",                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphClone
  m["cudaGraphClone"]                                          = {"hipGraphClone",                                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphCreate
  m["cudaGraphCreate"]                                         = {"hipGraphCreate",                                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphDebugDotPrint
  m["cudaGraphDebugDotPrint"]                                  = {"hipGraphDebugDotPrint",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphDestroy
  m["cudaGraphDestroy"]                                        = {"hipGraphDestroy",                                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphDestroyNode
  m["cudaGraphDestroyNode"]                                    = {"hipGraphDestroyNode",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecDestroy
  m["cudaGraphExecDestroy"]                                    = {"hipGraphExecDestroy",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphGetEdges
  m["cudaGraphGetEdges"]                                       = {"hipGraphGetEdges",                                       "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphGetEdges_v2
  m["cudaGraphGetEdges_v2"]                                    = {"hipGraphGetEdges_v2",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphGetNodes
  m["cudaGraphGetNodes"]                                       = {"hipGraphGetNodes",                                       "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphGetRootNodes
  m["cudaGraphGetRootNodes"]                                   = {"hipGraphGetRootNodes",                                   "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphHostNodeGetParams
  m["cudaGraphHostNodeGetParams"]                              = {"hipGraphHostNodeGetParams",                              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphHostNodeSetParams
  m["cudaGraphHostNodeSetParams"]                              = {"hipGraphHostNodeSetParams",                              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphInstantiate
  // NOTE: CUDA signature changed since 12.0
  m["cudaGraphInstantiate"]                                    = {"hipGraphInstantiate",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, CUDA_OVERLOADED};
  // cuGraphKernelNodeCopyAttributes
  m["cudaGraphKernelNodeCopyAttributes"]                       = {"hipGraphKernelNodeCopyAttributes",                       "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphKernelNodeGetAttribute
  m["cudaGraphKernelNodeGetAttribute"]                         = {"hipGraphKernelNodeGetAttribute",                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphKernelNodeSetAttribute
  m["cudaGraphKernelNodeSetAttribute"]                         = {"hipGraphKernelNodeSetAttribute",                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecKernelNodeSetParams
  m["cudaGraphExecKernelNodeSetParams"]                        = {"hipGraphExecKernelNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphExecMemcpyNodeSetParams"]                        = {"hipGraphExecMemcpyNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphExecMemsetNodeSetParams"]                        = {"hipGraphExecMemsetNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecHostNodeSetParams
  m["cudaGraphExecHostNodeSetParams"]                          = {"hipGraphExecHostNodeSetParams",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecUpdate
  // NOTE: CUDA signature has changed since 12.0
  m["cudaGraphExecUpdate"]                                     = {"hipGraphExecUpdate",                                     "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphKernelNodeGetParams
  m["cudaGraphKernelNodeGetParams"]                            = {"hipGraphKernelNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphKernelNodeSetParams
  m["cudaGraphKernelNodeSetParams"]                            = {"hipGraphKernelNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphLaunch
  m["cudaGraphLaunch"]                                         = {"hipGraphLaunch",                                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemcpyNodeGetParams
  m["cudaGraphMemcpyNodeGetParams"]                            = {"hipGraphMemcpyNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemcpyNodeSetParams
  m["cudaGraphMemcpyNodeSetParams"]                            = {"hipGraphMemcpyNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemsetNodeGetParams
  m["cudaGraphMemsetNodeGetParams"]                            = {"hipGraphMemsetNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemsetNodeSetParams
  m["cudaGraphMemsetNodeSetParams"]                            = {"hipGraphMemsetNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphNodeFindInClone
  m["cudaGraphNodeFindInClone"]                                = {"hipGraphNodeFindInClone",                                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphNodeGetDependencies
  m["cudaGraphNodeGetDependencies"]                            = {"hipGraphNodeGetDependencies",                            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphNodeGetDependencies_v2
  m["cudaGraphNodeGetDependencies_v2"]                         = {"hipGraphNodeGetDependencies_v2",                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeGetDependentNodes
  m["cudaGraphNodeGetDependentNodes"]                          = {"hipGraphNodeGetDependentNodes",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphNodeGetDependentNodes_v2
  m["cudaGraphNodeGetDependentNodes_v2"]                       = {"hipGraphNodeGetDependentNodes_v2",                       "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeGetEnabled
  m["cudaGraphNodeGetEnabled"]                                 = {"hipGraphNodeGetEnabled",                                 "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphNodeGetType
  m["cudaGraphNodeGetType"]                                    = {"hipGraphNodeGetType",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphRemoveDependencies
  m["cudaGraphRemoveDependencies"]                             = {"hipGraphRemoveDependencies",                             "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphRemoveDependencies_v2
  m["cudaGraphRemoveDependencies_v2"]                          = {"hipGraphRemoveDependencies_v2",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // no analogue
  m["cudaGraphAddMemcpyNodeToSymbol"]                          = {"hipGraphAddMemcpyNodeToSymbol",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphAddMemcpyNodeFromSymbol"]                        = {"hipGraphAddMemcpyNodeFromSymbol",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphAddMemcpyNode1D"]                                = {"hipGraphAddMemcpyNode1D",                                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphMemcpyNodeSetParamsToSymbol"]                    = {"hipGraphMemcpyNodeSetParamsToSymbol",                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphMemcpyNodeSetParamsFromSymbol"]                  = {"hipGraphMemcpyNodeSetParamsFromSymbol",                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphMemcpyNodeSetParams1D"]                          = {"hipGraphMemcpyNodeSetParams1D",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddEventRecordNode
  m["cudaGraphAddEventRecordNode"]                             = {"hipGraphAddEventRecordNode",                             "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphEventRecordNodeGetEvent
  m["cudaGraphEventRecordNodeGetEvent"]                        = {"hipGraphEventRecordNodeGetEvent",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphEventRecordNodeSetEvent
  m["cudaGraphEventRecordNodeSetEvent"]                        = {"hipGraphEventRecordNodeSetEvent",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddEventWaitNode
  m["cudaGraphAddEventWaitNode"]                               = {"hipGraphAddEventWaitNode",                               "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphEventWaitNodeGetEvent
  m["cudaGraphEventWaitNodeGetEvent"]                          = {"hipGraphEventWaitNodeGetEvent",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphEventWaitNodeSetEvent
  m["cudaGraphEventWaitNodeSetEvent"]                          = {"hipGraphEventWaitNodeSetEvent",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphExecMemcpyNodeSetParamsToSymbol"]                = {"hipGraphExecMemcpyNodeSetParamsToSymbol",                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphExecMemcpyNodeSetParamsFromSymbol"]              = {"hipGraphExecMemcpyNodeSetParamsFromSymbol",              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphExecMemcpyNodeSetParams1D"]                      = {"hipGraphExecMemcpyNodeSetParams1D",                      "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecChildGraphNodeSetParams
  m["cudaGraphExecChildGraphNodeSetParams"]                    = {"hipGraphExecChildGraphNodeSetParams",                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecEventRecordNodeSetEvent
  m["cudaGraphExecEventRecordNodeSetEvent"]                    = {"hipGraphExecEventRecordNodeSetEvent",                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecEventWaitNodeSetEvent
  m["cudaGraphExecEventWaitNodeSetEvent"]                      = {"hipGraphExecEventWaitNodeSetEvent",                      "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphUpload
  m["cudaGraphUpload"]                                         = {"hipGraphUpload",                                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddExternalSemaphoresSignalNode
  m["cudaGraphAddExternalSemaphoresSignalNode"]                = {"hipGraphAddExternalSemaphoresSignalNode",                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExternalSemaphoresSignalNodeGetParams
  m["cudaGraphExternalSemaphoresSignalNodeGetParams"]          = {"hipGraphExternalSemaphoresSignalNodeGetParams",          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExternalSemaphoresSignalNodeSetParams
  m["cudaGraphExternalSemaphoresSignalNodeSetParams"]          = {"hipGraphExternalSemaphoresSignalNodeSetParams",          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddExternalSemaphoresWaitNode
  m["cudaGraphAddExternalSemaphoresWaitNode"]                  = {"hipGraphAddExternalSemaphoresWaitNode",                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExternalSemaphoresWaitNodeGetParams
  m["cudaGraphExternalSemaphoresWaitNodeGetParams"]            = {"hipGraphExternalSemaphoresWaitNodeGetParams",            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExternalSemaphoresWaitNodeSetParams
  m["cudaGraphExternalSemaphoresWaitNodeSetParams"]            = {"hipGraphExternalSemaphoresWaitNodeSetParams",            "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecExternalSemaphoresSignalNodeSetParams
  m["cudaGraphExecExternalSemaphoresSignalNodeSetParams"]      = {"hipGraphExecExternalSemaphoresSignalNodeSetParams",      "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecExternalSemaphoresWaitNodeSetParams
  m["cudaGraphExecExternalSemaphoresWaitNodeSetParams"]        = {"hipGraphExecExternalSemaphoresWaitNodeSetParams",        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuUserObjectCreate
  m["cudaUserObjectCreate"]                                    = {"hipUserObjectCreate",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuUserObjectRetain
  m["cudaUserObjectRetain"]                                    = {"hipUserObjectRetain",                                    "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuUserObjectRelease
  m["cudaUserObjectRelease"]                                   = {"hipUserObjectRelease",                                   "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphRetainUserObject
  m["cudaGraphRetainUserObject"]                               = {"hipGraphRetainUserObject",                               "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphReleaseUserObject
  m["cudaGraphReleaseUserObject"]                              = {"hipGraphReleaseUserObject",                              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddMemAllocNode
  m["cudaGraphAddMemAllocNode"]                                = {"hipGraphAddMemAllocNode",                                "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemAllocNodeGetParams
  m["cudaGraphMemAllocNodeGetParams"]                          = {"hipGraphMemAllocNodeGetParams",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // no analogue
  m["cudaGraphAddMemFreeNode"]                                 = {"hipGraphAddMemFreeNode",                                 "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphMemFreeNodeGetParams
  m["cudaGraphMemFreeNodeGetParams"]                           = {"hipGraphMemFreeNodeGetParams",                           "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuDeviceGraphMemTrim
  m["cudaDeviceGraphMemTrim"]                                  = {"hipDeviceGraphMemTrim",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuDeviceGetGraphMemAttribute
  m["cudaDeviceGetGraphMemAttribute"]                          = {"hipDeviceGetGraphMemAttribute",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuDeviceSetGraphMemAttribute
  m["cudaDeviceSetGraphMemAttribute"]                          = {"hipDeviceSetGraphMemAttribute",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphInstantiateWithFlags
  // NOTE: CUDA signature changed since 12.0
  m["cudaGraphInstantiateWithFlags"]                           = {"hipGraphInstantiateWithFlags",                           "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphNodeSetEnabled
  m["cudaGraphNodeSetEnabled"]                                 = {"hipGraphNodeSetEnabled",                                 "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphInstantiateWithParams
  m["cudaGraphInstantiateWithParams"]                          = {"hipGraphInstantiateWithParams",                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecGetFlags
  m["cudaGraphExecGetFlags"]                                   = {"hipGraphExecGetFlags",                                   "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphAddNode
  m["cudaGraphAddNode"]                                        = {"hipGraphAddNode",                                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cuGraphAddNode_v2
  m["cudaGraphAddNode_v2"]                                     = {"hipGraphAddNode_v2",                                     "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeSetParams
  m["cudaGraphNodeSetParams"]                                  = {"hipGraphNodeSetParams",                                  "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphExecNodeSetParams
  m["cudaGraphExecNodeSetParams"]                              = {"hipGraphExecNodeSetParams",                              "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH};
  // cuGraphConditionalHandleCreate
  m["cudaGraphConditionalHandleCreate"]                        = {"hipGraphConditionalHandleCreate",                        "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeGetContainingGraph
  m["cudaGraphNodeGetContainingGraph"]                         = {"hipGraphNodeGetContainingGraph",                         "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeGetLocalId
  m["cudaGraphNodeGetLocalId"]                                 = {"hipGraphNodeGetLocalId",                                 "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphNodeGetToolsId
  m["cudaGraphNodeGetToolsId"]                                 = {"hipGraphNodeGetToolsId",                                 "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphGetId
  m["cudaGraphGetId"]                                           = {"hipGraphGetId",                                          "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // cuGraphExecGetId
  m["cudaGraphExecGetId"]                                      = {"hipGraphExecGetId",                                      "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};
  // no analogue
  m["cudaGraphConditionalHandleCreate_v2"]                     = {"hipGraphConditionalHandleCreate_v2",                     "", CONV_GRAPH, API_RUNTIME, SEC::GRAPH, HIP_UNSUPPORTED};

  // 31. Driver Entry Point Access
  // cuGetProcAddress
  m["cudaGetDriverEntryPoint"]                                 = {"hipGetDriverEntryPoint",                                 "", CONV_DRIVER_ENTRY_POINT, API_RUNTIME, SEC::DRIVER_ENTRY_POINT, CUDA_DEPRECATED};
  //
  m["cudaGetDriverEntryPointByVersion"]                        = {"hipGetDriverEntryPointByVersion",                        "", CONV_DRIVER_ENTRY_POINT, API_RUNTIME, SEC::DRIVER_ENTRY_POINT, HIP_UNSUPPORTED};

  // 32. Library Management
  // cuLibraryLoadData
  m["cudaLibraryLoadData"]                                     = {"hipLibraryLoadData",                                     "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};
  // cuLibraryLoadFromFile
  m["cudaLibraryLoadFromFile"]                                 = {"hipLibraryLoadFromFile",                                 "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};
  // cuLibraryUnload
  m["cudaLibraryUnload"]                                       = {"hipLibraryUnload",                                       "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};
  // cuLibraryGetKernel
  m["cudaLibraryGetKernel"]                                    = {"hipLibraryGetKernel",                                    "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};
  // cuLibraryGetGlobal
  m["cudaLibraryGetGlobal"]                                    = {"hipLibraryGetGlobal",                                    "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cuLibraryGetManaged
  m["cudaLibraryGetManaged"]                                   = {"hipLibraryGetManaged",                                   "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cuLibraryGetUnifiedFunction
  m["cudaLibraryGetUnifiedFunction"]                           = {"hipLibraryGetUnifiedFunction",                           "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cuKernelSetAttribute
  m["cudaKernelSetAttributeForDevice"]                         = {"hipKernelSetAttribute",                                  "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cuLibraryGetKernelCount
  m["cudaLibraryGetKernelCount"]                               = {"hipLibraryGetKernelCount",                               "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};
  // cuLibraryEnumerateKernels
  m["cudaLibraryEnumerateKernels"]                             = {"hipLibraryEnumerateKernels",                             "", CONV_LIBRARY, API_RUNTIME, SEC::LIBRARY};

  // 33. Execution Context Management
  //
  m["cudaDeviceGetDevResource"]                                = {"hipDeviceGetDevResource",                                "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaDevSmResourceSplitByCount"]                           = {"hipDevSmResourceSplitByCount",                           "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaDevSmResourceSplit"]                                  = {"hipDevSmResourceSplit",                                  "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaDevResourceGenerateDesc"]                             = {"hipDevResourceGenerateDesc",                             "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaGreenCtxCreate"]                                      = {"hipGreenCtxCreate",                                      "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxDestroy"]                                 = {"hipExecutionCtxDestroy",                                 "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxGetDevResource"]                          = {"hipExecutionCtxGetDevResource",                          "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxGetDevice"]                               = {"hipExecutionCtxGetDevice",                               "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxGetId"]                                   = {"hipExecutionCtxGetId",                                   "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxStreamCreate"]                            = {"hipExecutionCtxStreamCreate",                            "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxSynchronize"]                             = {"hipExecutionCtxSynchronize",                             "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaStreamGetDevResource"]                                = {"hipStreamGetDevResource",                                "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxRecordEvent"]                             = {"hipExecutionCtxRecordEvent",                             "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaExecutionCtxWaitEvent"]                               = {"hipExecutionCtxWaitEvent",                               "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};
  //
  m["cudaDeviceGetExecutionCtx"]                               = {"hipDeviceGetExecutionCtx",                               "", CONV_LIBRARY, API_RUNTIME, SEC::EXECUTION_CONTEXT_MANGEMENT, HIP_UNSUPPORTED};

  // 34. C++ API Routines
  m["cudaGetKernel"]                                           = {"hipGetKernel",                                           "", CONV_CPP, API_RUNTIME, SEC::CPP, HIP_UNSUPPORTED};

  // 35. Interactions with the CUDA Driver API
  m["cudaGetFuncBySymbol"]                                     = {"hipGetFuncBySymbol",                                     "", CONV_DRIVER_INTERACT, API_RUNTIME, SEC::DRIVER_INTERACT};

  // 36. Profiler Control
  // cuProfilerStart
  m["cudaProfilerStart"]                                       = {"hipProfilerStart",                                       "", CONV_PROFILER, API_RUNTIME, SEC::PROFILER, HIP_DEPRECATED};
  // cuProfilerStop
  m["cudaProfilerStop"]                                        = {"hipProfilerStop",                                        "", CONV_PROFILER, API_RUNTIME, SEC::PROFILER, HIP_DEPRECATED};

  // 37. Data types used by CUDA Runtime
  // NOTE: in a separate file

  // 38. Execution Control [REMOVED]
  // NOTE: Removed in CUDA 10.1
  // no analogue
  m["cudaConfigureCall"]                                       = {"hipConfigureCall",                                       "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION_REMOVED, CUDA_REMOVED};
  // no analogue
  // NOTE: Not equal to cuLaunch due to different signatures
  m["cudaLaunch"]                                              = {"hipLaunchByPtr",                                         "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION_REMOVED, CUDA_REMOVED};
  // no analogue
  m["cudaSetupArgument"]                                       = {"hipSetupArgument",                                       "", CONV_EXECUTION, API_RUNTIME, SEC::EXECUTION_REMOVED, CUDA_REMOVED};

  // 39. Texture Reference Management [REMOVED]
  // NOTE: Removed in CUDA 12.0
  // no analogue
  m["cudaBindTexture"]                                         = {"hipBindTexture",                                         "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaBindTexture2D"]                                       = {"hipBindTexture2D",                                       "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaBindTextureToArray"]                                  = {"hipBindTextureToArray",                                  "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaBindTextureToMipmappedArray"]                         = {"hipBindTextureToMipmappedArray",                         "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaGetTextureAlignmentOffset"]                           = {"hipGetTextureAlignmentOffset",                           "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaGetTextureReference"]                                 = {"hipGetTextureReference",                                 "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaUnbindTexture"]                                       = {"hipUnbindTexture",                                       "", CONV_TEXTURE, API_RUNTIME, SEC::TEXTURE_REMOVED, HIP_DEPRECATED | CUDA_REMOVED};

  // 40. Surface Reference Management [REMOVED]
  // NOTE: Removed in CUDA 12.0
  // no analogue
  m["cudaBindSurfaceToArray"]                                  = {"hipBindSurfaceToArray",                                  "", CONV_SURFACE, API_RUNTIME, SEC::SURFACE_REMOVED, HIP_UNSUPPORTED | CUDA_REMOVED};
  // no analogue
  m["cudaGetSurfaceReference"]                                 = {"hipGetSurfaceReference",                                 "", CONV_SURFACE, API_RUNTIME, SEC::SURFACE_REMOVED, HIP_UNSUPPORTED | CUDA_REMOVED};

  // 41. Profiler Control [REMOVED]
  // cuProfilerInitialize
  m["cudaProfilerInitialize"]                                  = {"hipProfilerInitialize",                                  "", CONV_PROFILER, API_RUNTIME, SEC::PROFILER_REMOVED, HIP_UNSUPPORTED | CUDA_REMOVED};

  // 42. Thread Management [REMOVED]
  // no analogue
  m["cudaThreadExit"]                                          = {"hipDeviceReset",                                         "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaThreadGetCacheConfig"]                                = {"hipDeviceGetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaThreadGetLimit"]                                      = {"hipThreadGetLimit",                                      "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaThreadSetCacheConfig"]                                = {"hipDeviceSetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, CUDA_DEPRECATED | CUDA_REMOVED};
  // no analogue
  m["cudaThreadSetLimit"]                                      = {"hipThreadSetLimit",                                      "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // cuCtxSynchronize
  m["cudaThreadSynchronize"]                                   = {"hipDeviceSynchronize",                                   "", CONV_THREAD, API_RUNTIME, SEC::THREAD_REMOVED, CUDA_DEPRECATED | CUDA_REMOVED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RUNTIME_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef,  cudaAPIversions> m;

  m["cudaDeviceGetNvSciSyncAttributes"]                        = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetP2PAttribute"]                               = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaCtxResetPersistingL2Cache"]                           = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaThreadExit"]                                          = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaThreadGetCacheConfig"]                                = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaThreadGetLimit"]                                      = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaThreadSetCacheConfig"]                                = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaThreadSetLimit"]                                      = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaThreadSynchronize"]                                   = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaStreamBeginCapture"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamCopyAttributes"]                                = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamEndCapture"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamGetAttribute"]                                  = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamSetAttribute"]                                  = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamIsCapturing"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamGetCaptureInfo"]                                = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaThreadExchangeStreamCaptureMode"]                     = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaDestroyExternalMemory"]                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaDestroyExternalSemaphore"]                            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryGetMappedBuffer"]                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryGetMappedMipmappedArray"]               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaImportExternalMemory"]                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaImportExternalSemaphore"]                             = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaSignalExternalSemaphoresAsync"]                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaWaitExternalSemaphoresAsync"]                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaFuncSetAttribute"]                                    = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaLaunchCooperativeKernel"]                             = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaLaunchCooperativeKernelMultiDevice"]                  = {CUDA_90,  CUDA_113, CUDA_130};
  m["cudaLaunchHostFunc"]                                      = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaSetDoubleForDevice"]                                  = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaSetDoubleForHost"]                                    = {CUDA_0,   CUDA_100, CUDA_130};
  m["cudaOccupancyAvailableDynamicSMemPerBlock"]               = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaMemAdvise"]                                           = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemPrefetchAsync"]                                    = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeGetAttribute"]                                = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeGetAttributes"]                               = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemcpyArrayToArray"]                                  = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaMemcpyFromArray"]                                     = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaMemcpyFromArrayAsync"]                                = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaMemcpyToArray"]                                       = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaMemcpyToArrayAsync"]                                  = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaGLMapBufferObject"]                                   = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLMapBufferObjectAsync"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLRegisterBufferObject"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLSetBufferObjectMapFlags"]                           = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLSetGLDevice"]                                       = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLUnmapBufferObject"]                                 = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLUnmapBufferObjectAsync"]                            = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaGLUnregisterBufferObject"]                            = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9MapResources"]                                    = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceGetMappedArray"]                          = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceGetMappedPitch"]                          = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceGetMappedPointer"]                        = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceGetMappedSize"]                           = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceGetSurfaceDimensions"]                    = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9ResourceSetMapFlags"]                             = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9UnmapResources"]                                  = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D9UnregisterResource"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10GetDirect3DDevice"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10MapResources"]                                   = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10RegisterResource"]                               = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceGetMappedArray"]                         = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceGetMappedPitch"]                         = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceGetMappedPointer"]                       = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceGetMappedSize"]                          = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceGetSurfaceDimensions"]                   = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10ResourceSetMapFlags"]                            = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10SetDirect3DDevice"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10UnmapResources"]                                 = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D10UnregisterResource"]                             = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D11GetDirect3DDevice"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaD3D11SetDirect3DDevice"]                              = {CUDA_0,   CUDA_100, CUDA_0  };
  m["cudaEGLStreamConsumerAcquireFrame"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamConsumerConnect"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamConsumerConnectWithFlags"]                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamConsumerDisconnect"]                         = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamConsumerReleaseFrame"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamProducerConnect"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamProducerDisconnect"]                         = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamProducerPresentFrame"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEGLStreamProducerReturnFrame"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEventCreateFromEGLSync"]                              = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaGraphicsEGLRegisterImage"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaGraphicsResourceGetMappedEglFrame"]                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaBindTexture"]                                         = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaBindTexture2D"]                                       = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaBindTextureToArray"]                                  = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaBindTextureToMipmappedArray"]                         = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaGetTextureAlignmentOffset"]                           = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaGetTextureReference"]                                 = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaUnbindTexture"]                                       = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaBindSurfaceToArray"]                                  = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaGetSurfaceReference"]                                 = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaGraphAddChildGraphNode"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddDependencies"]                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddEmptyNode"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddHostNode"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddKernelNode"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemcpyNode"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemsetNode"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphChildGraphNodeGetGraph"]                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphClone"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphCreate"]                                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphDestroy"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphDestroyNode"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphExecDestroy"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphGetEdges"]                                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphGetNodes"]                                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphGetRootNodes"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphHostNodeGetParams"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphHostNodeSetParams"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiate"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeCopyAttributes"]                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeGetAttribute"]                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeSetAttribute"]                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphExecKernelNodeSetParams"]                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphExecMemcpyNodeSetParams"]                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphExecMemsetNodeSetParams"]                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphExecHostNodeSetParams"]                          = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdate"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeGetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeSetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphLaunch"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphMemcpyNodeGetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphMemcpyNodeSetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphMemsetNodeGetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphMemsetNodeSetParams"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeFindInClone"]                                = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetDependencies"]                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetDependentNodes"]                          = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetType"]                                    = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGraphRemoveDependencies"]                             = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaGetFuncBySymbol"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaProfilerInitialize"]                                  = {CUDA_0,   CUDA_110, CUDA_120};
  m["cudaConfigureCall"]                                       = {CUDA_0,   CUDA_0,   CUDA_101};
  m["cudaLaunch"]                                              = {CUDA_0,   CUDA_0,   CUDA_101};
  m["cudaSetupArgument"]                                       = {CUDA_0,   CUDA_0,   CUDA_101};
  m["cudaDeviceGetTexture1DLinearMaxWidth"]                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEventRecordWithFlags"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaArrayGetSparseProperties"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemcpyNodeToSymbol"]                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemcpyNodeFromSymbol"]                        = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemcpyNode1D"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphMemcpyNodeSetParamsToSymbol"]                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphMemcpyNodeSetParamsFromSymbol"]                  = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphMemcpyNodeSetParams1D"]                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphAddEventRecordNode"]                             = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphEventRecordNodeGetEvent"]                        = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphEventRecordNodeSetEvent"]                        = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphAddEventWaitNode"]                               = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphEventWaitNodeGetEvent"]                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphEventWaitNodeSetEvent"]                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecMemcpyNodeSetParamsToSymbol"]                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecMemcpyNodeSetParamsFromSymbol"]              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecMemcpyNodeSetParams1D"]                      = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecChildGraphNodeSetParams"]                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecEventRecordNodeSetEvent"]                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphExecEventWaitNodeSetEvent"]                      = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphUpload"]                                         = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaMallocAsync"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaFreeAsync"]                                           = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMallocFromPoolAsync"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetDefaultMemPool"]                             = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaDeviceSetMemPool"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetMemPool"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaArrayGetPlane"]                                       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolTrimTo"]                                       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolSetAttribute"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolGetAttribute"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolSetAccess"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolGetAccess"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolCreate"]                                       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolDestroy"]                                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolExportToShareableHandle"]                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolImportFromShareableHandle"]                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolExportPointer"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolImportPointer"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphAddExternalSemaphoresSignalNode"]                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExternalSemaphoresSignalNodeGetParams"]          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExternalSemaphoresSignalNodeSetParams"]          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphAddExternalSemaphoresWaitNode"]                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExternalSemaphoresWaitNodeGetParams"]            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExternalSemaphoresWaitNodeSetParams"]            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExecExternalSemaphoresSignalNodeSetParams"]      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExecExternalSemaphoresWaitNodeSetParams"]        = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaDeviceFlushGPUDirectRDMAWrites"]                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotPrint"]                                  = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectCreate"]                                    = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectRetain"]                                    = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectRelease"]                                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphRetainUserObject"]                               = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphReleaseUserObject"]                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGetDriverEntryPoint"]                                 = {CUDA_113, CUDA_130, CUDA_0  };
  m["cudaGraphAddMemAllocNode"]                                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAllocNodeGetParams"]                          = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphAddMemFreeNode"]                                 = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemFreeNodeGetParams"]                           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaDeviceGraphMemTrim"]                                  = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetGraphMemAttribute"]                          = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaDeviceSetGraphMemAttribute"]                          = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateWithFlags"]                           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaArrayGetMemoryRequirements"]                          = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeSetEnabled"]                                 = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetEnabled"]                                 = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaLaunchKernelExC"]                                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaOccupancyMaxPotentialClusterSize"]                    = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaOccupancyMaxActiveClusters"]                          = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaCreateTextureObject_v2"]                              = {CUDA_118, CUDA_0,   CUDA_120};
  m["cudaGetTextureObjectTextureDesc_v2"]                      = {CUDA_118, CUDA_0,   CUDA_120};
  m["cudaInitDevice"]                                          = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaStreamGetId"]                                         = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateWithParams"]                          = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphExecGetFlags"]                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGetKernel"]                                           = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaMemPrefetchAsync_v2"]                                 = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemAdvise_v2"]                                        = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaGraphAddNode"]                                        = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeSetParams"]                                  = {CUDA_122, CUDA_0,   CUDA_0  };  
  m["cudaGraphExecNodeSetParams"]                              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaFuncGetName"]                                         = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaStreamBeginCaptureToGraph"]                           = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaStreamGetCaptureInfo_v3"]                             = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaStreamUpdateCaptureDependencies"]                     = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaStreamUpdateCaptureDependencies_v2"]                  = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphGetEdges_v2"]                                    = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetDependencies_v2"]                         = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetDependentNodes_v2"]                       = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphAddDependencies_v2"]                             = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphRemoveDependencies_v2"]                          = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphAddNode_v2"]                                     = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphConditionalHandleCreate"]                        = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetSharedMemConfig"]                            = {CUDA_0,   CUDA_124, CUDA_0  };
  m["cudaDeviceSetSharedMemConfig"]                            = {CUDA_0,   CUDA_124, CUDA_0  };
  m["cudaFuncSetSharedMemConfig"]                              = {CUDA_0,   CUDA_124, CUDA_0  };
  m["cudaDeviceRegisterAsyncNotification"]                     = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaDeviceUnregisterAsyncNotification"]                   = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaFuncGetParamInfo"]                                    = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGetDriverEntryPointByVersion"]                        = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cudaStreamGetDevice"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaEventElapsedTime_v2"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyBatchAsync"]                                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpy3DBatchAsync"]                                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryLoadData"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryLoadFromFile"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryUnload"]                                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryGetKernel"]                                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryGetGlobal"]                                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryGetManaged"]                                   = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryGetUnifiedFunction"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaKernelSetAttributeForDevice"]                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryGetKernelCount"]                               = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryEnumerateKernels"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetHostAtomicCapabilities"]                     = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetP2PAtomicCapabilities"]                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemDiscardBatchAsync"]                                = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemDiscardAndPrefetchBatchAsync"]                     = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemPrefetchBatchAsync"]                               = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemGetDefaultMemPool"]                                = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemGetMemPool"]                                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemSetMemPool"]                                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsRegisterCallback"]                                = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsUnregisterCallback"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsCurrent"]                                         = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsDumpToFile"]                                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsDumpToMemory"]                                    = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetContainingGraph"]                         = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetLocalId"]                                 = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetToolsId"]                                 = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGraphGetId"]                                          = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGraphExecGetId"]                                      = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGraphConditionalHandleCreate_v2"]                     = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetDevResource"]                                = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceSplitByCount"]                           = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceSplit"]                                  = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceGenerateDesc"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaGreenCtxCreate"]                                      = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxDestroy"]                                 = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxGetDevResource"]                          = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxGetDevice"]                               = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxGetId"]                                   = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxSynchronize"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaStreamGetDevResource"]                                = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxRecordEvent"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxWaitEvent"]                               = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDeviceGetExecutionCtx"]                               = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionCtxStreamCreate"]                            = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaFuncGetParamCount"]                                   = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaLaunchHostFunc_v2"]                                   = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemcpyWithAttributesAsync"]                           = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemcpy3DWithAttributesAsync"]                         = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeGetParams"]                                  = {CUDA_132, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_RUNTIME_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef,  hipAPIversions> m;

  m["hipHostAlloc"]                                            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChooseDevice"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetAttribute"]                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetByPCIBusId"]                                  = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetCacheConfig"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetLimit"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetP2PAttribute"]                                = {HIP_3080, HIP_0,    HIP_0   };
  m["hipDeviceGetPCIBusId"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetSharedMemConfig"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetStreamPriorityRange"]                         = {HIP_2000, HIP_0,    HIP_0   };
  m["hipDeviceReset"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceSetCacheConfig"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceSetSharedMemConfig"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceSynchronize"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetDevice"]                                            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetDeviceCount"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetDeviceFlags"]                                       = {HIP_3060, HIP_0,    HIP_0   };
  m["hipGetDeviceProperties"]                                  = {HIP_1060, HIP_0,    HIP_0   };
  m["hipIpcCloseMemHandle"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipIpcGetEventHandle"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipIpcGetMemHandle"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipIpcOpenEventHandle"]                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipIpcOpenMemHandle"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipSetDevice"]                                            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipSetDeviceFlags"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetErrorName"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetErrorString"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetLastError"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipPeekAtLastError"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamAddCallback"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamAttachMemAsync"]                                 = {HIP_3070, HIP_0,    HIP_0   };
  m["hipStreamCreate"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamCreateWithFlags"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamCreateWithPriority"]                             = {HIP_2000, HIP_0,    HIP_0   };
  m["hipStreamDestroy"]                                        = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamGetFlags"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamGetPriority"]                                    = {HIP_2000, HIP_0,    HIP_0   };
  m["hipStreamQuery"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamSynchronize"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipStreamWaitEvent"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventCreate"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventCreateWithFlags"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventDestroy"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventElapsedTime"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventQuery"]                                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventRecord"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipEventSynchronize"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipFuncGetAttributes"]                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["hipFuncSetAttribute"]                                     = {HIP_3090, HIP_0,    HIP_0   };
  m["hipFuncSetCacheConfig"]                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipFuncSetSharedMemConfig"]                               = {HIP_3090, HIP_0,    HIP_0   };
  m["hipLaunchCooperativeKernel"]                              = {HIP_2060, HIP_0,    HIP_0   };
  m["hipLaunchCooperativeKernelMultiDevice"]                   = {HIP_2060, HIP_0,    HIP_0   };
  m["hipLaunchKernel"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipOccupancyMaxActiveBlocksPerMultiprocessor"]            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]   = {HIP_2060, HIP_0,    HIP_0   };
  m["hipOccupancyMaxPotentialBlockSize"]                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipOccupancyMaxPotentialBlockSizeWithFlags"]              = {HIP_3050, HIP_0,    HIP_0   };
  m["hipFree"]                                                 = {HIP_1050, HIP_0,    HIP_0   };
  m["hipFreeArray"]                                            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipHostFree"]                                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipFreeMipmappedArray"]                                   = {HIP_3050, HIP_0,    HIP_0   };
  m["hipGetMipmappedArrayLevel"]                               = {HIP_3050, HIP_0,    HIP_0   };
  m["hipGetSymbolAddress"]                                     = {HIP_2000, HIP_0,    HIP_0   };
  m["hipGetSymbolSize"]                                        = {HIP_2000, HIP_0,    HIP_0   };
  m["hipHostMalloc"]                                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipHostGetFlags"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipHostRegister"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipHostUnregister"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMalloc"]                                               = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMalloc3D"]                                             = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMalloc3DArray"]                                        = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMallocArray"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipHostGetDevicePointer"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMallocManaged"]                                        = {HIP_2050, HIP_0,    HIP_0   };
  m["hipMallocMipmappedArray"]                                 = {HIP_3050, HIP_0,    HIP_0   };
  m["hipMallocPitch"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemAdvise"]                                            = {HIP_3070, HIP_0,    HIP_0   };
  m["hipMemcpy"]                                               = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpy2D"]                                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpy2DAsync"]                                        = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpy2DFromArray"]                                    = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMemcpy2DFromArrayAsync"]                               = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMemcpy2DToArray"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpy3D"]                                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpy3DAsync"]                                        = {HIP_2080, HIP_0,    HIP_0   };
  m["hipMemcpyAsync"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyFromSymbol"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyFromSymbolAsync"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyPeer"]                                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyPeerAsync"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyToSymbol"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyToSymbolAsync"]                                  = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemGetInfo"]                                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemPrefetchAsync"]                                     = {HIP_3070, HIP_0,    HIP_0   };
  m["hipMemRangeGetAttribute"]                                 = {HIP_3070, HIP_0,    HIP_0   };
  m["hipMemRangeGetAttributes"]                                = {HIP_3070, HIP_0,    HIP_0   };
  m["hipMemset"]                                               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemset2D"]                                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMemset2DAsync"]                                        = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemset3D"]                                             = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemset3DAsync"]                                        = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemsetAsync"]                                          = {HIP_1060, HIP_0,    HIP_0   };
  m["make_hipExtent"]                                          = {HIP_1070, HIP_0,    HIP_0   };
  m["make_hipPitchedPtr"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["make_hipPos"]                                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMemcpyFromArray"]                                      = {HIP_1090, HIP_3080, HIP_0   };
  m["hipMemcpyToArray"]                                        = {HIP_1060, HIP_3080, HIP_0   };
  m["hipPointerGetAttributes"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceCanAccessPeer"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["hipDeviceDisablePeerAccess"]                              = {HIP_1090, HIP_0,    HIP_0   };
  m["hipDeviceEnablePeerAccess"]                               = {HIP_1090, HIP_0,    HIP_0   };
  m["hipBindTexture"]                                          = {HIP_1060, HIP_3080, HIP_0   };
  m["hipBindTexture2D"]                                        = {HIP_1070, HIP_3080, HIP_0   };
  m["hipBindTextureToArray"]                                   = {HIP_1060, HIP_3080, HIP_0   };
  m["hipBindTextureToMipmappedArray"]                          = {HIP_1070, HIP_5070, HIP_0   };
  m["hipCreateChannelDesc"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipGetChannelDesc"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipGetTextureAlignmentOffset"]                            = {HIP_1090, HIP_3080, HIP_0   };
  m["hipGetTextureReference"]                                  = {HIP_1070, HIP_5030, HIP_0   };
  m["hipUnbindTexture"]                                        = {HIP_1060, HIP_3080, HIP_0   };
  m["hipCreateTextureObject"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipDestroyTextureObject"]                                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipGetTextureObjectResourceDesc"]                         = {HIP_1070, HIP_0,    HIP_0   };
  m["hipGetTextureObjectResourceViewDesc"]                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipGetTextureObjectTextureDesc"]                          = {HIP_1070, HIP_0,    HIP_0   };
  m["hipCreateSurfaceObject"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["hipDestroySurfaceObject"]                                 = {HIP_1090, HIP_0,    HIP_0   };
  m["hipDriverGetVersion"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipRuntimeGetVersion"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipProfilerStart"]                                        = {HIP_1060, HIP_3000, HIP_0   };
  m["hipProfilerStop"]                                         = {HIP_1060, HIP_3000, HIP_0   };
  m["hipConfigureCall"]                                        = {HIP_1090, HIP_0,    HIP_0   };
  m["hipLaunchByPtr"]                                          = {HIP_1090, HIP_0,    HIP_0   };
  m["hipSetupArgument"]                                        = {HIP_1090, HIP_0,    HIP_0   };
  m["hipImportExternalSemaphore"]                              = {HIP_4040, HIP_0,    HIP_0   };
  m["hipSignalExternalSemaphoresAsync"]                        = {HIP_4040, HIP_0,    HIP_0   };
  m["hipWaitExternalSemaphoresAsync"]                          = {HIP_4040, HIP_0,    HIP_0   };
  m["hipDestroyExternalSemaphore"]                             = {HIP_4040, HIP_0,    HIP_0   };
  m["hipImportExternalMemory"]                                 = {HIP_4030, HIP_0,    HIP_0   };
  m["hipExternalMemoryGetMappedBuffer"]                        = {HIP_4030, HIP_0,    HIP_0   };
  m["hipDestroyExternalMemory"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipMemcpy2DToArrayAsync"]                                 = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamBeginCapture"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamEndCapture"]                                     = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphCreate"]                                          = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphDestroy"]                                         = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecDestroy"]                                     = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphInstantiate"]                                     = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphLaunch"]                                          = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphAddKernelNode"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphAddMemcpyNode"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphAddMemsetNode"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphAddMemcpyNode1D"]                                 = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphGetNodes"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphGetRootNodes"]                                    = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphKernelNodeGetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphKernelNodeSetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphMemcpyNodeGetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphMemcpyNodeSetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphMemsetNodeGetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphMemsetNodeSetParams"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphExecKernelNodeSetParams"]                         = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphAddDependencies"]                                 = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphAddEmptyNode"]                                    = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGLGetDevices"]                                         = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphicsGLRegisterBuffer"]                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphicsMapResources"]                                 = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphicsResourceGetMappedPointer"]                     = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphicsUnmapResources"]                               = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphicsUnregisterResource"]                           = {HIP_4050, HIP_0,    HIP_0   };
  m["hipGraphRemoveDependencies"]                              = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphGetEdges"]                                        = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphNodeGetDependencies"]                             = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphNodeGetDependentNodes"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphNodeGetType"]                                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphDestroyNode"]                                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphNodeFindInClone"]                                 = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphInstantiateWithFlags"]                            = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecUpdate"]                                      = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecMemcpyNodeSetParams"]                         = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphMemcpyNodeSetParams1D"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecMemcpyNodeSetParams1D"]                       = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddMemcpyNodeFromSymbol"]                         = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphMemcpyNodeSetParamsFromSymbol"]                   = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecMemcpyNodeSetParamsFromSymbol"]               = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddMemcpyNodeToSymbol"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphMemcpyNodeSetParamsToSymbol"]                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecMemcpyNodeSetParamsToSymbol"]                 = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecMemsetNodeSetParams"]                         = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddHostNode"]                                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphHostNodeGetParams"]                               = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphHostNodeSetParams"]                               = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecHostNodeSetParams"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddChildGraphNode"]                               = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphChildGraphNodeGetGraph"]                          = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecChildGraphNodeSetParams"]                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddEventRecordNode"]                              = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphEventRecordNodeGetEvent"]                         = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphEventRecordNodeSetEvent"]                         = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecEventRecordNodeSetEvent"]                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphAddEventWaitNode"]                                = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphEventWaitNodeGetEvent"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphEventWaitNodeSetEvent"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphExecEventWaitNodeSetEvent"]                       = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphClone"]                                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipDeviceGetDefaultMemPool"]                              = {HIP_5020, HIP_0,    HIP_0   };
  m["hipDeviceSetMemPool"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipDeviceGetMemPool"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMallocAsync"]                                          = {HIP_5020, HIP_0,    HIP_0   };
  m["hipFreeAsync"]                                            = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolTrimTo"]                                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolSetAttribute"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolGetAttribute"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolSetAccess"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolGetAccess"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolCreate"]                                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolDestroy"]                                       = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMallocFromPoolAsync"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolExportToShareableHandle"]                       = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolImportFromShareableHandle"]                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolExportPointer"]                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolImportPointer"]                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipLaunchHostFunc"]                                       = {HIP_5020, HIP_0,    HIP_0   };
  m["hipThreadExchangeStreamCaptureMode"]                      = {HIP_5020, HIP_0,    HIP_0   };
  m["hipGraphKernelNodeSetAttribute"]                          = {HIP_5020, HIP_0,    HIP_0   };
  m["hipGraphKernelNodeGetAttribute"]                          = {HIP_5020, HIP_0,    HIP_0   };
  m["hipDeviceSetLimit"]                                       = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphUpload"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["hipDeviceGetGraphMemAttribute"]                           = {HIP_5030, HIP_0,    HIP_0   };
  m["hipDeviceSetGraphMemAttribute"]                           = {HIP_5030, HIP_0,    HIP_0   };
  m["hipDeviceGraphMemTrim"]                                   = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectCreate"]                                     = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectRelease"]                                    = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectRetain"]                                     = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphRetainUserObject"]                                = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphReleaseUserObject"]                               = {HIP_5030, HIP_0,    HIP_0   };
  m["hipOccupancyMaxPotentialBlockSizeVariableSMem"]           = {HIP_5050, HIP_0,    HIP_0   };
  m["hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags"]  = {HIP_5050, HIP_0,    HIP_0   };
  m["hipArrayGetInfo"]                                         = {HIP_5060, HIP_0,    HIP_0   };
  m["hipGraphAddExternalSemaphoresWaitNode"]                   = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphAddExternalSemaphoresSignalNode"]                 = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExternalSemaphoresSignalNodeSetParams"]           = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExternalSemaphoresSignalNodeGetParams"]           = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExternalSemaphoresWaitNodeGetParams"]             = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExternalSemaphoresWaitNodeSetParams"]             = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExecExternalSemaphoresSignalNodeSetParams"]       = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphExecExternalSemaphoresWaitNodeSetParams"]         = {HIP_5070, HIP_0,    HIP_0   };
  m["hipGraphInstantiateWithParams"]                           = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphAddNode"]                                         = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGetProcAddress"]                                       = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGetFuncBySymbol"]                                      = {HIP_6020, HIP_0,    HIP_0   };
  m["hipStreamBeginCaptureToGraph"]                            = {HIP_6020, HIP_0,    HIP_0   };
  m["hipSetValidDevices"]                                      = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpy2DArrayToArray"]                                 = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphExecGetFlags"]                                    = {HIP_6030, HIP_0,    HIP_0   };
  m["hipGraphNodeSetParams"]                                   = {HIP_6030, HIP_0,    HIP_0   };
  m["hipGraphExecNodeSetParams"]                               = {HIP_6030, HIP_0,    HIP_0   };
  m["hipLaunchKernelExC"]                                      = {HIP_7000, HIP_0,    HIP_0   };
  m["hipDeviceGetTexture1DLinearMaxWidth"]                     = {HIP_6040, HIP_0,    HIP_0   };
  m["hipStreamGetId"]                                          = {HIP_7010, HIP_0,    HIP_0   };
  m["hipStreamSetAttribute"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipStreamGetAttribute"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyBatchAsync"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DBatchAsync"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DPeer"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DPeerAsync"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryLoadData"]                                      = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryLoadFromFile"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryUnload"]                                        = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryGetKernel"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryGetKernelCount"]                                = {HIP_7010, HIP_0,    HIP_0   };
  m["hipGetDriverEntryPoint"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["hipStreamCopyAttributes"]                                 = {HIP_7020, HIP_0,    HIP_0   };
  m["hipOccupancyAvailableDynamicSMemPerBlock"]                = {HIP_7020, HIP_0,    HIP_0   };
  m["hipLibraryEnumerateKernels"]                              = {HIP_7020, HIP_0,    HIP_0   };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RUNTIME_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef,  cudaAPIChangedVersions> m;

  m["cudaGetDriverEntryPoint"]                                 = {CUDA_120};
  m["cudaStreamGetCaptureInfo"]                                = {CUDA_130};
  m["cudaStreamUpdateCaptureDependencies"]                     = {CUDA_130};
  m["cudaMemcpyBatchAsync"]                                    = {CUDA_130};
  m["cudaMemcpy3DBatchAsync"]                                  = {CUDA_130};
  m["cudaMemPrefetchAsync"]                                    = {CUDA_130};
  m["cudaMemAdvise"]                                           = {CUDA_130};
  m["cudaGraphGetEdges"]                                       = {CUDA_130};
  m["cudaGraphNodeGetDependencies"]                            = {CUDA_130};
  m["cudaGraphNodeGetDependentNodes"]                          = {CUDA_130};
  m["cudaGraphAddDependencies"]                                = {CUDA_130};
  m["cudaGraphRemoveDependencies"]                             = {CUDA_130};
  m["cudaGraphAddNode"]                                        = {CUDA_130};
  // [IMP] Changed semantics: Dst <-> Src
  m["cudaGraphKernelNodeCopyAttributes"]                       = {CUDA_130};

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_RUNTIME_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef,  hipAPIChangedVersions> m;

  m["hipDeviceGetTexture1DLinearMaxWidth"]                     = {HIP_7000};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIUnsupportedVersions> CUDA_RUNTIME_FUNCTION_UNSUPPORTED_VER_MAP = [] {
  std::map<llvm::StringRef,  cudaAPIUnsupportedVersions> m;

  m["cudaStreamGetCaptureInfo"]                                = {CUDA_130};
  m["cudaStreamUpdateCaptureDependencies"]                     = {CUDA_130};
  m["cudaMemcpyBatchAsync"]                                    = {CUDA_130};
  m["cudaMemcpy3DBatchAsync"]                                  = {CUDA_130};
  m["cudaMemAdvise"]                                           = {CUDA_130};
  m["cudaMemPrefetchAsync"]                                    = {CUDA_130};
  m["cudaGraphAddDependencies"]                                = {CUDA_130};
  m["cudaGraphGetEdges"]                                       = {CUDA_130};
  m["cudaGraphNodeGetDependencies"]                            = {CUDA_130};
  m["cudaGraphNodeGetDependentNodes"]                          = {CUDA_130};
  m["cudaGraphRemoveDependencies"]                             = {CUDA_130};
  m["cudaGraphAddNode"]                                        = {CUDA_130};
  m["cudaGetDriverEntryPoint"]                                 = {CUDA_113, CUDA_114, CUDA_115, CUDA_116, CUDA_117, CUDA_118};
  m["cudaGraphCreate"]                                         = {CUDA_132};

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_RUNTIME_API_SECTION_MAP = [] {
  std::map<unsigned int,  llvm::StringRef> m;

  m[SEC::DEVICE]                                               = "Device Management";
  m[SEC::DEVICE_DEPRECATED]                                    = "Device Management [DEPRECATED]";
  m[SEC::ERROR]                                                = "Error Handling";
  m[SEC::STREAM]                                               = "Stream Management";
  m[SEC::EVENT]                                                = "Event Management";
  m[SEC::EXTERNAL_RES]                                         = "External Resource Interoperability";
  m[SEC::EXECUTION]                                            = "Execution Control";
  m[SEC::EXECUTION_DEPRECATED]                                 = "Execution Control [DEPRECATED]";
  m[SEC::OCCUPANCY]                                            = "Occupancy";
  m[SEC::MEMORY]                                               = "Memory Management";
  m[SEC::MEMORY_DEPRECATED]                                    = "Memory Management [DEPRECATED]";
  m[SEC::ORDERED_MEMORY]                                       = "Stream Ordered Memory Allocator";
  m[SEC::UNIFIED]                                              = "Unified Addressing";
  m[SEC::PEER]                                                 = "Peer Device Memory Access";
  m[SEC::OPENGL]                                               = "OpenGL Interoperability";
  m[SEC::OPENGL_DEPRECATED]                                    = "OpenGL Interoperability [DEPRECATED]";
  m[SEC::D3D9]                                                 = "Direct3D 9 Interoperability";
  m[SEC::D3D9_DEPRECATED]                                      = "Direct3D 9 Interoperability [DEPRECATED]";
  m[SEC::D3D10]                                                = "Direct3D 10 Interoperability";
  m[SEC::D3D10_DEPRECATED]                                     = "Direct3D 10 Interoperability [DEPRECATED]";
  m[SEC::D3D11]                                                = "Direct3D 11 Interoperability";
  m[SEC::D3D11_DEPRECATED]                                     = "Direct3D 11 Interoperability [DEPRECATED]";
  m[SEC::VDPAU]                                                = "VDPAU Interoperability";
  m[SEC::EGL]                                                  = "EGL Interoperability";
  m[SEC::GRAPHICS]                                             = "Graphics Interoperability";
  m[SEC::TEXTURE]                                              = "Texture Object Management";
  m[SEC::SURFACE]                                              = "Surface Object Management";
  m[SEC::VERSION]                                              = "Version Management";
  m[SEC::ERROR_LOG]                                            = "Error Log Management";
  m[SEC::GRAPH]                                                = "Graph Management";
  m[SEC::DRIVER_ENTRY_POINT]                                   = "Driver Entry Point Access";
  m[SEC::LIBRARY]                                              = "Library Management";
  m[SEC::EXECUTION_CONTEXT_MANGEMENT]                          = "Execution Context Management";
  m[SEC::CPP]                                                  = "C++ API Routines";
  m[SEC::DRIVER_INTERACT]                                      = "Interactions with the CUDA Driver API";
  m[SEC::PROFILER]                                             = "Profiler Control";
  m[SEC::DATA_TYPES]                                           = "Data types used by CUDA Runtime";
  m[SEC::EXECUTION_REMOVED]                                    = "Execution Control [REMOVED]";
  m[SEC::TEXTURE_REMOVED]                                      = "Texture Reference Management [REMOVED]";
  m[SEC::SURFACE_REMOVED]                                      = "Surface Reference Management [REMOVED]";
  m[SEC::PROFILER_REMOVED]                                     = "Profiler Control [REMOVED]";
  m[SEC::THREAD_REMOVED]                                       = "Thread Management [REMOVED]";

  return m;
}();
