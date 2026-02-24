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

using SEC = driver::CUDA_DRIVER_API_SECTIONS;

// Map of all CUDA Driver API functions
const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  // 2. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  m["cuGetErrorName"]                                                    = {"hipDrvGetErrorName",                                          "", CONV_ERROR, API_DRIVER, SEC::ERROR};
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  m["cuGetErrorString"]                                                  = {"hipDrvGetErrorString",                                        "", CONV_ERROR, API_DRIVER, SEC::ERROR};

  // 3. Initialization
  // no analogue
  m["cuInit"]                                                            = {"hipInit",                                                     "", CONV_INIT, API_DRIVER, SEC::INIT};

  // 4. Version Management
  // cudaDriverGetVersion
  m["cuDriverGetVersion"]                                                = {"hipDriverGetVersion",                                         "", CONV_VERSION, API_DRIVER, SEC::VERSION};

  // 5. Device Management
  // cudaGetDevice
  // NOTE: cudaGetDevice has additional attr: int ordinal
  m["cuDeviceGet"]                                                       = {"hipDeviceGet",                                                "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // cudaDeviceGetAttribute
  m["cuDeviceGetAttribute"]                                              = {"hipDeviceGetAttribute",                                       "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // cudaGetDeviceCount
  m["cuDeviceGetCount"]                                                  = {"hipGetDeviceCount",                                           "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // no analogue
  m["cuDeviceGetLuid"]                                                   = {"hipDeviceGetLuid",                                            "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};
  // no analogue
  m["cuDeviceGetName"]                                                   = {"hipDeviceGetName",                                            "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // cudaDeviceGetNvSciSyncAttributes
  m["cuDeviceGetNvSciSyncAttributes"]                                    = {"hipDeviceGetNvSciSyncAttributes",                             "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};
  // no analogue
  m["cuDeviceGetUuid"]                                                   = {"hipDeviceGetUuid",                                            "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // no analogue
  m["cuDeviceGetUuid_v2"]                                                = {"hipDeviceGetUuid",                                            "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // no analogue
  m["cuDeviceTotalMem"]                                                  = {"hipDeviceTotalMem",                                           "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  m["cuDeviceTotalMem_v2"]                                               = {"hipDeviceTotalMem",                                           "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // NOTE: incompatible with cudaDeviceGetTexture1DLinearMaxWidth
  m["cuDeviceGetTexture1DLinearMaxWidth"]                                = {"hipDeviceGetTexture1DLinearMaxWidth",                         "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};
  // cudaDeviceSetMemPool
  m["cuDeviceSetMemPool"]                                                = {"hipDeviceSetMemPool",                                         "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // cudaDeviceGetMemPool
  m["cuDeviceGetMemPool"]                                                = {"hipDeviceGetMemPool",                                         "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  // cudaDeviceGetDefaultMemPool
  m["cuDeviceGetDefaultMemPool"]                                         = {"hipDeviceGetDefaultMemPool",                                  "", CONV_DEVICE, API_DRIVER, SEC::DEVICE};
  //
  m["cuDeviceGetExecAffinitySupport"]                                    = {"hipDeviceGetExecAffinitySupport",                             "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};
  // cudaDeviceFlushGPUDirectRDMAWrites
  m["cuFlushGPUDirectRDMAWrites"]                                        = {"hipDeviceFlushGPUDirectRDMAWrites",                           "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};
  // cudaDeviceGetHostAtomicCapabilities
  m["cuDeviceGetHostAtomicCapabilities"]                                 = {"hipDeviceGetHostAtomicCapabilities",                          "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED};

  // 6. Device Management [DEPRECATED]
  //
  m["cuDeviceComputeCapability"]                                         = {"hipDeviceComputeCapability",                                  "", CONV_DEVICE, API_DRIVER, SEC::DEVICE_DEPRECATED, CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaGetDeviceProperties due to different attributes: cudaDeviceProp and CUdevprop
  m["cuDeviceGetProperties"]                                             = {"hipGetDeviceProperties_",                                     "", CONV_DEVICE, API_DRIVER, SEC::DEVICE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 7. Primary Context Management
  // no analogues
  m["cuDevicePrimaryCtxGetState"]                                        = {"hipDevicePrimaryCtxGetState",                                 "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxRelease"]                                         = {"hipDevicePrimaryCtxRelease",                                  "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxRelease_v2"]                                      = {"hipDevicePrimaryCtxRelease",                                  "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxReset"]                                           = {"hipDevicePrimaryCtxReset",                                    "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxReset_v2"]                                        = {"hipDevicePrimaryCtxReset",                                    "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxRetain"]                                          = {"hipDevicePrimaryCtxRetain",                                   "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxSetFlags"]                                        = {"hipDevicePrimaryCtxSetFlags",                                 "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};
  m["cuDevicePrimaryCtxSetFlags_v2"]                                     = {"hipDevicePrimaryCtxSetFlags",                                 "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT, HIP_DEPRECATED};

  // 8. Context Management

  m["cuCtxCreate"]                                                       = {"hipCtxCreate",                                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_PARTIALLY_SUPPORTED | HIP_DEPRECATED};
  m["cuCtxCreate_v2"]                                                    = {"hipCtxCreate",                                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_PARTIALLY_SUPPORTED | HIP_DEPRECATED};
  m["cuCtxCreate_v3"]                                                    = {"hipCtxCreate_v3",                                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  // NOTE: cuCtxCreate_v4 equals cuCtxCreate since CUDA 13.0.0
  m["cuCtxCreate_v4"]                                                    = {"hipCtxCreate_v4",                                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  m["cuCtxDestroy"]                                                      = {"hipCtxDestroy",                                               "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxDestroy_v2"]                                                   = {"hipCtxDestroy",                                               "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxGetApiVersion"]                                                = {"hipCtxGetApiVersion",                                         "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxGetCacheConfig"]                                               = {"hipCtxGetCacheConfig",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxGetCurrent"]                                                   = {"hipCtxGetCurrent",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxGetDevice"]                                                    = {"hipCtxGetDevice",                                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxGetDevice_v2"]                                                 = {"hipCtxGetDevice_v2",                                          "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  // cudaGetDeviceFlags
  // TODO: rename to hipGetDeviceFlags
  m["cuCtxGetFlags"]                                                     = {"hipCtxGetFlags",                                              "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxSetFlags"]                                                     = {"hipCtxSetFlags",                                              "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  // cudaDeviceGetLimit
  m["cuCtxGetLimit"]                                                     = {"hipDeviceGetLimit",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT};
  // cudaDeviceGetSharedMemConfig
  // TODO: rename to hipDeviceGetSharedMemConfig
  m["cuCtxGetSharedMemConfig"]                                           = {"hipCtxGetSharedMemConfig",                                    "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED | CUDA_DEPRECATED};
  // cudaDeviceGetStreamPriorityRange
  m["cuCtxGetStreamPriorityRange"]                                       = {"hipDeviceGetStreamPriorityRange",                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT};
  m["cuCtxPopCurrent"]                                                   = {"hipCtxPopCurrent",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxPopCurrent_v2"]                                                = {"hipCtxPopCurrent",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxPushCurrent"]                                                  = {"hipCtxPushCurrent",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxPushCurrent_v2"]                                               = {"hipCtxPushCurrent",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  m["cuCtxSetCacheConfig"]                                               = {"hipCtxSetCacheConfig",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  // cudaCtxResetPersistingL2Cache
  m["cuCtxResetPersistingL2Cache"]                                       = {"hipCtxResetPersistingL2Cache",                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  m["cuCtxSetCurrent"]                                                   = {"hipCtxSetCurrent",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  // cudaDeviceSetLimit
  m["cuCtxSetLimit"]                                                     = {"hipDeviceSetLimit",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT};
  // cudaDeviceSetSharedMemConfig
  // TODO: rename to hipDeviceSetSharedMemConfig
  m["cuCtxSetSharedMemConfig"]                                           = {"hipCtxSetSharedMemConfig",                                    "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED | CUDA_DEPRECATED};
  // cudaDeviceSynchronize
  // TODO: rename to hipDeviceSynchronize
  m["cuCtxSynchronize"]                                                  = {"hipCtxSynchronize",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED};
  //
  m["cuCtxSynchronize_v2"]                                               = {"hipCtxSynchronize_v2",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuCtxGetExecAffinity"]                                              = {"hipCtxGetExecAffinity",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuCtxGetId"]                                                        = {"hipCtxGetId",                                                 "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuCtxWaitEvent"]                                                    = {"hipCtxWaitEvent",                                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED};

  // 9. Context Management [DEPRECATED]
  // no analogues
  m["cuCtxAttach"]                                                       = {"hipCtxAttach",                                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cuCtxDetach"]                                                       = {"hipCtxDetach",                                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 10. Module Management
  // no analogues
  m["cuLinkAddData"]                                                     = {"hiprtcLinkAddData",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkAddData_v2"]                                                  = {"hiprtcLinkAddData",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkAddFile"]                                                     = {"hiprtcLinkAddFile",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkAddFile_v2"]                                                  = {"hiprtcLinkAddFile",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkComplete"]                                                    = {"hiprtcLinkComplete",                                          "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkCreate"]                                                      = {"hiprtcLinkCreate",                                            "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkCreate_v2"]                                                   = {"hiprtcLinkCreate",                                            "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuLinkDestroy"]                                                     = {"hiprtcLinkDestroy",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleGetFunction"]                                               = {"hipModuleGetFunction",                                        "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleGetGlobal"]                                                 = {"hipModuleGetGlobal",                                          "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleGetGlobal_v2"]                                              = {"hipModuleGetGlobal",                                          "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleLoad"]                                                      = {"hipModuleLoad",                                               "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleLoadData"]                                                  = {"hipModuleLoadData",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleLoadDataEx"]                                                = {"hipModuleLoadDataEx",                                         "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleLoadFatBinary"]                                             = {"hipModuleLoadFatBinary",                                      "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleUnload"]                                                    = {"hipModuleUnload",                                             "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleGetLoadingMode"]                                            = {"hipModuleGetLoadingMode",                                     "", CONV_MODULE, API_DRIVER, SEC::MODULE, HIP_UNSUPPORTED};
  m["cuModuleGetFunctionCount"]                                          = {"hipModuleGetFunctionCount",                                   "", CONV_MODULE, API_DRIVER, SEC::MODULE};
  m["cuModuleEnumerateFunctions"]                                        = {"hipModuleEnumerateFunctions",                                 "", CONV_MODULE, API_DRIVER, SEC::MODULE, HIP_UNSUPPORTED};

  // 11. Module Management [DEPRECATED]
  m["cuModuleGetSurfRef"]                                                = {"hipModuleGetSurfRef",                                         "", CONV_MODULE, API_DRIVER, SEC::MODULE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cuModuleGetTexRef"]                                                 = {"hipModuleGetTexRef",                                          "", CONV_MODULE, API_DRIVER, SEC::MODULE_DEPRECATED, CUDA_DEPRECATED};

  // 12. Library Management
  // cudaLibraryLoadData
  m["cuLibraryLoadData"]                                                 = {"hipLibraryLoadData",                                          "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  // cudaLibraryLoadFromFile
  m["cuLibraryLoadFromFile"]                                             = {"hipLibraryLoadFromFile",                                      "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  // cudaLibraryUnload
  m["cuLibraryUnload"]                                                   = {"hipLibraryUnload",                                            "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  // cudaLibraryGetKernel
  m["cuLibraryGetKernel"]                                                = {"hipLibraryGetKernel",                                         "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  m["cuLibraryGetModule"]                                                = {"hipLibraryGetModule",                                         "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  m["cuKernelGetFunction"]                                               = {"hipKernelGetFunction",                                        "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cudaLibraryGetGlobal
  m["cuLibraryGetGlobal"]                                                = {"hipLibraryGetGlobal",                                         "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cudaLibraryGetManaged
  m["cuLibraryGetManaged"]                                               = {"hipLibraryGetManaged",                                        "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cudaLibraryGetUnifiedFunction
  m["cuLibraryGetUnifiedFunction"]                                       = {"hipLibraryGetUnifiedFunction",                                "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  m["cuKernelGetAttribute"]                                              = {"hipKernelGetAttribute",                                       "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  // cudaKernelSetAttributeForDevice
  m["cuKernelSetAttribute"]                                              = {"hipKernelSetAttribute",                                       "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  m["cuKernelSetCacheConfig"]                                            = {"hipKernelSetCacheConfig",                                     "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  m["cuKernelGetName"]                                                   = {"hipKernelGetName",                                            "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  // cudaLibraryGetKernelCount
  m["cuLibraryGetKernelCount"]                                           = {"hipLibraryGetKernelCount",                                    "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  // cudaLibraryEnumerateKernels
  m["cuLibraryEnumerateKernels"]                                         = {"hipLibraryEnumerateKernels",                                  "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};
  m["cuKernelGetParamInfo"]                                              = {"hipKernelGetParamInfo",                                       "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED};
  m["cuKernelGetLibrary"]                                                = {"hipKernelGetLibrary",                                         "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY};

  // 13. Memory Management
  // no analogue
  m["cuArray3DCreate"]                                                   = {"hipArray3DCreate",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArray3DCreate_v2"]                                                = {"hipArray3DCreate",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArray3DGetDescriptor"]                                            = {"hipArray3DGetDescriptor",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArray3DGetDescriptor_v2"]                                         = {"hipArray3DGetDescriptor",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArrayCreate"]                                                     = {"hipArrayCreate",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArrayCreate_v2"]                                                  = {"hipArrayCreate",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArrayDestroy"]                                                    = {"hipArrayDestroy",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArrayGetDescriptor"]                                              = {"hipArrayGetDescriptor",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuArrayGetDescriptor_v2"]                                           = {"hipArrayGetDescriptor",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  //
  m["cuMipmappedArrayGetMemoryRequirements"]                             = {"hipMipmappedArrayGetMemoryRequirements",                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaArrayGetMemoryRequirements
  m["cuArrayGetMemoryRequirements"]                                      = {"hipArrayGetMemoryRequirements",                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaDeviceGetByPCIBusId
  m["cuDeviceGetByPCIBusId"]                                             = {"hipDeviceGetByPCIBusId",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaDeviceGetPCIBusId
  m["cuDeviceGetPCIBusId"]                                               = {"hipDeviceGetPCIBusId",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaIpcCloseMemHandle
  m["cuIpcCloseMemHandle"]                                               = {"hipIpcCloseMemHandle",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaIpcGetEventHandle
  m["cuIpcGetEventHandle"]                                               = {"hipIpcGetEventHandle",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaIpcGetMemHandle
  m["cuIpcGetMemHandle"]                                                 = {"hipIpcGetMemHandle",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaIpcOpenEventHandle
  m["cuIpcOpenEventHandle"]                                              = {"hipIpcOpenEventHandle",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaIpcOpenMemHandle
  m["cuIpcOpenMemHandle"]                                                = {"hipIpcOpenMemHandle",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaMalloc
  m["cuMemAlloc"]                                                        = {"hipMalloc",                                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemAlloc_v2"]                                                     = {"hipMalloc",                                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  //
  m["cuMemAllocHost"]                                                    = {"hipMemAllocHost",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemAllocHost_v2"]                                                 = {"hipMemAllocHost",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaMallocManaged
  m["cuMemAllocManaged"]                                                 = {"hipMallocManaged",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMallocPitch due to different signatures
  m["cuMemAllocPitch"]                                                   = {"hipMemAllocPitch",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemAllocPitch_v2"]                                                = {"hipMemAllocPitch",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpy due to different signatures
  m["cuMemcpy"]                                                          = {"hipMemcpy_",                                                  "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaMemcpy2D due to different signatures
  m["cuMemcpy2D"]                                                        = {"hipMemcpyParam2D",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpy2D_v2"]                                                     = {"hipMemcpyParam2D",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpy2DAsync/hipMemcpy2DAsync due to different signatures
  m["cuMemcpy2DAsync"]                                                   = {"hipMemcpyParam2DAsync",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpy2DAsync_v2"]                                                = {"hipMemcpyParam2DAsync",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpy2DUnaligned"]                                               = {"hipDrvMemcpy2DUnaligned",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpy2DUnaligned_v2"]                                            = {"hipDrvMemcpy2DUnaligned",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpy3D due to different signatures
  m["cuMemcpy3D"]                                                        = {"hipDrvMemcpy3D",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpy3D_v2"]                                                     = {"hipDrvMemcpy3D",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DAsync due to different signatures
  m["cuMemcpy3DAsync"]                                                   = {"hipDrvMemcpy3DAsync",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpy3DAsync_v2"]                                                = {"hipDrvMemcpy3DAsync",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeer due to different signatures
  m["cuMemcpy3DPeer"]                                                    = {"hipMemcpy3DPeer_",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeerAsync due to different signatures
  m["cuMemcpy3DPeerAsync"]                                               = {"hipMemcpy3DPeerAsync_",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaMemcpyAsync due to different signatures
  m["cuMemcpyAsync"]                                                     = {"hipMemcpyAsync_",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaMemcpyArrayToArray due to different signatures
  m["cuMemcpyAtoA"]                                                      = {"hipMemcpyAtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyAtoA_v2"]                                                   = {"hipMemcpyAtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyAtoD"]                                                      = {"hipMemcpyAtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyAtoD_v2"]                                                   = {"hipMemcpyAtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyAtoH"]                                                      = {"hipMemcpyAtoH",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyAtoH_v2"]                                                   = {"hipMemcpyAtoH",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyAtoHAsync"]                                                 = {"hipMemcpyAtoHAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyAtoHAsync_v2"]                                              = {"hipMemcpyAtoHAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyDtoA"]                                                      = {"hipMemcpyDtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyDtoA_v2"]                                                   = {"hipMemcpyDtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyDtoD"]                                                      = {"hipMemcpyDtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyDtoD_v2"]                                                   = {"hipMemcpyDtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyDtoDAsync"]                                                 = {"hipMemcpyDtoDAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyDtoDAsync_v2"]                                              = {"hipMemcpyDtoDAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyDtoH"]                                                      = {"hipMemcpyDtoH",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyDtoH_v2"]                                                   = {"hipMemcpyDtoH",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyDtoHAsync"]                                                 = {"hipMemcpyDtoHAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyDtoHAsync_v2"]                                              = {"hipMemcpyDtoHAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyHtoA"]                                                      = {"hipMemcpyHtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyHtoA_v2"]                                                   = {"hipMemcpyHtoA",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyHtoAAsync"]                                                 = {"hipMemcpyHtoAAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyHtoAAsync_v2"]                                              = {"hipMemcpyHtoAAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyHtoD"]                                                      = {"hipMemcpyHtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyHtoD_v2"]                                                   = {"hipMemcpyHtoD",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemcpyHtoDAsync"]                                                 = {"hipMemcpyHtoDAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemcpyHtoDAsync_v2"]                                              = {"hipMemcpyHtoDAsync",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeer due to different signatures
  m["cuMemcpyPeer"]                                                      = {"hipMemcpyPeer_",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeerAsync due to different signatures
  m["cuMemcpyPeerAsync"]                                                 = {"hipMemcpyPeerAsync_",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaMemcpyBatchAsync
  m["cuMemcpyBatchAsync"]                                                = {"hipMemcpyBatchAsync",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  // cudaMemcpy3DBatchAsync
  m["cuMemcpy3DBatchAsync"]                                              = {"hipMemcpy3DBatchAsync",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_PARTIALLY_SUPPORTED};
  //
  m["cuMemBatchDecompressAsync"]                                         = {"hipMemBatchDecompressAsync",                                  "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaFree
  m["cuMemFree"]                                                         = {"hipFree",                                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemFree_v2"]                                                      = {"hipFree",                                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaFreeHost
  m["cuMemFreeHost"]                                                     = {"hipHostFree",                                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemGetAddressRange"]                                              = {"hipMemGetAddressRange",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemGetAddressRange_v2"]                                           = {"hipMemGetAddressRange",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaMemGetInfo
  m["cuMemGetInfo"]                                                      = {"hipMemGetInfo",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemGetInfo_v2"]                                                   = {"hipMemGetInfo",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaHostAlloc
  m["cuMemHostAlloc"]                                                    = {"hipHostAlloc",                                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaHostGetDevicePointer
  m["cuMemHostGetDevicePointer"]                                         = {"hipHostGetDevicePointer",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemHostGetDevicePointer_v2"]                                      = {"hipHostGetDevicePointer",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaHostGetFlags
  m["cuMemHostGetFlags"]                                                 = {"hipHostGetFlags",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaHostRegister
  m["cuMemHostRegister"]                                                 = {"hipHostRegister",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemHostRegister_v2"]                                              = {"hipHostRegister",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaHostUnregister
  m["cuMemHostUnregister"]                                               = {"hipHostUnregister",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD16"]                                                       = {"hipMemsetD16",                                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD16_v2"]                                                    = {"hipMemsetD16",                                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD16Async"]                                                  = {"hipMemsetD16Async",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D16"]                                                     = {"hipMemsetD2D16",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD2D16_v2"]                                                  = {"hipMemsetD2D16",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D16Async"]                                                = {"hipMemsetD2D16Async",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D32"]                                                     = {"hipMemsetD2D32",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD2D32_v2"]                                                  = {"hipMemsetD2D32",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D32Async"]                                                = {"hipMemsetD2D32Async",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D8"]                                                      = {"hipMemsetD2D8",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD2D8_v2"]                                                   = {"hipMemsetD2D8",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD2D8Async"]                                                 = {"hipMemsetD2D8Async",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaMemset
  m["cuMemsetD32"]                                                       = {"hipMemsetD32",                                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD32_v2"]                                                    = {"hipMemsetD32",                                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaMemsetAsync
  m["cuMemsetD32Async"]                                                  = {"hipMemsetD32Async",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD8"]                                                        = {"hipMemsetD8",                                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  m["cuMemsetD8_v2"]                                                     = {"hipMemsetD8",                                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  m["cuMemsetD8Async"]                                                   = {"hipMemsetD8Async",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // no analogue
  // NOTE: Not equal to cudaMallocMipmappedArray due to different signatures
  m["cuMipmappedArrayCreate"]                                            = {"hipMipmappedArrayCreate",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaFreeMipmappedArray due to different signatures
  m["cuMipmappedArrayDestroy"]                                           = {"hipMipmappedArrayDestroy",                                    "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaGetMipmappedArrayLevel due to different signatures
  m["cuMipmappedArrayGetLevel"]                                          = {"hipMipmappedArrayGetLevel",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED};
  // cudaArrayGetSparseProperties
  m["cuArrayGetSparseProperties"]                                        = {"hipArrayGetSparseProperties",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaArrayGetPlane
  m["cuArrayGetPlane"]                                                   = {"hipArrayGetPlane",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  //
  m["cuMemGetHandleForAddressRange"]                                     = {"hipMemGetHandleForAddressRange",                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY};
  // cudaDeviceRegisterAsyncNotification
  m["cuDeviceRegisterAsyncNotification"]                                 = {"hipDeviceRegisterAsyncNotification",                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};
  // cudaDeviceUnregisterAsyncNotification
  m["cuDeviceUnregisterAsyncNotification"]                               = {"hipDeviceUnregisterAsyncNotification",                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED};

  // 14. Virtual Memory Management
  // no analogue
  m["cuMemAddressFree"]                                                  = {"hipMemAddressFree",                                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemAddressReserve"]                                               = {"hipMemAddressReserve",                                        "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemCreate"]                                                       = {"hipMemCreate",                                                "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemExportToShareableHandle"]                                      = {"hipMemExportToShareableHandle",                               "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemGetAccess"]                                                    = {"hipMemGetAccess",                                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemGetAllocationGranularity"]                                     = {"hipMemGetAllocationGranularity",                              "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemGetAllocationPropertiesFromHandle"]                            = {"hipMemGetAllocationPropertiesFromHandle",                     "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemImportFromShareableHandle"]                                    = {"hipMemImportFromShareableHandle",                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemMap"]                                                          = {"hipMemMap",                                                   "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemRelease"]                                                      = {"hipMemRelease",                                               "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemRetainAllocationHandle"]                                       = {"hipMemRetainAllocationHandle",                                "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemSetAccess"]                                                    = {"hipMemSetAccess",                                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemUnmap"]                                                        = {"hipMemUnmap",                                                 "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};
  m["cuMemMapArrayAsync"]                                                = {"hipMemMapArrayAsync",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY};

  // 15. Stream Ordered Memory Allocator
  // cudaFreeAsync
  m["cuMemFreeAsync"]                                                    = {"hipFreeAsync",                                                "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMallocAsync
  m["cuMemAllocAsync"]                                                   = {"hipMallocAsync",                                              "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolTrimTo
  m["cuMemPoolTrimTo"]                                                   = {"hipMemPoolTrimTo",                                            "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolSetAttribute
  m["cuMemPoolSetAttribute"]                                             = {"hipMemPoolSetAttribute",                                      "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolGetAttribute
  m["cuMemPoolGetAttribute"]                                             = {"hipMemPoolGetAttribute",                                      "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolSetAccess
  m["cuMemPoolSetAccess"]                                                = {"hipMemPoolSetAccess",                                         "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolGetAccess
  m["cuMemPoolGetAccess"]                                                = {"hipMemPoolGetAccess",                                         "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolCreate
  m["cuMemPoolCreate"]                                                   = {"hipMemPoolCreate",                                            "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolDestroy
  m["cuMemPoolDestroy"]                                                  = {"hipMemPoolDestroy",                                           "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMallocFromPoolAsync
  m["cuMemAllocFromPoolAsync"]                                           = {"hipMallocFromPoolAsync",                                      "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolExportToShareableHandle
  m["cuMemPoolExportToShareableHandle"]                                  = {"hipMemPoolExportToShareableHandle",                           "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolImportFromShareableHandle
  m["cuMemPoolImportFromShareableHandle"]                                = {"hipMemPoolImportFromShareableHandle",                         "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolExportPointer
  m["cuMemPoolExportPointer"]                                            = {"hipMemPoolExportPointer",                                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemPoolImportPointer
  m["cuMemPoolImportPointer"]                                            = {"hipMemPoolImportPointer",                                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY};
  // cudaMemGetDefaultMemPool
  m["cuMemGetDefaultMemPool"]                                            = {"hipMemGetDefaultMemPool",                                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};
  // cudaMemGetMemPool
  m["cuMemGetMemPool"]                                                   = {"hipMemGetMemPool",                                            "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};
  // cudaMemSetMemPool
  m["cuMemSetMemPool"]                                                   = {"hipMemSetMemPool",                                            "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY, HIP_UNSUPPORTED};

  // 16. Multicast Object Management
  //
  m["cuMulticastCreate"]                                                 = {"hipMulticastCreate",                                          "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};
  //
  m["cuMulticastAddDevice"]                                              = {"hipMulticastAddDevice",                                       "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};
  //
  m["cuMulticastBindMem"]                                                = {"hipMulticastBindMem",                                         "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};
  //
  m["cuMulticastBindAddr"]                                               = {"hipMulticastBindAddr",                                        "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};
  //
  m["cuMulticastUnbind"]                                                 = {"hipMulticastUnbind",                                          "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};
  //
  m["cuMulticastGetGranularity"]                                         = {"hipMulticastGetGranularity",                                  "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED};

  // 17. Unified Addressing
  // cudaMemAdvise
  m["cuMemAdvise"]                                                       = {"hipMemAdvise",                                                "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_PARTIALLY_SUPPORTED};
  // cudaMemAdvise_v2
  m["cuMemAdvise_v2"]                                                    = {"hipMemAdvise_v2",                                             "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED};
  // cudaMemPrefetchAsync
  m["cuMemPrefetchAsync"]                                                = {"hipMemPrefetchAsync",                                         "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_PARTIALLY_SUPPORTED};
  // cudaMemPrefetchAsync_v2
  m["cuMemPrefetchAsync_v2"]                                             = {"hipMemPrefetchAsync_v2",                                      "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED};
  // cudaMemRangeGetAttribute
  m["cuMemRangeGetAttribute"]                                            = {"hipMemRangeGetAttribute",                                     "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED};
  // cudaMemRangeGetAttributes
  m["cuMemRangeGetAttributes"]                                           = {"hipMemRangeGetAttributes",                                    "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED};
  // no analogue
  m["cuPointerGetAttribute"]                                             = {"hipPointerGetAttribute",                                      "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED};
  // no analogue
  // NOTE: Not equal to cudaPointerGetAttributes due to different signatures
  m["cuPointerGetAttributes"]                                            = {"hipDrvPointerGetAttributes",                                  "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED};
  // no analogue
  m["cuPointerSetAttribute"]                                             = {"hipPointerSetAttribute",                                      "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED};
  // cudaMemPrefetchBatchAsync
  m["cuMemPrefetchBatchAsync"]                                           = {"hipMemPrefetchBatchAsync",                                    "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED};
  // cudaMemDiscardBatchAsync
  m["cuMemDiscardBatchAsync"]                                            = {"hipMemDiscardBatchAsync",                                     "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED};
  // cudaMemDiscardAndPrefetchBatchAsync
  m["cuMemDiscardAndPrefetchBatchAsync"]                                 = {"hipMemDiscardAndPrefetchBatchAsync",                          "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED};

  // 18. Stream Management
  // cudaStreamAddCallback
  m["cuStreamAddCallback"]                                               = {"hipStreamAddCallback",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamAttachMemAsync
  m["cuStreamAttachMemAsync"]                                            = {"hipStreamAttachMemAsync",                                     "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamBeginCapture
  m["cuStreamBeginCapture"]                                              = {"hipStreamBeginCapture",                                       "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  m["cuStreamBeginCapture_v2"]                                           = {"hipStreamBeginCapture",                                       "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  m["cuStreamBeginCapture_ptsz"]                                         = {"hipStreamBeginCapture_ptsz",                                  "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};
  // cudaStreamBeginCaptureToGraph
  m["cuStreamBeginCaptureToGraph"]                                       = {"hipStreamBeginCaptureToGraph",                                "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamCopyAttributes
  m["cuStreamCopyAttributes"]                                            = {"hipStreamCopyAttributes",                                     "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamCreateWithFlags
  m["cuStreamCreate"]                                                    = {"hipStreamCreateWithFlags",                                    "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamCreateWithPriority
  m["cuStreamCreateWithPriority"]                                        = {"hipStreamCreateWithPriority",                                 "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamDestroy
  m["cuStreamDestroy"]                                                   = {"hipStreamDestroy",                                            "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  m["cuStreamDestroy_v2"]                                                = {"hipStreamDestroy",                                            "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamEndCapture
  m["cuStreamEndCapture"]                                                = {"hipStreamEndCapture",                                         "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetAttribute
  m["cuStreamGetAttribute"]                                              = {"hipStreamGetAttribute",                                       "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetCaptureInfo
  m["cuStreamGetCaptureInfo"]                                            = {"hipStreamGetCaptureInfo",                                     "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_PARTIALLY_SUPPORTED};
  m["cuStreamGetCaptureInfo_v2"]                                         = {"hipStreamGetCaptureInfo_v2",                                  "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetCaptureInfo_v3
  m["cuStreamGetCaptureInfo_v3"]                                         = {"hipStreamGetCaptureInfo_v3",                                  "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};
  // cudaStreamUpdateCaptureDependencies
  m["cuStreamUpdateCaptureDependencies"]                                 = {"hipStreamUpdateCaptureDependencies",                          "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_PARTIALLY_SUPPORTED};
  // cudaStreamUpdateCaptureDependencies_v2
  m["cuStreamUpdateCaptureDependencies_v2"]                              = {"hipStreamUpdateCaptureDependencies_v2",                       "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};
  // no analogue
  m["cuStreamGetCtx"]                                                    = {"hipStreamGetContext",                                         "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};
  // no analogue
  m["cuStreamGetCtx_v2"]                                                 = {"hipStreamGetContext_v2",                                      "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};
  // cudaStreamGetFlags
  m["cuStreamGetFlags"]                                                  = {"hipStreamGetFlags",                                           "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetPriority
  m["cuStreamGetPriority"]                                               = {"hipStreamGetPriority",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamIsCapturing
  m["cuStreamIsCapturing"]                                               = {"hipStreamIsCapturing",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamQuery
  m["cuStreamQuery"]                                                     = {"hipStreamQuery",                                              "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamSetAttribute
  m["cuStreamSetAttribute"]                                              = {"hipStreamSetAttribute",                                       "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamSynchronize
  m["cuStreamSynchronize"]                                               = {"hipStreamSynchronize",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamWaitEvent
  m["cuStreamWaitEvent"]                                                 = {"hipStreamWaitEvent",                                          "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaThreadExchangeStreamCaptureMode
  m["cuThreadExchangeStreamCaptureMode"]                                 = {"hipThreadExchangeStreamCaptureMode",                          "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetId
  m["cuStreamGetId"]                                                     = {"hipStreamGetId",                                              "", CONV_STREAM, API_DRIVER, SEC::STREAM};
  // cudaStreamGetDevice
  m["cuStreamGetDevice"]                                                 = {"hipStreamGetDevice",                                          "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED};

  // 19. Event Management
  // cudaEventCreateWithFlags
  m["cuEventCreate"]                                                     = {"hipEventCreateWithFlags",                                     "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventDestroy
  m["cuEventDestroy"]                                                    = {"hipEventDestroy",                                             "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  m["cuEventDestroy_v2"]                                                 = {"hipEventDestroy",                                             "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventElapsedTime
  m["cuEventElapsedTime"]                                                = {"hipEventElapsedTime",                                         "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  //
  m["cuEventElapsedTime_v2"]                                             = {"hipEventElapsedTime",                                         "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventQuery
  m["cuEventQuery"]                                                      = {"hipEventQuery",                                               "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventRecord
  m["cuEventRecord"]                                                     = {"hipEventRecord",                                              "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventSynchronize
  m["cuEventSynchronize"]                                                = {"hipEventSynchronize",                                         "", CONV_EVENT, API_DRIVER, SEC::EVENT};
  // cudaEventRecordWithFlags
  m["cuEventRecordWithFlags"]                                            = {"hipEventRecordWithFlags",                                     "", CONV_EVENT, API_DRIVER, SEC::EVENT};

  // 20. External Resource Interoperability
  // cudaDestroyExternalMemory
  m["cuDestroyExternalMemory"]                                           = {"hipDestroyExternalMemory",                                    "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaDestroyExternalSemaphore
  m["cuDestroyExternalSemaphore"]                                        = {"hipDestroyExternalSemaphore",                                 "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaExternalMemoryGetMappedBuffer
  m["cuExternalMemoryGetMappedBuffer"]                                   = {"hipExternalMemoryGetMappedBuffer",                            "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaExternalMemoryGetMappedMipmappedArray
  m["cuExternalMemoryGetMappedMipmappedArray"]                           = {"hipExternalMemoryGetMappedMipmappedArray",                    "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES, HIP_UNSUPPORTED};
  // cudaImportExternalMemory
  m["cuImportExternalMemory"]                                            = {"hipImportExternalMemory",                                     "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaImportExternalSemaphore
  m["cuImportExternalSemaphore"]                                         = {"hipImportExternalSemaphore",                                  "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaSignalExternalSemaphoresAsync
  m["cuSignalExternalSemaphoresAsync"]                                   = {"hipSignalExternalSemaphoresAsync",                            "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};
  // cudaWaitExternalSemaphoresAsync
  m["cuWaitExternalSemaphoresAsync"]                                     = {"hipWaitExternalSemaphoresAsync",                              "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES};

  // 21. Stream Memory Operations
  // no analogues
  m["cuStreamBatchMemOp"]                                                = {"hipStreamBatchMemOp",                                         "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  m["cuStreamBatchMemOp_v2"]                                             = {"hipStreamBatchMemOp",                                         "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  m["cuStreamWaitValue32"]                                               = {"hipStreamWaitValue32",                                        "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  m["cuStreamWaitValue32_v2"]                                            = {"hipStreamWaitValue32",                                        "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  m["cuStreamWaitValue64"]                                               = {"hipStreamWaitValue64",                                        "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  m["cuStreamWaitValue64_v2"]                                            = {"hipStreamWaitValue64",                                        "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  m["cuStreamWriteValue32"]                                              = {"hipStreamWriteValue32",                                       "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  m["cuStreamWriteValue32_v2"]                                           = {"hipStreamWriteValue32",                                       "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  m["cuStreamWriteValue64"]                                              = {"hipStreamWriteValue64",                                       "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};
  // CUresult CUDAAPI cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  m["cuStreamWriteValue64_v2"]                                           = {"hipStreamWriteValue64",                                       "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY};

  // 22. Execution Control
  // no analogue
  m["cuFuncGetAttribute"]                                                = {"hipFuncGetAttribute",                                         "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION};
  // no analogue
  m["cuFuncGetModule"]                                                   = {"hipFuncGetModule",                                            "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaFuncSetAttribute due to different signatures
  m["cuFuncSetAttribute"]                                                = {"hipFuncSetAttribute_",                                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  m["cuFuncSetCacheConfig"]                                              = {"hipFuncSetCacheConfig_",                                      "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaFuncSetSharedMemConfig due to different signatures
  m["cuFuncSetSharedMemConfig"]                                          = {"hipFuncSetSharedMemConfig_",                                  "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernel due to different signatures
  m["cuLaunchCooperativeKernel"]                                         = {"hipModuleLaunchCooperativeKernel",                            "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernelMultiDevice due to different signatures
  m["cuLaunchCooperativeKernelMultiDevice"]                              = {"hipModuleLaunchCooperativeKernelMultiDevice",                 "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, CUDA_DEPRECATED};
  // cudaLaunchHostFunc
  m["cuLaunchHostFunc"]                                                  = {"hipLaunchHostFunc",                                           "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cudaLaunchKernel due to different signatures
  m["cuLaunchKernel"]                                                    = {"hipModuleLaunchKernel",                                       "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION};
  // no analogue
  // NOTE: Not equal to cudaLaunchKernelExC due to different signatures
  m["cuLaunchKernelEx"]                                                  = {"hipDrvLaunchKernelEx",                                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION};
  // cudaFuncGetName
  m["cuFuncGetName"]                                                     = {"hipFuncGetName",                                              "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  // cudaFuncGetParamInfo
  m["cuFuncGetParamInfo"]                                                = {"hipFuncGetParamInfo",                                         "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  //
  m["cuFuncIsLoaded"]                                                    = {"hipFuncIsLoaded",                                             "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};
  //
  m["cuFuncLoad"]                                                        = {"hipFuncLoad",                                                 "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED};

  // 23. Execution Control [DEPRECATED]
  // no analogue
  m["cuFuncSetBlockShape"]                                               = {"hipFuncSetBlockShape",                                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuFuncSetSharedSize"]                                               = {"hipFuncSetSharedSize",                                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  m["cuLaunch"]                                                          = {"hipLaunch",                                                   "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuLaunchGrid"]                                                      = {"hipLaunchGrid",                                               "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuLaunchGridAsync"]                                                 = {"hipLaunchGridAsync",                                          "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuParamSetf"]                                                       = {"hipParamSetf",                                                "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuParamSeti"]                                                       = {"hipParamSeti",                                                "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuParamSetSize"]                                                    = {"hipParamSetSize",                                             "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuParamSetTexRef"]                                                  = {"hipParamSetTexRef",                                           "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuParamSetv"]                                                       = {"hipParamSetv",                                                "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 24. Graph Management
  // cudaGraphAddChildGraphNode
  m["cuGraphAddChildGraphNode"]                                          = {"hipGraphAddChildGraphNode",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddDependencies
  m["cuGraphAddDependencies"]                                            = {"hipGraphAddDependencies",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphAddDependencies_v2
  m["cuGraphAddDependencies_v2"]                                         = {"hipGraphAddDependencies_v2",                                  "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // cudaGraphAddEmptyNode
  m["cuGraphAddEmptyNode"]                                               = {"hipGraphAddEmptyNode",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddHostNode
  m["cuGraphAddHostNode"]                                                = {"hipGraphAddHostNode",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddKernelNode
  m["cuGraphAddKernelNode"]                                              = {"hipGraphAddKernelNode",                                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue
  // NOTE: Not equal to cudaGraphAddMemcpyNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  m["cuGraphAddMemcpyNode"]                                              = {"hipDrvGraphAddMemcpyNode",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue
  // NOTE: Not equal to cudaGraphAddMemsetNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);
  m["cuGraphAddMemsetNode"]                                              = {"hipDrvGraphAddMemsetNode",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphChildGraphNodeGetGraph
  m["cuGraphChildGraphNodeGetGraph"]                                     = {"hipGraphChildGraphNodeGetGraph",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphClone
  m["cuGraphClone"]                                                      = {"hipGraphClone",                                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphCreate
  m["cuGraphCreate"]                                                     = {"hipGraphCreate",                                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphDebugDotPrint
  m["cuGraphDebugDotPrint"]                                              = {"hipGraphDebugDotPrint",                                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphDestroy
  m["cuGraphDestroy"]                                                    = {"hipGraphDestroy",                                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphDestroyNode
  m["cuGraphDestroyNode"]                                                = {"hipGraphDestroyNode",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecDestroy
  m["cuGraphExecDestroy"]                                                = {"hipGraphExecDestroy",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphGetEdges
  m["cuGraphGetEdges"]                                                   = {"hipGraphGetEdges",                                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphGetEdges_v2
  m["cuGraphGetEdges_v2"]                                                = {"hipGraphGetEdges_v2",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // cudaGraphGetNodes
  m["cuGraphGetNodes"]                                                   = {"hipGraphGetNodes",                                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphGetRootNodes
  m["cuGraphGetRootNodes"]                                               = {"hipGraphGetRootNodes",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphHostNodeGetParams
  m["cuGraphHostNodeGetParams"]                                          = {"hipGraphHostNodeGetParams",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphHostNodeSetParams
  m["cuGraphHostNodeSetParams"]                                          = {"hipGraphHostNodeSetParams",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphInstantiate
  m["cuGraphInstantiate"]                                                = {"hipGraphInstantiate",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  m["cuGraphInstantiate_v2"]                                             = {"hipGraphInstantiate",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphKernelNodeCopyAttributes
  m["cuGraphKernelNodeCopyAttributes"]                                   = {"hipGraphKernelNodeCopyAttributes",                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphKernelNodeGetAttribute
  m["cuGraphKernelNodeGetAttribute"]                                     = {"hipGraphKernelNodeGetAttribute",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecKernelNodeSetParams
  m["cuGraphExecKernelNodeSetParams"]                                    = {"hipGraphExecKernelNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphKernelNodeGetParams
  m["cuGraphKernelNodeGetParams"]                                        = {"hipGraphKernelNodeGetParams",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphKernelNodeSetAttribute
  m["cuGraphKernelNodeSetAttribute"]                                     = {"hipGraphKernelNodeSetAttribute",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphKernelNodeSetParams
  m["cuGraphKernelNodeSetParams"]                                        = {"hipGraphKernelNodeSetParams",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphLaunch
  m["cuGraphLaunch"]                                                     = {"hipGraphLaunch",                                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // NOTE: cudaGraphMemcpyNodeGetParams has a different signature
  m["cuGraphMemcpyNodeGetParams"]                                        = {"hipDrvGraphMemcpyNodeGetParams",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // NOTE: cudaGraphMemcpyNodeSetParams has a different signature
  m["cuGraphMemcpyNodeSetParams"]                                        = {"hipDrvGraphMemcpyNodeSetParams",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphMemsetNodeGetParams
  m["cuGraphMemsetNodeGetParams"]                                        = {"hipGraphMemsetNodeGetParams",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphMemsetNodeSetParams
  m["cuGraphMemsetNodeSetParams"]                                        = {"hipGraphMemsetNodeSetParams",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphNodeFindInClone
  m["cuGraphNodeFindInClone"]                                            = {"hipGraphNodeFindInClone",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphNodeGetDependencies
  m["cuGraphNodeGetDependencies"]                                        = {"hipGraphNodeGetDependencies",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphNodeGetDependencies_v2
  m["cuGraphNodeGetDependencies_v2"]                                     = {"hipGraphNodeGetDependencies_v2",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // cudaGraphNodeGetDependentNodes
  m["cuGraphNodeGetDependentNodes"]                                      = {"hipGraphNodeGetDependentNodes",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphNodeGetDependentNodes_v2
  m["cuGraphNodeGetDependentNodes_v2"]                                   = {"hipGraphNodeGetDependentNodes_v2",                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // cudaGraphNodeGetEnabled
  m["cuGraphNodeGetEnabled"]                                             = {"hipGraphNodeGetEnabled",                                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphNodeGetType
  m["cuGraphNodeGetType"]                                                = {"hipGraphNodeGetType",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphNodeSetEnabled
  m["cuGraphNodeSetEnabled"]                                             = {"hipGraphNodeSetEnabled",                                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphRemoveDependencies
  m["cuGraphRemoveDependencies"]                                         = {"hipGraphRemoveDependencies",                                  "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphRemoveDependencies_v2
  m["cuGraphRemoveDependencies_v2"]                                      = {"hipGraphRemoveDependencies_v2",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // no analogue
  m["cuGraphExecMemcpyNodeSetParams"]                                    = {"hipDrvGraphExecMemcpyNodeSetParams",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue
  m["cuGraphExecMemsetNodeSetParams"]                                    = {"hipDrvGraphExecMemsetNodeSetParams",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecHostNodeSetParams
  m["cuGraphExecHostNodeSetParams"]                                      = {"hipGraphExecHostNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // TODO: take into account the new signature since 12.0
  // cudaGraphExecUpdate
  m["cuGraphExecUpdate"]                                                 = {"hipGraphExecUpdate",                                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddEventRecordNode
  m["cuGraphAddEventRecordNode"]                                         = {"hipGraphAddEventRecordNode",                                  "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphEventRecordNodeGetEvent
  m["cuGraphEventRecordNodeGetEvent"]                                    = {"hipGraphEventRecordNodeGetEvent",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphEventRecordNodeSetEvent
  m["cuGraphEventRecordNodeSetEvent"]                                    = {"hipGraphEventRecordNodeSetEvent",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddEventWaitNode
  m["cuGraphAddEventWaitNode"]                                           = {"hipGraphAddEventWaitNode",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphEventWaitNodeGetEvent
  m["cuGraphEventWaitNodeGetEvent"]                                      = {"hipGraphEventWaitNodeGetEvent",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphEventWaitNodeSetEvent
  m["cuGraphEventWaitNodeSetEvent"]                                      = {"hipGraphEventWaitNodeSetEvent",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecChildGraphNodeSetParams
  m["cuGraphExecChildGraphNodeSetParams"]                                = {"hipGraphExecChildGraphNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecEventRecordNodeSetEvent
  m["cuGraphExecEventRecordNodeSetEvent"]                                = {"hipGraphExecEventRecordNodeSetEvent",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecEventWaitNodeSetEvent
  m["cuGraphExecEventWaitNodeSetEvent"]                                  = {"hipGraphExecEventWaitNodeSetEvent",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphUpload
  m["cuGraphUpload"]                                                     = {"hipGraphUpload",                                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddExternalSemaphoresSignalNode
  m["cuGraphAddExternalSemaphoresSignalNode"]                            = {"hipGraphAddExternalSemaphoresSignalNode",                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExternalSemaphoresSignalNodeGetParams
  m["cuGraphExternalSemaphoresSignalNodeGetParams"]                      = {"hipGraphExternalSemaphoresSignalNodeGetParams",               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExternalSemaphoresSignalNodeSetParams
  m["cuGraphExternalSemaphoresSignalNodeSetParams"]                      = {"hipGraphExternalSemaphoresSignalNodeSetParams",               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddExternalSemaphoresWaitNode
  m["cuGraphAddExternalSemaphoresWaitNode"]                              = {"hipGraphAddExternalSemaphoresWaitNode",                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExternalSemaphoresWaitNodeGetParams
  m["cuGraphExternalSemaphoresWaitNodeGetParams"]                        = {"hipGraphExternalSemaphoresWaitNodeGetParams",                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExternalSemaphoresWaitNodeSetParams
  m["cuGraphExternalSemaphoresWaitNodeSetParams"]                        = {"hipGraphExternalSemaphoresWaitNodeSetParams",                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecExternalSemaphoresSignalNodeSetParams
  m["cuGraphExecExternalSemaphoresSignalNodeSetParams"]                  = {"hipGraphExecExternalSemaphoresSignalNodeSetParams",           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecExternalSemaphoresWaitNodeSetParams
  m["cuGraphExecExternalSemaphoresWaitNodeSetParams"]                    = {"hipGraphExecExternalSemaphoresWaitNodeSetParams",             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaUserObjectCreate
  m["cuUserObjectCreate"]                                                = {"hipUserObjectCreate",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaUserObjectRetain
  m["cuUserObjectRetain"]                                                = {"hipUserObjectRetain",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaUserObjectRelease
  m["cuUserObjectRelease"]                                               = {"hipUserObjectRelease",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphRetainUserObject
  m["cuGraphRetainUserObject"]                                           = {"hipGraphRetainUserObject",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphReleaseUserObject
  m["cuGraphReleaseUserObject"]                                          = {"hipGraphReleaseUserObject",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddMemAllocNode
  m["cuGraphAddMemAllocNode"]                                            = {"hipGraphAddMemAllocNode",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphMemAllocNodeGetParams
  m["cuGraphMemAllocNodeGetParams"]                                      = {"hipGraphMemAllocNodeGetParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue
  m["cuGraphAddMemFreeNode"]                                             = {"hipDrvGraphAddMemFreeNode",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphMemFreeNodeGetParams
  m["cuGraphMemFreeNodeGetParams"]                                       = {"hipGraphMemFreeNodeGetParams",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaDeviceGraphMemTrim
  m["cuDeviceGraphMemTrim"]                                              = {"hipDeviceGraphMemTrim",                                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaDeviceGetGraphMemAttribute
  m["cuDeviceGetGraphMemAttribute"]                                      = {"hipDeviceGetGraphMemAttribute",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaDeviceSetGraphMemAttribute
  m["cuDeviceSetGraphMemAttribute"]                                      = {"hipDeviceSetGraphMemAttribute",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphInstantiateWithFlags
  m["cuGraphInstantiateWithFlags"]                                       = {"hipGraphInstantiateWithFlags",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue yet
  m["cuGraphAddBatchMemOpNode"]                                          = {"hipGraphAddBatchMemOpNode",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue yet
  m["cuGraphBatchMemOpNodeGetParams"]                                    = {"hipGraphBatchMemOpNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue yet
  m["cuGraphBatchMemOpNodeSetParams"]                                    = {"hipGraphBatchMemOpNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // no analogue yet
  m["cuGraphExecBatchMemOpNodeSetParams"]                                = {"hipGraphExecBatchMemOpNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphInstantiateWithParams
  m["cuGraphInstantiateWithParams"]                                      = {"hipGraphInstantiateWithParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecGetFlags
  m["cuGraphExecGetFlags"]                                               = {"hipGraphExecGetFlags",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphAddNode
  m["cuGraphAddNode"]                                                    = {"hipGraphAddNode",                                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_PARTIALLY_SUPPORTED};
  // cudaGraphAddNode_v2
  // NOTE: cuGraphAddNode_v2 equals cuGraphAddNode since CUDA 13.0.0
  m["cuGraphAddNode_v2"]                                                 = {"hipGraphAddNode_v2",                                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};
  // cudaGraphNodeSetParams
  m["cuGraphNodeSetParams"]                                              = {"hipGraphNodeSetParams",                                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphExecNodeSetParams
  m["cuGraphExecNodeSetParams"]                                          = {"hipGraphExecNodeSetParams",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH};
  // cudaGraphConditionalHandleCreate
  m["cuGraphConditionalHandleCreate"]                                    = {"hipGraphConditionalHandleCreate",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED};

  // 25. Occupancy
  // cudaOccupancyAvailableDynamicSMemPerBlock
  m["cuOccupancyAvailableDynamicSMemPerBlock"]                           = {"hipOccupancyAvailableDynamicSMemPerBlock",                    "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY};
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  m["cuOccupancyMaxActiveBlocksPerMultiprocessor"]                       = {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",          "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY};
  // cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  m["cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]              = {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY};
  // cudaOccupancyMaxPotentialBlockSize
  m["cuOccupancyMaxPotentialBlockSize"]                                  = {"hipModuleOccupancyMaxPotentialBlockSize",                     "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY};
  // cudaOccupancyMaxPotentialBlockSizeWithFlags
  m["cuOccupancyMaxPotentialBlockSizeWithFlags"]                         = {"hipModuleOccupancyMaxPotentialBlockSizeWithFlags",            "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY};
  // cudaOccupancyMaxPotentialClusterSize
  m["cuOccupancyMaxPotentialClusterSize"]                                = {"hipOccupancyMaxPotentialClusterSize",                         "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY, HIP_UNSUPPORTED};
  // cudaOccupancyMaxActiveClusters
  m["cuOccupancyMaxActiveClusters"]                                      = {"hipOccupancyMaxActiveClusters",                               "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY, HIP_UNSUPPORTED};

  // 26. Texture Reference Management [DEPRECATED]
  // no analogues
  m["cuTexRefGetAddress"]                                                = {"hipTexRefGetAddress",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetAddress_v2"]                                             = {"hipTexRefGetAddress",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetAddressMode"]                                            = {"hipTexRefGetAddressMode",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetArray"]                                                  = {"hipTexRefGetArray",                                           "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, CUDA_DEPRECATED | HIP_DEPRECATED};
  m["cuTexRefGetBorderColor"]                                            = {"hipTexRefGetBorderColor",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, CUDA_DEPRECATED | HIP_DEPRECATED};
  m["cuTexRefGetFilterMode"]                                             = {"hipTexRefGetFilterMode",                                      "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetFlags"]                                                  = {"hipTexRefGetFlags",                                           "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetFormat"]                                                 = {"hipTexRefGetFormat",                                          "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetMaxAnisotropy"]                                          = {"hipTexRefGetMaxAnisotropy",                                   "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetMipmapFilterMode"]                                       = {"hipTexRefGetMipmapFilterMode",                                "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetMipmapLevelBias"]                                        = {"hipTexRefGetMipmapLevelBias",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefGetMipmapLevelClamp"]                                       = {"hipTexRefGetMipmapLevelClamp",                                "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  // TODO: [HIP] fix typo hipTexRefGetMipMappedArray -> hipTexRefGetMipmappedArray
  m["cuTexRefGetMipmappedArray"]                                         = {"hipTexRefGetMipMappedArray",                                  "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetAddress"]                                                = {"hipTexRefSetAddress",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetAddress_v2"]                                             = {"hipTexRefSetAddress",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetAddress2D"]                                              = {"hipTexRefSetAddress2D",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetAddress2D_v2"]                                           = {"hipTexRefSetAddress2D",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_DEPRECATED};
  m["cuTexRefSetAddress2D_v3"]                                           = {"hipTexRefSetAddress2D",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_DEPRECATED};
  m["cuTexRefSetAddressMode"]                                            = {"hipTexRefSetAddressMode",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetArray"]                                                  = {"hipTexRefSetArray",                                           "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetBorderColor"]                                            = {"hipTexRefSetBorderColor",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetFilterMode"]                                             = {"hipTexRefSetFilterMode",                                      "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetFlags"]                                                  = {"hipTexRefSetFlags",                                           "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetFormat"]                                                 = {"hipTexRefSetFormat",                                          "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetMaxAnisotropy"]                                          = {"hipTexRefSetMaxAnisotropy",                                   "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetMipmapFilterMode"]                                       = {"hipTexRefSetMipmapFilterMode",                                "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetMipmapLevelBias"]                                        = {"hipTexRefSetMipmapLevelBias",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetMipmapLevelClamp"]                                       = {"hipTexRefSetMipmapLevelClamp",                                "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefSetMipmappedArray"]                                         = {"hipTexRefSetMipmappedArray",                                  "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED};
  m["cuTexRefCreate"]                                                    = {"hipTexRefCreate",                                             "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cuTexRefDestroy"]                                                   = {"hipTexRefDestroy",                                            "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 27. Surface Reference Management [DEPRECATED]
  // no analogues
  m["cuSurfRefGetArray"]                                                 = {"hipSurfRefGetArray",                                          "", CONV_SURFACE, API_DRIVER, SEC::SURFACE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cuSurfRefSetArray"]                                                 = {"hipSurfRefSetArray",                                          "", CONV_SURFACE, API_DRIVER, SEC::SURFACE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 28. Texture Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateTextureObject due to different signatures
  m["cuTexObjectCreate"]                                                 = {"hipTexObjectCreate",                                          "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE};
  // cudaDestroyTextureObject
  m["cuTexObjectDestroy"]                                                = {"hipTexObjectDestroy",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE};
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectResourceDesc due to different signatures
  m["cuTexObjectGetResourceDesc"]                                        = {"hipTexObjectGetResourceDesc",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE};
  // cudaGetTextureObjectResourceViewDesc
  m["cuTexObjectGetResourceViewDesc"]                                    = {"hipTexObjectGetResourceViewDesc",                             "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE};
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  m["cuTexObjectGetTextureDesc"]                                         = {"hipTexObjectGetTextureDesc",                                  "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE};

  // 29. Surface Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateSurfaceObject due to different signatures
  m["cuSurfObjectCreate"]                                                = {"hipSurfObjectCreate",                                         "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED};
  // cudaDestroySurfaceObject
  m["cuSurfObjectDestroy"]                                               = {"hipSurfObjectDestroy",                                        "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED};
  // no analogue
  // NOTE: Not equal to cudaGetSurfaceObjectResourceDesc due to different signatures
  m["cuSurfObjectGetResourceDesc"]                                       = {"hipSurfObjectGetResourceDesc",                                "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED};

  // 30. Tensor Map Object Managment
  //
  m["cuTensorMapEncodeTiled"]                                            = {"hipTensorMapEncodeTiled",                                     "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED};
  //
  m["cuTensorMapEncodeIm2col"]                                           = {"hipTensorMapEncodeIm2col",                                    "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED};
  //
  m["cuTensorMapReplaceAddress"]                                         = {"hipTensorMapReplaceAddress",                                  "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED};
  //
  m["cuTensorMapEncodeIm2colWide"]                                       = {"hipTensorMapEncodeIm2colWide",                                "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED};

  // 31. Peer Context Memory Access
  // no analogue
  // NOTE: Not equal to cudaDeviceEnablePeerAccess due to different signatures
  m["cuCtxEnablePeerAccess"]                                             = {"hipCtxEnablePeerAccess",                                      "", CONV_PEER, API_DRIVER, SEC::PEER, HIP_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaDeviceDisablePeerAccess due to different signatures
  m["cuCtxDisablePeerAccess"]                                            = {"hipCtxDisablePeerAccess",                                     "", CONV_PEER, API_DRIVER, SEC::PEER, HIP_DEPRECATED};
  // cudaDeviceCanAccessPeer
  m["cuDeviceCanAccessPeer"]                                             = {"hipDeviceCanAccessPeer",                                      "", CONV_PEER, API_DRIVER, SEC::PEER};
  // cudaDeviceGetP2PAttribute
  m["cuDeviceGetP2PAttribute"]                                           = {"hipDeviceGetP2PAttribute",                                    "", CONV_PEER, API_DRIVER, SEC::PEER};
  // cudaDeviceGetP2PAtomicCapabilities
  m["cuDeviceGetP2PAtomicCapabilities"]                                  = {"hipDeviceGetP2PAtomicCapabilities",                           "", CONV_PEER, API_DRIVER, SEC::PEER, HIP_UNSUPPORTED};

  // 32. Graphics Interoperability
  // cudaGraphicsMapResources
  m["cuGraphicsMapResources"]                                            = {"hipGraphicsMapResources",                                     "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};
  // cudaGraphicsResourceGetMappedMipmappedArray
  m["cuGraphicsResourceGetMappedMipmappedArray"]                         = {"hipGraphicsResourceGetMappedMipmappedArray",                  "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED};
  // cudaGraphicsResourceGetMappedPointer
  m["cuGraphicsResourceGetMappedPointer"]                                = {"hipGraphicsResourceGetMappedPointer",                         "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};
  // cudaGraphicsResourceGetMappedPointer
  m["cuGraphicsResourceGetMappedPointer_v2"]                             = {"hipGraphicsResourceGetMappedPointer",                         "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};
  // cudaGraphicsResourceSetMapFlags
  m["cuGraphicsResourceSetMapFlags"]                                     = {"hipGraphicsResourceSetMapFlags",                              "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED};
  // cudaGraphicsResourceSetMapFlags
  m["cuGraphicsResourceSetMapFlags_v2"]                                  = {"hipGraphicsResourceSetMapFlags",                              "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED};
  // cudaGraphicsSubResourceGetMappedArray
  m["cuGraphicsSubResourceGetMappedArray"]                               = {"hipGraphicsSubResourceGetMappedArray",                        "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};
  // cudaGraphicsUnmapResources
  m["cuGraphicsUnmapResources"]                                          = {"hipGraphicsUnmapResources",                                   "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};
  // cudaGraphicsUnregisterResource
  m["cuGraphicsUnregisterResource"]                                      = {"hipGraphicsUnregisterResource",                               "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS};

  // 33. Driver Entry Point Access
  // cudaGetDriverEntryPoint
  m["cuGetProcAddress"]                                                  = {"hipGetProcAddress",                                           "", CONV_DRIVER_ENTRY_POINT, API_DRIVER, SEC::DRIVER_ENTRY_POINT};

  // 34. Coredump Attributes Control API
  //
  m["cuCoredumpGetAttribute"]                                            = {"hipCoredumpGetAttribute",                                     "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED};
  //
  m["cuCoredumpGetAttributeGlobal"]                                      = {"hipCoredumpGetAttributeGlobal",                               "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED};
  //
  m["cuCoredumpSetAttribute"]                                            = {"hipCoredumpSetAttribute",                                     "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED};
  //
  m["cuCoredumpSetAttributeGlobal"]                                      = {"hipCoredumpSetAttributeGlobal",                               "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED};

  // 35. Green Contexts
  //
  m["cuGreenCtxCreate"]                                                  = {"hipGreenCtxCreate",                                           "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxDestroy"]                                                 = {"hipGreenCtxDestroy",                                          "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuCtxFromGreenCtx"]                                                 = {"hipCtxFromGreenCtx",                                          "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuDeviceGetDevResource"]                                            = {"hipDeviceGetDevResource",                                     "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuCtxGetDevResource"]                                               = {"hipCtxGetDevResource",                                        "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxGetDevResource"]                                          = {"hipGreenCtxGetDevResource",                                   "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuDevSmResourceSplitByCount"]                                       = {"hipDevSmResourceSplitByCount",                                "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuDevResourceGenerateDesc"]                                         = {"hipDevResourceGenerateDesc",                                  "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxRecordEvent"]                                             = {"hipGreenCtxRecordEvent",                                      "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxWaitEvent"]                                               = {"hipGreenCtxWaitEvent",                                        "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuStreamGetGreenCtx"]                                               = {"hipStreamGetGreenCtx",                                        "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxStreamCreate"]                                            = {"hipGreenCtxStreamCreate",                                     "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};
  //
  m["cuGreenCtxGetId"]                                                   = {"hipGreenCtxGetId",                                            "", CONV_GREEN_CONTEXT, API_DRIVER, SEC::GREEN_CONTEXT, HIP_UNSUPPORTED};

  // 36. Error Log Management Functions
  // cudaLogsRegisterCallback
  m["cuLogsRegisterCallback"]                                            = {"hipLogsRegisterCallback",                                     "", CONV_ERROR_LOG, API_DRIVER, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cudaLogsUnregisterCallback
  m["cuLogsUnregisterCallback"]                                          = {"hipLogsUnregisterCallback",                                   "", CONV_ERROR_LOG, API_DRIVER, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cudaLogsCurrent
  m["cuLogsCurrent"]                                                     = {"hipLogsCurrent",                                              "", CONV_ERROR_LOG, API_DRIVER, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cudaLogsDumpToFile
  m["cuLogsDumpToFile"]                                                  = {"hipLogsDumpToFile",                                           "", CONV_ERROR_LOG, API_DRIVER, SEC::ERROR_LOG, HIP_UNSUPPORTED};
  // cudaLogsDumpToMemory
  m["cuLogsDumpToMemory"]                                                = {"hipLogsDumpToMemory",                                         "", CONV_ERROR_LOG, API_DRIVER, SEC::ERROR_LOG, HIP_UNSUPPORTED};

  // 37. Checkpointing
  //
  m["cuCheckpointProcessGetRestoreThreadId"]                             = {"hipCheckpointProcessGetRestoreThreadId",                      "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};
  //
  m["cuCheckpointProcessGetState"]                                       = {"hipCheckpointProcessGetState",                                "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};
  //
  m["cuCheckpointProcessLock"]                                           = {"hipCheckpointProcessLock",                                    "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};
  //
  m["cuCheckpointProcessCheckpoint"]                                     = {"hipCheckpointProcessCheckpoint",                              "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};
  //
  m["cuCheckpointProcessRestore"]                                        = {"hipCheckpointProcessRestore",                                 "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};
  //
  m["cuCheckpointProcessUnlock"]                                         = {"hipCheckpointProcessUnlock",                                  "", CONV_COREDUMP, API_DRIVER, SEC::CHECKPOINTING, HIP_UNSUPPORTED};

  // 38. Profiler Control [DEPRECATED]
  // cudaProfilerInitialize
  m["cuProfilerInitialize"]                                              = {"hipProfilerInitialize",                                       "", CONV_PROFILER, API_DRIVER, SEC::PROFILER_DEPRECATED, HIP_UNSUPPORTED};

  // 39. Profiler Control
  // cudaProfilerStart
  m["cuProfilerStart"]                                                   = {"hipProfilerStart",                                            "", CONV_PROFILER, API_DRIVER, SEC::PROFILER};
  // cudaProfilerStop
  m["cuProfilerStop"]                                                    = {"hipProfilerStop",                                             "", CONV_PROFILER, API_DRIVER, SEC::PROFILER};

  // 40. OpenGL Interoperability
  // cudaGLGetDevices
  m["cuGLGetDevices"]                                                    = {"hipGLGetDevices",                                             "", CONV_OPENGL, API_DRIVER, SEC::OPENGL};
  // cudaGraphicsGLRegisterBuffer
  m["cuGraphicsGLRegisterBuffer"]                                        = {"hipGraphicsGLRegisterBuffer",                                 "", CONV_OPENGL, API_DRIVER, SEC::OPENGL};
  // cudaGraphicsGLRegisterImage
  m["cuGraphicsGLRegisterImage"]                                         = {"hipGraphicsGLRegisterImage",                                  "", CONV_OPENGL, API_DRIVER, SEC::OPENGL};
  // cudaWGLGetDevice
  m["cuWGLGetDevice"]                                                    = {"hipWGLGetDevice",                                             "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED};

  // 40. OpenGL Interoperability [DEPRECATED]
  // no analogue
  m["cuGLCtxCreate"]                                                     = {"hipGLCtxCreate",                                              "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuGLInit"]                                                          = {"hipGLInit",                                                   "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObject due to different signatures
  m["cuGLMapBufferObject"]                                               = {"hipGLMapBufferObject_",                                       "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObjectAsync due to different signatures
  m["cuGLMapBufferObjectAsync"]                                          = {"hipGLMapBufferObjectAsync_",                                  "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaGLRegisterBufferObject
  m["cuGLRegisterBufferObject"]                                          = {"hipGLRegisterBufferObject",                                   "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaGLSetBufferObjectMapFlags
  m["cuGLSetBufferObjectMapFlags"]                                       = {"hipGLSetBufferObjectMapFlags",                                "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaGLUnmapBufferObject
  m["cuGLUnmapBufferObject"]                                             = {"hipGLUnmapBufferObject",                                      "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaGLUnmapBufferObjectAsync
  m["cuGLUnmapBufferObjectAsync"]                                        = {"hipGLUnmapBufferObjectAsync",                                 "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaGLUnregisterBufferObject
  m["cuGLUnregisterBufferObject"]                                        = {"hipGLUnregisterBufferObject",                                 "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 41. Direct3D 9 Interoperability
  // no analogue
  m["cuD3D9CtxCreate"]                                                   = {"hipD3D9CtxCreate",                                            "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};
    // no analogue
  m["cuD3D9CtxCreateOnDevice"]                                           = {"hipD3D9CtxCreateOnDevice",                                    "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};
  // cudaD3D9GetDevice
  m["cuD3D9GetDevice"]                                                   = {"hipD3D9GetDevice",                                            "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};
  // cudaD3D9GetDevices
  m["cuD3D9GetDevices"]                                                  = {"hipD3D9GetDevices",                                           "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};
  // cudaD3D9GetDirect3DDevice
  m["cuD3D9GetDirect3DDevice"]                                           = {"hipD3D9GetDirect3DDevice",                                    "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};
  // cudaGraphicsD3D9RegisterResource
  m["cuGraphicsD3D9RegisterResource"]                                    = {"hipGraphicsD3D9RegisterResource",                             "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED};

  // 41. Direct3D 9 Interoperability [DEPRECATED]
  // cudaD3D9MapResources
  m["cuD3D9MapResources"]                                                = {"hipD3D9MapResources",                                         "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9RegisterResource
  m["cuD3D9RegisterResource"]                                            = {"hipD3D9RegisterResource",                                     "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceGetMappedArray
  m["cuD3D9ResourceGetMappedArray"]                                      = {"hipD3D9ResourceGetMappedArray",                               "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceGetMappedPitch
  m["cuD3D9ResourceGetMappedPitch"]                                      = {"hipD3D9ResourceGetMappedPitch",                               "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceGetMappedPointer
  m["cuD3D9ResourceGetMappedPointer"]                                    = {"hipD3D9ResourceGetMappedPointer",                             "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceGetMappedSize
  m["cuD3D9ResourceGetMappedSize"]                                       = {"hipD3D9ResourceGetMappedSize",                                "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceGetSurfaceDimensions
  m["cuD3D9ResourceGetSurfaceDimensions"]                                = {"hipD3D9ResourceGetSurfaceDimensions",                         "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9ResourceSetMapFlags
  m["cuD3D9ResourceSetMapFlags"]                                         = {"hipD3D9ResourceSetMapFlags",                                  "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9UnmapResources
  m["cuD3D9UnmapResources"]                                              = {"hipD3D9UnmapResources",                                       "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D9UnregisterResource
  m["cuD3D9UnregisterResource"]                                          = {"hipD3D9UnregisterResource",                                   "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 42. Direct3D 10 Interoperability
  // cudaD3D10GetDevice
  m["cuD3D10GetDevice"]                                                  = {"hipD3D10GetDevice",                                           "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED};
  // cudaD3D10GetDevices
  m["cuD3D10GetDevices"]                                                 = {"hipD3D10GetDevices",                                          "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED};
  // cudaGraphicsD3D10RegisterResource
  m["cuGraphicsD3D10RegisterResource"]                                   = {"hipGraphicsD3D10RegisterResource",                            "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED};

  // 42. Direct3D 10 Interoperability [DEPRECATED]
  // no analogue
  m["cuD3D10CtxCreate"]                                                  = {"hipD3D10CtxCreate",                                           "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuD3D10CtxCreateOnDevice"]                                          = {"hipD3D10CtxCreateOnDevice",                                   "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10GetDirect3DDevice
  m["cuD3D10GetDirect3DDevice"]                                          = {"hipD3D10GetDirect3DDevice",                                   "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10MapResources
  m["cuD3D10MapResources"]                                               = {"hipD3D10MapResources",                                        "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10RegisterResource
  m["cuD3D10RegisterResource"]                                           = {"hipD3D10RegisterResource",                                    "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceGetMappedArray
  m["cuD3D10ResourceGetMappedArray"]                                     = {"hipD3D10ResourceGetMappedArray",                              "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceGetMappedPitch
  m["cuD3D10ResourceGetMappedPitch"]                                     = {"hipD3D10ResourceGetMappedPitch",                              "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceGetMappedPointer
  m["cuD3D10ResourceGetMappedPointer"]                                   = {"hipD3D10ResourceGetMappedPointer",                            "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceGetMappedSize
  m["cuD3D10ResourceGetMappedSize"]                                      = {"hipD3D10ResourceGetMappedSize",                               "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceGetSurfaceDimensions
  m["cuD3D10ResourceGetSurfaceDimensions"]                               = {"hipD3D10ResourceGetSurfaceDimensions",                        "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10ResourceSetMapFlags
  m["cuD3D10ResourceSetMapFlags"]                                        = {"hipD3D10ResourceSetMapFlags",                                 "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10UnmapResources
  m["cuD3D10UnmapResources"]                                             = {"hipD3D10UnmapResources",                                      "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D10UnregisterResource
  m["cuD3D10UnregisterResource"]                                         = {"hipD3D10UnregisterResource",                                  "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 43. Direct3D 11 Interoperability
  // cudaD3D11GetDevice
  m["cuD3D11GetDevice"]                                                  = {"hipD3D11GetDevice",                                           "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED};
  // cudaD3D11GetDevices
  m["cuD3D11GetDevices"]                                                 = {"hipD3D11GetDevices",                                          "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED};
  // cudaGraphicsD3D11RegisterResource
  m["cuGraphicsD3D11RegisterResource"]                                   = {"hipGraphicsD3D11RegisterResource",                            "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED};

  // 43. Direct3D 11 Interoperability [DEPRECATED]
  // no analogue
  m["cuD3D11CtxCreate"]                                                  = {"hipD3D11CtxCreate",                                           "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // no analogue
  m["cuD3D11CtxCreateOnDevice"]                                          = {"hipD3D11CtxCreateOnDevice",                                   "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  // cudaD3D11GetDirect3DDevice
  m["cuD3D11GetDirect3DDevice"]                                          = {"hipD3D11GetDirect3DDevice",                                   "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED};

  // 44. VDPAU Interoperability
  // cudaGraphicsVDPAURegisterOutputSurface
  m["cuGraphicsVDPAURegisterOutputSurface"]                              = {"hipGraphicsVDPAURegisterOutputSurface",                       "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED};
  // cudaGraphicsVDPAURegisterVideoSurface
  m["cuGraphicsVDPAURegisterVideoSurface"]                               = {"hipGraphicsVDPAURegisterVideoSurface",                        "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED};
  // cudaVDPAUGetDevice
  m["cuVDPAUGetDevice"]                                                  = {"hipVDPAUGetDevice",                                           "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED};
  // no analogue
  m["cuVDPAUCtxCreate"]                                                  = {"hipVDPAUCtxCreate",                                           "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED};

  // 45. EGL Interoperability
  // cudaEGLStreamConsumerAcquireFrame
  m["cuEGLStreamConsumerAcquireFrame"]                                   = {"hipEGLStreamConsumerAcquireFrame",                            "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamConsumerConnect
  m["cuEGLStreamConsumerConnect"]                                        = {"hipEGLStreamConsumerConnect",                                 "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamConsumerConnectWithFlags
  m["cuEGLStreamConsumerConnectWithFlags"]                               = {"hipEGLStreamConsumerConnectWithFlags",                        "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamConsumerDisconnect
  m["cuEGLStreamConsumerDisconnect"]                                     = {"hipEGLStreamConsumerDisconnect",                              "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamConsumerReleaseFrame
  m["cuEGLStreamConsumerReleaseFrame"]                                   = {"hipEGLStreamConsumerReleaseFrame",                            "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamProducerConnect
  m["cuEGLStreamProducerConnect"]                                        = {"hipEGLStreamProducerConnect",                                 "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamProducerDisconnect
  m["cuEGLStreamProducerDisconnect"]                                     = {"hipEGLStreamProducerDisconnect",                              "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamProducerPresentFrame
  m["cuEGLStreamProducerPresentFrame"]                                   = {"hipEGLStreamProducerPresentFrame",                            "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEGLStreamProducerReturnFrame
  m["cuEGLStreamProducerReturnFrame"]                                    = {"hipEGLStreamProducerReturnFrame",                             "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaGraphicsEGLRegisterImage
  m["cuGraphicsEGLRegisterImage"]                                        = {"hipGraphicsEGLRegisterImage",                                 "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaGraphicsResourceGetMappedEglFrame
  m["cuGraphicsResourceGetMappedEglFrame"]                               = {"hipGraphicsResourceGetMappedEglFrame",                        "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};
  // cudaEventCreateFromEGLSync
  m["cuEventCreateFromEGLSync"]                                          = {"hipEventCreateFromEGLSync",                                   "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DRIVER_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cuDeviceGetLuid"]                                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuDeviceGetNvSciSyncAttributes"]                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuDeviceGetUuid"]                                                   = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cuDeviceComputeCapability"]                                         = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuDeviceGetProperties"]                                             = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuDevicePrimaryCtxRelease_v2"]                                      = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuDevicePrimaryCtxReset_v2"]                                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuDevicePrimaryCtxSetFlags_v2"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuCtxResetPersistingL2Cache"]                                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuMemAddressFree"]                                                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemAddressReserve"]                                               = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemCreate"]                                                       = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemExportToShareableHandle"]                                      = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemGetAccess"]                                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemGetAllocationGranularity"]                                     = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemGetAllocationPropertiesFromHandle"]                            = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemImportFromShareableHandle"]                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemMap"]                                                          = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemRelease"]                                                      = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemRetainAllocationHandle"]                                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuMemSetAccess"]                                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemUnmap"]                                                        = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuMemAdvise"]                                                       = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuMemPrefetchAsync"]                                                = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuMemRangeGetAttribute"]                                            = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuMemRangeGetAttributes"]                                           = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuStreamBeginCapture"]                                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuStreamBeginCapture_v2"]                                           = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cuStreamBeginCapture_ptsz"]                                         = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cuStreamCopyAttributes"]                                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuStreamEndCapture"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuStreamGetAttribute"]                                              = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuStreamGetCaptureInfo"]                                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cuStreamGetCtx"]                                                    = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cuStreamIsCapturing"]                                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuStreamSetAttribute"]                                              = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuThreadExchangeStreamCaptureMode"]                                 = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cuDestroyExternalMemory"]                                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuDestroyExternalSemaphore"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuExternalMemoryGetMappedBuffer"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuExternalMemoryGetMappedMipmappedArray"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuImportExternalMemory"]                                            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuImportExternalSemaphore"]                                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuSignalExternalSemaphoresAsync"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuWaitExternalSemaphoresAsync"]                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuStreamBatchMemOp"]                                                = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuStreamWaitValue32"]                                               = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuStreamWaitValue64"]                                               = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cuStreamWriteValue32"]                                              = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuStreamWriteValue64"]                                              = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cuFuncGetModule"]                                                   = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuFuncSetAttribute"]                                                = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cuLaunchCooperativeKernel"]                                         = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cuLaunchCooperativeKernelMultiDevice"]                              = {CUDA_90,  CUDA_113, CUDA_0  };
  m["cuLaunchHostFunc"]                                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuFuncSetBlockShape"]                                               = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuFuncSetSharedSize"]                                               = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuLaunch"]                                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuLaunchGrid"]                                                      = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuLaunchGridAsync"]                                                 = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuParamSetf"]                                                       = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuParamSeti"]                                                       = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuParamSetSize"]                                                    = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuParamSetTexRef"]                                                  = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuParamSetv"]                                                       = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGraphAddChildGraphNode"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddDependencies"]                                            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddEmptyNode"]                                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddHostNode"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddKernelNode"]                                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddMemcpyNode"]                                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphAddMemsetNode"]                                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphChildGraphNodeGetGraph"]                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphClone"]                                                      = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphCreate"]                                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphDestroy"]                                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphDestroyNode"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphExecDestroy"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphGetEdges"]                                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphGetNodes"]                                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphGetRootNodes"]                                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphHostNodeGetParams"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphHostNodeSetParams"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphInstantiate"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphInstantiate_v2"]                                             = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuGraphKernelNodeCopyAttributes"]                                   = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuGraphKernelNodeGetAttribute"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuGraphExecKernelNodeSetParams"]                                    = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cuGraphKernelNodeGetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphKernelNodeSetAttribute"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuGraphKernelNodeSetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphLaunch"]                                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphMemcpyNodeGetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphMemcpyNodeSetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphMemsetNodeGetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphMemsetNodeSetParams"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphNodeFindInClone"]                                            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetDependencies"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetDependentNodes"]                                      = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetType"]                                                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphRemoveDependencies"]                                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cuGraphExecMemcpyNodeSetParams"]                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuGraphExecMemsetNodeSetParams"]                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuGraphExecHostNodeSetParams"]                                      = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuGraphExecUpdate"]                                                 = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cuOccupancyAvailableDynamicSMemPerBlock"]                           = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cuTexRefGetAddress"]                                                = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetAddress_v2"]                                             = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetAddressMode"]                                            = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetArray"]                                                  = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetBorderColor"]                                            = {CUDA_80,  CUDA_110, CUDA_0  };
  m["cuTexRefGetFilterMode"]                                             = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetFlags"]                                                  = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetFormat"]                                                 = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetMaxAnisotropy"]                                          = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetMipmapFilterMode"]                                       = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetMipmapLevelBias"]                                        = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetMipmapLevelClamp"]                                       = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefGetMipmappedArray"]                                         = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetAddress"]                                                = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetAddress_v2"]                                             = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetAddress2D"]                                              = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetAddressMode"]                                            = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetArray"]                                                  = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetBorderColor"]                                            = {CUDA_80,  CUDA_110, CUDA_0  };
  m["cuTexRefSetFilterMode"]                                             = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetFlags"]                                                  = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetFormat"]                                                 = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetMaxAnisotropy"]                                          = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetMipmapFilterMode"]                                       = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetMipmapLevelBias"]                                        = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetMipmapLevelClamp"]                                       = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefSetMipmappedArray"]                                         = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefCreate"]                                                    = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuTexRefDestroy"]                                                   = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuSurfRefGetArray"]                                                 = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuSurfRefSetArray"]                                                 = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuDeviceGetP2PAttribute"]                                           = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cuProfilerInitialize"]                                              = {CUDA_0,   CUDA_110, CUDA_0  };
  m["cuGLCtxCreate"]                                                     = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLInit"]                                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLMapBufferObject"]                                               = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLMapBufferObjectAsync"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLRegisterBufferObject"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLSetBufferObjectMapFlags"]                                       = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLUnmapBufferObject"]                                             = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLUnmapBufferObjectAsync"]                                        = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuGLUnregisterBufferObject"]                                        = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9MapResources"]                                                = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9RegisterResource"]                                            = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceGetMappedArray"]                                      = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceGetMappedPitch"]                                      = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceGetMappedPointer"]                                    = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceGetMappedSize"]                                       = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceGetSurfaceDimensions"]                                = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9ResourceSetMapFlags"]                                         = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9UnmapResources"]                                              = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D9UnregisterResource"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10CtxCreate"]                                                  = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10CtxCreateOnDevice"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10GetDirect3DDevice"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10MapResources"]                                               = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10RegisterResource"]                                           = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceGetMappedArray"]                                     = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceGetMappedPitch"]                                     = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceGetMappedPointer"]                                   = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceGetMappedSize"]                                      = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceGetSurfaceDimensions"]                               = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10ResourceSetMapFlags"]                                        = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10UnmapResources"]                                             = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D10UnregisterResource"]                                         = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D11CtxCreate"]                                                  = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D11CtxCreateOnDevice"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuD3D11GetDirect3DDevice"]                                          = {CUDA_0,   CUDA_92,  CUDA_0  };
  m["cuEGLStreamConsumerAcquireFrame"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamConsumerConnect"]                                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamConsumerConnectWithFlags"]                               = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamConsumerDisconnect"]                                     = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamConsumerReleaseFrame"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamProducerConnect"]                                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamProducerDisconnect"]                                     = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamProducerPresentFrame"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEGLStreamProducerReturnFrame"]                                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuGraphicsEGLRegisterImage"]                                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuGraphicsResourceGetMappedEglFrame"]                               = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuEventCreateFromEGLSync"]                                          = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cuDeviceGetTexture1DLinearMaxWidth"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuArrayGetSparseProperties"]                                        = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuMemMapArrayAsync"]                                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuEventRecordWithFlags"]                                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphAddEventRecordNode"]                                         = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphEventRecordNodeGetEvent"]                                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphEventRecordNodeSetEvent"]                                    = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphAddEventWaitNode"]                                           = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphEventWaitNodeGetEvent"]                                      = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphEventWaitNodeSetEvent"]                                      = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphExecChildGraphNodeSetParams"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphExecEventRecordNodeSetEvent"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphExecEventWaitNodeSetEvent"]                                  = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuGraphUpload"]                                                     = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cuDeviceSetMemPool"]                                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuDeviceGetMemPool"]                                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuDeviceGetDefaultMemPool"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuArrayGetPlane"]                                                   = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemFreeAsync"]                                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemAllocAsync"]                                                   = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolTrimTo"]                                                   = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolSetAttribute"]                                             = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolGetAttribute"]                                             = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolSetAccess"]                                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolGetAccess"]                                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolCreate"]                                                   = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolDestroy"]                                                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemAllocFromPoolAsync"]                                           = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolExportToShareableHandle"]                                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolImportFromShareableHandle"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolExportPointer"]                                            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuMemPoolImportPointer"]                                            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphAddExternalSemaphoresSignalNode"]                            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExternalSemaphoresSignalNodeGetParams"]                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExternalSemaphoresSignalNodeSetParams"]                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphAddExternalSemaphoresWaitNode"]                              = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExternalSemaphoresWaitNodeGetParams"]                        = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExternalSemaphoresWaitNodeSetParams"]                        = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExecExternalSemaphoresSignalNodeSetParams"]                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuGraphExecExternalSemaphoresWaitNodeSetParams"]                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cuStreamGetCaptureInfo_v2"]                                         = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuStreamUpdateCaptureDependencies"]                                 = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuGraphDebugDotPrint"]                                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuUserObjectCreate"]                                                = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuUserObjectRetain"]                                                = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuUserObjectRelease"]                                               = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuGraphRetainUserObject"]                                           = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuGraphReleaseUserObject"]                                          = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuGetProcAddress"]                                                  = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuFlushGPUDirectRDMAWrites"]                                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cuCtxCreate_v3"]                                                    = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuDeviceGetUuid_v2"]                                                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuDeviceGetExecAffinitySupport"]                                    = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuCtxGetExecAffinity"]                                              = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuGraphAddMemAllocNode"]                                            = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuGraphMemAllocNodeGetParams"]                                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuGraphAddMemFreeNode"]                                             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuGraphMemFreeNodeGetParams"]                                       = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuDeviceGraphMemTrim"]                                              = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuDeviceGetGraphMemAttribute"]                                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuDeviceSetGraphMemAttribute"]                                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuGraphInstantiateWithFlags"]                                       = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cuArrayGetMemoryRequirements"]                                      = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cuMipmappedArrayGetMemoryRequirements"]                             = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cuGraphNodeSetEnabled"]                                             = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetEnabled"]                                             = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cuMemGetHandleForAddressRange"]                                     = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuModuleGetLoadingMode"]                                            = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuStreamWaitValue32_v2"]                                            = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuStreamWaitValue64_v2"]                                            = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuStreamWriteValue32_v2"]                                           = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuStreamWriteValue64_v2"]                                           = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuStreamBatchMemOp_v2"]                                             = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuGraphAddBatchMemOpNode"]                                          = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuGraphBatchMemOpNodeGetParams"]                                    = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuGraphBatchMemOpNodeSetParams"]                                    = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuGraphExecBatchMemOpNodeSetParams"]                                = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cuLaunchKernelEx"]                                                  = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cuOccupancyMaxPotentialClusterSize"]                                = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cuOccupancyMaxActiveClusters"]                                      = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cuCtxGetId"]                                                        = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuModuleGetTexRef"]                                                 = {CUDA_0,   CUDA_120, CUDA_0  };
  m["cuModuleGetSurfRef"]                                                = {CUDA_0,   CUDA_120, CUDA_0  };
  m["cuLibraryLoadData"]                                                 = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryLoadFromFile"]                                             = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryUnload"]                                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryGetKernel"]                                                = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryGetModule"]                                                = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuKernelGetFunction"]                                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryGetGlobal"]                                                = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryGetManaged"]                                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuLibraryGetUnifiedFunction"]                                       = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuKernelGetAttribute"]                                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuKernelSetAttribute"]                                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuKernelSetCacheConfig"]                                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuStreamGetId"]                                                     = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuGraphInstantiateWithParams"]                                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuTensorMapEncodeTiled"]                                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuTensorMapEncodeIm2col"]                                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuTensorMapReplaceAddress"]                                         = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuGraphExecGetFlags"]                                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cuCtxSetFlags"]                                                     = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastCreate"]                                                 = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastAddDevice"]                                              = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastBindMem"]                                                = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastBindAddr"]                                               = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastUnbind"]                                                 = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMulticastGetGranularity"]                                         = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuCoredumpGetAttribute"]                                            = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuCoredumpGetAttributeGlobal"]                                      = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuCoredumpSetAttribute"]                                            = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuCoredumpSetAttributeGlobal"]                                      = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cuMemPrefetchAsync_v2"]                                             = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cuMemAdvise_v2"]                                                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cuGraphAddNode"]                                                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cuGraphNodeSetParams"]                                              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cuGraphExecNodeSetParams"]                                          = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cuKernelGetName"]                                                   = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuStreamBeginCaptureToGraph"]                                       = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuStreamGetCaptureInfo_v3"]                                         = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuStreamUpdateCaptureDependencies_v2"]                              = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuFuncGetName"]                                                     = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphGetEdges_v2"]                                                = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetDependencies_v2"]                                     = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphAddDependencies_v2"]                                         = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphRemoveDependencies_v2"]                                      = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphAddNode_v2"]                                                 = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphConditionalHandleCreate"]                                    = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuGraphNodeGetDependentNodes_v2"]                                   = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cuCtxGetSharedMemConfig"]                                           = {CUDA_0,   CUDA_0,   CUDA_124};
  m["cuCtxSetSharedMemConfig"]                                           = {CUDA_0,   CUDA_0,   CUDA_124};
  m["cuModuleGetFunctionCount"]                                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuModuleEnumerateFunctions"]                                        = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuLibraryGetKernelCount"]                                           = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuLibraryEnumerateKernels"]                                         = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuKernelGetParamInfo"]                                              = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuDeviceRegisterAsyncNotification"]                                 = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuDeviceUnregisterAsyncNotification"]                               = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuFuncSetSharedMemConfig"]                                          = {CUDA_0,   CUDA_0,   CUDA_124};
  m["cuFuncGetParamInfo"]                                                = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuFuncIsLoaded"]                                                    = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuFuncLoad"]                                                        = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuGreenCtxCreate"]                                                  = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuGreenCtxDestroy"]                                                 = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuCtxFromGreenCtx"]                                                 = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuDeviceGetDevResource"]                                            = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuCtxGetDevResource"]                                               = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuGreenCtxGetDevResource"]                                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuDevSmResourceSplitByCount"]                                       = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuDevResourceGenerateDesc"]                                         = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuGreenCtxRecordEvent"]                                             = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuGreenCtxWaitEvent"]                                               = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuStreamGetGreenCtx"]                                               = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cuCtxCreate_v4"]                                                    = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cuCtxWaitEvent"]                                                    = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cuKernelGetLibrary"]                                                = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cuStreamGetCtx_v2"]                                                 = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cuGreenCtxStreamCreate"]                                            = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cuMemcpyBatchAsync"]                                                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuMemcpy3DBatchAsync"]                                              = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuMemBatchDecompressAsync"]                                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuStreamGetDevice"]                                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuEventElapsedTime_v2"]                                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessGetRestoreThreadId"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessGetState"]                                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessLock"]                                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessCheckpoint"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessRestore"]                                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuCheckpointProcessUnlock"]                                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cuLogsRegisterCallback"]                                            = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cuLogsUnregisterCallback"]                                          = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cuLogsCurrent"]                                                     = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cuLogsDumpToFile"]                                                  = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cuLogsDumpToMemory"]                                                = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cuDeviceGetHostAtomicCapabilities"]                                 = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuCtxGetDevice_v2"]                                                 = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuCtxSynchronize_v2"]                                               = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemGetDefaultMemPool"]                                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemGetMemPool"]                                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemSetMemPool"]                                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemPrefetchBatchAsync"]                                           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemDiscardBatchAsync"]                                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuMemDiscardAndPrefetchBatchAsync"]                                 = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuGreenCtxGetId"]                                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cuDeviceGetP2PAtomicCapabilities"]                                  = {CUDA_130, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_DRIVER_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipInit"]                                                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDriverGetVersion"]                                               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGet"]                                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceGetName"]                                                  = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceTotalMem"]                                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDeviceComputeCapability"]                                        = {HIP_1060, HIP_0,    HIP_0   };
  m["hipDevicePrimaryCtxGetState"]                                       = {HIP_1090, HIP_6010, HIP_0   };
  m["hipDevicePrimaryCtxRelease"]                                        = {HIP_1090, HIP_6010, HIP_0   };
  m["hipDevicePrimaryCtxReset"]                                          = {HIP_1090, HIP_6010, HIP_0   };
  m["hipDevicePrimaryCtxRetain"]                                         = {HIP_1090, HIP_6010, HIP_0   };
  m["hipDevicePrimaryCtxSetFlags"]                                       = {HIP_1090, HIP_6010, HIP_0   };
  m["hipCtxCreate"]                                                      = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxDestroy"]                                                     = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxGetApiVersion"]                                               = {HIP_1090, HIP_1090, HIP_0   };
  m["hipCtxGetCacheConfig"]                                              = {HIP_1090, HIP_1090, HIP_0   };
  m["hipCtxGetCurrent"]                                                  = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxGetDevice"]                                                   = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxGetFlags"]                                                    = {HIP_1090, HIP_1090, HIP_0   };
  m["hipCtxGetSharedMemConfig"]                                          = {HIP_1090, HIP_1090, HIP_0   };
  m["hipDeviceGetStreamPriorityRange"]                                   = {HIP_2000, HIP_0,    HIP_0   };
  m["hipCtxPopCurrent"]                                                  = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxPushCurrent"]                                                 = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxSetCacheConfig"]                                              = {HIP_1090, HIP_1090, HIP_0   };
  m["hipCtxSetCurrent"]                                                  = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxSetSharedMemConfig"]                                          = {HIP_1090, HIP_1090, HIP_0   };
  m["hipCtxSynchronize"]                                                 = {HIP_1090, HIP_1090, HIP_0   };
  m["hipModuleGetFunction"]                                              = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleGetGlobal"]                                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleGetTexRef"]                                                = {HIP_1070, HIP_0,    HIP_0   };
  m["hipModuleLoad"]                                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleLoadData"]                                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleLoadDataEx"]                                               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleUnload"]                                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipArray3DCreate"]                                                  = {HIP_1071, HIP_0,    HIP_0   };
  m["hipArrayCreate"]                                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemAllocPitch"]                                                  = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMemAllocHost"]                                                   = {HIP_3000, HIP_3000, HIP_0   };
  m["hipMemcpyParam2D"]                                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMemcpyParam2DAsync"]                                             = {HIP_2080, HIP_0,    HIP_0   };
  m["hipDrvMemcpy3D"]                                                    = {HIP_3050, HIP_0,    HIP_0   };
  m["hipDrvMemcpy3DAsync"]                                               = {HIP_3050, HIP_0,    HIP_0   };
  m["hipMemcpyAtoH"]                                                     = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemcpyDtoD"]                                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyDtoDAsync"]                                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyDtoH"]                                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyDtoHAsync"]                                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyHtoA"]                                                     = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemcpyHtoD"]                                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemcpyHtoDAsync"]                                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemGetAddressRange"]                                             = {HIP_1090, HIP_0,    HIP_0   };
  m["hipMemsetD16"]                                                      = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMemsetD16Async"]                                                 = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMemsetD32"]                                                      = {HIP_2030, HIP_0,    HIP_0   };
  m["hipMemsetD32Async"]                                                 = {HIP_2030, HIP_0,    HIP_0   };
  m["hipMemsetD8"]                                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMemsetD8Async"]                                                  = {HIP_3000, HIP_0,    HIP_0   };
  m["hipMipmappedArrayCreate"]                                           = {HIP_3050, HIP_5070, HIP_0   };
  m["hipMipmappedArrayDestroy"]                                          = {HIP_3050, HIP_5070, HIP_0   };
  m["hipMipmappedArrayGetLevel"]                                         = {HIP_3050, HIP_5070, HIP_0   };
  m["hipFuncGetAttribute"]                                               = {HIP_2080, HIP_0,    HIP_0   };
  m["hipModuleLaunchKernel"]                                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipModuleOccupancyMaxActiveBlocksPerMultiprocessor"]                = {HIP_3050, HIP_0,    HIP_0   };
  m["hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"]       = {HIP_3050, HIP_0,    HIP_0   };
  m["hipModuleOccupancyMaxPotentialBlockSize"]                           = {HIP_3050, HIP_0,    HIP_0   };
  m["hipModuleOccupancyMaxPotentialBlockSizeWithFlags"]                  = {HIP_3050, HIP_0,    HIP_0   };
  m["hipTexRefGetAddress"]                                               = {HIP_3000, HIP_4030, HIP_0   };
  m["hipTexRefGetAddressMode"]                                           = {HIP_3000, HIP_4030, HIP_0   };
  m["hipTexRefGetArray"]                                                 = {HIP_3000, HIP_6010, HIP_0   };
  m["hipTexRefGetFilterMode"]                                            = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetFlags"]                                                 = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetFormat"]                                                = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetMaxAnisotropy"]                                         = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetMipmapFilterMode"]                                      = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetMipmapLevelBias"]                                       = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetMipmapLevelClamp"]                                      = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefGetMipMappedArray"]                                        = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefSetAddress"]                                               = {HIP_1070, HIP_4030, HIP_0   };
  m["hipTexRefSetAddress2D"]                                             = {HIP_1070, HIP_4030, HIP_0   };
  m["hipTexRefSetAddressMode"]                                           = {HIP_1090, HIP_5030, HIP_0   };
  m["hipTexRefSetArray"]                                                 = {HIP_1090, HIP_5030, HIP_0   };
  m["hipTexRefSetBorderColor"]                                           = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefSetFilterMode"]                                            = {HIP_1090, HIP_5030, HIP_0   };
  m["hipTexRefSetFlags"]                                                 = {HIP_1090, HIP_5030, HIP_0   };
  m["hipTexRefSetFormat"]                                                = {HIP_1090, HIP_5030, HIP_0   };
  m["hipTexRefSetMaxAnisotropy"]                                         = {HIP_3050, HIP_4030, HIP_0   };
  m["hipTexRefSetMipmapFilterMode"]                                      = {HIP_3050, HIP_5030, HIP_0   };
  m["hipTexRefSetMipmapLevelBias"]                                       = {HIP_3050, HIP_5030, HIP_0   };
  m["hipTexRefSetMipmapLevelClamp"]                                      = {HIP_3050, HIP_5030, HIP_0   };
  m["hipTexRefSetMipmappedArray"]                                        = {HIP_3050, HIP_5030, HIP_0   };
  m["hipTexObjectCreate"]                                                = {HIP_3050, HIP_0,    HIP_0   };
  m["hipTexObjectDestroy"]                                               = {HIP_3050, HIP_0,    HIP_0   };
  m["hipTexObjectGetResourceDesc"]                                       = {HIP_3050, HIP_0,    HIP_0   };
  m["hipTexObjectGetResourceViewDesc"]                                   = {HIP_3050, HIP_0,    HIP_0   };
  m["hipTexObjectGetTextureDesc"]                                        = {HIP_3050, HIP_0,    HIP_0   };
  m["hipCtxEnablePeerAccess"]                                            = {HIP_1060, HIP_1090, HIP_0   };
  m["hipCtxDisablePeerAccess"]                                           = {HIP_1060, HIP_1090, HIP_0   };
  m["hipStreamWaitValue32"]                                              = {HIP_4020, HIP_0,    HIP_0   };
  m["hipStreamWaitValue64"]                                              = {HIP_4020, HIP_0,    HIP_0   };
  m["hipStreamWriteValue32"]                                             = {HIP_4020, HIP_0,    HIP_0   };
  m["hipStreamWriteValue64"]                                             = {HIP_4020, HIP_0,    HIP_0   };
  m["hipArrayDestroy"]                                                   = {HIP_4020, HIP_0,    HIP_0   };
  m["hipDrvMemcpy2DUnaligned"]                                           = {HIP_4020, HIP_0,    HIP_0   };
  m["hipPointerGetAttribute"]                                            = {HIP_5000, HIP_0,    HIP_0   };
  m["hipDrvPointerGetAttributes"]                                        = {HIP_5000, HIP_0,    HIP_0   };
  m["hipStreamGetCaptureInfo"]                                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipStreamGetCaptureInfo_v2"]                                        = {HIP_5000, HIP_0,    HIP_0   };
  m["hipStreamIsCapturing"]                                              = {HIP_5000, HIP_0,    HIP_0   };
  m["hipStreamUpdateCaptureDependencies"]                                = {HIP_5000, HIP_0,    HIP_0   };
  m["hipGraphicsGLRegisterImage"]                                        = {HIP_5010, HIP_0,    HIP_0   };
  m["hipGraphicsSubResourceGetMappedArray"]                              = {HIP_5010, HIP_0,    HIP_0   };
  m["hipDeviceGetUuid"]                                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAddressFree"]                                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAddressReserve"]                                              = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemCreate"]                                                      = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemExportToShareableHandle"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemGetAccess"]                                                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemGetAllocationGranularity"]                                    = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemGetAllocationPropertiesFromHandle"]                           = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemImportFromShareableHandle"]                                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemMap"]                                                         = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemMapArrayAsync"]                                               = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemRelease"]                                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemRetainAllocationHandle"]                                      = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemSetAccess"]                                                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemUnmap"]                                                       = {HIP_5020, HIP_0,    HIP_0   };
  m["hiprtcLinkCreate"]                                                  = {HIP_5030, HIP_0,    HIP_0   };
  m["hiprtcLinkAddFile"]                                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["hiprtcLinkAddData"]                                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["hiprtcLinkComplete"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["hiprtcLinkDestroy"]                                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["hipDrvGetErrorName"]                                                = {HIP_5040, HIP_0,    HIP_0   };
  m["hipDrvGetErrorString"]                                              = {HIP_5040, HIP_0,    HIP_0   };
  m["hipPointerSetAttribute"]                                            = {HIP_5050, HIP_0,    HIP_0   };
  m["hipModuleLaunchCooperativeKernel"]                                  = {HIP_5050, HIP_0,    HIP_0   };
  m["hipModuleLaunchCooperativeKernelMultiDevice"]                       = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphAddMemAllocNode"]                                           = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphMemAllocNodeGetParams"]                                     = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphAddMemFreeNode"]                                            = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphMemFreeNodeGetParams"]                                      = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotPrint"]                                             = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphKernelNodeCopyAttributes"]                                  = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphNodeSetEnabled"]                                            = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphNodeGetEnabled"]                                            = {HIP_5050, HIP_0,    HIP_0   };
  m["hipArrayGetDescriptor"]                                             = {HIP_5060, HIP_0,    HIP_0   };
  m["hipArray3DGetDescriptor"]                                           = {HIP_5060, HIP_0,    HIP_0   };
  m["hipDrvGraphAddMemcpyNode"]                                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipDrvGraphAddMemsetNode"]                                          = {HIP_6010, HIP_0,    HIP_0   };
  m["hipTexRefGetBorderColor"]                                           = {HIP_6010, HIP_6010, HIP_0   };
  m["hipMemcpyAtoD"]                                                     = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpyDtoA"]                                                     = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpyAtoA"]                                                     = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpyAtoHAsync"]                                                = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpyHtoAAsync"]                                                = {HIP_6020, HIP_0,    HIP_0   };
  m["hipDrvGraphAddMemFreeNode"]                                         = {HIP_6030, HIP_0,    HIP_0   };
  m["hipDrvGraphMemcpyNodeGetParams"]                                    = {HIP_6030, HIP_0,    HIP_0   };
  m["hipDrvGraphMemcpyNodeSetParams"]                                    = {HIP_6030, HIP_0,    HIP_0   };
  m["hipDrvGraphExecMemcpyNodeSetParams"]                                = {HIP_6030, HIP_0,    HIP_0   };
  m["hipDrvGraphExecMemsetNodeSetParams"]                                = {HIP_6030, HIP_0,    HIP_0   };
  m["hipStreamBatchMemOp"]                                               = {HIP_6040, HIP_0,    HIP_0   };
  m["hipGraphAddBatchMemOpNode"]                                         = {HIP_6040, HIP_0,    HIP_0   };
  m["hipGraphBatchMemOpNodeGetParams"]                                   = {HIP_6040, HIP_0,    HIP_0   };
  m["hipGraphBatchMemOpNodeSetParams"]                                   = {HIP_6040, HIP_0,    HIP_0   };
  m["hipGraphExecBatchMemOpNodeSetParams"]                               = {HIP_6040, HIP_0,    HIP_0   };
  m["hipEventRecordWithFlags"]                                           = {HIP_6040, HIP_0,    HIP_0   };
  m["hipDrvLaunchKernelEx"]                                              = {HIP_7000, HIP_0,    HIP_0   };
  m["hipMemGetHandleForAddressRange"]                                    = {HIP_7000, HIP_0,    HIP_0   };
  m["hipMemsetD2D8"]                                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemsetD2D8Async"]                                                = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemsetD2D16"]                                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemsetD2D16Async"]                                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemsetD2D32"]                                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemsetD2D32Async"]                                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipModuleLoadFatBinary"]                                            = {HIP_7010, HIP_0,    HIP_0   };
  m["hipModuleGetFunctionCount"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["hipKernelGetName"]                                                  = {HIP_7020, HIP_0,    HIP_0   };
  m["hipKernelGetLibrary"]                                               = {HIP_7020, HIP_0,    HIP_0   };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_DRIVER_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIChangedVersions> m;

  m["cuGetProcAddress"]                                                  = {CUDA_120};
  m["cuGraphAddNode"]                                                    = {CUDA_130};
  m["cuCtxCreate"]                                                       = {CUDA_130};
  m["cuMemcpyBatchAsync"]                                                = {CUDA_130};
  m["cuMemcpy3DBatchAsync"]                                              = {CUDA_130};
  m["cuMemPrefetchAsync"]                                                = {CUDA_130};
  m["cuMemAdvise"]                                                       = {CUDA_130};
  m["cuStreamGetCaptureInfo"]                                            = {CUDA_130};
  m["cuStreamUpdateCaptureDependencies"]                                 = {CUDA_130};
  m["cuGraphGetEdges"]                                                   = {CUDA_130};
  m["cuGraphNodeGetDependencies"]                                        = {CUDA_130};
  m["cuGraphNodeGetDependentNodes"]                                      = {CUDA_130};
  m["cuGraphAddDependencies"]                                            = {CUDA_130};
  m["cuGraphRemoveDependencies"]                                         = {CUDA_130};

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_DRIVER_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIChangedVersions> m;

  m["hipCtxGetApiVersion"]                                               = {HIP_7000};
  m["hipDrvGraphAddMemsetNode"]                                          = {HIP_7000};
  m["hipDrvGraphExecMemsetNodeSetParams"]                                = {HIP_7000};
  m["hipMemcpyHtoD"]                                                     = {HIP_7000};
  m["hipMemcpyHtoDAsync"]                                                = {HIP_7000};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIUnsupportedVersions> CUDA_DRIVER_FUNCTION_UNSUPPORTED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIUnsupportedVersions> m;

  m["cuCtxCreate"]                                                       = {CUDA_130};
  m["cuCtxCreate_v2"]                                                    = {CUDA_130};
  m["cuMemcpyBatchAsync"]                                                = {CUDA_130};
  m["cuMemcpy3DBatchAsync"]                                              = {CUDA_130};
  m["cuMemAdvise"]                                                       = {CUDA_130};
  m["cuMemPrefetchAsync"]                                                = {CUDA_130};
  m["cuStreamGetCaptureInfo"]                                            = {CUDA_130};
  m["cuStreamUpdateCaptureDependencies"]                                 = {CUDA_130};
  m["cuGraphAddDependencies"]                                            = {CUDA_130};
  m["cuGraphGetEdges"]                                                   = {CUDA_130};
  m["cuGraphNodeGetDependencies"]                                        = {CUDA_130};
  m["cuGraphNodeGetDependentNodes"]                                      = {CUDA_130};
  m["cuGraphRemoveDependencies"]                                         = {CUDA_130};
  m["cuGraphAddNode"]                                                    = {CUDA_130};
  m["cuGetProcAddress"]                                                  = {CUDA_113, CUDA_114, CUDA_115, CUDA_116, CUDA_117, CUDA_118};

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_DRIVER_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[SEC::DATA_TYPES]                                                     = "CUDA Driver Data Types";
  m[SEC::ERROR]                                                          = "Error Handling";
  m[SEC::INIT]                                                           = "Initialization";
  m[SEC::VERSION]                                                        = "Version Management";
  m[SEC::DEVICE]                                                         = "Device Management";
  m[SEC::DEVICE_DEPRECATED]                                              = "Device Management [DEPRECATED]";
  m[SEC::PRIMARY_CONTEXT]                                                = "Primary Context Management";
  m[SEC::CONTEXT]                                                        = "Context Management";
  m[SEC::CONTEXT_DEPRECATED]                                             = "Context Management [DEPRECATED]";
  m[SEC::MODULE]                                                         = "Module Management";
  m[SEC::MODULE_DEPRECATED]                                              = "Module Management [DEPRECATED]";
  m[SEC::LIBRARY]                                                        = "Library Management";
  m[SEC::MEMORY]                                                         = "Memory Management";
  m[SEC::VIRTUAL_MEMORY]                                                 = "Virtual Memory Management";
  m[SEC::ORDERED_MEMORY]                                                 = "Stream Ordered Memory Allocator";
  m[SEC::MULTICAST]                                                      = "Multicast Object Management";
  m[SEC::UNIFIED]                                                        = "Unified Addressing";
  m[SEC::STREAM]                                                         = "Stream Management";
  m[SEC::EVENT]                                                          = "Event Management";
  m[SEC::EXTERNAL_RES]                                                   = "External Resource Interoperability";
  m[SEC::STREAM_MEMORY]                                                  = "Stream Memory Operations";
  m[SEC::EXECUTION]                                                      = "Execution Control";
  m[SEC::EXECUTION_DEPRECATED]                                           = "Execution Control [DEPRECATED]";
  m[SEC::GRAPH]                                                          = "Graph Management";
  m[SEC::OCCUPANCY]                                                      = "Occupancy";
  m[SEC::TEXTURE_DEPRECATED]                                             = "Texture Reference Management [DEPRECATED]";
  m[SEC::SURFACE_DEPRECATED]                                             = "Surface Reference Management [DEPRECATED]";
  m[SEC::TEXTURE]                                                        = "Texture Object Management";
  m[SEC::SURFACE]                                                        = "Surface Object Management";
  m[SEC::TENSOR]                                                         = "Tensor Map Object Managment";
  m[SEC::PEER]                                                           = "Peer Context Memory Access";
  m[SEC::GRAPHICS]                                                       = "Graphics Interoperability";
  m[SEC::DRIVER_ENTRY_POINT]                                             = "Driver Entry Point Access";
  m[SEC::COREDUMP]                                                       = "Coredump Attributes Control API";
  m[SEC::GREEN_CONTEXT]                                                  = "Green Contexts";
  m[SEC::ERROR_LOG]                                                      = "Error Log Management";
  m[SEC::CHECKPOINTING]                                                  = "Checkpointing";
  m[SEC::PROFILER_DEPRECATED]                                            = "Profiler Control [DEPRECATED]";
  m[SEC::PROFILER]                                                       = "Profiler Control";
  m[SEC::OPENGL]                                                         = "OpenGL Interoperability";
  m[SEC::D3D9]                                                           = "Direct3D 9 Interoperability";
  m[SEC::D3D10]                                                          = "Direct3D 10 Interoperability";
  m[SEC::D3D11]                                                          = "Direct3D 11 Interoperability";
  m[SEC::VDPAU]                                                          = "VDPAU Interoperability";
  m[SEC::EGL]                                                            = "EGL Interoperability";

  return m;
}();
