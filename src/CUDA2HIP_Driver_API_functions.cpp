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
const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP {
  // 2. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  {"cuGetErrorName",                                       {"hipDrvGetErrorName",                                      "", CONV_ERROR, API_DRIVER, SEC::ERROR}},
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  {"cuGetErrorString",                                     {"hipDrvGetErrorString",                                    "", CONV_ERROR, API_DRIVER, SEC::ERROR}},

  // 3. Initialization
  // no analogue
  {"cuInit",                                               {"hipInit",                                                 "", CONV_INIT, API_DRIVER, SEC::INIT}},

  // 4. Version Management
  // cudaDriverGetVersion
  {"cuDriverGetVersion",                                   {"hipDriverGetVersion",                                     "", CONV_VERSION, API_DRIVER, SEC::VERSION}},

  // 5. Device Management
  // cudaGetDevice
  // NOTE: cudaGetDevice has additional attr: int ordinal
  {"cuDeviceGet",                                          {"hipDeviceGet",                                            "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaDeviceGetAttribute
  {"cuDeviceGetAttribute",                                 {"hipDeviceGetAttribute",                                   "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaGetDeviceCount
  {"cuDeviceGetCount",                                     {"hipGetDeviceCount",                                       "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // no analogue
  {"cuDeviceGetLuid",                                      {"hipDeviceGetLuid",                                        "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceGetName",                                      {"hipDeviceGetName",                                        "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaDeviceGetNvSciSyncAttributes
  {"cuDeviceGetNvSciSyncAttributes",                       {"hipDeviceGetNvSciSyncAttributes",                         "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceGetUuid",                                      {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // no analogue
  {"cuDeviceGetUuid_v2",                                   {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // no analogue
  {"cuDeviceTotalMem",                                     {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  {"cuDeviceTotalMem_v2",                                  {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaDeviceGetTexture1DLinearMaxWidth
  {"cuDeviceGetTexture1DLinearMaxWidth",                   {"hipDeviceGetTexture1DLinearMaxWidth",                     "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED}},
  // cudaDeviceSetMemPool
  {"cuDeviceSetMemPool",                                   {"hipDeviceSetMemPool",                                     "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaDeviceGetMemPool
  {"cuDeviceGetMemPool",                                   {"hipDeviceGetMemPool",                                     "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  // cudaDeviceGetDefaultMemPool
  {"cuDeviceGetDefaultMemPool",                            {"hipDeviceGetDefaultMemPool",                              "", CONV_DEVICE, API_DRIVER, SEC::DEVICE}},
  //
  {"cuDeviceGetExecAffinitySupport",                       {"hipDeviceGetExecAffinitySupport",                         "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED}},
  // cudaDeviceFlushGPUDirectRDMAWrites
  {"cuFlushGPUDirectRDMAWrites",                           {"hipDeviceFlushGPUDirectRDMAWrites",                       "", CONV_DEVICE, API_DRIVER, SEC::DEVICE, HIP_UNSUPPORTED}},

  // 6. Device Management [DEPRECATED]
  //
  {"cuDeviceComputeCapability",                            {"hipDeviceComputeCapability",                              "", CONV_DEVICE, API_DRIVER, SEC::DEVICE_DEPRECATED, CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGetDeviceProperties due to different attributes: cudaDeviceProp and CUdevprop
  {"cuDeviceGetProperties",                                {"hipGetDeviceProperties_",                                 "", CONV_DEVICE, API_DRIVER, SEC::DEVICE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 7. Primary Context Management
  // no analogues
  {"cuDevicePrimaryCtxGetState",                           {"hipDevicePrimaryCtxGetState",                             "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxRelease",                            {"hipDevicePrimaryCtxRelease",                              "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxRelease_v2",                         {"hipDevicePrimaryCtxRelease",                              "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxReset",                              {"hipDevicePrimaryCtxReset",                                "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxReset_v2",                           {"hipDevicePrimaryCtxReset",                                "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxRetain",                             {"hipDevicePrimaryCtxRetain",                               "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxSetFlags",                           {"hipDevicePrimaryCtxSetFlags",                             "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},
  {"cuDevicePrimaryCtxSetFlags_v2",                        {"hipDevicePrimaryCtxSetFlags",                             "", CONV_CONTEXT, API_DRIVER, SEC::PRIMARY_CONTEXT}},

  // 8. Context Management
  // no analogues, except a few
  {"cuCtxCreate",                                          {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxCreate_v2",                                       {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxCreate_v3",                                       {"hipCtxCreate_v3",                                         "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED}},
  {"cuCtxDestroy",                                         {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxDestroy_v2",                                      {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxGetApiVersion",                                   {"hipCtxGetApiVersion",                                     "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxGetCacheConfig",                                  {"hipCtxGetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxGetCurrent",                                      {"hipCtxGetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxGetDevice",                                       {"hipCtxGetDevice",                                         "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  // cudaGetDeviceFlags
  // TODO: rename to hipGetDeviceFlags
  {"cuCtxGetFlags",                                        {"hipCtxGetFlags",                                          "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxSetFlags",                                        {"hipCtxSetFlags",                                          "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED}},
  // cudaDeviceGetLimit
  {"cuCtxGetLimit",                                        {"hipDeviceGetLimit",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT}},
  // cudaDeviceGetSharedMemConfig
  // TODO: rename to hipDeviceGetSharedMemConfig
  {"cuCtxGetSharedMemConfig",                              {"hipCtxGetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  // cudaDeviceGetStreamPriorityRange
  {"cuCtxGetStreamPriorityRange",                          {"hipDeviceGetStreamPriorityRange",                         "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT}},
  {"cuCtxPopCurrent",                                      {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxPopCurrent_v2",                                   {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxPushCurrent",                                     {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxPushCurrent_v2",                                  {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  {"cuCtxSetCacheConfig",                                  {"hipCtxSetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  // cudaCtxResetPersistingL2Cache
  {"cuCtxResetPersistingL2Cache",                          {"hipCtxResetPersistingL2Cache",                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED}},
  {"cuCtxSetCurrent",                                      {"hipCtxSetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  // cudaDeviceSetLimit
  {"cuCtxSetLimit",                                        {"hipDeviceSetLimit",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT}},
  // cudaDeviceSetSharedMemConfig
  // TODO: rename to hipDeviceSetSharedMemConfig
  {"cuCtxSetSharedMemConfig",                              {"hipCtxSetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  // cudaDeviceSynchronize
  // TODO: rename to hipDeviceSynchronize
  {"cuCtxSynchronize",                                     {"hipCtxSynchronize",                                       "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_DEPRECATED}},
  //
  {"cuCtxGetExecAffinity",                                 {"hipCtxGetExecAffinity",                                   "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED}},
  //
  {"cuCtxGetId",                                           {"hipCtxGetId",                                             "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT, HIP_UNSUPPORTED}},

  // 9. Context Management [DEPRECATED]
  // no analogues
  {"cuCtxAttach",                                          {"hipCtxAttach",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuCtxDetach",                                          {"hipCtxDetach",                                            "", CONV_CONTEXT, API_DRIVER, SEC::CONTEXT_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 10. Module Management
  // no analogues
  {"cuLinkAddData",                                        {"hiprtcLinkAddData",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkAddData_v2",                                     {"hiprtcLinkAddData",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkAddFile",                                        {"hiprtcLinkAddFile",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkAddFile_v2",                                     {"hiprtcLinkAddFile",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkComplete",                                       {"hiprtcLinkComplete",                                      "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkCreate",                                         {"hiprtcLinkCreate",                                        "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkCreate_v2",                                      {"hiprtcLinkCreate",                                        "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuLinkDestroy",                                        {"hiprtcLinkDestroy",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleGetFunction",                                  {"hipModuleGetFunction",                                    "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleGetGlobal",                                    {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleGetGlobal_v2",                                 {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleLoad",                                         {"hipModuleLoad",                                           "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleLoadData",                                     {"hipModuleLoadData",                                       "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleLoadDataEx",                                   {"hipModuleLoadDataEx",                                     "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleLoadFatBinary",                                {"hipModuleLoadFatBinary",                                  "", CONV_MODULE, API_DRIVER, SEC::MODULE, HIP_UNSUPPORTED}},
  {"cuModuleUnload",                                       {"hipModuleUnload",                                         "", CONV_MODULE, API_DRIVER, SEC::MODULE}},
  {"cuModuleGetLoadingMode",                               {"hipModuleGetLoadingMode",                                 "", CONV_MODULE, API_DRIVER, SEC::MODULE, HIP_UNSUPPORTED}},

  // 11. Module Management [DEPRECATED]
  {"cuModuleGetSurfRef",                                   {"hipModuleGetSurfRef",                                     "", CONV_MODULE, API_DRIVER, SEC::MODULE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuModuleGetTexRef",                                    {"hipModuleGetTexRef",                                      "", CONV_MODULE, API_DRIVER, SEC::MODULE_DEPRECATED, CUDA_DEPRECATED}},

  // 12. Library Management
  {"cuLibraryLoadData",                                    {"hipLibraryLoadData",                                      "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryLoadFromFile",                                {"hipLibraryLoadFromFile",                                  "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryUnload",                                      {"hipLibraryUnload",                                        "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryGetKernel",                                   {"hipLibraryGetKernel",                                     "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryGetModule",                                   {"hipLibraryGetModule",                                     "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuKernelGetFunction",                                  {"hipKernelGetFunction",                                    "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryGetGlobal",                                   {"hipLibraryGetGlobal",                                     "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryGetManaged",                                  {"hipLibraryGetManaged",                                    "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuLibraryGetUnifiedFunction",                          {"hipLibraryGetUnifiedFunction",                            "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuKernelGetAttribute",                                 {"hipKernelGetAttribute",                                   "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuKernelSetAttribute",                                 {"hipKernelSetAttribute",                                   "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuKernelSetCacheConfig",                               {"hipKernelSetCacheConfig",                                 "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},
  {"cuKernelGetName",                                      {"hipKernelGetName",                                        "", CONV_LIBRARY, API_DRIVER, SEC::LIBRARY, HIP_UNSUPPORTED}},

  // 13. Memory Management
  // no analogue
  {"cuArray3DCreate",                                      {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArray3DCreate_v2",                                   {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArray3DGetDescriptor",                               {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArray3DGetDescriptor_v2",                            {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArrayCreate",                                        {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArrayCreate_v2",                                     {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArrayDestroy",                                       {"hipArrayDestroy",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArrayGetDescriptor",                                 {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuArrayGetDescriptor_v2",                              {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  //
  {"cuMipmappedArrayGetMemoryRequirements",                {"hipMipmappedArrayGetMemoryRequirements",                  "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // cudaArrayGetMemoryRequirements
  {"cuArrayGetMemoryRequirements",                         {"hipArrayGetMemoryRequirements",                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // cudaDeviceGetByPCIBusId
  {"cuDeviceGetByPCIBusId",                                {"hipDeviceGetByPCIBusId",                                  "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaDeviceGetPCIBusId
  {"cuDeviceGetPCIBusId",                                  {"hipDeviceGetPCIBusId",                                    "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaIpcCloseMemHandle
  {"cuIpcCloseMemHandle",                                  {"hipIpcCloseMemHandle",                                    "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaIpcGetEventHandle
  {"cuIpcGetEventHandle",                                  {"hipIpcGetEventHandle",                                    "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaIpcGetMemHandle
  {"cuIpcGetMemHandle",                                    {"hipIpcGetMemHandle",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaIpcOpenEventHandle
  {"cuIpcOpenEventHandle",                                 {"hipIpcOpenEventHandle",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaIpcOpenMemHandle
  {"cuIpcOpenMemHandle",                                   {"hipIpcOpenMemHandle",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaMalloc
  {"cuMemAlloc",                                           {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemAlloc_v2",                                        {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  //
  {"cuMemAllocHost",                                       {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemAllocHost_v2",                                    {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaMallocManaged
  {"cuMemAllocManaged",                                    {"hipMallocManaged",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMallocPitch due to different signatures
  {"cuMemAllocPitch",                                      {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemAllocPitch_v2",                                   {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy due to different signatures
  {"cuMemcpy",                                             {"hipMemcpy_",                                              "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2D due to different signatures
  {"cuMemcpy2D",                                           {"hipMemcpyParam2D",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpy2D_v2",                                        {"hipMemcpyParam2D",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2DAsync/hipMemcpy2DAsync due to different signatures
  {"cuMemcpy2DAsync",                                      {"hipMemcpyParam2DAsync",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpy2DAsync_v2",                                   {"hipMemcpyParam2DAsync",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpy2DUnaligned",                                  {"hipDrvMemcpy2DUnaligned",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpy2DUnaligned_v2",                               {"hipDrvMemcpy2DUnaligned",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3D due to different signatures
  {"cuMemcpy3D",                                           {"hipDrvMemcpy3D",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpy3D_v2",                                        {"hipDrvMemcpy3D",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DAsync due to different signatures
  {"cuMemcpy3DAsync",                                      {"hipDrvMemcpy3DAsync",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpy3DAsync_v2",                                   {"hipDrvMemcpy3DAsync",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeer due to different signatures
  {"cuMemcpy3DPeer",                                       {"hipMemcpy3DPeer_",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeerAsync due to different signatures
  {"cuMemcpy3DPeerAsync",                                  {"hipMemcpy3DPeerAsync_",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyAsync due to different signatures
  {"cuMemcpyAsync",                                        {"hipMemcpyAsync_",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyArrayToArray due to different signatures
  {"cuMemcpyAtoA",                                         {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoA_v2",                                      {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoD",                                         {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoD_v2",                                      {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoH",                                         {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyAtoH_v2",                                      {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyAtoHAsync",                                    {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoHAsync_v2",                                 {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoA",                                         {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemcpyDtoA_v2",                                      {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoD",                                         {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyDtoD_v2",                                      {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyDtoDAsync",                                    {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyDtoDAsync_v2",                                 {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyDtoH",                                         {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyDtoH_v2",                                      {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyDtoHAsync",                                    {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyDtoHAsync_v2",                                 {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyHtoA",                                         {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyHtoA_v2",                                      {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyHtoAAsync",                                    {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemcpyHtoAAsync_v2",                                 {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyHtoD",                                         {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyHtoD_v2",                                      {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemcpyHtoDAsync",                                    {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemcpyHtoDAsync_v2",                                 {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeer due to different signatures
  {"cuMemcpyPeer",                                         {"hipMemcpyPeer_",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeerAsync due to different signatures
  {"cuMemcpyPeerAsync",                                    {"hipMemcpyPeerAsync_",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // cudaFree
  {"cuMemFree",                                            {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemFree_v2",                                         {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaFreeHost
  {"cuMemFreeHost",                                        {"hipHostFree",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemGetAddressRange",                                 {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemGetAddressRange_v2",                              {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaMemGetInfo
  {"cuMemGetInfo",                                         {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemGetInfo_v2",                                      {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaHostAlloc
  {"cuMemHostAlloc",                                       {"hipHostAlloc",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaHostGetDevicePointer
  {"cuMemHostGetDevicePointer",                            {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemHostGetDevicePointer_v2",                         {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaHostGetFlags
  {"cuMemHostGetFlags",                                    {"hipHostGetFlags",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaHostRegister
  {"cuMemHostRegister",                                    {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemHostRegister_v2",                                 {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaHostUnregister
  {"cuMemHostUnregister",                                  {"hipHostUnregister",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemsetD16",                                          {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemsetD16_v2",                                       {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemsetD16Async",                                     {"hipMemsetD16Async",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemsetD2D16",                                        {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemsetD2D16_v2",                                     {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D16Async",                                   {"hipMemsetD2D16Async",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32",                                        {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemsetD2D32_v2",                                     {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32Async",                                   {"hipMemsetD2D32Async",                                     "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8",                                         {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  {"cuMemsetD2D8_v2",                                      {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8Async",                                    {"hipMemsetD2D8Async",                                      "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // cudaMemset
  {"cuMemsetD32",                                          {"hipMemsetD32",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemsetD32_v2",                                       {"hipMemsetD32",                                            "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // cudaMemsetAsync
  {"cuMemsetD32Async",                                     {"hipMemsetD32Async",                                       "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemsetD8",                                           {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  {"cuMemsetD8_v2",                                        {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  {"cuMemsetD8Async",                                      {"hipMemsetD8Async",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY}},
  // no analogue
  // NOTE: Not equal to cudaMallocMipmappedArray due to different signatures
  {"cuMipmappedArrayCreate",                               {"hipMipmappedArrayCreate",                                 "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaFreeMipmappedArray due to different signatures
  {"cuMipmappedArrayDestroy",                              {"hipMipmappedArrayDestroy",                                "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGetMipmappedArrayLevel due to different signatures
  {"cuMipmappedArrayGetLevel",                             {"hipMipmappedArrayGetLevel",                               "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_DEPRECATED}},
  // cudaArrayGetSparseProperties
  {"cuArrayGetSparseProperties",                           {"hipArrayGetSparseProperties",                             "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  // cudaArrayGetPlane
  {"cuArrayGetPlane",                                      {"hipArrayGetPlane",                                        "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},
  //
  {"cuMemGetHandleForAddressRange",                        {"hipMemGetHandleForAddressRange",                          "", CONV_MEMORY, API_DRIVER, SEC::MEMORY, HIP_UNSUPPORTED}},

  // 14. Virtual Memory Management
  // no analogue
  {"cuMemAddressFree",                                     {"hipMemAddressFree",                                       "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemAddressReserve",                                  {"hipMemAddressReserve",                                    "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemCreate",                                          {"hipMemCreate",                                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemExportToShareableHandle",                         {"hipMemExportToShareableHandle",                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemGetAccess",                                       {"hipMemGetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemGetAllocationGranularity",                        {"hipMemGetAllocationGranularity",                          "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemGetAllocationPropertiesFromHandle",               {"hipMemGetAllocationPropertiesFromHandle",                 "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemImportFromShareableHandle",                       {"hipMemImportFromShareableHandle",                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemMap",                                             {"hipMemMap",                                               "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemRelease",                                         {"hipMemRelease",                                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemRetainAllocationHandle",                          {"hipMemRetainAllocationHandle",                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemSetAccess",                                       {"hipMemSetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemUnmap",                                           {"hipMemUnmap",                                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},
  {"cuMemMapArrayAsync",                                   {"hipMemMapArrayAsync",                                     "", CONV_VIRTUAL_MEMORY, API_DRIVER, SEC::VIRTUAL_MEMORY}},

  // 15. Stream Ordered Memory Allocator
  // cudaFreeAsync
  {"cuMemFreeAsync",                                       {"hipFreeAsync",                                            "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMallocAsync
  {"cuMemAllocAsync",                                      {"hipMallocAsync",                                          "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolTrimTo
  {"cuMemPoolTrimTo",                                      {"hipMemPoolTrimTo",                                        "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolSetAttribute
  {"cuMemPoolSetAttribute",                                {"hipMemPoolSetAttribute",                                  "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolGetAttribute
  {"cuMemPoolGetAttribute",                                {"hipMemPoolGetAttribute",                                  "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolSetAccess
  {"cuMemPoolSetAccess",                                   {"hipMemPoolSetAccess",                                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolGetAccess
  {"cuMemPoolGetAccess",                                   {"hipMemPoolGetAccess",                                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolCreate
  {"cuMemPoolCreate",                                      {"hipMemPoolCreate",                                        "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolDestroy
  {"cuMemPoolDestroy",                                     {"hipMemPoolDestroy",                                       "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMallocFromPoolAsync
  {"cuMemAllocFromPoolAsync",                              {"hipMallocFromPoolAsync",                                  "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolExportToShareableHandle
  {"cuMemPoolExportToShareableHandle",                     {"hipMemPoolExportToShareableHandle",                       "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolImportFromShareableHandle
  {"cuMemPoolImportFromShareableHandle",                   {"hipMemPoolImportFromShareableHandle",                     "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolExportPointer
  {"cuMemPoolExportPointer",                               {"hipMemPoolExportPointer",                                 "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},
  // cudaMemPoolImportPointer
  {"cuMemPoolImportPointer",                               {"hipMemPoolImportPointer",                                 "", CONV_ORDERED_MEMORY, API_DRIVER, SEC::ORDERED_MEMORY}},

  // 16. Multicast Object Management
  //
  {"cuMulticastCreate",                                    {"hipMulticastCreate",                                      "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},
  //
  {"cuMulticastAddDevice",                                 {"hipMulticastAddDevice",                                   "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},
  //
  {"cuMulticastBindMem",                                   {"hipMulticastBindMem",                                     "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},
  //
  {"cuMulticastBindAddr",                                  {"hipMulticastBindAddr",                                    "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},
  //
  {"cuMulticastUnbind",                                    {"hipMulticastUnbind",                                      "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},
  //
  {"cuMulticastGetGranularity",                            {"hipMulticastGetGranularity",                              "", CONV_MULTICAST, API_DRIVER, SEC::MULTICAST, HIP_UNSUPPORTED}},

  // 17. Unified Addressing
  // cudaMemAdvise
  {"cuMemAdvise",                                          {"hipMemAdvise",                                            "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // cudaMemAdvise_v2
  {"cuMemAdvise_v2",                                       {"hipMemAdvise_v2",                                         "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED}},
  // cudaMemPrefetchAsync
  {"cuMemPrefetchAsync",                                   {"hipMemPrefetchAsync",                                     "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // cudaMemPrefetchAsync_v2
  {"cuMemPrefetchAsync_v2",                                {"hipMemPrefetchAsync_v2",                                  "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED, HIP_UNSUPPORTED}},
  // cudaMemRangeGetAttribute
  {"cuMemRangeGetAttribute",                               {"hipMemRangeGetAttribute",                                 "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // cudaMemRangeGetAttributes
  {"cuMemRangeGetAttributes",                              {"hipMemRangeGetAttributes",                                "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // no analogue
  {"cuPointerGetAttribute",                                {"hipPointerGetAttribute",                                  "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // no analogue
  // NOTE: Not equal to cudaPointerGetAttributes due to different signatures
  {"cuPointerGetAttributes",                               {"hipDrvPointerGetAttributes",                              "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},
  // no analogue
  {"cuPointerSetAttribute",                                {"hipPointerSetAttribute",                                  "", CONV_UNIFIED, API_DRIVER, SEC::UNIFIED}},

  // 18. Stream Management
  // cudaStreamAddCallback
  {"cuStreamAddCallback",                                  {"hipStreamAddCallback",                                    "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamAttachMemAsync
  {"cuStreamAttachMemAsync",                               {"hipStreamAttachMemAsync",                                 "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamBeginCapture
  {"cuStreamBeginCapture",                                 {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  {"cuStreamBeginCapture_v2",                              {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  {"cuStreamBeginCapture_ptsz",                            {"hipStreamBeginCapture_ptsz",                              "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamBeginCaptureToGraph
  {"cuStreamBeginCaptureToGraph",                          {"hipStreamBeginCaptureToGraph",                            "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamCopyAttributes
  {"cuStreamCopyAttributes",                               {"hipStreamCopyAttributes",                                 "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamCreateWithFlags
  {"cuStreamCreate",                                       {"hipStreamCreateWithFlags",                                "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamCreateWithPriority
  {"cuStreamCreateWithPriority",                           {"hipStreamCreateWithPriority",                             "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamDestroy
  {"cuStreamDestroy",                                      {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  {"cuStreamDestroy_v2",                                   {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamEndCapture
  {"cuStreamEndCapture",                                   {"hipStreamEndCapture",                                     "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamGetAttribute
  {"cuStreamGetAttribute",                                 {"hipStreamGetAttribute",                                   "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamGetCaptureInfo
  {"cuStreamGetCaptureInfo",                               {"hipStreamGetCaptureInfo",                                 "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  {"cuStreamGetCaptureInfo_v2",                            {"hipStreamGetCaptureInfo_v2",                              "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamGetCaptureInfo_v3
  {"cuStreamGetCaptureInfo_v3",                            {"hipStreamGetCaptureInfo_v3",                              "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamUpdateCaptureDependencies
  {"cuStreamUpdateCaptureDependencies",                    {"hipStreamUpdateCaptureDependencies",                      "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamUpdateCaptureDependencies_v2
  {"cuStreamUpdateCaptureDependencies_v2",                 {"hipStreamUpdateCaptureDependencies_v2",                   "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // no analogue
  {"cuStreamGetCtx",                                       {"hipStreamGetContext",                                     "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamGetFlags
  {"cuStreamGetFlags",                                     {"hipStreamGetFlags",                                       "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamGetPriority
  {"cuStreamGetPriority",                                  {"hipStreamGetPriority",                                    "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamIsCapturing
  {"cuStreamIsCapturing",                                  {"hipStreamIsCapturing",                                    "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamQuery
  {"cuStreamQuery",                                        {"hipStreamQuery",                                          "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamSetAttribute
  {"cuStreamSetAttribute",                                 {"hipStreamSetAttribute",                                   "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},
  // cudaStreamSynchronize
  {"cuStreamSynchronize",                                  {"hipStreamSynchronize",                                    "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamWaitEvent
  {"cuStreamWaitEvent",                                    {"hipStreamWaitEvent",                                      "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaThreadExchangeStreamCaptureMode
  {"cuThreadExchangeStreamCaptureMode",                    {"hipThreadExchangeStreamCaptureMode",                      "", CONV_STREAM, API_DRIVER, SEC::STREAM}},
  // cudaStreamGetId
  {"cuStreamGetId",                                        {"hipStreamGetId",                                          "", CONV_STREAM, API_DRIVER, SEC::STREAM, HIP_UNSUPPORTED}},

  // 19. Event Management
  // cudaEventCreateWithFlags
  {"cuEventCreate",                                        {"hipEventCreateWithFlags",                                 "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventDestroy
  {"cuEventDestroy",                                       {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  {"cuEventDestroy_v2",                                    {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventElapsedTime
  {"cuEventElapsedTime",                                   {"hipEventElapsedTime",                                     "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventQuery
  {"cuEventQuery",                                         {"hipEventQuery",                                           "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventRecord
  {"cuEventRecord",                                        {"hipEventRecord",                                          "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventSynchronize
  {"cuEventSynchronize",                                   {"hipEventSynchronize",                                     "", CONV_EVENT, API_DRIVER, SEC::EVENT}},
  // cudaEventRecordWithFlags
  {"cuEventRecordWithFlags",                               {"hipEventRecordWithFlags",                                 "", CONV_EVENT, API_DRIVER, SEC::EVENT, HIP_UNSUPPORTED}},

  // 20. External Resource Interoperability
  // cudaDestroyExternalMemory
  {"cuDestroyExternalMemory",                              {"hipDestroyExternalMemory",                                "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaDestroyExternalSemaphore
  {"cuDestroyExternalSemaphore",                           {"hipDestroyExternalSemaphore",                             "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaExternalMemoryGetMappedBuffer
  {"cuExternalMemoryGetMappedBuffer",                      {"hipExternalMemoryGetMappedBuffer",                        "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaExternalMemoryGetMappedMipmappedArray
  {"cuExternalMemoryGetMappedMipmappedArray",              {"hipExternalMemoryGetMappedMipmappedArray",                "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES, HIP_UNSUPPORTED}},
  // cudaImportExternalMemory
  {"cuImportExternalMemory",                               {"hipImportExternalMemory",                                 "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaImportExternalSemaphore
  {"cuImportExternalSemaphore",                            {"hipImportExternalSemaphore",                              "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaSignalExternalSemaphoresAsync
  {"cuSignalExternalSemaphoresAsync",                      {"hipSignalExternalSemaphoresAsync",                        "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},
  // cudaWaitExternalSemaphoresAsync
  {"cuWaitExternalSemaphoresAsync",                        {"hipWaitExternalSemaphoresAsync",                          "", CONV_EXTERNAL_RES, API_DRIVER, SEC::EXTERNAL_RES}},

  // 21. Stream Memory Operations
  // no analogues
  {"cuStreamBatchMemOp",                                   {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY, HIP_UNSUPPORTED}},
  {"cuStreamBatchMemOp_v2",                                {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY, HIP_UNSUPPORTED}},
  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  {"cuStreamWaitValue32",                                  {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  {"cuStreamWaitValue32_v2",                               {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  {"cuStreamWaitValue64",                                  {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  {"cuStreamWaitValue64_v2",                               {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  {"cuStreamWriteValue32",                                 {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  {"cuStreamWriteValue32_v2",                              {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  {"cuStreamWriteValue64",                                 {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},
  // CUresult CUDAAPI cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  {"cuStreamWriteValue64_v2",                              {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, SEC::STREAM_MEMORY}},

  // 22. Execution Control
  // no analogue
  {"cuFuncGetAttribute",                                   {"hipFuncGetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION}},
  // no analogue
  {"cuFuncGetModule",                                      {"hipFuncGetModule",                                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetAttribute due to different signatures
  {"cuFuncSetAttribute",                                   {"hipFuncSetAttribute_",                                    "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  {"cuFuncSetCacheConfig",                                 {"hipFuncSetCacheConfig_",                                  "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetSharedMemConfig due to different signatures
  {"cuFuncSetSharedMemConfig",                             {"hipFuncSetSharedMemConfig_",                              "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernel due to different signatures
  {"cuLaunchCooperativeKernel",                            {"hipModuleLaunchCooperativeKernel",                        "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernelMultiDevice due to different signatures
  {"cuLaunchCooperativeKernelMultiDevice",                 {"hipModuleLaunchCooperativeKernelMultiDevice",             "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, CUDA_DEPRECATED}},
  // cudaLaunchHostFunc
  {"cuLaunchHostFunc",                                     {"hipLaunchHostFunc",                                       "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION}},
  // no analogue
  // NOTE: Not equal to cudaLaunchKernel due to different signatures
  {"cuLaunchKernel",                                       {"hipModuleLaunchKernel",                                   "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION}},
  // no analogue
  // NOTE: Not equal to cudaLaunchKernelExC due to different signatures
  {"cuLaunchKernelEx",                                     {"hipLaunchKernelEx",                                       "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},
  // cudaFuncGetName
  {"cuFuncGetName",                                        {"hipFuncGetName",                                          "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION, HIP_UNSUPPORTED}},

  // 23. Execution Control [DEPRECATED]
  // no analogue
  {"cuFuncSetBlockShape",                                  {"hipFuncSetBlockShape",                                    "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuFuncSetSharedSize",                                  {"hipFuncSetSharedSize",                                    "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  {"cuLaunch",                                             {"hipLaunch",                                               "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuLaunchGrid",                                         {"hipLaunchGrid",                                           "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuLaunchGridAsync",                                    {"hipLaunchGridAsync",                                      "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetf",                                          {"hipParamSetf",                                            "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSeti",                                          {"hipParamSeti",                                            "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetSize",                                       {"hipParamSetSize",                                         "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetTexRef",                                     {"hipParamSetTexRef",                                       "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetv",                                          {"hipParamSetv",                                            "", CONV_EXECUTION, API_DRIVER, SEC::EXECUTION_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 24. Graph Management
  // cudaGraphAddChildGraphNode
  {"cuGraphAddChildGraphNode",                             {"hipGraphAddChildGraphNode",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddDependencies
  {"cuGraphAddDependencies",                               {"hipGraphAddDependencies",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddDependencies_v2
  {"cuGraphAddDependencies_v2",                            {"hipGraphAddDependencies_v2",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // cudaGraphAddEmptyNode
  {"cuGraphAddEmptyNode",                                  {"hipGraphAddEmptyNode",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddHostNode
  {"cuGraphAddHostNode",                                   {"hipGraphAddHostNode",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddKernelNode
  {"cuGraphAddKernelNode",                                 {"hipGraphAddKernelNode",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // no analogue
  // NOTE: Not equal to cudaGraphAddMemcpyNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  {"cuGraphAddMemcpyNode",                                 {"hipDrvGraphAddMemcpyNode",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // no analogue
  {"cuGraphAddMemsetNode",                                 {"hipDrvGraphAddMemsetNode",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphChildGraphNodeGetGraph
  {"cuGraphChildGraphNodeGetGraph",                        {"hipGraphChildGraphNodeGetGraph",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphClone
  {"cuGraphClone",                                         {"hipGraphClone",                                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphCreate
  {"cuGraphCreate",                                        {"hipGraphCreate",                                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphDebugDotPrint
  {"cuGraphDebugDotPrint",                                 {"hipGraphDebugDotPrint",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphDestroy
  {"cuGraphDestroy",                                       {"hipGraphDestroy",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphDestroyNode
  {"cuGraphDestroyNode",                                   {"hipGraphDestroyNode",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphExecDestroy
  {"cuGraphExecDestroy",                                   {"hipGraphExecDestroy",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphGetEdges
  {"cuGraphGetEdges",                                      {"hipGraphGetEdges",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphGetEdges_v2
  {"cuGraphGetEdges_v2",                                   {"hipGraphGetEdges_v2",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // cudaGraphGetNodes
  {"cuGraphGetNodes",                                      {"hipGraphGetNodes",                                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphGetRootNodes
  {"cuGraphGetRootNodes",                                  {"hipGraphGetRootNodes",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphHostNodeGetParams
  {"cuGraphHostNodeGetParams",                             {"hipGraphHostNodeGetParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphHostNodeSetParams
  {"cuGraphHostNodeSetParams",                             {"hipGraphHostNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphInstantiate
  {"cuGraphInstantiate",                                   {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  {"cuGraphInstantiate_v2",                                {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphKernelNodeCopyAttributes
  {"cuGraphKernelNodeCopyAttributes",                      {"hipGraphKernelNodeCopyAttributes",                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphKernelNodeGetAttribute
  {"cuGraphKernelNodeGetAttribute",                        {"hipGraphKernelNodeGetAttribute",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphExecKernelNodeSetParams
  {"cuGraphExecKernelNodeSetParams",                       {"hipGraphExecKernelNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphKernelNodeGetParams
  {"cuGraphKernelNodeGetParams",                           {"hipGraphKernelNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphKernelNodeSetAttribute
  {"cuGraphKernelNodeSetAttribute",                        {"hipGraphKernelNodeSetAttribute",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphKernelNodeSetParams
  {"cuGraphKernelNodeSetParams",                           {"hipGraphKernelNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphLaunch
  {"cuGraphLaunch",                                        {"hipGraphLaunch",                                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphMemcpyNodeGetParams (?)
  {"cuGraphMemcpyNodeGetParams",                           {"hipDrvGraphMemcpyNodeGetParams",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphMemcpyNodeSetParams (?)
  {"cuGraphMemcpyNodeSetParams",                           {"hipDrvGraphMemcpyNodeSetParams",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphMemsetNodeGetParams
  {"cuGraphMemsetNodeGetParams",                           {"hipGraphMemsetNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphMemsetNodeSetParams
  {"cuGraphMemsetNodeSetParams",                           {"hipGraphMemsetNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeFindInClone
  {"cuGraphNodeFindInClone",                               {"hipGraphNodeFindInClone",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeGetDependencies
  {"cuGraphNodeGetDependencies",                           {"hipGraphNodeGetDependencies",                             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeGetDependencies_v2
  {"cuGraphNodeGetDependencies_v2",                        {"hipGraphNodeGetDependencies_v2",                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetDependentNodes
  {"cuGraphNodeGetDependentNodes",                         {"hipGraphNodeGetDependentNodes",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeGetDependentNodes_v2
  {"cuGraphNodeGetDependentNodes_v2",                      {"hipGraphNodeGetDependentNodes_v2",                        "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetEnabled
  {"cuGraphNodeGetEnabled",                                {"hipGraphNodeGetEnabled",                                  "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeGetType
  {"cuGraphNodeGetType",                                   {"hipGraphNodeGetType",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphNodeSetEnabled
  {"cuGraphNodeSetEnabled",                                {"hipGraphNodeSetEnabled",                                  "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphRemoveDependencies
  {"cuGraphRemoveDependencies",                            {"hipGraphRemoveDependencies",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphRemoveDependencies_v2
  {"cuGraphRemoveDependencies_v2",                         {"hipGraphRemoveDependencies_v2",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // no analogue
  {"cuGraphExecMemcpyNodeSetParams",                       {"hipDrvGraphExecMemcpyNodeSetParams",                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // no analogue
  {"cuGraphExecMemsetNodeSetParams",                       {"hipDrvGraphExecMemsetNodeSetParams",                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExecHostNodeSetParams
  {"cuGraphExecHostNodeSetParams",                         {"hipGraphExecHostNodeSetParams",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // TODO: take into account the new signature since 12.0
  // cudaGraphExecUpdate
  {"cuGraphExecUpdate",                                    {"hipGraphExecUpdate",                                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddEventRecordNode
  {"cuGraphAddEventRecordNode",                            {"hipGraphAddEventRecordNode",                              "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphEventRecordNodeGetEvent
  {"cuGraphEventRecordNodeGetEvent",                       {"hipGraphEventRecordNodeGetEvent",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphEventRecordNodeSetEvent
  {"cuGraphEventRecordNodeSetEvent",                       {"hipGraphEventRecordNodeSetEvent",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddEventWaitNode
  {"cuGraphAddEventWaitNode",                              {"hipGraphAddEventWaitNode",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphEventWaitNodeGetEvent
  {"cuGraphEventWaitNodeGetEvent",                         {"hipGraphEventWaitNodeGetEvent",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphEventWaitNodeSetEvent
  {"cuGraphEventWaitNodeSetEvent",                         {"hipGraphEventWaitNodeSetEvent",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphExecChildGraphNodeSetParams
  {"cuGraphExecChildGraphNodeSetParams",                   {"hipGraphExecChildGraphNodeSetParams",                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphExecEventRecordNodeSetEvent
  {"cuGraphExecEventRecordNodeSetEvent",                   {"hipGraphExecEventRecordNodeSetEvent",                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphExecEventWaitNodeSetEvent
  {"cuGraphExecEventWaitNodeSetEvent",                     {"hipGraphExecEventWaitNodeSetEvent",                       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphUpload
  {"cuGraphUpload",                                        {"hipGraphUpload",                                          "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddExternalSemaphoresSignalNode
  {"cuGraphAddExternalSemaphoresSignalNode",               {"hipGraphAddExternalSemaphoresSignalNode",                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExternalSemaphoresSignalNodeGetParams
  {"cuGraphExternalSemaphoresSignalNodeGetParams",         {"hipGraphExternalSemaphoresSignalNodeGetParams",           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExternalSemaphoresSignalNodeSetParams
  {"cuGraphExternalSemaphoresSignalNodeSetParams",         {"hipGraphExternalSemaphoresSignalNodeSetParams",           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphAddExternalSemaphoresWaitNode
  {"cuGraphAddExternalSemaphoresWaitNode",                 {"hipGraphAddExternalSemaphoresWaitNode",                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExternalSemaphoresWaitNodeGetParams
  {"cuGraphExternalSemaphoresWaitNodeGetParams",           {"hipGraphExternalSemaphoresWaitNodeGetParams",             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExternalSemaphoresWaitNodeSetParams
  {"cuGraphExternalSemaphoresWaitNodeSetParams",           {"hipGraphExternalSemaphoresWaitNodeSetParams",             "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExecExternalSemaphoresSignalNodeSetParams
  {"cuGraphExecExternalSemaphoresSignalNodeSetParams",     {"hipGraphExecExternalSemaphoresSignalNodeSetParams",       "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExecExternalSemaphoresWaitNodeSetParams
  {"cuGraphExecExternalSemaphoresWaitNodeSetParams",       {"hipGraphExecExternalSemaphoresWaitNodeSetParams",         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaUserObjectCreate
  {"cuUserObjectCreate",                                   {"hipUserObjectCreate",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaUserObjectRetain
  {"cuUserObjectRetain",                                   {"hipUserObjectRetain",                                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaUserObjectRelease
  {"cuUserObjectRelease",                                  {"hipUserObjectRelease",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphRetainUserObject
  {"cuGraphRetainUserObject",                              {"hipGraphRetainUserObject",                                "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphReleaseUserObject
  {"cuGraphReleaseUserObject",                             {"hipGraphReleaseUserObject",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphAddMemAllocNode
  {"cuGraphAddMemAllocNode",                               {"hipGraphAddMemAllocNode",                                 "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphMemAllocNodeGetParams
  {"cuGraphMemAllocNodeGetParams",                         {"hipGraphMemAllocNodeGetParams",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // no analogue
  {"cuGraphAddMemFreeNode",                                {"hipDrvGraphAddMemFreeNode",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphMemFreeNodeGetParams
  {"cuGraphMemFreeNodeGetParams",                          {"hipGraphMemFreeNodeGetParams",                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaDeviceGraphMemTrim
  {"cuDeviceGraphMemTrim",                                 {"hipDeviceGraphMemTrim",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaDeviceGetGraphMemAttribute
  {"cuDeviceGetGraphMemAttribute",                         {"hipDeviceGetGraphMemAttribute",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaDeviceSetGraphMemAttribute
  {"cuDeviceSetGraphMemAttribute",                         {"hipDeviceSetGraphMemAttribute",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphInstantiateWithFlags
  {"cuGraphInstantiateWithFlags",                          {"hipGraphInstantiateWithFlags",                            "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  //
  {"cuGraphAddBatchMemOpNode",                             {"hipGraphAddBatchMemOpNode",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  //
  {"cuGraphBatchMemOpNodeGetParams",                       {"hipGraphBatchMemOpNodeGetParams",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  //
  {"cuGraphBatchMemOpNodeSetParams",                       {"hipGraphBatchMemOpNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  //
  {"cuGraphExecBatchMemOpNodeSetParams",                   {"hipGraphExecBatchMemOpNodeSetParams",                     "", CONV_GRAPH, API_DRIVER, SEC::GRAPH}},
  // cudaGraphInstantiateWithParams
  {"cuGraphInstantiateWithParams",                         {"hipGraphInstantiateWithParams",                           "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExecGetFlags
  {"cuGraphExecGetFlags",                                  {"hipGraphExecGetFlags",                                    "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphAddNode
  {"cuGraphAddNode",                                       {"hipGraphAddNode",                                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphAddNode_v2
  {"cuGraphAddNode_v2",                                    {"hipGraphAddNode_v2",                                      "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},
  // cudaGraphNodeSetParams
  {"cuGraphNodeSetParams",                                 {"hipGraphNodeSetParams",                                   "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphExecNodeSetParams
  {"cuGraphExecNodeSetParams",                             {"hipGraphExecNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_EXPERIMENTAL}},
  // cudaGraphConditionalHandleCreate
  {"cuGraphConditionalHandleCreate",                       {"hipGraphConditionalHandleCreate",                         "", CONV_GRAPH, API_DRIVER, SEC::GRAPH, HIP_UNSUPPORTED}},

  // 25. Occupancy
  // cudaOccupancyAvailableDynamicSMemPerBlock
  {"cuOccupancyAvailableDynamicSMemPerBlock",              {"hipModuleOccupancyAvailableDynamicSMemPerBlock",          "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY, HIP_UNSUPPORTED}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",      "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags","", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY}},
  // cudaOccupancyMaxPotentialBlockSize
  {"cuOccupancyMaxPotentialBlockSize",                     {"hipModuleOccupancyMaxPotentialBlockSize",                 "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY}},
  // cudaOccupancyMaxPotentialBlockSizeWithFlags
  {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipModuleOccupancyMaxPotentialBlockSizeWithFlags",        "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY}},
  // cudaOccupancyMaxPotentialClusterSize
  {"cuOccupancyMaxPotentialClusterSize",                   {"hipOccupancyMaxPotentialClusterSize",                     "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY, HIP_UNSUPPORTED}},
  // cudaOccupancyMaxActiveClusters
  {"cuOccupancyMaxActiveClusters",                         {"hipOccupancyMaxActiveClusters",                           "", CONV_OCCUPANCY, API_DRIVER, SEC::OCCUPANCY, HIP_UNSUPPORTED}},

  // 26. Texture Reference Management [DEPRECATED]
  // no analogues
  {"cuTexRefGetAddress",                                   {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetAddress_v2",                                {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetAddressMode",                               {"hipTexRefGetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetArray",                                     {"hipTexRefGetArray",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, CUDA_DEPRECATED | HIP_REMOVED}},
  {"cuTexRefGetBorderColor",                               {"hipTexRefGetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, CUDA_DEPRECATED | HIP_UNSUPPORTED}},
  {"cuTexRefGetFilterMode",                                {"hipTexRefGetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetFlags",                                     {"hipTexRefGetFlags",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetFormat",                                    {"hipTexRefGetFormat",                                      "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetMaxAnisotropy",                             {"hipTexRefGetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetMipmapFilterMode",                          {"hipTexRefGetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetMipmapLevelBias",                           {"hipTexRefGetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefGetMipmapLevelClamp",                          {"hipTexRefGetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  // TODO: [HIP] fix typo hipTexRefGetMipMappedArray -> hipTexRefGetMipmappedArray
  {"cuTexRefGetMipmappedArray",                            {"hipTexRefGetMipMappedArray",                              "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetAddress",                                   {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetAddress_v2",                                {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetAddress2D",                                 {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetAddress2D_v2",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_DEPRECATED}},
  {"cuTexRefSetAddress2D_v3",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_DEPRECATED}},
  {"cuTexRefSetAddressMode",                               {"hipTexRefSetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetArray",                                     {"hipTexRefSetArray",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetBorderColor",                               {"hipTexRefSetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetFilterMode",                                {"hipTexRefSetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetFlags",                                     {"hipTexRefSetFlags",                                       "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetFormat",                                    {"hipTexRefSetFormat",                                      "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetMaxAnisotropy",                             {"hipTexRefSetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetMipmapFilterMode",                          {"hipTexRefSetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetMipmapLevelBias",                           {"hipTexRefSetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetMipmapLevelClamp",                          {"hipTexRefSetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefSetMipmappedArray",                            {"hipTexRefSetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, DEPRECATED}},
  {"cuTexRefCreate",                                       {"hipTexRefCreate",                                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuTexRefDestroy",                                      {"hipTexRefDestroy",                                        "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 27. Surface Reference Management [DEPRECATED]
  // no analogues
  {"cuSurfRefGetArray",                                    {"hipSurfRefGetArray",                                      "", CONV_SURFACE, API_DRIVER, SEC::SURFACE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuSurfRefSetArray",                                    {"hipSurfRefSetArray",                                      "", CONV_SURFACE, API_DRIVER, SEC::SURFACE_DEPRECATED, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 28. Texture Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateTextureObject due to different signatures
  {"cuTexObjectCreate",                                    {"hipTexObjectCreate",                                      "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE}},
  // cudaDestroyTextureObject
  {"cuTexObjectDestroy",                                   {"hipTexObjectDestroy",                                     "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectResourceDesc due to different signatures
  {"cuTexObjectGetResourceDesc",                           {"hipTexObjectGetResourceDesc",                             "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE}},
  // cudaGetTextureObjectResourceViewDesc
  {"cuTexObjectGetResourceViewDesc",                       {"hipTexObjectGetResourceViewDesc",                         "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                            {"hipTexObjectGetTextureDesc",                              "", CONV_TEXTURE, API_DRIVER, SEC::TEXTURE}},

  // 29. Surface Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateSurfaceObject due to different signatures
  {"cuSurfObjectCreate",                                   {"hipSurfObjectCreate",                                     "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED}},
  // cudaDestroySurfaceObject
  {"cuSurfObjectDestroy",                                  {"hipSurfObjectDestroy",                                    "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetSurfaceObjectResourceDesc due to different signatures
  {"cuSurfObjectGetResourceDesc",                          {"hipSurfObjectGetResourceDesc",                            "", CONV_TEXTURE, API_DRIVER, SEC::SURFACE, HIP_UNSUPPORTED}},

  // 30. Tensor Core Management
  //
  {"cuTensorMapEncodeTiled",                               {"hipTensorMapEncodeTiled",                                 "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED}},
  //
  {"cuTensorMapEncodeIm2col",                              {"hipTensorMapEncodeIm2col",                                "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED}},
  //
  {"cuTensorMapReplaceAddress",                            {"hipTensorMapReplaceAddress",                              "", CONV_TENSOR, API_DRIVER, SEC::TENSOR, HIP_UNSUPPORTED}},

  // 31. Peer Context Memory Access
  // no analogue
  // NOTE: Not equal to cudaDeviceEnablePeerAccess due to different signatures
  {"cuCtxEnablePeerAccess",                                {"hipCtxEnablePeerAccess",                                  "", CONV_PEER, API_DRIVER, SEC::PEER, HIP_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaDeviceDisablePeerAccess due to different signatures
  {"cuCtxDisablePeerAccess",                               {"hipCtxDisablePeerAccess",                                 "", CONV_PEER, API_DRIVER, SEC::PEER, HIP_DEPRECATED}},
  // cudaDeviceCanAccessPeer
  {"cuDeviceCanAccessPeer",                                {"hipDeviceCanAccessPeer",                                  "", CONV_PEER, API_DRIVER, SEC::PEER}},
  // cudaDeviceGetP2PAttribute
  {"cuDeviceGetP2PAttribute",                              {"hipDeviceGetP2PAttribute",                                "", CONV_PEER, API_DRIVER, SEC::PEER}},

  // 32. Graphics Interoperability
  // cudaGraphicsMapResources
  {"cuGraphicsMapResources",                               {"hipGraphicsMapResources",                                 "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},
  // cudaGraphicsResourceGetMappedMipmappedArray
  {"cuGraphicsResourceGetMappedMipmappedArray",            {"hipGraphicsResourceGetMappedMipmappedArray",              "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer",                   {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer_v2",                {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags",                        {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags_v2",                     {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS, HIP_UNSUPPORTED}},
  // cudaGraphicsSubResourceGetMappedArray
  {"cuGraphicsSubResourceGetMappedArray",                  {"hipGraphicsSubResourceGetMappedArray",                    "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},
  // cudaGraphicsUnmapResources
  {"cuGraphicsUnmapResources",                             {"hipGraphicsUnmapResources",                               "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},
  // cudaGraphicsUnregisterResource
  {"cuGraphicsUnregisterResource",                         {"hipGraphicsUnregisterResource",                           "", CONV_GRAPHICS, API_DRIVER, SEC::GRAPHICS}},

  // 33. Driver Entry Point Access
  // cudaGetDriverEntryPoint
  {"cuGetProcAddress",                                     {"hipGetProcAddress",                                       "", CONV_DRIVER_ENTRY_POINT, API_DRIVER, SEC::DRIVER_ENTRY_POINT, HIP_EXPERIMENTAL}},

  // 34. Coredump Attributes Control API
  //
  {"cuCoredumpGetAttribute",                               {"hipCoredumpGetAttribute",                                 "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED}},
  //
  {"cuCoredumpGetAttributeGlobal",                         {"hipCoredumpGetAttributeGlobal",                           "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED}},
  //
  {"cuCoredumpSetAttribute",                               {"hipCoredumpSetAttribute",                                 "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED}},
  //
  {"cuCoredumpSetAttributeGlobal",                         {"hipCoredumpSetAttributeGlobal",                           "", CONV_COREDUMP, API_DRIVER, SEC::COREDUMP, HIP_UNSUPPORTED}},

  // 35. Profiler Control [DEPRECATED]
  // cudaProfilerInitialize
  {"cuProfilerInitialize",                                 {"hipProfilerInitialize",                                   "", CONV_PROFILER, API_DRIVER, SEC::PROFILER_DEPRECATED, HIP_UNSUPPORTED}},

  // 36. Profiler Control
  // cudaProfilerStart
  {"cuProfilerStart",                                      {"hipProfilerStart",                                        "", CONV_PROFILER, API_DRIVER, SEC::PROFILER}},
  // cudaProfilerStop
  {"cuProfilerStop",                                       {"hipProfilerStop",                                         "", CONV_PROFILER, API_DRIVER, SEC::PROFILER}},

  // 37. OpenGL Interoperability
  // cudaGLGetDevices
  {"cuGLGetDevices",                                       {"hipGLGetDevices",                                         "", CONV_OPENGL, API_DRIVER, SEC::OPENGL}},
  // cudaGraphicsGLRegisterBuffer
  {"cuGraphicsGLRegisterBuffer",                           {"hipGraphicsGLRegisterBuffer",                             "", CONV_OPENGL, API_DRIVER, SEC::OPENGL}},
  // cudaGraphicsGLRegisterImage
  {"cuGraphicsGLRegisterImage",                            {"hipGraphicsGLRegisterImage",                              "", CONV_OPENGL, API_DRIVER, SEC::OPENGL}},
  // cudaWGLGetDevice
  {"cuWGLGetDevice",                                       {"hipWGLGetDevice",                                         "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED}},

  // 37. OpenGL Interoperability [DEPRECATED]
  // no analogue
  {"cuGLCtxCreate",                                        {"hipGLCtxCreate",                                          "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuGLInit",                                             {"hipGLInit",                                               "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObject due to different signatures
  {"cuGLMapBufferObject",                                  {"hipGLMapBufferObject_",                                   "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObjectAsync due to different signatures
  {"cuGLMapBufferObjectAsync",                             {"hipGLMapBufferObjectAsync_",                              "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLRegisterBufferObject
  {"cuGLRegisterBufferObject",                             {"hipGLRegisterBufferObject",                               "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLSetBufferObjectMapFlags
  {"cuGLSetBufferObjectMapFlags",                          {"hipGLSetBufferObjectMapFlags",                            "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnmapBufferObject
  {"cuGLUnmapBufferObject",                                {"hipGLUnmapBufferObject",                                  "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnmapBufferObjectAsync
  {"cuGLUnmapBufferObjectAsync",                           {"hipGLUnmapBufferObjectAsync",                             "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnregisterBufferObject
  {"cuGLUnregisterBufferObject",                           {"hipGLUnregisterBufferObject",                             "", CONV_OPENGL, API_DRIVER, SEC::OPENGL, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 38. Direct3D 9 Interoperability
  // no analogue
  {"cuD3D9CtxCreate",                                      {"hipD3D9CtxCreate",                                        "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},
    // no analogue
  {"cuD3D9CtxCreateOnDevice",                              {"hipD3D9CtxCreateOnDevice",                                "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevice
  {"cuD3D9GetDevice",                                      {"hipD3D9GetDevice",                                        "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevices
  {"cuD3D9GetDevices",                                     {"hipD3D9GetDevices",                                       "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},
  // cudaD3D9GetDirect3DDevice
  {"cuD3D9GetDirect3DDevice",                              {"hipD3D9GetDirect3DDevice",                                "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D9RegisterResource
  {"cuGraphicsD3D9RegisterResource",                       {"hipGraphicsD3D9RegisterResource",                         "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED}},

  // 38. Direct3D 9 Interoperability [DEPRECATED]
  // cudaD3D9MapResources
  {"cuD3D9MapResources",                                   {"hipD3D9MapResources",                                     "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9RegisterResource
  {"cuD3D9RegisterResource",                               {"hipD3D9RegisterResource",                                 "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedArray
  {"cuD3D9ResourceGetMappedArray",                         {"hipD3D9ResourceGetMappedArray",                           "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedPitch
  {"cuD3D9ResourceGetMappedPitch",                         {"hipD3D9ResourceGetMappedPitch",                           "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedPointer
  {"cuD3D9ResourceGetMappedPointer",                       {"hipD3D9ResourceGetMappedPointer",                         "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedSize
  {"cuD3D9ResourceGetMappedSize",                          {"hipD3D9ResourceGetMappedSize",                            "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetSurfaceDimensions
  {"cuD3D9ResourceGetSurfaceDimensions",                   {"hipD3D9ResourceGetSurfaceDimensions",                     "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceSetMapFlags
  {"cuD3D9ResourceSetMapFlags",                            {"hipD3D9ResourceSetMapFlags",                              "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9UnmapResources
  {"cuD3D9UnmapResources",                                 {"hipD3D9UnmapResources",                                   "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9UnregisterResource
  {"cuD3D9UnregisterResource",                             {"hipD3D9UnregisterResource",                               "", CONV_D3D9, API_DRIVER, SEC::D3D9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 39. Direct3D 10 Interoperability
  // cudaD3D10GetDevice
  {"cuD3D10GetDevice",                                     {"hipD3D10GetDevice",                                       "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED}},
  // cudaD3D10GetDevices
  {"cuD3D10GetDevices",                                    {"hipD3D10GetDevices",                                      "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D10RegisterResource
  {"cuGraphicsD3D10RegisterResource",                      {"hipGraphicsD3D10RegisterResource",                        "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED}},

  // 39. Direct3D 10 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D10CtxCreate",                                     {"hipD3D10CtxCreate",                                       "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuD3D10CtxCreateOnDevice",                             {"hipD3D10CtxCreateOnDevice",                               "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10GetDirect3DDevice
  {"cuD3D10GetDirect3DDevice",                             {"hipD3D10GetDirect3DDevice",                               "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10MapResources
  {"cuD3D10MapResources",                                  {"hipD3D10MapResources",                                    "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10RegisterResource
  {"cuD3D10RegisterResource",                              {"hipD3D10RegisterResource",                                "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedArray
  {"cuD3D10ResourceGetMappedArray",                        {"hipD3D10ResourceGetMappedArray",                          "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedPitch
  {"cuD3D10ResourceGetMappedPitch",                        {"hipD3D10ResourceGetMappedPitch",                          "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedPointer
  {"cuD3D10ResourceGetMappedPointer",                      {"hipD3D10ResourceGetMappedPointer",                        "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedSize
  {"cuD3D10ResourceGetMappedSize",                         {"hipD3D10ResourceGetMappedSize",                           "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetSurfaceDimensions
  {"cuD3D10ResourceGetSurfaceDimensions",                  {"hipD3D10ResourceGetSurfaceDimensions",                    "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceSetMapFlags
  {"cuD3D10ResourceSetMapFlags",                           {"hipD3D10ResourceSetMapFlags",                             "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10UnmapResources
  {"cuD3D10UnmapResources",                                {"hipD3D10UnmapResources",                                  "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10UnregisterResource
  {"cuD3D10UnregisterResource",                            {"hipD3D10UnregisterResource",                              "", CONV_D3D10, API_DRIVER, SEC::D3D10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 40. Direct3D 11 Interoperability
  // cudaD3D11GetDevice
  {"cuD3D11GetDevice",                                     {"hipD3D11GetDevice",                                       "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED}},
  // cudaD3D11GetDevices
  {"cuD3D11GetDevices",                                    {"hipD3D11GetDevices",                                      "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D11RegisterResource
  {"cuGraphicsD3D11RegisterResource",                      {"hipGraphicsD3D11RegisterResource",                        "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED}},

  // 40. Direct3D 11 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D11CtxCreate",                                     {"hipD3D11CtxCreate",                                       "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuD3D11CtxCreateOnDevice",                             {"hipD3D11CtxCreateOnDevice",                               "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D11GetDirect3DDevice
  {"cuD3D11GetDirect3DDevice",                             {"hipD3D11GetDirect3DDevice",                               "", CONV_D3D11, API_DRIVER, SEC::D3D11, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 41. VDPAU Interoperability
  // cudaGraphicsVDPAURegisterOutputSurface
  {"cuGraphicsVDPAURegisterOutputSurface",                 {"hipGraphicsVDPAURegisterOutputSurface",                   "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED}},
  // cudaGraphicsVDPAURegisterVideoSurface
  {"cuGraphicsVDPAURegisterVideoSurface",                  {"hipGraphicsVDPAURegisterVideoSurface",                    "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED}},
  // cudaVDPAUGetDevice
  {"cuVDPAUGetDevice",                                     {"hipVDPAUGetDevice",                                       "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED}},
  // no analogue
  {"cuVDPAUCtxCreate",                                     {"hipVDPAUCtxCreate",                                       "", CONV_VDPAU, API_DRIVER, SEC::VDPAU, HIP_UNSUPPORTED}},

  // 42. EGL Interoperability
  // cudaEGLStreamConsumerAcquireFrame
  {"cuEGLStreamConsumerAcquireFrame",                      {"hipEGLStreamConsumerAcquireFrame",                        "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnect
  {"cuEGLStreamConsumerConnect",                           {"hipEGLStreamConsumerConnect",                             "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnectWithFlags
  {"cuEGLStreamConsumerConnectWithFlags",                  {"hipEGLStreamConsumerConnectWithFlags",                    "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerDisconnect
  {"cuEGLStreamConsumerDisconnect",                        {"hipEGLStreamConsumerDisconnect",                          "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerReleaseFrame
  {"cuEGLStreamConsumerReleaseFrame",                      {"hipEGLStreamConsumerReleaseFrame",                        "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerConnect
  {"cuEGLStreamProducerConnect",                           {"hipEGLStreamProducerConnect",                             "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerDisconnect
  {"cuEGLStreamProducerDisconnect",                        {"hipEGLStreamProducerDisconnect",                          "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerPresentFrame
  {"cuEGLStreamProducerPresentFrame",                      {"hipEGLStreamProducerPresentFrame",                        "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerReturnFrame
  {"cuEGLStreamProducerReturnFrame",                       {"hipEGLStreamProducerReturnFrame",                         "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaGraphicsEGLRegisterImage
  {"cuGraphicsEGLRegisterImage",                           {"hipGraphicsEGLRegisterImage",                             "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedEglFrame
  {"cuGraphicsResourceGetMappedEglFrame",                  {"hipGraphicsResourceGetMappedEglFrame",                    "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
  // cudaEventCreateFromEGLSync
  {"cuEventCreateFromEGLSync",                             {"hipEventCreateFromEGLSync",                               "", CONV_EGL, API_DRIVER, SEC::EGL, HIP_UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DRIVER_FUNCTION_VER_MAP {
  {"cuDeviceGetLuid",                                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetNvSciSyncAttributes",                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetUuid",                                      {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"cuDeviceComputeCapability",                            {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuDeviceGetProperties",                                {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuDevicePrimaryCtxRelease_v2",                         {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuDevicePrimaryCtxReset_v2",                           {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuDevicePrimaryCtxSetFlags_v2",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuCtxResetPersistingL2Cache",                          {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuMemAddressFree",                                     {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemAddressReserve",                                  {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemCreate",                                          {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemExportToShareableHandle",                         {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemGetAccess",                                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemGetAllocationGranularity",                        {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemGetAllocationPropertiesFromHandle",               {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemImportFromShareableHandle",                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemMap",                                             {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemRelease",                                         {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemRetainAllocationHandle",                          {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuMemSetAccess",                                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemUnmap",                                           {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuMemAdvise",                                          {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuMemPrefetchAsync",                                   {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuMemRangeGetAttribute",                               {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuMemRangeGetAttributes",                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuStreamBeginCapture",                                 {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuStreamBeginCapture_v2",                              {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cuStreamBeginCapture_ptsz",                            {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cuStreamCopyAttributes",                               {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuStreamEndCapture",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuStreamGetAttribute",                                 {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuStreamGetCaptureInfo",                               {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cuStreamGetCtx",                                       {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"cuStreamIsCapturing",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuStreamSetAttribute",                                 {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuThreadExchangeStreamCaptureMode",                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cuDestroyExternalMemory",                              {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuDestroyExternalSemaphore",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuExternalMemoryGetMappedBuffer",                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuExternalMemoryGetMappedMipmappedArray",              {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuImportExternalMemory",                               {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuImportExternalSemaphore",                            {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuSignalExternalSemaphoresAsync",                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuWaitExternalSemaphoresAsync",                        {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuStreamBatchMemOp",                                   {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuStreamWaitValue32",                                  {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuStreamWaitValue64",                                  {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cuStreamWriteValue32",                                 {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuStreamWriteValue64",                                 {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cuFuncGetModule",                                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuFuncSetAttribute",                                   {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cuLaunchCooperativeKernel",                            {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cuLaunchCooperativeKernelMultiDevice",                 {CUDA_90,  CUDA_113, CUDA_0  }},
  {"cuLaunchHostFunc",                                     {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuFuncSetBlockShape",                                  {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuFuncSetSharedSize",                                  {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuLaunch",                                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuLaunchGrid",                                         {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuLaunchGridAsync",                                    {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuParamSetf",                                          {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuParamSeti",                                          {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuParamSetSize",                                       {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuParamSetTexRef",                                     {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuParamSetv",                                          {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGraphAddChildGraphNode",                             {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddDependencies",                               {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddEmptyNode",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddHostNode",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddKernelNode",                                 {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddMemcpyNode",                                 {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphAddMemsetNode",                                 {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphChildGraphNodeGetGraph",                        {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphClone",                                         {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphCreate",                                        {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphDestroy",                                       {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphDestroyNode",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphExecDestroy",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphGetEdges",                                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphGetNodes",                                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphGetRootNodes",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphHostNodeGetParams",                             {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphHostNodeSetParams",                             {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphInstantiate",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphInstantiate_v2",                                {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuGraphKernelNodeCopyAttributes",                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuGraphKernelNodeGetAttribute",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuGraphExecKernelNodeSetParams",                       {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cuGraphKernelNodeGetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphKernelNodeSetAttribute",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuGraphKernelNodeSetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphLaunch",                                        {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphMemcpyNodeGetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphMemcpyNodeSetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphMemsetNodeGetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphMemsetNodeSetParams",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeFindInClone",                               {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetDependencies",                           {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetDependentNodes",                         {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetType",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphRemoveDependencies",                            {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cuGraphExecMemcpyNodeSetParams",                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuGraphExecMemsetNodeSetParams",                       {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuGraphExecHostNodeSetParams",                         {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuGraphExecUpdate",                                    {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cuOccupancyAvailableDynamicSMemPerBlock",              {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cuTexRefGetAddress",                                   {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetAddress_v2",                                {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetAddressMode",                               {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetArray",                                     {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetBorderColor",                               {CUDA_80,  CUDA_110, CUDA_0  }},
  {"cuTexRefGetFilterMode",                                {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetFlags",                                     {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetFormat",                                    {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetMaxAnisotropy",                             {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetMipmapFilterMode",                          {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetMipmapLevelBias",                           {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetMipmapLevelClamp",                          {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefGetMipmappedArray",                            {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetAddress",                                   {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetAddress_v2",                                {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetAddress2D",                                 {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetAddressMode",                               {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetArray",                                     {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetBorderColor",                               {CUDA_80,  CUDA_110, CUDA_0  }},
  {"cuTexRefSetFilterMode",                                {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetFlags",                                     {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetFormat",                                    {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetMaxAnisotropy",                             {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetMipmapFilterMode",                          {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetMipmapLevelBias",                           {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetMipmapLevelClamp",                          {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefSetMipmappedArray",                            {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefCreate",                                       {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexRefDestroy",                                      {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuSurfRefGetArray",                                    {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuSurfRefSetArray",                                    {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuDeviceGetP2PAttribute",                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cuProfilerInitialize",                                 {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuGLCtxCreate",                                        {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLInit",                                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLMapBufferObject",                                  {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLMapBufferObjectAsync",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLRegisterBufferObject",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLSetBufferObjectMapFlags",                          {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLUnmapBufferObject",                                {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLUnmapBufferObjectAsync",                           {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuGLUnregisterBufferObject",                           {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9MapResources",                                   {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9RegisterResource",                               {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceGetMappedArray",                         {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceGetMappedPitch",                         {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceGetMappedPointer",                       {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceGetMappedSize",                          {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceGetSurfaceDimensions",                   {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9ResourceSetMapFlags",                            {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9UnmapResources",                                 {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D9UnregisterResource",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10CtxCreate",                                     {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10CtxCreateOnDevice",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10GetDirect3DDevice",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10MapResources",                                  {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10RegisterResource",                              {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceGetMappedArray",                        {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceGetMappedPitch",                        {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceGetMappedPointer",                      {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceGetMappedSize",                         {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceGetSurfaceDimensions",                  {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10ResourceSetMapFlags",                           {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10UnmapResources",                                {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D10UnregisterResource",                            {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D11CtxCreate",                                     {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D11CtxCreateOnDevice",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuD3D11GetDirect3DDevice",                             {CUDA_0,   CUDA_92,  CUDA_0  }},
  {"cuEGLStreamConsumerAcquireFrame",                      {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamConsumerConnect",                           {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamConsumerConnectWithFlags",                  {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamConsumerDisconnect",                        {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamConsumerReleaseFrame",                      {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamProducerConnect",                           {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamProducerDisconnect",                        {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamProducerPresentFrame",                      {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEGLStreamProducerReturnFrame",                       {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuGraphicsEGLRegisterImage",                           {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuGraphicsResourceGetMappedEglFrame",                  {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuEventCreateFromEGLSync",                             {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cuDeviceGetTexture1DLinearMaxWidth",                   {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuArrayGetSparseProperties",                           {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuMemMapArrayAsync",                                   {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuEventRecordWithFlags",                               {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphAddEventRecordNode",                            {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphEventRecordNodeGetEvent",                       {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphEventRecordNodeSetEvent",                       {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphAddEventWaitNode",                              {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphEventWaitNodeGetEvent",                         {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphEventWaitNodeSetEvent",                         {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphExecChildGraphNodeSetParams",                   {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphExecEventRecordNodeSetEvent",                   {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphExecEventWaitNodeSetEvent",                     {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuGraphUpload",                                        {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cuDeviceSetMemPool",                                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetMemPool",                                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetDefaultMemPool",                            {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuArrayGetPlane",                                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemFreeAsync",                                       {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemAllocAsync",                                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolTrimTo",                                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolSetAttribute",                                {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolGetAttribute",                                {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolSetAccess",                                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolGetAccess",                                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolCreate",                                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolDestroy",                                     {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemAllocFromPoolAsync",                              {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolExportToShareableHandle",                     {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolImportFromShareableHandle",                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolExportPointer",                               {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuMemPoolImportPointer",                               {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphAddExternalSemaphoresSignalNode",               {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExternalSemaphoresSignalNodeGetParams",         {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExternalSemaphoresSignalNodeSetParams",         {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphAddExternalSemaphoresWaitNode",                 {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExternalSemaphoresWaitNodeGetParams",           {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExternalSemaphoresWaitNodeSetParams",           {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExecExternalSemaphoresSignalNodeSetParams",     {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuGraphExecExternalSemaphoresWaitNodeSetParams",       {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cuStreamGetCaptureInfo_v2",                            {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuStreamUpdateCaptureDependencies",                    {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuGraphDebugDotPrint",                                 {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuUserObjectCreate",                                   {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuUserObjectRetain",                                   {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuUserObjectRelease",                                  {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuGraphRetainUserObject",                              {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuGraphReleaseUserObject",                             {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuGetProcAddress",                                     {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuFlushGPUDirectRDMAWrites",                           {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cuCtxCreate_v3",                                       {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetUuid_v2",                                   {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetExecAffinitySupport",                       {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuCtxGetExecAffinity",                                 {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuGraphAddMemAllocNode",                               {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuGraphMemAllocNodeGetParams",                         {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuGraphAddMemFreeNode",                                {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuGraphMemFreeNodeGetParams",                          {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuDeviceGraphMemTrim",                                 {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuDeviceGetGraphMemAttribute",                         {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuDeviceSetGraphMemAttribute",                         {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuGraphInstantiateWithFlags",                          {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cuArrayGetMemoryRequirements",                         {CUDA_116, CUDA_0,   CUDA_0  }},
  {"cuMipmappedArrayGetMemoryRequirements",                {CUDA_116, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeSetEnabled",                                {CUDA_116, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetEnabled",                                {CUDA_116, CUDA_0,   CUDA_0  }},
  {"cuMemGetHandleForAddressRange",                        {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuModuleGetLoadingMode",                               {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuStreamWaitValue32_v2",                               {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuStreamWaitValue64_v2",                               {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuStreamWriteValue32_v2",                              {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuStreamWriteValue64_v2",                              {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuStreamBatchMemOp_v2",                                {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuGraphAddBatchMemOpNode",                             {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuGraphBatchMemOpNodeGetParams",                       {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuGraphBatchMemOpNodeSetParams",                       {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuGraphExecBatchMemOpNodeSetParams",                   {CUDA_117, CUDA_0,   CUDA_0  }},
  {"cuLaunchKernelEx",                                     {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cuOccupancyMaxPotentialClusterSize",                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cuOccupancyMaxActiveClusters",                         {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cuCtxGetId",                                           {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuModuleGetTexRef",                                    {CUDA_0,   CUDA_120, CUDA_0  }},
  {"cuModuleGetSurfRef",                                   {CUDA_0,   CUDA_120, CUDA_0  }},
  {"cuLibraryLoadData",                                    {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryLoadFromFile",                                {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryUnload",                                      {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryGetKernel",                                   {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryGetModule",                                   {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuKernelGetFunction",                                  {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryGetGlobal",                                   {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryGetManaged",                                  {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuLibraryGetUnifiedFunction",                          {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuKernelGetAttribute",                                 {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuKernelSetAttribute",                                 {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuKernelSetCacheConfig",                               {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuStreamGetId",                                        {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuGraphInstantiateWithParams",                         {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuTensorMapEncodeTiled",                               {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuTensorMapEncodeIm2col",                              {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuTensorMapReplaceAddress",                            {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuGraphExecGetFlags",                                  {CUDA_120, CUDA_0,   CUDA_0  }},
  {"cuCtxSetFlags",                                        {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastCreate",                                    {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastAddDevice",                                 {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastBindMem",                                   {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastBindAddr",                                  {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastUnbind",                                    {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMulticastGetGranularity",                            {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuCoredumpGetAttribute",                               {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuCoredumpGetAttributeGlobal",                         {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuCoredumpSetAttribute",                               {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuCoredumpSetAttributeGlobal",                         {CUDA_121, CUDA_0,   CUDA_0  }},
  {"cuMemPrefetchAsync_v2",                                {CUDA_122, CUDA_0,   CUDA_0  }},
  {"cuMemAdvise_v2",                                       {CUDA_122, CUDA_0,   CUDA_0  }},
  {"cuGraphAddNode",                                       {CUDA_122, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeSetParams",                                 {CUDA_122, CUDA_0,   CUDA_0  }},
  {"cuGraphExecNodeSetParams",                             {CUDA_122, CUDA_0,   CUDA_0  }},
  {"cuKernelGetName",                                      {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuStreamBeginCaptureToGraph",                          {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuStreamGetCaptureInfo_v3",                            {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuStreamUpdateCaptureDependencies_v2",                 {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuFuncGetName",                                        {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphGetEdges_v2",                                   {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetDependencies_v2",                        {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphAddDependencies_v2",                            {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphRemoveDependencies_v2",                         {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphAddNode_v2",                                    {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphConditionalHandleCreate",                       {CUDA_123, CUDA_0,   CUDA_0  }},
  {"cuGraphNodeGetDependentNodes_v2",                      {CUDA_123, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_DRIVER_FUNCTION_VER_MAP {
  {"hipInit",                                              {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDriverGetVersion",                                  {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGet",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetName",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceTotalMem",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceComputeCapability",                           {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDevicePrimaryCtxGetState",                          {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDevicePrimaryCtxRelease",                           {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDevicePrimaryCtxReset",                             {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDevicePrimaryCtxRetain",                            {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDevicePrimaryCtxSetFlags",                          {HIP_1090, HIP_0,    HIP_0   }},
  {"hipCtxCreate",                                         {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxDestroy",                                        {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxGetApiVersion",                                  {HIP_1090, HIP_1090, HIP_0   }},
  {"hipCtxGetCacheConfig",                                 {HIP_1090, HIP_1090, HIP_0   }},
  {"hipCtxGetCurrent",                                     {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxGetDevice",                                      {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxGetFlags",                                       {HIP_1090, HIP_1090, HIP_0   }},
  {"hipCtxGetSharedMemConfig",                             {HIP_1090, HIP_1090, HIP_0   }},
  {"hipDeviceGetStreamPriorityRange",                      {HIP_2000, HIP_0,    HIP_0   }},
  {"hipCtxPopCurrent",                                     {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxPushCurrent",                                    {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxSetCacheConfig",                                 {HIP_1090, HIP_1090, HIP_0   }},
  {"hipCtxSetCurrent",                                     {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxSetSharedMemConfig",                             {HIP_1090, HIP_1090, HIP_0   }},
  {"hipCtxSynchronize",                                    {HIP_1090, HIP_1090, HIP_0   }},
  {"hipModuleGetFunction",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleGetGlobal",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleGetTexRef",                                   {HIP_1070, HIP_0,    HIP_0   }},
  {"hipModuleLoad",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleLoadData",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleLoadDataEx",                                  {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleUnload",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipArray3DCreate",                                     {HIP_1071, HIP_0,    HIP_0   }},
  {"hipArrayCreate",                                       {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemAllocPitch",                                     {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMemAllocHost",                                      {HIP_3000, HIP_3000, HIP_0   }},
  {"hipMemcpyParam2D",                                     {HIP_1070, HIP_0,    HIP_0   }},
  {"hipMemcpyParam2DAsync",                                {HIP_2080, HIP_0,    HIP_0   }},
  {"hipDrvMemcpy3D",                                       {HIP_3050, HIP_0,    HIP_0   }},
  {"hipDrvMemcpy3DAsync",                                  {HIP_3050, HIP_0,    HIP_0   }},
  {"hipMemcpyAtoH",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemcpyDtoD",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyDtoDAsync",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyDtoH",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyDtoHAsync",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyHtoA",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemcpyHtoD",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyHtoDAsync",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemGetAddressRange",                                {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemsetD16",                                         {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMemsetD16Async",                                    {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMemsetD32",                                         {HIP_2030, HIP_0,    HIP_0   }},
  {"hipMemsetD32Async",                                    {HIP_2030, HIP_0,    HIP_0   }},
  {"hipMemsetD8",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemsetD8Async",                                     {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMipmappedArrayCreate",                              {HIP_3050, HIP_5070, HIP_0   }},
  {"hipMipmappedArrayDestroy",                             {HIP_3050, HIP_5070, HIP_0   }},
  {"hipMipmappedArrayGetLevel",                            {HIP_3050, HIP_5070, HIP_0   }},
  {"hipFuncGetAttribute",                                  {HIP_2080, HIP_0,    HIP_0   }},
  {"hipModuleLaunchKernel",                                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",   {HIP_3050, HIP_0,    HIP_0   }},
  {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {HIP_3050, HIP_0,    HIP_0   }},
  {"hipModuleOccupancyMaxPotentialBlockSize",              {HIP_3050, HIP_0,    HIP_0   }},
  {"hipModuleOccupancyMaxPotentialBlockSizeWithFlags",     {HIP_3050, HIP_0,    HIP_0   }},
  {"hipTexRefGetAddress",                                  {HIP_3000, HIP_4030, HIP_0   }},
  {"hipTexRefGetAddressMode",                              {HIP_3000, HIP_4030, HIP_0   }},
  {"hipTexRefGetArray",                                    {HIP_3000, HIP_0,    HIP_4020}},
  {"hipTexRefGetFilterMode",                               {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetFlags",                                    {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetFormat",                                   {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetMaxAnisotropy",                            {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetMipmapFilterMode",                         {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetMipmapLevelBias" ,                         {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetMipmapLevelClamp",                         {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefGetMipMappedArray",                           {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefSetAddress",                                  {HIP_1070, HIP_4030, HIP_0   }},
  {"hipTexRefSetAddress2D",                                {HIP_1070, HIP_4030, HIP_0   }},
  {"hipTexRefSetAddressMode",                              {HIP_1090, HIP_5030, HIP_0   }},
  {"hipTexRefSetArray",                                    {HIP_1090, HIP_5030, HIP_0   }},
  {"hipTexRefSetBorderColor",                              {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefSetFilterMode",                               {HIP_1090, HIP_5030, HIP_0   }},
  {"hipTexRefSetFlags",                                    {HIP_1090, HIP_5030, HIP_0   }},
  {"hipTexRefSetFormat",                                   {HIP_1090, HIP_5030, HIP_0   }},
  {"hipTexRefSetMaxAnisotropy",                            {HIP_3050, HIP_4030, HIP_0   }},
  {"hipTexRefSetMipmapFilterMode",                         {HIP_3050, HIP_5030, HIP_0   }},
  {"hipTexRefSetMipmapLevelBias",                          {HIP_3050, HIP_5030, HIP_0   }},
  {"hipTexRefSetMipmapLevelClamp",                         {HIP_3050, HIP_5030, HIP_0   }},
  {"hipTexRefSetMipmappedArray",                           {HIP_3050, HIP_5030, HIP_0   }},
  {"hipTexObjectCreate",                                   {HIP_3050, HIP_0,    HIP_0   }},
  {"hipTexObjectDestroy",                                  {HIP_3050, HIP_0,    HIP_0   }},
  {"hipTexObjectGetResourceDesc",                          {HIP_3050, HIP_0,    HIP_0   }},
  {"hipTexObjectGetResourceViewDesc",                      {HIP_3050, HIP_0,    HIP_0   }},
  {"hipTexObjectGetTextureDesc",                           {HIP_3050, HIP_0,    HIP_0   }},
  {"hipCtxEnablePeerAccess",                               {HIP_1060, HIP_1090, HIP_0   }},
  {"hipCtxDisablePeerAccess",                              {HIP_1060, HIP_1090, HIP_0   }},
  {"hipStreamWaitValue32",                                 {HIP_4020, HIP_0,    HIP_0   }},
  {"hipStreamWaitValue64",                                 {HIP_4020, HIP_0,    HIP_0   }},
  {"hipStreamWriteValue32",                                {HIP_4020, HIP_0,    HIP_0   }},
  {"hipStreamWriteValue64",                                {HIP_4020, HIP_0,    HIP_0   }},
  {"hipArrayDestroy",                                      {HIP_4020, HIP_0,    HIP_0   }},
  {"hipDrvMemcpy2DUnaligned",                              {HIP_4020, HIP_0,    HIP_0   }},
  {"hipPointerGetAttribute",                               {HIP_5000, HIP_0,    HIP_0   }},
  {"hipDrvPointerGetAttributes",                           {HIP_5000, HIP_0,    HIP_0   }},
  {"hipStreamGetCaptureInfo",                              {HIP_5000, HIP_0,    HIP_0   }},
  {"hipStreamGetCaptureInfo_v2",                           {HIP_5000, HIP_0,    HIP_0   }},
  {"hipStreamIsCapturing",                                 {HIP_5000, HIP_0,    HIP_0   }},
  {"hipStreamUpdateCaptureDependencies",                   {HIP_5000, HIP_0,    HIP_0   }},
  {"hipGraphicsGLRegisterImage",                           {HIP_5010, HIP_0,    HIP_0   }},
  {"hipGraphicsSubResourceGetMappedArray",                 {HIP_5010, HIP_0,    HIP_0   }},
  {"hipDeviceGetUuid",                                     {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemAddressFree",                                    {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemAddressReserve",                                 {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemCreate",                                         {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemExportToShareableHandle",                        {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemGetAccess",                                      {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemGetAllocationGranularity",                       {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemGetAllocationPropertiesFromHandle",              {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemImportFromShareableHandle",                      {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemMap",                                            {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemMapArrayAsync",                                  {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemRelease",                                        {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemRetainAllocationHandle",                         {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemSetAccess",                                      {HIP_5020, HIP_0,    HIP_0   }},
  {"hipMemUnmap",                                          {HIP_5020, HIP_0,    HIP_0   }},
  {"hiprtcLinkCreate",                                     {HIP_5030, HIP_0,    HIP_0   }},
  {"hiprtcLinkAddFile",                                    {HIP_5030, HIP_0,    HIP_0   }},
  {"hiprtcLinkAddData",                                    {HIP_5030, HIP_0,    HIP_0   }},
  {"hiprtcLinkComplete",                                   {HIP_5030, HIP_0,    HIP_0   }},
  {"hiprtcLinkDestroy",                                    {HIP_5030, HIP_0,    HIP_0   }},
  {"hipDrvGetErrorName",                                   {HIP_5040, HIP_0,    HIP_0   }},
  {"hipDrvGetErrorString",                                 {HIP_5040, HIP_0,    HIP_0   }},
  {"hipPointerSetAttribute",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"hipModuleLaunchCooperativeKernel",                     {HIP_5050, HIP_0,    HIP_0   }},
  {"hipModuleLaunchCooperativeKernelMultiDevice",          {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphAddMemAllocNode",                              {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphMemAllocNodeGetParams",                        {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphAddMemFreeNode",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphMemFreeNodeGetParams",                         {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphDebugDotPrint",                                {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphKernelNodeCopyAttributes",                     {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphNodeSetEnabled",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"hipGraphNodeGetEnabled",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"hipArrayGetDescriptor",                                {HIP_5060, HIP_0,    HIP_0   }},
  {"hipArray3DGetDescriptor",                              {HIP_5060, HIP_0,    HIP_0   }},
  {"hipDrvGraphAddMemcpyNode",                             {HIP_6000, HIP_0,    HIP_0,  }},
  {"hipGetProcAddress",                                    {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphMemcpyNodeGetParams",                       {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphMemcpyNodeSetParams",                       {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphAddMemsetNode",                             {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphAddMemFreeNode",                            {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphExecMemcpyNodeSetParams",                   {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGraphExecMemsetNodeSetParams",                   {HIP_6010, HIP_0,    HIP_0,  HIP_LATEST}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_DRIVER_API_SECTION_MAP {
  {SEC::DATA_TYPES, "CUDA Driver Data Types"},
  {SEC::ERROR, "Error Handling"},
  {SEC::INIT, "Initialization"},
  {SEC::VERSION, "Version Management"},
  {SEC::DEVICE, "Device Management"},
  {SEC::DEVICE_DEPRECATED, "Device Management [DEPRECATED]"},
  {SEC::PRIMARY_CONTEXT, "Primary Context Management"},
  {SEC::CONTEXT, "Context Management"},
  {SEC::CONTEXT_DEPRECATED, "Context Management [DEPRECATED]"},
  {SEC::MODULE, "Module Management"},
  {SEC::MODULE_DEPRECATED, "Module Management [DEPRECATED]"},
  {SEC::LIBRARY, "Library Management"},
  {SEC::MEMORY, "Memory Management"},
  {SEC::VIRTUAL_MEMORY, "Virtual Memory Management"},
  {SEC::ORDERED_MEMORY, "Stream Ordered Memory Allocator"},
  {SEC::MULTICAST, "Multicast Object Management"},
  {SEC::UNIFIED, "Unified Addressing"},
  {SEC::STREAM, "Stream Management"},
  {SEC::EVENT, "Event Management"},
  {SEC::EXTERNAL_RES, "External Resource Interoperability"},
  {SEC::STREAM_MEMORY, "Stream Memory Operations"},
  {SEC::EXECUTION, "Execution Control"},
  {SEC::EXECUTION_DEPRECATED, "Execution Control [DEPRECATED]"},
  {SEC::GRAPH, "Graph Management"},
  {SEC::OCCUPANCY, "Occupancy"},
  {SEC::TEXTURE_DEPRECATED, "Texture Reference Management [DEPRECATED]"},
  {SEC::SURFACE_DEPRECATED, "Surface Reference Management [DEPRECATED]"},
  {SEC::TEXTURE, "Texture Object Management"},
  {SEC::SURFACE, "Surface Object Management"},
  {SEC::TENSOR, "Tensor Core Management"},
  {SEC::PEER, "Peer Context Memory Access"},
  {SEC::GRAPHICS, "Graphics Interoperability"},
  {SEC::DRIVER_ENTRY_POINT, "Driver Entry Point Access"},
  {SEC::COREDUMP, "Coredump Attributes Control API"},
  {SEC::PROFILER_DEPRECATED, "Profiler Control [DEPRECATED]"},
  {SEC::PROFILER, "Profiler Control"},
  {SEC::OPENGL, "OpenGL Interoperability"},
  {SEC::D3D9, "Direct3D 9 Interoperability"},
  {SEC::D3D10, "Direct3D 10 Interoperability"},
  {SEC::D3D11, "Direct3D 11 Interoperability"},
  {SEC::VDPAU, "VDPAU Interoperability"},
  {SEC::EGL, "EGL Interoperability"},
};
