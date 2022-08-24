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

// Map of all CUDA Driver API functions
const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP {
  // 2. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  {"cuGetErrorName",                                       {"hipDrvGetErrorName",                                      "", CONV_ERROR, API_DRIVER, 2, HIP_EXPERIMENTAL}},
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  {"cuGetErrorString",                                     {"hipDrvGetErrorString",                                    "", CONV_ERROR, API_DRIVER, 2, HIP_EXPERIMENTAL}},

  // 3. Initialization
  // no analogue
  {"cuInit",                                               {"hipInit",                                                 "", CONV_INIT, API_DRIVER, 3}},

  // 4. Version Management
  // cudaDriverGetVersion
  {"cuDriverGetVersion",                                   {"hipDriverGetVersion",                                     "", CONV_VERSION, API_DRIVER, 4}},

  // 5. Device Management
  // cudaGetDevice
  // NOTE: cudaGetDevice has additional attr: int ordinal
  {"cuDeviceGet",                                          {"hipDeviceGet",                                            "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaDeviceGetAttribute
  {"cuDeviceGetAttribute",                                 {"hipDeviceGetAttribute",                                   "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaGetDeviceCount
  {"cuDeviceGetCount",                                     {"hipGetDeviceCount",                                       "", CONV_DEVICE, API_DRIVER, 5}},
  // no analogue
  {"cuDeviceGetLuid",                                      {"hipDeviceGetLuid",                                        "", CONV_DEVICE, API_DRIVER, 5, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceGetName",                                      {"hipDeviceGetName",                                        "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaDeviceGetNvSciSyncAttributes
  {"cuDeviceGetNvSciSyncAttributes",                       {"hipDeviceGetNvSciSyncAttributes",                         "", CONV_DEVICE, API_DRIVER, 5, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceGetUuid",                                      {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, 5}},
  // no analogue
  {"cuDeviceGetUuid_v2",                                   {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, 5}},
  // no analogue
  {"cuDeviceTotalMem",                                     {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, 5}},
  {"cuDeviceTotalMem_v2",                                  {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaDeviceGetTexture1DLinearMaxWidth
  {"cuDeviceGetTexture1DLinearMaxWidth",                   {"hipDeviceGetTexture1DLinearMaxWidth",                     "", CONV_DEVICE, API_DRIVER, 5, HIP_UNSUPPORTED}},
  // cudaDeviceSetMemPool
  {"cuDeviceSetMemPool",                                   {"hipDeviceSetMemPool",                                     "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaDeviceGetMemPool
  {"cuDeviceGetMemPool",                                   {"hipDeviceGetMemPool",                                     "", CONV_DEVICE, API_DRIVER, 5}},
  // cudaDeviceGetDefaultMemPool
  {"cuDeviceGetDefaultMemPool",                            {"hipDeviceGetDefaultMemPool",                              "", CONV_DEVICE, API_DRIVER, 5}},
  //
  {"cuDeviceGetExecAffinitySupport",                       {"hipDeviceGetExecAffinitySupport",                         "", CONV_DEVICE, API_DRIVER, 5, HIP_UNSUPPORTED}},

  // 6. Device Management [DEPRECATED]
  {"cuDeviceComputeCapability",                            {"hipDeviceComputeCapability",                              "", CONV_DEVICE, API_DRIVER, 6, CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGetDeviceProperties due to different attributes: cudaDeviceProp and CUdevprop
  {"cuDeviceGetProperties",                                {"hipGetDeviceProperties_",                                 "", CONV_DEVICE, API_DRIVER, 6, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 7. Primary Context Management
  // no analogues
  {"cuDevicePrimaryCtxGetState",                           {"hipDevicePrimaryCtxGetState",                             "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxRelease",                            {"hipDevicePrimaryCtxRelease",                              "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxRelease_v2",                         {"hipDevicePrimaryCtxRelease",                              "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxReset",                              {"hipDevicePrimaryCtxReset",                                "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxReset_v2",                           {"hipDevicePrimaryCtxReset",                                "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxRetain",                             {"hipDevicePrimaryCtxRetain",                               "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxSetFlags",                           {"hipDevicePrimaryCtxSetFlags",                             "", CONV_CONTEXT, API_DRIVER, 7}},
  {"cuDevicePrimaryCtxSetFlags_v2",                        {"hipDevicePrimaryCtxSetFlags",                             "", CONV_CONTEXT, API_DRIVER, 7}},

  // 8. Context Management
  // no analogues, except a few
  {"cuCtxCreate",                                          {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxCreate_v2",                                       {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxCreate_v3",                                       {"hipCtxCreate_v3",                                         "", CONV_CONTEXT, API_DRIVER, 8, HIP_UNSUPPORTED}},
  {"cuCtxDestroy",                                         {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxDestroy_v2",                                      {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxGetApiVersion",                                   {"hipCtxGetApiVersion",                                     "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxGetCacheConfig",                                  {"hipCtxGetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxGetCurrent",                                      {"hipCtxGetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxGetDevice",                                       {"hipCtxGetDevice",                                         "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaGetDeviceFlags
  // TODO: rename to hipGetDeviceFlags
  {"cuCtxGetFlags",                                        {"hipCtxGetFlags",                                          "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaDeviceGetLimit
  {"cuCtxGetLimit",                                        {"hipDeviceGetLimit",                                       "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceGetSharedMemConfig
  // TODO: rename to hipDeviceGetSharedMemConfig
  {"cuCtxGetSharedMemConfig",                              {"hipCtxGetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaDeviceGetStreamPriorityRange
  {"cuCtxGetStreamPriorityRange",                          {"hipDeviceGetStreamPriorityRange",                         "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxPopCurrent",                                      {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxPopCurrent_v2",                                   {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxPushCurrent",                                     {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxPushCurrent_v2",                                  {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  {"cuCtxSetCacheConfig",                                  {"hipCtxSetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaCtxResetPersistingL2Cache
  {"cuCtxResetPersistingL2Cache",                          {"hipCtxResetPersistingL2Cache",                            "", CONV_CONTEXT, API_DRIVER, 8, HIP_UNSUPPORTED}},
  {"cuCtxSetCurrent",                                      {"hipCtxSetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaDeviceSetLimit
  {"cuCtxSetLimit",                                        {"hipDeviceSetLimit",                                       "", CONV_CONTEXT, API_DRIVER, 8, HIP_EXPERIMENTAL}},
  // cudaDeviceSetSharedMemConfig
  // TODO: rename to hipDeviceSetSharedMemConfig
  {"cuCtxSetSharedMemConfig",                              {"hipCtxSetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  // cudaDeviceSynchronize
  // TODO: rename to hipDeviceSynchronize
  {"cuCtxSynchronize",                                     {"hipCtxSynchronize",                                       "", CONV_CONTEXT, API_DRIVER, 8, HIP_DEPRECATED}},
  //
  {"cuCtxGetExecAffinity",                                 {"hipCtxGetExecAffinity",                                   "", CONV_CONTEXT, API_DRIVER, 8, HIP_UNSUPPORTED}},

  // 9. Context Management [DEPRECATED]
  // no analogues
  {"cuCtxAttach",                                          {"hipCtxAttach",                                            "", CONV_CONTEXT, API_DRIVER, 9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuCtxDetach",                                          {"hipCtxDetach",                                            "", CONV_CONTEXT, API_DRIVER, 9, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 10. Module Management
  // no analogues
  {"cuLinkAddData",                                        {"hiprtcLinkAddData",                                       "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkAddData_v2",                                     {"hiprtcLinkAddData",                                       "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkAddFile",                                        {"hiprtcLinkAddFile",                                       "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkAddFile_v2",                                     {"hiprtcLinkAddFile",                                       "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkComplete",                                       {"hiprtcLinkComplete",                                      "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkCreate",                                         {"hiprtcLinkCreate",                                        "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkCreate_v2",                                      {"hiprtcLinkCreate",                                        "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuLinkDestroy",                                        {"hiprtcLinkDestroy",                                       "", CONV_MODULE, API_DRIVER, 10, HIP_EXPERIMENTAL}},
  {"cuModuleGetFunction",                                  {"hipModuleGetFunction",                                    "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleGetGlobal",                                    {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleGetGlobal_v2",                                 {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleGetSurfRef",                                   {"hipModuleGetSurfRef",                                     "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuModuleGetTexRef",                                    {"hipModuleGetTexRef",                                      "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleLoad",                                         {"hipModuleLoad",                                           "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleLoadData",                                     {"hipModuleLoadData",                                       "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleLoadDataEx",                                   {"hipModuleLoadDataEx",                                     "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleLoadFatBinary",                                {"hipModuleLoadFatBinary",                                  "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuModuleUnload",                                       {"hipModuleUnload",                                         "", CONV_MODULE, API_DRIVER, 10}},
  {"cuModuleGetLoadingMode",                               {"hipModuleGetLoadingMode",                                 "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},

  // 11. Memory Management
  // no analogue
  {"cuArray3DCreate",                                      {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArray3DCreate_v2",                                   {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArray3DGetDescriptor",                               {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArray3DGetDescriptor_v2",                            {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArrayCreate",                                        {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArrayCreate_v2",                                     {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArrayDestroy",                                       {"hipArrayDestroy",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArrayGetDescriptor",                                 {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor_v2",                              {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  //
  {"cuMipmappedArrayGetMemoryRequirements",                {"hipMipmappedArrayGetMemoryRequirements",                  "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaArrayGetMemoryRequirements
  {"cuArrayGetMemoryRequirements",                         {"hipArrayGetMemoryRequirements",                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaDeviceGetByPCIBusId
  {"cuDeviceGetByPCIBusId",                                {"hipDeviceGetByPCIBusId",                                  "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaDeviceGetPCIBusId
  {"cuDeviceGetPCIBusId",                                  {"hipDeviceGetPCIBusId",                                    "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcCloseMemHandle
  {"cuIpcCloseMemHandle",                                  {"hipIpcCloseMemHandle",                                    "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcGetEventHandle
  {"cuIpcGetEventHandle",                                  {"hipIpcGetEventHandle",                                    "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcGetMemHandle
  {"cuIpcGetMemHandle",                                    {"hipIpcGetMemHandle",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcOpenEventHandle
  {"cuIpcOpenEventHandle",                                 {"hipIpcOpenEventHandle",                                   "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcOpenMemHandle
  {"cuIpcOpenMemHandle",                                   {"hipIpcOpenMemHandle",                                     "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaMalloc
  {"cuMemAlloc",                                           {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemAlloc_v2",                                        {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, 11}},
  //
  {"cuMemAllocHost",                                       {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemAllocHost_v2",                                    {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaMallocManaged
  {"cuMemAllocManaged",                                    {"hipMallocManaged",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMallocPitch due to different signatures
  {"cuMemAllocPitch",                                      {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemAllocPitch_v2",                                   {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy due to different signatures
  {"cuMemcpy",                                             {"hipMemcpy_",                                              "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2D due to different signatures
  {"cuMemcpy2D",                                           {"hipMemcpyParam2D",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpy2D_v2",                                        {"hipMemcpyParam2D",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2DAsync/hipMemcpy2DAsync due to different signatures
  {"cuMemcpy2DAsync",                                      {"hipMemcpyParam2DAsync",                                   "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpy2DAsync_v2",                                   {"hipMemcpyParam2DAsync",                                   "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpy2DUnaligned",                                  {"hipDrvMemcpy2DUnaligned",                                 "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpy2DUnaligned_v2",                               {"hipDrvMemcpy2DUnaligned",                                 "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3D due to different signatures
  {"cuMemcpy3D",                                           {"hipDrvMemcpy3D",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpy3D_v2",                                        {"hipDrvMemcpy3D",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DAsync due to different signatures
  {"cuMemcpy3DAsync",                                      {"hipDrvMemcpy3DAsync",                                     "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpy3DAsync_v2",                                   {"hipDrvMemcpy3DAsync",                                     "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeer due to different signatures
  {"cuMemcpy3DPeer",                                       {"hipMemcpy3DPeer_",                                        "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeerAsync due to different signatures
  {"cuMemcpy3DPeerAsync",                                  {"hipMemcpy3DPeerAsync_",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyAsync due to different signatures
  {"cuMemcpyAsync",                                        {"hipMemcpyAsync_",                                         "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyArrayToArray due to different signatures
  {"cuMemcpyAtoA",                                         {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoA_v2",                                      {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoD",                                         {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoD_v2",                                      {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoH",                                         {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyAtoH_v2",                                      {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyAtoHAsync",                                    {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoHAsync_v2",                                 {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoA",                                         {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpyDtoA_v2",                                      {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoD",                                         {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyDtoD_v2",                                      {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyDtoDAsync",                                    {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyDtoDAsync_v2",                                 {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyDtoH",                                         {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyDtoH_v2",                                      {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyDtoHAsync",                                    {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyDtoHAsync_v2",                                 {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyHtoA",                                         {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyHtoA_v2",                                      {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyHtoAAsync",                                    {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpyHtoAAsync_v2",                                 {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyHtoD",                                         {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyHtoD_v2",                                      {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemcpyHtoDAsync",                                    {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemcpyHtoDAsync_v2",                                 {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeer due to different signatures
  {"cuMemcpyPeer",                                         {"hipMemcpyPeer_",                                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeerAsync due to different signatures
  {"cuMemcpyPeerAsync",                                    {"hipMemcpyPeerAsync_",                                     "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaFree
  {"cuMemFree",                                            {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemFree_v2",                                         {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaFreeHost
  {"cuMemFreeHost",                                        {"hipHostFree",                                             "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemGetAddressRange",                                 {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemGetAddressRange_v2",                              {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaMemGetInfo
  {"cuMemGetInfo",                                         {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemGetInfo_v2",                                      {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostAlloc
  {"cuMemHostAlloc",                                       {"hipHostAlloc",                                            "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostGetDevicePointer
  {"cuMemHostGetDevicePointer",                            {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemHostGetDevicePointer_v2",                         {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostGetFlags
  {"cuMemHostGetFlags",                                    {"hipHostGetFlags",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostRegister
  {"cuMemHostRegister",                                    {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemHostRegister_v2",                                 {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostUnregister
  {"cuMemHostUnregister",                                  {"hipHostUnregister",                                       "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemsetD16",                                          {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemsetD16_v2",                                       {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemsetD16Async",                                     {"hipMemsetD16Async",                                       "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemsetD2D16",                                        {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemsetD2D16_v2",                                     {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D16Async",                                   {"hipMemsetD2D16Async",                                     "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32",                                        {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemsetD2D32_v2",                                     {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32Async",                                   {"hipMemsetD2D32Async",                                     "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8",                                         {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemsetD2D8_v2",                                      {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8Async",                                    {"hipMemsetD2D8Async",                                      "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaMemset
  {"cuMemsetD32",                                          {"hipMemsetD32",                                            "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemsetD32_v2",                                       {"hipMemsetD32",                                            "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaMemsetAsync
  {"cuMemsetD32Async",                                     {"hipMemsetD32Async",                                       "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemsetD8",                                           {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemsetD8_v2",                                        {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  {"cuMemsetD8Async",                                      {"hipMemsetD8Async",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaMallocMipmappedArray due to different signatures
  {"cuMipmappedArrayCreate",                               {"hipMipmappedArrayCreate",                                 "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaFreeMipmappedArray due to different signatures
  {"cuMipmappedArrayDestroy",                              {"hipMipmappedArrayDestroy",                                "", CONV_MEMORY, API_DRIVER, 11}},
  // no analogue
  // NOTE: Not equal to cudaGetMipmappedArrayLevel due to different signatures
  {"cuMipmappedArrayGetLevel",                             {"hipMipmappedArrayGetLevel",                               "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaArrayGetSparseProperties
  {"cuArrayGetSparseProperties",                           {"hipArrayGetSparseProperties",                             "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaArrayGetPlane
  {"cuArrayGetPlane",                                      {"hipArrayGetPlane",                                        "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  //
  {"cuMemGetHandleForAddressRange",                        {"hipMemGetHandleForAddressRange",                          "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},

  // 12. Virtual Memory Management
  // no analogue
  {"cuMemAddressFree",                                     {"hipMemAddressFree",                                       "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemAddressReserve",                                  {"hipMemAddressReserve",                                    "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemCreate",                                          {"hipMemCreate",                                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemExportToShareableHandle",                         {"hipMemExportToShareableHandle",                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemGetAccess",                                       {"hipMemGetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemGetAllocationGranularity",                        {"hipMemGetAllocationGranularity",                          "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemGetAllocationPropertiesFromHandle",               {"hipMemGetAllocationPropertiesFromHandle",                 "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemImportFromShareableHandle",                       {"hipMemImportFromShareableHandle",                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemMap",                                             {"hipMemMap",                                               "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemRelease",                                         {"hipMemRelease",                                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemRetainAllocationHandle",                          {"hipMemRetainAllocationHandle",                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemSetAccess",                                       {"hipMemSetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemUnmap",                                           {"hipMemUnmap",                                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},
  {"cuMemMapArrayAsync",                                   {"hipMemMapArrayAsync",                                     "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12}},

  // 13. Stream Ordered Memory Allocator
  // cudaFreeAsync
  {"cuMemFreeAsync",                                       {"hipFreeAsync",                                            "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMallocAsync
  {"cuMemAllocAsync",                                      {"hipMallocAsync",                                          "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolTrimTo
  {"cuMemPoolTrimTo",                                      {"hipMemPoolTrimTo",                                        "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolSetAttribute
  {"cuMemPoolSetAttribute",                                {"hipMemPoolSetAttribute",                                  "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolGetAttribute
  {"cuMemPoolGetAttribute",                                {"hipMemPoolGetAttribute",                                  "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolSetAccess
  {"cuMemPoolSetAccess",                                   {"hipMemPoolSetAccess",                                     "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolGetAccess
  {"cuMemPoolGetAccess",                                   {"hipMemPoolGetAccess",                                     "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolCreate
  {"cuMemPoolCreate",                                      {"hipMemPoolCreate",                                        "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolDestroy
  {"cuMemPoolDestroy",                                     {"hipMemPoolDestroy",                                       "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMallocFromPoolAsync
  {"cuMemAllocFromPoolAsync",                              {"hipMallocFromPoolAsync",                                  "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolExportToShareableHandle
  {"cuMemPoolExportToShareableHandle",                     {"hipMemPoolExportToShareableHandle",                       "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolImportFromShareableHandle
  {"cuMemPoolImportFromShareableHandle",                   {"hipMemPoolImportFromShareableHandle",                     "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolExportPointer
  {"cuMemPoolExportPointer",                               {"hipMemPoolExportPointer",                                 "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},
  // cudaMemPoolImportPointer
  {"cuMemPoolImportPointer",                               {"hipMemPoolImportPointer",                                 "", CONV_STREAM_ORDERED_MEMORY, API_DRIVER, 13}},

  // 14. Unified Addressing
  // cudaMemAdvise
  {"cuMemAdvise",                                          {"hipMemAdvise",                                            "", CONV_ADDRESSING, API_DRIVER, 14}},
  // TODO: double check cudaMemPrefetchAsync
  {"cuMemPrefetchAsync",                                   {"hipMemPrefetchAsync",                                     "", CONV_ADDRESSING, API_DRIVER, 14}},
  // cudaMemRangeGetAttribute
  {"cuMemRangeGetAttribute",                               {"hipMemRangeGetAttribute",                                 "", CONV_ADDRESSING, API_DRIVER, 14}},
  // cudaMemRangeGetAttributes
  {"cuMemRangeGetAttributes",                              {"hipMemRangeGetAttributes",                                "", CONV_ADDRESSING, API_DRIVER, 14}},
  // no analogue
  {"cuPointerGetAttribute",                                {"hipPointerGetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, 14}},
  // no analogue
  // NOTE: Not equal to cudaPointerGetAttributes due to different signatures
  {"cuPointerGetAttributes",                               {"hipDrvPointerGetAttributes",                              "", CONV_ADDRESSING, API_DRIVER, 14}},
  // no analogue
  {"cuPointerSetAttribute",                                {"hipPointerSetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, 14, HIP_UNSUPPORTED}},

  // 15. Stream Management
  // cudaStreamAddCallback
  {"cuStreamAddCallback",                                  {"hipStreamAddCallback",                                    "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamAttachMemAsync
  {"cuStreamAttachMemAsync",                               {"hipStreamAttachMemAsync",                                 "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamBeginCapture
  {"cuStreamBeginCapture",                                 {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, 15}},
  {"cuStreamBeginCapture_v2",                              {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, 15}},
  {"cuStreamBeginCapture_ptsz",                            {"hipStreamBeginCapture_ptsz",                              "", CONV_STREAM, API_DRIVER, 15, HIP_UNSUPPORTED}},
  // cudaStreamCopyAttributes
  {"cuStreamCopyAttributes",                               {"hipStreamCopyAttributes",                                 "", CONV_STREAM, API_DRIVER, 15, HIP_UNSUPPORTED}},
  // cudaStreamCreateWithFlags
  {"cuStreamCreate",                                       {"hipStreamCreateWithFlags",                                "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamCreateWithPriority
  {"cuStreamCreateWithPriority",                           {"hipStreamCreateWithPriority",                             "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamDestroy
  {"cuStreamDestroy",                                      {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, 15}},
  {"cuStreamDestroy_v2",                                   {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamEndCapture
  {"cuStreamEndCapture",                                   {"hipStreamEndCapture",                                     "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamGetAttribute
  {"cuStreamGetAttribute",                                 {"hipStreamGetAttribute",                                   "", CONV_STREAM, API_DRIVER, 15, HIP_UNSUPPORTED}},
  // cudaStreamGetCaptureInfo
  {"cuStreamGetCaptureInfo",                               {"hipStreamGetCaptureInfo",                                 "", CONV_STREAM, API_DRIVER, 15}},
  {"cuStreamGetCaptureInfo_v2",                            {"hipStreamGetCaptureInfo_v2",                              "", CONV_STREAM, API_DRIVER, 15}},
  //
  {"cuStreamUpdateCaptureDependencies",                    {"hipStreamUpdateCaptureDependencies",                      "", CONV_STREAM, API_DRIVER, 15}},
  // no analogue
  {"cuStreamGetCtx",                                       {"hipStreamGetContext",                                     "", CONV_STREAM, API_DRIVER, 15, HIP_UNSUPPORTED}},
  // cudaStreamGetFlags
  {"cuStreamGetFlags",                                     {"hipStreamGetFlags",                                       "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamGetPriority
  {"cuStreamGetPriority",                                  {"hipStreamGetPriority",                                    "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamIsCapturing
  {"cuStreamIsCapturing",                                  {"hipStreamIsCapturing",                                    "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamQuery
  {"cuStreamQuery",                                        {"hipStreamQuery",                                          "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamSetAttribute
  {"cuStreamSetAttribute",                                 {"hipStreamSetAttribute",                                   "", CONV_STREAM, API_DRIVER, 15, HIP_UNSUPPORTED}},
  // cudaStreamSynchronize
  {"cuStreamSynchronize",                                  {"hipStreamSynchronize",                                    "", CONV_STREAM, API_DRIVER, 15}},
  // cudaStreamWaitEvent
  {"cuStreamWaitEvent",                                    {"hipStreamWaitEvent",                                      "", CONV_STREAM, API_DRIVER, 15}},
  // cudaThreadExchangeStreamCaptureMode
  {"cuThreadExchangeStreamCaptureMode",                    {"hipThreadExchangeStreamCaptureMode",                      "", CONV_STREAM, API_DRIVER, 15}},

  // 16. Event Management
  // cudaEventCreateWithFlags
  {"cuEventCreate",                                        {"hipEventCreateWithFlags",                                 "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventDestroy
  {"cuEventDestroy",                                       {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, 16}},
  {"cuEventDestroy_v2",                                    {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventElapsedTime
  {"cuEventElapsedTime",                                   {"hipEventElapsedTime",                                     "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventQuery
  {"cuEventQuery",                                         {"hipEventQuery",                                           "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventRecord
  {"cuEventRecord",                                        {"hipEventRecord",                                          "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventSynchronize
  {"cuEventSynchronize",                                   {"hipEventSynchronize",                                     "", CONV_EVENT, API_DRIVER, 16}},
  // cudaEventRecordWithFlags
  {"cuEventRecordWithFlags",                               {"hipEventRecordWithFlags",                                 "", CONV_EVENT, API_DRIVER, 16, HIP_UNSUPPORTED}},

  // 17. External Resource Interoperability
  // cudaDestroyExternalMemory
  {"cuDestroyExternalMemory",                              {"hipDestroyExternalMemory",                                "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaDestroyExternalSemaphore
  {"cuDestroyExternalSemaphore",                           {"hipDestroyExternalSemaphore",                             "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaExternalMemoryGetMappedBuffer
  {"cuExternalMemoryGetMappedBuffer",                      {"hipExternalMemoryGetMappedBuffer",                        "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaExternalMemoryGetMappedMipmappedArray
  {"cuExternalMemoryGetMappedMipmappedArray",              {"hipExternalMemoryGetMappedMipmappedArray",                "", CONV_EXT_RES, API_DRIVER, 17, HIP_UNSUPPORTED}},
  // cudaImportExternalMemory
  {"cuImportExternalMemory",                               {"hipImportExternalMemory",                                 "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaImportExternalSemaphore
  {"cuImportExternalSemaphore",                            {"hipImportExternalSemaphore",                              "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaSignalExternalSemaphoresAsync
  {"cuSignalExternalSemaphoresAsync",                      {"hipSignalExternalSemaphoresAsync",                        "", CONV_EXT_RES, API_DRIVER, 17}},
  // cudaWaitExternalSemaphoresAsync
  {"cuWaitExternalSemaphoresAsync",                        {"hipWaitExternalSemaphoresAsync",                          "", CONV_EXT_RES, API_DRIVER, 17}},

  // 18. Stream Memory Operations
  // no analogues
  {"cuStreamBatchMemOp",                                   {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, 18, HIP_UNSUPPORTED}},
  {"cuStreamBatchMemOp_v2",                                {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, 18, HIP_UNSUPPORTED}},

  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  {"cuStreamWaitValue32",                                  {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  {"cuStreamWaitValue32_v2",                               {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  {"cuStreamWaitValue64",                                  {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  {"cuStreamWaitValue64_v2",                               {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  {"cuStreamWriteValue32",                                 {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, int32_t value, unsigned int flags);
  {"cuStreamWriteValue32_v2",                              {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  {"cuStreamWriteValue64",                                 {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 18}},
  // CUresult CUDAAPI cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, int64_t value, unsigned int flags);
  {"cuStreamWriteValue64_v2",                              {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 18}},

  // 19. Execution Control
  // no analogue
  {"cuFuncGetAttribute",                                   {"hipFuncGetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, 19}},
  // no analogue
  {"cuFuncGetModule",                                      {"hipFuncGetModule",                                        "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetAttribute due to different signatures
  {"cuFuncSetAttribute",                                   {"hipFuncSetAttribute_",                                    "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  {"cuFuncSetCacheConfig",                                 {"hipFuncSetCacheConfig_",                                  "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetSharedMemConfig due to different signatures
  {"cuFuncSetSharedMemConfig",                             {"hipFuncSetSharedMemConfig_",                              "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernel due to different signatures
  {"cuLaunchCooperativeKernel",                            {"hipLaunchCooperativeKernel_",                             "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernelMultiDevice due to different signatures
  {"cuLaunchCooperativeKernelMultiDevice",                 {"hipLaunchCooperativeKernelMultiDevice_",                  "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaLaunchHostFunc
  {"cuLaunchHostFunc",                                     {"hipLaunchHostFunc",                                       "", CONV_EXECUTION, API_DRIVER, 19}},
  // no analogue
  // NOTE: Not equal to cudaLaunchKernel due to different signatures
  {"cuLaunchKernel",                                       {"hipModuleLaunchKernel",                                   "", CONV_EXECUTION, API_DRIVER, 19}},

  // 20. Execution Control [DEPRECATED]
  // no analogue
  {"cuFuncSetBlockShape",                                  {"hipFuncSetBlockShape",                                    "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuFuncSetSharedSize",                                  {"hipFuncSetSharedSize",                                    "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  {"cuLaunch",                                             {"hipLaunch",                                               "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuLaunchGrid",                                         {"hipLaunchGrid",                                           "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuLaunchGridAsync",                                    {"hipLaunchGridAsync",                                      "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetf",                                          {"hipParamSetf",                                            "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSeti",                                          {"hipParamSeti",                                            "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetSize",                                       {"hipParamSetSize",                                         "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetTexRef",                                     {"hipParamSetTexRef",                                       "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuParamSetv",                                          {"hipParamSetv",                                            "", CONV_EXECUTION, API_DRIVER, 20, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 21. Graph Management
  // cudaGraphAddChildGraphNode
  {"cuGraphAddChildGraphNode",                             {"hipGraphAddChildGraphNode",                               "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddDependencies
  {"cuGraphAddDependencies",                               {"hipGraphAddDependencies",                                 "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddEmptyNode
  {"cuGraphAddEmptyNode",                                  {"hipGraphAddEmptyNode",                                    "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddHostNode
  {"cuGraphAddHostNode",                                   {"hipGraphAddHostNode",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddKernelNode
  {"cuGraphAddKernelNode",                                 {"hipGraphAddKernelNode",                                   "", CONV_GRAPH, API_DRIVER, 21}},
  // no analogue
  // NOTE: Not equal to cudaGraphAddMemcpyNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  {"cuGraphAddMemcpyNode",                                 {"hipGraphAddMemcpyNode",                                   "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGraphAddMemsetNode due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  {"cuGraphAddMemsetNode",                                 {"hipGraphAddMemsetNode",                                   "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphChildGraphNodeGetGraph
  {"cuGraphChildGraphNodeGetGraph",                        {"hipGraphChildGraphNodeGetGraph",                          "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphClone
  {"cuGraphClone",                                         {"hipGraphClone",                                           "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphCreate
  {"cuGraphCreate",                                        {"hipGraphCreate",                                          "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphDebugDotPrint
  {"cuGraphDebugDotPrint",                                 {"hipGraphDebugDotPrint",                                   "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphDestroy
  {"cuGraphDestroy",                                       {"hipGraphDestroy",                                         "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphDestroyNode
  {"cuGraphDestroyNode",                                   {"hipGraphDestroyNode",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecDestroy
  {"cuGraphExecDestroy",                                   {"hipGraphExecDestroy",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphGetEdges
  {"cuGraphGetEdges",                                      {"hipGraphGetEdges",                                        "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphGetNodes
  {"cuGraphGetNodes",                                      {"hipGraphGetNodes",                                        "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphGetRootNodes
  {"cuGraphGetRootNodes",                                  {"hipGraphGetRootNodes",                                    "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphHostNodeGetParams
  {"cuGraphHostNodeGetParams",                             {"hipGraphHostNodeGetParams",                               "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphHostNodeSetParams
  {"cuGraphHostNodeSetParams",                             {"hipGraphHostNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphInstantiate
  {"cuGraphInstantiate",                                   {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  {"cuGraphInstantiate_v2",                                {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphKernelNodeCopyAttributes
  {"cuGraphKernelNodeCopyAttributes",                      {"hipGraphKernelNodeCopyAttributes",                        "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeGetAttribute
  {"cuGraphKernelNodeGetAttribute",                        {"hipGraphKernelNodeGetAttribute",                          "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecKernelNodeSetParams
  {"cuGraphExecKernelNodeSetParams",                       {"hipGraphExecKernelNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphKernelNodeGetParams
  {"cuGraphKernelNodeGetParams",                           {"hipGraphKernelNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphKernelNodeSetAttribute
  {"cuGraphKernelNodeSetAttribute",                        {"hipGraphKernelNodeSetAttribute",                          "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphKernelNodeSetParams
  {"cuGraphKernelNodeSetParams",                           {"hipGraphKernelNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphLaunch
  {"cuGraphLaunch",                                        {"hipGraphLaunch",                                          "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphMemcpyNodeGetParams
  {"cuGraphMemcpyNodeGetParams",                           {"hipGraphMemcpyNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphMemcpyNodeSetParams
  {"cuGraphMemcpyNodeSetParams",                           {"hipGraphMemcpyNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphMemsetNodeGetParams
  {"cuGraphMemsetNodeGetParams",                           {"hipGraphMemsetNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphMemsetNodeSetParams
  {"cuGraphMemsetNodeSetParams",                           {"hipGraphMemsetNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphNodeFindInClone
  {"cuGraphNodeFindInClone",                               {"hipGraphNodeFindInClone",                                 "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphNodeGetDependencies
  {"cuGraphNodeGetDependencies",                           {"hipGraphNodeGetDependencies",                             "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphNodeGetDependentNodes
  {"cuGraphNodeGetDependentNodes",                         {"hipGraphNodeGetDependentNodes",                           "", CONV_GRAPH, API_DRIVER, 21}},
  //
  {"cuGraphNodeGetEnabled",                                {"hipGraphNodeGetEnabled",                                  "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetType
  {"cuGraphNodeGetType",                                   {"hipGraphNodeGetType",                                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphNodeSetEnabled
  {"cuGraphNodeSetEnabled",                                {"hipGraphNodeSetEnabled",                                  "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphRemoveDependencies
  {"cuGraphRemoveDependencies",                            {"hipGraphRemoveDependencies",                              "", CONV_GRAPH, API_DRIVER, 21}},
  // no analogue
  // NOTE: Not equal to cudaGraphExecMemcpyNodeSetParams due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
  {"cuGraphExecMemcpyNodeSetParams",                       {"hipGraphExecMemcpyNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGraphExecMemcpyNodeSetParams due to different signatures:
  // DRIVER: CUresult CUDAAPI cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx);
  // RUNTIME: cudaError_t CUDARTAPI cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
  {"cuGraphExecMemsetNodeSetParams",                       {"hipGraphExecMemsetNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExecHostNodeSetParams
  {"cuGraphExecHostNodeSetParams",                         {"hipGraphExecHostNodeSetParams",                           "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecUpdate
  {"cuGraphExecUpdate",                                    {"hipGraphExecUpdate",                                      "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddEventRecordNode
  {"cuGraphAddEventRecordNode",                            {"hipGraphAddEventRecordNode",                              "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphEventRecordNodeGetEvent
  {"cuGraphEventRecordNodeGetEvent",                       {"hipGraphEventRecordNodeGetEvent",                         "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphEventRecordNodeSetEvent
  {"cuGraphEventRecordNodeSetEvent",                       {"hipGraphEventRecordNodeSetEvent",                         "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphAddEventWaitNode
  {"cuGraphAddEventWaitNode",                              {"hipGraphAddEventWaitNode",                                "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphEventWaitNodeGetEvent
  {"cuGraphEventWaitNodeGetEvent",                         {"hipGraphEventWaitNodeGetEvent",                           "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphEventWaitNodeSetEvent
  {"cuGraphEventWaitNodeSetEvent",                         {"hipGraphEventWaitNodeSetEvent",                           "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecChildGraphNodeSetParams
  {"cuGraphExecChildGraphNodeSetParams",                   {"hipGraphExecChildGraphNodeSetParams",                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecEventRecordNodeSetEvent
  {"cuGraphExecEventRecordNodeSetEvent",                   {"hipGraphExecEventRecordNodeSetEvent",                     "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphExecEventWaitNodeSetEvent
  {"cuGraphExecEventWaitNodeSetEvent",                     {"hipGraphExecEventWaitNodeSetEvent",                       "", CONV_GRAPH, API_DRIVER, 21}},
  // cudaGraphUpload
  {"cuGraphUpload",                                        {"hipGraphUpload",                                          "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaGraphAddExternalSemaphoresSignalNode
  {"cuGraphAddExternalSemaphoresSignalNode",               {"hipGraphAddExternalSemaphoresSignalNode",                 "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExternalSemaphoresSignalNodeGetParams
  {"cuGraphExternalSemaphoresSignalNodeGetParams",         {"hipGraphExternalSemaphoresSignalNodeGetParams",           "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExternalSemaphoresSignalNodeSetParams
  {"cuGraphExternalSemaphoresSignalNodeSetParams",         {"hipGraphExternalSemaphoresSignalNodeSetParams",           "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphAddExternalSemaphoresWaitNode
  {"cuGraphAddExternalSemaphoresWaitNode",                 {"hipGraphAddExternalSemaphoresWaitNode",                   "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExternalSemaphoresWaitNodeGetParams
  {"cuGraphExternalSemaphoresWaitNodeGetParams",           {"hipGraphExternalSemaphoresWaitNodeGetParams",             "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExternalSemaphoresWaitNodeSetParams
  {"cuGraphExternalSemaphoresWaitNodeSetParams",           {"hipGraphExternalSemaphoresWaitNodeSetParams",             "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExecExternalSemaphoresSignalNodeSetParams
  {"cuGraphExecExternalSemaphoresSignalNodeSetParams",     {"hipGraphExecExternalSemaphoresSignalNodeSetParams",       "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphExecExternalSemaphoresWaitNodeSetParams
  {"cuGraphExecExternalSemaphoresWaitNodeSetParams",       {"hipGraphExecExternalSemaphoresWaitNodeSetParams",         "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaUserObjectCreate
  {"cuUserObjectCreate",                                   {"hipUserObjectCreate",                                     "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaUserObjectRetain
  {"cuUserObjectRetain",                                   {"hipUserObjectRetain",                                     "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaUserObjectRelease
  {"cuUserObjectRelease",                                  {"hipUserObjectRelease",                                    "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaGraphRetainUserObject
  {"cuGraphRetainUserObject",                              {"hipGraphRetainUserObject",                                "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaGraphReleaseUserObject
  {"cuGraphReleaseUserObject",                             {"hipGraphReleaseUserObject",                               "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaGraphAddMemAllocNode
  {"cuGraphAddMemAllocNode",                               {"hipGraphAddMemAllocNode",                                 "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphMemAllocNodeGetParams
  {"cuGraphMemAllocNodeGetParams",                         {"hipGraphMemAllocNodeGetParams",                           "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphAddMemFreeNode
  {"cuGraphAddMemFreeNode",                                {"hipGraphAddMemFreeNode",                                  "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaGraphMemFreeNodeGetParams
  {"cuGraphMemFreeNodeGetParams",                          {"hipGraphMemFreeNodeGetParams",                            "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaDeviceGraphMemTrim
  {"cuDeviceGraphMemTrim",                                 {"hipDeviceGraphMemTrim",                                   "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaDeviceGetGraphMemAttribute
  {"cuDeviceGetGraphMemAttribute",                         {"hipDeviceGetGraphMemAttribute",                           "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaDeviceSetGraphMemAttribute
  {"cuDeviceSetGraphMemAttribute",                         {"hipDeviceSetGraphMemAttribute",                           "", CONV_GRAPH, API_DRIVER, 21, HIP_EXPERIMENTAL}},
  // cudaGraphInstantiateWithFlags
  {"cuGraphInstantiateWithFlags",                          {"hipGraphInstantiateWithFlags",                            "", CONV_GRAPH, API_DRIVER, 21}},
  //
  {"cuGraphAddBatchMemOpNode",                             {"hipGraphAddBatchMemOpNode",                               "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  //
  {"cuGraphBatchMemOpNodeGetParams",                       {"hipGraphBatchMemOpNodeGetParams",                         "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  //
  {"cuGraphBatchMemOpNodeSetParams",                       {"hipGraphBatchMemOpNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},
  //
  {"cuGraphExecBatchMemOpNodeSetParams",                   {"hipGraphExecBatchMemOpNodeSetParams",                     "", CONV_GRAPH, API_DRIVER, 21, HIP_UNSUPPORTED}},

  // 22. Occupancy
  // cudaOccupancyAvailableDynamicSMemPerBlock
  {"cuOccupancyAvailableDynamicSMemPerBlock",              {"hipModuleOccupancyAvailableDynamicSMemPerBlock",          "", CONV_OCCUPANCY, API_DRIVER, 22, HIP_UNSUPPORTED}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",      "", CONV_OCCUPANCY, API_DRIVER, 22}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags","", CONV_OCCUPANCY, API_DRIVER, 22}},
  // cudaOccupancyMaxPotentialBlockSize
  {"cuOccupancyMaxPotentialBlockSize",                     {"hipModuleOccupancyMaxPotentialBlockSize",                 "", CONV_OCCUPANCY, API_DRIVER, 22}},
  // cudaOccupancyMaxPotentialBlockSizeWithFlags
  {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipModuleOccupancyMaxPotentialBlockSizeWithFlags",        "", CONV_OCCUPANCY, API_DRIVER, 22}},

  // 23. Texture Reference Management [DEPRECATED]
  // no analogues
  {"cuTexRefGetAddress",                                   {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetAddress_v2",                                {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetAddressMode",                               {"hipTexRefGetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetArray",                                     {"hipTexRefGetArray",                                       "", CONV_TEXTURE, API_DRIVER, 23, CUDA_DEPRECATED | HIP_REMOVED}},
  {"cuTexRefGetBorderColor",                               {"hipTexRefGetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, 23, CUDA_DEPRECATED | HIP_UNSUPPORTED}},
  {"cuTexRefGetFilterMode",                                {"hipTexRefGetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetFlags",                                     {"hipTexRefGetFlags",                                       "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetFormat",                                    {"hipTexRefGetFormat",                                      "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetMaxAnisotropy",                             {"hipTexRefGetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetMipmapFilterMode",                          {"hipTexRefGetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetMipmapLevelBias",                           {"hipTexRefGetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefGetMipmapLevelClamp",                          {"hipTexRefGetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  // TODO: [HIP] fix typo hipTexRefGetMipMappedArray -> hipTexRefGetMipmappedArray
  {"cuTexRefGetMipmappedArray",                            {"hipTexRefGetMipMappedArray",                              "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetAddress",                                   {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetAddress_v2",                                {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetAddress2D",                                 {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetAddress2D_v2",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 23, HIP_DEPRECATED}},
  {"cuTexRefSetAddress2D_v3",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 23, HIP_DEPRECATED}},
  {"cuTexRefSetAddressMode",                               {"hipTexRefSetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetArray",                                     {"hipTexRefSetArray",                                       "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetBorderColor",                               {"hipTexRefSetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetFilterMode",                                {"hipTexRefSetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetFlags",                                     {"hipTexRefSetFlags",                                       "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetFormat",                                    {"hipTexRefSetFormat",                                      "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetMaxAnisotropy",                             {"hipTexRefSetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetMipmapFilterMode",                          {"hipTexRefSetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetMipmapLevelBias",                           {"hipTexRefSetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetMipmapLevelClamp",                          {"hipTexRefSetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefSetMipmappedArray",                            {"hipTexRefSetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, 23, DEPRECATED}},
  {"cuTexRefCreate",                                       {"hipTexRefCreate",                                         "", CONV_TEXTURE, API_DRIVER, 23, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuTexRefDestroy",                                      {"hipTexRefDestroy",                                        "", CONV_TEXTURE, API_DRIVER, 23, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 24. Surface Reference Management [DEPRECATED]
  // no analogues
  {"cuSurfRefGetArray",                                    {"hipSurfRefGetArray",                                      "", CONV_SURFACE, API_DRIVER, 24, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"cuSurfRefSetArray",                                    {"hipSurfRefSetArray",                                      "", CONV_SURFACE, API_DRIVER, 24, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 25. Texture Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateTextureObject due to different signatures
  {"cuTexObjectCreate",                                    {"hipTexObjectCreate",                                      "", CONV_TEXTURE, API_DRIVER, 25}},
  // cudaDestroyTextureObject
  {"cuTexObjectDestroy",                                   {"hipTexObjectDestroy",                                     "", CONV_TEXTURE, API_DRIVER, 25}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectResourceDesc due to different signatures
  {"cuTexObjectGetResourceDesc",                           {"hipTexObjectGetResourceDesc",                             "", CONV_TEXTURE, API_DRIVER, 25}},
  // cudaGetTextureObjectResourceViewDesc
  {"cuTexObjectGetResourceViewDesc",                       {"hipTexObjectGetResourceViewDesc",                         "", CONV_TEXTURE, API_DRIVER, 25}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                            {"hipTexObjectGetTextureDesc",                              "", CONV_TEXTURE, API_DRIVER, 25}},

  // 26. Surface Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateSurfaceObject due to different signatures
  {"cuSurfObjectCreate",                                   {"hipSurfObjectCreate",                                     "", CONV_TEXTURE, API_DRIVER, 26, HIP_UNSUPPORTED}},
  // cudaDestroySurfaceObject
  {"cuSurfObjectDestroy",                                  {"hipSurfObjectDestroy",                                    "", CONV_TEXTURE, API_DRIVER, 26, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetSurfaceObjectResourceDesc due to different signatures
  {"cuSurfObjectGetResourceDesc",                          {"hipSurfObjectGetResourceDesc",                            "", CONV_TEXTURE, API_DRIVER, 26, HIP_UNSUPPORTED}},

  // 27. Peer Context Memory Access
  // no analogue
  // NOTE: Not equal to cudaDeviceEnablePeerAccess due to different signatures
  {"cuCtxEnablePeerAccess",                                {"hipCtxEnablePeerAccess",                                  "", CONV_PEER, API_DRIVER, 27, HIP_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaDeviceDisablePeerAccess due to different signatures
  {"cuCtxDisablePeerAccess",                               {"hipCtxDisablePeerAccess",                                 "", CONV_PEER, API_DRIVER, 27, HIP_DEPRECATED}},
  // cudaDeviceCanAccessPeer
  {"cuDeviceCanAccessPeer",                                {"hipDeviceCanAccessPeer",                                  "", CONV_PEER, API_DRIVER, 27}},
  // cudaDeviceGetP2PAttribute
  {"cuDeviceGetP2PAttribute",                              {"hipDeviceGetP2PAttribute",                                "", CONV_PEER, API_DRIVER, 27}},

  // 28. Graphics Interoperability
  // cudaGraphicsMapResources
  {"cuGraphicsMapResources",                               {"hipGraphicsMapResources",                                 "", CONV_GRAPHICS, API_DRIVER, 28}},
  // cudaGraphicsResourceGetMappedMipmappedArray
  {"cuGraphicsResourceGetMappedMipmappedArray",            {"hipGraphicsResourceGetMappedMipmappedArray",              "", CONV_GRAPHICS, API_DRIVER, 28, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer",                   {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, 28}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer_v2",                {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, 28}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags",                        {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, 28, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags_v2",                     {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, 28, HIP_UNSUPPORTED}},
  // cudaGraphicsSubResourceGetMappedArray
  {"cuGraphicsSubResourceGetMappedArray",                  {"hipGraphicsSubResourceGetMappedArray",                    "", CONV_GRAPHICS, API_DRIVER, 28}},
  // cudaGraphicsUnmapResources
  {"cuGraphicsUnmapResources",                             {"hipGraphicsUnmapResources",                               "", CONV_GRAPHICS, API_DRIVER, 28}},
  // cudaGraphicsUnregisterResource
  {"cuGraphicsUnregisterResource",                         {"hipGraphicsUnregisterResource",                           "", CONV_GRAPHICS, API_DRIVER, 28}},

  // 29. Driver Entry Point Access
  // cudaGetDriverEntryPoint
  {"cuGetProcAddress",                                     {"hipGetProcAddress",                                       "", CONV_PROFILER, API_DRIVER, 29, HIP_UNSUPPORTED}},
  // cudaDeviceFlushGPUDirectRDMAWrites
  {"cuFlushGPUDirectRDMAWrites",                           {"hipDeviceFlushGPUDirectRDMAWrites",                       "", CONV_PROFILER, API_DRIVER, 29, HIP_UNSUPPORTED}},

  // 30. Profiler Control [DEPRECATED]
  // cudaProfilerInitialize
  {"cuProfilerInitialize",                                 {"hipProfilerInitialize",                                   "", CONV_PROFILER, API_DRIVER, 30, HIP_UNSUPPORTED}},

  // 31. Profiler Control
  // cudaProfilerStart
  {"cuProfilerStart",                                      {"hipProfilerStart",                                        "", CONV_PROFILER, API_DRIVER, 31}},
  // cudaProfilerStop
  {"cuProfilerStop",                                       {"hipProfilerStop",                                         "", CONV_PROFILER, API_DRIVER, 31}},

  // 32. OpenGL Interoperability
  // cudaGLGetDevices
  {"cuGLGetDevices",                                       {"hipGLGetDevices",                                         "", CONV_OPENGL, API_DRIVER, 32}},
  // cudaGraphicsGLRegisterBuffer
  {"cuGraphicsGLRegisterBuffer",                           {"hipGraphicsGLRegisterBuffer",                             "", CONV_OPENGL, API_DRIVER, 32}},
  // cudaGraphicsGLRegisterImage
  {"cuGraphicsGLRegisterImage",                            {"hipGraphicsGLRegisterImage",                              "", CONV_OPENGL, API_DRIVER, 32}},
  // cudaWGLGetDevice
  {"cuWGLGetDevice",                                       {"hipWGLGetDevice",                                         "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED}},

  // 32. OpenGL Interoperability [DEPRECATED]
  // no analogue
  {"cuGLCtxCreate",                                        {"hipGLCtxCreate",                                          "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuGLInit",                                             {"hipGLInit",                                               "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObject due to different signatures
  {"cuGLMapBufferObject",                                  {"hipGLMapBufferObject_",                                   "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObjectAsync due to different signatures
  {"cuGLMapBufferObjectAsync",                             {"hipGLMapBufferObjectAsync_",                              "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLRegisterBufferObject
  {"cuGLRegisterBufferObject",                             {"hipGLRegisterBufferObject",                               "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLSetBufferObjectMapFlags
  {"cuGLSetBufferObjectMapFlags",                          {"hipGLSetBufferObjectMapFlags",                            "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnmapBufferObject
  {"cuGLUnmapBufferObject",                                {"hipGLUnmapBufferObject",                                  "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnmapBufferObjectAsync
  {"cuGLUnmapBufferObjectAsync",                           {"hipGLUnmapBufferObjectAsync",                             "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaGLUnregisterBufferObject
  {"cuGLUnregisterBufferObject",                           {"hipGLUnregisterBufferObject",                             "", CONV_OPENGL, API_DRIVER, 32, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 33. VDPAU Interoperability
  // cudaGraphicsVDPAURegisterOutputSurface
  {"cuGraphicsVDPAURegisterOutputSurface",                 {"hipGraphicsVDPAURegisterOutputSurface",                   "", CONV_VDPAU, API_DRIVER, 33, HIP_UNSUPPORTED}},
  // cudaGraphicsVDPAURegisterVideoSurface
  {"cuGraphicsVDPAURegisterVideoSurface",                  {"hipGraphicsVDPAURegisterVideoSurface",                    "", CONV_VDPAU, API_DRIVER, 33, HIP_UNSUPPORTED}},
  // cudaVDPAUGetDevice
  {"cuVDPAUGetDevice",                                     {"hipVDPAUGetDevice",                                       "", CONV_VDPAU, API_DRIVER, 33, HIP_UNSUPPORTED}},
  // no analogue
  {"cuVDPAUCtxCreate",                                     {"hipVDPAUCtxCreate",                                       "", CONV_VDPAU, API_DRIVER, 33, HIP_UNSUPPORTED}},

  // 34. EGL Interoperability
  // cudaEGLStreamConsumerAcquireFrame
  {"cuEGLStreamConsumerAcquireFrame",                      {"hipEGLStreamConsumerAcquireFrame",                        "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnect
  {"cuEGLStreamConsumerConnect",                           {"hipEGLStreamConsumerConnect",                             "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnectWithFlags
  {"cuEGLStreamConsumerConnectWithFlags",                  {"hipEGLStreamConsumerConnectWithFlags",                    "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerDisconnect
  {"cuEGLStreamConsumerDisconnect",                        {"hipEGLStreamConsumerDisconnect",                          "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerReleaseFrame
  {"cuEGLStreamConsumerReleaseFrame",                      {"hipEGLStreamConsumerReleaseFrame",                        "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerConnect
  {"cuEGLStreamProducerConnect",                           {"hipEGLStreamProducerConnect",                             "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerDisconnect
  {"cuEGLStreamProducerDisconnect",                        {"hipEGLStreamProducerDisconnect",                          "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerPresentFrame
  {"cuEGLStreamProducerPresentFrame",                      {"hipEGLStreamProducerPresentFrame",                        "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerReturnFrame
  {"cuEGLStreamProducerReturnFrame",                       {"hipEGLStreamProducerReturnFrame",                         "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaGraphicsEGLRegisterImage
  {"cuGraphicsEGLRegisterImage",                           {"hipGraphicsEGLRegisterImage",                             "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedEglFrame
  {"cuGraphicsResourceGetMappedEglFrame",                  {"hipGraphicsResourceGetMappedEglFrame",                    "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaEventCreateFromEGLSync
  {"cuEventCreateFromEGLSync",                             {"hipEventCreateFromEGLSync",                               "", CONV_EGL, API_DRIVER, 34, HIP_UNSUPPORTED}},

  // 35. Direct3D 9 Interoperability
  // no analogue
  {"cuD3D9CtxCreate",                                      {"hipD3D9CtxCreate",                                        "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},
    // no analogue
  {"cuD3D9CtxCreateOnDevice",                              {"hipD3D9CtxCreateOnDevice",                                "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevice
  {"cuD3D9GetDevice",                                      {"hipD3D9GetDevice",                                        "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevices
  {"cuD3D9GetDevices",                                     {"hipD3D9GetDevices",                                       "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaD3D9GetDirect3DDevice
  {"cuD3D9GetDirect3DDevice",                              {"hipD3D9GetDirect3DDevice",                                "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D9RegisterResource
  {"cuGraphicsD3D9RegisterResource",                       {"hipGraphicsD3D9RegisterResource",                         "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED}},

  // 35. Direct3D 9 Interoperability [DEPRECATED]
  // cudaD3D9MapResources
  {"cuD3D9MapResources",                                   {"hipD3D9MapResources",                                     "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9RegisterResource
  {"cuD3D9RegisterResource",                               {"hipD3D9RegisterResource",                                 "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedArray
  {"cuD3D9ResourceGetMappedArray",                         {"hipD3D9ResourceGetMappedArray",                           "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedPitch
  {"cuD3D9ResourceGetMappedPitch",                         {"hipD3D9ResourceGetMappedPitch",                           "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedPointer
  {"cuD3D9ResourceGetMappedPointer",                       {"hipD3D9ResourceGetMappedPointer",                         "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetMappedSize
  {"cuD3D9ResourceGetMappedSize",                          {"hipD3D9ResourceGetMappedSize",                            "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceGetSurfaceDimensions
  {"cuD3D9ResourceGetSurfaceDimensions",                   {"hipD3D9ResourceGetSurfaceDimensions",                     "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9ResourceSetMapFlags
  {"cuD3D9ResourceSetMapFlags",                            {"hipD3D9ResourceSetMapFlags",                              "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9UnmapResources
  {"cuD3D9UnmapResources",                                 {"hipD3D9UnmapResources",                                   "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D9UnregisterResource
  {"cuD3D9UnregisterResource",                             {"hipD3D9UnregisterResource",                               "", CONV_D3D9, API_DRIVER, 35, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 36. Direct3D 10 Interoperability
  // cudaD3D10GetDevice
  {"cuD3D10GetDevice",                                     {"hipD3D10GetDevice",                                       "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED}},
  // cudaD3D10GetDevices
  {"cuD3D10GetDevices",                                    {"hipD3D10GetDevices",                                      "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D10RegisterResource
  {"cuGraphicsD3D10RegisterResource",                      {"hipGraphicsD3D10RegisterResource",                        "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED}},

  // 36. Direct3D 10 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D10CtxCreate",                                     {"hipD3D10CtxCreate",                                       "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuD3D10CtxCreateOnDevice",                             {"hipD3D10CtxCreateOnDevice",                               "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10GetDirect3DDevice
  {"cuD3D10GetDirect3DDevice",                             {"hipD3D10GetDirect3DDevice",                               "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10MapResources
  {"cuD3D10MapResources",                                  {"hipD3D10MapResources",                                    "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10RegisterResource
  {"cuD3D10RegisterResource",                              {"hipD3D10RegisterResource",                                "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedArray
  {"cuD3D10ResourceGetMappedArray",                        {"hipD3D10ResourceGetMappedArray",                          "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedPitch
  {"cuD3D10ResourceGetMappedPitch",                        {"hipD3D10ResourceGetMappedPitch",                          "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedPointer
  {"cuD3D10ResourceGetMappedPointer",                      {"hipD3D10ResourceGetMappedPointer",                        "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetMappedSize
  {"cuD3D10ResourceGetMappedSize",                         {"hipD3D10ResourceGetMappedSize",                           "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceGetSurfaceDimensions
  {"cuD3D10ResourceGetSurfaceDimensions",                  {"hipD3D10ResourceGetSurfaceDimensions",                    "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10ResourceSetMapFlags
  {"cuD3D10ResourceSetMapFlags",                           {"hipD3D10ResourceSetMapFlags",                             "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10UnmapResources
  {"cuD3D10UnmapResources",                                {"hipD3D10UnmapResources",                                  "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D10UnregisterResource
  {"cuD3D10UnregisterResource",                            {"hipD3D10UnregisterResource",                              "", CONV_D3D10, API_DRIVER, 36, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 37. Direct3D 11 Interoperability
  // cudaD3D11GetDevice
  {"cuD3D11GetDevice",                                     {"hipD3D11GetDevice",                                       "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED}},
  // cudaD3D11GetDevices
  {"cuD3D11GetDevices",                                    {"hipD3D11GetDevices",                                      "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D11RegisterResource
  {"cuGraphicsD3D11RegisterResource",                      {"hipGraphicsD3D11RegisterResource",                        "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED}},

  // 37. Direct3D 11 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D11CtxCreate",                                     {"hipD3D11CtxCreate",                                       "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cuD3D11CtxCreateOnDevice",                             {"hipD3D11CtxCreateOnDevice",                               "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cudaD3D11GetDirect3DDevice
  {"cuD3D11GetDirect3DDevice",                             {"hipD3D11GetDirect3DDevice",                               "", CONV_D3D11, API_DRIVER, 37, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
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
  {"hipMipmappedArrayCreate",                              {HIP_3050, HIP_0,    HIP_0   }},
  {"hipMipmappedArrayDestroy",                             {HIP_3050, HIP_0,    HIP_0   }},
  {"hipMipmappedArrayGetLevel",                            {HIP_3050, HIP_0,    HIP_0   }},
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
  {"hipDrvGetErrorName",                                   {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipDrvGetErrorString",                                 {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hiprtcLinkCreate",                                     {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hiprtcLinkAddFile",                                    {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hiprtcLinkAddData",                                    {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hiprtcLinkComplete",                                   {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hiprtcLinkDestroy",                                    {HIP_5030, HIP_0,    HIP_0,  HIP_LATEST}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_DRIVER_API_SECTION_MAP {
  {1, "CUDA Driver Data Types"},
  {2, "Error Handling"},
  {3, "Initialization"},
  {4, "Version Management"},
  {5, "Device Management"},
  {6, "Device Management [DEPRECATED]"},
  {7, "Primary Context Management"},
  {8, "Context Management"},
  {9, "Context Management [DEPRECATED]"},
  {10, "Module Management"},
  {11, "Memory Management"},
  {12, "Virtual Memory Management"},
  {13, "Stream Ordered Memory Allocator"},
  {14, "Unified Addressing"},
  {15, "Stream Management"},
  {16, "Event Management"},
  {17, "External Resource Interoperability"},
  {18, "Stream Memory Operations"},
  {19, "Execution Control"},
  {20, "Execution Control [DEPRECATED]"},
  {21, "Graph Management"},
  {22, "Occupancy"},
  {23, "Texture Reference Management [DEPRECATED]"},
  {24, "Surface Reference Management [DEPRECATED]"},
  {25, "Texture Object Management"},
  {26, "Surface Object Management"},
  {27, "Peer Context Memory Access"},
  {28, "Graphics Interoperability"},
  {29, "Driver Entry Point Access"},
  {30, "Profiler Control [DEPRECATED]"},
  {31, "Profiler Control"},
  {32, "OpenGL Interoperability"},
  {33, "VDPAU Interoperability"},
  {34, "EGL Interoperability"},
  {35, "Direct3D 9 Interoperability"},
  {36, "Direct3D 10 Interoperability"},
  {37, "Direct3D 11 Interoperability"},
};
