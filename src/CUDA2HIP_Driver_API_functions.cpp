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
  {"cuGetErrorName",                                       {"hipGetErrorName_",                                        "", CONV_ERROR, API_DRIVER, 2, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  {"cuGetErrorString",                                     {"hipGetErrorString_",                                      "", CONV_ERROR, API_DRIVER, 2, HIP_UNSUPPORTED}},

  // 3. Initialization
  // no analogue
  {"cuInit",                                               {"hipInit",                                                 "", CONV_INIT, API_DRIVER, 3}},

  // 4. Version Management
  // cudaDriverGetVersion
  {"cuDriverGetVersion",                                   {"hipDriverGetVersion",                                     "", CONV_VERSION, API_DRIVER, 4}},

  // 5. Device Management
  // cudaGetDevice
  // NOTE: cudaGetDevice has additional attr: int ordinal
  {"cuDeviceGet",                                          {"hipGetDevice",                                            "", CONV_DEVICE, API_DRIVER, 5}},
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
  {"cuDeviceGetUuid",                                      {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, 5, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceTotalMem",                                     {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, 5}},
  {"cuDeviceTotalMem_v2",                                  {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER, 5}},

  // 6. Device Management [DEPRECATED]
  {"cuDeviceComputeCapability",                            {"hipDeviceComputeCapability",                              "", CONV_DEVICE, API_DRIVER, 6, DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGetDeviceProperties due to different attributes: cudaDeviceProp and CUdevprop
  {"cuDeviceGetProperties",                                {"hipGetDeviceProperties_",                                 "", CONV_DEVICE, API_DRIVER, 6, HIP_UNSUPPORTED | DEPRECATED}},

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
  {"cuCtxCreate",                                          {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxCreate_v2",                                       {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxDestroy",                                         {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxDestroy_v2",                                      {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxGetApiVersion",                                   {"hipCtxGetApiVersion",                                     "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxGetCacheConfig",                                  {"hipCtxGetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxGetCurrent",                                      {"hipCtxGetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxGetDevice",                                       {"hipCtxGetDevice",                                         "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaGetDeviceFlags
  // TODO: rename to hipGetDeviceFlags
  {"cuCtxGetFlags",                                        {"hipCtxGetFlags",                                          "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceGetLimit
  {"cuCtxGetLimit",                                        {"hipDeviceGetLimit",                                       "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceGetSharedMemConfig
  // TODO: rename to hipDeviceGetSharedMemConfig
  {"cuCtxGetSharedMemConfig",                              {"hipCtxGetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceGetStreamPriorityRange
  {"cuCtxGetStreamPriorityRange",                          {"hipDeviceGetStreamPriorityRange",                         "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxPopCurrent",                                      {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxPopCurrent_v2",                                   {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxPushCurrent",                                     {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxPushCurrent_v2",                                  {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER, 8}},
  {"cuCtxSetCacheConfig",                                  {"hipCtxSetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaCtxResetPersistingL2Cache
  {"cuCtxResetPersistingL2Cache",                          {"hipCtxResetPersistingL2Cache",                            "", CONV_CONTEXT, API_DRIVER, 8, HIP_UNSUPPORTED}},
  {"cuCtxSetCurrent",                                      {"hipCtxSetCurrent",                                        "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceSetLimit
  {"cuCtxSetLimit",                                        {"hipDeviceSetLimit",                                       "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceSetSharedMemConfig
  // TODO: rename to hipDeviceSetSharedMemConfig
  {"cuCtxSetSharedMemConfig",                              {"hipCtxSetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER, 8}},
  // cudaDeviceSynchronize
  // TODO: rename to hipDeviceSynchronize
  {"cuCtxSynchronize",                                     {"hipCtxSynchronize",                                       "", CONV_CONTEXT, API_DRIVER, 8}},

  // 9. Context Management [DEPRECATED]
  // no analogues
  {"cuCtxAttach",                                          {"hipCtxAttach",                                            "", CONV_CONTEXT, API_DRIVER, 9, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuCtxDetach",                                          {"hipCtxDetach",                                            "", CONV_CONTEXT, API_DRIVER, 9, HIP_UNSUPPORTED | DEPRECATED}},

  // 10. Module Management
  // no analogues
  {"cuLinkAddData",                                        {"hipLinkAddData",                                          "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkAddData_v2",                                     {"hipLinkAddData",                                          "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkAddFile",                                        {"hipLinkAddFile",                                          "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkAddFile_v2",                                     {"hipLinkAddFile",                                          "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkComplete",                                       {"hipLinkComplete",                                         "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkCreate",                                         {"hipLinkCreate",                                           "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkCreate_v2",                                      {"hipLinkCreate",                                           "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
  {"cuLinkDestroy",                                        {"hipLinkDestroy",                                          "", CONV_MODULE, API_DRIVER, 10, HIP_UNSUPPORTED}},
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

  // 11. Memory Management
  // no analogue
  {"cuArray3DCreate",                                      {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArray3DCreate_v2",                                   {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArray3DGetDescriptor",                               {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArray3DGetDescriptor_v2",                            {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArrayCreate",                                        {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArrayCreate_v2",                                     {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuArrayDestroy",                                       {"hipArrayDestroy",                                         "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor",                                 {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor_v2",                              {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaDeviceGetByPCIBusId
  {"cuDeviceGetByPCIBusId",                                {"hipDeviceGetByPCIBusId",                                  "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaDeviceGetPCIBusId
  {"cuDeviceGetPCIBusId",                                  {"hipDeviceGetPCIBusId",                                    "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcCloseMemHandle
  {"cuIpcCloseMemHandle",                                  {"hipIpcCloseMemHandle",                                    "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcGetEventHandle
  {"cuIpcGetEventHandle",                                  {"hipIpcGetEventHandle",                                    "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaIpcGetMemHandle
  {"cuIpcGetMemHandle",                                    {"hipIpcGetMemHandle",                                      "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaIpcOpenEventHandle
  {"cuIpcOpenEventHandle",                                 {"hipIpcOpenEventHandle",                                   "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // cudaIpcOpenMemHandle
  {"cuIpcOpenMemHandle",                                   {"hipIpcOpenMemHandle",                                     "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaMalloc
  {"cuMemAlloc",                                           {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemAlloc_v2",                                        {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER, 11}},
  // cudaHostAlloc
  {"cuMemAllocHost",                                       {"hipHostMalloc",                                           "", CONV_MEMORY, API_DRIVER, 11}},
  {"cuMemAllocHost_v2",                                    {"hipHostMalloc",                                           "", CONV_MEMORY, API_DRIVER, 11}},
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
  {"cuMemcpy2DUnaligned",                                  {"hipMemcpy2DUnaligned",                                    "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  {"cuMemcpy2DUnaligned_v2",                               {"hipMemcpy2DUnaligned",                                    "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
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
  {"cuMemHostAlloc",                                       {"hipHostMalloc",                                           "", CONV_MEMORY, API_DRIVER, 11}},
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
  {"cuMipmappedArrayCreate",                               {"hipMipmappedArrayCreate",                                 "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFreeMipmappedArray due to different signatures
  {"cuMipmappedArrayDestroy",                              {"hipMipmappedArrayDestroy",                                "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetMipmappedArrayLevel due to different signatures
  {"cuMipmappedArrayGetLevel",                             {"hipMipmappedArrayGetLevel",                               "", CONV_MEMORY, API_DRIVER, 11, HIP_UNSUPPORTED}},

  // 12. Virtual Memory Management
  // no analogue
  {"cuMemAddressFree",                                     {"hipMemAddressFree",                                       "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemAddressReserve",                                  {"hipMemAddressReserve",                                    "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemCreate",                                          {"hipMemCreate",                                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemExportToShareableHandle",                         {"hipMemExportToShareableHandle",                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemGetAccess",                                       {"hipMemGetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemGetAllocationGranularity",                        {"hipMemGetAllocationGranularity",                          "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemGetAllocationPropertiesFromHandle",               {"hipMemGetAllocationPropertiesFromHandle",                 "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemImportFromShareableHandle",                       {"hipMemImportFromShareableHandle",                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemMap",                                             {"hipMemMap",                                               "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemRelease",                                         {"hipMemRelease",                                           "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemRetainAllocationHandle",                          {"hipMemRetainAllocationHandle",                            "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemSetAccess",                                       {"hipMemSetAccess",                                         "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},
  {"cuMemUnmap",                                           {"hipMemUnmap",                                             "", CONV_VIRTUAL_MEMORY, API_DRIVER, 12, HIP_UNSUPPORTED}},

  // 13. Unified Addressing
  // cudaMemAdvise
  {"cuMemAdvise",                                          {"hipMemAdvise",                                            "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // TODO: double check cudaMemPrefetchAsync
  {"cuMemPrefetchAsync",                                   {"hipMemPrefetchAsync_",                                    "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // cudaMemRangeGetAttribute
  {"cuMemRangeGetAttribute",                               {"hipMemRangeGetAttribute",                                 "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // cudaMemRangeGetAttributes
  {"cuMemRangeGetAttributes",                              {"hipMemRangeGetAttributes",                                "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // no analogue
  {"cuPointerGetAttribute",                                {"hipPointerGetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaPointerGetAttributes due to different signatures
  {"cuPointerGetAttributes",                               {"hipPointerGetAttributes_",                                "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},
  // no analogue
  {"cuPointerSetAttribute",                                {"hipPointerSetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, 13, HIP_UNSUPPORTED}},

  // 14. Stream Management
  // cudaStreamAddCallback
  {"cuStreamAddCallback",                                  {"hipStreamAddCallback",                                    "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamAttachMemAsync
  {"cuStreamAttachMemAsync",                               {"hipStreamAttachMemAsync",                                 "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamBeginCapture
  {"cuStreamBeginCapture",                                 {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  {"cuStreamBeginCapture_v2",                              {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  {"cuStreamBeginCapture_ptsz",                            {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamCopyAttributes
  {"cuStreamCopyAttributes",                               {"hipStreamCopyAttributes",                                 "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamCreateWithFlags
  {"cuStreamCreate",                                       {"hipStreamCreateWithFlags",                                "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamCreateWithPriority
  {"cuStreamCreateWithPriority",                           {"hipStreamCreateWithPriority",                             "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamDestroy
  {"cuStreamDestroy",                                      {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, 14}},
  {"cuStreamDestroy_v2",                                   {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamEndCapture
  {"cuStreamEndCapture",                                   {"hipStreamEndCapture",                                     "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamGetAttribute
  {"cuStreamGetAttribute",                                 {"hipStreamGetAttribute",                                   "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamGetCaptureInfo
  {"cuStreamGetCaptureInfo",                               {"hipStreamGetCaptureInfo",                                 "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // no analogue
  {"cuStreamGetCtx",                                       {"hipStreamGetContext",                                     "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamGetFlags
  {"cuStreamGetFlags",                                     {"hipStreamGetFlags",                                       "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamGetPriority
  {"cuStreamGetPriority",                                  {"hipStreamGetPriority",                                    "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamIsCapturing
  {"cuStreamIsCapturing",                                  {"hipStreamIsCapturing",                                    "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamQuery
  {"cuStreamQuery",                                        {"hipStreamQuery",                                          "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamSetAttribute
  {"cuStreamSetAttribute",                                 {"hipStreamSetAttribute",                                   "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},
  // cudaStreamSynchronize
  {"cuStreamSynchronize",                                  {"hipStreamSynchronize",                                    "", CONV_STREAM, API_DRIVER, 14}},
  // cudaStreamWaitEvent
  {"cuStreamWaitEvent",                                    {"hipStreamWaitEvent",                                      "", CONV_STREAM, API_DRIVER, 14}},
  // cudaThreadExchangeStreamCaptureMode
  {"cuThreadExchangeStreamCaptureMode",                    {"hipThreadExchangeStreamCaptureMode",                      "", CONV_STREAM, API_DRIVER, 14, HIP_UNSUPPORTED}},

  // 15. Event Management
  // cudaEventCreateWithFlags
  {"cuEventCreate",                                        {"hipEventCreateWithFlags",                                 "", CONV_EVENT, API_DRIVER, 15}},
  // cudaEventDestroy
  {"cuEventDestroy",                                       {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, 15}},
  {"cuEventDestroy_v2",                                    {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER, 15}},
  // cudaEventElapsedTime
  {"cuEventElapsedTime",                                   {"hipEventElapsedTime",                                     "", CONV_EVENT, API_DRIVER, 15}},
  // cudaEventQuery
  {"cuEventQuery",                                         {"hipEventQuery",                                           "", CONV_EVENT, API_DRIVER, 15}},
  // cudaEventRecord
  {"cuEventRecord",                                        {"hipEventRecord",                                          "", CONV_EVENT, API_DRIVER, 15}},
  // cudaEventSynchronize
  {"cuEventSynchronize",                                   {"hipEventSynchronize",                                     "", CONV_EVENT, API_DRIVER, 15}},

  // 16. External Resource Interoperability
  // cudaDestroyExternalMemory
  {"cuDestroyExternalMemory",                              {"hipDestroyExternalMemory",                                "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaDestroyExternalSemaphore
  {"cuDestroyExternalSemaphore",                           {"hipDestroyExternalSemaphore",                             "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaExternalMemoryGetMappedBuffer
  {"cuExternalMemoryGetMappedBuffer",                      {"hipExternalMemoryGetMappedBuffer",                        "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaExternalMemoryGetMappedMipmappedArray
  {"cuExternalMemoryGetMappedMipmappedArray",              {"hipExternalMemoryGetMappedMipmappedArray",                "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaImportExternalMemory
  {"cuImportExternalMemory",                               {"hipImportExternalMemory",                                 "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaImportExternalSemaphore
  {"cuImportExternalSemaphore",                            {"hipImportExternalSemaphore",                              "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaSignalExternalSemaphoresAsync
  {"cuSignalExternalSemaphoresAsync",                      {"hipSignalExternalSemaphoresAsync",                        "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},
  // cudaWaitExternalSemaphoresAsync
  {"cuWaitExternalSemaphoresAsync",                        {"hipWaitExternalSemaphoresAsync",                          "", CONV_EXT_RES, API_DRIVER, 16, HIP_UNSUPPORTED}},

  // 17. Stream Memory Operations
  // no analogues
  {"cuStreamBatchMemOp",                                   {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, 17, HIP_UNSUPPORTED}},
  {"cuStreamWaitValue32",                                  {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 17, HIP_UNSUPPORTED}},
  {"cuStreamWaitValue64",                                  {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, 17, HIP_UNSUPPORTED}},
  {"cuStreamWriteValue32",                                 {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 17, HIP_UNSUPPORTED}},
  {"cuStreamWriteValue64",                                 {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, 17, HIP_UNSUPPORTED}},

  // 18. Execution Control
  // no analogue
  {"cuFuncGetAttribute",                                   {"hipFuncGetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, 18}},
  // no analogue
  {"cuFuncGetModule",                                      {"hipFuncGetModule",                                        "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetAttribute due to different signatures
  {"cuFuncSetAttribute",                                   {"hipFuncSetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  {"cuFuncSetCacheConfig",                                 {"hipFuncSetCacheConfig",                                   "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetSharedMemConfig due to different signatures
  {"cuFuncSetSharedMemConfig",                             {"hipFuncSetSharedMemConfig",                               "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernel due to different signatures
  {"cuLaunchCooperativeKernel",                            {"hipLaunchCooperativeKernel_",                             "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernelMultiDevice due to different signatures
  {"cuLaunchCooperativeKernelMultiDevice",                 {"hipLaunchCooperativeKernelMultiDevice_",                  "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // cudaLaunchHostFunc
  {"cuLaunchHostFunc",                                     {"hipLaunchHostFunc",                                       "", CONV_EXECUTION, API_DRIVER, 18, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchKernel due to different signatures
  {"cuLaunchKernel",                                       {"hipModuleLaunchKernel",                                   "", CONV_EXECUTION, API_DRIVER, 18}},

  // 19. Execution Control [DEPRECATED]
  // no analogue
  {"cuFuncSetBlockShape",                                  {"hipFuncSetBlockShape",                                    "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuFuncSetSharedSize",                                  {"hipFuncSetSharedSize",                                    "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  {"cuLaunch",                                             {"hipLaunch",                                               "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuLaunchGrid",                                         {"hipLaunchGrid",                                           "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuLaunchGridAsync",                                    {"hipLaunchGridAsync",                                      "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuParamSetf",                                          {"hipParamSetf",                                            "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuParamSeti",                                          {"hipParamSeti",                                            "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuParamSetSize",                                       {"hipParamSetSize",                                         "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuParamSetTexRef",                                     {"hipParamSetTexRef",                                       "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuParamSetv",                                          {"hipParamSetv",                                            "", CONV_EXECUTION, API_DRIVER, 19, HIP_UNSUPPORTED | DEPRECATED}},

  // 20. Graph Management
  // cudaGraphAddChildGraphNode
  {"cuGraphAddChildGraphNode",                             {"hipGraphAddChildGraphNode",                               "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddDependencies
  {"cuGraphAddDependencies",                               {"hipGraphAddDependencies",                                 "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddEmptyNode
  {"cuGraphAddEmptyNode",                                  {"hipGraphAddEmptyNode",                                    "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddHostNode
  {"cuGraphAddHostNode",                                   {"hipGraphAddHostNode",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddKernelNode
  {"cuGraphAddKernelNode",                                 {"hipGraphAddKernelNode",                                   "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddMemcpyNode
  {"cuGraphAddMemcpyNode",                                 {"hipGraphAddMemcpyNode",                                   "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphAddMemsetNode
  {"cuGraphAddMemsetNode",                                 {"hipGraphAddMemsetNode",                                   "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphChildGraphNodeGetGraph
  {"cuGraphChildGraphNodeGetGraph",                        {"hipGraphChildGraphNodeGetGraph",                          "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphClone
  {"cuGraphClone",                                         {"hipGraphClone",                                           "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphCreate
  {"cuGraphCreate",                                        {"hipGraphCreate",                                          "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphDestroy
  {"cuGraphDestroy",                                       {"hipGraphDestroy",                                         "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphDestroyNode
  {"cuGraphDestroyNode",                                   {"hipGraphDestroyNode",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecDestroy
  {"cuGraphExecDestroy",                                   {"hipGraphExecDestroy",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphGetEdges
  {"cuGraphGetEdges",                                      {"hipGraphGetEdges",                                        "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphGetNodes
  {"cuGraphGetNodes",                                      {"hipGraphGetNodes",                                        "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphGetRootNodes
  {"cuGraphGetRootNodes",                                  {"hipGraphGetRootNodes",                                    "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphHostNodeGetParams
  {"cuGraphHostNodeGetParams",                             {"hipGraphHostNodeGetParams",                               "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphHostNodeSetParams
  {"cuGraphHostNodeSetParams",                             {"hipGraphHostNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphInstantiate
  {"cuGraphInstantiate",                                   {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  {"cuGraphInstantiate_v2",                                {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeCopyAttributes
  {"cuGraphKernelNodeCopyAttributes",                      {"hipGraphKernelNodeCopyAttributes",                        "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeGetAttribute
  {"cuGraphKernelNodeGetAttribute",                        {"hipGraphKernelNodeGetAttribute",                          "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecKernelNodeSetParams
  {"cuGraphExecKernelNodeSetParams",                       {"hipGraphExecKernelNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeGetParams
  {"cuGraphKernelNodeGetParams",                           {"hipGraphKernelNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeSetAttribute
  {"cuGraphKernelNodeSetAttribute",                        {"hipGraphKernelNodeSetAttribute",                          "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeSetParams
  {"cuGraphKernelNodeSetParams",                           {"hipGraphKernelNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphLaunch
  {"cuGraphLaunch",                                        {"hipGraphLaunch",                                          "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphMemcpyNodeGetParams
  {"cuGraphMemcpyNodeGetParams",                           {"hipGraphMemcpyNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphMemcpyNodeSetParams
  {"cuGraphMemcpyNodeSetParams",                           {"hipGraphMemcpyNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphMemsetNodeGetParams
  {"cuGraphMemsetNodeGetParams",                           {"hipGraphMemsetNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphMemsetNodeSetParams
  {"cuGraphMemsetNodeSetParams",                           {"hipGraphMemsetNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphNodeFindInClone
  {"cuGraphNodeFindInClone",                               {"hipGraphNodeFindInClone",                                 "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetDependencies
  {"cuGraphNodeGetDependencies",                           {"hipGraphNodeGetDependencies",                             "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetDependentNodes
  {"cuGraphNodeGetDependentNodes",                         {"hipGraphNodeGetDependentNodes",                           "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetType
  {"cuGraphNodeGetType",                                   {"hipGraphNodeGetType",                                     "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphRemoveDependencies
  {"cuGraphRemoveDependencies",                            {"hipGraphRemoveDependencies",                              "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecMemcpyNodeSetParams
  {"cuGraphExecMemcpyNodeSetParams",                       {"hipGraphExecMemcpyNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecMemsetNodeSetParams
  {"cuGraphExecMemsetNodeSetParams",                       {"hipGraphExecMemsetNodeSetParams",                         "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecHostNodeSetParams
  {"cuGraphExecHostNodeSetParams",                         {"hipGraphExecHostNodeSetParams",                           "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},
  // cudaGraphExecUpdate
  {"cuGraphExecUpdate",                                    {"hipGraphExecUpdate",                                      "", CONV_GRAPH, API_DRIVER, 20, HIP_UNSUPPORTED}},

  // 21. Occupancy
  // cudaOccupancyAvailableDynamicSMemPerBlock
  {"cuOccupancyAvailableDynamicSMemPerBlock",              {"hipOccupancyAvailableDynamicSMemPerBlock",                "", CONV_OCCUPANCY, API_DRIVER, 21, HIP_UNSUPPORTED}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipDrvOccupancyMaxActiveBlocksPerMultiprocessor",         "", CONV_OCCUPANCY, API_DRIVER, 21}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipDrvOccupancyMaxActiveBlocksPerMultiprocessorWithFlags","", CONV_OCCUPANCY, API_DRIVER, 21}},
  // cudaOccupancyMaxPotentialBlockSize
  {"cuOccupancyMaxPotentialBlockSize",                     {"hipOccupancyMaxPotentialBlockSize",                       "", CONV_OCCUPANCY, API_DRIVER, 21}},
  // cudaOccupancyMaxPotentialBlockSizeWithFlags
  {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipOccupancyMaxPotentialBlockSizeWithFlags",              "", CONV_OCCUPANCY, API_DRIVER, 21, HIP_UNSUPPORTED}},

  // 22. Texture Reference Management [DEPRECATED]
  // no analogues
  {"cuTexRefGetAddress",                                   {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefGetAddress_v2",                                {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefGetAddressMode",                               {"hipTexRefGetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefGetArray",                                     {"hipTexRefGetArray",                                       "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefGetBorderColor",                               {"hipTexRefGetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetFilterMode",                                {"hipTexRefGetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetFlags",                                     {"hipTexRefGetFlags",                                       "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetFormat",                                    {"hipTexRefGetFormat",                                      "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetMaxAnisotropy",                             {"hipTexRefGetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetMipmapFilterMode",                          {"hipTexRefGetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetMipmapLevelBias",                           {"hipTexRefGetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetMipmapLevelClamp",                          {"hipTexRefGetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefGetMipmappedArray",                            {"hipTexRefGetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetAddress",                                   {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetAddress_v2",                                {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetAddress2D",                                 {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetAddress2D_v2",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 22}},
  {"cuTexRefSetAddress2D_v3",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER, 22}},
  {"cuTexRefSetAddressMode",                               {"hipTexRefSetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetArray",                                     {"hipTexRefSetArray",                                       "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetBorderColor",                               {"hipTexRefSetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetFilterMode",                                {"hipTexRefSetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetFlags",                                     {"hipTexRefSetFlags",                                       "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetFormat",                                    {"hipTexRefSetFormat",                                      "", CONV_TEXTURE, API_DRIVER, 22, DEPRECATED}},
  {"cuTexRefSetMaxAnisotropy",                             {"hipTexRefSetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetMipmapFilterMode",                          {"hipTexRefSetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetMipmapLevelBias",                           {"hipTexRefSetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetMipmapLevelClamp",                          {"hipTexRefSetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefSetMipmappedArray",                            {"hipTexRefSetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefCreate",                                       {"hipTexRefCreate",                                         "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuTexRefDestroy",                                      {"hipTexRefDestroy",                                        "", CONV_TEXTURE, API_DRIVER, 22, HIP_UNSUPPORTED | DEPRECATED}},

  // 23. Surface Reference Management [DEPRECATED]
  // no analogues
  {"cuSurfRefGetArray",                                    {"hipSurfRefGetArray",                                      "", CONV_SURFACE, API_DRIVER, 23, HIP_UNSUPPORTED | DEPRECATED}},
  {"cuSurfRefSetArray",                                    {"hipSurfRefSetArray",                                      "", CONV_SURFACE, API_DRIVER, 23, HIP_UNSUPPORTED | DEPRECATED}},

  // 24. Texture Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateTextureObject due to different signatures
  {"cuTexObjectCreate",                                    {"hipTexObjectCreate",                                      "", CONV_TEXTURE, API_DRIVER, 24, HIP_UNSUPPORTED}},
  // cudaDestroyTextureObject
  {"cuTexObjectDestroy",                                   {"hipTexObjectDestroy",                                     "", CONV_TEXTURE, API_DRIVER, 24, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectResourceDesc due to different signatures
  {"cuTexObjectGetResourceDesc",                           {"hipTexObjectGetResourceDesc",                             "", CONV_TEXTURE, API_DRIVER, 24, HIP_UNSUPPORTED}},
  // cudaGetTextureObjectResourceViewDesc
  {"cuTexObjectGetResourceViewDesc",                       {"hipTexObjectGetResourceViewDesc",                         "", CONV_TEXTURE, API_DRIVER, 24, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                            {"hipTexObjectGetTextureDesc",                              "", CONV_TEXTURE, API_DRIVER, 24, HIP_UNSUPPORTED}},

  // 25. Surface Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateSurfaceObject due to different signatures
  {"cuSurfObjectCreate",                                   {"hipSurfObjectCreate",                                     "", CONV_TEXTURE, API_DRIVER, 25, HIP_UNSUPPORTED}},
  // cudaDestroySurfaceObject
  {"cuSurfObjectDestroy",                                  {"hipSurfObjectDestroy",                                    "", CONV_TEXTURE, API_DRIVER, 25, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetSurfaceObjectResourceDesc due to different signatures
  {"cuSurfObjectGetResourceDesc",                          {"hipSurfObjectGetResourceDesc",                            "", CONV_TEXTURE, API_DRIVER, 25, HIP_UNSUPPORTED}},

  // 26. Peer Context Memory Access
  // no analogue
  // NOTE: Not equal to cudaDeviceEnablePeerAccess due to different signatures
  {"cuCtxEnablePeerAccess",                                {"hipCtxEnablePeerAccess",                                  "", CONV_PEER, API_DRIVER, 26}},
  // no analogue
  // NOTE: Not equal to cudaDeviceDisablePeerAccess due to different signatures
  {"cuCtxDisablePeerAccess",                               {"hipCtxDisablePeerAccess",                                 "", CONV_PEER, API_DRIVER, 26}},
  // cudaDeviceCanAccessPeer
  {"cuDeviceCanAccessPeer",                                {"hipDeviceCanAccessPeer",                                  "", CONV_PEER, API_DRIVER, 26}},
  // cudaDeviceGetP2PAttribute
  {"cuDeviceGetP2PAttribute",                              {"hipDeviceGetP2PAttribute",                                "", CONV_PEER, API_DRIVER, 26, HIP_UNSUPPORTED}},

  // 27. Graphics Interoperability
  // cudaGraphicsMapResources
  {"cuGraphicsMapResources",                               {"hipGraphicsMapResources",                                 "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedMipmappedArray
  {"cuGraphicsResourceGetMappedMipmappedArray",            {"hipGraphicsResourceGetMappedMipmappedArray",              "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer",                   {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer_v2",                {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags",                        {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags_v2",                     {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsSubResourceGetMappedArray
  {"cuGraphicsSubResourceGetMappedArray",                  {"hipGraphicsSubResourceGetMappedArray",                    "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsUnmapResources
  {"cuGraphicsUnmapResources",                             {"hipGraphicsUnmapResources",                               "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},
  // cudaGraphicsUnregisterResource
  {"cuGraphicsUnregisterResource",                         {"hipGraphicsUnregisterResource",                           "", CONV_GRAPHICS, API_DRIVER, 27, HIP_UNSUPPORTED}},

  // 28. Profiler Control [DEPRECATED]
  // cudaProfilerInitialize
  {"cuProfilerInitialize",                                 {"hipProfilerInitialize",                                   "", CONV_PROFILER, API_DRIVER, 28, HIP_UNSUPPORTED}},

  // 29. Profiler Control
  // cudaProfilerStart
  {"cuProfilerStart",                                      {"hipProfilerStart",                                        "", CONV_PROFILER, API_DRIVER, 29}},
  // cudaProfilerStop
  {"cuProfilerStop",                                       {"hipProfilerStop",                                         "", CONV_PROFILER, API_DRIVER, 29}},

  // 30. OpenGL Interoperability
  // cudaGLGetDevices
  {"cuGLGetDevices",                                       {"hipGLGetDevices",                                         "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED}},
  // cudaGraphicsGLRegisterBuffer
  {"cuGraphicsGLRegisterBuffer",                           {"hipGraphicsGLRegisterBuffer",                             "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED}},
  // cudaGraphicsGLRegisterImage
  {"cuGraphicsGLRegisterImage",                            {"hipGraphicsGLRegisterImage",                              "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED}},
  // cudaWGLGetDevice
  {"cuWGLGetDevice",                                       {"hipWGLGetDevice",                                         "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED}},

  // 30. OpenGL Interoperability [DEPRECATED]
  // no analogue
  {"cuGLCtxCreate",                                        {"hipGLCtxCreate",                                          "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuGLInit",                                             {"hipGLInit",                                               "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObject due to different signatures
  {"cuGLMapBufferObject",                                  {"hipGLMapBufferObject_",                                   "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cudaGLMapBufferObjectAsync due to different signatures
  {"cuGLMapBufferObjectAsync",                             {"hipGLMapBufferObjectAsync_",                              "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaGLRegisterBufferObject
  {"cuGLRegisterBufferObject",                             {"hipGLRegisterBufferObject",                               "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaGLSetBufferObjectMapFlags
  {"cuGLSetBufferObjectMapFlags",                          {"hipGLSetBufferObjectMapFlags",                            "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaGLUnmapBufferObject
  {"cuGLUnmapBufferObject",                                {"hipGLUnmapBufferObject",                                  "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaGLUnmapBufferObjectAsync
  {"cuGLUnmapBufferObjectAsync",                           {"hipGLUnmapBufferObjectAsync",                             "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaGLUnregisterBufferObject
  {"cuGLUnregisterBufferObject",                           {"hipGLUnregisterBufferObject",                             "", CONV_OPENGL, API_DRIVER, 30, HIP_UNSUPPORTED | DEPRECATED}},

  // 31. Direct3D 9 Interoperability
  // no analogue
  {"cuD3D9CtxCreate",                                      {"hipD3D9CtxCreate",                                        "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},
    // no analogue
  {"cuD3D9CtxCreateOnDevice",                              {"hipD3D9CtxCreateOnDevice",                                "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevice
  {"cuD3D9GetDevice",                                      {"hipD3D9GetDevice",                                        "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevices
  {"cuD3D9GetDevices",                                     {"hipD3D9GetDevices",                                       "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},
  // cudaD3D9GetDirect3DDevice
  {"cuD3D9GetDirect3DDevice",                              {"hipD3D9GetDirect3DDevice",                                "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D9RegisterResource
  {"cuGraphicsD3D9RegisterResource",                       {"hipGraphicsD3D9RegisterResource",                         "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED}},

  // 31. Direct3D 9 Interoperability [DEPRECATED]
  // cudaD3D9MapResources
  {"cuD3D9MapResources",                                   {"hipD3D9MapResources",                                     "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9RegisterResource
  {"cuD3D9RegisterResource",                               {"hipD3D9RegisterResource",                                 "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceGetMappedArray
  {"cuD3D9ResourceGetMappedArray",                         {"hipD3D9ResourceGetMappedArray",                           "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceGetMappedPitch
  {"cuD3D9ResourceGetMappedPitch",                         {"hipD3D9ResourceGetMappedPitch",                           "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceGetMappedPointer
  {"cuD3D9ResourceGetMappedPointer",                       {"hipD3D9ResourceGetMappedPointer",                         "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceGetMappedSize
  {"cuD3D9ResourceGetMappedSize",                          {"hipD3D9ResourceGetMappedSize",                            "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceGetSurfaceDimensions
  {"cuD3D9ResourceGetSurfaceDimensions",                   {"hipD3D9ResourceGetSurfaceDimensions",                     "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9ResourceSetMapFlags
  {"cuD3D9ResourceSetMapFlags",                            {"hipD3D9ResourceSetMapFlags",                              "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9UnmapResources
  {"cuD3D9UnmapResources",                                 {"hipD3D9UnmapResources",                                   "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D9UnregisterResource
  {"cuD3D9UnregisterResource",                             {"hipD3D9UnregisterResource",                               "", CONV_D3D9, API_DRIVER, 31, HIP_UNSUPPORTED | DEPRECATED}},

  // 32. Direct3D 10 Interoperability
  // cudaD3D10GetDevice
  {"cuD3D10GetDevice",                                     {"hipD3D10GetDevice",                                       "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED}},
  // cudaD3D10GetDevices
  {"cuD3D10GetDevices",                                    {"hipD3D10GetDevices",                                      "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D10RegisterResource
  {"cuGraphicsD3D10RegisterResource",                      {"hipGraphicsD3D10RegisterResource",                        "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED}},

  // 32. Direct3D 10 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D10CtxCreate",                                     {"hipD3D10CtxCreate",                                       "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuD3D10CtxCreateOnDevice",                             {"hipD3D10CtxCreateOnDevice",                               "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10GetDirect3DDevice
  {"cuD3D10GetDirect3DDevice",                             {"hipD3D10GetDirect3DDevice",                               "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10MapResources
  {"cuD3D10MapResources",                                  {"hipD3D10MapResources",                                    "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10RegisterResource
  {"cuD3D10RegisterResource",                              {"hipD3D10RegisterResource",                                "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceGetMappedArray
  {"cuD3D10ResourceGetMappedArray",                        {"hipD3D10ResourceGetMappedArray",                          "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceGetMappedPitch
  {"cuD3D10ResourceGetMappedPitch",                        {"hipD3D10ResourceGetMappedPitch",                          "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceGetMappedPointer
  {"cuD3D10ResourceGetMappedPointer",                      {"hipD3D10ResourceGetMappedPointer",                        "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceGetMappedSize
  {"cuD3D10ResourceGetMappedSize",                         {"hipD3D10ResourceGetMappedSize",                           "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceGetSurfaceDimensions
  {"cuD3D10ResourceGetSurfaceDimensions",                  {"hipD3D10ResourceGetSurfaceDimensions",                    "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10ResourceSetMapFlags
  {"cuD3D10ResourceSetMapFlags",                           {"hipD3D10ResourceSetMapFlags",                             "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10UnmapResources
  {"cuD3D10UnmapResources",                                {"hipD3D10UnmapResources",                                  "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D10UnregisterResource
  {"cuD3D10UnregisterResource",                            {"hipD3D10UnregisterResource",                              "", CONV_D3D10, API_DRIVER, 32, HIP_UNSUPPORTED | DEPRECATED}},

  // 33. Direct3D 11 Interoperability
  // cudaD3D11GetDevice
  {"cuD3D11GetDevice",                                     {"hipD3D11GetDevice",                                       "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED}},
  // cudaD3D11GetDevices
  {"cuD3D11GetDevices",                                    {"hipD3D11GetDevices",                                      "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D11RegisterResource
  {"cuGraphicsD3D11RegisterResource",                      {"hipGraphicsD3D11RegisterResource",                        "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED}},

  // 33. Direct3D 11 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D11CtxCreate",                                     {"hipD3D11CtxCreate",                                       "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cuD3D11CtxCreateOnDevice",                             {"hipD3D11CtxCreateOnDevice",                               "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED | DEPRECATED}},
  // cudaD3D11GetDirect3DDevice
  {"cuD3D11GetDirect3DDevice",                             {"hipD3D11GetDirect3DDevice",                               "", CONV_D3D11, API_DRIVER, 33, HIP_UNSUPPORTED | DEPRECATED}},

  // 34. VDPAU Interoperability
  // cudaGraphicsVDPAURegisterOutputSurface
  {"cuGraphicsVDPAURegisterOutputSurface",                 {"hipGraphicsVDPAURegisterOutputSurface",                   "", CONV_VDPAU, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaGraphicsVDPAURegisterVideoSurface
  {"cuGraphicsVDPAURegisterVideoSurface",                  {"hipGraphicsVDPAURegisterVideoSurface",                    "", CONV_VDPAU, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // cudaVDPAUGetDevice
  {"cuVDPAUGetDevice",                                     {"hipVDPAUGetDevice",                                       "", CONV_VDPAU, API_DRIVER, 34, HIP_UNSUPPORTED}},
  // no analogue
  {"cuVDPAUCtxCreate",                                     {"hipVDPAUCtxCreate",                                       "", CONV_VDPAU, API_DRIVER, 34, HIP_UNSUPPORTED}},

  // 35. EGL Interoperability
  // cudaEGLStreamConsumerAcquireFrame
  {"cuEGLStreamConsumerAcquireFrame",                      {"hipEGLStreamConsumerAcquireFrame",                        "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnect
  {"cuEGLStreamConsumerConnect",                           {"hipEGLStreamConsumerConnect",                             "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnectWithFlags
  {"cuEGLStreamConsumerConnectWithFlags",                  {"hipEGLStreamConsumerConnectWithFlags",                    "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerDisconnect
  {"cuEGLStreamConsumerDisconnect",                        {"hipEGLStreamConsumerDisconnect",                          "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerReleaseFrame
  {"cuEGLStreamConsumerReleaseFrame",                      {"hipEGLStreamConsumerReleaseFrame",                        "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerConnect
  {"cuEGLStreamProducerConnect",                           {"hipEGLStreamProducerConnect",                             "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerDisconnect
  {"cuEGLStreamProducerDisconnect",                        {"hipEGLStreamProducerDisconnect",                          "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerPresentFrame
  {"cuEGLStreamProducerPresentFrame",                      {"hipEGLStreamProducerPresentFrame",                        "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerReturnFrame
  {"cuEGLStreamProducerReturnFrame",                       {"hipEGLStreamProducerReturnFrame",                         "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaGraphicsEGLRegisterImage
  {"cuGraphicsEGLRegisterImage",                           {"hipGraphicsEGLRegisterImage",                             "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedEglFrame
  {"cuGraphicsResourceGetMappedEglFrame",                  {"hipGraphicsResourceGetMappedEglFrame",                    "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
  // cudaEventCreateFromEGLSync
  {"cuEventCreateFromEGLSync",                             {"hipEventCreateFromEGLSync",                               "", CONV_EGL, API_DRIVER, 35, HIP_UNSUPPORTED}},
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
  {"cuLaunchCooperativeKernelMultiDevice",                 {CUDA_90,  CUDA_0,   CUDA_0  }},
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
  {13, "Unified Addressing"},
  {14, "Stream Management"},
  {15, "Event Management"},
  {16, "External Resource Interoperability"},
  {17, "Stream Memory Operations"},
  {18, "Execution Control"},
  {19, "Execution Control [DEPRECATED]"},
  {20, "Graph Management"},
  {21, "Occupancy"},
  {22, "Texture Reference Management [DEPRECATED]"},
  {23, "Surface Reference Management [DEPRECATED]"},
  {24, "Texture Object Management"},
  {25, "Surface Object Management"},
  {26, "Peer Context Memory Access"},
  {27, "Graphics Interoperability"},
  {28, "Profiler Control [DEPRECATED]"},
  {29, "Profiler Control"},
  {30, "OpenGL Interoperability"},
  {31, "Direct3D 9 Interoperability"},
  {32, "Direct3D 10 Interoperability"},
  {33, "Direct3D 11 Interoperability"},
  {34, "VDPAU Interoperability"},
  {35, "EGL Interoperability"},
};
