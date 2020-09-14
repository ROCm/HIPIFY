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

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP {
  // 1. Device Management
  // no analogue
  {"cudaChooseDevice",                                        {"hipChooseDevice",                                        "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuDeviceGetAttribute
  {"cudaDeviceGetAttribute",                                  {"hipDeviceGetAttribute",                                  "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuDeviceGetByPCIBusId
  {"cudaDeviceGetByPCIBusId",                                 {"hipDeviceGetByPCIBusId",                                 "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  {"cudaDeviceGetCacheConfig",                                {"hipDeviceGetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxGetLimit
  {"cudaDeviceGetLimit",                                      {"hipDeviceGetLimit",                                      "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuDeviceGetNvSciSyncAttributes
  {"cudaDeviceGetNvSciSyncAttributes",                        {"hipDeviceGetNvSciSyncAttributes",                        "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
  // cuDeviceGetP2PAttribute
  {"cudaDeviceGetP2PAttribute",                               {"hipDeviceGetP2PAttribute",                               "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
  // cuDeviceGetPCIBusId
  {"cudaDeviceGetPCIBusId",                                   {"hipDeviceGetPCIBusId",                                   "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxGetSharedMemConfig
  {"cudaDeviceGetSharedMemConfig",                            {"hipDeviceGetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxGetStreamPriorityRange
  {"cudaDeviceGetStreamPriorityRange",                        {"hipDeviceGetStreamPriorityRange",                        "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  {"cudaDeviceReset",                                         {"hipDeviceReset",                                         "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  {"cudaDeviceSetCacheConfig",                                {"hipDeviceSetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxSetLimit
  {"cudaDeviceSetLimit",                                      {"hipDeviceSetLimit",                                      "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxSetSharedMemConfig
  {"cudaDeviceSetSharedMemConfig",                            {"hipDeviceSetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxSynchronize
  {"cudaDeviceSynchronize",                                   {"hipDeviceSynchronize",                                   "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuDeviceGet
  // NOTE: cuDeviceGet has no attr: int ordinal
  {"cudaGetDevice",                                           {"hipGetDevice",                                           "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuDeviceGetCount
  {"cudaGetDeviceCount",                                      {"hipGetDeviceCount",                                      "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxGetFlags
  // TODO: rename to hipGetDeviceFlags
  {"cudaGetDeviceFlags",                                      {"hipCtxGetFlags",                                         "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  // NOTE: Not equal to cuDeviceGetProperties due to different attributes: CUdevprop and cudaDeviceProp
  {"cudaGetDeviceProperties",                                 {"hipGetDeviceProperties",                                 "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuIpcCloseMemHandle
  {"cudaIpcCloseMemHandle",                                   {"hipIpcCloseMemHandle",                                   "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuIpcGetEventHandle
  {"cudaIpcGetEventHandle",                                   {"hipIpcGetEventHandle",                                   "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuIpcGetMemHandle
  {"cudaIpcGetMemHandle",                                     {"hipIpcGetMemHandle",                                     "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuIpcOpenEventHandle
  {"cudaIpcOpenEventHandle",                                  {"hipIpcOpenEventHandle",                                  "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuIpcOpenMemHandle
  {"cudaIpcOpenMemHandle",                                    {"hipIpcOpenMemHandle",                                    "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  {"cudaSetDevice",                                           {"hipSetDevice",                                           "", CONV_DEVICE, API_RUNTIME, 1}},
  // cuCtxGetFlags
  {"cudaSetDeviceFlags",                                      {"hipSetDeviceFlags",                                      "", CONV_DEVICE, API_RUNTIME, 1}},
  // no analogue
  {"cudaSetValidDevices",                                     {"hipSetValidDevices",                                     "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},

  // 2. Thread Management [DEPRECATED]
  // no analogue
  {"cudaThreadExit",                                          {"hipDeviceReset",                                         "", CONV_THREAD, API_RUNTIME, 2, DEPRECATED}},
  // no analogue
  {"cudaThreadGetCacheConfig",                                {"hipDeviceGetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, 2, DEPRECATED}},
  // no analogue
  {"cudaThreadGetLimit",                                      {"hipThreadGetLimit",                                      "", CONV_THREAD, API_RUNTIME, 2, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaThreadSetCacheConfig",                                {"hipDeviceSetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, 2, DEPRECATED}},
  // no analogue
  {"cudaThreadSetLimit",                                      {"hipThreadSetLimit",                                      "", CONV_THREAD, API_RUNTIME, 2, HIP_UNSUPPORTED | DEPRECATED}},
  // cuCtxSynchronize
  {"cudaThreadSynchronize",                                   {"hipDeviceSynchronize",                                   "", CONV_THREAD, API_RUNTIME, 2, DEPRECATED}},

  // 3. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  {"cudaGetErrorName",                                        {"hipGetErrorName",                                        "", CONV_ERROR, API_RUNTIME, 3}},
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  {"cudaGetErrorString",                                      {"hipGetErrorString",                                      "", CONV_ERROR, API_RUNTIME, 3}},
  // no analogue
  {"cudaGetLastError",                                        {"hipGetLastError",                                        "", CONV_ERROR, API_RUNTIME, 3}},
  // no analogue
  {"cudaPeekAtLastError",                                     {"hipPeekAtLastError",                                     "", CONV_ERROR, API_RUNTIME, 3}},

  // 4. Stream Management
  // cuStreamAddCallback
  {"cudaStreamAddCallback",                                   {"hipStreamAddCallback",                                   "", CONV_STREAM, API_RUNTIME, 4}},
  // cuCtxResetPersistingL2Cache
  {"cudaCtxResetPersistingL2Cache",                           {"hipCtxResetPersistingL2Cache",                           "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamAttachMemAsync
  {"cudaStreamAttachMemAsync",                                {"hipStreamAttachMemAsync",                                "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamBeginCapture
  {"cudaStreamBeginCapture",                                  {"hipStreamBeginCapture",                                  "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamCopyAttributes
  {"cudaStreamCopyAttributes",                                {"hipStreamCopyAttributes",                                "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuStreamCreate due to different signatures
  {"cudaStreamCreate",                                        {"hipStreamCreate",                                        "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamCreate
  {"cudaStreamCreateWithFlags",                               {"hipStreamCreateWithFlags",                               "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamCreateWithPriority
  {"cudaStreamCreateWithPriority",                            {"hipStreamCreateWithPriority",                            "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamDestroy
  {"cudaStreamDestroy",                                       {"hipStreamDestroy",                                       "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamEndCapture
  {"cudaStreamEndCapture",                                    {"hipStreamEndCapture",                                    "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamGetAttribute
  {"cudaStreamGetAttribute",                                  {"hipStreamGetAttribute",                                  "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamSetAttribute
  {"cudaStreamSetAttribute",                                  {"hipStreamSetAttribute",                                  "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamGetFlags
  {"cudaStreamGetFlags",                                      {"hipStreamGetFlags",                                      "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamGetPriority
  {"cudaStreamGetPriority",                                   {"hipStreamGetPriority",                                   "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamIsCapturing
  {"cudaStreamIsCapturing",                                   {"hipStreamIsCapturing",                                   "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamGetCaptureInfo
  {"cudaStreamGetCaptureInfo",                                {"hipStreamGetCaptureInfo",                                "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},
  // cuStreamQuery
  {"cudaStreamQuery",                                         {"hipStreamQuery",                                         "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamSynchronize
  {"cudaStreamSynchronize",                                   {"hipStreamSynchronize",                                   "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamWaitEvent
  {"cudaStreamWaitEvent",                                     {"hipStreamWaitEvent",                                     "", CONV_STREAM, API_RUNTIME, 4}},
  // cuThreadExchangeStreamCaptureMode
  {"cudaThreadExchangeStreamCaptureMode",                     {"hipThreadExchangeStreamCaptureMode",                     "", CONV_STREAM, API_RUNTIME, 4, HIP_UNSUPPORTED}},

  // 5. Event Management
  // no analogue
  // NOTE: Not equal to cuEventCreate due to different signatures
  {"cudaEventCreate",                                         {"hipEventCreate",                                         "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventCreate
  {"cudaEventCreateWithFlags",                                {"hipEventCreateWithFlags",                                "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventDestroy
  {"cudaEventDestroy",                                        {"hipEventDestroy",                                        "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventElapsedTime
  {"cudaEventElapsedTime",                                    {"hipEventElapsedTime",                                    "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventQuery
  {"cudaEventQuery",                                          {"hipEventQuery",                                          "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventRecord
  {"cudaEventRecord",                                         {"hipEventRecord",                                         "", CONV_EVENT, API_RUNTIME, 5}},
  // cuEventSynchronize
  {"cudaEventSynchronize",                                    {"hipEventSynchronize",                                    "", CONV_EVENT, API_RUNTIME, 5}},

  // 6. External Resource Interoperability
  // cuDestroyExternalMemory
  {"cudaDestroyExternalMemory",                               {"hipDestroyExternalMemory",                               "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuDestroyExternalSemaphore
  {"cudaDestroyExternalSemaphore",                            {"hipDestroyExternalSemaphore",                            "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedBuffer
  {"cudaExternalMemoryGetMappedBuffer",                       {"hipExternalMemoryGetMappedBuffer",                       "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedMipmappedArray
  {"cudaExternalMemoryGetMappedMipmappedArray",               {"hipExternalMemoryGetMappedMipmappedArray",               "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuImportExternalMemory
  {"cudaImportExternalMemory",                                {"hipImportExternalMemory",                                "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuImportExternalSemaphore
  {"cudaImportExternalSemaphore",                             {"hipImportExternalSemaphore",                             "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuSignalExternalSemaphoresAsync
  {"cudaSignalExternalSemaphoresAsync",                       {"hipSignalExternalSemaphoresAsync",                       "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuWaitExternalSemaphoresAsync
  {"cudaWaitExternalSemaphoresAsync",                         {"hipWaitExternalSemaphoresAsync",                         "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},

  // 7. Execution Control
  // no analogue
  {"cudaFuncGetAttributes",                                   {"hipFuncGetAttributes",                                   "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  {"cudaFuncSetAttribute",                                    {"hipFuncSetAttribute",                                    "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuFuncSetCacheConfig due to different signatures
  {"cudaFuncSetCacheConfig",                                  {"hipFuncSetCacheConfig",                                  "", CONV_DEVICE, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuFuncSetSharedMemConfig due to different signatures
  {"cudaFuncSetSharedMemConfig",                              {"hipFuncSetSharedMemConfig",                              "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetParameterBuffer",                                  {"hipGetParameterBuffer",                                  "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetParameterBufferV2",                                {"hipGetParameterBufferV2",                                "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernel due to different signatures
  {"cudaLaunchCooperativeKernel",                             {"hipLaunchCooperativeKernel",                             "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernelMultiDevice due to different signatures
  {"cudaLaunchCooperativeKernelMultiDevice",                  {"hipLaunchCooperativeKernelMultiDevice",                  "", CONV_EXECUTION, API_RUNTIME, 7}},
  // cuLaunchHostFunc
  {"cudaLaunchHostFunc",                                      {"hipLaunchHostFunc",                                      "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchKernel due to different signatures
  {"cudaLaunchKernel",                                        {"hipLaunchKernel",                                        "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  {"cudaSetDoubleForDevice",                                  {"hipSetDoubleForDevice",                                  "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaSetDoubleForHost",                                    {"hipSetDoubleForHost",                                    "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED | DEPRECATED}},

  // 8. Occupancy
  // cuOccupancyAvailableDynamicSMemPerBlock
  {"cudaOccupancyAvailableDynamicSMemPerBlock",               {"hipOccupancyAvailableDynamicSMemPerBlock",               "", CONV_OCCUPANCY, API_RUNTIME, 8, HIP_UNSUPPORTED}},
  // cuOccupancyMaxActiveBlocksPerMultiprocessor
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessor",           {"hipOccupancyMaxActiveBlocksPerMultiprocessor",           "", CONV_OCCUPANCY, API_RUNTIME, 8}},
  // cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  "", CONV_OCCUPANCY, API_RUNTIME, 8}},
  // cuOccupancyMaxPotentialBlockSize
  {"cudaOccupancyMaxPotentialBlockSize",                      {"hipOccupancyMaxPotentialBlockSize",                      "", CONV_OCCUPANCY, API_RUNTIME, 8}},
  // cuOccupancyMaxPotentialBlockSizeWithFlags
  {"cudaOccupancyMaxPotentialBlockSizeWithFlags",             {"hipOccupancyMaxPotentialBlockSizeWithFlags",             "", CONV_OCCUPANCY, API_RUNTIME, 8, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMem",          {"hipOccupancyMaxPotentialBlockSizeVariableSMem",          "", CONV_OCCUPANCY, API_RUNTIME, 8, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", "", CONV_OCCUPANCY, API_RUNTIME, 8, HIP_UNSUPPORTED}},

  // 9. Memory Management
  // no analogue
  {"cudaArrayGetInfo",                                        {"hipArrayGetInfo",                                        "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // cuMemFree
  {"cudaFree",                                                {"hipFree",                                                "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaFreeArray",                                           {"hipFreeArray",                                           "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemFreeHost
  {"cudaFreeHost",                                            {"hipHostFree",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayDestroy due to different signatures
  {"cudaFreeMipmappedArray",                                  {"hipFreeMipmappedArray",                                  "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayGetLevel due to different signatures
  {"cudaGetMipmappedArrayLevel",                              {"hipGetMipmappedArrayLevel",                              "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetSymbolAddress",                                    {"hipGetSymbolAddress",                                    "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaGetSymbolSize",                                       {"hipGetSymbolSize",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostAlloc
  {"cudaHostAlloc",                                           {"hipHostMalloc",                                          "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostGetDevicePointer
  {"cudaHostGetDevicePointer",                                {"hipHostGetDevicePointer",                                "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostGetFlags
  {"cudaHostGetFlags",                                        {"hipHostGetFlags",                                        "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostRegister
  {"cudaHostRegister",                                        {"hipHostRegister",                                        "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostUnregister
  {"cudaHostUnregister",                                      {"hipHostUnregister",                                      "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemAlloc
  {"cudaMalloc",                                              {"hipMalloc",                                              "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMalloc3D",                                            {"hipMalloc3D",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMalloc3DArray",                                       {"hipMalloc3DArray",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMallocArray",                                         {"hipMallocArray",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostAlloc
  {"cudaMallocHost",                                          {"hipHostMalloc",                                          "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemAllocManaged
  {"cudaMallocManaged",                                       {"hipMallocManaged",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayCreate due to different signatures
  {"cudaMallocMipmappedArray",                                {"hipMallocMipmappedArray",                                "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemAllocPitch due to different signatures
  {"cudaMallocPitch",                                         {"hipMallocPitch",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemAdvise
  {"cudaMemAdvise",                                           {"hipMemAdvise",                                           "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy due to different signatures
  {"cudaMemcpy",                                              {"hipMemcpy",                                              "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2D due to different signatures
  {"cudaMemcpy2D",                                            {"hipMemcpy2D",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpy2DArrayToArray",                                {"hipMemcpy2DArrayToArray",                                "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2DAsync due to different signatures
  {"cudaMemcpy2DAsync",                                       {"hipMemcpy2DAsync",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpy2DFromArray",                                   {"hipMemcpy2DFromArray",                                   "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpy2DFromArrayAsync",                              {"hipMemcpy2DFromArrayAsync",                              "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpy2DToArray",                                     {"hipMemcpy2DToArray",                                     "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpy2DToArrayAsync",                                {"hipMemcpy2DToArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3D due to different signatures
  {"cudaMemcpy3D",                                            {"hipMemcpy3D",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DAsync due to different signatures
  {"cudaMemcpy3DAsync",                                       {"hipMemcpy3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeer due to different signatures
  {"cudaMemcpy3DPeer",                                        {"hipMemcpy3DPeer",                                        "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeerAsync due to different signatures
  {"cudaMemcpy3DPeerAsync",                                   {"hipMemcpy3DPeerAsync",                                   "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpyAsync due to different signatures
  {"cudaMemcpyAsync",                                         {"hipMemcpyAsync",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpyFromSymbol",                                    {"hipMemcpyFromSymbol",                                    "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpyFromSymbolAsync",                               {"hipMemcpyFromSymbolAsync",                               "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemcpyPeer due to different signatures
  {"cudaMemcpyPeer",                                          {"hipMemcpyPeer",                                          "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemcpyPeerAsync due to different signatures
  {"cudaMemcpyPeerAsync",                                     {"hipMemcpyPeerAsync",                                     "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpyToSymbol",                                      {"hipMemcpyToSymbol",                                      "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemcpyToSymbolAsync",                                 {"hipMemcpyToSymbolAsync",                                 "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemGetInfo
  {"cudaMemGetInfo",                                          {"hipMemGetInfo",                                          "", CONV_MEMORY, API_RUNTIME, 9}},
  // TODO: double check cuMemPrefetchAsync
  {"cudaMemPrefetchAsync",                                    {"hipMemPrefetchAsync",                                    "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttribute
  {"cudaMemRangeGetAttribute",                                {"hipMemRangeGetAttribute",                                "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttributes
  {"cudaMemRangeGetAttributes",                               {"hipMemRangeGetAttributes",                               "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // cuMemsetD32 - hipMemsetD32
  {"cudaMemset",                                              {"hipMemset",                                              "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemset2D",                                            {"hipMemset2D",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemset2DAsync",                                       {"hipMemset2DAsync",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemset3D",                                            {"hipMemset3D",                                            "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaMemset3DAsync",                                       {"hipMemset3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemsetD32Async
  {"cudaMemsetAsync",                                         {"hipMemsetAsync",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"make_cudaExtent",                                         {"make_hipExtent",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"make_cudaPitchedPtr",                                     {"make_hipPitchedPtr",                                     "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"make_cudaPos",                                            {"make_hipPos",                                            "", CONV_MEMORY, API_RUNTIME, 9}},

  // 10. Memory Management [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuMemcpyAtoA due to different signatures
  {"cudaMemcpyArrayToArray",                                  {"hipMemcpyArrayToArray",                                  "", CONV_MEMORY, API_RUNTIME, 10, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaMemcpyFromArray",                                     {"hipMemcpyFromArray",                                     "", CONV_MEMORY, API_RUNTIME, 10, DEPRECATED}},
  // no analogue
  {"cudaMemcpyFromArrayAsync",                                {"hipMemcpyFromArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, 10, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaMemcpyToArray",                                       {"hipMemcpyToArray",                                       "", CONV_MEMORY, API_RUNTIME, 10, DEPRECATED}},
  // no analogue
  {"cudaMemcpyToArrayAsync",                                  {"hipMemcpyToArrayAsync",                                  "", CONV_MEMORY, API_RUNTIME, 10, DEPRECATED}},

  // 11. Unified Addressing
  // no analogue
  // NOTE: Not equal to cuPointerGetAttributes due to different signatures
  {"cudaPointerGetAttributes",                                {"hipPointerGetAttributes",                                "", CONV_ADDRESSING, API_RUNTIME, 11}},

  // 12. Peer Device Memory Access
  // cuDeviceCanAccessPeer
  {"cudaDeviceCanAccessPeer",                                 {"hipDeviceCanAccessPeer",                                 "", CONV_PEER, API_RUNTIME, 12}},
  // no analogue
  // NOTE: Not equal to cuCtxDisablePeerAccess due to different signatures
  {"cudaDeviceDisablePeerAccess",                             {"hipDeviceDisablePeerAccess",                             "", CONV_PEER, API_RUNTIME, 12}},
  // no analogue
  // NOTE: Not equal to cuCtxEnablePeerAccess due to different signatures
  {"cudaDeviceEnablePeerAccess",                              {"hipDeviceEnablePeerAccess",                              "", CONV_PEER, API_RUNTIME, 12}},

  // 13. OpenGL Interoperability
  // cuGLGetDevices
  {"cudaGLGetDevices",                                        {"hipGLGetDevices",                                        "", CONV_OPENGL, API_RUNTIME, 13, HIP_UNSUPPORTED}},
  // cuGraphicsGLRegisterBuffer
  {"cudaGraphicsGLRegisterBuffer",                            {"hipGraphicsGLRegisterBuffer",                            "", CONV_OPENGL, API_RUNTIME, 13, HIP_UNSUPPORTED}},
  // cuGraphicsGLRegisterImage
  {"cudaGraphicsGLRegisterImage",                             {"hipGraphicsGLRegisterImage",                             "", CONV_OPENGL, API_RUNTIME, 13, HIP_UNSUPPORTED}},
  // cuWGLGetDevice
  {"cudaWGLGetDevice",                                        {"hipWGLGetDevice",                                        "", CONV_OPENGL, API_RUNTIME, 13, HIP_UNSUPPORTED}},

  // 14. OpenGL Interoperability [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObject due to different signatures
  {"cudaGLMapBufferObject",                                   {"hipGLMapBufferObject",                                   "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObjectAsync due to different signatures
  {"cudaGLMapBufferObjectAsync",                              {"hipGLMapBufferObjectAsync",                              "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // cuGLRegisterBufferObject
  {"cudaGLRegisterBufferObject",                              {"hipGLRegisterBufferObject",                              "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // cuGLSetBufferObjectMapFlags
  {"cudaGLSetBufferObjectMapFlags",                           {"hipGLSetBufferObjectMapFlags",                           "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaGLSetGLDevice",                                       {"hipGLSetGLDevice",                                       "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // cuGLUnmapBufferObject
  {"cudaGLUnmapBufferObject",                                 {"hipGLUnmapBufferObject",                                 "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // cuGLUnmapBufferObjectAsync
  {"cudaGLUnmapBufferObjectAsync",                            {"hipGLUnmapBufferObjectAsync",                            "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},
  // cuGLUnregisterBufferObject
  {"cudaGLUnregisterBufferObject",                            {"hipGLUnregisterBufferObject",                            "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED | DEPRECATED}},

  // 15. Direct3D 9 Interoperability
  // cuD3D9GetDevice
  {"cudaD3D9GetDevice",                                       {"hipD3D9GetDevice",                                       "", CONV_D3D9, API_RUNTIME, 15, HIP_UNSUPPORTED}},
  // cuD3D9GetDevices
  {"cudaD3D9GetDevices",                                      {"hipD3D9GetDevices",                                      "", CONV_D3D9, API_RUNTIME, 15, HIP_UNSUPPORTED}},
  // cuD3D9GetDirect3DDevice
  {"cudaD3D9GetDirect3DDevice",                               {"hipD3D9GetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, 15, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaD3D9SetDirect3DDevice",                               {"hipD3D9SetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, 15, HIP_UNSUPPORTED}},
  // cuGraphicsD3D9RegisterResource
  {"cudaGraphicsD3D9RegisterResource",                        {"hipGraphicsD3D9RegisterResource",                        "", CONV_D3D9, API_RUNTIME, 15, HIP_UNSUPPORTED}},

  // 16. Direct3D 9 Interoperability[DEPRECATED]
  // cuD3D9MapResources
  {"cudaD3D9MapResources",                                    {"hipD3D9MapResources",                                    "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9RegisterResource
  // NOTE: cudaD3D9RegisterResource is not marked as deprecated function even in CUDA 11.0
  {"cudaD3D9RegisterResource",                                {"hipD3D9RegisterResource",                                "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetMappedArray
  {"cudaD3D9ResourceGetMappedArray",                          {"hipD3D9ResourceGetMappedArray",                          "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9ResourceGetMappedPitch
  {"cudaD3D9ResourceGetMappedPitch",                          {"hipD3D9ResourceGetMappedPitch",                          "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9ResourceGetMappedPointer
  {"cudaD3D9ResourceGetMappedPointer",                        {"hipD3D9ResourceGetMappedPointer",                        "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9ResourceGetMappedSize
  {"cudaD3D9ResourceGetMappedSize",                           {"hipD3D9ResourceGetMappedSize",                           "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9ResourceGetSurfaceDimensions
  {"cudaD3D9ResourceGetSurfaceDimensions",                    {"hipD3D9ResourceGetSurfaceDimensions",                    "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9ResourceSetMapFlags
  {"cudaD3D9ResourceSetMapFlags",                             {"hipD3D9ResourceSetMapFlags",                             "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9UnmapResources
  {"cudaD3D9UnmapResources",                                  {"hipD3D9UnmapResources",                                  "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D9UnregisterResource
  {"cudaD3D9UnregisterResource",                              {"hipD3D9UnregisterResource",                              "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED | DEPRECATED}},

  // 17. Direct3D 10 Interoperability
  // cuD3D10GetDevice
  {"cudaD3D10GetDevice",                                      {"hipD3D10GetDevice",                                      "", CONV_D3D10, API_RUNTIME, 17, HIP_UNSUPPORTED}},
  // cuD3D10GetDevices
  {"cudaD3D10GetDevices",                                     {"hipD3D10GetDevices",                                     "", CONV_D3D10, API_RUNTIME, 17, HIP_UNSUPPORTED}},
  // cuGraphicsD3D10RegisterResource
  {"cudaGraphicsD3D10RegisterResource",                       {"hipGraphicsD3D10RegisterResource",                       "", CONV_D3D10, API_RUNTIME, 17, HIP_UNSUPPORTED}},

  // 18. Direct3D 10 Interoperability [DEPRECATED]
  // cuD3D10GetDirect3DDevice
  {"cudaD3D10GetDirect3DDevice",                              {"hipD3D10GetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10MapResources
  {"cudaD3D10MapResources",                                   {"hipD3D10MapResources",                                   "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10RegisterResource
  {"cudaD3D10RegisterResource",                               {"hipD3D10RegisterResource",                               "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceGetMappedArray
  {"cudaD3D10ResourceGetMappedArray",                         {"hipD3D10ResourceGetMappedArray",                         "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceGetMappedPitch
  {"cudaD3D10ResourceGetMappedPitch",                         {"hipD3D10ResourceGetMappedPitch",                         "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceGetMappedPointer
  {"cudaD3D10ResourceGetMappedPointer",                       {"hipD3D10ResourceGetMappedPointer",                       "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceGetMappedSize
  {"cudaD3D10ResourceGetMappedSize",                          {"hipD3D10ResourceGetMappedSize",                          "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceGetSurfaceDimensions
  {"cudaD3D10ResourceGetSurfaceDimensions",                   {"hipD3D10ResourceGetSurfaceDimensions",                   "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10ResourceSetMapFlags
  {"cudaD3D10ResourceSetMapFlags",                            {"hipD3D10ResourceSetMapFlags",                            "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaD3D10SetDirect3DDevice",                              {"hipD3D10SetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10UnmapResources
  {"cudaD3D10UnmapResources",                                 {"hipD3D10UnmapResources",                                 "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},
  // cuD3D10UnregisterResource
  {"cudaD3D10UnregisterResource",                             {"hipD3D10UnregisterResource",                             "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED | DEPRECATED}},

  // 19. Direct3D 11 Interoperability
  // cuD3D11GetDevice
  {"cudaD3D11GetDevice",                                      {"hipD3D11GetDevice",                                      "", CONV_D3D11, API_RUNTIME, 19, HIP_UNSUPPORTED}},
  // cuD3D11GetDevices
  {"cudaD3D11GetDevices",                                     {"hipD3D11GetDevices",                                     "", CONV_D3D11, API_RUNTIME, 19, HIP_UNSUPPORTED}},
  // cuGraphicsD3D11RegisterResource
  {"cudaGraphicsD3D11RegisterResource",                       {"hipGraphicsD3D11RegisterResource",                       "", CONV_D3D11, API_RUNTIME, 19, HIP_UNSUPPORTED}},

  // 20. Direct3D 11 Interoperability [DEPRECATED]
  // cuD3D11GetDirect3DDevice
  {"cudaD3D11GetDirect3DDevice",                              {"hipD3D11GetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, 20, HIP_UNSUPPORTED | DEPRECATED}},
  // no analogue
  {"cudaD3D11SetDirect3DDevice",                              {"hipD3D11SetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, 20, HIP_UNSUPPORTED | DEPRECATED}},

  // 21. VDPAU Interoperability
  // cuGraphicsVDPAURegisterOutputSurface
  {"cudaGraphicsVDPAURegisterOutputSurface",                  {"hipGraphicsVDPAURegisterOutputSurface",                  "", CONV_VDPAU, API_RUNTIME, 21, HIP_UNSUPPORTED}},
  // cuGraphicsVDPAURegisterVideoSurface
  {"cudaGraphicsVDPAURegisterVideoSurface",                   {"hipGraphicsVDPAURegisterVideoSurface",                   "", CONV_VDPAU, API_RUNTIME, 21, HIP_UNSUPPORTED}},
  // cuVDPAUGetDevice
  {"cudaVDPAUGetDevice",                                      {"hipVDPAUGetDevice",                                      "", CONV_VDPAU, API_RUNTIME, 21, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaVDPAUSetVDPAUDevice",                                 {"hipVDPAUSetDevice",                                      "", CONV_VDPAU, API_RUNTIME, 21, HIP_UNSUPPORTED}},

  // 22. EGL Interoperability
  // cuEGLStreamConsumerAcquireFrame
  {"cudaEGLStreamConsumerAcquireFrame",                       {"hipEGLStreamConsumerAcquireFrame",                       "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnect
  {"cudaEGLStreamConsumerConnect",                            {"hipEGLStreamConsumerConnect",                            "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnectWithFlags
  {"cudaEGLStreamConsumerConnectWithFlags",                   {"hipEGLStreamConsumerConnectWithFlags",                   "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerDisconnect
  {"cudaEGLStreamConsumerDisconnect",                         {"hipEGLStreamConsumerDisconnect",                         "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerReleaseFrame
  {"cudaEGLStreamConsumerReleaseFrame",                       {"hipEGLStreamConsumerReleaseFrame",                       "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerConnect
  {"cudaEGLStreamProducerConnect",                            {"hipEGLStreamProducerConnect",                            "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerDisconnect
  {"cudaEGLStreamProducerDisconnect",                         {"hipEGLStreamProducerDisconnect",                         "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerPresentFrame
  {"cudaEGLStreamProducerPresentFrame",                       {"hipEGLStreamProducerPresentFrame",                       "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerReturnFrame
  {"cudaEGLStreamProducerReturnFrame",                        {"hipEGLStreamProducerReturnFrame",                        "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuEventCreateFromEGLSync
  {"cudaEventCreateFromEGLSync",                              {"hipEventCreateFromEGLSync",                              "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuGraphicsEGLRegisterImage
  {"cudaGraphicsEGLRegisterImage",                            {"hipGraphicsEGLRegisterImage",                            "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedEglFrame
  {"cudaGraphicsResourceGetMappedEglFrame",                   {"hipGraphicsResourceGetMappedEglFrame",                   "", CONV_EGL, API_RUNTIME, 22, HIP_UNSUPPORTED}},

  // 23. Graphics Interoperability
  // cuGraphicsMapResources
  {"cudaGraphicsMapResources",                                {"hipGraphicsMapResources",                                "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedMipmappedArray
  {"cudaGraphicsResourceGetMappedMipmappedArray",             {"hipGraphicsResourceGetMappedMipmappedArray",             "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedPointer
  {"cudaGraphicsResourceGetMappedPointer",                    {"hipGraphicsResourceGetMappedPointer",                    "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsResourceSetMapFlags
  {"cudaGraphicsResourceSetMapFlags",                         {"hipGraphicsResourceSetMapFlags",                         "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsSubResourceGetMappedArray
  {"cudaGraphicsSubResourceGetMappedArray",                   {"hipGraphicsSubResourceGetMappedArray",                   "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsUnmapResources
  {"cudaGraphicsUnmapResources",                              {"hipGraphicsUnmapResources",                              "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsUnregisterResource
  {"cudaGraphicsUnregisterResource",                          {"hipGraphicsUnregisterResource",                          "", CONV_GRAPHICS, API_RUNTIME, 23, HIP_UNSUPPORTED}},

  // 24. Texture Reference Management [DEPRECATED]
  // no analogue
  {"cudaBindTexture",                                         {"hipBindTexture",                                         "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // no analogue
  {"cudaBindTexture2D",                                       {"hipBindTexture2D",                                       "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // no analogue
  {"cudaBindTextureToArray",                                  {"hipBindTextureToArray",                                  "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // no analogue
  // NOTE: Unsupported yet on NVCC path
  {"cudaBindTextureToMipmappedArray",                         {"hipBindTextureToMipmappedArray",                         "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // no analogue
  {"cudaCreateChannelDesc",                                   {"hipCreateChannelDesc",                                   "", CONV_TEXTURE, API_RUNTIME, 24}},
  // no analogue
  {"cudaGetChannelDesc",                                      {"hipGetChannelDesc",                                      "", CONV_TEXTURE, API_RUNTIME, 24}},
  // no analogue
  {"cudaGetTextureAlignmentOffset",                           {"hipGetTextureAlignmentOffset",                           "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // TODO: double check cuModuleGetTexRef
  // NOTE: Unsupported yet on NVCC path
  {"cudaGetTextureReference",                                 {"hipGetTextureReference",                                 "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},
  // no analogue
  {"cudaUnbindTexture",                                       {"hipUnbindTexture",                                       "", CONV_TEXTURE, API_RUNTIME, 24, DEPRECATED}},

  // 25. Surface Reference Management [DEPRECATED]
  // no analogue
  {"cudaBindSurfaceToArray",                                  {"hipBindSurfaceToArray",                                  "", CONV_SURFACE, API_RUNTIME, 25, HIP_UNSUPPORTED | DEPRECATED}},
  // TODO: double check cuModuleGetSurfRef
  {"cudaGetSurfaceReference",                                 {"hipGetSurfaceReference",                                 "", CONV_SURFACE, API_RUNTIME, 25, HIP_UNSUPPORTED | DEPRECATED}},

  // 26. Texture Object Management
  // no analogue
  // NOTE: Not equal to cuTexObjectCreate due to different signatures
  {"cudaCreateTextureObject",                                 {"hipCreateTextureObject",                                 "", CONV_TEXTURE, API_RUNTIME, 26}},
  // cuTexObjectDestroy
  {"cudaDestroyTextureObject",                                {"hipDestroyTextureObject",                                "", CONV_TEXTURE, API_RUNTIME, 26}},
  // no analogue
  // NOTE: Not equal to cuTexObjectGetResourceDesc due to different signatures
  {"cudaGetTextureObjectResourceDesc",                        {"hipGetTextureObjectResourceDesc",                        "", CONV_TEXTURE, API_RUNTIME, 26}},
  // cuTexObjectGetResourceViewDesc
  {"cudaGetTextureObjectResourceViewDesc",                    {"hipGetTextureObjectResourceViewDesc",                    "", CONV_TEXTURE, API_RUNTIME, 26}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                               {"hipGetTextureObjectTextureDesc",                         "", CONV_TEXTURE, API_RUNTIME, 26}},

  // 27. Surface Object Management
  // no analogue
  // NOTE: Not equal to cuSurfObjectCreate due to different signatures
  {"cudaCreateSurfaceObject",                                 {"hipCreateSurfaceObject",                                 "", CONV_SURFACE, API_RUNTIME, 27}},
  // cuSurfObjectDestroy
  {"cudaDestroySurfaceObject",                                {"hipDestroySurfaceObject",                                "", CONV_SURFACE, API_RUNTIME, 27}},
  // no analogue
  // NOTE: Not equal to cuSurfObjectGetResourceDesc due to different signatures
  {"cudaGetSurfaceObjectResourceDesc",                        {"hipGetSurfaceObjectResourceDesc",                        "", CONV_SURFACE, API_RUNTIME, 27, HIP_UNSUPPORTED}},

  // 28. Version Management
  // cuDriverGetVersion
  {"cudaDriverGetVersion",                                    {"hipDriverGetVersion",                                    "", CONV_VERSION, API_RUNTIME, 28}},
  // no analogue
  {"cudaRuntimeGetVersion",                                   {"hipRuntimeGetVersion",                                   "", CONV_VERSION, API_RUNTIME, 28}},

  // 29. Graph Management
  // cuGraphAddChildGraphNode
  {"cudaGraphAddChildGraphNode",                              {"hipGraphAddChildGraphNode",                              "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddDependencies
  {"cudaGraphAddDependencies",                                {"hipGraphAddDependencies",                                "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddEmptyNode
  {"cudaGraphAddEmptyNode",                                   {"hipGraphAddEmptyNode",                                   "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddHostNode
  {"cudaGraphAddHostNode",                                    {"hipGraphAddHostNode",                                    "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddKernelNode
  {"cudaGraphAddKernelNode",                                  {"hipGraphAddKernelNode",                                  "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddMemcpyNode
  {"cudaGraphAddMemcpyNode",                                  {"hipGraphAddMemcpyNode",                                  "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphAddMemsetNode
  {"cudaGraphAddMemsetNode",                                  {"hipGraphAddMemsetNode",                                  "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphChildGraphNodeGetGraph
  {"cudaGraphChildGraphNodeGetGraph",                         {"hipGraphChildGraphNodeGetGraph",                         "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphClone
  {"cudaGraphClone",                                          {"hipGraphClone",                                          "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphCreate
  {"cudaGraphCreate",                                         {"hipGraphCreate",                                         "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphDestroy
  {"cudaGraphDestroy",                                        {"hipGraphDestroy",                                        "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphDestroyNode
  {"cudaGraphDestroyNode",                                    {"hipGraphDestroyNode",                                    "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecDestroy
  {"cudaGraphExecDestroy",                                    {"hipGraphExecDestroy",                                    "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphGetEdges
  {"cudaGraphGetEdges",                                       {"hipGraphGetEdges",                                       "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphGetNodes
  {"cudaGraphGetNodes",                                       {"hipGraphGetNodes",                                       "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphGetRootNodes
  {"cudaGraphGetRootNodes",                                   {"hipGraphGetRootNodes",                                   "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphHostNodeGetParams
  {"cudaGraphHostNodeGetParams",                              {"hipGraphHostNodeGetParams",                              "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphHostNodeSetParams
  {"cudaGraphHostNodeSetParams",                              {"hipGraphHostNodeSetParams",                              "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphInstantiate
  {"cudaGraphInstantiate",                                    {"hipGraphInstantiate",                                    "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeCopyAttributes
  {"cudaGraphKernelNodeCopyAttributes",                       {"hipGraphKernelNodeCopyAttributes",                       "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeGetAttribute
  {"cudaGraphKernelNodeGetAttribute",                         {"hipGraphKernelNodeGetAttribute",                         "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeSetAttribute
  {"cudaGraphKernelNodeSetAttribute",                         {"hipGraphKernelNodeSetAttribute",                         "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecKernelNodeSetParams
  {"cudaGraphExecKernelNodeSetParams",                        {"hipGraphExecKernelNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecMemcpyNodeSetParams
  {"cudaGraphExecMemcpyNodeSetParams",                        {"hipGraphExecMemcpyNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecMemsetNodeSetParams
  {"cudaGraphExecMemsetNodeSetParams",                        {"hipGraphExecMemsetNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecHostNodeSetParams
  {"cudaGraphExecHostNodeSetParams",                          {"hipGraphExecHostNodeSetParams",                          "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphExecUpdate
  {"cudaGraphExecUpdate",                                     {"hipGraphExecUpdate",                                     "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeGetParams
  {"cudaGraphKernelNodeGetParams",                            {"hipGraphKernelNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeSetParams
  {"cudaGraphKernelNodeSetParams",                            {"hipGraphKernelNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphLaunch
  {"cudaGraphLaunch",                                         {"hipGraphLaunch",                                         "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphMemcpyNodeGetParams
  {"cudaGraphMemcpyNodeGetParams",                            {"hipGraphMemcpyNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphMemcpyNodeSetParams
  {"cudaGraphMemcpyNodeSetParams",                            {"hipGraphMemcpyNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphMemsetNodeGetParams
  {"cudaGraphMemsetNodeGetParams",                            {"hipGraphMemsetNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphMemsetNodeSetParams
  {"cudaGraphMemsetNodeSetParams",                            {"hipGraphMemsetNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphNodeFindInClone
  {"cudaGraphNodeFindInClone",                                {"hipGraphNodeFindInClone",                                "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependencies
  {"cudaGraphNodeGetDependencies",                            {"hipGraphNodeGetDependencies",                            "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependentNodes
  {"cudaGraphNodeGetDependentNodes",                          {"hipGraphNodeGetDependentNodes",                          "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphNodeGetType
  {"cudaGraphNodeGetType",                                    {"hipGraphNodeGetType",                                    "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},
  // cuGraphRemoveDependencies
  {"cudaGraphRemoveDependencies",                             {"hipGraphRemoveDependencies",                             "", CONV_GRAPH, API_RUNTIME, 29, HIP_UNSUPPORTED}},

  // 30. C++ API Routines
  // TODO

  // 31. Interactions with the CUDA Driver API
  {"cudaGetFuncBySymbol",                                     {"hipGetFuncBySymbol",                                     "", CONV_INTERACTION, API_RUNTIME, 31, HIP_UNSUPPORTED}},

  // 32. Profiler Control [DEPRECATED]
  // cuProfilerInitialize
  {"cudaProfilerInitialize",                                  {"hipProfilerInitialize",                                  "", CONV_PROFILER, API_RUNTIME, 32, HIP_UNSUPPORTED}},

  // 33. Profiler Control
  // cuProfilerStart
  {"cudaProfilerStart",                                       {"hipProfilerStart",                                       "", CONV_PROFILER, API_RUNTIME, 33}},
  // cuProfilerStop
  {"cudaProfilerStop",                                        {"hipProfilerStop",                                        "", CONV_PROFILER, API_RUNTIME, 33}},

  // 34. Data types used by CUDA Runtime
  // NOTE: in a separate file

  // 35. Execution Control [REMOVED]
  // NOTE: Removed in CUDA 10.1
  // no analogue
  {"cudaConfigureCall",                                       {"hipConfigureCall",                                       "", CONV_EXECUTION, API_RUNTIME, 35, REMOVED}},
  // no analogue
  // NOTE: Not equal to cuLaunch due to different signatures
  {"cudaLaunch",                                              {"hipLaunchByPtr",                                         "", CONV_EXECUTION, API_RUNTIME, 35, REMOVED}},
  // no analogue
  {"cudaSetupArgument",                                       {"hipSetupArgument",                                       "", CONV_EXECUTION, API_RUNTIME, 35, REMOVED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RUNTIME_FUNCTION_VER_MAP {
  {"cudaDeviceGetNvSciSyncAttributes",                        {CUDA_102, CUDA_0,   CUDA_0  }},
  {"cudaDeviceGetP2PAttribute",                               {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaCtxResetPersistingL2Cache",                           {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaThreadExit",                                          {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaThreadGetCacheConfig",                                {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaThreadGetLimit",                                      {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaThreadSetCacheConfig",                                {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaThreadSetLimit",                                      {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaThreadSynchronize",                                   {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaStreamBeginCapture",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaStreamCopyAttributes",                                {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaStreamEndCapture",                                    {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaStreamGetAttribute",                                  {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaStreamSetAttribute",                                  {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaStreamIsCapturing",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaStreamGetCaptureInfo",                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cudaThreadExchangeStreamCaptureMode",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cudaDestroyExternalMemory",                               {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaDestroyExternalSemaphore",                            {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaExternalMemoryGetMappedBuffer",                       {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaExternalMemoryGetMappedMipmappedArray",               {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaImportExternalMemory",                                {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaImportExternalSemaphore",                             {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaSignalExternalSemaphoresAsync",                       {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaWaitExternalSemaphoresAsync",                         {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaFuncSetAttribute",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaLaunchCooperativeKernel",                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaLaunchCooperativeKernelMultiDevice",                  {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaLaunchHostFunc",                                      {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaSetDoubleForDevice",                                  {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaSetDoubleForHost",                                    {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaOccupancyAvailableDynamicSMemPerBlock",               {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaMemAdvise",                                           {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaMemPrefetchAsync",                                    {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaMemRangeGetAttribute",                                {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaMemRangeGetAttributes",                               {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaMemcpyArrayToArray",                                  {CUDA_0,   CUDA_101, CUDA_0  }},
  {"cudaMemcpyFromArray",                                     {CUDA_0,   CUDA_101, CUDA_0  }},
  {"cudaMemcpyFromArrayAsync",                                {CUDA_0,   CUDA_101, CUDA_0  }},
  {"cudaMemcpyToArray",                                       {CUDA_0,   CUDA_101, CUDA_0  }},
  {"cudaMemcpyToArrayAsync",                                  {CUDA_0,   CUDA_101, CUDA_0  }},
  {"cudaGLMapBufferObject",                                   {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLMapBufferObjectAsync",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLRegisterBufferObject",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLSetBufferObjectMapFlags",                           {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLSetGLDevice",                                       {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLUnmapBufferObject",                                 {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLUnmapBufferObjectAsync",                            {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaGLUnregisterBufferObject",                            {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9MapResources",                                    {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceGetMappedArray",                          {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceGetMappedPitch",                          {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceGetMappedPointer",                        {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceGetMappedSize",                           {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceGetSurfaceDimensions",                    {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9ResourceSetMapFlags",                             {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9UnmapResources",                                  {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D9UnregisterResource",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10GetDirect3DDevice",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10MapResources",                                   {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10RegisterResource",                               {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceGetMappedArray",                         {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceGetMappedPitch",                         {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceGetMappedPointer",                       {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceGetMappedSize",                          {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceGetSurfaceDimensions",                   {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10ResourceSetMapFlags",                            {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10SetDirect3DDevice",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10UnmapResources",                                 {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D10UnregisterResource",                             {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D11GetDirect3DDevice",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaD3D11SetDirect3DDevice",                              {CUDA_0,   CUDA_100, CUDA_0  }},
  {"cudaEGLStreamConsumerAcquireFrame",                       {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamConsumerConnect",                            {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamConsumerConnectWithFlags",                   {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamConsumerDisconnect",                         {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamConsumerReleaseFrame",                       {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamProducerConnect",                            {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamProducerDisconnect",                         {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamProducerPresentFrame",                       {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEGLStreamProducerReturnFrame",                        {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaEventCreateFromEGLSync",                              {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaGraphicsEGLRegisterImage",                            {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaGraphicsResourceGetMappedEglFrame",                   {CUDA_91,  CUDA_0,   CUDA_0  }},
  {"cudaBindTexture",                                         {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaBindTexture2D",                                       {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaBindTextureToArray",                                  {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaBindTextureToMipmappedArray",                         {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaGetTextureAlignmentOffset",                           {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaGetTextureReference",                                 {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaUnbindTexture",                                       {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaBindSurfaceToArray",                                  {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaGetSurfaceReference",                                 {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cuTexObjectGetTextureDesc",                               {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaCreateSurfaceObject",                                 {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaDestroySurfaceObject",                                {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaGetSurfaceObjectResourceDesc",                        {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaDriverGetVersion",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaRuntimeGetVersion",                                   {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaGraphAddChildGraphNode",                              {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddDependencies",                                {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddEmptyNode",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddHostNode",                                    {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddKernelNode",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemcpyNode",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemsetNode",                                  {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphChildGraphNodeGetGraph",                         {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphClone",                                          {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphCreate",                                         {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphDestroy",                                        {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphDestroyNode",                                    {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecDestroy",                                    {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphGetEdges",                                       {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphGetNodes",                                       {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphGetRootNodes",                                   {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphHostNodeGetParams",                              {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphHostNodeSetParams",                              {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphInstantiate",                                    {CUDA_100, CUDA_0,   CUDA_0  }},
  {"cudaGraphKernelNodeCopyAttributes",                       {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphKernelNodeGetAttribute",                         {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphKernelNodeSetAttribute",                         {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecKernelNodeSetParams",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecMemcpyNodeSetParams",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecMemsetNodeSetParams",                        {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecHostNodeSetParams",                          {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecUpdate",                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphKernelNodeGetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphKernelNodeSetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphLaunch",                                         {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemcpyNodeGetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemcpyNodeSetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemsetNodeGetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemsetNodeSetParams",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphNodeFindInClone",                                {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphNodeGetDependencies",                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphNodeGetDependentNodes",                          {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphNodeGetType",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGraphRemoveDependencies",                             {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaGetFuncBySymbol",                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cudaProfilerInitialize",                                  {CUDA_0,   CUDA_110, CUDA_0  }},
  {"cudaConfigureCall",                                       {CUDA_0,   CUDA_0,   CUDA_101}},
  {"cudaLaunch",                                              {CUDA_0,   CUDA_0,   CUDA_101}},
  {"cudaSetupArgument",                                       {CUDA_0,   CUDA_0,   CUDA_101}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_RUNTIME_API_SECTION_MAP {
  {1, "Device Management"},
  {2, "Thread Management [DEPRECATED]"},
  {3, "Error Handling"},
  {4, "Stream Management"},
  {5, "Event Management"},
  {6, "External Resource Interoperability"},
  {7, "Execution Control"},
  {8, "Occupancy"},
  {9, "Memory Management"},
  {10, "Memory Management [DEPRECATED]"},
  {11, "Unified Addressing"},
  {12, "Peer Device Memory Access"},
  {13, "OpenGL Interoperability"},
  {14, "OpenGL Interoperability [DEPRECATED]"},
  {15, "Direct3D 9 Interoperability"},
  {16, "Direct3D 9 Interoperability [DEPRECATED]"},
  {17, "Direct3D 10 Interoperability"},
  {18, "Direct3D 10 Interoperability [DEPRECATED]"},
  {19, "Direct3D 11 Interoperability"},
  {20, "Direct3D 11 Interoperability [DEPRECATED]"},
  {21, "VDPAU Interoperability"},
  {22, "EGL Interoperability"},
  {23, "Graphics Interoperability"},
  {24, "Texture Reference Management [DEPRECATED]"},
  {25, "Surface Reference Management [DEPRECATED]"},
  {26, "Texture Object Management"},
  {27, "Surface Object Management"},
  {28, "Version Management"},
  {29, "Graph Management"},
  {30, "C++ API Routines"},
  {31, "Interactions with the CUDA Driver API"},
  {32, "Profiler Control [DEPRECATED]"},
  {33, "Profiler Control"},
  {34, "Data types used by CUDA Runtime"},
  {35, "Execution Control [REMOVED]"},
};
