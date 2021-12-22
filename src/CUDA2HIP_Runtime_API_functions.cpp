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
  // cuFlushGPUDirectRDMAWrites
  {"cudaDeviceFlushGPUDirectRDMAWrites",                      {"hipDeviceFlushGPUDirectRDMAWrites",                      "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
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
  {"cudaDeviceGetP2PAttribute",                               {"hipDeviceGetP2PAttribute",                               "", CONV_DEVICE, API_RUNTIME, 1}},
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
  {"cudaDeviceSetLimit",                                      {"hipDeviceSetLimit",                                      "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
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
  {"cudaGetDeviceFlags",                                      {"hipGetDeviceFlags",                                      "", CONV_DEVICE, API_RUNTIME, 1}},
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
  // cuDeviceGetTexture1DLinearMaxWidth
  {"cudaDeviceGetTexture1DLinearMaxWidth",                    {"hipDeviceGetTexture1DLinearMaxWidth",                    "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
  // cuDeviceGetDefaultMemPool
  {"cudaDeviceGetDefaultMemPool",                             {"hipDeviceGetDefaultMemPool",                             "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
  // cuDeviceSetMemPool
  {"cudaDeviceSetMemPool",                                    {"hipDeviceSetMemPool",                                    "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},
  // cuDeviceGetMemPool
  {"cudaDeviceGetMemPool",                                    {"hipDeviceGetMemPool",                                    "", CONV_DEVICE, API_RUNTIME, 1, HIP_UNSUPPORTED}},

  // 2. Thread Management [DEPRECATED]
  // no analogue
  {"cudaThreadExit",                                          {"hipDeviceReset",                                         "", CONV_THREAD, API_RUNTIME, 2, CUDA_DEPRECATED}},
  // no analogue
  {"cudaThreadGetCacheConfig",                                {"hipDeviceGetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, 2, CUDA_DEPRECATED}},
  // no analogue
  {"cudaThreadGetLimit",                                      {"hipThreadGetLimit",                                      "", CONV_THREAD, API_RUNTIME, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaThreadSetCacheConfig",                                {"hipDeviceSetCacheConfig",                                "", CONV_THREAD, API_RUNTIME, 2, CUDA_DEPRECATED}},
  // no analogue
  {"cudaThreadSetLimit",                                      {"hipThreadSetLimit",                                      "", CONV_THREAD, API_RUNTIME, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuCtxSynchronize
  {"cudaThreadSynchronize",                                   {"hipDeviceSynchronize",                                   "", CONV_THREAD, API_RUNTIME, 2, CUDA_DEPRECATED}},

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
  {"cudaStreamAttachMemAsync",                                {"hipStreamAttachMemAsync",                                "", CONV_STREAM, API_RUNTIME, 4}},
  // cuStreamBeginCapture
  {"cudaStreamBeginCapture",                                  {"hipStreamBeginCapture",                                  "", CONV_STREAM, API_RUNTIME, 4}},
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
  {"cudaStreamEndCapture",                                    {"hipStreamEndCapture",                                    "", CONV_STREAM, API_RUNTIME, 4}},
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
  // cuEventRecordWithFlags
  {"cudaEventRecordWithFlags",                                {"hipEventRecordWithFlags",                                "", CONV_EVENT, API_RUNTIME, 5, HIP_UNSUPPORTED}},

  // 6. External Resource Interoperability
  // cuDestroyExternalMemory
  {"cudaDestroyExternalMemory",                               {"hipDestroyExternalMemory",                               "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuDestroyExternalSemaphore
  {"cudaDestroyExternalSemaphore",                            {"hipDestroyExternalSemaphore",                            "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuExternalMemoryGetMappedBuffer
  {"cudaExternalMemoryGetMappedBuffer",                       {"hipExternalMemoryGetMappedBuffer",                       "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuExternalMemoryGetMappedMipmappedArray
  {"cudaExternalMemoryGetMappedMipmappedArray",               {"hipExternalMemoryGetMappedMipmappedArray",               "", CONV_EXT_RES, API_RUNTIME, 6, HIP_UNSUPPORTED}},
  // cuImportExternalMemory
  {"cudaImportExternalMemory",                                {"hipImportExternalMemory",                                "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuImportExternalSemaphore
  {"cudaImportExternalSemaphore",                             {"hipImportExternalSemaphore",                             "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuSignalExternalSemaphoresAsync
  {"cudaSignalExternalSemaphoresAsync",                       {"hipSignalExternalSemaphoresAsync",                       "", CONV_EXT_RES, API_RUNTIME, 6}},
  // cuWaitExternalSemaphoresAsync
  {"cudaWaitExternalSemaphoresAsync",                         {"hipWaitExternalSemaphoresAsync",                         "", CONV_EXT_RES, API_RUNTIME, 6}},

  // 7. Execution Control
  // no analogue
  {"cudaFuncGetAttributes",                                   {"hipFuncGetAttributes",                                   "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuFuncSetAttribute due to different signatures
  {"cudaFuncSetAttribute",                                    {"hipFuncSetAttribute",                                    "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuFuncSetCacheConfig due to different signatures
  {"cudaFuncSetCacheConfig",                                  {"hipFuncSetCacheConfig",                                  "", CONV_DEVICE, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuFuncSetSharedMemConfig due to different signatures
  {"cudaFuncSetSharedMemConfig",                              {"hipFuncSetSharedMemConfig",                              "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  {"cudaGetParameterBuffer",                                  {"hipGetParameterBuffer",                                  "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetParameterBufferV2",                                {"hipGetParameterBufferV2",                                "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernel due to different signatures
  {"cudaLaunchCooperativeKernel",                             {"hipLaunchCooperativeKernel",                             "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernelMultiDevice due to different signatures
  {"cudaLaunchCooperativeKernelMultiDevice",                  {"hipLaunchCooperativeKernelMultiDevice",                  "", CONV_EXECUTION, API_RUNTIME, 7, CUDA_DEPRECATED}},
  // cuLaunchHostFunc
  {"cudaLaunchHostFunc",                                      {"hipLaunchHostFunc",                                      "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchKernel due to different signatures
  {"cudaLaunchKernel",                                        {"hipLaunchKernel",                                        "", CONV_EXECUTION, API_RUNTIME, 7}},
  // no analogue
  {"cudaSetDoubleForDevice",                                  {"hipSetDoubleForDevice",                                  "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaSetDoubleForHost",                                    {"hipSetDoubleForHost",                                    "", CONV_EXECUTION, API_RUNTIME, 7, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

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
  {"cudaOccupancyMaxPotentialBlockSizeWithFlags",             {"hipOccupancyMaxPotentialBlockSizeWithFlags",             "", CONV_OCCUPANCY, API_RUNTIME, 8}},
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
  {"cudaFreeMipmappedArray",                                  {"hipFreeMipmappedArray",                                  "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayGetLevel due to different signatures
  {"cudaGetMipmappedArrayLevel",                              {"hipGetMipmappedArrayLevel",                              "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaGetSymbolAddress",                                    {"hipGetSymbolAddress",                                    "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  {"cudaGetSymbolSize",                                       {"hipGetSymbolSize",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemHostAlloc
  {"cudaHostAlloc",                                           {"hipHostAlloc",                                           "", CONV_MEMORY, API_RUNTIME, 9}},
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
  {"cudaMallocHost",                                          {"hipMallocHost",                                          "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemAllocManaged
  {"cudaMallocManaged",                                       {"hipMallocManaged",                                       "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayCreate due to different signatures
  {"cudaMallocMipmappedArray",                                {"hipMallocMipmappedArray",                                "", CONV_MEMORY, API_RUNTIME, 9}},
  // no analogue
  // NOTE: Not equal to cuMemAllocPitch due to different signatures
  {"cudaMallocPitch",                                         {"hipMallocPitch",                                         "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemAdvise
  {"cudaMemAdvise",                                           {"hipMemAdvise",                                           "", CONV_MEMORY, API_RUNTIME, 9}},
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
  {"cudaMemcpy2DToArrayAsync",                                {"hipMemcpy2DToArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, 9}},
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
  {"cudaMemPrefetchAsync",                                    {"hipMemPrefetchAsync",                                    "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemRangeGetAttribute
  {"cudaMemRangeGetAttribute",                                {"hipMemRangeGetAttribute",                                "", CONV_MEMORY, API_RUNTIME, 9}},
  // cuMemRangeGetAttributes
  {"cudaMemRangeGetAttributes",                               {"hipMemRangeGetAttributes",                               "", CONV_MEMORY, API_RUNTIME, 9}},
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
  // cuArrayGetSparseProperties
  {"cudaArrayGetSparseProperties",                            {"hipArrayGetSparseProperties",                            "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},
  // cuArrayGetPlane
  {"cudaArrayGetPlane",                                       {"hipArrayGetPlane",                                       "", CONV_MEMORY, API_RUNTIME, 9, HIP_UNSUPPORTED}},

  // 10. Memory Management [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuMemcpyAtoA due to different signatures
  {"cudaMemcpyArrayToArray",                                  {"hipMemcpyArrayToArray",                                  "", CONV_MEMORY, API_RUNTIME, 10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaMemcpyFromArray",                                     {"hipMemcpyFromArray",                                     "", CONV_MEMORY, API_RUNTIME, 10, DEPRECATED}},
  // no analogue
  {"cudaMemcpyFromArrayAsync",                                {"hipMemcpyFromArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, 10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaMemcpyToArray",                                       {"hipMemcpyToArray",                                       "", CONV_MEMORY, API_RUNTIME, 10, DEPRECATED}},
  // no analogue
  {"cudaMemcpyToArrayAsync",                                  {"hipMemcpyToArrayAsync",                                  "", CONV_MEMORY, API_RUNTIME, 10, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 11. Stream Ordered Memory Allocator

  // no analogue
  {"cudaMallocAsync",                                         {"hipMallocAsync",                                         "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemFreeAsync?
  // TODO: double check cuMemFreeAsync
  {"cudaFreeAsync",                                           {"hipFreeAsync",                                           "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaMallocFromPoolAsync",                                 {"hipMallocFromPoolAsync",                                 "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolTrimTo
  {"cudaMemPoolTrimTo",                                       {"hipMemPoolTrimTo",                                       "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolSetAttribute
  {"cudaMemPoolSetAttribute",                                 {"hipMemPoolSetAttribute",                                 "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolGetAttribute
  {"cudaMemPoolGetAttribute",                                 {"hipMemPoolGetAttribute",                                 "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolSetAccess
  {"cudaMemPoolSetAccess",                                    {"hipMemPoolSetAccess",                                    "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolGetAccess
  {"cudaMemPoolGetAccess",                                    {"hipMemPoolGetAccess",                                    "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolCreate
  {"cudaMemPoolCreate",                                       {"hipMemPoolCreate",                                       "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolDestroy
  {"cudaMemPoolDestroy",                                      {"hipMemPoolDestroy",                                      "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolExportToShareableHandle
  {"cudaMemPoolExportToShareableHandle",                      {"hipMemPoolExportToShareableHandle",                      "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolImportFromShareableHandle
  {"cudaMemPoolImportFromShareableHandle",                    {"hipMemPoolImportFromShareableHandle",                    "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolExportPointer
  {"cudaMemPoolExportPointer",                                {"hipMemPoolExportPointer",                                "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},
  // cuMemPoolImportPointer
  {"cudaMemPoolImportPointer",                                {"hipMemPoolImportPointer",                                "", CONV_MEMORY, API_RUNTIME, 11, HIP_UNSUPPORTED}},

  // 12. Unified Addressing
  // no analogue
  // NOTE: Not equal to cuPointerGetAttributes due to different signatures
  {"cudaPointerGetAttributes",                                {"hipPointerGetAttributes",                                "", CONV_ADDRESSING, API_RUNTIME, 12}},

  // 13. Peer Device Memory Access
  // cuDeviceCanAccessPeer
  {"cudaDeviceCanAccessPeer",                                 {"hipDeviceCanAccessPeer",                                 "", CONV_PEER, API_RUNTIME, 13}},
  // no analogue
  // NOTE: Not equal to cuCtxDisablePeerAccess due to different signatures
  {"cudaDeviceDisablePeerAccess",                             {"hipDeviceDisablePeerAccess",                             "", CONV_PEER, API_RUNTIME, 13}},
  // no analogue
  // NOTE: Not equal to cuCtxEnablePeerAccess due to different signatures
  {"cudaDeviceEnablePeerAccess",                              {"hipDeviceEnablePeerAccess",                              "", CONV_PEER, API_RUNTIME, 13}},

  // 14. OpenGL Interoperability
  // cuGLGetDevices
  {"cudaGLGetDevices",                                        {"hipGLGetDevices",                                        "", CONV_OPENGL, API_RUNTIME, 14}},
  // cuGraphicsGLRegisterBuffer
  {"cudaGraphicsGLRegisterBuffer",                            {"hipGraphicsGLRegisterBuffer",                            "", CONV_OPENGL, API_RUNTIME, 14}},
  // cuGraphicsGLRegisterImage
  {"cudaGraphicsGLRegisterImage",                             {"hipGraphicsGLRegisterImage",                             "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED}},
  // cuWGLGetDevice
  {"cudaWGLGetDevice",                                        {"hipWGLGetDevice",                                        "", CONV_OPENGL, API_RUNTIME, 14, HIP_UNSUPPORTED}},

  // 15. OpenGL Interoperability [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObject due to different signatures
  {"cudaGLMapBufferObject",                                   {"hipGLMapBufferObject",                                   "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObjectAsync due to different signatures
  {"cudaGLMapBufferObjectAsync",                              {"hipGLMapBufferObjectAsync",                              "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuGLRegisterBufferObject
  {"cudaGLRegisterBufferObject",                              {"hipGLRegisterBufferObject",                              "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuGLSetBufferObjectMapFlags
  {"cudaGLSetBufferObjectMapFlags",                           {"hipGLSetBufferObjectMapFlags",                           "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaGLSetGLDevice",                                       {"hipGLSetGLDevice",                                       "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuGLUnmapBufferObject
  {"cudaGLUnmapBufferObject",                                 {"hipGLUnmapBufferObject",                                 "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuGLUnmapBufferObjectAsync
  {"cudaGLUnmapBufferObjectAsync",                            {"hipGLUnmapBufferObjectAsync",                            "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuGLUnregisterBufferObject
  {"cudaGLUnregisterBufferObject",                            {"hipGLUnregisterBufferObject",                            "", CONV_OPENGL, API_RUNTIME, 15, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 16. Direct3D 9 Interoperability
  // cuD3D9GetDevice
  {"cudaD3D9GetDevice",                                       {"hipD3D9GetDevice",                                       "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},
  // cuD3D9GetDevices
  {"cudaD3D9GetDevices",                                      {"hipD3D9GetDevices",                                      "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},
  // cuD3D9GetDirect3DDevice
  {"cudaD3D9GetDirect3DDevice",                               {"hipD3D9GetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaD3D9SetDirect3DDevice",                               {"hipD3D9SetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},
  // cuGraphicsD3D9RegisterResource
  {"cudaGraphicsD3D9RegisterResource",                        {"hipGraphicsD3D9RegisterResource",                        "", CONV_D3D9, API_RUNTIME, 16, HIP_UNSUPPORTED}},

  // 17. Direct3D 9 Interoperability[DEPRECATED]
  // cuD3D9MapResources
  {"cudaD3D9MapResources",                                    {"hipD3D9MapResources",                                    "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9RegisterResource
  // NOTE: cudaD3D9RegisterResource is not marked as deprecated function even in CUDA 11.0
  {"cudaD3D9RegisterResource",                                {"hipD3D9RegisterResource",                                "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetMappedArray
  {"cudaD3D9ResourceGetMappedArray",                          {"hipD3D9ResourceGetMappedArray",                          "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9ResourceGetMappedPitch
  {"cudaD3D9ResourceGetMappedPitch",                          {"hipD3D9ResourceGetMappedPitch",                          "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9ResourceGetMappedPointer
  {"cudaD3D9ResourceGetMappedPointer",                        {"hipD3D9ResourceGetMappedPointer",                        "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9ResourceGetMappedSize
  {"cudaD3D9ResourceGetMappedSize",                           {"hipD3D9ResourceGetMappedSize",                           "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9ResourceGetSurfaceDimensions
  {"cudaD3D9ResourceGetSurfaceDimensions",                    {"hipD3D9ResourceGetSurfaceDimensions",                    "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9ResourceSetMapFlags
  {"cudaD3D9ResourceSetMapFlags",                             {"hipD3D9ResourceSetMapFlags",                             "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9UnmapResources
  {"cudaD3D9UnmapResources",                                  {"hipD3D9UnmapResources",                                  "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D9UnregisterResource
  {"cudaD3D9UnregisterResource",                              {"hipD3D9UnregisterResource",                              "", CONV_D3D9, API_RUNTIME, 17, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 18. Direct3D 10 Interoperability
  // cuD3D10GetDevice
  {"cudaD3D10GetDevice",                                      {"hipD3D10GetDevice",                                      "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED}},
  // cuD3D10GetDevices
  {"cudaD3D10GetDevices",                                     {"hipD3D10GetDevices",                                     "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED}},
  // cuGraphicsD3D10RegisterResource
  {"cudaGraphicsD3D10RegisterResource",                       {"hipGraphicsD3D10RegisterResource",                       "", CONV_D3D10, API_RUNTIME, 18, HIP_UNSUPPORTED}},

  // 19. Direct3D 10 Interoperability [DEPRECATED]
  // cuD3D10GetDirect3DDevice
  {"cudaD3D10GetDirect3DDevice",                              {"hipD3D10GetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10MapResources
  {"cudaD3D10MapResources",                                   {"hipD3D10MapResources",                                   "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10RegisterResource
  {"cudaD3D10RegisterResource",                               {"hipD3D10RegisterResource",                               "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceGetMappedArray
  {"cudaD3D10ResourceGetMappedArray",                         {"hipD3D10ResourceGetMappedArray",                         "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceGetMappedPitch
  {"cudaD3D10ResourceGetMappedPitch",                         {"hipD3D10ResourceGetMappedPitch",                         "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceGetMappedPointer
  {"cudaD3D10ResourceGetMappedPointer",                       {"hipD3D10ResourceGetMappedPointer",                       "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceGetMappedSize
  {"cudaD3D10ResourceGetMappedSize",                          {"hipD3D10ResourceGetMappedSize",                          "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceGetSurfaceDimensions
  {"cudaD3D10ResourceGetSurfaceDimensions",                   {"hipD3D10ResourceGetSurfaceDimensions",                   "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10ResourceSetMapFlags
  {"cudaD3D10ResourceSetMapFlags",                            {"hipD3D10ResourceSetMapFlags",                            "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaD3D10SetDirect3DDevice",                              {"hipD3D10SetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10UnmapResources
  {"cudaD3D10UnmapResources",                                 {"hipD3D10UnmapResources",                                 "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // cuD3D10UnregisterResource
  {"cudaD3D10UnregisterResource",                             {"hipD3D10UnregisterResource",                             "", CONV_D3D10, API_RUNTIME, 19, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 20. Direct3D 11 Interoperability
  // cuD3D11GetDevice
  {"cudaD3D11GetDevice",                                      {"hipD3D11GetDevice",                                      "", CONV_D3D11, API_RUNTIME, 20, HIP_UNSUPPORTED}},
  // cuD3D11GetDevices
  {"cudaD3D11GetDevices",                                     {"hipD3D11GetDevices",                                     "", CONV_D3D11, API_RUNTIME, 20, HIP_UNSUPPORTED}},
  // cuGraphicsD3D11RegisterResource
  {"cudaGraphicsD3D11RegisterResource",                       {"hipGraphicsD3D11RegisterResource",                       "", CONV_D3D11, API_RUNTIME, 20, HIP_UNSUPPORTED}},

  // 21. Direct3D 11 Interoperability [DEPRECATED]
  // cuD3D11GetDirect3DDevice
  {"cudaD3D11GetDirect3DDevice",                              {"hipD3D11GetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, 21, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // no analogue
  {"cudaD3D11SetDirect3DDevice",                              {"hipD3D11SetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, 21, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 22. VDPAU Interoperability
  // cuGraphicsVDPAURegisterOutputSurface
  {"cudaGraphicsVDPAURegisterOutputSurface",                  {"hipGraphicsVDPAURegisterOutputSurface",                  "", CONV_VDPAU, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuGraphicsVDPAURegisterVideoSurface
  {"cudaGraphicsVDPAURegisterVideoSurface",                   {"hipGraphicsVDPAURegisterVideoSurface",                   "", CONV_VDPAU, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // cuVDPAUGetDevice
  {"cudaVDPAUGetDevice",                                      {"hipVDPAUGetDevice",                                      "", CONV_VDPAU, API_RUNTIME, 22, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaVDPAUSetVDPAUDevice",                                 {"hipVDPAUSetDevice",                                      "", CONV_VDPAU, API_RUNTIME, 22, HIP_UNSUPPORTED}},

  // 23. EGL Interoperability
  // cuEGLStreamConsumerAcquireFrame
  {"cudaEGLStreamConsumerAcquireFrame",                       {"hipEGLStreamConsumerAcquireFrame",                       "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnect
  {"cudaEGLStreamConsumerConnect",                            {"hipEGLStreamConsumerConnect",                            "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnectWithFlags
  {"cudaEGLStreamConsumerConnectWithFlags",                   {"hipEGLStreamConsumerConnectWithFlags",                   "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerDisconnect
  {"cudaEGLStreamConsumerDisconnect",                         {"hipEGLStreamConsumerDisconnect",                         "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerReleaseFrame
  {"cudaEGLStreamConsumerReleaseFrame",                       {"hipEGLStreamConsumerReleaseFrame",                       "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerConnect
  {"cudaEGLStreamProducerConnect",                            {"hipEGLStreamProducerConnect",                            "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerDisconnect
  {"cudaEGLStreamProducerDisconnect",                         {"hipEGLStreamProducerDisconnect",                         "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerPresentFrame
  {"cudaEGLStreamProducerPresentFrame",                       {"hipEGLStreamProducerPresentFrame",                       "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerReturnFrame
  {"cudaEGLStreamProducerReturnFrame",                        {"hipEGLStreamProducerReturnFrame",                        "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuEventCreateFromEGLSync
  {"cudaEventCreateFromEGLSync",                              {"hipEventCreateFromEGLSync",                              "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsEGLRegisterImage
  {"cudaGraphicsEGLRegisterImage",                            {"hipGraphicsEGLRegisterImage",                            "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedEglFrame
  {"cudaGraphicsResourceGetMappedEglFrame",                   {"hipGraphicsResourceGetMappedEglFrame",                   "", CONV_EGL, API_RUNTIME, 23, HIP_UNSUPPORTED}},

  // 24. Graphics Interoperability
  // cuGraphicsMapResources
  {"cudaGraphicsMapResources",                                {"hipGraphicsMapResources",                                "", CONV_GRAPHICS, API_RUNTIME, 24}},
  // cuGraphicsResourceGetMappedMipmappedArray
  {"cudaGraphicsResourceGetMappedMipmappedArray",             {"hipGraphicsResourceGetMappedMipmappedArray",             "", CONV_GRAPHICS, API_RUNTIME, 24, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedPointer
  {"cudaGraphicsResourceGetMappedPointer",                    {"hipGraphicsResourceGetMappedPointer",                    "", CONV_GRAPHICS, API_RUNTIME, 24}},
  // cuGraphicsResourceSetMapFlags
  {"cudaGraphicsResourceSetMapFlags",                         {"hipGraphicsResourceSetMapFlags",                         "", CONV_GRAPHICS, API_RUNTIME, 24, HIP_UNSUPPORTED}},
  // cuGraphicsSubResourceGetMappedArray
  {"cudaGraphicsSubResourceGetMappedArray",                   {"hipGraphicsSubResourceGetMappedArray",                   "", CONV_GRAPHICS, API_RUNTIME, 24, HIP_UNSUPPORTED}},
  // cuGraphicsUnmapResources
  {"cudaGraphicsUnmapResources",                              {"hipGraphicsUnmapResources",                              "", CONV_GRAPHICS, API_RUNTIME, 24}},
  // cuGraphicsUnregisterResource
  {"cudaGraphicsUnregisterResource",                          {"hipGraphicsUnregisterResource",                          "", CONV_GRAPHICS, API_RUNTIME, 24}},

  // 25. Texture Reference Management [DEPRECATED]
  // no analogue
  {"cudaBindTexture",                                         {"hipBindTexture",                                         "", CONV_TEXTURE, API_RUNTIME, 25, DEPRECATED}},
  // no analogue
  {"cudaBindTexture2D",                                       {"hipBindTexture2D",                                       "", CONV_TEXTURE, API_RUNTIME, 25, DEPRECATED}},
  // no analogue
  {"cudaBindTextureToArray",                                  {"hipBindTextureToArray",                                  "", CONV_TEXTURE, API_RUNTIME, 25, DEPRECATED}},
  // no analogue
  // NOTE: Unsupported yet on NVCC path
  {"cudaBindTextureToMipmappedArray",                         {"hipBindTextureToMipmappedArray",                         "", CONV_TEXTURE, API_RUNTIME, 25, CUDA_DEPRECATED}},
  // no analogue
  {"cudaCreateChannelDesc",                                   {"hipCreateChannelDesc",                                   "", CONV_TEXTURE, API_RUNTIME, 25}},
  // no analogue
  {"cudaGetChannelDesc",                                      {"hipGetChannelDesc",                                      "", CONV_TEXTURE, API_RUNTIME, 25}},
  // no analogue
  {"cudaGetTextureAlignmentOffset",                           {"hipGetTextureAlignmentOffset",                           "", CONV_TEXTURE, API_RUNTIME, 25, DEPRECATED}},
  // TODO: double check cuModuleGetTexRef
  // NOTE: Unsupported yet on NVCC path
  {"cudaGetTextureReference",                                 {"hipGetTextureReference",                                 "", CONV_TEXTURE, API_RUNTIME, 25, CUDA_DEPRECATED}},
  // no analogue
  {"cudaUnbindTexture",                                       {"hipUnbindTexture",                                       "", CONV_TEXTURE, API_RUNTIME, 25, DEPRECATED}},

  // 26. Surface Reference Management [DEPRECATED]
  // no analogue
  {"cudaBindSurfaceToArray",                                  {"hipBindSurfaceToArray",                                  "", CONV_SURFACE, API_RUNTIME, 26, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  // TODO: double check cuModuleGetSurfRef
  {"cudaGetSurfaceReference",                                 {"hipGetSurfaceReference",                                 "", CONV_SURFACE, API_RUNTIME, 26, HIP_UNSUPPORTED | CUDA_DEPRECATED}},

  // 27. Texture Object Management
  // no analogue
  // NOTE: Not equal to cuTexObjectCreate due to different signatures
  {"cudaCreateTextureObject",                                 {"hipCreateTextureObject",                                 "", CONV_TEXTURE, API_RUNTIME, 27}},
  // cuTexObjectDestroy
  {"cudaDestroyTextureObject",                                {"hipDestroyTextureObject",                                "", CONV_TEXTURE, API_RUNTIME, 27}},
  // no analogue
  // NOTE: Not equal to cuTexObjectGetResourceDesc due to different signatures
  {"cudaGetTextureObjectResourceDesc",                        {"hipGetTextureObjectResourceDesc",                        "", CONV_TEXTURE, API_RUNTIME, 27}},
  // cuTexObjectGetResourceViewDesc
  {"cudaGetTextureObjectResourceViewDesc",                    {"hipGetTextureObjectResourceViewDesc",                    "", CONV_TEXTURE, API_RUNTIME, 27}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                               {"hipGetTextureObjectTextureDesc",                         "", CONV_TEXTURE, API_RUNTIME, 27}},

  // 28. Surface Object Management
  // no analogue
  // NOTE: Not equal to cuSurfObjectCreate due to different signatures
  {"cudaCreateSurfaceObject",                                 {"hipCreateSurfaceObject",                                 "", CONV_SURFACE, API_RUNTIME, 28}},
  // cuSurfObjectDestroy
  {"cudaDestroySurfaceObject",                                {"hipDestroySurfaceObject",                                "", CONV_SURFACE, API_RUNTIME, 28}},
  // no analogue
  // NOTE: Not equal to cuSurfObjectGetResourceDesc due to different signatures
  {"cudaGetSurfaceObjectResourceDesc",                        {"hipGetSurfaceObjectResourceDesc",                        "", CONV_SURFACE, API_RUNTIME, 28, HIP_UNSUPPORTED}},

  // 29. Version Management
  // cuDriverGetVersion
  {"cudaDriverGetVersion",                                    {"hipDriverGetVersion",                                    "", CONV_VERSION, API_RUNTIME, 29}},
  // no analogue
  {"cudaRuntimeGetVersion",                                   {"hipRuntimeGetVersion",                                   "", CONV_VERSION, API_RUNTIME, 29}},

  // 30. Graph Management
  // cuGraphAddChildGraphNode
  {"cudaGraphAddChildGraphNode",                              {"hipGraphAddChildGraphNode",                              "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddDependencies
  {"cudaGraphAddDependencies",                                {"hipGraphAddDependencies",                                "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphAddEmptyNode
  {"cudaGraphAddEmptyNode",                                   {"hipGraphAddEmptyNode",                                   "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphAddHostNode
  {"cudaGraphAddHostNode",                                    {"hipGraphAddHostNode",                                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddKernelNode
  {"cudaGraphAddKernelNode",                                  {"hipGraphAddKernelNode",                                  "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphAddMemcpyNode
  {"cudaGraphAddMemcpyNode",                                  {"hipGraphAddMemcpyNode",                                  "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphAddMemsetNode
  {"cudaGraphAddMemsetNode",                                  {"hipGraphAddMemsetNode",                                  "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphChildGraphNodeGetGraph
  {"cudaGraphChildGraphNodeGetGraph",                         {"hipGraphChildGraphNodeGetGraph",                         "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphClone
  {"cudaGraphClone",                                          {"hipGraphClone",                                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphCreate
  {"cudaGraphCreate",                                         {"hipGraphCreate",                                         "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphDebugDotPrint
  {"cudaGraphDebugDotPrint",                                  {"hipGraphDebugDotPrint",                                  "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphDestroy
  {"cudaGraphDestroy",                                        {"hipGraphDestroy",                                        "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphDestroyNode
  {"cudaGraphDestroyNode",                                    {"hipGraphDestroyNode",                                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecDestroy
  {"cudaGraphExecDestroy",                                    {"hipGraphExecDestroy",                                    "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphGetEdges
  {"cudaGraphGetEdges",                                       {"hipGraphGetEdges",                                       "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphGetNodes
  {"cudaGraphGetNodes",                                       {"hipGraphGetNodes",                                       "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphGetRootNodes
  {"cudaGraphGetRootNodes",                                   {"hipGraphGetRootNodes",                                   "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphHostNodeGetParams
  {"cudaGraphHostNodeGetParams",                              {"hipGraphHostNodeGetParams",                              "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphHostNodeSetParams
  {"cudaGraphHostNodeSetParams",                              {"hipGraphHostNodeSetParams",                              "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphInstantiate
  {"cudaGraphInstantiate",                                    {"hipGraphInstantiate",                                    "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphKernelNodeCopyAttributes
  {"cudaGraphKernelNodeCopyAttributes",                       {"hipGraphKernelNodeCopyAttributes",                       "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeGetAttribute
  {"cudaGraphKernelNodeGetAttribute",                         {"hipGraphKernelNodeGetAttribute",                         "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeSetAttribute
  {"cudaGraphKernelNodeSetAttribute",                         {"hipGraphKernelNodeSetAttribute",                         "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecKernelNodeSetParams
  {"cudaGraphExecKernelNodeSetParams",                        {"hipGraphExecKernelNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphExecMemcpyNodeSetParams
  {"cudaGraphExecMemcpyNodeSetParams",                        {"hipGraphExecMemcpyNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecMemsetNodeSetParams
  {"cudaGraphExecMemsetNodeSetParams",                        {"hipGraphExecMemsetNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecHostNodeSetParams
  {"cudaGraphExecHostNodeSetParams",                          {"hipGraphExecHostNodeSetParams",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecUpdate
  {"cudaGraphExecUpdate",                                     {"hipGraphExecUpdate",                                     "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeGetParams
  {"cudaGraphKernelNodeGetParams",                            {"hipGraphKernelNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphKernelNodeSetParams
  {"cudaGraphKernelNodeSetParams",                            {"hipGraphKernelNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphLaunch
  {"cudaGraphLaunch",                                         {"hipGraphLaunch",                                         "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphMemcpyNodeGetParams
  {"cudaGraphMemcpyNodeGetParams",                            {"hipGraphMemcpyNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphMemcpyNodeSetParams
  {"cudaGraphMemcpyNodeSetParams",                            {"hipGraphMemcpyNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphMemsetNodeGetParams
  {"cudaGraphMemsetNodeGetParams",                            {"hipGraphMemsetNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphMemsetNodeSetParams
  {"cudaGraphMemsetNodeSetParams",                            {"hipGraphMemsetNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, 30}},
  // cuGraphNodeFindInClone
  {"cudaGraphNodeFindInClone",                                {"hipGraphNodeFindInClone",                                "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependencies
  {"cudaGraphNodeGetDependencies",                            {"hipGraphNodeGetDependencies",                            "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependentNodes
  {"cudaGraphNodeGetDependentNodes",                          {"hipGraphNodeGetDependentNodes",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphNodeGetType
  {"cudaGraphNodeGetType",                                    {"hipGraphNodeGetType",                                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphRemoveDependencies
  {"cudaGraphRemoveDependencies",                             {"hipGraphRemoveDependencies",                             "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphAddMemcpyNodeToSymbol",                          {"hipGraphAddMemcpyNodeToSymbol",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphAddMemcpyNodeFromSymbol",                        {"hipGraphAddMemcpyNodeFromSymbol",                        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphAddMemcpyNode1D",                                {"hipGraphAddMemcpyNode1D",                                "", CONV_GRAPH, API_RUNTIME, 30}},
  // no analogue
  {"cudaGraphMemcpyNodeSetParamsToSymbol",                    {"hipGraphMemcpyNodeSetParamsToSymbol",                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphMemcpyNodeSetParamsFromSymbol",                  {"hipGraphMemcpyNodeSetParamsFromSymbol",                  "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphMemcpyNodeSetParams1D",                          {"hipGraphMemcpyNodeSetParams1D",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddEventRecordNode
  {"cudaGraphAddEventRecordNode",                             {"hipGraphAddEventRecordNode",                             "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphEventRecordNodeGetEvent
  {"cudaGraphEventRecordNodeGetEvent",                        {"hipGraphEventRecordNodeGetEvent",                        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphEventRecordNodeSetEvent
  {"cudaGraphEventRecordNodeSetEvent",                        {"hipGraphEventRecordNodeSetEvent",                        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddEventWaitNode
  {"cudaGraphAddEventWaitNode",                               {"hipGraphAddEventWaitNode",                               "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphEventWaitNodeGetEvent
  {"cudaGraphEventWaitNodeGetEvent",                          {"hipGraphEventWaitNodeGetEvent",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphEventWaitNodeSetEvent
  {"cudaGraphEventWaitNodeSetEvent",                          {"hipGraphEventWaitNodeSetEvent",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphExecMemcpyNodeSetParamsToSymbol",                {"hipGraphExecMemcpyNodeSetParamsToSymbol",                "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphExecMemcpyNodeSetParamsFromSymbol",              {"hipGraphExecMemcpyNodeSetParamsFromSymbol",              "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGraphExecMemcpyNodeSetParams1D",                      {"hipGraphExecMemcpyNodeSetParams1D",                      "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecChildGraphNodeSetParams
  {"cudaGraphExecChildGraphNodeSetParams",                    {"hipGraphExecChildGraphNodeSetParams",                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecEventRecordNodeSetEvent
  {"cudaGraphExecEventRecordNodeSetEvent",                    {"hipGraphExecEventRecordNodeSetEvent",                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecEventWaitNodeSetEvent
  {"cudaGraphExecEventWaitNodeSetEvent",                      {"hipGraphExecEventWaitNodeSetEvent",                      "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphUpload
  {"cudaGraphUpload",                                         {"hipGraphUpload",                                         "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddExternalSemaphoresSignalNode
  {"cudaGraphAddExternalSemaphoresSignalNode",                {"hipGraphAddExternalSemaphoresSignalNode",                "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExternalSemaphoresSignalNodeGetParams
  {"cudaGraphExternalSemaphoresSignalNodeGetParams",          {"hipGraphExternalSemaphoresSignalNodeGetParams",          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExternalSemaphoresSignalNodeSetParams
  {"cudaGraphExternalSemaphoresSignalNodeSetParams",          {"hipGraphExternalSemaphoresSignalNodeSetParams",          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddExternalSemaphoresWaitNode
  {"cudaGraphAddExternalSemaphoresWaitNode",                  {"hipGraphAddExternalSemaphoresWaitNode",                  "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExternalSemaphoresWaitNodeGetParams
  {"cudaGraphExternalSemaphoresWaitNodeGetParams",            {"hipGraphExternalSemaphoresWaitNodeGetParams",            "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExternalSemaphoresWaitNodeSetParams
  {"cudaGraphExternalSemaphoresWaitNodeSetParams",            {"hipGraphExternalSemaphoresWaitNodeSetParams",            "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecExternalSemaphoresSignalNodeSetParams
  {"cudaGraphExecExternalSemaphoresSignalNodeSetParams",      {"hipGraphExecExternalSemaphoresSignalNodeSetParams",      "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphExecExternalSemaphoresWaitNodeSetParams
  {"cudaGraphExecExternalSemaphoresWaitNodeSetParams",        {"hipGraphExecExternalSemaphoresWaitNodeSetParams",        "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuUserObjectCreate
  {"cudaUserObjectCreate",                                    {"hipUserObjectCreate",                                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuUserObjectRetain
  {"cudaUserObjectRetain",                                    {"hipUserObjectRetain",                                    "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuUserObjectRelease
  {"cudaUserObjectRelease",                                   {"hipUserObjectRelease",                                   "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphRetainUserObject
  {"cudaGraphRetainUserObject",                               {"hipGraphRetainUserObject",                               "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphReleaseUserObject
  {"cudaGraphReleaseUserObject",                              {"hipGraphReleaseUserObject",                              "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddMemAllocNode
  {"cudaGraphAddMemAllocNode",                                {"hipGraphAddMemAllocNode",                                "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphMemAllocNodeGetParams
  {"cudaGraphMemAllocNodeGetParams",                          {"hipGraphMemAllocNodeGetParams",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphAddMemFreeNode
  {"cudaGraphAddMemFreeNode",                                 {"hipGraphAddMemFreeNode",                                 "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphMemFreeNodeGetParams
  {"cudaGraphMemFreeNodeGetParams",                           {"hipGraphMemFreeNodeGetParams",                           "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuDeviceGraphMemTrim
  {"cudaDeviceGraphMemTrim",                                  {"hipDeviceGraphMemTrim",                                  "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuDeviceGetGraphMemAttribute
  {"cudaDeviceGetGraphMemAttribute",                          {"hipDeviceGetGraphMemAttribute",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuDeviceSetGraphMemAttribute
  {"cudaDeviceSetGraphMemAttribute",                          {"hipDeviceSetGraphMemAttribute",                          "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},
  // cuGraphInstantiateWithFlags
  {"cudaGraphInstantiateWithFlags",                           {"hipGraphInstantiateWithFlags",                           "", CONV_GRAPH, API_RUNTIME, 30, HIP_UNSUPPORTED}},

  // 31. Driver Entry Point Access
  // cuGetProcAddress
  {"cudaGetDriverEntryPoint",                                 {"hipGetProcAddress",                                      "", CONV_GRAPH, API_RUNTIME, 31, HIP_UNSUPPORTED}},

  // 32. C++ API Routines
  // TODO

  // 33. Interactions with the CUDA Driver API
  {"cudaGetFuncBySymbol",                                     {"hipGetFuncBySymbol",                                     "", CONV_INTERACTION, API_RUNTIME, 33, HIP_UNSUPPORTED}},

  // 34. Profiler Control [DEPRECATED]
  // cuProfilerInitialize
  {"cudaProfilerInitialize",                                  {"hipProfilerInitialize",                                  "", CONV_PROFILER, API_RUNTIME, 34, HIP_UNSUPPORTED}},

  // 35. Profiler Control
  // cuProfilerStart
  {"cudaProfilerStart",                                       {"hipProfilerStart",                                       "", CONV_PROFILER, API_RUNTIME, 35, HIP_DEPRECATED}},
  // cuProfilerStop
  {"cudaProfilerStop",                                        {"hipProfilerStop",                                        "", CONV_PROFILER, API_RUNTIME, 35, HIP_DEPRECATED}},

  // 36. Data types used by CUDA Runtime
  // NOTE: in a separate file

  // 37. Execution Control [REMOVED]
  // NOTE: Removed in CUDA 10.1
  // no analogue
  {"cudaConfigureCall",                                       {"hipConfigureCall",                                       "", CONV_EXECUTION, API_RUNTIME, 37, CUDA_REMOVED}},
  // no analogue
  // NOTE: Not equal to cuLaunch due to different signatures
  {"cudaLaunch",                                              {"hipLaunchByPtr",                                         "", CONV_EXECUTION, API_RUNTIME, 37, CUDA_REMOVED}},
  // no analogue
  {"cudaSetupArgument",                                       {"hipSetupArgument",                                       "", CONV_EXECUTION, API_RUNTIME, 37, CUDA_REMOVED}},
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
  {"cudaLaunchCooperativeKernelMultiDevice",                  {CUDA_90,  CUDA_113, CUDA_0  }},
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
  {"cudaDeviceGetTexture1DLinearMaxWidth",                    {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaEventRecordWithFlags",                                {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaArrayGetSparseProperties",                            {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemcpyNodeToSymbol",                          {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemcpyNodeFromSymbol",                        {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemcpyNode1D",                                {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemcpyNodeSetParamsToSymbol",                    {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemcpyNodeSetParamsFromSymbol",                  {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemcpyNodeSetParams1D",                          {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddEventRecordNode",                             {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphEventRecordNodeGetEvent",                        {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphEventRecordNodeSetEvent",                        {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddEventWaitNode",                               {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphEventWaitNodeGetEvent",                          {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphEventWaitNodeSetEvent",                          {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecMemcpyNodeSetParamsToSymbol",                {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecMemcpyNodeSetParamsFromSymbol",              {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecMemcpyNodeSetParams1D",                      {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecChildGraphNodeSetParams",                    {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecEventRecordNodeSetEvent",                    {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecEventWaitNodeSetEvent",                      {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaGraphUpload",                                         {CUDA_111, CUDA_0,   CUDA_0  }},
  {"cudaMallocAsync",                                         {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaFreeAsync",                                           {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMallocFromPoolAsync",                                 {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaDeviceGetDefaultMemPool",                             {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaDeviceSetMemPool",                                    {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaDeviceGetMemPool",                                    {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaArrayGetPlane",                                       {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolTrimTo",                                       {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolSetAttribute",                                 {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolGetAttribute",                                 {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolSetAccess",                                    {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolGetAccess",                                    {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolCreate",                                       {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolDestroy",                                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolExportToShareableHandle",                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolImportFromShareableHandle",                    {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolExportPointer",                                {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaMemPoolImportPointer",                                {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddExternalSemaphoresSignalNode",                {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExternalSemaphoresSignalNodeGetParams",          {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExternalSemaphoresSignalNodeSetParams",          {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddExternalSemaphoresWaitNode",                  {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExternalSemaphoresWaitNodeGetParams",            {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExternalSemaphoresWaitNodeSetParams",            {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecExternalSemaphoresSignalNodeSetParams",      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaGraphExecExternalSemaphoresWaitNodeSetParams",        {CUDA_112, CUDA_0,   CUDA_0  }},
  {"cudaDeviceFlushGPUDirectRDMAWrites",                      {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaGraphDebugDotPrint",                                  {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaUserObjectCreate",                                    {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaUserObjectRetain",                                    {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaUserObjectRelease",                                   {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaGraphRetainUserObject",                               {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaGraphReleaseUserObject",                              {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaGetDriverEntryPoint",                                 {CUDA_113, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemAllocNode",                                {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemAllocNodeGetParams",                          {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaGraphAddMemFreeNode",                                 {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaGraphMemFreeNodeGetParams",                           {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaDeviceGraphMemTrim",                                  {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaDeviceGetGraphMemAttribute",                          {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaDeviceSetGraphMemAttribute",                          {CUDA_114, CUDA_0,   CUDA_0  }},
  {"cudaGraphInstantiateWithFlags",                           {CUDA_114, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_RUNTIME_FUNCTION_VER_MAP {
  {"hipChooseDevice",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetAttribute",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetByPCIBusId",                                  {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetCacheConfig",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetLimit",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetP2PAttribute",                                {HIP_3080, HIP_0,    HIP_0   }},
  {"hipDeviceGetPCIBusId",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetSharedMemConfig",                             {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetStreamPriorityRange",                         {HIP_2000, HIP_0,    HIP_0   }},
  {"hipDeviceReset",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceSetCacheConfig",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceSetSharedMemConfig",                             {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceSynchronize",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetDevice",                                            {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetDeviceCount",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetDeviceFlags",                                       {HIP_3060, HIP_0,    HIP_0   }},
  {"hipGetDeviceProperties",                                  {HIP_1060, HIP_0,    HIP_0   }},
  {"hipIpcCloseMemHandle",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipIpcGetEventHandle",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipIpcGetMemHandle",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipIpcOpenEventHandle",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipIpcOpenMemHandle",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipSetDevice",                                            {HIP_1060, HIP_0,    HIP_0   }},
  {"hipSetDeviceFlags",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceReset",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceGetCacheConfig",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetErrorName",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetErrorString",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetLastError",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipPeekAtLastError",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamAddCallback",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamAttachMemAsync",                                 {HIP_3070, HIP_0,    HIP_0   }},
  {"hipStreamCreate",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamCreateWithFlags",                                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamCreateWithPriority",                             {HIP_2000, HIP_0,    HIP_0   }},
  {"hipStreamDestroy",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamGetFlags",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamGetPriority",                                    {HIP_2000, HIP_0,    HIP_0   }},
  {"hipStreamQuery",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamSynchronize",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipStreamWaitEvent",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventCreate",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventCreateWithFlags",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventDestroy",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventElapsedTime",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventQuery",                                           {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventRecord",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipEventSynchronize",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipFuncGetAttributes",                                    {HIP_1090, HIP_0,    HIP_0   }},
  {"hipFuncSetAttribute",                                     {HIP_3090, HIP_0,    HIP_0   }},
  {"hipFuncSetCacheConfig",                                   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipFuncSetSharedMemConfig",                               {HIP_3090, HIP_0,    HIP_0   }},
  {"hipLaunchCooperativeKernel",                              {HIP_2060, HIP_0,    HIP_0   }},
  {"hipLaunchCooperativeKernelMultiDevice",                   {HIP_2060, HIP_0,    HIP_0   }},
  {"hipLaunchKernel",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipOccupancyMaxActiveBlocksPerMultiprocessor",            {HIP_1060, HIP_0,    HIP_0   }},
  {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",   {HIP_2060, HIP_0,    HIP_0   }},
  {"hipOccupancyMaxPotentialBlockSize",                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipOccupancyMaxPotentialBlockSizeWithFlags",              {HIP_3050, HIP_0,    HIP_0   }},
  {"hipFree",                                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"hipFreeArray",                                            {HIP_1060, HIP_0,    HIP_0   }},
  {"hipHostFree",                                             {HIP_1060, HIP_0,    HIP_0   }},
  {"hipFreeMipmappedArray",                                   {HIP_3050, HIP_0,    HIP_0   }},
  {"hipGetMipmappedArrayLevel",                               {HIP_3050, HIP_0,    HIP_0   }},
  {"hipGetSymbolAddress",                                     {HIP_2000, HIP_0,    HIP_0   }},
  {"hipGetSymbolSize",                                        {HIP_2000, HIP_0,    HIP_0   }},
  {"hipHostMalloc",                                           {HIP_1060, HIP_0,    HIP_0   }},
  {"hipHostGetFlags",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipHostRegister",                                         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipHostUnregister",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMalloc",                                               {HIP_1050, HIP_0,    HIP_0   }},
  {"hipMalloc3D",                                             {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMalloc3DArray",                                        {HIP_1070, HIP_0,    HIP_0   }},
  {"hipMallocArray",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipHostGetDevicePointer",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMallocManaged",                                        {HIP_2050, HIP_0,    HIP_0   }},
  {"hipMallocMipmappedArray",                                 {HIP_3050, HIP_0,    HIP_0   }},
  {"hipMallocPitch",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemAdvise",                                            {HIP_3070, HIP_0,    HIP_0   }},
  {"hipMemcpy",                                               {HIP_1050, HIP_0,    HIP_0   }},
  {"hipMemcpy2D",                                             {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpy2DAsync",                                        {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpy2DFromArray",                                    {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMemcpy2DFromArrayAsync",                               {HIP_3000, HIP_0,    HIP_0   }},
  {"hipMemcpy2DToArray",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpy3D",                                             {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpy3DAsync",                                        {HIP_2080, HIP_0,    HIP_0   }},
  {"hipMemcpyAsync",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyFromSymbol",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyFromSymbolAsync",                                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyPeer",                                           {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyPeerAsync",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyToSymbol",                                       {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemcpyToSymbolAsync",                                  {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemGetInfo",                                           {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemPrefetchAsync",                                     {HIP_3070, HIP_0,    HIP_0   }},
  {"hipMemRangeGetAttribute",                                 {HIP_3070, HIP_0,    HIP_0   }},
  {"hipMemRangeGetAttributes",                                {HIP_3070, HIP_0,    HIP_0   }},
  {"hipMemset",                                               {HIP_1060, HIP_0,    HIP_0   }},
  {"hipMemset2D",                                             {HIP_1070, HIP_0,    HIP_0   }},
  {"hipMemset2DAsync",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemset3D",                                             {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemset3DAsync",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipMemsetAsync",                                          {HIP_1060, HIP_0,    HIP_0   }},
  {"make_hipExtent",                                          {HIP_1070, HIP_0,    HIP_0   }},
  {"make_hipPitchedPtr",                                      {HIP_1070, HIP_0,    HIP_0   }},
  {"make_hipPos",                                             {HIP_1070, HIP_0,    HIP_0   }},
  {"hipMemcpyFromArray",                                      {HIP_1090, HIP_3080, HIP_0   }},
  {"hipMemcpyToArray",                                        {HIP_1060, HIP_3080, HIP_0   }},
  {"hipPointerGetAttributes",                                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDeviceCanAccessPeer",                                  {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDeviceDisablePeerAccess",                              {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDeviceEnablePeerAccess",                               {HIP_1090, HIP_0,    HIP_0   }},
  {"hipBindTexture",                                          {HIP_1060, HIP_3080, HIP_0   }},
  {"hipBindTexture2D",                                        {HIP_1070, HIP_3080, HIP_0   }},
  {"hipBindTextureToArray",                                   {HIP_1060, HIP_3080, HIP_0   }},
  {"hipBindTextureToMipmappedArray",                          {HIP_1070, HIP_0,    HIP_0   }},
  {"hipCreateChannelDesc",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipGetChannelDesc",                                       {HIP_1070, HIP_0,    HIP_0   }},
  {"hipGetTextureAlignmentOffset",                            {HIP_1090, HIP_3080, HIP_0   }},
  {"hipGetTextureReference",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"hipUnbindTexture",                                        {HIP_1060, HIP_3080, HIP_0   }},
  {"hipCreateTextureObject",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"hipDestroyTextureObject",                                 {HIP_1070, HIP_0,    HIP_0   }},
  {"hipGetTextureObjectResourceDesc",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"hipGetTextureObjectResourceViewDesc",                     {HIP_1070, HIP_0,    HIP_0   }},
  {"hipGetTextureObjectTextureDesc",                          {HIP_1070, HIP_0,    HIP_0   }},
  {"hipCreateSurfaceObject",                                  {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDestroySurfaceObject",                                 {HIP_1090, HIP_0,    HIP_0   }},
  {"hipDriverGetVersion",                                     {HIP_1060, HIP_0,    HIP_0   }},
  {"hipRuntimeGetVersion",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipProfilerStart",                                        {HIP_1060, HIP_3000, HIP_0   }},
  {"hipProfilerStop",                                         {HIP_1060, HIP_3000, HIP_0   }},
  {"hipConfigureCall",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipLaunchByPtr",                                          {HIP_1090, HIP_0,    HIP_0   }},
  {"hipSetupArgument",                                        {HIP_1090, HIP_0,    HIP_0   }},
  {"hipImportExternalSemaphore",                              {HIP_4040, HIP_0,    HIP_0   }},
  {"hipSignalExternalSemaphoresAsync",                        {HIP_4040, HIP_0,    HIP_0   }},
  {"hipWaitExternalSemaphoresAsync",                          {HIP_4040, HIP_0,    HIP_0   }},
  {"hipDestroyExternalSemaphore",                             {HIP_4040, HIP_0,    HIP_0   }},
  {"hipImportExternalMemory",                                 {HIP_4030, HIP_0,    HIP_0   }},
  {"hipExternalMemoryGetMappedBuffer",                        {HIP_4030, HIP_0,    HIP_0   }},
  {"hipDestroyExternalMemory",                                {HIP_4030, HIP_0,    HIP_0   }},
  {"hipMemcpy2DToArrayAsync",                                 {HIP_4030, HIP_0,    HIP_0   }},
  {"hipStreamBeginCapture",                                   {HIP_4030, HIP_0,    HIP_0   }},
  {"hipStreamEndCapture",                                     {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphCreate",                                          {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphDestroy",                                         {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphExecDestroy",                                     {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphInstantiate",                                     {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphLaunch",                                          {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphAddKernelNode",                                   {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphAddMemcpyNode",                                   {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphAddMemsetNode",                                   {HIP_4030, HIP_0,    HIP_0   }},
  {"hipGraphAddMemcpyNode1D",                                 {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphGetNodes",                                        {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphGetRootNodes",                                    {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphKernelNodeGetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphKernelNodeSetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphMemcpyNodeGetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphMemcpyNodeSetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphMemsetNodeGetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphMemsetNodeSetParams",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphExecKernelNodeSetParams",                         {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphAddDependencies",                                 {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphAddEmptyNode",                                    {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGLGetDevices",                                         {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphicsGLRegisterBuffer",                             {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphicsMapResources",                                 {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphicsResourceGetMappedPointer",                     {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphicsUnmapResources",                               {HIP_4050, HIP_0,    HIP_0   }},
  {"hipGraphicsUnregisterResource",                           {HIP_4050, HIP_0,    HIP_0   }},
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
  {11, "Stream Ordered Memory Allocator"},
  {12, "Unified Addressing"},
  {13, "Peer Device Memory Access"},
  {14, "OpenGL Interoperability"},
  {15, "OpenGL Interoperability [DEPRECATED]"},
  {16, "Direct3D 9 Interoperability"},
  {17, "Direct3D 9 Interoperability [DEPRECATED]"},
  {18, "Direct3D 10 Interoperability"},
  {19, "Direct3D 10 Interoperability [DEPRECATED]"},
  {20, "Direct3D 11 Interoperability"},
  {21, "Direct3D 11 Interoperability [DEPRECATED]"},
  {22, "VDPAU Interoperability"},
  {23, "EGL Interoperability"},
  {24, "Graphics Interoperability"},
  {25, "Texture Reference Management [DEPRECATED]"},
  {26, "Surface Reference Management [DEPRECATED]"},
  {27, "Texture Object Management"},
  {28, "Surface Object Management"},
  {29, "Version Management"},
  {30, "Graph Management"},
  {31, "Driver Entry Point Access"},
  {32, "C++ API Routines"},
  {33, "Interactions with the CUDA Driver API"},
  {34, "Profiler Control [DEPRECATED]"},
  {35, "Profiler Control"},
  {36, "Data types used by CUDA Runtime"},
  {37, "Execution Control [REMOVED]"},
};
