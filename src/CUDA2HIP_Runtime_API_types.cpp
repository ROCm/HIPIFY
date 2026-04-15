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

// Maps the names of CUDA RUNTIME API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_TYPE_NAME_MAP = [] {

  std::map<llvm::StringRef, hipCounter> m;

  // 1. Structs

  // no analogue
  m["cudaChannelFormatDesc"]                                    = {"hipChannelFormatDesc",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // no analogue
  m["cudaDeviceProp"]                                           = {"hipDeviceProp_t",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaEglFrame"]                                             = {"hipEglFrame",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaEglFrame_st"]                                          = {"hipEglFrame",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogue
  m["cudaEglPlaneDesc"]                                         = {"hipEglPlaneDesc",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaEglPlaneDesc_st"]                                      = {"hipEglPlaneDesc",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogue
  m["cudaExtent"]                                               = {"hipExtent",                                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXTERNAL_MEMORY_BUFFER_DESC
  m["cudaExternalMemoryBufferDesc"]                             = {"hipExternalMemoryBufferDesc",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXTERNAL_MEMORY_HANDLE_DESC
  m["cudaExternalMemoryHandleDesc"]                             = {"hipExternalMemoryHandleDesc",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
  m["cudaExternalMemoryMipmappedArrayDesc"]                     = {"HIP_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC",                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
  m["cudaExternalSemaphoreHandleDesc"]                          = {"hipExternalSemaphoreHandleDesc",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
  m["cudaExternalSemaphoreSignalParams"]                        = {"hipExternalSemaphoreSignalParams",                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  m["cudaExternalSemaphoreSignalParams_v1"]                     = {"hipExternalSemaphoreSignalParams",                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED | CUDA_REMOVED};

  // CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
  m["cudaExternalSemaphoreWaitParams"]                          = {"hipExternalSemaphoreWaitParams",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  m["cudaExternalSemaphoreWaitParams_v1"]                       = {"hipExternalSemaphoreWaitParams",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED | CUDA_REMOVED};

  // no analogue
  m["cudaFuncAttributes"]                                       = {"hipFuncAttributes",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_HOST_NODE_PARAMS
  m["cudaHostNodeParams"]                                       = {"hipHostNodeParams",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_HOST_NODE_PARAMS_v2
  m["cudaHostNodeParamsV2"]                                     = {"hipHostNodeParams_v2",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUipcEventHandle
  m["cudaIpcEventHandle_t"]                                     = {"hipIpcEventHandle_t",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUipcEventHandle_st
  m["cudaIpcEventHandle_st"]                                    = {"hipIpcEventHandle_st",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUipcMemHandle
  m["cudaIpcMemHandle_t"]                                       = {"hipIpcMemHandle_t",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUipcMemHandle_st
  m["cudaIpcMemHandle_st"]                                      = {"hipIpcMemHandle_st",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_KERNEL_NODE_PARAMS
  m["cudaKernelNodeParams"]                                     = {"hipKernelNodeParams",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_KERNEL_NODE_PARAMS_v2_st
  m["cudaKernelNodeParamsV2"]                                   = {"hipKernelNodeParams_v2",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogue
  // CUDA_LAUNCH_PARAMS struct differs
  m["cudaLaunchParams"]                                         = {"hipLaunchParams",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_REMOVED};

  // no analogue
  // NOTE: HIP struct is bigger and contains cudaMemcpy3DParms only in the beginning
  m["cudaMemcpy3DParms"]                                        = {"hipMemcpy3DParms",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaMemcpy3DPeerParms"]                                    = {"hipMemcpy3DPeerParms",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  //
  m["cudaMemsetParams"]                                         = {"hipMemsetParams",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  //
  m["cudaMemsetParamsV2"]                                       = {"hipMemsetParams_v2",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogue
  m["cudaPitchedPtr"]                                           = {"hipPitchedPtr",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaPointerAttributes"]                                    = {"hipPointerAttribute_t",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaPos"]                                                  = {"hipPos",                                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  // NOTE: CUDA_RESOURCE_DESC struct differs
  m["cudaResourceDesc"]                                         = {"hipResourceDesc",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // NOTE: CUDA_RESOURCE_VIEW_DESC has reserved bytes in the end
  m["cudaResourceViewDesc"]                                     = {"hipResourceViewDesc",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  // NOTE: CUDA_TEXTURE_DESC differs
  m["cudaTextureDesc"]                                          = {"hipTextureDesc",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // NOTE: the same struct and its name
  m["CUuuid_st"]                                                = {"hipUUID_t",                                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // NOTE: possibly CUsurfref is analogue
  m["surfaceReference"]                                         = {"surfaceReference",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_REMOVED};

  // NOTE: possibly CUtexref_st is analogue
  m["textureReference"]                                         = {"textureReference",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  m["texture"]                                                  = {"texture",                                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_REMOVED};

  // the same - CUevent_st
  m["CUevent_st"]                                               = {"ihipEvent_t",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUevent
  m["cudaEvent_t"]                                              = {"hipEvent_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUextMemory_st
  m["CUexternalMemory_st"]                                      = {"hipExtMemory_st",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUexternalMemory
  m["cudaExternalMemory_t"]                                     = {"hipExternalMemory_t",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUextSemaphore_st
  m["CUexternalSemaphore_st"]                                   = {"hipExtSemaphore_st",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUexternalSemaphore
  m["cudaExternalSemaphore_t"]                                  = {"hipExternalSemaphore_t",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // the same - CUgraph_st
  m["CUgraph_st"]                                               = {"ihipGraph",                                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraph
  m["cudaGraph_t"]                                              = {"hipGraph_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // the same -CUgraphExec_st
  m["CUgraphExec_st"]                                           = {"hipGraphExec",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphExec
  m["cudaGraphExec_t"]                                          = {"hipGraphExec_t",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphicsResource_st
  m["cudaGraphicsResource"]                                     = {"hipGraphicsResource",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphicsResource
  m["cudaGraphicsResource_t"]                                   = {"hipGraphicsResource_t",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // the same - CUgraphNode_st
  m["CUgraphNode_st"]                                           = {"hipGraphNode",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphNode
  m["cudaGraphNode_t"]                                          = {"hipGraphNode_t",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUeglStreamConnection_st
  m["CUeglStreamConnection_st"]                                 = {"hipEglStreamConnection",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUeglStreamConnection
  m["cudaEglStreamConnection"]                                  = {"hipEglStreamConnection",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUarray_st
  m["cudaArray"]                                                = {"hipArray",                                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUarray
  m["cudaArray_t"]                                              = {"hipArray_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // no analogue
  m["cudaArray_const_t"]                                        = {"hipArray_const_t",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmipmappedArray_st
  m["cudaMipmappedArray"]                                       = {"hipMipmappedArray",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUmipmappedArray
  m["cudaMipmappedArray_t"]                                     = {"hipMipmappedArray_t",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // no analogue
  m["cudaMipmappedArray_const_t"]                               = {"hipMipmappedArray_const_t",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // the same - CUstream_st
  m["CUstream_st"]                                              = {"ihipStream_t",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUstream
  m["cudaStream_t"]                                             = {"hipStream_t",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUfunction
  m["cudaFunction_t"]                                           = {"hipFunction_t",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUaccessPolicyWindow_st
  m["cudaAccessPolicyWindow"]                                   = {"hipAccessPolicyWindow",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_ARRAY_SPARSE_PROPERTIES_st
  m["cudaArraySparseProperties"]                                = {"hipArraySparseProperties",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUmemLocation_st
  m["cudaMemLocation"]                                          = {"hipMemLocation",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemAccessDesc_st
  m["cudaMemAccessDesc"]                                        = {"hipMemAccessDesc",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemPoolProps_st
  m["cudaMemPoolProps"]                                         = {"hipMemPoolProps",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemPoolPtrExportData_st
  m["cudaMemPoolPtrExportData"]                                 = {"hipMemPoolPtrExportData",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
  m["cudaExternalSemaphoreSignalNodeParams"]                    = {"hipExternalSemaphoreSignalNodeParams",                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st
  m["cudaExternalSemaphoreSignalNodeParamsV2"]                  = {"hipExternalSemaphoreSignalNodeParams",                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
  m["cudaExternalSemaphoreWaitNodeParams"]                      = {"hipExternalSemaphoreWaitNodeParams",                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st
  m["cudaExternalSemaphoreWaitNodeParamsV2"]                    = {"hipExternalSemaphoreWaitNodeParams",                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_MEM_ALLOC_NODE_PARAMS_st
  m["cudaMemAllocNodeParams"]                                   = {"hipMemAllocNodeParams",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_MEM_ALLOC_NODE_PARAMS_v2_st
  m["cudaMemAllocNodeParamsV2"]                                 = {"hipMemAllocNodeParams_v2",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUDA_MEM_FREE_NODE_PARAMS_st
  m["cudaMemFreeNodeParams"]                                    = {"hipMemFreeNodeParams",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_CHILD_GRAPH_NODE_PARAMS_st
  m["cudaChildGraphNodeParams"]                                 = {"hipChildGraphNodeParams",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EVENT_RECORD_NODE_PARAMS_st
  m["cudaEventRecordNodeParams"]                                = {"hipEventRecordNodeParams",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_EVENT_WAIT_NODE_PARAMS_st
  m["cudaEventWaitNodeParams"]                                  = {"hipEventWaitNodeParams",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphNodeParams_st
  m["cudaGraphNodeParams"]                                      = {"hipGraphNodeParams",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_ARRAY_MEMORY_REQUIREMENTS_st
  m["cudaArrayMemoryRequirements"]                              = {"hipArrayMemoryRequirements",                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUlaunchMemSyncDomainMap_st
  m["cudaLaunchMemSyncDomainMap_st"]                            = {"hipLaunchMemSyncDomainMap",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUlaunchMemSyncDomainMap
  m["cudaLaunchMemSyncDomainMap"]                               = {"hipLaunchMemSyncDomainMap",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUkernel
  m["cudaKernel_t"]                                             = {"hipKernel_t",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_MEMCPY_NODE_PARAMS
  m["cudaMemcpyNodeParams"]                                     = {"hipMemcpyNodeParams",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_CONDITIONAL_NODE_PARAMS
  m["cudaConditionalNodeParams"]                                = {"hipConditionalNodeParams",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphEdgeData_st
  m["cudaGraphEdgeData_st"]                                     = {"hipGraphEdgeData",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphEdgeData
  m["cudaGraphEdgeData"]                                        = {"hipGraphEdgeData",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaGraphKernelNodeUpdate"]                                = {"hipGraphKernelNodeUpdate",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUmemcpyAttributes
  m["cudaMemcpyAttributes"]                                     = {"hipMemcpyAttributes",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUdevSmResource_st
  m["cudaDevSmResource"]                                        = {"hipDevSmResource",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevWorkqueueConfigResource_st
  m["cudaDevWorkqueueConfigResource"]                           = {"hipDevWorkqueueConfigResource",                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevWorkqueueResource_st
  m["cudaDevWorkqueueResource"]                                 = {"hipDevWorkqueueResource",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // 2. Unions

  // CUstreamAttrValue
  m["cudaStreamAttrValue"]                                      = {"hipLaunchAttributeValue",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUkernelNodeAttrValue
  m["cudaKernelNodeAttrValue"]                                  = {"hipKernelNodeAttrValue",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUlaunchAttributeValue
  m["cudaLaunchAttributeValue"]                                 = {"hipLaunchAttributeValue",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUlaunchAttribute_st
  m["cudaLaunchAttribute_st"]                                   = {"hipLaunchAttribute_st",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUlaunchAttribute
  m["cudaLaunchAttribute"]                                      = {"hipLaunchAttribute",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // NOTE: CUlaunchConfig_st struct differs
  m["cudaLaunchConfig_st"]                                      = {"hipLaunchConfig_st",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // NOTE: CUlaunchConfig struct differs
  m["cudaLaunchConfig_t"]                                       = {"hipLaunchConfig_t",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_GRAPH_INSTANTIATE_PARAMS_st
  m["cudaGraphInstantiateParams_st"]                            = {"hipGraphInstantiateParams",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_PARAMS
  m["cudaGraphInstantiateParams"]                               = {"hipGraphInstantiateParams",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphExecUpdateResultInfo_st
  m["cudaGraphExecUpdateResultInfo_st"]                         = {"hipGraphExecUpdateResultInfo",                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUgraphExecUpdateResultInfo
  m["cudaGraphExecUpdateResultInfo"]                            = {"hipGraphExecUpdateResultInfo",                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUmemFabricHandle_st
  m["cudaMemFabricHandle_st"]                                   = {"hipMemFabricHandle",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUmemFabricHandle
  m["cudaMemFabricHandle_t"]                                    = {"hipMemFabricHandle",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphDeviceNode
  m["cudaGraphDeviceNode_t"]                                    = {"hipGraphDeviceNode",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUasyncNotificationInfo_st
  m["cudaAsyncNotificationInfo"]                                = {"hipAsyncNotificationInfo",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUasyncNotificationInfo
  m["cudaAsyncNotificationInfo_t"]                              = {"hipAsyncNotificationInfo",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUoffset3D
  m["cudaOffset3D"]                                             = {"hipOffset3D",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemcpy3DOperand
  m["cudaMemcpy3DOperand"]                                      = {"hipMemcpy3DOperand",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUDA_MEMCPY3D_BATCH_OP
  m["cudaMemcpy3DBatchOp"]                                      = {"hipMemcpy3DBatchOp",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUlibraryHostUniversalFunctionAndDataTable
  m["cudalibraryHostUniversalFunctionAndDataTable"]             = {"hipLibraryHostUniversalFunctionAndDataTable",              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUlibrary
  m["cudaLibrary_t"]                                            = {"hipLibrary_t",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUdevResourceDesc_st
  m["CUdevResourceDesc_st"]                                     = {"hipDevResourceDesc",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUdevResourceDesc
  m["cudaDevResourceDesc_t"]                                    = {"hipDevResourceDesc_t",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  //
  m["cudaExecutionContext_st"]                                  = {"hipExecutionContext",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  //
  m["cudaExecutionContext_t"]                                   = {"hipExecutionContext_t",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CU_DEV_SM_RESOURCE_GROUP_PARAMS_st
  m["cudaDevSmResourceGroupParams_st"]                          = {"hipDevSmResourceGroupParams",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_SM_RESOURCE_GROUP_PARAMS
  m["cudaDevSmResourceGroupParams"]                             = {"hipDevSmResourceGroupParams",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevResource_st
  m["cudaDevResource_st"]                                       = {"hipDevResource",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUdevResource
  m["cudaDevResource"]                                          = {"hipDevResource",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // 3. Enums

  // no analogue
  m["cudaCGScope"]                                              = {"hipCGScope",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaCGScope enum values
  m["cudaCGScopeInvalid"]                                       = {"hipCGScopeInvalid",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  m["cudaCGScopeGrid"]                                          = {"hipCGScopeGrid",                                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  m["cudaCGScopeMultiGrid"]                                     = {"hipCGScopeMultiGrid",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2

  // no analogue
  m["cudaChannelFormatKind"]                                    = {"hipChannelFormatKind",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaChannelFormatKind enum values
  m["cudaChannelFormatKindSigned"]                              = {"hipChannelFormatKindSigned",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  m["cudaChannelFormatKindUnsigned"]                            = {"hipChannelFormatKindUnsigned",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  m["cudaChannelFormatKindFloat"]                               = {"hipChannelFormatKindFloat",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  m["cudaChannelFormatKindNone"]                                = {"hipChannelFormatKindNone",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  m["cudaChannelFormatKindNV12"]                                = {"hipChannelFormatKindNV12",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 4
  m["cudaChannelFormatKindUnsignedNormalized8X1"]               = {"hipChannelFormatKindUnsignedNormalized8X1",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 5
  m["cudaChannelFormatKindUnsignedNormalized8X2"]               = {"hipChannelFormatKindUnsignedNormalized8X2",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 6
  m["cudaChannelFormatKindUnsignedNormalized8X4"]               = {"hipChannelFormatKindUnsignedNormalized8X4",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 7
  m["cudaChannelFormatKindUnsignedNormalized16X1"]              = {"hipChannelFormatKindUnsignedNormalized16X1",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8
  m["cudaChannelFormatKindUnsignedNormalized16X2"]              = {"hipChannelFormatKindUnsignedNormalized16X2",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 9
  m["cudaChannelFormatKindUnsignedNormalized16X4"]              = {"hipChannelFormatKindUnsignedNormalized16X4",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 10
  m["cudaChannelFormatKindSignedNormalized8X1"]                 = {"hipChannelFormatKindSignedNormalized8X1",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 11
  m["cudaChannelFormatKindSignedNormalized8X2"]                 = {"hipChannelFormatKindSignedNormalized8X2",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 12
  m["cudaChannelFormatKindSignedNormalized8X4"]                 = {"hipChannelFormatKindSignedNormalized8X4",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 13
  m["cudaChannelFormatKindSignedNormalized16X1"]                = {"hipChannelFormatKindSignedNormalized16X1",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 14
  m["cudaChannelFormatKindSignedNormalized16X2"]                = {"hipChannelFormatKindSignedNormalized16X2",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 15
  m["cudaChannelFormatKindSignedNormalized16X4"]                = {"hipChannelFormatKindSignedNormalized16X4",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 16
  m["cudaChannelFormatKindUnsignedBlockCompressed1"]            = {"hipChannelFormatKindUnsignedBlockCompressed1",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 17
  m["cudaChannelFormatKindUnsignedBlockCompressed1SRGB"]        = {"hipChannelFormatKindUnsignedBlockCompressed1SRGB",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 18
  m["cudaChannelFormatKindUnsignedBlockCompressed2"]            = {"hipChannelFormatKindUnsignedBlockCompressed2",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 19
  m["cudaChannelFormatKindUnsignedBlockCompressed2SRGB"]        = {"hipChannelFormatKindUnsignedBlockCompressed2SRGB",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 20
  m["cudaChannelFormatKindUnsignedBlockCompressed3"]            = {"hipChannelFormatKindUnsignedBlockCompressed3",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 21
  m["cudaChannelFormatKindUnsignedBlockCompressed3SRGB"]        = {"hipChannelFormatKindUnsignedBlockCompressed3SRGB",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 22
  m["cudaChannelFormatKindUnsignedBlockCompressed4"]            = {"hipChannelFormatKindUnsignedBlockCompressed4",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 23
  m["cudaChannelFormatKindSignedBlockCompressed4"]              = {"hipChannelFormatKindSignedBlockCompressed4",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 24
  m["cudaChannelFormatKindUnsignedBlockCompressed5"]            = {"hipChannelFormatKindUnsignedBlockCompressed5",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 25
  m["cudaChannelFormatKindSignedBlockCompressed5"]              = {"hipChannelFormatKindSignedBlockCompressed5",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 26
  m["cudaChannelFormatKindUnsignedBlockCompressed6H"]           = {"hipChannelFormatKindUnsignedBlockCompressed6H",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 27
  m["cudaChannelFormatKindSignedBlockCompressed6H"]             = {"hipChannelFormatKindSignedBlockCompressed6H",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 28
  m["cudaChannelFormatKindUnsignedBlockCompressed7"]            = {"hipChannelFormatKindUnsignedBlockCompressed7",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 29
  m["cudaChannelFormatKindUnsignedBlockCompressed7SRGB"]        = {"hipChannelFormatKindUnsignedBlockCompressed7SRGB",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 30
  m["cudaChannelFormatKindUnsignedNormalized1010102"]           = {"hipChannelFormatKindUnsignedNormalized1010102",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 31

  // CUcomputemode
  m["cudaComputeMode"]                                          = {"hipComputeMode",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaComputeMode enum values
  // CU_COMPUTEMODE_DEFAULT
  m["cudaComputeModeDefault"]                                   = {"hipComputeModeDefault",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_COMPUTEMODE_EXCLUSIVE
  m["cudaComputeModeExclusive"]                                 = {"hipComputeModeExclusive",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_COMPUTEMODE_PROHIBITED
  m["cudaComputeModeProhibited"]                                = {"hipComputeModeProhibited",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_COMPUTEMODE_EXCLUSIVE_PROCESS
  m["cudaComputeModeExclusiveProcess"]                          = {"hipComputeModeExclusiveProcess",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUdevice_attribute
  m["cudaDeviceAttr"]                                           = {"hipDeviceAttribute_t",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaDeviceAttr enum values
  // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
  m["cudaDevAttrMaxThreadsPerBlock"]                            = {"hipDeviceAttributeMaxThreadsPerBlock",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  1
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
  m["cudaDevAttrMaxBlockDimX"]                                  = {"hipDeviceAttributeMaxBlockDimX",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  2
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
  m["cudaDevAttrMaxBlockDimY"]                                  = {"hipDeviceAttributeMaxBlockDimY",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  3
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
  m["cudaDevAttrMaxBlockDimZ"]                                  = {"hipDeviceAttributeMaxBlockDimZ",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  4
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
  m["cudaDevAttrMaxGridDimX"]                                   = {"hipDeviceAttributeMaxGridDimX",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  5
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
  m["cudaDevAttrMaxGridDimY"]                                   = {"hipDeviceAttributeMaxGridDimY",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  6
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
  m["cudaDevAttrMaxGridDimZ"]                                   = {"hipDeviceAttributeMaxGridDimZ",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  7
  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
  m["cudaDevAttrMaxSharedMemoryPerBlock"]                       = {"hipDeviceAttributeMaxSharedMemoryPerBlock",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  8
  // CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
  m["cudaDevAttrTotalConstantMemory"]                           = {"hipDeviceAttributeTotalConstantMemory",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  9
  // CU_DEVICE_ATTRIBUTE_WARP_SIZE
  m["cudaDevAttrWarpSize"]                                      = {"hipDeviceAttributeWarpSize",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 10
  // CU_DEVICE_ATTRIBUTE_MAX_PITCH
  m["cudaDevAttrMaxPitch"]                                      = {"hipDeviceAttributeMaxPitch",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 11
  // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
  m["cudaDevAttrMaxRegistersPerBlock"]                          = {"hipDeviceAttributeMaxRegistersPerBlock",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 12
  // CU_DEVICE_ATTRIBUTE_CLOCK_RATE
  m["cudaDevAttrClockRate"]                                     = {"hipDeviceAttributeClockRate",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 13
  // CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
  m["cudaDevAttrTextureAlignment"]                              = {"hipDeviceAttributeTextureAlignment",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 14
  // CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  // NOTE: Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  m["cudaDevAttrGpuOverlap"]                                    = {"hipDeviceAttributeAsyncEngineCount",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 15
  // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
  m["cudaDevAttrMultiProcessorCount"]                           = {"hipDeviceAttributeMultiprocessorCount",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 16
  // CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
  m["cudaDevAttrKernelExecTimeout"]                             = {"hipDeviceAttributeKernelExecTimeout",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 17
  // CU_DEVICE_ATTRIBUTE_INTEGRATED
  m["cudaDevAttrIntegrated"]                                    = {"hipDeviceAttributeIntegrated",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 18
  // CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
  m["cudaDevAttrCanMapHostMemory"]                              = {"hipDeviceAttributeCanMapHostMemory",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 19
  // CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
  m["cudaDevAttrComputeMode"]                                   = {"hipDeviceAttributeComputeMode",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 20
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
  m["cudaDevAttrMaxTexture1DWidth"]                             = {"hipDeviceAttributeMaxTexture1DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 21
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
  m["cudaDevAttrMaxTexture2DWidth"]                             = {"hipDeviceAttributeMaxTexture2DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 22
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
  m["cudaDevAttrMaxTexture2DHeight"]                            = {"hipDeviceAttributeMaxTexture2DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 23
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
  m["cudaDevAttrMaxTexture3DWidth"]                             = {"hipDeviceAttributeMaxTexture3DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 24
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
  m["cudaDevAttrMaxTexture3DHeight"]                            = {"hipDeviceAttributeMaxTexture3DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 25
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
  m["cudaDevAttrMaxTexture3DDepth"]                             = {"hipDeviceAttributeMaxTexture3DDepth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 26
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture2DLayeredWidth"]                      = {"hipDeviceAttributeMaxTexture2DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 27
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  // CUDA only
  m["cudaDevAttrMaxTexture2DLayeredHeight"]                     = {"hipDeviceAttributeMaxTexture2DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 28
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
  m["cudaDevAttrMaxTexture2DLayeredLayers"]                     = {"hipDeviceAttributeMaxTexture2DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 29
  // CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT
  // CUDA only
  m["cudaDevAttrSurfaceAlignment"]                              = {"hipDeviceAttributeSurfaceAlignment",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 30
  // CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
  m["cudaDevAttrConcurrentKernels"]                             = {"hipDeviceAttributeConcurrentKernels",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 31
  // CU_DEVICE_ATTRIBUTE_ECC_ENABLED
  m["cudaDevAttrEccEnabled"]                                    = {"hipDeviceAttributeEccEnabled",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 32
  // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
  m["cudaDevAttrPciBusId"]                                      = {"hipDeviceAttributePciBusId",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 33
  // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
  m["cudaDevAttrPciDeviceId"]                                   = {"hipDeviceAttributePciDeviceId",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 34
  // CU_DEVICE_ATTRIBUTE_TCC_DRIVER
  // CUDA only
  m["cudaDevAttrTccDriver"]                                     = {"hipDeviceAttributeTccDriver",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 35
  // CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
  m["cudaDevAttrMemoryClockRate"]                               = {"hipDeviceAttributeMemoryClockRate",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 36
  // CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
  m["cudaDevAttrGlobalMemoryBusWidth"]                          = {"hipDeviceAttributeMemoryBusWidth",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 37
  // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
  m["cudaDevAttrL2CacheSize"]                                   = {"hipDeviceAttributeL2CacheSize",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 38
  // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
  m["cudaDevAttrMaxThreadsPerMultiProcessor"]                   = {"hipDeviceAttributeMaxThreadsPerMultiProcessor",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 39
  // CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
  // CUDA only
  m["cudaDevAttrAsyncEngineCount"]                              = {"hipDeviceAttributeAsyncEngineCount",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 40
  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
  // CUDA only
  m["cudaDevAttrUnifiedAddressing"]                             = {"hipDeviceAttributeUnifiedAddressing",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 41
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture1DLayeredWidth"]                      = {"hipDeviceAttributeMaxTexture1DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 42
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
  m["cudaDevAttrMaxTexture1DLayeredLayers"]                     = {"hipDeviceAttributeMaxTexture1DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 43
  // 44 - no
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture2DGatherWidth"]                       = {"hipDeviceAttributeMaxTexture2DGather",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 45
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT
  // CUDA only
  m["cudaDevAttrMaxTexture2DGatherHeight"]                      = {"hipDeviceAttributeMaxTexture2DGather",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 46
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
  // CUDA only
  m["cudaDevAttrMaxTexture3DWidthAlt"]                          = {"hipDeviceAttributeMaxTexture3DAlt",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 47
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
  // CUDA only
  m["cudaDevAttrMaxTexture3DHeightAlt"]                         = {"hipDeviceAttributeMaxTexture3DAlt",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 48
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
  // CUDA only
  m["cudaDevAttrMaxTexture3DDepthAlt"]                          = {"hipDeviceAttributeMaxTexture3DAlt",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 49
  // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
  m["cudaDevAttrPciDomainId"]                                   = {"hipDeviceAttributePciDomainId",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 50
  // CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT
  m["cudaDevAttrTexturePitchAlignment"]                         = {"hipDeviceAttributeTexturePitchAlignment",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 51
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTextureCubemapWidth"]                        = {"hipDeviceAttributeMaxTextureCubemap",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 52
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTextureCubemapLayeredWidth"]                 = {"hipDeviceAttributeMaxTextureCubemapLayered",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 53
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
  m["cudaDevAttrMaxTextureCubemapLayeredLayers"]                = {"hipDeviceAttributeMaxTextureCubemapLayeredLayers",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 54
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH
  m["cudaDevAttrMaxSurface1DWidth"]                             = {"hipDeviceAttributeMaxSurface1D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 55
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH
  m["cudaDevAttrMaxSurface2DWidth"]                             = {"hipDeviceAttributeMaxSurface2D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 56
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT
  m["cudaDevAttrMaxSurface2DHeight"]                            = {"hipDeviceAttributeMaxSurface2D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 57
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH
  m["cudaDevAttrMaxSurface3DWidth"]                             = {"hipDeviceAttributeMaxSurface3D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 58
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT
  m["cudaDevAttrMaxSurface3DHeight"]                            = {"hipDeviceAttributeMaxSurface3D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 59
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH
  m["cudaDevAttrMaxSurface3DDepth"]                             = {"hipDeviceAttributeMaxSurface3D",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 60
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxSurface1DLayeredWidth"]                      = {"hipDeviceAttributeMaxSurface1DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 61
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
  m["cudaDevAttrMaxSurface1DLayeredLayers"]                     = {"hipDeviceAttributeMaxSurface1DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 62
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxSurface2DLayeredWidth"]                      = {"hipDeviceAttributeMaxSurface2DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 63
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
  // CUDA only
  m["cudaDevAttrMaxSurface2DLayeredHeight"]                     = {"hipDeviceAttributeMaxSurface2DLayered",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 64
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LA  YERS
  m["cudaDevAttrMaxSurface2DLayeredLayers"]                     = {"hipDeviceAttributeMaxSurface2DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 65
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH
  // CUDA only
  m["cudaDevAttrMaxSurfaceCubemapWidth"]                        = {"hipDeviceAttributeMaxSurfaceCubemap",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 66
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxSurfaceCubemapLayeredWidth"]                 = {"hipDeviceAttributeMaxSurfaceCubemapLayered",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 67
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
  m["cudaDevAttrMaxSurfaceCubemapLayeredLayers"]                = {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 68
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH
  m["cudaDevAttrMaxTexture1DLinearWidth"]                       = {"hipDeviceAttributeMaxTexture1DLinear",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 69
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture2DLinearWidth"]                       = {"hipDeviceAttributeMaxTexture2DLinear",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 70
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
  // CUDA only
  m["cudaDevAttrMaxTexture2DLinearHeight"]                      = {"hipDeviceAttributeMaxTexture2DLinear",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 71
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH
  // CUDA only
  m["cudaDevAttrMaxTexture2DLinearPitch"]                       = {"hipDeviceAttributeMaxTexture2DLinear",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 72
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture2DMipmappedWidth"]                    = {"hipDeviceAttributeMaxTexture2DMipmap",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 73
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
  // CUDA only
  m["cudaDevAttrMaxTexture2DMipmappedHeight"]                   = {"hipDeviceAttributeMaxTexture2DMipmap",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 74
  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  m["cudaDevAttrComputeCapabilityMajor"]                        = {"hipDeviceAttributeComputeCapabilityMajor",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 75
  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  m["cudaDevAttrComputeCapabilityMinor"]                        = {"hipDeviceAttributeComputeCapabilityMinor",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 76
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
  // CUDA only
  m["cudaDevAttrMaxTexture1DMipmappedWidth"]                    = {"hipDeviceAttributeMaxTexture1DMipmap",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 77
  // CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED
  // CUDA only
  m["cudaDevAttrStreamPrioritiesSupported"]                     = {"hipDeviceAttributeStreamPrioritiesSupported",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 78
  // CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
  // CUDA only
  m["cudaDevAttrGlobalL1CacheSupported"]                        = {"hipDeviceAttributeGlobalL1CacheSupported",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 79
  // CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED
  m["cudaDevAttrLocalL1CacheSupported"]                         = {"hipDeviceAttributeLocalL1CacheSupported",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 80
  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
  m["cudaDevAttrMaxSharedMemoryPerMultiprocessor"]              = {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 81
  // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
  m["cudaDevAttrMaxRegistersPerMultiprocessor"]                 = {"hipDeviceAttributeMaxRegistersPerMultiprocessor",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 82
  // CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
  m["cudaDevAttrManagedMemory"]                                 = {"hipDeviceAttributeManagedMemory",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 83
  // CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD
  m["cudaDevAttrIsMultiGpuBoard"]                               = {"hipDeviceAttributeIsMultiGpuBoard",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 84
  // CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID
  // CUDA only
  m["cudaDevAttrMultiGpuBoardGroupID"]                          = {"hipDeviceAttributeMultiGpuBoardGroupID",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 85
  // CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
  // CUDA only
  m["cudaDevAttrHostNativeAtomicSupported"]                     = {"hipDeviceAttributeHostNativeAtomicSupported",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 86
  // CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
  // CUDA only
  m["cudaDevAttrSingleToDoublePrecisionPerfRatio"]              = {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio",       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 87
  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
  m["cudaDevAttrPageableMemoryAccess"]                          = {"hipDeviceAttributePageableMemoryAccess",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 88
  // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  m["cudaDevAttrConcurrentManagedAccess"]                       = {"hipDeviceAttributeConcurrentManagedAccess",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 89
  // CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED
  // CUDA only
  m["cudaDevAttrComputePreemptionSupported"]                    = {"hipDeviceAttributeComputePreemptionSupported",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 90
  // CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
  // CUDA only
  m["cudaDevAttrCanUseHostPointerForRegisteredMem"]             = {"hipDeviceAttributeCanUseHostPointerForRegisteredMem",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 91
  // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS
  m["cudaDevAttrReserved92"]                                    = {"hipDeviceAttributeCanUseStreamMemOps",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 92
  // CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
  m["cudaDevAttrReserved93"]                                    = {"hipDeviceAttributeCanUse64BitStreamMemOps",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 93
  // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR
  m["cudaDevAttrReserved94"]                                    = {"hipDeviceAttributeCanUseStreamWaitValue",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 94
  // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
  m["cudaDevAttrCooperativeLaunch"]                             = {"hipDeviceAttributeCooperativeLaunch",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 95
  // CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH
  m["cudaDevAttrCooperativeMultiDeviceLaunch"]                  = {"hipDeviceAttributeCooperativeMultiDeviceLaunch",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED| CUDA_REMOVED}; // 96
  //
  m["cudaDevAttrReserved96"]                                    = {"hipDeviceAttributeCooperativeMultiDeviceLaunch",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 96

  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
  // CUDA only
  m["cudaDevAttrMaxSharedMemoryPerBlockOptin"]                  = {"hipDeviceAttributeSharedMemPerBlockOptin",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 97
  // CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES
  m["cudaDevAttrCanFlushRemoteWrites"]                          = {"hipDeviceAttributeCanFlushRemoteWrites",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 98
  // CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED
  m["cudaDevAttrHostRegisterSupported"]                         = {"hipDeviceAttributeHostRegisterSupported",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 99
  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
  m["cudaDevAttrPageableMemoryAccessUsesHostPageTables"]        = {"hipDeviceAttributePageableMemoryAccessUsesHostPageTables", "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 100
  // CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
  m["cudaDevAttrDirectManagedMemAccessFromHost"]                = {"hipDeviceAttributeDirectManagedMemAccessFromHost",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 101
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR
  // CUDA only
  m["cudaDevAttrMaxBlocksPerMultiprocessor"]                    = {"hipDeviceAttributeMaxBlocksPerMultiprocessor",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 106
  // CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE
  m["cudaDevAttrMaxPersistingL2CacheSize"]                      = {"hipDeviceAttributeMaxPersistingL2CacheSize",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 108
  // CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE
  m["cudaDevAttrMaxAccessPolicyWindowSize"]                     = {"hipDeviceAttributeMaxAccessPolicyWindowSize",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 109
  // CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK
  m["cudaDevAttrReservedSharedMemoryPerBlock"]                  = {"hipDeviceAttributeReservedSharedMemoryPerBlock",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 111
  // CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED
  m["cudaDevAttrSparseCudaArraySupported"]                      = {"hipDeviceAttributeSparseCudaArraySupported",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 112
  // CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED
  m["cudaDevAttrHostRegisterReadOnlySupported"]                 = {"hipDeviceAttributeReadOnlyHostRestigerSupported",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 113
  // CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
  m["cudaDevAttrMaxTimelineSemaphoreInteropSupported"]          = {"hipDeviceAttributeMaxTimelineSemaphoreInteropSupported",   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}; // 114
  // CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
  m["cudaDevAttrTimelineSemaphoreInteropSupported"]             = {"hipDeviceAttributeTimelineSemaphoreInteropSupported",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 114
  // CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED
  m["cudaDevAttrMemoryPoolsSupported"]                          = {"hipDeviceAttributeMemoryPoolsSupported",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 115
  // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
  m["cudaDevAttrGPUDirectRDMASupported"]                        = {"hipDeviceAttributeGPUDirectRDMASupported",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 116
  // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
  m["cudaDevAttrGPUDirectRDMAFlushWritesOptions"]               = {"hipDeviceAttributeGpuDirectRdmaFlushWritesOptions",        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 117
  // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
  m["cudaDevAttrGPUDirectRDMAWritesOrdering"]                   = {"hipDeviceAttributeGpuDirectRdmaWritesOrdering",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 118
  // CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES
  m["cudaDevAttrMemoryPoolSupportedHandleTypes"]                = {"hipDeviceAttributeMempoolSupportedHandleTypes",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 119
  // CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH
  m["cudaDevAttrClusterLaunch"]                                 = {"hipDeviceAttributeClusterLaunch",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 120
  // CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED
  m["cudaDevAttrDeferredMappingCudaArraySupported"]             = {"hipDeviceAttributeDeferredMappingCudaArraySupported",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 121
  // CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2
  m["cudaDevAttrReserved122"]                                   = {"hipDevAttrReserved122",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 122
  // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2
  m["cudaDevAttrReserved123"]                                   = {"hipDevAttrReserved123",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 123
  // CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
  m["cudaDevAttrReserved124"]                                   = {"hipDevAttrReserved124",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 124
  // CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED
  m["cudaDevAttrIpcEventSupport"]                               = {"hipDevAttrIpcEventSupport",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 125
  // CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT
  m["cudaDevAttrMemSyncDomainCount"]                            = {"hipDevAttrMemSyncDomainCount",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 126
  // CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED
  m["cudaDevAttrReserved127"]                                   = {"hipDeviceAttributeTensorMapAccessSupported",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 127
  // CUDA only
  m["cudaDevAttrReserved128"]                                   = {"hipDevAttrReserved128",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 128
  // CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS
  m["cudaDevAttrReserved129"]                                   = {"hipDeviceAttributeUnifiedFunctionPointers",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 129
  // CU_DEVICE_ATTRIBUTE_NUMA_CONFIG
  m["cudaDevAttrNumaConfig"]                                    = {"hipDeviceAttributeNumaConfig",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 130
  // CU_DEVICE_ATTRIBUTE_NUMA_ID
  m["cudaDevAttrNumaId"]                                        = {"hipDeviceAttributeNumaId",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 131
  // CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
  m["cudaDevAttrReserved132"]                                   = {"hipDeviceAttributeMulticastSupported",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 132
  // CU_DEVICE_ATTRIBUTE_MPS_ENABLED
  m["cudaDevAttrMpsEnabled"]                                    = {"hipDeviceAttributeMpsEnables",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 133
  // CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
  m["cudaDevAttrHostNumaId"]                                    = {"hipDeviceAttributeHostNumaId",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 134
  // CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED
  m["cudaDevAttrD3D12CigSupported"]                             = {"hipDeviceAttributeD3D12CigSupported",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 135
  // CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED
  m["cudaDevAttrVulkanCigSupported"]                            = {"hipDevAttrVulkanCigSupported",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 138
  // CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID
  m["cudaDevAttrGpuPciDeviceId"]                                = {"hipDeviceAttributePciDeviceId",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 139
  // CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID
  m["cudaDevAttrGpuPciSubsystemId"]                             = {"hipDeviceAttributeGpuPciSubsystemId",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 140
  //
  m["cudaDevAttrReserved141"]                                   = {"hipDevAttrReserved141",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 141
  // CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED
  m["cudaDevAttrHostNumaMemoryPoolsSupported"]                  = {"hipDeviceAttributeHostNumaMemoryPoolsSupported",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 142
  // CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED
  m["cudaDevAttrHostNumaMultinodeIpcSupported"]                 = {"hipDeviceAttributeHostNumaMultinodeIpcSupported",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 143
  // CU_DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED
  m["cudaDevAttrHostMemoryPoolsSupported"]                      = {"hipDeviceAttributeHostMemoryPoolsSupported",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 144
  //
  m["cudaDevAttrReserved145"]                                   = {"hipDevAttrReserved145",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 145
  // CU_DEVICE_ATTRIBUTE_ONLY_PARTIAL_HOST_NATIVE_ATOMIC_SUPPORTED
  m["cudaDevAttrOnlyPartialHostNativeAtomicSupported"]          = {"hipDevAttributeOnlyPartialHostNativeAtomicSupported",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 147
  // CU_DEVICE_ATTRIBUTE_MAX
  m["cudaDevAttrMax"]                                           = {"hipDeviceAttributeMax",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevice_P2PAttribute
  m["cudaDeviceP2PAttr"]                                        = {"hipDeviceP2PAttr",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaDeviceP2PAttr enum values
  // CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01
  m["cudaDevP2PAttrPerformanceRank"]                            = {"hipDevP2PAttrPerformanceRank",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02
  m["cudaDevP2PAttrAccessSupported"]                            = {"hipDevP2PAttrAccessSupported",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03
  m["cudaDevP2PAttrNativeAtomicSupported"]                      = {"hipDevP2PAttrNativeAtomicSupported",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 0x04
  m["cudaDevP2PAttrCudaArrayAccessSupported"]                   = {"hipDevP2PAttrHipArrayAccessSupported",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED
  m["cudaDevP2PAttrOnlyPartialNativeAtomicSupported"]           = {"hipDevP2PAttrOnlyPartialNativeAtomicSupported",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 5

  // cudaEGL.h - presented only on Linux in nvidia-cuda-dev package
  // CUeglColorFormat
  m["cudaEglColorFormat"]                                       = {"hipEglColorFormat",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEglColorFormat enum values
  // CU_EGL_COLOR_FORMAT_YUV420_PLANAR = 0x00
  m["cudaEglColorFormatYUV420Planar"]                           = {"hipEglColorFormatYUV420Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR= 0x01
  m["cudaEglColorFormatYUV420SemiPlanar"]                       = {"hipEglColorFormatYUV420SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_EGL_COLOR_FORMAT_YUV422_PLANAR = 0x02
  m["cudaEglColorFormatYUV422Planar"]                           = {"hipEglColorFormatYUV422Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2
  // CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR = 0x03
  m["cudaEglColorFormatYUV422SemiPlanar"]                       = {"hipEglColorFormatYUV422SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 3
  // CU_EGL_COLOR_FORMAT_RGB = 0x04
  m["cudaEglColorFormatRGB"]                                    = {"hipEglColorFormatRGB",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 4
  // CU_EGL_COLOR_FORMAT_BGR = 0x05
  m["cudaEglColorFormatBGR"]                                    = {"hipEglColorFormatBGR",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 5
  // CU_EGL_COLOR_FORMAT_ARGB = 0x06
  m["cudaEglColorFormatARGB"]                                   = {"hipEglColorFormatARGB",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 6
  // CU_EGL_COLOR_FORMAT_RGBA = 0x07
  m["cudaEglColorFormatRGBA"]                                   = {"hipEglColorFormatRGBA",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 7
  // CU_EGL_COLOR_FORMAT_L = 0x08
  m["cudaEglColorFormatL"]                                      = {"hipEglColorFormatL",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8
  // CU_EGL_COLOR_FORMAT_R = 0x09
  m["cudaEglColorFormatR"]                                      = {"hipEglColorFormatR",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 9
  // CU_EGL_COLOR_FORMAT_YUV444_PLANAR = 0x0A
  m["cudaEglColorFormatYUV444Planar"]                           = {"hipEglColorFormatYUV444Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 10
  // CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR = 0x0B
  m["cudaEglColorFormatYUV444SemiPlanar"]                       = {"hipEglColorFormatYUV444SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 11
  // CU_EGL_COLOR_FORMAT_YUYV_422 = 0x0C
  m["cudaEglColorFormatYUYV422"]                                = {"hipEglColorFormatYUYV422",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 12
  // CU_EGL_COLOR_FORMAT_UYVY_422 = 0x0D
  m["cudaEglColorFormatUYVY422"]                                = {"hipEglColorFormatUYVY422",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 13
  // CU_EGL_COLOR_FORMAT_ABGR = 0x0E
  m["cudaEglColorFormatABGR"]                                   = {"hipEglColorFormatABGR",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 14
  // CU_EGL_COLOR_FORMAT_BGRA = 0x0F
  m["cudaEglColorFormatBGRA"]                                   = {"hipEglColorFormatBGRA",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 15
  // CU_EGL_COLOR_FORMAT_A = 0x10
  m["cudaEglColorFormatA"]                                      = {"hipEglColorFormatA",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 16
  // CU_EGL_COLOR_FORMAT_RG = 0x11
  m["cudaEglColorFormatRG"]                                     = {"hipEglColorFormatRG",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 17
  // CU_EGL_COLOR_FORMAT_AYUV = 0x12
  m["cudaEglColorFormatAYUV"]                                   = {"hipEglColorFormatAYUV",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 18
  // CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR = 0x13
  m["cudaEglColorFormatYVU444SemiPlanar"]                       = {"hipEglColorFormatYVU444SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 19
  // CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR = 0x14
  m["cudaEglColorFormatYVU422SemiPlanar"]                       = {"hipEglColorFormatYVU422SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 20
  // CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR = 0x15
  m["cudaEglColorFormatYVU420SemiPlanar"]                       = {"hipEglColorFormatYVU420SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 21
  // CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR = 0x16
  m["cudaEglColorFormatY10V10U10_444SemiPlanar"]                = {"hipEglColorFormatY10V10U10_444SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 22
  // CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR = 0x17
  m["cudaEglColorFormatY10V10U10_420SemiPlanar"]                = {"hipEglColorFormatY10V10U10_420SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 23
  // CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR = 0x18
  m["cudaEglColorFormatY12V12U12_444SemiPlanar"]                = {"hipEglColorFormatY12V12U12_444SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 24
  // CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR = 0x19
  m["cudaEglColorFormatY12V12U12_420SemiPlanar"]                = {"hipEglColorFormatY12V12U12_420SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 25
  // CU_EGL_COLOR_FORMAT_VYUY_ER = 0x1A
  m["cudaEglColorFormatVYUY_ER"]                                = {"hipEglColorFormatVYUY_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 26
  // CU_EGL_COLOR_FORMAT_UYVY_ER = 0x1B
  m["cudaEglColorFormatUYVY_ER"]                                = {"hipEglColorFormatUYVY_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 27
  // CU_EGL_COLOR_FORMAT_YUYV_ER = 0x1C
  m["cudaEglColorFormatYUYV_ER"]                                = {"hipEglColorFormatYUYV_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 28
  // CU_EGL_COLOR_FORMAT_YVYU_ER = 0x1D
  m["cudaEglColorFormatYVYU_ER"]                                = {"hipEglColorFormatYVYU_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 29
  // CU_EGL_COLOR_FORMAT_YUV_ER = 0x1E
  m["cudaEglColorFormatYUV_ER"]                                 = {"hipEglColorFormatYUV_ER",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 30
  // CU_EGL_COLOR_FORMAT_YUVA_ER = 0x1F
  m["cudaEglColorFormatYUVA_ER"]                                = {"hipEglColorFormatYUVA_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 31
  // CU_EGL_COLOR_FORMAT_AYUV_ER = 0x20
  m["cudaEglColorFormatAYUV_ER"]                                = {"hipEglColorFormatAYUV_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 32
  // CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER = 0x21
  m["cudaEglColorFormatYUV444Planar_ER"]                        = {"hipEglColorFormatYUV444Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 33
  // CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER = 0x22
  m["cudaEglColorFormatYUV422Planar_ER"]                        = {"hipEglColorFormatYUV422Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 34
  // CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER = 0x23
  m["cudaEglColorFormatYUV420Planar_ER"]                        = {"hipEglColorFormatYUV420Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 35
  // CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER = 0x24
  m["cudaEglColorFormatYUV444SemiPlanar_ER"]                    = {"hipEglColorFormatYUV444SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 36
  // CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER = 0x25
  m["cudaEglColorFormatYUV422SemiPlanar_ER"]                    = {"hipEglColorFormatYUV422SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 37
  // CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER = 0x26
  m["cudaEglColorFormatYUV420SemiPlanar_ER"]                    = {"hipEglColorFormatYUV420SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 38
  // CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER = 0x27
  m["cudaEglColorFormatYVU444Planar_ER"]                        = {"hipEglColorFormatYVU444Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 39
  // CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER = 0x28
  m["cudaEglColorFormatYVU422Planar_ER"]                        = {"hipEglColorFormatYVU422Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 40
  // CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER = 0x29
  m["cudaEglColorFormatYVU420Planar_ER"]                        = {"hipEglColorFormatYVU420Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 41
  // CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER = 0x2A
  m["cudaEglColorFormatYVU444SemiPlanar_ER"]                    = {"hipEglColorFormatYVU444SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 42
  // CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER = 0x2B
  m["cudaEglColorFormatYVU422SemiPlanar_ER"]                    = {"hipEglColorFormatYVU422SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 43
  // CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER = 0x2C
  m["cudaEglColorFormatYVU420SemiPlanar_ER"]                    = {"hipEglColorFormatYVU420SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 44
  // CU_EGL_COLOR_FORMAT_BAYER_RGGB = 0x2D
  m["cudaEglColorFormatBayerRGGB"]                              = {"hipEglColorFormatBayerRGGB",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 45
  // CU_EGL_COLOR_FORMAT_BAYER_BGGR = 0x2E
  m["cudaEglColorFormatBayerBGGR"]                              = {"hipEglColorFormatBayerBGGR",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 46
  // CU_EGL_COLOR_FORMAT_BAYER_GRBG = 0x2F
  m["cudaEglColorFormatBayerGRBG"]                              = {"hipEglColorFormatBayerGRBG",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 47
  // CU_EGL_COLOR_FORMAT_BAYER_GBRG = 0x30
  m["cudaEglColorFormatBayerGBRG"]                              = {"hipEglColorFormatBayerGBRG",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 48
  // CU_EGL_COLOR_FORMAT_BAYER10_RGGB = 0x31
  m["cudaEglColorFormatBayer10RGGB"]                            = {"hipEglColorFormatBayer10RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 49
  // CU_EGL_COLOR_FORMAT_BAYER10_BGGR = 0x32
  m["cudaEglColorFormatBayer10BGGR"]                            = {"hipEglColorFormatBayer10BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 50
  // CU_EGL_COLOR_FORMAT_BAYER10_GRBG = 0x33
  m["cudaEglColorFormatBayer10GRBG"]                            = {"hipEglColorFormatBayer10GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 51
  // CU_EGL_COLOR_FORMAT_BAYER10_GBRG = 0x34
  m["cudaEglColorFormatBayer10GBRG"]                            = {"hipEglColorFormatBayer10GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 52
  // CU_EGL_COLOR_FORMAT_BAYER12_RGGB = 0x35
  m["cudaEglColorFormatBayer12RGGB"]                            = {"hipEglColorFormatBayer12RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 53
  // CU_EGL_COLOR_FORMAT_BAYER12_BGGR = 0x36
  m["cudaEglColorFormatBayer12BGGR"]                            = {"hipEglColorFormatBayer12BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 54
  // CU_EGL_COLOR_FORMAT_BAYER12_GRBG = 0x37
  m["cudaEglColorFormatBayer12GRBG"]                            = {"hipEglColorFormatBayer12GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 55
  // CU_EGL_COLOR_FORMAT_BAYER12_GBRG = 0x38
  m["cudaEglColorFormatBayer12GBRG"]                            = {"hipEglColorFormatBayer12GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 56
  // CU_EGL_COLOR_FORMAT_BAYER14_RGGB = 0x39
  m["cudaEglColorFormatBayer14RGGB"]                            = {"hipEglColorFormatBayer14RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 57
  // CU_EGL_COLOR_FORMAT_BAYER14_BGGR = 0x3A
  m["cudaEglColorFormatBayer14BGGR"]                            = {"hipEglColorFormatBayer14BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 58
  // CU_EGL_COLOR_FORMAT_BAYER14_GRBG = 0x3B
  m["cudaEglColorFormatBayer14GRBG"]                            = {"hipEglColorFormatBayer14GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 59
  // CU_EGL_COLOR_FORMAT_BAYER14_GBRG = 0x3C
  m["cudaEglColorFormatBayer14GBRG"]                            = {"hipEglColorFormatBayer14GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 60
  // CU_EGL_COLOR_FORMAT_BAYER20_RGGB = 0x3D
  m["cudaEglColorFormatBayer20RGGB"]                            = {"hipEglColorFormatBayer20RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 61
  // CU_EGL_COLOR_FORMAT_BAYER20_BGGR = 0x3E
  m["cudaEglColorFormatBayer20BGGR"]                            = {"hipEglColorFormatBayer20BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 62
  // CU_EGL_COLOR_FORMAT_BAYER20_GRBG = 0x3F
  m["cudaEglColorFormatBayer20GRBG"]                            = {"hipEglColorFormatBayer20GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 63
  // CU_EGL_COLOR_FORMAT_BAYER20_GBRG = 0x40
  m["cudaEglColorFormatBayer20GBRG"]                            = {"hipEglColorFormatBayer20GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 64
  // CU_EGL_COLOR_FORMAT_YVU444_PLANAR = 0x41
  m["cudaEglColorFormatYVU444Planar"]                           = {"hipEglColorFormatYVU444Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 65
  // CU_EGL_COLOR_FORMAT_YVU422_PLANAR = 0x42
  m["cudaEglColorFormatYVU422Planar"]                           = {"hipEglColorFormatYVU422Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 66
  // CU_EGL_COLOR_FORMAT_YVU420_PLANAR = 0x43
  m["cudaEglColorFormatYVU420Planar"]                           = {"hipEglColorFormatYVU420Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 67
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB = 0x44
  m["cudaEglColorFormatBayerIspRGGB"]                           = {"hipEglColorFormatBayerIspRGGB",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 68
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR = 0x45
  m["cudaEglColorFormatBayerIspBGGR"]                           = {"hipEglColorFormatBayerIspBGGR",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 69
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG = 0x46
  m["cudaEglColorFormatBayerIspGRBG"]                           = {"hipEglColorFormatBayerIspGRBG",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 70
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG = 0x47
  m["cudaEglColorFormatBayerIspGBRG"]                           = {"hipEglColorFormatBayerIspGBRG",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 71
  //
  m["cudaEglColorFormatBayerBCCR"]                              = {"hipEglColorFormatBayerBCCR",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 72
  //
  m["cudaEglColorFormatBayerRCCB"]                              = {"hipEglColorFormatBayerRCCB",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 73
  //
  m["cudaEglColorFormatBayerCRBC"]                              = {"hipEglColorFormatBayerCRBC",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 74
  //
  m["cudaEglColorFormatBayerCBRC"]                              = {"hipEglColorFormatBayerCBRC",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 75
  //
  m["cudaEglColorFormatBayer10CCCC"]                            = {"hipEglColorFormatBayer10CCCC",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 76
  //
  m["cudaEglColorFormatBayer12BCCR"]                            = {"hipEglColorFormatBayer12BCCR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 77
  //
  m["cudaEglColorFormatBayer12RCCB"]                            = {"hipEglColorFormatBayer12RCCB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 78
  //
  m["cudaEglColorFormatBayer12CRBC"]                            = {"hipEglColorFormatBayer12CRBC",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 79
  //
  m["cudaEglColorFormatBayer12CBRC"]                            = {"hipEglColorFormatBayer12CBRC",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 80
  //
  m["cudaEglColorFormatBayer12CCCC"]                            = {"hipEglColorFormatBayer12CCCC",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 81
  //
  m["cudaEglColorFormatY"]                                      = {"hipEglColorFormatY",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 82
  //
  m["cudaEglColorFormatYUV420SemiPlanar_2020"]                  = {"hipEglColorFormatYUV420SemiPlanar_2020",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 83
  //
  m["cudaEglColorFormatYVU420SemiPlanar_2020"]                  = {"hipEglColorFormatYVU420SemiPlanar_2020",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 84
  //
  m["cudaEglColorFormatYUV420Planar_2020"]                      = {"hipEglColorFormatYUV420Planar_2020",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 85
  //
  m["cudaEglColorFormatYVU420Planar_2020"]                      = {"hipEglColorFormatYVU420Planar_2020",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 86
  //
  m["cudaEglColorFormatYUV420SemiPlanar_709"]                   = {"hipEglColorFormatYUV420SemiPlanar_709",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 87
  //
  m["cudaEglColorFormatYVU420SemiPlanar_709"]                   = {"hipEglColorFormatYVU420SemiPlanar_709",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 88
  //
  m["cudaEglColorFormatYUV420Planar_709"]                       = {"hipEglColorFormatYUV420Planar_709",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 89
  //
  m["cudaEglColorFormatYVU420Planar_709"]                       = {"hipEglColorFormatYVU420Planar_709",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 90
  //
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_709"]            = {"hipEglColorFormatY10V10U10_420SemiPlanar_709",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 91
  //
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_2020"]           = {"hipEglColorFormatY10V10U10_420SemiPlanar_2020",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 92
  //
  m["cudaEglColorFormatY10V10U10_422SemiPlanar_2020"]           = {"hipEglColorFormatY10V10U10_422SemiPlanar_2020",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 93
  //
  m["cudaEglColorFormatY10V10U10_422SemiPlanar"]                = {"hipEglColorFormatY10V10U10_422SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 94
  //
  m["cudaEglColorFormatY10V10U10_422SemiPlanar_709"]            = {"hipEglColorFormatY10V10U10_422SemiPlanar_709",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 95
  //
  m["cudaEglColorFormatY_ER"]                                   = {"hipEglColorFormatY_ER",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 96
  //
  m["cudaEglColorFormatY_709_ER"]                               = {"hipEglColorFormatY_709_ER",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 97
  //
  m["cudaEglColorFormatY10_ER"]                                 = {"hipEglColorFormatY10_ER",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 98
  //
  m["cudaEglColorFormatY10_709_ER"]                             = {"hipEglColorFormatY10_709_ER",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 99
  //
  m["cudaEglColorFormatY12_ER"]                                 = {"hipEglColorFormatY12_ER",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 100
  //
  m["cudaEglColorFormatY12_709_ER"]                             = {"hipEglColorFormatY12_709_ER",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 101
  //
  m["cudaEglColorFormatYUVA"]                                   = {"hipEglColorFormatYUVA",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 102
  //
  m["cudaEglColorFormatYVYU"]                                   = {"hipEglColorFormatYVYU",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 104
  //
  m["cudaEglColorFormatVYUY"]                                   = {"hipEglColorFormatVYUY",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 105
  //
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_ER"]             = {"hipEglColorFormatY10V10U10_420SemiPlanar_ER",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 106
  //
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER"]         = {"hipEglColorFormatY10V10U10_420SemiPlanar_709_ER",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 107
  //
  m["cudaEglColorFormatY10V10U10_444SemiPlanar_ER"]             = {"hipEglColorFormatY10V10U10_444SemiPlanar_ER",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 108
  //
  m["cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER"]         = {"hipEglColorFormatY10V10U10_444SemiPlanar_709_ER",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 109
  //
  m["cudaEglColorFormatY12V12U12_420SemiPlanar_ER"]             = {"hipEglColorFormatY12V12U12_420SemiPlanar_ER",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 110
  //
  m["cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER"]         = {"hipEglColorFormatY12V12U12_420SemiPlanar_709_ER",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 111
  //
  m["cudaEglColorFormatY12V12U12_444SemiPlanar_ER"]             = {"hipEglColorFormatY12V12U12_444SemiPlanar_ER",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 112
  //
  m["cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER"]         = {"hipEglColorFormatY12V12U12_444SemiPlanar_709_ER",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 113
  //
  m["cudaEglColorFormatUYVY709"]                                = {"hipEglColorFormatUYVY709",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 114
  //
  m["cudaEglColorFormatUYVY709_ER"]                             = {"hipEglColorFormatUYVY709_ER",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 115
  //
  m["cudaEglColorFormatUYVY2020"]                               = {"hipEglColorFormatUYVY2020",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 116

  // CUeglFrameType
  m["cudaEglFrameType"]                                         = {"hipEglFrameType",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEglFrameType enum values
  // CU_EGL_FRAME_TYPE_ARRAY
  m["cudaEglFrameTypeArray"]                                    = {"hipEglFrameTypeArray",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_EGL_FRAME_TYPE_PITCH
  m["cudaEglFrameTypePitch"]                                    = {"hipEglFrameTypePitch",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1

  // CUeglResourceLocationFlags
  m["cudaEglResourceLocationFlags"]                             = {"hipEglResourceLocationFlags",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEglResourceLocationFlagss enum values
  // CU_EGL_RESOURCE_LOCATION_SYSMEM
  m["cudaEglResourceLocationSysmem"]                            = {"hipEglResourceLocationSysmem",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x00
  // CU_EGL_RESOURCE_LOCATION_VIDMEM
  m["cudaEglResourceLocationVidmem"]                            = {"hipEglResourceLocationVidmem",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x01

  // CUresult
  m["cudaError"]                                                = {"hipError_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  m["cudaError_t"]                                              = {"hipError_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaError enum values
  // CUDA_SUCCESS
  m["cudaSuccess"]                                              = {"hipSuccess",                                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CUDA_ERROR_INVALID_VALUE
  m["cudaErrorInvalidValue"]                                    = {"hipErrorInvalidValue",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CUDA_ERROR_OUT_OF_MEMORY
  m["cudaErrorMemoryAllocation"]                                = {"hipErrorOutOfMemory",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CUDA_ERROR_NOT_INITIALIZED
  m["cudaErrorInitializationError"]                             = {"hipErrorNotInitialized",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CUDA_ERROR_DEINITIALIZED
  m["cudaErrorCudartUnloading"]                                 = {"hipErrorDeinitialized",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CUDA_ERROR_PROFILER_DISABLED
  m["cudaErrorProfilerDisabled"]                                = {"hipErrorProfilerDisabled",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 5
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_NOT_INITIALIZED
  m["cudaErrorProfilerNotInitialized"]                          = {"hipErrorProfilerNotInitialized",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 6
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STARTED
  m["cudaErrorProfilerAlreadyStarted"]                          = {"hipErrorProfilerAlreadyStarted",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 7
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STOPPED
  m["cudaErrorProfilerAlreadyStopped"]                          = {"hipErrorProfilerAlreadyStopped",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 8
  // no analogue
  m["cudaErrorInvalidConfiguration"]                            = {"hipErrorInvalidConfiguration",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 9
  //
  m["cudaErrorVersionTranslation"]                              = {"hipErrorVersionTranslation",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 10
  // no analogue
  m["cudaErrorInvalidPitchValue"]                               = {"hipErrorInvalidPitchValue",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 12
  // no analogue
  m["cudaErrorInvalidSymbol"]                                   = {"hipErrorInvalidSymbol",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 13
  // Deprecated since CUDA 10.1
  // no analogue
  m["cudaErrorInvalidHostPointer"]                              = {"hipErrorInvalidHostPointer",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 16
  // Deprecated since CUDA 10.1
  // no analogue
  m["cudaErrorInvalidDevicePointer"]                            = {"hipErrorInvalidDevicePointer",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 17
  // no analogue
  m["cudaErrorInvalidTexture"]                                  = {"hipErrorInvalidTexture",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 18
  // no analogue
  m["cudaErrorInvalidTextureBinding"]                           = {"hipErrorInvalidTextureBinding",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 19
  // no analogue
  m["cudaErrorInvalidChannelDescriptor"]                        = {"hipErrorInvalidChannelDescriptor",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 20
  // no analogue
  m["cudaErrorInvalidMemcpyDirection"]                          = {"hipErrorInvalidMemcpyDirection",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 21
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorAddressOfConstant"]                               = {"hipErrorAddressOfConstant",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 22
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorTextureFetchFailed"]                              = {"hipErrorTextureFetchFailed",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 23
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorTextureNotBound"]                                 = {"hipErrorTextureNotBound",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 24
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorSynchronizationError"]                            = {"hipErrorSynchronizationError",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 25
  // no analogue
  m["cudaErrorInvalidFilterSetting"]                            = {"hipErrorInvalidFilterSetting",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 26
  // no analogue
  m["cudaErrorInvalidNormSetting"]                              = {"hipErrorInvalidNormSetting",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 27
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorMixedDeviceExecution"]                            = {"hipErrorMixedDeviceExecution",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 28
  // Deprecated since CUDA 4.1
  // no analogue
  m["cudaErrorNotYetImplemented"]                               = {"hipErrorNotYetImplemented",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 31
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorMemoryValueTooLarge"]                             = {"hipErrorMemoryValueTooLarge",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 32
  // CUDA_ERROR_STUB_LIBRARY
  m["cudaErrorStubLibrary"]                                     = {"hipErrorStubLibrary",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 34
  // no analogue
  m["cudaErrorInsufficientDriver"]                              = {"hipErrorInsufficientDriver",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 35
  // CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER
  m["cudaErrorCallRequiresNewerDriver"]                         = {"hipErrorCallRequiresNewerDriver",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 36
  // no analogue
  m["cudaErrorInvalidSurface"]                                  = {"hipErrorInvalidSurface",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 37
  // no analogue
  m["cudaErrorDuplicateVariableName"]                           = {"hipErrorDuplicateVariableName",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 43
  // no analogue
  m["cudaErrorDuplicateTextureName"]                            = {"hipErrorDuplicateTextureName",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 44
  // no analogue
  m["cudaErrorDuplicateSurfaceName"]                            = {"hipErrorDuplicateSurfaceName",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 45
  // CUDA_ERROR_DEVICE_UNAVAILABLE
  m["cudaErrorDevicesUnavailable"]                              = {"hipErrorDeviceUnavailable",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 46
  // no analogue
  m["cudaErrorIncompatibleDriverContext"]                       = {"hipErrorIncompatibleDriverContext",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 49
  // no analogue
  m["cudaErrorMissingConfiguration"]                            = {"hipErrorMissingConfiguration",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 52
  // Deprecated since CUDA 3.1
  // no analogue
  m["cudaErrorPriorLaunchFailure"]                              = {"hipErrorPriorLaunchFailure",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 53
  // no analogue
  m["cudaErrorLaunchMaxDepthExceeded"]                          = {"hipErrorLaunchMaxDepthExceeded",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 65
  // no analogue
  m["cudaErrorLaunchFileScopedTex"]                             = {"hipErrorLaunchFileScopedTex",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 66
  // no analogue
  m["cudaErrorLaunchFileScopedSurf"]                            = {"hipErrorLaunchFileScopedSurf",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 67
  // no analogue
  m["cudaErrorSyncDepthExceeded"]                               = {"hipErrorSyncDepthExceeded",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 68
  // no analogue
  m["cudaErrorLaunchPendingCountExceeded"]                      = {"hipErrorLaunchPendingCountExceeded",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 69
  // no analogue
  m["cudaErrorInvalidDeviceFunction"]                           = {"hipErrorInvalidDeviceFunction",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 98
  // CUDA_ERROR_NO_DEVICE
  m["cudaErrorNoDevice"]                                        = {"hipErrorNoDevice",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 100
  // CUDA_ERROR_INVALID_DEVICE
  m["cudaErrorInvalidDevice"]                                   = {"hipErrorInvalidDevice",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 101
  // CUDA_ERROR_DEVICE_NOT_LICENSED
  m["cudaErrorDeviceNotLicensed"]                               = {"hipErrorDeviceNotLicensed",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 102
  // no analogue
  m["cudaErrorSoftwareValidityNotEstablished"]                  = {"hipErrorSoftwareValidityNotEstablished",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 103
  // no analogue
  m["cudaErrorStartupFailure"]                                  = {"hipErrorStartupFailure",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 127
  // CUDA_ERROR_INVALID_IMAGE
  m["cudaErrorInvalidKernelImage"]                              = {"hipErrorInvalidImage",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 200
  // CUDA_ERROR_INVALID_CONTEXT
  m["cudaErrorDeviceUninitialized"]                             = {"hipErrorInvalidContext",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 201
  // CUDA_ERROR_MAP_FAILED
  m["cudaErrorMapBufferObjectFailed"]                           = {"hipErrorMapFailed",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 205
  // CUDA_ERROR_UNMAP_FAILED
  m["cudaErrorUnmapBufferObjectFailed"]                         = {"hipErrorUnmapFailed",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 206
  // CUDA_ERROR_ARRAY_IS_MAPPED
  m["cudaErrorArrayIsMapped"]                                   = {"hipErrorArrayIsMapped",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 207
  // CUDA_ERROR_ALREADY_MAPPED
  m["cudaErrorAlreadyMapped"]                                   = {"hipErrorAlreadyMapped",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 208
  // CUDA_ERROR_NO_BINARY_FOR_GPU
  m["cudaErrorNoKernelImageForDevice"]                          = {"hipErrorNoBinaryForGpu",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 209
  // CUDA_ERROR_ALREADY_ACQUIRED
  m["cudaErrorAlreadyAcquired"]                                 = {"hipErrorAlreadyAcquired",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 210
  // CUDA_ERROR_NOT_MAPPED
  m["cudaErrorNotMapped"]                                       = {"hipErrorNotMapped",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 211
  // CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  m["cudaErrorNotMappedAsArray"]                                = {"hipErrorNotMappedAsArray",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 212
  // CUDA_ERROR_NOT_MAPPED_AS_POINTER
  m["cudaErrorNotMappedAsPointer"]                              = {"hipErrorNotMappedAsPointer",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 213
  // CUDA_ERROR_ECC_UNCORRECTABLE
  m["cudaErrorECCUncorrectable"]                                = {"hipErrorECCNotCorrectable",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 214
  // CUDA_ERROR_UNSUPPORTED_LIMIT
  m["cudaErrorUnsupportedLimit"]                                = {"hipErrorUnsupportedLimit",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 215
  // CUDA_ERROR_CONTEXT_ALREADY_IN_USE
  m["cudaErrorDeviceAlreadyInUse"]                              = {"hipErrorContextAlreadyInUse",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 216
  // CUDA_ERROR_PEER_ACCESS_UNSUPPORTED
  m["cudaErrorPeerAccessUnsupported"]                           = {"hipErrorPeerAccessUnsupported",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 217
  // CUDA_ERROR_INVALID_PTX
  m["cudaErrorInvalidPtx"]                                      = {"hipErrorInvalidKernelFile",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 218
  // CUDA_ERROR_INVALID_GRAPHICS_CONTEXT
  m["cudaErrorInvalidGraphicsContext"]                          = {"hipErrorInvalidGraphicsContext",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 219
  // CUDA_ERROR_NVLINK_UNCORRECTABLE
  m["cudaErrorNvlinkUncorrectable"]                             = {"hipErrorNvlinkUncorrectable",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 220
  // CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  m["cudaErrorJitCompilerNotFound"]                             = {"hipErrorJitCompilerNotFound",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 221
  // CUDA_ERROR_UNSUPPORTED_PTX_VERSION
  m["cudaErrorUnsupportedPtxVersion"]                           = {"hipErrorUnsupportedPtxVersion",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 222
  // CUDA_ERROR_JIT_COMPILATION_DISABLED
  m["cudaErrorJitCompilationDisabled"]                          = {"hipErrorJitCompilationDisabled",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 223
  // CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY
  m["cudaErrorUnsupportedExecAffinity"]                         = {"hipErrorUnsupportedExecAffinity",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 224
  // CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC
  m["cudaErrorUnsupportedDevSideSync"]                          = {"hipErrorUnsupportedDevSideSync",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 225
  // CUDA_ERROR_CONTAINED
  m["cudaErrorContained"]                                       = {"hipErrorContained",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 226
  // CUDA_ERROR_INVALID_SOURCE
  m["cudaErrorInvalidSource"]                                   = {"hipErrorInvalidSource",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 300
  // CUDA_ERROR_FILE_NOT_FOUND
  m["cudaErrorFileNotFound"]                                    = {"hipErrorFileNotFound",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 301
  // CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND
  m["cudaErrorSharedObjectSymbolNotFound"]                      = {"hipErrorSharedObjectSymbolNotFound",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 302
  // CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  m["cudaErrorSharedObjectInitFailed"]                          = {"hipErrorSharedObjectInitFailed",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 303
  // CUDA_ERROR_OPERATING_SYSTEM
  m["cudaErrorOperatingSystem"]                                 = {"hipErrorOperatingSystem",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 304
  // CUDA_ERROR_INVALID_HANDLE
  m["cudaErrorInvalidResourceHandle"]                           = {"hipErrorInvalidHandle",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 400
  // CUDA_ERROR_ILLEGAL_STATE
  m["cudaErrorIllegalState"]                                    = {"hipErrorIllegalState",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 401
  // CUDA_ERROR_LOSSY_QUERY
  m["cudaErrorLossyQuery"]                                      = {"hipErrorLossyQuery",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 402
  // CUDA_ERROR_NOT_FOUND
  m["cudaErrorSymbolNotFound"]                                  = {"hipErrorNotFound",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 500
  // CUDA_ERROR_NOT_READY
  m["cudaErrorNotReady"]                                        = {"hipErrorNotReady",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 600
 // CUDA_ERROR_ILLEGAL_ADDRESS
  m["cudaErrorIllegalAddress"]                                  = {"hipErrorIllegalAddress",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 700
  // CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
  m["cudaErrorLaunchOutOfResources"]                            = {"hipErrorLaunchOutOfResources",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 701
  // CUDA_ERROR_LAUNCH_TIMEOUT
  m["cudaErrorLaunchTimeout"]                                   = {"hipErrorLaunchTimeOut",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 702
  // CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING
  m["cudaErrorLaunchIncompatibleTexturing"]                     = {"hipErrorLaunchIncompatibleTexturing",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 703
  // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
  m["cudaErrorPeerAccessAlreadyEnabled"]                        = {"hipErrorPeerAccessAlreadyEnabled",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 704
  // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
  m["cudaErrorPeerAccessNotEnabled"]                            = {"hipErrorPeerAccessNotEnabled",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 705
  // CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  m["cudaErrorSetOnActiveProcess"]                              = {"hipErrorSetOnActiveProcess",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 708
  // CUDA_ERROR_CONTEXT_IS_DESTROYED
  m["cudaErrorContextIsDestroyed"]                              = {"hipErrorContextIsDestroyed",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 709
  // CUDA_ERROR_ASSERT
  m["cudaErrorAssert"]                                          = {"hipErrorAssert",                                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 710
  // CUDA_ERROR_TOO_MANY_PEERS
  m["cudaErrorTooManyPeers"]                                    = {"hipErrorTooManyPeers",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 711
  // CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
  m["cudaErrorHostMemoryAlreadyRegistered"]                     = {"hipErrorHostMemoryAlreadyRegistered",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 712
  // CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
  m["cudaErrorHostMemoryNotRegistered"]                         = {"hipErrorHostMemoryNotRegistered",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 713
  // CUDA_ERROR_HARDWARE_STACK_ERROR
  m["cudaErrorHardwareStackError"]                              = {"hipErrorHardwareStackError",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 714
  // CUDA_ERROR_ILLEGAL_INSTRUCTION
  m["cudaErrorIllegalInstruction"]                              = {"hipErrorIllegalInstruction",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 715
  // CUDA_ERROR_MISALIGNED_ADDRESS
  m["cudaErrorMisalignedAddress"]                               = {"hipErrorMisalignedAddress",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 716
  // CUDA_ERROR_INVALID_ADDRESS_SPACE
  m["cudaErrorInvalidAddressSpace"]                             = {"hipErrorInvalidAddressSpace",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 717
  // CUDA_ERROR_INVALID_PC
  m["cudaErrorInvalidPc"]                                       = {"hipErrorInvalidPc",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 718
  // CUDA_ERROR_LAUNCH_FAILED
  m["cudaErrorLaunchFailure"]                                   = {"hipErrorLaunchFailure",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 719
  // CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
  m["cudaErrorCooperativeLaunchTooLarge"]                       = {"hipErrorCooperativeLaunchTooLarge",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 720
  // CUDA_ERROR_TENSOR_MEMORY_LEAK
  m["cudaErrorTensorMemoryLeak"]                                = {"hipErrorTensorMemoryLeak",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 721
  // CUDA_ERROR_NOT_PERMITTED
  m["cudaErrorNotPermitted"]                                    = {"hipErrorNotPermitted",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 800
  // CUDA_ERROR_NOT_SUPPORTED
  m["cudaErrorNotSupported"]                                    = {"hipErrorNotSupported",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 801
  // CUDA_ERROR_SYSTEM_NOT_READY
  m["cudaErrorSystemNotReady"]                                  = {"hipErrorSystemNotReady",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 802
  // CUDA_ERROR_SYSTEM_DRIVER_MISMATCH
  m["cudaErrorSystemDriverMismatch"]                            = {"hipErrorSystemDriverMismatch",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 803
  // CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
  m["cudaErrorCompatNotSupportedOnDevice"]                      = {"hipErrorCompatNotSupportedOnDevice",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 804
  // CUDA_ERROR_MPS_CONNECTION_FAILED
  m["cudaErrorMpsConnectionFailed"]                             = {"hipErrorMpsConnectionFailed",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 805
  // CUDA_ERROR_MPS_RPC_FAILURE
  m["cudaErrorMpsRpcFailure"]                                   = {"hipErrorMpsRpcFailed",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 806
  // CUDA_ERROR_MPS_SERVER_NOT_READY
  m["cudaErrorMpsServerNotReady"]                               = {"hipErrorMpsServerNotReady",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 807
  // CUDA_ERROR_MPS_MAX_CLIENTS_REACHED
  m["cudaErrorMpsMaxClientsReached"]                            = {"hipErrorMpsMaxClientsReached",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 808
  // CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED
  m["cudaErrorMpsMaxConnectionsReached"]                        = {"hipErrorMpsMaxConnectionsReached",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 809
  // CUDA_ERROR_MPS_CLIENT_TERMINATED
  m["cudaErrorMpsClientTerminated"]                             = {"hipErrorMpsClientTerminated",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 810
  // CUDA_ERROR_CDP_NOT_SUPPORTED
  m["cudaErrorCdpNotSupported"]                                 = {"hipErrorCdpNotUnsupported",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 811
  // CUDA_ERROR_CDP_VERSION_MISMATCH
  m["cudaErrorCdpVersionMismatch"]                              = {"hipErrorCdpVersionMismatch",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 812
  // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
  m["cudaErrorStreamCaptureUnsupported"]                        = {"hipErrorStreamCaptureUnsupported",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 900
  // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
  m["cudaErrorStreamCaptureInvalidated"]                        = {"hipErrorStreamCaptureInvalidated",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 901
  // CUDA_ERROR_STREAM_CAPTURE_MERGE
  m["cudaErrorStreamCaptureMerge"]                              = {"hipErrorStreamCaptureMerge",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 902
  // CUDA_ERROR_STREAM_CAPTURE_UNMATCHED
  m["cudaErrorStreamCaptureUnmatched"]                          = {"hipErrorStreamCaptureUnmatched",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 903
  // CUDA_ERROR_STREAM_CAPTURE_UNJOINED
  m["cudaErrorStreamCaptureUnjoined"]                           = {"hipErrorStreamCaptureUnjoined",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 904
  // CUDA_ERROR_STREAM_CAPTURE_ISOLATION
  m["cudaErrorStreamCaptureIsolation"]                          = {"hipErrorStreamCaptureIsolation",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 905
  // CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
  m["cudaErrorStreamCaptureImplicit"]                           = {"hipErrorStreamCaptureImplicit",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 906
  // CUDA_ERROR_CAPTURED_EVENT
  m["cudaErrorCapturedEvent"]                                   = {"hipErrorCapturedEvent",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 907
  // CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
  m["cudaErrorStreamCaptureWrongThread"]                        = {"hipErrorStreamCaptureWrongThread",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 908
  // CUDA_ERROR_TIMEOUT
  m["cudaErrorTimeout"]                                         = {"hipErrorTimeout",                                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 909
  // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
  m["cudaErrorGraphExecUpdateFailure"]                          = {"hipErrorGraphExecUpdateFailure",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 910
  // CUDA_ERROR_EXTERNAL_DEVICE
  m["cudaErrorExternalDevice"]                                  = {"hipErrorExternalDevice",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 911
  // CUDA_ERROR_INVALID_CLUSTER_SIZE
  m["cudaErrorInvalidClusterSize"]                              = {"hipErrorInvalidClusterSize",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 912
  // CUDA_ERROR_FUNCTION_NOT_LOADED
  m["cudaErrorFunctionNotLoaded"]                               = {"hipErrorFunctionNotLoaded",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 913
  // CUDA_ERROR_INVALID_RESOURCE_TYPE
  m["cudaErrorInvalidResourceType"]                             = {"hipErrorInvalidResourceType",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 914
  // CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION
  m["cudaErrorInvalidResourceConfiguration"]                    = {"hipErrorInvalidResourceConfiguration",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 915
  // CUDA_ERROR_STREAM_DETACHED
  m["cudaErrorStreamDetached"]                                  = {"hipErrorStreamDetached",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 917
  // CUDA_ERROR_UNKNOWN
  m["cudaErrorUnknown"]                                         = {"hipErrorUnknown",                                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 999
  // Deprecated since CUDA 4.1
  m["cudaErrorApiFailureBase"]                                  = {"hipErrorApiFailureBase",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_DEPRECATED}; // 10000

  // CUexternalMemoryHandleType
  m["cudaExternalMemoryHandleType"]                             = {"hipExternalMemoryHandleType",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaExternalMemoryHandleType enum values
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
  m["cudaExternalMemoryHandleTypeOpaqueFd"]                     = {"hipExternalMemoryHandleTypeOpaqueFd",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
  m["cudaExternalMemoryHandleTypeOpaqueWin32"]                  = {"hipExternalMemoryHandleTypeOpaqueWin32",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
  m["cudaExternalMemoryHandleTypeOpaqueWin32Kmt"]               = {"hipExternalMemoryHandleTypeOpaqueWin32Kmt",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
  m["cudaExternalMemoryHandleTypeD3D12Heap"]                    = {"hipExternalMemoryHandleTypeD3D12Heap",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
  m["cudaExternalMemoryHandleTypeD3D12Resource"]                = {"hipExternalMemoryHandleTypeD3D12Resource",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 5
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE
  m["cudaExternalMemoryHandleTypeD3D11Resource"]                = {"hipExternalMemoryHandleTypeD3D11Resource",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 6
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT
  m["cudaExternalMemoryHandleTypeD3D11ResourceKmt"]             = {"hipExternalMemoryHandleTypeD3D11ResourceKmt",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 7
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
  m["cudaExternalMemoryHandleTypeNvSciBuf"]                     = {"hipExternalMemoryHandleTypeNvSciBuf",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8

  // CUexternalSemaphoreHandleType
  m["cudaExternalSemaphoreHandleType"]                          = {"hipExternalSemaphoreHandleType",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaExternalSemaphoreHandleType enum values
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
  m["cudaExternalSemaphoreHandleTypeOpaqueFd"]                  = {"hipExternalSemaphoreHandleTypeOpaqueFd",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
  m["cudaExternalSemaphoreHandleTypeOpaqueWin32"]               = {"hipExternalSemaphoreHandleTypeOpaqueWin32",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
  m["cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt"]            = {"hipExternalSemaphoreHandleTypeOpaqueWin32Kmt",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
  m["cudaExternalSemaphoreHandleTypeD3D12Fence"]                = {"hipExternalSemaphoreHandleTypeD3D12Fence",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE
  m["cudaExternalSemaphoreHandleTypeD3D11Fence"]                = {"hipExternalSemaphoreHandleTypeD3D11Fence",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 5
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC
  m["cudaExternalSemaphoreHandleTypeNvSciSync"]                 = {"hipExternalSemaphoreHandleTypeNvSciSync",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 6
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX
  m["cudaExternalSemaphoreHandleTypeKeyedMutex"]                = {"hipExternalSemaphoreHandleTypeKeyedMutex",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 7
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT
  m["cudaExternalSemaphoreHandleTypeKeyedMutexKmt"]             = {"hipExternalSemaphoreHandleTypeKeyedMutexKmt",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD
  m["cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd"]       = {"hipExternalSemaphoreHandleTypeTimelineSemaphoreFd",        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 9
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32
  m["cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32"]    = {"hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32",     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 9

  // CUfunction_attribute
  // NOTE: only last, starting from 8, values are presented and are equal to Driver's ones
  m["cudaFuncAttribute"]                                        = {"hipFuncAttribute",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaFuncAttribute enum values
  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
  m["cudaFuncAttributeMaxDynamicSharedMemorySize"]              = {"hipFuncAttributeMaxDynamicSharedMemorySize",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  8
  // CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
  m["cudaFuncAttributePreferredSharedMemoryCarveout"]           = {"hipFuncAttributePreferredSharedMemoryCarveout",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; //  9
  // CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
  m["cudaFuncAttributeClusterDimMustBeSet"]                     = {"hipFuncAttributeClusterDimMustBeSet",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 10
  // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
  m["cudaFuncAttributeRequiredClusterWidth"]                    = {"hipFuncAttributeRequiredClusterWidth",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 11
  // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
  m["cudaFuncAttributeRequiredClusterHeight"]                   = {"hipFuncAttributeRequiredClusterHeight",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 12
  // CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
  m["cudaFuncAttributeRequiredClusterDepth"]                    = {"hipFuncAttributeRequiredClusterDepth",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 13
  // CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
  m["cudaFuncAttributeNonPortableClusterSizeAllowed"]           = {"hipFuncAttributeNonPortableClusterSizeAllowed",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 14
  // CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
  m["cudaFuncAttributeClusterSchedulingPolicyPreference"]       = {"hipFuncAttributeClusterSchedulingPolicyPreference",        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 15
  // CU_FUNC_ATTRIBUTE_MAX
  m["cudaFuncAttributeMax"]                                     = {"hipFuncAttributeMax",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 16

  // CUfunc_cache
  m["cudaFuncCache"]                                            = {"hipFuncCache_t",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaFuncCache enum values
  // CU_FUNC_CACHE_PREFER_NONE = 0x00
  m["cudaFuncCachePreferNone"]                                  = {"hipFuncCachePreferNone",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_FUNC_CACHE_PREFER_SHARED = 0x01
  m["cudaFuncCachePreferShared"]                                = {"hipFuncCachePreferShared",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_FUNC_CACHE_PREFER_L1 = 0x02
  m["cudaFuncCachePreferL1"]                                    = {"hipFuncCachePreferL1",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_FUNC_CACHE_PREFER_EQUAL = 0x03
  m["cudaFuncCachePreferEqual"]                                 = {"hipFuncCachePreferEqual",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUarray_cubemap_face
  m["cudaGraphicsCubeFace"]                                     = {"hipGraphicsCubeFace",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGraphicsCubeFace enum values
  // CU_CUBEMAP_FACE_POSITIVE_X
  m["cudaGraphicsCubeFacePositiveX"]                            = {"hipGraphicsCubeFacePositiveX",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x00
  // CU_CUBEMAP_FACE_NEGATIVE_X
  m["cudaGraphicsCubeFaceNegativeX"]                            = {"hipGraphicsCubeFaceNegativeX",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x01
  // CU_CUBEMAP_FACE_POSITIVE_Y
  m["cudaGraphicsCubeFacePositiveY"]                            = {"hipGraphicsCubeFacePositiveY",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x02
  // CU_CUBEMAP_FACE_NEGATIVE_Y
  m["cudaGraphicsCubeFaceNegativeY"]                            = {"hipGraphicsCubeFaceNegativeY",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x03
  // CU_CUBEMAP_FACE_POSITIVE_Z
  m["cudaGraphicsCubeFacePositiveZ"]                            = {"hipGraphicsCubeFacePositiveZ",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x04
  // CU_CUBEMAP_FACE_NEGATIVE_Z
  m["cudaGraphicsCubeFaceNegativeZ"]                            = {"hipGraphicsCubeFaceNegativeZ",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x05

  // CUgraphicsMapResourceFlags
  m["cudaGraphicsMapFlags"]                                     = {"hipGraphicsMapFlags",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGraphicsMapFlags enum values
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00
  m["cudaGraphicsMapFlagsNone"]                                 = {"hipGraphicsMapFlagsNone",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  m["cudaGraphicsMapFlagsReadOnly"]                             = {"hipGraphicsMapFlagsReadOnly",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
  m["cudaGraphicsMapFlagsWriteDiscard"]                         = {"hipGraphicsMapFlagsWriteDiscard",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2

  // CUgraphicsRegisterFlags
  m["cudaGraphicsRegisterFlags"]                                = {"hipGraphicsRegisterFlags",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphicsRegisterFlags enum values
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00
  m["cudaGraphicsRegisterFlagsNone"]                            = {"hipGraphicsRegisterFlagsNone",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  m["cudaGraphicsRegisterFlagsReadOnly"]                        = {"hipGraphicsRegisterFlagsReadOnly",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02
  m["cudaGraphicsRegisterFlagsWriteDiscard"]                    = {"hipGraphicsRegisterFlagsWriteDiscard",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04
  m["cudaGraphicsRegisterFlagsSurfaceLoadStore"]                = {"hipGraphicsRegisterFlagsSurfaceLoadStore",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08
  m["cudaGraphicsRegisterFlagsTextureGather"]                   = {"hipGraphicsRegisterFlagsTextureGather",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 8

  // CUgraphNodeType
  m["cudaGraphNodeType"]                                        = {"hipGraphNodeType",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphNodeType enum values
  // CU_GRAPH_NODE_TYPE_KERNEL = 0
  m["cudaGraphNodeTypeKernel"]                                  = {"hipGraphNodeTypeKernel",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_GRAPH_NODE_TYPE_MEMCPY = 1
  m["cudaGraphNodeTypeMemcpy"]                                  = {"hipGraphNodeTypeMemcpy",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_GRAPH_NODE_TYPE_MEMSET = 2
  m["cudaGraphNodeTypeMemset"]                                  = {"hipGraphNodeTypeMemset",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_GRAPH_NODE_TYPE_HOST = 3
  m["cudaGraphNodeTypeHost"]                                    = {"hipGraphNodeTypeHost",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x03
  // CU_GRAPH_NODE_TYPE_GRAPH = 4
  m["cudaGraphNodeTypeGraph"]                                   = {"hipGraphNodeTypeGraph",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CU_GRAPH_NODE_TYPE_EMPTY = 5
  m["cudaGraphNodeTypeEmpty"]                                   = {"hipGraphNodeTypeEmpty",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x05
  // CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6
  m["cudaGraphNodeTypeWaitEvent"]                               = {"hipGraphNodeTypeWaitEvent",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x06
  // CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7
  m["cudaGraphNodeTypeEventRecord"]                             = {"hipGraphNodeTypeEventRecord",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x07
  // CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
  m["cudaGraphNodeTypeExtSemaphoreSignal"]                      = {"hipGraphNodeTypeExtSemaphoreSignal",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x08
  // CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9
  m["cudaGraphNodeTypeExtSemaphoreWait"]                        = {"hipGraphNodeTypeExtSemaphoreWait",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x09
  // CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10
  m["cudaGraphNodeTypeMemAlloc"]                                = {"hipGraphNodeTypeMemAlloc",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0a
  // CU_GRAPH_NODE_TYPE_MEM_FREE = 11
  m["cudaGraphNodeTypeMemFree"]                                 = {"hipGraphNodeTypeMemFree",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0b
  // CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
  m["cudaGraphNodeTypeConditional"]                             = {"hipGraphNodeTypeConditional",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0d
  // CU_GRAPH_NODE_TYPE_COUNT
  m["cudaGraphNodeTypeCount"]                                   = {"hipGraphNodeTypeCount",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphExecUpdateResult
  m["cudaGraphExecUpdateResult"]                                = {"hipGraphExecUpdateResult",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphExecUpdateResult enum values
  // CU_GRAPH_EXEC_UPDATE_SUCCESS
  m["cudaGraphExecUpdateSuccess"]                               = {"hipGraphExecUpdateSuccess",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0
  // CU_GRAPH_EXEC_UPDATE_ERROR
  m["cudaGraphExecUpdateError"]                                 = {"hipGraphExecUpdateError",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1
  // CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED
  m["cudaGraphExecUpdateErrorTopologyChanged"]                  = {"hipGraphExecUpdateErrorTopologyChanged",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x2
  // CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED
  m["cudaGraphExecUpdateErrorNodeTypeChanged"]                  = {"hipGraphExecUpdateErrorNodeTypeChanged",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x3
  // CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED
  m["cudaGraphExecUpdateErrorFunctionChanged"]                  = {"hipGraphExecUpdateErrorFunctionChanged",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x4
  // CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED
  m["cudaGraphExecUpdateErrorParametersChanged"]                = {"hipGraphExecUpdateErrorParametersChanged",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x5
  // CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED
  m["cudaGraphExecUpdateErrorNotSupported"]                     = {"hipGraphExecUpdateErrorNotSupported",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x6
  // CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
  m["cudaGraphExecUpdateErrorUnsupportedFunctionChange"]        = {"hipGraphExecUpdateErrorUnsupportedFunctionChange",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x7
  // CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED
  m["cudaGraphExecUpdateErrorAttributesChanged"]                = {"hipGraphExecUpdateErrorAttributesChanged",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x8

  // CUlimit
  m["cudaLimit"]                                                = {"hipLimit_t",                                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaLimit enum values
  // CU_LIMIT_STACK_SIZE
  m["cudaLimitStackSize"]                                       = {"hipLimitStackSize",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_LIMIT_PRINTF_FIFO_SIZE
  m["cudaLimitPrintfFifoSize"]                                  = {"hipLimitPrintfFifoSize",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_LIMIT_MALLOC_HEAP_SIZE
  m["cudaLimitMallocHeapSize"]                                  = {"hipLimitMallocHeapSize",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
  m["cudaLimitDevRuntimeSyncDepth"]                             = {"hipLimitDevRuntimeSyncDepth",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x03
  // CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
  m["cudaLimitDevRuntimePendingLaunchCount"]                    = {"hipLimitDevRuntimePendingLaunchCount",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x04
  // CU_LIMIT_MAX_L2_FETCH_GRANULARITY
  m["cudaLimitMaxL2FetchGranularity"]                           = {"hipLimitMaxL2FetchGranularity",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x05
  // CU_LIMIT_PERSISTING_L2_CACHE_SIZE
  m["cudaLimitPersistingL2CacheSize"]                           = {"hipLimitPersistingL2CacheSize",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x06

  // no analogue
  m["cudaMemcpyKind"]                                           = {"hipMemcpyKind",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemcpyKind enum values
  m["cudaMemcpyHostToHost"]                                     = {"hipMemcpyHostToHost",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  m["cudaMemcpyHostToDevice"]                                   = {"hipMemcpyHostToDevice",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  m["cudaMemcpyDeviceToHost"]                                   = {"hipMemcpyDeviceToHost",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  m["cudaMemcpyDeviceToDevice"]                                 = {"hipMemcpyDeviceToDevice",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  m["cudaMemcpyDefault"]                                        = {"hipMemcpyDefault",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4

  // CUmem_advise
  m["cudaMemoryAdvise"]                                         = {"hipMemoryAdvise",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemoryAdvise enum values
  // CU_MEM_ADVISE_SET_READ_MOSTLY
  m["cudaMemAdviseSetReadMostly"]                               = {"hipMemAdviseSetReadMostly",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_MEM_ADVISE_UNSET_READ_MOSTLY
  m["cudaMemAdviseUnsetReadMostly"]                             = {"hipMemAdviseUnsetReadMostly",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_MEM_ADVISE_SET_PREFERRED_LOCATION
  m["cudaMemAdviseSetPreferredLocation"]                        = {"hipMemAdviseSetPreferredLocation",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION
  m["cudaMemAdviseUnsetPreferredLocation"]                      = {"hipMemAdviseUnsetPreferredLocation",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_MEM_ADVISE_SET_ACCESSED_BY
  m["cudaMemAdviseSetAccessedBy"]                               = {"hipMemAdviseSetAccessedBy",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 5
  // CU_MEM_ADVISE_UNSET_ACCESSED_BY
  m["cudaMemAdviseUnsetAccessedBy"]                             = {"hipMemAdviseUnsetAccessedBy",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 6

  // no analogue
  // NOTE: CUmemorytype is partial analogue
  m["cudaMemoryType"]                                           = {"hipMemoryType",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemoryType enum values
  m["cudaMemoryTypeUnregistered"]                               = {"hipMemoryTypeUnregistered",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  m["cudaMemoryTypeHost"]                                       = {"hipMemoryTypeHost",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  m["cudaMemoryTypeDevice"]                                     = {"hipMemoryTypeDevice",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  m["cudaMemoryTypeManaged"]                                    = {"hipMemoryTypeManaged",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUmem_range_attribute
  m["cudaMemRangeAttribute"]                                    = {"hipMemRangeAttribute",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemRangeAttribute enum values
  // CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
  m["cudaMemRangeAttributeReadMostly"]                          = {"hipMemRangeAttributeReadMostly",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
  m["cudaMemRangeAttributePreferredLocation"]                   = {"hipMemRangeAttributePreferredLocation",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
  m["cudaMemRangeAttributeAccessedBy"]                          = {"hipMemRangeAttributeAccessedBy",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
  m["cudaMemRangeAttributeLastPrefetchLocation"]                = {"hipMemRangeAttributeLastPrefetchLocation",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE
  m["cudaMemRangeAttributePreferredLocationType"]               = {"hipMemRangeAttributePreferredLocationType",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 5
  // CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID
  m["cudaMemRangeAttributePreferredLocationId"]                 = {"hipMemRangeAttributePreferredLocationId",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 6
  // CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE
  m["cudaMemRangeAttributeLastPrefetchLocationType"]            = {"hipMemRangeAttributeLastPrefetchLocationType",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 7
  // CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID
  m["cudaMemRangeAttributeLastPrefetchLocationId"]              = {"hipMemRangeAttributeLastPrefetchLocationId",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8

  // no analogue
  m["cudaOutputMode"]                                           = {"hipOutputMode",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_REMOVED};
  m["cudaOutputMode_t"]                                         = {"hipOutputMode",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_REMOVED};
  // cudaOutputMode enum values
  m["cudaKeyValuePair"]                                         = {"hipKeyValuePair",                                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_REMOVED}; // 0x00
  m["cudaCSV"]                                                  = {"hipCSV",                                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_REMOVED}; // 0x01

  // CUresourcetype
  m["cudaResourceType"]                                         = {"hipResourceType",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaResourceType enum values
  // CU_RESOURCE_TYPE_ARRAY
  m["cudaResourceTypeArray"]                                    = {"hipResourceTypeArray",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
  m["cudaResourceTypeMipmappedArray"]                           = {"hipResourceTypeMipmappedArray",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_RESOURCE_TYPE_LINEAR
  m["cudaResourceTypeLinear"]                                   = {"hipResourceTypeLinear",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_RESOURCE_TYPE_PITCH2D
  m["cudaResourceTypePitch2D"]                                  = {"hipResourceTypePitch2D",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x03

  // CUresourceViewFormat
  m["cudaResourceViewFormat"]                                   = {"hipResourceViewFormat",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // enum cudaResourceViewFormat
  // CU_RES_VIEW_FORMAT_NONE
  m["cudaResViewFormatNone"]                                    = {"hipResViewFormatNone",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_RES_VIEW_FORMAT_UINT_1X8
  m["cudaResViewFormatUnsignedChar1"]                           = {"hipResViewFormatUnsignedChar1",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_RES_VIEW_FORMAT_UINT_2X8
  m["cudaResViewFormatUnsignedChar2"]                           = {"hipResViewFormatUnsignedChar2",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_RES_VIEW_FORMAT_UINT_4X8
  m["cudaResViewFormatUnsignedChar4"]                           = {"hipResViewFormatUnsignedChar4",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x03
  // CU_RES_VIEW_FORMAT_SINT_1X8
  m["cudaResViewFormatSignedChar1"]                             = {"hipResViewFormatSignedChar1",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CU_RES_VIEW_FORMAT_SINT_2X8
  m["cudaResViewFormatSignedChar2"]                             = {"hipResViewFormatSignedChar2",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x05
  // CU_RES_VIEW_FORMAT_SINT_4X8
  m["cudaResViewFormatSignedChar4"]                             = {"hipResViewFormatSignedChar4",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x06
  // CU_RES_VIEW_FORMAT_UINT_1X16
  m["cudaResViewFormatUnsignedShort1"]                          = {"hipResViewFormatUnsignedShort1",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x07
  // CU_RES_VIEW_FORMAT_UINT_2X16
  m["cudaResViewFormatUnsignedShort2"]                          = {"hipResViewFormatUnsignedShort2",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x08
  // CU_RES_VIEW_FORMAT_UINT_4X16
  m["cudaResViewFormatUnsignedShort4"]                          = {"hipResViewFormatUnsignedShort4",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x09
  // CU_RES_VIEW_FORMAT_SINT_1X16
  m["cudaResViewFormatSignedShort1"]                            = {"hipResViewFormatSignedShort1",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0a
  // CU_RES_VIEW_FORMAT_SINT_2X16
  m["cudaResViewFormatSignedShort2"]                            = {"hipResViewFormatSignedShort2",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0b
  // CU_RES_VIEW_FORMAT_SINT_4X16
  m["cudaResViewFormatSignedShort4"]                            = {"hipResViewFormatSignedShort4",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0c
  // CU_RES_VIEW_FORMAT_UINT_1X32
  m["cudaResViewFormatUnsignedInt1"]                            = {"hipResViewFormatUnsignedInt1",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0d
  // CU_RES_VIEW_FORMAT_UINT_2X32
  m["cudaResViewFormatUnsignedInt2"]                            = {"hipResViewFormatUnsignedInt2",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0e
  // CU_RES_VIEW_FORMAT_UINT_4X32
  m["cudaResViewFormatUnsignedInt4"]                            = {"hipResViewFormatUnsignedInt4",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0f
  // CU_RES_VIEW_FORMAT_SINT_1X32
  m["cudaResViewFormatSignedInt1"]                              = {"hipResViewFormatSignedInt1",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x10
  // CU_RES_VIEW_FORMAT_SINT_2X32
  m["cudaResViewFormatSignedInt2"]                              = {"hipResViewFormatSignedInt2",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x11
  // CU_RES_VIEW_FORMAT_SINT_4X32
  m["cudaResViewFormatSignedInt4"]                              = {"hipResViewFormatSignedInt4",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x12
  // CU_RES_VIEW_FORMAT_FLOAT_1X16
  m["cudaResViewFormatHalf1"]                                   = {"hipResViewFormatHalf1",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x13
  // CU_RES_VIEW_FORMAT_FLOAT_2X16
  m["cudaResViewFormatHalf2"]                                   = {"hipResViewFormatHalf2",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x14
  // CU_RES_VIEW_FORMAT_FLOAT_4X16
  m["cudaResViewFormatHalf4"]                                   = {"hipResViewFormatHalf4",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x15
  // CU_RES_VIEW_FORMAT_FLOAT_1X32
  m["cudaResViewFormatFloat1"]                                  = {"hipResViewFormatFloat1",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x16
  // CU_RES_VIEW_FORMAT_FLOAT_2X32
  m["cudaResViewFormatFloat2"]                                  = {"hipResViewFormatFloat2",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x17
  // CU_RES_VIEW_FORMAT_FLOAT_4X32
  m["cudaResViewFormatFloat4"]                                  = {"hipResViewFormatFloat4",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x18
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC1
  m["cudaResViewFormatUnsignedBlockCompressed1"]                = {"hipResViewFormatUnsignedBlockCompressed1",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x19
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC2
  m["cudaResViewFormatUnsignedBlockCompressed2"]                = {"hipResViewFormatUnsignedBlockCompressed2",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1a
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC3
  m["cudaResViewFormatUnsignedBlockCompressed3"]                = {"hipResViewFormatUnsignedBlockCompressed3",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1b
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC4
  m["cudaResViewFormatUnsignedBlockCompressed4"]                = {"hipResViewFormatUnsignedBlockCompressed4",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1c
  // CU_RES_VIEW_FORMAT_SIGNED_BC4
  m["cudaResViewFormatSignedBlockCompressed4"]                  = {"hipResViewFormatSignedBlockCompressed4",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1d
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC5
  m["cudaResViewFormatUnsignedBlockCompressed5"]                = {"hipResViewFormatUnsignedBlockCompressed5",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1e
  // CU_RES_VIEW_FORMAT_SIGNED_BC5
  m["cudaResViewFormatSignedBlockCompressed5"]                  = {"hipResViewFormatSignedBlockCompressed5",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1f
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC6H
  m["cudaResViewFormatUnsignedBlockCompressed6H"]               = {"hipResViewFormatUnsignedBlockCompressed6H",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x20
  // CU_RES_VIEW_FORMAT_SIGNED_BC6H
  m["cudaResViewFormatSignedBlockCompressed6H"]                 = {"hipResViewFormatSignedBlockCompressed6H",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x21
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC7
  m["cudaResViewFormatUnsignedBlockCompressed7"]                = {"hipResViewFormatUnsignedBlockCompressed7",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x22

  // CUshared_carveout
  m["cudaSharedCarveout"]                                       = {"hipSharedCarveout",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaSharedCarveout enum values
  // CU_SHAREDMEM_CARVEOUT_DEFAULT
  m["cudaSharedmemCarveoutDefault"]                             = {"hipSharedmemCarveoutDefault",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // -1
  // CU_SHAREDMEM_CARVEOUT_MAX_SHARED
  m["cudaSharedmemCarveoutMaxShared"]                           = {"hipSharedmemCarveoutMaxShared",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 100
  // CU_SHAREDMEM_CARVEOUT_MAX_L1
  m["cudaSharedmemCarveoutMaxL1"]                               = {"hipSharedmemCarveoutMaxL1",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0

  // CUsharedconfig
  m["cudaSharedMemConfig"]                                      = {"hipSharedMemConfig",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED};
  // cudaSharedMemConfig enum values
  // CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00
  m["cudaSharedMemBankSizeDefault"]                             = {"hipSharedMemBankSizeDefault",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01
  m["cudaSharedMemBankSizeFourByte"]                            = {"hipSharedMemBankSizeFourByte",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02
  m["cudaSharedMemBankSizeEightByte"]                           = {"hipSharedMemBankSizeEightByte",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2

  // CUstreamCaptureStatus
  m["cudaStreamCaptureStatus"]                                  = {"hipStreamCaptureStatus",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaStreamCaptureStatus enum values
  // CU_STREAM_CAPTURE_STATUS_NONE
  m["cudaStreamCaptureStatusNone"]                              = {"hipStreamCaptureStatusNone",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_STREAM_CAPTURE_STATUS_ACTIVE
  m["cudaStreamCaptureStatusActive"]                            = {"hipStreamCaptureStatusActive",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_STREAM_CAPTURE_STATUS_INVALIDATED
  m["cudaStreamCaptureStatusInvalidated"]                       = {"hipStreamCaptureStatusInvalidated",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2

  // CUstreamCaptureMode
  m["cudaStreamCaptureMode"]                                    = {"hipStreamCaptureMode",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaStreamCaptureMode enum values
  // CU_STREAM_CAPTURE_MODE_GLOBAL
  m["cudaStreamCaptureModeGlobal"]                              = {"hipStreamCaptureModeGlobal",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
  m["cudaStreamCaptureModeThreadLocal"]                         = {"hipStreamCaptureModeThreadLocal",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_STREAM_CAPTURE_MODE_RELAXED
  m["cudaStreamCaptureModeRelaxed"]                             = {"hipStreamCaptureModeRelaxed",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2

  // no analogue
  m["cudaSurfaceBoundaryMode"]                                  = {"hipSurfaceBoundaryMode",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaSurfaceBoundaryMode enum values
  m["cudaBoundaryModeZero"]                                     = {"hipBoundaryModeZero",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  m["cudaBoundaryModeClamp"]                                    = {"hipBoundaryModeClamp",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  m["cudaBoundaryModeTrap"]                                     = {"hipBoundaryModeTrap",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2

  // no analogue
  m["cudaSurfaceFormatMode"]                                    = {"hipSurfaceFormatMode",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // enum cudaSurfaceFormatMode
  m["cudaFormatModeForced"]                                     = {"hipFormatModeForced",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  m["cudaFormatModeAuto"]                                       = {"hipFormatModeAuto",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1

  // CUaddress_mode_enum
  m["cudaTextureAddressMode"]                                   = {"hipTextureAddressMode",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaTextureAddressMode enum values
  // CU_TR_ADDRESS_MODE_WRAP
  m["cudaAddressModeWrap"]                                      = {"hipAddressModeWrap",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_TR_ADDRESS_MODE_CLAMP
  m["cudaAddressModeClamp"]                                     = {"hipAddressModeClamp",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_TR_ADDRESS_MODE_MIRROR
  m["cudaAddressModeMirror"]                                    = {"hipAddressModeMirror",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_TR_ADDRESS_MODE_BORDER
  m["cudaAddressModeBorder"]                                    = {"hipAddressModeBorder",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUfilter_mode
  m["cudaTextureFilterMode"]                                    = {"hipTextureFilterMode",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaTextureFilterMode enum values
  // CU_TR_FILTER_MODE_POINT
  m["cudaFilterModePoint"]                                      = {"hipFilterModePoint",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_TR_FILTER_MODE_LINEAR
  m["cudaFilterModeLinear"]                                     = {"hipFilterModeLinear",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1

  // no analogue
  m["cudaTextureReadMode"]                                      = {"hipTextureReadMode",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaTextureReadMode enum values
  m["cudaReadModeElementType"]                                  = {"hipReadModeElementType",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  m["cudaReadModeNormalizedFloat"]                              = {"hipReadModeNormalizedFloat",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1

  // CUGLDeviceList
  m["cudaGLDeviceList"]                                         = {"hipGLDeviceList",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGLDeviceList enum values
  // CU_GL_DEVICE_LIST_ALL = 0x01
  m["cudaGLDeviceListAll"]                                      = {"hipGLDeviceListAll",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_GL_DEVICE_LIST_CURRENT_FRAME = 0x02
  m["cudaGLDeviceListCurrentFrame"]                             = {"hipGLDeviceListCurrentFrame",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_GL_DEVICE_LIST_NEXT_FRAME = 0x03
  m["cudaGLDeviceListNextFrame"]                                = {"hipGLDeviceListNextFrame",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUGLmap_flags
  m["cudaGLMapFlags"]                                           = {"hipGLMapFlags",                                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGLMapFlags enum values
  // CU_GL_MAP_RESOURCE_FLAGS_NONE = 0x00
  m["cudaGLMapFlagsNone"]                                       = {"hipGLMapFlagsNone",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  m["cudaGLMapFlagsReadOnly"]                                   = {"hipGLMapFlagsReadOnly",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
  m["cudaGLMapFlagsWriteDiscard"]                               = {"hipGLMapFlagsWriteDiscard",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2

  // CUd3d9DeviceList
  m["cudaD3D9DeviceList"]                                       = {"hipD3D9DeviceList",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUd3d9DeviceList enum values
  // CU_D3D9_DEVICE_LIST_ALL = 0x01
  m["cudaD3D9DeviceListAll"]                                    = {"HIP_D3D9_DEVICE_LIST_ALL",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_D3D9_DEVICE_LIST_CURRENT_FRAME = 0x02
  m["cudaD3D9DeviceListCurrentFrame"]                           = {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2
  // CU_D3D9_DEVICE_LIST_NEXT_FRAME = 0x03
  m["cudaD3D9DeviceListNextFrame"]                              = {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 3

  // CUd3d9map_flags
  m["cudaD3D9MapFlags"]                                         = {"hipD3D9MapFlags",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D9MapFlags enum values
  // CU_D3D9_MAPRESOURCE_FLAGS_NONE = 0x00
  m["cudaD3D9MapFlagsNone"]                                     = {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_D3D9_MAPRESOURCE_FLAGS_READONLY = 0x01
  m["cudaD3D9MapFlagsReadOnly"]                                 = {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02
  m["cudaD3D9MapFlagsWriteDiscard"]                             = {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2

  // CUd3d9Register_flags
  m["cudaD3D9RegisterFlags"]                                    = {"hipD3D9RegisterFlags",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D9RegisterFlags enum values
  // CU_D3D9_REGISTER_FLAGS_NONE = 0x00
  m["cudaD3D9RegisterFlagsNone"]                                = {"HIP_D3D9_REGISTER_FLAGS_NONE",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_D3D9_REGISTER_FLAGS_ARRAY = 0x01
  m["cudaD3D9RegisterFlagsArray"]                               = {"HIP_D3D9_REGISTER_FLAGS_ARRAY",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1

  // CUd3d10DeviceList
  m["cudaD3D10DeviceList"]                                      = {"hipd3d10DeviceList",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D10DeviceList enum values
  // CU_D3D10_DEVICE_LIST_ALL = 0x01
  m["cudaD3D10DeviceListAll"]                                   = {"HIP_D3D10_DEVICE_LIST_ALL",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_D3D10_DEVICE_LIST_CURRENT_FRAME = 0x02
  m["cudaD3D10DeviceListCurrentFrame"]                          = {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2
  // CU_D3D10_DEVICE_LIST_NEXT_FRAME = 0x03
  m["cudaD3D10DeviceListNextFrame"]                             = {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 3

  // CUd3d10map_flags
  m["cudaD3D10MapFlags"]                                        = {"hipD3D10MapFlags",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D10MapFlags enum values
  // CU_D3D10_MAPRESOURCE_FLAGS_NONE = 0x00
  m["cudaD3D10MapFlagsNone"]                                    = {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_D3D10_MAPRESOURCE_FLAGS_READONLY = 0x01
  m["cudaD3D10MapFlagsReadOnly"]                                = {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02
  m["cudaD3D10MapFlagsWriteDiscard"]                            = {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2

  // CUd3d10Register_flags
  m["cudaD3D10RegisterFlags"]                                   = {"hipD3D10RegisterFlags",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D10RegisterFlags enum values
  // CU_D3D10_REGISTER_FLAGS_NONE = 0x00
  m["cudaD3D10RegisterFlagsNone"]                               = {"HIP_D3D10_REGISTER_FLAGS_NONE",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_D3D10_REGISTER_FLAGS_ARRAY = 0x01
  m["cudaD3D10RegisterFlagsArray"]                              = {"HIP_D3D10_REGISTER_FLAGS_ARRAY",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1

  // CUd3d11DeviceList
  m["cudaD3D11DeviceList"]                                      = {"hipd3d11DeviceList",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaD3D11DeviceList enum values
  // CU_D3D11_DEVICE_LIST_ALL = 0x01
  m["cudaD3D11DeviceListAll"]                                   = {"HIP_D3D11_DEVICE_LIST_ALL",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1
  // CU_D3D11_DEVICE_LIST_CURRENT_FRAME = 0x02
  m["cudaD3D11DeviceListCurrentFrame"]                          = {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 2
  // CU_D3D11_DEVICE_LIST_NEXT_FRAME = 0x03
  m["cudaD3D11DeviceListNextFrame"]                             = {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 3

  // no analogue
  m["libraryPropertyType"]                                      = {"hipLibraryPropertyType_t",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["libraryPropertyType_t"]                                    = {"hipLibraryPropertyType_t",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUaccessProperty
  m["cudaAccessProperty"]                                       = {"hipAccessProperty",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CU_ACCESS_PROPERTY_NORMAL
  m["cudaAccessPropertyNormal"]                                 = {"hipAccessPropertyNormal",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_ACCESS_PROPERTY_STREAMING
  m["cudaAccessPropertyStreaming"]                              = {"hipAccessPropertyStreaming",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_ACCESS_PROPERTY_PERSISTING
  m["cudaAccessPropertyPersisting"]                             = {"hipAccessPropertyPersisting",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2

  // CUsynchronizationPolicy
  m["cudaSynchronizationPolicy"]                                = {"hipSynchronizationPolicy",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CU_SYNC_POLICY_AUTO
  m["cudaSyncPolicyAuto"]                                       = {"hipSyncPolicyAuto",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_SYNC_POLICY_SPIN
  m["cudaSyncPolicySpin"]                                       = {"hipSyncPolicySpin",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_SYNC_POLICY_YIELD
  m["cudaSyncPolicyYield"]                                      = {"hipSyncPolicyYield",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_SYNC_POLICY_BLOCKING_SYNC
  m["cudaSyncPolicyBlockingSync"]                               = {"hipSyncPolicyBlockingSync",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4

  // CUkernelNodeAttrID
  m["cudaKernelNodeAttrID"]                                     = {"hipKernelNodeAttrID",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW
  m["cudaKernelNodeAttributeAccessPolicyWindow"]                = {"hipKernelNodeAttributeAccessPolicyWindow",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE
  m["cudaKernelNodeAttributeCooperative"]                       = {"hipKernelNodeAttributeCooperative",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_KERNEL_NODE_ATTRIBUTE_PRIORITY
  m["cudaKernelNodeAttributePriority"]                          = {"hipKernelNodeAttributePriority",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 8

  // CUmemPool_attribute
  m["cudaMemPoolAttr"]                                          = {"hipMemPoolAttr",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemPoolAttr enum values
  // CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
  m["cudaMemPoolReuseFollowEventDependencies"]                  = {"hipMemPoolReuseFollowEventDependencies",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1
  // CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
  m["cudaMemPoolReuseAllowOpportunistic"]                       = {"hipMemPoolReuseAllowOpportunistic",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x2
  // CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
  m["cudaMemPoolReuseAllowInternalDependencies"]                = {"hipMemPoolReuseAllowInternalDependencies",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x3
  // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
  m["cudaMemPoolAttrReleaseThreshold"]                          = {"hipMemPoolAttrReleaseThreshold",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x4
  // CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
  m["cudaMemPoolAttrReservedMemCurrent"]                        = {"hipMemPoolAttrReservedMemCurrent",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x5
  // CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
  m["cudaMemPoolAttrReservedMemHigh"]                           = {"hipMemPoolAttrReservedMemHigh",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x6
  // CU_MEMPOOL_ATTR_USED_MEM_CURRENT
  m["cudaMemPoolAttrUsedMemCurrent"]                            = {"hipMemPoolAttrUsedMemCurrent",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x7
  // CU_MEMPOOL_ATTR_USED_MEM_HIGH
  m["cudaMemPoolAttrUsedMemHigh"]                               = {"hipMemPoolAttrUsedMemHigh",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x8
  // CU_MEMPOOL_ATTR_ALLOCATION_TYPE
  m["cudaMemPoolAttrAllocationType"]                            = {"hipMemPoolAttrAllocationType",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x9
  // CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES
  m["cudaMemPoolAttrExportHandleTypes"]                         = {"hipMemPoolAttrExportHandleTypes",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0xA
  // CU_MEMPOOL_ATTR_LOCATION_ID
  m["cudaMemPoolAttrLocationId"]                                = {"hipMemPoolAttrLocationId",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0xB
  // CU_MEMPOOL_ATTR_LOCATION_TYPE
  m["cudaMemPoolAttrLocationType"]                              = {"hipMemPoolAttrLocationType",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0xC
  // CU_MEMPOOL_ATTR_MAX_POOL_SIZE
  m["cudaMemPoolAttrMaxPoolSize"]                               = {"hipMemPoolAttrMaxPoolSize",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0xD
  // CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED
  m["cudaMemPoolAttrHwDecompressEnabled"]                       = {"hipMemPoolAttrHwDecompressEnabled",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0xE

  // CUmemLocationType
  m["cudaMemLocationType"]                                      = {"hipMemLocationType",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemLocationType enum values
  // CU_MEM_LOCATION_TYPE_INVALID
  m["cudaMemLocationTypeInvalid"]                               = {"hipMemLocationTypeInvalid",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_MEM_LOCATION_TYPE_NONE
  m["cudaMemLocationTypeNone"]                                  = {"hipMemLocationTypeNone",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_MEM_LOCATION_TYPE_DEVICE
  m["cudaMemLocationTypeDevice"]                                = {"hipMemLocationTypeDevice",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_MEM_LOCATION_TYPE_HOST
  m["cudaMemLocationTypeHost"]                                  = {"hipMemLocationTypeHost",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_MEM_LOCATION_TYPE_HOST_NUMA
  m["cudaMemLocationTypeHostNuma"]                              = {"hipMemLocationTypeHostNuma",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT
  m["cudaMemLocationTypeHostNumaCurrent"]                       = {"hipMemLocationTypeHostNumaCurrent",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4

  // CUmemAllocationType
  m["cudaMemAllocationType"]                                    = {"hipMemAllocationType",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUmemAllocationType enum values
  // CU_MEM_ALLOCATION_TYPE_INVALID
  m["cudaMemAllocationTypeInvalid"]                             = {"hipMemAllocationTypeInvalid",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0
  // CU_MEM_ALLOCATION_TYPE_PINNED
  m["cudaMemAllocationTypePinned"]                              = {"hipMemAllocationTypePinned",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1
  // CU_MEM_ALLOCATION_TYPE_MANAGED
  m["cudaMemAllocationTypeManaged"]                             = {"hipMemAllocationTypeManaged",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x2
  // CU_MEM_ALLOCATION_TYPE_MAX
  m["cudaMemAllocationTypeMax"]                                 = {"hipMemAllocationTypeMax",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x7FFFFFFF

  // CUmemAccess_flags
  m["cudaMemAccessFlags"]                                       = {"hipMemAccessFlags",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemAccessFlags enum values
  // CU_MEM_ACCESS_FLAGS_PROT_NONE
  m["cudaMemAccessFlagsProtNone"]                               = {"hipMemAccessFlagsProtNone",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_MEM_ACCESS_FLAGS_PROT_READ
  m["cudaMemAccessFlagsProtRead"]                               = {"hipMemAccessFlagsProtRead",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
  m["cudaMemAccessFlagsProtReadWrite"]                          = {"hipMemAccessFlagsProtReadWrite",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3

  // CUmemAllocationHandleType
  m["cudaMemAllocationHandleType"]                              = {"hipMemAllocationHandleType",                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemAllocationHandleType enum values
  // CU_MEM_HANDLE_TYPE_NONE
  m["cudaMemHandleTypeNone"]                                    = {"hipMemHandleTypeNone",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
  m["cudaMemHandleTypePosixFileDescriptor"]                     = {"hipMemHandleTypePosixFileDescriptor",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_MEM_HANDLE_TYPE_WIN32
  m["cudaMemHandleTypeWin32"]                                   = {"hipMemHandleTypeWin32",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_MEM_HANDLE_TYPE_WIN32_KMT
  m["cudaMemHandleTypeWin32Kmt"]                                = {"hipMemHandleTypeWin32Kmt",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_MEM_HANDLE_TYPE_FABRIC
  m["cudaMemHandleTypeFabric"]                                  = {"hipMemHandleTypeFabric",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 8

  // CUstreamUpdateCaptureDependencies_flags
  m["cudaStreamUpdateCaptureDependenciesFlags"]                 = {"hipStreamUpdateCaptureDependenciesFlags",                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaStreamUpdateCaptureDependenciesFlags enum values
  // CU_STREAM_ADD_CAPTURE_DEPENDENCIES
  m["cudaStreamAddCaptureDependencies"]                         = {"hipStreamAddCaptureDependencies",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0
  // CU_STREAM_SET_CAPTURE_DEPENDENCIES
  m["cudaStreamSetCaptureDependencies"]                         = {"hipStreamSetCaptureDependencies",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1

  // CUuserObject_flags
  m["cudaUserObjectFlags"]                                      = {"hipUserObjectFlags",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaUserObjectFlags enum values
  // CU_USER_OBJECT_NO_DESTRUCTOR_SYNC
  m["cudaUserObjectNoDestructorSync"]                           = {"hipUserObjectNoDestructorSync",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1

  // CUuserObjectRetain_flags
  m["cudaUserObjectRetainFlags"]                                = {"hipUserObjectRetainFlags",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaUserObjectRetainFlags enum values
  // CU_GRAPH_USER_OBJECT_MOVE
  m["cudaGraphUserObjectMove"]                                  = {"hipGraphUserObjectMove",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1

  // CUflushGPUDirectRDMAWritesOptions
  m["cudaFlushGPUDirectRDMAWritesOptions"]                      = {"hipFlushGPUDirectRDMAWritesOptions",                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaFlushGPUDirectRDMAWritesOptions enum values
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST
  m["cudaFlushGPUDirectRDMAWritesOptionHost"]                   = {"hipFlushGPUDirectRDMAWritesOptionHost",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<0
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS
  m["cudaFlushGPUDirectRDMAWritesOptionMemOps"]                 = {"hipFlushGPUDirectRDMAWritesOptionMemOps",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<1

  // CUGPUDirectRDMAWritesOrdering
  m["cudaGPUDirectRDMAWritesOrdering"]                          = {"hipGPUDirectRDMAWritesOrdering",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGPUDirectRDMAWritesOrdering enum values
  // CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE
  m["cudaGPUDirectRDMAWritesOrderingNone"]                      = {"hipGPUDirectRDMAWritesOrderingNone",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
  m["cudaGPUDirectRDMAWritesOrderingOwner"]                     = {"hipGPUDirectRDMAWritesOrderingOwner",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 100
  // CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES
  m["cudaGPUDirectRDMAWritesOrderingAllDevices"]                = {"hipGPUDirectRDMAWritesOrderingAllDevices",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 200

  // CUflushGPUDirectRDMAWritesScope
  m["cudaFlushGPUDirectRDMAWritesScope"]                        = {"hipFlushGPUDirectRDMAWritesScope",                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaFlushGPUDirectRDMAWritesScope enum values
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER
  m["cudaFlushGPUDirectRDMAWritesToOwner"]                      = {"hipFlushGPUDirectRDMAWritesToOwner",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 100
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES
  m["cudaFlushGPUDirectRDMAWritesToAllDevices"]                 = {"hipFlushGPUDirectRDMAWritesToAllDevices",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 200

  // CUflushGPUDirectRDMAWritesTarget
  m["cudaFlushGPUDirectRDMAWritesTarget"]                       = {"hipFlushGPUDirectRDMAWritesTarget",                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaFlushGPUDirectRDMAWritesTarget enum values
  // CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX
  m["cudaFlushGPUDirectRDMAWritesTargetCurrentDevice"]          = {"hipFlushGPUDirectRDMAWritesTargetCurrentDevice",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdriverProcAddress_flags
  m["cudaGetDriverEntryPointFlags"]                             = {"hipGetDriverEntryPointFlags",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGetDriverEntryPointFlags enum values
  // CU_GET_PROC_ADDRESS_DEFAULT
  m["cudaEnableDefault"]                                        = {"hipEnableDefault",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x0
  // CU_GET_PROC_ADDRESS_LEGACY_STREAM
  m["cudaEnableLegacyStream"]                                   = {"hipEnableLegacyStream",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x1
  // CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
  m["cudaEnablePerThreadDefaultStream"]                         = {"hipEnablePerThreadDefaultStream",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0x2

  // CUgraphDebugDot_flags
  m["cudaGraphDebugDotFlags"]                                   = {"hipGraphDebugDotFlags",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphDebugDotFlags enum values
  // CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE
  m["cudaGraphDebugDotFlagsVerbose"]                            = {"hipGraphDebugDotFlagsVerbose",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<0
  // CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS
  m["cudaGraphDebugDotFlagsKernelNodeParams"]                   = {"hipGraphDebugDotFlagsKernelNodeParams",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<2
  // CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS
  m["cudaGraphDebugDotFlagsMemcpyNodeParams"]                   = {"hipGraphDebugDotFlagsMemcpyNodeParams",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<3
  // CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS
  m["cudaGraphDebugDotFlagsMemsetNodeParams"]                   = {"hipGraphDebugDotFlagsMemsetNodeParams",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<4
  // CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS
  m["cudaGraphDebugDotFlagsHostNodeParams"]                     = {"hipGraphDebugDotFlagsHostNodeParams",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<5
  // CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS
  m["cudaGraphDebugDotFlagsEventNodeParams"]                    = {"hipGraphDebugDotFlagsEventNodeParams",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<6
  // CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS
  m["cudaGraphDebugDotFlagsExtSemasSignalNodeParams"]           = {"hipGraphDebugDotFlagsExtSemasSignalNodeParams",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<7
  // CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS
  m["cudaGraphDebugDotFlagsExtSemasWaitNodeParams"]             = {"hipGraphDebugDotFlagsExtSemasWaitNodeParams",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<8
  // CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES
  m["cudaGraphDebugDotFlagsKernelNodeAttributes"]               = {"hipGraphDebugDotFlagsKernelNodeAttributes",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<9
  // CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES
  m["cudaGraphDebugDotFlagsHandles"]                            = {"hipGraphDebugDotFlagsHandles",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1<<10
  // CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS
  m["cudaGraphDebugDotFlagsConditionalNodeParams"]              = {"hipGraphDebugDotFlagsConditionalNodeParams",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 1<<15

  // CUgraphMem_attribute
  m["cudaGraphMemAttributeType"]                                = {"hipGraphMemAttributeType",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphMemAttributeType enum values
  // CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT
  m["cudaGraphMemAttrUsedMemCurrent"]                           = {"hipGraphMemAttrUsedMemCurrent",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GRAPH_MEM_ATTR_USED_MEM_HIGH
  m["cudaGraphMemAttrUsedMemHigh"]                              = {"hipGraphMemAttrUsedMemHigh",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT
  m["cudaGraphMemAttrReservedMemCurrent"]                       = {"hipGraphMemAttrReservedMemCurrent",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH
  m["cudaGraphMemAttrReservedMemHigh"]                          = {"hipGraphMemAttrReservedMemHigh",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphInstantiate_flags
  m["cudaGraphInstantiateFlags"]                                = {"hipGraphInstantiateFlags",                                 "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphInstantiateFlags enum values
  // CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
  m["cudaGraphInstantiateFlagAutoFreeOnLaunch"]                 = {"hipGraphInstantiateFlagAutoFreeOnLaunch",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD
  m["cudaGraphInstantiateFlagUpload"]                           = {"hipGraphInstantiateFlagUpload",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH
  m["cudaGraphInstantiateFlagDeviceLaunch"]                     = {"hipGraphInstantiateFlagDeviceLaunch",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
  m["cudaGraphInstantiateFlagUseNodePriority"]                  = {"hipGraphInstantiateFlagUseNodePriority",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUclusterSchedulingPolicy
  m["cudaClusterSchedulingPolicy"]                              = {"hipClusterSchedulingPolicy",                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaClusterSchedulingPolicy enum values
  // CU_CLUSTER_SCHEDULING_POLICY_DEFAULT
  m["cudaClusterSchedulingPolicyDefault"]                       = {"hipClusterSchedulingPolicyDefault",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_CLUSTER_SCHEDULING_POLICY_SPREAD
  m["cudaClusterSchedulingPolicySpread"]                        = {"hipClusterSchedulingPolicySpread",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING
  m["cudaClusterSchedulingPolicyLoadBalancing"]                 = {"hipClusterSchedulingPolicyLoadBalancing",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUlaunchAttributeID
  m["cudaLaunchAttributeID"]                                    = {"hipLaunchAttributeID",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaLaunchAttributeID enum values
  // CU_LAUNCH_ATTRIBUTE_IGNORE
  m["cudaLaunchAttributeIgnore"]                                = {"hipLaunchAttributeIgnore",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW
  m["cudaLaunchAttributeAccessPolicyWindow"]                    = {"hipLaunchAttributeAccessPolicyWindow",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_COOPERATIVE
  m["cudaLaunchAttributeCooperative"]                           = {"hipLaunchAttributeCooperative",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY
  m["cudaLaunchAttributeSynchronizationPolicy"]                 = {"hipLaunchAttributeSynchronizationPolicy",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
  m["cudaLaunchAttributeClusterDimension"]                      = {"hipLaunchAttributeClusterDimension",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
  m["cudaLaunchAttributeClusterSchedulingPolicyPreference"]     = {"hipLaunchAttributeClusterSchedulingPolicyPreference",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
  m["cudaLaunchAttributeProgrammaticStreamSerialization"]       = {"hipLaunchAttributeProgrammaticStreamSerialization",        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT
  m["cudaLaunchAttributeProgrammaticEvent"]                     = {"hipLaunchAttributeProgrammaticEvent",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_PRIORITY
  m["cudaLaunchAttributePriority"]                              = {"hipLaunchAttributePriority",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
  m["cudaLaunchAttributeMemSyncDomainMap"]                      = {"hipLaunchAttributeMemSyncDomainMap",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN
  m["cudaLaunchAttributeMemSyncDomain"]                         = {"hipLaunchAttributeMemSyncDomain",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION
  m["cudaLaunchAttributePreferredClusterDimension"]             = {"hipLaunchAttributePreferredClusterDimension",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT
  m["cudaLaunchAttributeLaunchCompletionEvent"]                 = {"hipLaunchAttributeLaunchCompletionEvent",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
  m["cudaLaunchAttributeDeviceUpdatableKernelNode"]             = {"hipLaunchAttributeDeviceUpdatableKernelNode",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
  m["cudaLaunchAttributePreferredSharedMemoryCarveout"]         = {"hipLaunchAttributePreferredSharedMemoryCarveout",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING
  m["cudaLaunchAttributeNvlinkUtilCentricScheduling"]           = {"hipLaunchAttributeNvlinkUtilCentricScheduling",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphInstantiateResult
  m["cudaGraphInstantiateResult"]                               = {"hipGraphInstantiateResult",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaGraphInstantiateResult enum values
  // CUDA_GRAPH_INSTANTIATE_SUCCESS
  m["cudaGraphInstantiateSuccess"]                              = {"hipGraphInstantiateSuccess",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_ERROR
  m["cudaGraphInstantiateError"]                                = {"hipGraphInstantiateError",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE
  m["cudaGraphInstantiateInvalidStructure"]                     = {"hipGraphInstantiateInvalidStructure",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED
  m["cudaGraphInstantiateNodeOperationNotSupported"]            = {"hipGraphInstantiateNodeOperationNotSupported",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED
  m["cudaGraphInstantiateMultipleDevicesNotSupported"]          = {"hipGraphInstantiateMultipleDevicesNotSupported",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED
  m["cudaGraphInstantiateConditionalHandleUnused"]              = {"hipGraphInstantiateConditionalHandleUnused",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdriverProcAddressQueryResult
  m["cudaDriverEntryPointQueryResult"]                          = {"hipDriverEntryPointQueryResult",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaDriverEntryPointQueryResult enum values
  // CU_GET_PROC_ADDRESS_SUCCESS
  m["cudaDriverEntryPointSuccess"]                              = {"hipDriverEntryPointSuccess",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
  m["cudaDriverEntryPointSymbolNotFound"]                       = {"hipDriverEntryPointSymbolNotFound",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT
  m["cudaDriverEntryPointVersionNotSufficent"]                  = {"hipDriverEntryPointVersionNotSufficent",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUlaunchMemSyncDomain
  m["cudaLaunchMemSyncDomain"]                                  = {"hipLaunchMemSyncDomain",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaLaunchMemSyncDomain enum values
  // CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT
  m["cudaLaunchMemSyncDomainDefault"]                           = {"hipLaunchMemSyncDomainDefault",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE
  m["cudaLaunchMemSyncDomainRemote"]                            = {"hipLaunchMemSyncDomainRemote",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUdeviceNumaConfig
  m["cudaDeviceNumaConfig"]                                     = {"hipDeviceNumaConfig",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaDeviceNumaConfig enum values
  // CU_DEVICE_NUMA_CONFIG_NONE
  m["cudaDeviceNumaConfigNone"]                                 = {"hipDeviceNumaConfigNone",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEVICE_NUMA_CONFIG_NUMA_NODE
  m["cudaDeviceNumaConfigNumaNode"]                             = {"hipDeviceNumaConfigNumaNode",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogues
  m["cudaGraphConditionalHandleFlags"]                          = {"hipGraphConditionalHandleFlags",                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGraphConditionalHandleFlags enum values
  //
  m["cudaGraphCondAssignDefault"]                               = {"hipGraphCondAssignDefault",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphConditionalNodeType
  m["cudaGraphConditionalNodeType"]                             = {"hipGraphConditionalNodeType",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUgraphConditionalNodeType enum values
  // CU_GRAPH_COND_TYPE_IF
  m["cudaGraphCondTypeIf"]                                      = {"hipGraphCondTypeIf",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_GRAPH_COND_TYPE_WHILE
  m["cudaGraphCondTypeWhile"]                                   = {"hipGraphCondTypeWhile",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_GRAPH_COND_TYPE_SWITCH
  m["cudaGraphCondTypeSwitch"]                                  = {"hipGraphCondTypeSwitch",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphDependencyType
  m["cudaGraphDependencyType"]                                  = {"hipGraphDependencyType",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphDependencyType_enum
  m["cudaGraphDependencyType_enum"]                             = {"hipGraphDependencyType",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // CUgraphDependencyType enum values
  // CU_GRAPH_DEPENDENCY_TYPE_DEFAULT
  m["cudaGraphDependencyTypeDefault"]                           = {"hipGraphDependencyTypeDefault",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
  m["cudaGraphDependencyTypeProgrammatic"]                      = {"hipGraphDependencyTypeProgrammatic",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // no analogue
  m["cudaGraphKernelNodeField"]                                 = {"hipGraphKernelNodeField",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaGraphKernelNodeField enum values
  m["cudaGraphKernelNodeFieldInvalid"]                          = {"hipGraphKernelNodeFieldInvalid",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaGraphKernelNodeFieldGridDim"]                          = {"hipGraphKernelNodeFieldGridDim",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaGraphKernelNodeFieldParam"]                            = {"hipGraphKernelNodeFieldParam",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaGraphKernelNodeFieldEnabled"]                          = {"hipGraphKernelNodeFieldEnabled",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUmemcpyFlags
  m["cudaMemcpyFlags"]                                          = {"hipMemcpyFlags",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemcpyFlags enum values
  // CU_MEMCPY_FLAG_DEFAULT
  m["cudaMemcpyFlagDefault"]                                    = {"hipMemcpyFlagDefault",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE
  m["cudaMemcpyFlagPreferOverlapWithCompute"]                   = {"hipMemcpyFlagPreferOverlapWithCompute",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemcpySrcAccessOrder
  m["cudaMemcpySrcAccessOrder"]                                 = {"hipMemcpySrcAccessOrder",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemcpySrcAccessOrder enum values
  // CU_MEMCPY_SRC_ACCESS_ORDER_INVALID
  m["cudaMemcpySrcAccessOrderInvalid"]                          = {"hipMemcpySrcAccessOrderInvalid",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
  m["cudaMemcpySrcAccessOrderStream"]                           = {"hipMemcpySrcAccessOrderStream",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL
  m["cudaMemcpySrcAccessOrderDuringApiCall"]                    = {"hipMemcpySrcAccessOrderDuringApiCall",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_SRC_ACCESS_ORDER_ANY
  m["cudaMemcpySrcAccessOrderAny"]                              = {"hipMemcpySrcAccessOrderAny",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_SRC_ACCESS_ORDER_MAX
  m["cudaMemcpySrcAccessOrderMax"]                              = {"hipMemcpySrcAccessOrderMax",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemcpy3DOperandType
  m["cudaMemcpy3DOperandType"]                                  = {"hipMemcpy3DOperandType",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaMemcpy3DOperandType enum values
  // CU_MEMCPY_OPERAND_TYPE_POINTER
  m["cudaMemcpyOperandTypePointer"]                             = {"hipMemcpyOperandTypePointer",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_OPERAND_TYPE_ARRAY
  m["cudaMemcpyOperandTypeArray"]                               = {"hipMemcpyOperandTypeArray",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEMCPY_OPERAND_TYPE_MAX
  m["cudaMemcpyOperandTypeMax"]                                 = {"hipMemcpyOperandTypeMax",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  //
  m["CUDAlogLevel"]                                             = {"hipLogLevel",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  //
  m["CUDAlogLevel_enum"]                                        = {"hipLogLevel",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUDAlogLevel enum values
  //
  m["cudaLogLevelError"]                                        = {"hipLogLevelError",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  //
  m["cudaLogLevelWarning"]                                      = {"hipLogLevelWarning",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // NOTE: HIP doesn't have JIT; this dummy enum is used for syntactical compatibility
  // CUjit_option
  m["cudaJitOption"]                                            = {"hipJitOption",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaJitOption enum values
  // CU_JIT_MAX_REGISTERS
  m["cudaJitMaxRegisters"]                                      = {"hipJitOptionMaxRegisters",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_JIT_THREADS_PER_BLOCK
  m["cudaJitThreadsPerBlock"]                                   = {"hipJitOptionThreadsPerBlock",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_JIT_WALL_TIME
  m["cudaJitWallTime"]                                          = {"hipJitOptionWallTime",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_JIT_INFO_LOG_BUFFER
  m["cudaJitInfoLogBuffer"]                                     = {"hipJitOptionInfoLogBuffer",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 3
  // CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
  m["cudaJitInfoLogBufferSizeBytes"]                            = {"hipJitOptionInfoLogBufferSizeBytes",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 4
  // CU_JIT_ERROR_LOG_BUFFER
  m["cudaJitErrorLogBuffer"]                                    = {"hipJitOptionErrorLogBuffer",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 5
  // CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
  m["cudaJitErrorLogBufferSizeBytes"]                           = {"hipJitOptionErrorLogBufferSizeBytes",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 6
  // CU_JIT_OPTIMIZATION_LEVEL
  m["cudaJitOptimizationLevel"]                                 = {"hipJitOptionOptimizationLevel",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 7
  // CU_JIT_FALLBACK_STRATEGY
  m["cudaJitFallbackStrategy"]                                  = {"hipJitOptionFallbackStrategy",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 10
  // CU_JIT_GENERATE_DEBUG_INFO
  m["cudaJitGenerateDebugInfo"]                                 = {"hipJitOptionGenerateDebugInfo",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 11
  // CU_JIT_LOG_VERBOSE
  m["cudaJitLogVerbose"]                                        = {"hipJitOptionLogVerbose",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 12
  // CU_JIT_GENERATE_LINE_INFO
  m["cudaJitGenerateLineInfo"]                                  = {"hipJitOptionGenerateLineInfo",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 13
  // CU_JIT_CACHE_MODE
  m["cudaJitCacheMode"]                                         = {"hipJitOptionCacheMode",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 14
  // CU_JIT_POSITION_INDEPENDENT_CODE
  m["cudaJitPositionIndependentCode"]                           = {"hipJitOptionPositionIndependentCode",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 30
  // CU_JIT_MIN_CTA_PER_SM
  m["cudaJitMinCtaPerSm"]                                       = {"hipJitOptionMinCTAPerSM",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 31
  // CU_JIT_MAX_THREADS_PER_BLOCK
  m["cudaJitMaxThreadsPerBlock"]                                = {"hipJitOptionMaxThreadsPerBlock",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 32
  // CU_JIT_OVERRIDE_DIRECTIVE_VALUES
  m["cudaJitOverrideDirectiveValues"]                           = {"hipJitOptionOverrideDirectiveValues",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES}; // 33

  // CUlibraryOption
  m["cudaLibraryOption"]                                        = {"hipLibraryOption",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};
  // cudaLibraryOption enum values
  // CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE
  m["cudaLibraryHostUniversalFunctionAndDataTable"]             = {"hipLibraryHostUniversalFunctionAndDataTable",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};
  // CU_LIBRARY_BINARY_IS_PRESERVED
  m["cudaLibraryBinaryIsPreserved"]                             = {"hipLibraryBinaryIsPreserved",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES};

  // CUjit_cacheMode
  m["cudaJit_CacheMode"]                                        = {"hipJitCacheMode",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaJit_CacheMode enum values
  // CU_JIT_CACHE_OPTION_NONE
  m["cudaJitCacheOptionNone"]                                   = {"hipJitCacheModeOptionNone",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_JIT_CACHE_OPTION_CG
  m["cudaJitCacheOptionCG"]                                     = {"hipJitCacheModeOptionCG",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_JIT_CACHE_OPTION_CA
  m["cudaJitCacheOptionCA"]                                     = {"hipJitCacheModeOptionCA",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUjit_fallback
  m["cudaJit_Fallback"]                                         = {"hipJitFallback",                                           "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUjit_fallback enum values
  // CU_PREFER_PTX
  m["cudaPreferPtx"]                                            = {"hipJitFallbackPreferPtx",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0
  // CU_PREFER_BINARY
  m["cudaPreferBinary"]                                         = {"hipJitFallbackPreferBinary",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUgraphChildGraphNodeOwnership
  m["cudaGraphChildGraphNodeOwnership"]                         = {"hipGraphChildGraphNodeOwnership",                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUgraphChildGraphNodeOwnership enum values
  // CU_GRAPH_CHILD_GRAPH_OWNERSHIP_CLONE
  m["cudaGraphChildGraphOwnershipClone"]                        = {"hipGraphChildGraphOwnershipClone",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE
  m["cudaGraphChildGraphOwnershipMove"]                         = {"hipGraphChildGraphOwnershipMove",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUatomicOperation
  m["cudaAtomicOperation"]                                      = {"hipAtomicOperation",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaAtomicOperation enum values
  // CU_ATOMIC_OPERATION_INTEGER_ADD
  m["cudaAtomicOperationIntegerAdd"]                            = {"hipAtomicOperationIntegerAdd",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_INTEGER_MIN
  m["cudaAtomicOperationIntegerMin"]                            = {"hipAtomicOperationIntegerMin",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_INTEGER_MAX
  m["cudaAtomicOperationIntegerMax"]                            = {"hipAtomicOperationIntegerMax",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_INTEGER_INCREMENT
  m["cudaAtomicOperationIntegerIncrement"]                      = {"hipAtomicOperationIntegerIncrement",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_INTEGER_DECREMENT
  m["cudaAtomicOperationIntegerDecrement"]                      = {"hipAtomicOperationIntegerDecrement",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_AND
  m["cudaAtomicOperationAnd"]                                   = {"hipAtomicOperationAnd",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_OR
  m["cudaAtomicOperationOr"]                                    = {"hipAtomicOperationOr",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_XOR
  m["cudaAtomicOperationXOR"]                                   = {"hipAtomicOperationXOR",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_EXCHANGE
  m["cudaAtomicOperationExchange"]                              = {"hipAtomicOperationExchange",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_CAS
  m["cudaAtomicOperationCAS"]                                   = {"hipAtomicOperationCAS",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_FLOAT_ADD
  m["cudaAtomicOperationFloatAdd"]                              = {"hipAtomicOperationFloatAdd",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_FLOAT_MIN
  m["cudaAtomicOperationFloatMin"]                              = {"hipAtomicOperationFloatMin",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_OPERATION_FLOAT_MAX
  m["cudaAtomicOperationFloatMax"]                              = {"hipAtomicOperationFloatMax",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUatomicOperationCapability
  m["cudaAtomicOperationCapability"]                            = {"hipAtomicOperationCapability",                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaAtomicOperationCapability enum values
  // CU_ATOMIC_CAPABILITY_SIGNED
  m["cudaAtomicCapabilitySigned"]                               = {"hipAtomicCapabilitySigned",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_UNSIGNED
  m["cudaAtomicCapabilityUnsigned"]                             = {"hipAtomicCapabilityUnsigned",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_REDUCTION
  m["cudaAtomicCapabilityReduction"]                            = {"hipAtomicCapabilityReduction",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_SCALAR_32
  m["cudaAtomicCapabilityScalar32"]                             = {"hipAtomicCapabilityScalar32",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_SCALAR_64
  m["cudaAtomicCapabilityScalar64"]                             = {"hipAtomicCapabilityScalar64",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_SCALAR_128
  m["cudaAtomicCapabilityScalar128"]                            = {"hipAtomicCapabilityScalar128",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_ATOMIC_CAPABILITY_VECTOR_32x4
  m["cudaAtomicCapabilityVector32x4"]                           = {"hipAtomicCapabilityVector32x4",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // [ToDo] Move to a separated Library types, common for Runtime, Driver and Libraries APIs
  m["cudaEmulationStrategy_t"]                                  = {"hipEmulationStrategy_t",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaEmulationStrategy"]                                    = {"hipEmulationStrategy",                                     "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEmulationStrategy enum values
  m["CUDA_EMULATION_STRATEGY_DEFAULT"]                          = {"HIP_EMULATION_STRATEGY_DEFAULT",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_STRATEGY_PERFORMANT"]                       = {"HIP_EMULATION_STRATEGY_PERFORMANT",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_STRATEGY_EAGER"]                            = {"HIP_EMULATION_STRATEGY_EAGER",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // [ToDo] Move to a separated Library types, common for Runtime, Driver and Libraries APIs
  m["cudaEmulationMantissaControl_t"]                           = {"hipEmulationMantissaControl_t",                            "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaEmulationMantissaControl"]                             = {"hipEmulationMantissaControl",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEmulationMantissaControl enum values
  m["CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC"]                  = {"HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_MANTISSA_CONTROL_FIXED"]                    = {"HIP_EMULATION_MANTISSA_CONTROL_FIXED",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // [ToDo] Move to a separated Library types, common for Runtime, Driver and Libraries APIs
  m["cudaEmulationSpecialValuesSupport_t"]                      = {"hipEmulationSpecialValuesSupport_t",                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["cudaEmulationSpecialValuesSupport"]                        = {"hipEmulationSpecialValuesSupport",                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaEmulationSpecialValuesSupport enum values
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT"]            = {"HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE"]               = {"HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY"]           = {"HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN"]                = {"HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevSmResourceGroup_flags
  m["cudaDevSmResourceGroup_flags"]                             = {"hipDevSmResourceGroup_flags",                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaDevSmResourceGroupDefault enum values
  // CU_DEV_SM_RESOURCE_GROUP_DEFAULT
  m["cudaDevSmResourceGroupDefault"]                            = {"hipDevSmResourceGroupDefault",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_SM_RESOURCE_GROUP_BACKFILL
  m["cudaDevSmResourceGroupBackfill"]                           = {"hipDevSmResourceGroupBackfill",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevSmResourceSplitByCount_flags
  m["cudaDevSmResourceSplitByCount_flags"]                      = {"hipDevSmResourceSplitByCount_flags",                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaDevSmResourceSplitByCount_flags enum values
  // CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING
  m["cudaDevSmResourceSplitIgnoreSmCoscheduling"]               = {"hipDevSmResourceSplitIgnoreSmCoscheduling",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE
  m["cudaDevSmResourceSplitMaxPotentialClusterSize"]            = {"hipDevSmResourceSplitMaxPotentialClusterSize",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevResourceType
  m["cudaDevResourceType"]                                      = {"hipDevResourceType",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaDevResourceType enum values
  // CU_DEV_RESOURCE_TYPE_INVALID
  m["cudaDevResourceTypeInvalid"]                               = {"hipDevResourceTypeInvalid",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_RESOURCE_TYPE_SM
  m["cudaDevResourceTypeSm"]                                    = {"hipDevResourceTypeSm",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG
  m["cudaDevResourceTypeWorkqueueConfig"]                       = {"hipDevResourceTypeWorkqueueConfig",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_DEV_RESOURCE_TYPE_WORKQUEUE
  m["cudaDevResourceTypeWorkqueue"]                             = {"hipDevResourceTypeWorkqueue",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUdevWorkqueueConfigScope
  m["cudaDevWorkqueueConfigScope"]                              = {"hipDevWorkqueueConfigScope",                               "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaDevWorkqueueConfigScope enum values
  // CU_WORKQUEUE_SCOPE_DEVICE_CTX
  m["cudaDevWorkqueueConfigScopeDeviceCtx"]                     = {"HIP_WORKQUEUE_SCOPE_DEVICE_CTX",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED
  m["cudaDevWorkqueueConfigScopeGreenCtxBalanced"]              = {"HIP_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUhostTaskSyncMode
  m["cudaHostTaskSyncMode"]                                     = {"hipHostTaskSyncMode",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // cudaHostTaskSyncMode enum values
  // CU_HOST_TASK_BLOCKING
  m["cudaHostTaskBlocking"]                                     = {"hipHostTaskBlocking",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_HOST_TASK_SPINWAIT
  m["cudaHostTaskSpinWait"]                                     = {"hipHostTaskSpinWait",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // 4. Typedefs

  // CUhostFn
  m["cudaHostFn_t"]                                             = {"hipHostFn_t",                                              "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUstreamCallback
  m["cudaStreamCallback_t"]                                     = {"hipStreamCallback_t",                                      "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUsurfObject
  m["cudaSurfaceObject_t"]                                      = {"hipSurfaceObject_t",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUtexObject
  m["cudaTextureObject_t"]                                      = {"hipTextureObject_t",                                       "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUuuid
  m["cudaUUID_t"]                                               = {"hipUUID",                                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUmemoryPool
  m["cudaMemPool_t"]                                            = {"hipMemPool_t",                                             "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUuserObject
  m["cudaUserObject_t"]                                         = {"hipUserObject_t",                                          "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES};

  // CUgraphConditionalHandle
  m["cudaGraphConditionalHandle"]                               = {"hipGraphConditionalHandle",                                "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUasyncCallbackEntry_st
  m["cudaAsyncCallbackEntry"]                                   = {"hipAsyncCallbackEntry",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUasyncCallbackHandle
  m["cudaAsyncCallbackHandle_t"]                                = {"hipAsyncCallbackHandle",                                   "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUasyncCallback
  m["cudaAsyncCallback"]                                        = {"hipAsyncCallback",                                         "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // CUlogsCallbackHandle
  m["cudaLogsCallbackHandle"]                                   = {"hipLogsCallbackHandle",                                    "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUlogsCallbackEntry_st
  m["CUlogsCallbackEntry_st"]                                   = {"hipLogsCallbackEntry_st",                                  "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // no analogue
  m["cudaLogsCallback_t"]                                       = {"hipLogsCallback_t",                                        "", CONV_TYPE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  // 5. Defines

  // no analogue
  m["CUDA_EGL_MAX_PLANES"]                                      = {"HIP_EGL_MAX_PLANES",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 3
  // CU_IPC_HANDLE_SIZE
  m["CUDA_IPC_HANDLE_SIZE"]                                     = {"HIP_IPC_HANDLE_SIZE",                                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 64
  // no analogue
  m["cudaArrayDefault"]                                         = {"hipArrayDefault",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CUDA_ARRAY3D_LAYERED
  m["cudaArrayLayered"]                                         = {"hipArrayLayered",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CUDA_ARRAY3D_SURFACE_LDST
  m["cudaArraySurfaceLoadStore"]                                = {"hipArraySurfaceLoadStore",                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CUDA_ARRAY3D_CUBEMAP
  m["cudaArrayCubemap"]                                         = {"hipArrayCubemap",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CUDA_ARRAY3D_TEXTURE_GATHER
  m["cudaArrayTextureGather"]                                   = {"hipArrayTextureGather",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x08
  // CUDA_ARRAY3D_COLOR_ATTACHMENT
  m["cudaArrayColorAttachment"]                                 = {"hipArrayColorAttachment",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x20
  // CUDA_ARRAY3D_SPARSE
  m["cudaArraySparse"]                                          = {"hipArraySparse",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x40
  // CUDA_ARRAY3D_DEFERRED_MAPPING
  m["cudaArrayDeferredMapping"]                                 = {"hipArrayDeferredMapping",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x80
  // CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
  m["cudaCooperativeLaunchMultiDeviceNoPreSync"]                = {"hipCooperativeLaunchMultiDeviceNoPreSync",                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, CUDA_REMOVED}; // 0x01
  // CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
  m["cudaCooperativeLaunchMultiDeviceNoPostSync"]               = {"hipCooperativeLaunchMultiDeviceNoPostSync",                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, CUDA_REMOVED}; // 0x02
  // CU_DEVICE_CPU ((CUdevice)-1)
  m["cudaCpuDeviceId"]                                          = {"hipCpuDeviceId",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // ((int)-1)
  // CU_DEVICE_INVALID ((CUdevice)-2)
  m["cudaInvalidDeviceId"]                                      = {"hipInvalidDeviceId",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // ((int)-2)
  // CU_CTX_BLOCKING_SYNC
  // NOTE: Deprecated since CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
  m["cudaDeviceBlockingSync"]                                   = {"hipDeviceScheduleBlockingSync",                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, CUDA_DEPRECATED}; // 0x04
  // CU_CTX_LMEM_RESIZE_TO_MAX
  // NOTE: hipDeviceLmemResizeToMax = 0x16
  m["cudaDeviceLmemResizeToMax"]                                = {"hipDeviceLmemResizeToMax",                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x10
  // CU_CTX_SYNC_MEMOPS
  m["cudaDeviceSyncMemops"]                                     = {"hipDeviceSyncMemops",                                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x80
  // CU_CTX_MAP_HOST
  m["cudaDeviceMapHost"]                                        = {"hipDeviceMapHost",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x08
  // CU_CTX_FLAGS_MASK
  m["cudaDeviceMask"]                                           = {"hipDeviceMask",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x1f
  // no analogue
  m["cudaDevicePropDontCare"]                                   = {"hipDevicePropDontCare",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED | CUDA_REMOVED};
  // CU_CTX_SCHED_AUTO
  m["cudaDeviceScheduleAuto"]                                   = {"hipDeviceScheduleAuto",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_CTX_SCHED_SPIN
  m["cudaDeviceScheduleSpin"]                                   = {"hipDeviceScheduleSpin",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_CTX_SCHED_YIELD
  m["cudaDeviceScheduleYield"]                                  = {"hipDeviceScheduleYield",                                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_CTX_SCHED_BLOCKING_SYNC
  m["cudaDeviceScheduleBlockingSync"]                           = {"hipDeviceScheduleBlockingSync",                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CU_CTX_SCHED_MASK
  m["cudaDeviceScheduleMask"]                                   = {"hipDeviceScheduleMask",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x07
  // CU_EVENT_DEFAULT
  m["cudaEventDefault"]                                         = {"hipEventDefault",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_EVENT_BLOCKING_SYNC
  m["cudaEventBlockingSync"]                                    = {"hipEventBlockingSync",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_EVENT_DISABLE_TIMING
  m["cudaEventDisableTiming"]                                   = {"hipEventDisableTiming",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_EVENT_INTERPROCESS
  m["cudaEventInterprocess"]                                    = {"hipEventInterprocess",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CU_EVENT_RECORD_DEFAULT
  m["cudaEventRecordDefault"]                                   = {"hipEventRecordDefault",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_EVENT_RECORD_EXTERNAL
  m["cudaEventRecordExternal"]                                  = {"hipEventRecordExternal",                                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_EVENT_WAIT_DEFAULT
  m["cudaEventWaitDefault"]                                     = {"hipEventWaitDefault",                                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x00
  // CU_EVENT_WAIT_EXTERNAL
  m["cudaEventWaitExternal"]                                    = {"hipEventWaitExternal",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x01
  // CUDA_EXTERNAL_MEMORY_DEDICATED
  m["cudaExternalMemoryDedicated"]                              = {"hipExternalMemoryDedicated",                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x1
  // CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC
  m["cudaExternalSemaphoreSignalSkipNvSciBufMemSync"]           = {"hipExternalSemaphoreSignalSkipNvSciBufMemSync",            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x01
  // CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC
  m["cudaExternalSemaphoreWaitSkipNvSciBufMemSync"]             = {"hipExternalSemaphoreWaitSkipNvSciBufMemSync",              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x02
  // CUDA_NVSCISYNC_ATTR_SIGNAL
  m["cudaNvSciSyncAttrSignal"]                                  = {"hipNvSciSyncAttrSignal",                                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x1
  // CUDA_NVSCISYNC_ATTR_WAIT
  m["cudaNvSciSyncAttrWait"]                                    = {"hipNvSciSyncAttrWait",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x2
  // no analogue
  m["cudaHostAllocDefault"]                                     = {"hipHostMallocDefault",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_MEMHOSTALLOC_PORTABLE
  m["cudaHostAllocPortable"]                                    = {"hipHostMallocPortable",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_MEMHOSTALLOC_DEVICEMAP
  m["cudaHostAllocMapped"]                                      = {"hipHostMallocMapped",                                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_MEMHOSTALLOC_WRITECOMBINED
  m["cudaHostAllocWriteCombined"]                               = {"hipHostMallocWriteCombined",                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // no analogue
  m["cudaHostRegisterDefault"]                                  = {"hipHostRegisterDefault",                                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_MEMHOSTREGISTER_PORTABLE
  m["cudaHostRegisterPortable"]                                 = {"hipHostRegisterPortable",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_MEMHOSTREGISTER_DEVICEMAP
  m["cudaHostRegisterMapped"]                                   = {"hipHostRegisterMapped",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_MEMHOSTREGISTER_IOMEMORY
  m["cudaHostRegisterIoMemory"]                                 = {"hipHostRegisterIoMemory",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // CU_MEMHOSTREGISTER_READ_ONLY
  m["cudaHostRegisterReadOnly"]                                 = {"hipHostRegisterReadOnly",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x08
  // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
  m["cudaIpcMemLazyEnablePeerAccess"]                           = {"hipIpcMemLazyEnablePeerAccess",                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_MEM_ATTACH_GLOBAL
  m["cudaMemAttachGlobal"]                                      = {"hipMemAttachGlobal",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_MEM_ATTACH_HOST
  m["cudaMemAttachHost"]                                        = {"hipMemAttachHost",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // CU_MEM_ATTACH_SINGLE
  m["cudaMemAttachSingle"]                                      = {"hipMemAttachSingle",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x04
  // no analogue
  m["cudaTextureType1D"]                                        = {"hipTextureType1D",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // no analogue
  m["cudaTextureType2D"]                                        = {"hipTextureType2D",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x02
  // no analogue
  m["cudaTextureType3D"]                                        = {"hipTextureType3D",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x03
  // no analogue
  m["cudaTextureTypeCubemap"]                                   = {"hipTextureTypeCubemap",                                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x0C
  // no analogue
  m["cudaTextureType1DLayered"]                                 = {"hipTextureType1DLayered",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0xF1
  // no analogue
  m["cudaTextureType2DLayered"]                                 = {"hipTextureType2DLayered",                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0xF2
  // no analogue
  m["cudaTextureTypeCubemapLayered"]                            = {"hipTextureTypeCubemapLayered",                             "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0xFC
  // CU_OCCUPANCY_DEFAULT
  m["cudaOccupancyDefault"]                                     = {"hipOccupancyDefault",                                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
  m["cudaOccupancyDisableCachingOverride"]                      = {"hipOccupancyDisableCachingOverride",                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_STREAM_DEFAULT
  m["cudaStreamDefault"]                                        = {"hipStreamDefault",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x00
  // CU_STREAM_NON_BLOCKING
  m["cudaStreamNonBlocking"]                                    = {"hipStreamNonBlocking",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0x01
  // CU_STREAM_LEGACY ((CUstream)0x1)
  m["cudaStreamLegacy"]                                         = {"hipStreamLegacy",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // ((cudaStream_t)0x1)
  // CU_STREAM_PER_THREAD ((CUstream)0x2)
  m["cudaStreamPerThread"]                                      = {"hipStreamPerThread",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // ((cudaStream_t)0x2)
  // CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL
  m["cudaArraySparsePropertiesSingleMipTail"]                   = {"hipArraySparsePropertiesSingleMipTail",                    "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x1
  // CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_DIMENSION
  m["cudaKernelNodeAttributeClusterDimension"]                  = {"hipKernelNodeAttributeClusterDimension",                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // cudaLaunchAttributeClusterDimension
  // CU_KERNEL_NODE_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
  m["cudaKernelNodeAttributeClusterSchedulingPolicyPreference"] = {"hipKernelNodeAttributeClusterSchedulingPolicyPreference",  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // cudaLaunchAttributeClusterSchedulingPolicyPreference
  // CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
  m["cudaKernelNodeAttributeMemSyncDomainMap"]                  = {"hipKernelNodeAttributeMemSyncDomainMap",                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // cudaLaunchAttributeMemSyncDomainMap
  // CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN
  m["cudaKernelNodeAttributeMemSyncDomain"]                     = {"hipKernelNodeAttributeMemSyncDomain",                      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // cudaLaunchAttributeMemSyncDomain
  // CU_KERNEL_NODE_ATTRIBUTE_MEM_SYNC_DOMAIN
  m["cudaKernelNodeAttributePreferredSharedMemoryCarveout"]     = {"hipKernelNodeAttributePreferredSharedMemoryCarveout",      "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // cudaLaunchAttributePreferredSharedMemoryCarveout
  //
  m["cudaInitDeviceFlagsAreValid"]                              = {"hipInitDeviceFlagsAreValid",                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x01
  // CUstreamAttrID
  m["cudaStreamAttrID"]                                         = {"hipLaunchAttributeID",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributeID
  // CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW
  m["cudaStreamAttributeAccessPolicyWindow"]                    = {"hipLaunchAttributeAccessPolicyWindow",                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributeAccessPolicyWindow
  // CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY
  m["cudaStreamAttributeSynchronizationPolicy"]                 = {"hipLaunchAttributeSynchronizationPolicy",                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributeSynchronizationPolicy
  // CU_STREAM_ATTRIBUTE_PRIORITY
  m["cudaStreamAttributePriority"]                              = {"hipLaunchAttributePriority",                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributePriority
  // CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP
  m["cudaStreamAttributeMemSyncDomainMap"]                      = {"hipLaunchAttributeMemSyncDomainMap",                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributeMemSyncDomainMap
  // CU_STREAM_ATTRIBUTE_MEM_SYNC_DOMAIN
  m["cudaStreamAttributeMemSyncDomain"]                         = {"hipLaunchAttributeMemSyncDomain",                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // cudaLaunchAttributeMemSyncDomain
  // CU_GRAPH_KERNEL_NODE_PORT_DEFAULT
  m["cudaGraphKernelNodePortDefault"]                           = {"hipGraphKernelNodePortDefault",                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 0
  // CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC
  m["cudaGraphKernelNodePortProgrammatic"]                      = {"hipGraphKernelNodePortProgrammatic",                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 1
  // CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER
  m["cudaGraphKernelNodePortLaunchCompletion"]                  = {"hipGraphKernelNodePortLaunchCompletion",                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES}; // 2
  // CU_KERNEL_NODE_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE
  m["cudaKernelNodeAttributeDeviceUpdatableKernelNode"]         = {"hipKernelNodeAttributeDeviceUpdatableKernelNode",          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CU_LAUNCH_ATTRIBUTE_NVLINK_UTIL_CENTRIC_SCHEDULING
  m["cudaKernelNodeAttributeNvlinkUtilCentricScheduling"]       = {"hipLaunchAttributeNvlinkUtilCentricScheduling",            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};
  // CUlogIterator
  m["cudaLogIterator"]                                          = {"hipLogIterator",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED};

  m["CUDART_INF_F"]                                             = {"HIP_INF_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_NAN_F"]                                             = {"HIP_NAN_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_MIN_DENORM_F"]                                      = {"HIP_MIN_DENORM_F",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_MAX_NORMAL_F"]                                      = {"HIP_MAX_NORMAL_F",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_NEG_ZERO_F"]                                        = {"HIP_NEG_ZERO_F",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_ZERO_F"]                                            = {"HIP_ZERO_F",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_ONE_F"]                                             = {"HIP_ONE_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF_F"]                                       = {"HIP_SQRT_HALF_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF_HI_F"]                                    = {"HIP_SQRT_HALF_HI_F",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF_LO_F"]                                    = {"HIP_SQRT_HALF_LO_F",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_TWO_F"]                                        = {"HIP_SQRT_TWO_F",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_THIRD_F"]                                           = {"HIP_THIRD_F",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO4_F"]                                            = {"HIP_PIO4_F",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO2_F"]                                            = {"HIP_PIO2_F",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_3PIO4_F"]                                           = {"HIP_3PIO4_F",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_2_OVER_PI_F"]                                       = {"HIP_2_OVER_PI_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_2_OVER_PI_F"]                                  = {"HIP_SQRT_2_OVER_PI_F",                                     "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PI_F"]                                              = {"HIP_PI_F",                                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2E_F"]                                             = {"HIP_L2E_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2T_F"]                                             = {"HIP_L2T_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2_F"]                                             = {"HIP_LG2_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LGE_F"]                                             = {"HIP_LGE_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_F"]                                             = {"HIP_LN2_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNT_F"]                                             = {"HIP_LNT_F",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNPI_F"]                                            = {"HIP_LNPI_F",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_M126_F"]                                     = {"HIP_TWO_TO_M126_F",                                        "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_126_F"]                                      = {"HIP_TWO_TO_126_F",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_NORM_HUGE_F"]                                       = {"HIP_NORM_HUGE_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_23_F"]                                       = {"HIP_TWO_TO_23_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_24_F"]                                       = {"HIP_TWO_TO_24_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_31_F"]                                       = {"HIP_TWO_TO_31_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_32_F"]                                       = {"HIP_TWO_TO_32_F",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_REMQUO_BITS_F"]                                     = {"HIP_REMQUO_BITS_F",                                        "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_REMQUO_MASK_F"]                                     = {"HIP_REMQUO_MASK_F",                                        "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TRIG_PLOSS_F"]                                      = {"HIP_TRIG_PLOSS_F",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_INF"]                                               = {"HIP_INF",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_NAN"]                                               = {"HIP_NAN",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_NEG_ZERO"]                                          = {"HIP_NEG_ZERO",                                             "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_MIN_DENORM"]                                        = {"HIP_MIN_DENORM",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_ZERO"]                                              = {"HIP_ZERO",                                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_ONE"]                                               = {"HIP_ONE",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_TWO"]                                          = {"HIP_SQRT_TWO",                                             "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF"]                                         = {"HIP_SQRT_HALF",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF_HI"]                                      = {"HIP_SQRT_HALF_HI",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_HALF_LO"]                                      = {"HIP_SQRT_HALF_LO",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_THIRD"]                                             = {"HIP_THIRD",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWOTHIRD"]                                          = {"HIP_TWOTHIRD",                                             "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO4"]                                              = {"HIP_PIO4",                                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO4_HI"]                                           = {"HIP_PIO4_HI",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO4_LO"]                                           = {"HIP_PIO4_LO",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO2"]                                              = {"HIP_PIO2",                                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO2_HI"]                                           = {"HIP_PIO2_HI",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PIO2_LO"]                                           = {"HIP_PIO2_LO",                                              "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_3PIO4"]                                             = {"HIP_3PIO4",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_2_OVER_PI"]                                         = {"HIP_2_OVER_PI",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PI"]                                                = {"HIP_PI",                                                   "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PI_HI"]                                             = {"HIP_PI_HI",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_PI_LO"]                                             = {"HIP_PI_LO",                                                "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_2PI"]                                          = {"HIP_SQRT_2PI",                                             "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_2PI_HI"]                                       = {"HIP_SQRT_2PI_HI",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_2PI_LO"]                                       = {"HIP_SQRT_2PI_LO",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_PIO2"]                                         = {"HIP_SQRT_PIO2",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_PIO2_HI"]                                      = {"HIP_SQRT_PIO2_HI",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_PIO2_LO"]                                      = {"HIP_SQRT_PIO2_LO",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_SQRT_2OPI"]                                         = {"HIP_SQRT_2OPI",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2E"]                                               = {"HIP_L2E",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2E_HI"]                                            = {"HIP_L2E_HI",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2E_LO"]                                            = {"HIP_L2E_LO",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_L2T"]                                               = {"HIP_L2T",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2"]                                               = {"HIP_LG2",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2_HI"]                                            = {"HIP_LG2_HI",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2_LO"]                                            = {"HIP_LG2_LO",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LGE"]                                               = {"HIP_LGE",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LGE_HI"]                                            = {"HIP_LGE_HI",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LGE_LO"]                                            = {"HIP_LGE_LO",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2"]                                               = {"HIP_LN2",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_HI"]                                            = {"HIP_LN2_HI",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_LO"]                                            = {"HIP_LN2_LO",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNT"]                                               = {"HIP_LNT",                                                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNT_HI"]                                            = {"HIP_LNT_HI",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNT_LO"]                                            = {"HIP_LNT_LO",                                               "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LNPI"]                                              = {"HIP_LNPI",                                                 "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_X_1024"]                                        = {"HIP_LN2_X_1024",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_X_1025"]                                        = {"HIP_LN2_X_1025",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LN2_X_1075"]                                        = {"HIP_LN2_X_1075",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2_X_1024"]                                        = {"HIP_LG2_X_1024",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_LG2_X_1075"]                                        = {"HIP_LG2_X_1075",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_23"]                                         = {"HIP_TWO_TO_23",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_52"]                                         = {"HIP_TWO_TO_52",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_53"]                                         = {"HIP_TWO_TO_53",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_54"]                                         = {"HIP_TWO_TO_54",                                            "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_M54"]                                        = {"HIP_TWO_TO_M54",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TWO_TO_M1022"]                                      = {"HIP_TWO_TO_M1022",                                         "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_TRIG_PLOSS"]                                        = {"HIP_TRIG_PLOSS",                                           "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  m["CUDART_DBL2INT_CVT"]                                       = {"HIP_DBL2INT_CVT",                                          "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES};
  // CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS
  m["cudaMemPoolCreateUsageHwDecompress"]                       = {"HIP_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS",                  "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 0x2

  m["RESOURCE_ABI_BYTES"]                                       = {"RESOURCE_ABI_BYTES",                                       "", CONV_DEFINE, API_RUNTIME, SEC::DATA_TYPES, HIP_UNSUPPORTED}; // 40

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RUNTIME_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cudaEglFrame"]                                             = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglFrame_st"]                                          = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglPlaneDesc"]                                         = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglPlaneDesc_st"]                                      = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryBufferDesc"]                             = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleDesc"]                             = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryMipmappedArrayDesc"]                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleDesc"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreSignalParams"]                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreWaitParams"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaHostNodeParams"]                                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeParams"]                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaLaunchParams"]                                         = {CUDA_90,  CUDA_0,   CUDA_130};
  m["cudaMemsetParams"]                                         = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUexternalMemory_st"]                                      = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemory_t"]                                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUexternalSemaphore_st"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphore_t"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUgraph_st"]                                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraph_t"]                                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUgraphExec_st"]                                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphExec_t"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUgraphNode_st"]                                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNode_t"]                                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUeglStreamConnection_st"]                                 = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglStreamConnection"]                                  = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaFunction_t"]                                           = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaAccessPolicyWindow"]                                   = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamAttrValue"]                                      = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttrValue"]                                  = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaCGScope"]                                              = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaCGScopeInvalid"]                                       = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaCGScopeGrid"]                                          = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaCGScopeMultiGrid"]                                     = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostNativeAtomicSupported"]                     = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrSingleToDoublePrecisionPerfRatio"]              = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrPageableMemoryAccess"]                          = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrConcurrentManagedAccess"]                       = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrComputePreemptionSupported"]                    = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrCanUseHostPointerForRegisteredMem"]             = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved92"]                                    = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved93"]                                    = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved94"]                                    = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrCooperativeLaunch"]                             = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrCooperativeMultiDeviceLaunch"]                  = {CUDA_90,  CUDA_114, CUDA_130};
  m["cudaDevAttrMaxSharedMemoryPerBlockOptin"]                  = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrCanFlushRemoteWrites"]                          = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostRegisterSupported"]                         = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrPageableMemoryAccessUsesHostPageTables"]        = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrDirectManagedMemAccessFromHost"]                = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrMaxBlocksPerMultiprocessor"]                    = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReservedSharedMemoryPerBlock"]                  = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaDeviceP2PAttr"]                                        = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevP2PAttrPerformanceRank"]                            = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevP2PAttrAccessSupported"]                            = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevP2PAttrNativeAtomicSupported"]                      = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaDevP2PAttrCudaArrayAccessSupported"]                   = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormat"]                                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV422Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV422SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatRGB"]                                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBGR"]                                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatARGB"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatRGBA"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatL"]                                      = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatR"]                                      = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV444Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV444SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUYV422"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatUYVY422"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatABGR"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBGRA"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatA"]                                      = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatRG"]                                     = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatAYUV"]                                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU444SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU422SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420SemiPlanar"]                       = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_444SemiPlanar"]                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_420SemiPlanar"]                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_444SemiPlanar"]                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_420SemiPlanar"]                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatVYUY_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatUYVY_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUYV_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVYU_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV_ER"]                                 = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUVA_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatAYUV_ER"]                                = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV444Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV422Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV444SemiPlanar_ER"]                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV422SemiPlanar_ER"]                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420SemiPlanar_ER"]                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU444Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU422Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420Planar_ER"]                        = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU444SsemiPlanar_ER"]                   = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU422SemiPlanar_ER"]                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420SemiPlanar_ER"]                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerRGGB"]                              = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerBGGR"]                              = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerGRBG"]                              = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerGBRG"]                              = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer10RGGB"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer10BGGR"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer10GRBG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer10GBRG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12RGGB"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12BGGR"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12GRBG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12GBRG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer14RGGB"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer14BGGR"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer14GRBG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer14GBRG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer20RGGB"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer20BGGR"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer20GRBG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer20GBRG"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU444Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU422Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420Planar"]                           = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerIspRGGB"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerIspBGGR"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerIspGRBG"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerIspGBRG"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cudaEglFrameType"]                                         = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglFrameTypeArray"]                                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglFrameTypePitch"]                                    = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglResourceLocationFlags"]                             = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglResourceLocationSysmem"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaEglResourceLocationVidmem"]                            = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaErrorProfilerNotInitialized"]                          = {CUDA_0,   CUDA_50,  CUDA_0  };
  m["cudaErrorProfilerAlreadyStarted"]                          = {CUDA_0,   CUDA_50,  CUDA_0  };
  m["cudaErrorProfilerAlreadyStopped"]                          = {CUDA_0,   CUDA_50,  CUDA_0  };
  m["cudaErrorInvalidHostPointer"]                              = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaErrorInvalidDevicePointer"]                            = {CUDA_0,   CUDA_101, CUDA_0  };
  m["cudaErrorAddressOfConstant"]                               = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorTextureFetchFailed"]                              = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorTextureNotBound"]                                 = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorSynchronizationError"]                            = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorMixedDeviceExecution"]                            = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorNotYetImplemented"]                               = {CUDA_0,   CUDA_41,  CUDA_0  };
  m["cudaErrorMemoryValueTooLarge"]                             = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorPriorLaunchFailure"]                              = {CUDA_0,   CUDA_31,  CUDA_0  };
  m["cudaErrorArrayIsMapped"]                                   = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorAlreadyMapped"]                                   = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorDeviceUninitialized"]                             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaErrorAlreadyAcquired"]                                 = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorNotMapped"]                                       = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorNotMappedAsArray"]                                = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorNotMappedAsPointer"]                              = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorNvlinkUncorrectable"]                             = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaErrorJitCompilerNotFound"]                             = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaErrorInvalidSource"]                                   = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorFileNotFound"]                                    = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorIllegalState"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorSymbolNotFound"]                                  = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorLaunchIncompatibleTexturing"]                     = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorContextIsDestroyed"]                              = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorCooperativeLaunchTooLarge"]                       = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaErrorSystemNotReady"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorSystemDriverMismatch"]                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorCompatNotSupportedOnDevice"]                      = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureUnsupported"]                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureInvalidated"]                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureMerge"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureUnmatched"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureUnjoined"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureIsolation"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureImplicit"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorCapturedEvent"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaErrorStreamCaptureWrongThread"]                        = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaErrorTimeout"]                                         = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaErrorGraphExecUpdateFailure"]                          = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaErrorApiFailureBase"]                                  = {CUDA_0,   CUDA_41,  CUDA_0  };
  m["cudaExternalMemoryHandleType"]                             = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeOpaqueFd"]                     = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeOpaqueWin32"]                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeOpaqueWin32Kmt"]               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeD3D12Heap"]                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeD3D12Resource"]                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeD3D11Resource"]                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeD3D11ResourceKmt"]             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryHandleTypeNvSciBuf"]                     = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleType"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeOpaqueFd"]                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeOpaqueWin32"]               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt"]            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeD3D12Fence"]                = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeD3D11Fence"]                = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeNvSciSync"]                 = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeKeyedMutex"]                = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeKeyedMutexKmt"]             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaFuncAttribute"]                                        = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeMaxDynamicSharedMemorySize"]              = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaFuncAttributePreferredSharedMemoryCarveout"]           = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeMax"]                                     = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaGraphNodeType"]                                        = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeKernel"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeMemcpy"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeMemset"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeHost"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeGraph"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeEmpty"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeCount"]                                   = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateResult"]                                = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateSuccess"]                               = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateError"]                                 = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorTopologyChanged"]                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorNodeTypeChanged"]                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorFunctionChanged"]                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorParametersChanged"]                = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorNotSupported"]                     = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaLimitMaxL2FetchGranularity"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaLimitPersistingL2CacheSize"]                           = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaMemoryAdvise"]                                         = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseSetReadMostly"]                               = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseUnsetReadMostly"]                             = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseSetPreferredLocation"]                        = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseUnsetPreferredLocation"]                      = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseSetAccessedBy"]                               = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemAdviseUnsetAccessedBy"]                             = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemoryTypeManaged"]                                    = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttribute"]                                    = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributeReadMostly"]                          = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributePreferredLocation"]                   = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributeAccessedBy"]                          = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributeLastPrefetchLocation"]                = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaSharedCarveout"]                                       = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaSharedmemCarveoutDefault"]                             = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaSharedmemCarveoutMaxShared"]                           = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaSharedmemCarveoutMaxL1"]                               = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureStatus"]                                  = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureStatusNone"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureStatusActive"]                            = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureStatusInvalidated"]                       = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureMode"]                                    = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureModeGlobal"]                              = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureModeThreadLocal"]                         = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cudaStreamCaptureModeRelaxed"]                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["libraryPropertyType"]                                      = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["libraryPropertyType_t"]                                    = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaAccessProperty"]                                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaAccessPropertyNormal"]                                 = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaAccessPropertyStreaming"]                              = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaAccessPropertyPersisting"]                             = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaSynchronizationPolicy"]                                = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaSyncPolicyAuto"]                                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaSyncPolicySpin"]                                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaSyncPolicyYield"]                                      = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaSyncPolicyBlockingSync"]                               = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamAttrID"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamAttributeAccessPolicyWindow"]                    = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaStreamAttributeSynchronizationPolicy"]                 = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttrID"]                                     = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeAccessPolicyWindow"]                = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeCooperative"]                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cudaHostFn_t"]                                             = {CUDA_100, CUDA_0,   CUDA_0  };
  m["CUDA_EGL_MAX_PLANES"]                                      = {CUDA_91,  CUDA_0,   CUDA_0  };
  m["cudaArrayColorAttachment"]                                 = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaCooperativeLaunchMultiDeviceNoPreSync"]                = {CUDA_90,  CUDA_0,   CUDA_130};
  m["cudaCooperativeLaunchMultiDeviceNoPostSync"]               = {CUDA_90,  CUDA_0,   CUDA_130};
  m["cudaCpuDeviceId"]                                          = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaInvalidDeviceId"]                                      = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cudaExternalMemoryDedicated"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaMemoryTypeUnregistered"]                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreSignalSkipNvSciBufMemSync"]           = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreWaitSkipNvSciBufMemSync"]             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaNvSciSyncAttrSignal"]                                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaNvSciSyncAttrWait"]                                    = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cudaHostRegisterIoMemory"]                                 = {CUDA_75,  CUDA_0,   CUDA_0  };
  m["cudaHostRegisterReadOnly"]                                 = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEventRecordDefault"]                                   = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEventRecordExternal"]                                  = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEventWaitDefault"]                                     = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaArraySparse"]                                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaErrorStubLibrary"]                                     = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaErrorCallRequiresNewerDriver"]                         = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaErrorDeviceNotLicensed"]                               = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaErrorUnsupportedPtxVersion"]                           = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaArraySparsePropertiesSingleMipTail"]                   = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaArraySparseProperties"]                                = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaDevAttrSparseCudaArraySupported"]                      = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostRegisterReadOnlySupported"]                 = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeWaitEvent"]                               = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeEventRecord"]                             = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaErrorSoftwareValidityNotEstablished"]                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaErrorJitCompilationDisabled"]                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindNV12"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMaxTimelineSemaphoreInteropSupported"]          = {CUDA_112, CUDA_115, CUDA_130};
  m["cudaDevAttrMemoryPoolsSupported"]                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttr"]                                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolReuseFollowEventDependencies"]                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolReuseAllowOpportunistic"]                       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolReuseAllowInternalDependencies"]                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrReleaseThreshold"]                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemLocationType"]                                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeInvalid"]                               = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeDevice"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemLocation"]                                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAccessFlags"]                                       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAccessFlagsProtNone"]                               = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAccessFlagsProtRead"]                               = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAccessFlagsProtReadWrite"]                          = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAccessDesc"]                                        = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationType"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationTypeInvalid"]                             = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationTypePinned"]                              = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationTypeMax"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationHandleType"]                              = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemHandleTypeNone"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemHandleTypePosixFileDescriptor"]                     = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemHandleTypeWin32"]                                   = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemHandleTypeWin32Kmt"]                                = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolProps"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaMemPoolPtrExportData"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd"]       = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32"]    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreSignalParams_v1"]                     = {CUDA_112, CUDA_112, CUDA_113};
  m["cudaExternalSemaphoreWaitParams_v1"]                       = {CUDA_112, CUDA_112, CUDA_113};
  m["cudaMemPool_t"]                                            = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreSignalNodeParams"]                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreWaitNodeParams"]                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorUnsupportedFunctionChange"]        = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaStreamUpdateCaptureDependenciesFlags"]                 = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaStreamAddCaptureDependencies"]                         = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaStreamSetCaptureDependencies"]                         = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectFlags"]                                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectNoDestructorSync"]                           = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObjectRetainFlags"]                                = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphUserObjectMove"]                                  = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesOptions"]                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesOptionHost"]                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesOptionMemOps"]                 = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGPUDirectRDMAWritesOrdering"]                          = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGPUDirectRDMAWritesOrderingNone"]                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGPUDirectRDMAWritesOrderingOwner"]                     = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGPUDirectRDMAWritesOrderingAllDevices"]                = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesScope"]                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesToOwner"]                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesToAllDevices"]                 = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesTarget"]                       = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaFlushGPUDirectRDMAWritesTargetCurrentDevice"]          = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMaxPersistingL2CacheSize"]                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMaxAccessPolicyWindowSize"]                     = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrGPUDirectRDMASupported"]                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrGPUDirectRDMAFlushWritesOptions"]               = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrGPUDirectRDMAWritesOrdering"]                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMemoryPoolSupportedHandleTypes"]                = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrReservedMemCurrent"]                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrReservedMemHigh"]                           = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrUsedMemCurrent"]                            = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrUsedMemHigh"]                               = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaUserObject_t"]                                         = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGetDriverEntryPointFlags"]                             = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaEnableDefault"]                                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaEnableLegacyStream"]                                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaEnablePerThreadDefaultStream"]                         = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlags"]                                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsVerbose"]                            = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsKernelNodeParams"]                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsMemcpyNodeParams"]                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsMemsetNodeParams"]                   = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsHostNodeParams"]                     = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsEventNodeParams"]                    = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsExtSemasSignalNodeParams"]           = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsExtSemasWaitNodeParams"]             = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsKernelNodeAttributes"]               = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsHandles"]                            = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cudaErrorUnsupportedExecAffinity"]                         = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsConnectionFailed"]                             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsRpcFailure"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsServerNotReady"]                               = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsMaxClientsReached"]                            = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsMaxConnectionsReached"]                        = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMax"]                                           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaMemAllocNodeParams"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAttributeType"]                                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAttrUsedMemCurrent"]                           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAttrUsedMemHigh"]                              = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAttrReservedMemCurrent"]                       = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphMemAttrReservedMemHigh"]                          = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeExtSemaphoreSignal"]                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeExtSemaphoreWait"]                        = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeMemAlloc"]                                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeMemFree"]                                 = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateFlags"]                                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateFlagAutoFreeOnLaunch"]                 = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized8X1"]               = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized8X2"]               = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized8X4"]               = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized16X1"]              = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized16X2"]              = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized16X4"]              = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized8X1"]                 = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized8X2"]                 = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized8X4"]                 = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized16X1"]                = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized16X2"]                = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedNormalized16X4"]                = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed1"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed1SRGB"]        = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed2"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed2SRGB"]        = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed3"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed3SRGB"]        = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed4"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedBlockCompressed4"]              = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed5"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedBlockCompressed5"]              = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed6H"]           = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindSignedBlockCompressed6H"]             = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed7"]            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedBlockCompressed7SRGB"]        = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaDevAttrTimelineSemaphoreInteropSupported"]             = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cudaArrayDeferredMapping"]                                 = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaArrayMemoryRequirements"]                              = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaDevAttrDeferredMappingCudaArraySupported"]             = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateErrorAttributesChanged"]                = {CUDA_116, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributePriority"]                          = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateFlagUseNodePriority"]                  = {CUDA_117, CUDA_0,   CUDA_0  };
  m["cudaErrorMpsClientTerminated"]                             = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaErrorInvalidClusterSize"]                              = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaClusterSchedulingPolicy"]                              = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaClusterSchedulingPolicyDefault"]                       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaClusterSchedulingPolicySpread"]                        = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaClusterSchedulingPolicyLoadBalancing"]                 = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeClusterDimMustBeSet"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeRequiredClusterWidth"]                    = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeRequiredClusterHeight"]                   = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeRequiredClusterDepth"]                    = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeNonPortableClusterSizeAllowed"]           = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaFuncAttributeClusterSchedulingPolicyPreference"]       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaDevAttrClusterLaunch"]                                 = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeID"]                                    = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeIgnore"]                                = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeAccessPolicyWindow"]                    = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeCooperative"]                           = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeSynchronizationPolicy"]                 = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeClusterDimension"]                      = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeClusterSchedulingPolicyPreference"]     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeProgrammaticStreamSerialization"]       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeProgrammaticEvent"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributePriority"]                              = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeValue"]                                 = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttribute_st"]                                   = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttribute"]                                      = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchConfig_st"]                                      = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaLaunchConfig_t"]                                       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeClusterDimension"]                  = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeClusterSchedulingPolicyPreference"] = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cudaInitDeviceFlagsAreValid"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaErrorCdpNotSupported"]                                 = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaErrorCdpVersionMismatch"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaOutputMode"]                                           = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaOutputMode_t"]                                         = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaKeyValuePair"]                                         = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaCSV"]                                                  = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaDevAttrReserved122"]                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved123"]                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved124"]                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDevAttrIpcEventSupport"]                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMemSyncDomainCount"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDevicePropDontCare"]                                   = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaGraphInstantiateResult"]                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateSuccess"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateError"]                                = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateInvalidStructure"]                     = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateNodeOperationNotSupported"]            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateMultipleDevicesNotSupported"]          = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateParams_st"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateParams"]                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateResultInfo_st"]                         = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphExecUpdateResultInfo"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDriverEntryPointQueryResult"]                          = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDriverEntryPointSuccess"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDriverEntryPointSymbolNotFound"]                       = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaDriverEntryPointVersionNotSufficent"]                  = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateFlagUpload"]                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateFlagDeviceLaunch"]                     = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchMemSyncDomain"]                                  = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchMemSyncDomainDefault"]                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchMemSyncDomainRemote"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchMemSyncDomainMap_st"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchMemSyncDomainMap"]                               = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeMemSyncDomainMap"]                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeMemSyncDomain"]                         = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaStreamAttributePriority"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaStreamAttributeMemSyncDomainMap"]                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaStreamAttributeMemSyncDomain"]                         = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeMemSyncDomainMap"]                  = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeMemSyncDomain"]                     = {CUDA_120, CUDA_0,   CUDA_0  };
  m["texture"]                                                  = {CUDA_0,   CUDA_0,   CUDA_120};
  m["surfaceReference"]                                         = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cudaDeviceSyncMemops"]                                     = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaErrorUnsupportedDevSideSync"]                          = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved127"]                                   = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved128"]                                   = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved129"]                                   = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved132"]                                   = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaKernel_t"]                                             = {CUDA_121, CUDA_0,   CUDA_0  };
  m["cudaMemcpyNodeParams"]                                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemsetParamsV2"]                                       = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaHostNodeParamsV2"]                                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributePreferredLocationType"]               = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributePreferredLocationId"]                 = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributeLastPrefetchLocationType"]            = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemRangeAttributeLastPrefetchLocationId"]              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDevAttrNumaConfig"]                                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDevAttrNumaId"]                                        = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostNumaId"]                                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeHost"]                                  = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeHostNuma"]                              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeHostNumaCurrent"]                       = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemAllocNodeParamsV2"]                                 = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaMemFreeNodeParams"]                                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeParamsV2"]                                   = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreSignalNodeParamsV2"]                  = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaExternalSemaphoreWaitNodeParamsV2"]                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaChildGraphNodeParams"]                                 = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaEventRecordNodeParams"]                                = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaEventWaitNodeParams"]                                  = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeParams"]                                      = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDeviceNumaConfig"]                                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDeviceNumaConfigNone"]                                 = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaDeviceNumaConfigNumaNode"]                             = {CUDA_122, CUDA_0,   CUDA_0  };
  m["cudaErrorLossyQuery"]                                      = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaDevAttrMpsEnabled"]                                    = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaMemFabricHandle_st"]                                   = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaMemFabricHandle_t"]                                    = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphConditionalHandle"]                               = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphConditionalHandleFlags"]                          = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphCondAssignDefault"]                               = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphConditionalNodeType"]                             = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphCondTypeIf"]                                      = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphCondTypeWhile"]                                   = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaConditionalNodeParams"]                                = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphNodeTypeConditional"]                             = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphDependencyType"]                                  = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphDependencyType_enum"]                             = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphDependencyTypeDefault"]                           = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphDependencyTypeProgrammatic"]                      = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphEdgeData_st"]                                     = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphEdgeData"]                                        = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodePortDefault"]                           = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodePortProgrammatic"]                      = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodePortLaunchCompletion"]                  = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaGraphDebugDotFlagsConditionalNodeParams"]              = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeLaunchCompletionEvent"]                 = {CUDA_123, CUDA_0,   CUDA_0  };
  m["cudaSharedMemConfig"]                                      = {CUDA_0,   CUDA_124, CUDA_0  };
  m["cudaMemHandleTypeFabric"]                                  = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphDeviceNode_t"]                                    = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeField"]                                 = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeFieldInvalid"]                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeFieldGridDim"]                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeFieldParam"]                            = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeFieldEnabled"]                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaGraphKernelNodeUpdate"]                                = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeDeviceUpdatableKernelNode"]             = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeDeviceUpdatableKernelNode"]         = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaAsyncCallback"]                                        = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaAsyncCallbackEntry"]                                   = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaAsyncCallbackHandle_t"]                                = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaAsyncNotificationInfo"]                                = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaAsyncNotificationInfo_t"]                              = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cudaStreamLegacy"]                                         = {CUDA_90,  CUDA_0,   CUDA_0  };
  m["cudaDevAttrD3D12CigSupported"]                             = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributePreferredSharedMemoryCarveout"]         = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributePreferredSharedMemoryCarveout"]     = {CUDA_125, CUDA_0,   CUDA_0  };
  m["cudaErrorFunctionNotLoaded"]                               = {CUDA_126, CUDA_0,   CUDA_0  };
  m["cudaErrorInvalidResourceType"]                             = {CUDA_126, CUDA_0,   CUDA_0  };
  m["cudaErrorInvalidResourceConfiguration"]                    = {CUDA_126, CUDA_0,   CUDA_0  };
  m["cudaErrorContained"]                                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaErrorTensorMemoryLeak"]                                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaChannelFormatKindUnsignedNormalized1010102"]           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaDevAttrGpuPciDeviceId"]                                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaDevAttrGpuPciSubsystemId"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostNumaMultinodeIpcSupported"]                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemPoolCreateUsageHwDecompress"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyFlags"]                                          = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyFlagDefault"]                                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyFlagPreferOverlapWithCompute"]                   = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrder"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrderInvalid"]                          = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrderStream"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrderDuringApiCall"]                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrderAny"]                              = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpySrcAccessOrderMax"]                              = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyAttributes"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpy3DOperandType"]                                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyOperandTypePointer"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyOperandTypeArray"]                               = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpyOperandTypeMax"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaOffset3D"]                                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpy3DOperand"]                                      = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaMemcpy3DBatchOp"]                                      = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitOption"]                                            = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitMaxRegisters"]                                      = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitThreadsPerBlock"]                                   = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitWallTime"]                                          = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitInfoLogBuffer"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitInfoLogBufferSizeBytes"]                            = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitErrorLogBuffer"]                                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitErrorLogBufferSizeBytes"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitOptimizationLevel"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitFallbackStrategy"]                                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitGenerateDebugInfo"]                                 = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitLogVerbose"]                                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitGenerateLineInfo"]                                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitCacheMode"]                                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitPositionIndependentCode"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitMinCtaPerSm"]                                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitMaxThreadsPerBlock"]                                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitOverrideDirectiveValues"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryOption"]                                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryHostUniversalFunctionAndDataTable"]             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudalibraryHostUniversalFunctionAndDataTable"]             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibraryBinaryIsPreserved"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJit_CacheMode"]                                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitCacheOptionNone"]                                   = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitCacheOptionCG"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJitCacheOptionCA"]                                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaJit_Fallback"]                                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaPreferPtx"]                                            = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaPreferBinary"]                                         = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLibrary_t"]                                            = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaGraphCondTypeSwitch"]                                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaGraphInstantiateConditionalHandleUnused"]              = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributePreferredClusterDimension"]             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatUYVY709"]                                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatUYVY709_ER"]                             = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatUYVY2020"]                               = {CUDA_128, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerBCCR"]                              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerRCCB"]                              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerCRBC"]                              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayerCBRC"]                              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer10CCCC"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12BCCR"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12RCCB"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12CRBC"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12CBRC"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatBayer12CCCC"]                            = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY"]                                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420SemiPlanar_2020"]                  = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420SemiPlanar_2020"]                  = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420Planar_2020"]                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420Planar_2020"]                      = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420SemiPlanar_709"]                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420SemiPlanar_709"]                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUV420Planar_709"]                       = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVU420Planar_709"]                       = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_709"]            = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_2020"]           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_422SemiPlanar_2020"]           = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_422SemiPlanar"]                = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_422SemiPlanar_709"]            = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY_ER"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY_709_ER"]                               = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10_ER"]                                 = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10_709_ER"]                             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12_ER"]                                 = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12_709_ER"]                             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYUVA"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatYVYU"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatVYUY"]                                   = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_ER"]             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_420SemiPlanar_709_ER"]         = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_444SemiPlanar_ER"]             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY10V10U10_444SemiPlanar_709_ER"]         = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_420SemiPlanar_ER"]             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_420SemiPlanar_709_ER"]         = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_444SemiPlanar_ER"]             = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaEglColorFormatY12V12U12_444SemiPlanar_709_ER"]         = {CUDA_114, CUDA_0,   CUDA_0  };
  m["cudaDevAttrVulkanCigSupported"]                            = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved141"]                                   = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostNumaMemoryPoolsSupported"]                  = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaGraphChildGraphNodeOwnership"]                         = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaGraphChildGraphOwnershipClone"]                        = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaGraphChildGraphOwnershipMove"]                         = {CUDA_129, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved96"]                                    = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaDevAttrHostMemoryPoolsSupported"]                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaDevAttrReserved145"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaDevAttrOnlyPartialHostNativeAtomicSupported"]          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemLocationTypeNone"]                                  = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaMemAllocationTypeManaged"]                             = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaDevP2PAttrOnlyPartialNativeAtomicSupported"]           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperation"]                                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationIntegerAdd"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationIntegerMin"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationIntegerMax"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationIntegerIncrement"]                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationIntegerDecrement"]                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationAnd"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationOr"]                                    = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationXOR"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationExchange"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationCAS"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationFloatAdd"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationFloatMin"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationFloatMax"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicOperationCapability"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilitySigned"]                               = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityUnsigned"]                             = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityReduction"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityScalar32"]                             = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityScalar64"]                             = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityScalar128"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaAtomicCapabilityVector32x4"]                           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLaunchAttributeNvlinkUtilCentricScheduling"]           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUDAlogLevel"]                                             = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUDAlogLevel_enum"]                                        = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogLevelError"]                                        = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogLevelWarning"]                                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsCallbackHandle"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUlogsCallbackEntry_st"]                                   = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaKernelNodeAttributeNvlinkUtilCentricScheduling"]       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogIterator"]                                          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaLogsCallback_t"]                                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaEmulationStrategy_t"]                                  = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaEmulationStrategy"]                                    = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUDA_EMULATION_STRATEGY_DEFAULT"]                          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUDA_EMULATION_STRATEGY_PERFORMANT"]                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUDA_EMULATION_STRATEGY_EAGER"]                            = {CUDA_130, CUDA_0,   CUDA_0  };
  m["cudaEmulationMantissaControl_t"]                           = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["cudaEmulationMantissaControl"]                             = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC"]                  = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_MANTISSA_CONTROL_FIXED"]                    = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["cudaEmulationSpecialValuesSupport_t"]                      = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["cudaEmulationSpecialValuesSupport"]                        = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT"]            = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE"]               = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY"]           = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN"]                = {CUDA_130, CUDA_0,   CUDA_0  }; // [#2143] CUDA 13.0.2
  m["cudaErrorStreamDetached"]                                  = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceDesc_t"]                                    = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionContext_st"]                                  = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaExecutionContext_t"]                                   = {CUDA_131, CUDA_0,   CUDA_0  };
  m["RESOURCE_ABI_BYTES"]                                       = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceGroup_flags"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceGroupDefault"]                            = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceGroupBackfill"]                           = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceSplitByCount_flags"]                      = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceSplitIgnoreSmCoscheduling"]               = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceSplitMaxPotentialClusterSize"]            = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceType"]                                      = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceTypeInvalid"]                               = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceTypeSm"]                                    = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceTypeWorkqueueConfig"]                       = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResourceTypeWorkqueue"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResource"]                                        = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevWorkqueueConfigScope"]                              = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevWorkqueueConfigScopeDeviceCtx"]                     = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevWorkqueueConfigScopeGreenCtxBalanced"]              = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevWorkqueueConfigResource"]                           = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevWorkqueueResource"]                                 = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceGroupParams_st"]                          = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevSmResourceGroupParams"]                             = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResource_st"]                                       = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaDevResource"]                                          = {CUDA_131, CUDA_0,   CUDA_0  };
  m["cudaErrorVersionTranslation"]                              = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaHostTaskSyncMode"]                                     = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaHostTaskBlocking"]                                     = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaHostTaskSpinWait"]                                     = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrAllocationType"]                            = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrExportHandleTypes"]                         = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrLocationId"]                                = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrLocationType"]                              = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrMaxPoolSize"]                               = {CUDA_132, CUDA_0,   CUDA_0  };
  m["cudaMemPoolAttrHwDecompressEnabled"]                       = {CUDA_132, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RUNTIME_TYPE_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIChangedVersions> m;

  m["cudaExternalSemaphoreSignalParams"]                        = {CUDA_130};
  m["cudaExternalSemaphoreWaitParams"]                          = {CUDA_130};
  m["cudaLaunchAttributeValue"]                                 = {CUDA_130};
  m["cudaMemcpyNodeParams"]                                     = {CUDA_131};
  m["cudaConditionalNodeParams"]                                = {CUDA_131};
  m["cudaKernelNodeParamsV2"]                                   = {CUDA_131};
  m["cudaMemsetParamsV2"]                                       = {CUDA_131};
  m["cudaHostNodeParamsV2"]                                     = {CUDA_132};
  m["cudaPointerAttributes"]                                    = {CUDA_132};

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_RUNTIME_TYPE_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIChangedVersions> m;

  m["hipLaunchAttributeValue"]                                  = {HIP_7010};

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_RUNTIME_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipHostRegisterDefault"]                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipArrayDefault"]                                          = {HIP_1070, HIP_0,    HIP_0   };
  m["hipFuncAttribute"]                                         = {HIP_3090, HIP_0,    HIP_0   };
  m["hipFuncAttributeMaxDynamicSharedMemorySize"]               = {HIP_3090, HIP_0,    HIP_0   };
  m["hipFuncAttributePreferredSharedMemoryCarveout"]            = {HIP_3090, HIP_0,    HIP_0   };
  m["hipFuncAttributeMax"]                                      = {HIP_3090, HIP_0,    HIP_0   };
  m["hipChannelFormatKind"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChannelFormatKindSigned"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChannelFormatKindUnsigned"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChannelFormatKindFloat"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChannelFormatKindNone"]                                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipChannelFormatDesc"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipArray_const_t"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipMipmappedArray_const_t"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipResourceType"]                                          = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceTypeArray"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceTypeMipmappedArray"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceTypeLinear"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceTypePitch2D"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceViewFormat"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatNone"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedChar1"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedChar2"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedChar4"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedChar1"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedChar2"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedChar4"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedShort1"]                           = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedShort2"]                           = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedShort4"]                           = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedShort1"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedShort2"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedShort4"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedInt1"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedInt2"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedInt4"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedInt1"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedInt2"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedInt4"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatHalf1"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatHalf2"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatHalf4"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatFloat1"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatFloat2"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatFloat4"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed1"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed2"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed3"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed4"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedBlockCompressed4"]                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed5"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedBlockCompressed5"]                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed6H"]                = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatSignedBlockCompressed6H"]                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResViewFormatUnsignedBlockCompressed7"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceDesc"]                                          = {HIP_1070, HIP_0,    HIP_0   };
  m["hipResourceViewDesc"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMemcpyKind"]                                            = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpyHostToHost"]                                      = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpyHostToDevice"]                                    = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpyDeviceToHost"]                                    = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpyDeviceToDevice"]                                  = {HIP_1050, HIP_0,    HIP_0   };
  m["hipMemcpyDefault"]                                         = {HIP_1050, HIP_0,    HIP_0   };
  m["hipPitchedPtr"]                                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipExtent"]                                                = {HIP_1070, HIP_0,    HIP_0   };
  m["hipPos"]                                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipMemcpy3DParms"]                                         = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureAddressMode"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipAddressModeWrap"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipAddressModeClamp"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipAddressModeMirror"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipAddressModeBorder"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipSurfaceBoundaryMode"]                                   = {HIP_1090, HIP_0,    HIP_0   };
  m["hipBoundaryModeZero"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["hipBoundaryModeTrap"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["hipBoundaryModeClamp"]                                     = {HIP_1090, HIP_0,    HIP_0   };
  m["hipSurfaceObject_t"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["surfaceReference"]                                         = {HIP_1090, HIP_0,    HIP_0   };
  m["hipTextureType1D"]                                         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipTextureType2D"]                                         = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureType3D"]                                         = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureTypeCubemap"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureType1DLayered"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureType2DLayered"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureTypeCubemapLayered"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureFilterMode"]                                     = {HIP_1060, HIP_0,    HIP_0   };
  m["hipFilterModePoint"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipFilterModeLinear"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureReadMode"]                                       = {HIP_1060, HIP_0,    HIP_0   };
  m["hipReadModeElementType"]                                   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipReadModeNormalizedFloat"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipTextureDesc"]                                           = {HIP_1070, HIP_0,    HIP_0   };
  m["hipPointerAttribute_t"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipLaunchParams"]                                          = {HIP_2060, HIP_0,    HIP_0   };
  m["hipStreamCallback_t"]                                      = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidConfiguration"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidSymbol"]                                    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidDevicePointer"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidMemcpyDirection"]                           = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInsufficientDriver"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipErrorMissingConfiguration"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorPriorLaunchFailure"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidDeviceFunction"]                            = {HIP_1060, HIP_0,    HIP_0   };
  m["hipErrorInvalidPitchValue"]                                = {HIP_4020, HIP_0,    HIP_0   };
  m["hipExternalMemoryHandleDesc"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipExternalMemoryBufferDesc"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipExternalSemaphoreHandleDesc"]                           = {HIP_4040, HIP_0,    HIP_0   };
  m["hipExternalSemaphoreSignalParams"]                         = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphNodeType"]                                         = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeKernel"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeMemcpy"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeMemset"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeHost"]                                     = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeGraph"]                                    = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeEmpty"]                                    = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeWaitEvent"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeEventRecord"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeCount"]                                    = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNode"]                                             = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphNode_t"]                                           = {HIP_4030, HIP_0,    HIP_0   };
  m["hipHostFn_t"]                                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipMemsetParams"]                                          = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateResult"]                                 = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateSuccess"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateError"]                                  = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorTopologyChanged"]                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorNodeTypeChanged"]                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorFunctionChanged"]                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorParametersChanged"]                 = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorNotSupported"]                      = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExecUpdateErrorUnsupportedFunctionChange"]         = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureMode"]                                     = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureModeGlobal"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureModeThreadLocal"]                          = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureModeRelaxed"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureStatus"]                                   = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureStatusNone"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureStatusActive"]                             = {HIP_4030, HIP_0,    HIP_0   };
  m["hipStreamCaptureStatusInvalidated"]                        = {HIP_4030, HIP_0,    HIP_0   };
  m["ihipGraph"]                                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraph_t"]                                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExec"]                                             = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphExec_t"]                                           = {HIP_4030, HIP_0,    HIP_0   };
  m["hipGraphicsResource"]                                      = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsResource_t"]                                    = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGLDeviceList"]                                          = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGLDeviceListAll"]                                       = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGLDeviceListCurrentFrame"]                              = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGLDeviceListNextFrame"]                                 = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlags"]                                 = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlagsNone"]                             = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlagsReadOnly"]                         = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlagsWriteDiscard"]                     = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlagsSurfaceLoadStore"]                 = {HIP_4040, HIP_0,    HIP_0   };
  m["hipGraphicsRegisterFlagsTextureGather"]                    = {HIP_4040, HIP_0,    HIP_0   };
  m["hipErrorIllegalState"]                                     = {HIP_5000, HIP_0,    HIP_0   };
  m["hipErrorGraphExecUpdateFailure"]                           = {HIP_5000, HIP_0,    HIP_0   };
  m["hipDeviceAttributeMultiGpuBoardGroupID"]                   = {HIP_5000, HIP_0,    HIP_0   };
  m["hipUUID"]                                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipUUID_t"]                                                = {HIP_5020, HIP_0,    HIP_0   };
  m["hipKernelNodeAttrID"]                                      = {HIP_5020, HIP_0,    HIP_0   };
  m["hipKernelNodeAttributeAccessPolicyWindow"]                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipKernelNodeAttributeCooperative"]                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipAccessProperty"]                                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipAccessPropertyNormal"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipAccessPropertyStreaming"]                               = {HIP_5020, HIP_0,    HIP_0   };
  m["hipAccessPropertyPersisting"]                              = {HIP_5020, HIP_0,    HIP_0   };
  m["hipAccessPolicyWindow"]                                    = {HIP_5020, HIP_0,    HIP_0   };
  m["hipKernelNodeAttrValue"]                                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipDeviceAttributeMemoryPoolsSupported"]                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPool_t"]                                             = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttr"]                                           = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolReuseFollowEventDependencies"]                   = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolReuseAllowOpportunistic"]                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolReuseAllowInternalDependencies"]                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttrReleaseThreshold"]                           = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttrReservedMemCurrent"]                         = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttrReservedMemHigh"]                            = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttrUsedMemCurrent"]                             = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolAttrUsedMemHigh"]                                = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemLocationType"]                                       = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemLocationTypeInvalid"]                                = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemLocationTypeDevice"]                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemLocation"]                                           = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAccessFlags"]                                        = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAccessFlagsProtNone"]                                = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAccessFlagsProtRead"]                                = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAccessFlagsProtReadWrite"]                           = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAccessDesc"]                                         = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAllocationType"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAllocationTypeInvalid"]                              = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAllocationTypePinned"]                               = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAllocationTypeMax"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemAllocationHandleType"]                               = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemHandleTypeNone"]                                     = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemHandleTypePosixFileDescriptor"]                      = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemHandleTypeWin32"]                                    = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemHandleTypeWin32Kmt"]                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolProps"]                                          = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemPoolPtrExportData"]                                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateFlags"]                                 = {HIP_5020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateFlagAutoFreeOnLaunch"]                  = {HIP_5020, HIP_0,    HIP_0   };
  m["hipMemoryTypeManaged"]                                     = {HIP_5030, HIP_0,    HIP_0   };
  m["hipLimitStackSize"]                                        = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeExtSemaphoreSignal"]                       = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeExtSemaphoreWait"]                         = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphMemAttributeType"]                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphMemAttrUsedMemCurrent"]                            = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphMemAttrUsedMemHigh"]                               = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphMemAttrReservedMemCurrent"]                        = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphMemAttrReservedMemHigh"]                           = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectFlags"]                                       = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectNoDestructorSync"]                            = {HIP_5030, HIP_0,    HIP_0   };
  m["hipUserObjectRetainFlags"]                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["hipGraphUserObjectMove"]                                   = {HIP_5030, HIP_0,    HIP_0   };
  m["hipOccupancyDisableCachingOverride"]                       = {HIP_5050, HIP_0,    HIP_0   };
  m["hipExternalMemoryDedicated"]                               = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeMemAlloc"]                                 = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphNodeTypeMemFree"]                                  = {HIP_5050, HIP_0,    HIP_0   };
  m["hipMemAllocNodeParams"]                                    = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlags"]                                    = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsVerbose"]                             = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsKernelNodeParams"]                    = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsMemcpyNodeParams"]                    = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsMemsetNodeParams"]                    = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsHostNodeParams"]                      = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsEventNodeParams"]                     = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsExtSemasSignalNodeParams"]            = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsExtSemasWaitNodeParams"]              = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsKernelNodeAttributes"]                = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphDebugDotFlagsHandles"]                             = {HIP_5050, HIP_0,    HIP_0   };
  m["hipGraphInstantiateFlagUpload"]                            = {HIP_5060, HIP_0,    HIP_0   };
  m["hipGraphInstantiateFlagDeviceLaunch"]                      = {HIP_5060, HIP_0,    HIP_0   };
  m["hipGraphInstantiateFlagUseNodePriority"]                   = {HIP_5060, HIP_0,    HIP_0   };
  m["hipHostRegisterReadOnly"]                                  = {HIP_5060, HIP_0,    HIP_0   };
  m["hipFlushGPUDirectRDMAWritesOptions"]                       = {HIP_6010, HIP_0,    HIP_0   };
  m["hipFlushGPUDirectRDMAWritesOptionHost"]                    = {HIP_6010, HIP_0,    HIP_0   };
  m["hipFlushGPUDirectRDMAWritesOptionMemOps"]                  = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGPUDirectRDMAWritesOrdering"]                           = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGPUDirectRDMAWritesOrderingNone"]                       = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGPUDirectRDMAWritesOrderingOwner"]                      = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGPUDirectRDMAWritesOrderingAllDevices"]                 = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGraphInstantiateResult"]                                = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateSuccess"]                               = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateError"]                                 = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateInvalidStructure"]                      = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateNodeOperationNotSupported"]             = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateMultipleDevicesNotSupported"]           = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphInstantiateParams"]                                = {HIP_6020, HIP_0,    HIP_0   };
  m["hipMemcpyNodeParams"]                                      = {HIP_6010, HIP_0,    HIP_0   };
  m["hipChildGraphNodeParams"]                                  = {HIP_6010, HIP_0,    HIP_0   };
  m["hipEventWaitNodeParams"]                                   = {HIP_6010, HIP_0,    HIP_0   };
  m["hipEventRecordNodeParams"]                                 = {HIP_6010, HIP_0,    HIP_0   };
  m["hipMemFreeNodeParams"]                                     = {HIP_6010, HIP_0,    HIP_0   };
  m["hipGraphNodeParams"]                                       = {HIP_6010, HIP_0,    HIP_0   };
  m["hipLaunchAttributeID"]                                     = {HIP_6020, HIP_0,    HIP_0   };
  m["hipLaunchAttributeAccessPolicyWindow"]                     = {HIP_6020, HIP_0,    HIP_0   };
  m["hipLaunchAttributeCooperative"]                            = {HIP_6020, HIP_0,    HIP_0   };
  m["hipLaunchAttributePriority"]                               = {HIP_6020, HIP_0,    HIP_0   };
  m["hipLaunchAttributeValue"]                                  = {HIP_6020, HIP_0,    HIP_0   };
  m["hipKernelNodeAttributePriority"]                           = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphKernelNodePortDefault"]                            = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphKernelNodePortLaunchCompletion"]                   = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphKernelNodePortProgrammatic"]                       = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphDependencyType"]                                   = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphDependencyTypeDefault"]                            = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphDependencyTypeProgrammatic"]                       = {HIP_6020, HIP_0,    HIP_0   };
  m["hipGraphEdgeData"]                                         = {HIP_6020, HIP_0,    HIP_0   };
  m["HIP_INF_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_NAN_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_MIN_DENORM_F"]                                         = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_MAX_NORMAL_F"]                                         = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_NEG_ZERO_F"]                                           = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_ZERO_F"]                                               = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_ONE_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF_HI_F"]                                       = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF_LO_F"]                                       = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_SQRT_TWO_F"]                                           = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_THIRD_F"]                                              = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_PIO4_F"]                                               = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_PIO2_F"]                                               = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_3PIO4_F"]                                              = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_2_OVER_PI_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_SQRT_2_OVER_PI_F"]                                     = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_PI_F"]                                                 = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_L2E_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_L2T_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_LG2_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_LGE_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_LN2_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_LNT_F"]                                                = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_LNPI_F"]                                               = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_M126_F"]                                        = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_126_F"]                                         = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_NORM_HUGE_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_23_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_24_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_31_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_32_F"]                                          = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_REMQUO_BITS_F"]                                        = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_REMQUO_MASK_F"]                                        = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_TRIG_PLOSS_F"]                                         = {HIP_5030, HIP_0,    HIP_0   };
  m["HIP_INF"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_NAN"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_NEG_ZERO"]                                             = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_MIN_DENORM"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_ZERO"]                                                 = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_ONE"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_TWO"]                                             = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF_HI"]                                         = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_HALF_LO"]                                         = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_THIRD"]                                                = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWOTHIRD"]                                             = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO4"]                                                 = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO4_HI"]                                              = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO4_LO"]                                              = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO2"]                                                 = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO2_HI"]                                              = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PIO2_LO"]                                              = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_3PIO4"]                                                = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_2_OVER_PI"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PI"]                                                   = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PI_HI"]                                                = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_PI_LO"]                                                = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_2PI"]                                             = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_2PI_HI"]                                          = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_2PI_LO"]                                          = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_PIO2"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_PIO2_HI"]                                         = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_PIO2_LO"]                                         = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_SQRT_2OPI"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_L2E"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_L2E_HI"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_L2E_LO"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_L2T"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LG2"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LG2_HI"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LG2_LO"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LGE"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LGE_HI"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LGE_LO"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2_HI"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2_LO"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LNT"]                                                  = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LNT_HI"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LNT_LO"]                                               = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LNPI"]                                                 = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2_X_1024"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2_X_1025"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LN2_X_1075"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LG2_X_1024"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_LG2_X_1075"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_23"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_52"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_53"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_54"]                                            = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_M54"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TWO_TO_M1022"]                                         = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_TRIG_PLOSS"]                                           = {HIP_5070, HIP_0,    HIP_0   };
  m["HIP_DBL2INT_CVT"]                                          = {HIP_5070, HIP_0,    HIP_0   };
  m["hipMemoryTypeUnregistered"]                                = {HIP_6000, HIP_0,    HIP_0   };
  m["hipErrorInvalidChannelDescriptor"]                         = {HIP_6040, HIP_0,    HIP_0   };
  m["hipErrorInvalidTexture"]                                   = {HIP_6040, HIP_0,    HIP_0   };
  m["hipEventRecordDefault"]                                    = {HIP_6040, HIP_0,    HIP_0   };
  m["hipEventRecordExternal"]                                   = {HIP_6040, HIP_0,    HIP_0   };
  m["hipLaunchAttribute_st"]                                    = {HIP_7000, HIP_0,    HIP_0   };
  m["hipLaunchAttribute"]                                       = {HIP_7000, HIP_0,    HIP_0   };
  m["hipLaunchConfig_st"]                                       = {HIP_7000, HIP_0,    HIP_0   };
  m["hipLaunchConfig_t"]                                        = {HIP_7000, HIP_0,    HIP_0   };
  m["hipMemLocationTypeNone"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemLocationTypeHost"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemLocationTypeHostNuma"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemLocationTypeHostNumaCurrent"]                        = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyFlags"]                                           = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyFlagDefault"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyFlagPreferOverlapWithCompute"]                    = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrder"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrderInvalid"]                           = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrderStream"]                            = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrderDuringApiCall"]                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrderAny"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpySrcAccessOrderMax"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyAttributes"]                                      = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DOperandType"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyOperandTypePointer"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyOperandTypeArray"]                                = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpyOperandTypeMax"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["hipOffset3D"]                                              = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DOperand"]                                       = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DBatchOp"]                                       = {HIP_7010, HIP_0,    HIP_0   };
  m["hipMemcpy3DPeerParms"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["hipDriverEntryPointQueryResult"]                           = {HIP_7010, HIP_0,    HIP_0   };
  m["hipDriverEntryPointSuccess"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["hipDriverEntryPointSymbolNotFound"]                        = {HIP_7010, HIP_0,    HIP_0   };
  m["hipDriverEntryPointVersionNotSufficent"]                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipEnableDefault"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["hipEnableLegacyStream"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["hipEnablePerThreadDefaultStream"]                          = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchMemSyncDomainMap"]                                = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchMemSyncDomain"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchMemSyncDomainDefault"]                            = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchMemSyncDomainRemote"]                             = {HIP_7010, HIP_0,    HIP_0   };
  m["hipSynchronizationPolicy"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["hipSyncPolicyAuto"]                                        = {HIP_7010, HIP_0,    HIP_0   };
  m["hipSyncPolicySpin"]                                        = {HIP_7010, HIP_0,    HIP_0   };
  m["hipSyncPolicyYield"]                                       = {HIP_7010, HIP_0,    HIP_0   };
  m["hipSyncPolicyBlockingSync"]                                = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchAttributeSynchronizationPolicy"]                  = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchAttributeMemSyncDomainMap"]                       = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLaunchAttributeMemSyncDomain"]                          = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryOption"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryHostUniversalFunctionAndDataTable"]              = {HIP_7010, HIP_0,    HIP_0   };
  m["hipLibraryBinaryIsPreserved"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["hipDeviceAttributeHostNumaId"]                             = {HIP_7020, HIP_0,    HIP_0   };

  return m;
}();
