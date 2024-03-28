// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main() {
  printf("01. CUDA Driver API Structs synthetic test\n");

  // CHECK: HIP_ARRAY3D_DESCRIPTOR array3D_st;
  // CHECK-NEXT: HIP_ARRAY3D_DESCRIPTOR array3D;
  CUDA_ARRAY3D_DESCRIPTOR_st array3D_st;
  CUDA_ARRAY3D_DESCRIPTOR array3D;

  // CHECK: HIP_ARRAY_DESCRIPTOR array_descr_st;
  // CHECK-NEXT: HIP_ARRAY_DESCRIPTOR array_descr;
  CUDA_ARRAY_DESCRIPTOR_st array_descr_st;
  CUDA_ARRAY_DESCRIPTOR array_descr;
#define __CUDA_API_VERSION_INTERNAL
  // CHECK: HIP_ARRAY_DESCRIPTOR array_descr_v1_st;
  // CHECK-NEXT: HIP_ARRAY_DESCRIPTOR array_descr_v1;
  CUDA_ARRAY_DESCRIPTOR_v1_st array_descr_v1_st;
  CUDA_ARRAY_DESCRIPTOR_v1 array_descr_v1;
#undef __CUDA_API_VERSION_INTERNAL

  // CHECK: hip_Memcpy2D memcpy2D_st;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D_v1_st;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D_v1;
  CUDA_MEMCPY2D_st memcpy2D_st;
  CUDA_MEMCPY2D_v1_st memcpy2D_v1_st;
  CUDA_MEMCPY2D memcpy2D;
  CUDA_MEMCPY2D_v1 memcpy2D_v1;

  // CHECK: HIP_MEMCPY3D memcpy3D_st;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D_v1_st;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D_v1;
  CUDA_MEMCPY3D_st memcpy3D_st;
  CUDA_MEMCPY3D_v1_st memcpy3D_v1_st;
  CUDA_MEMCPY3D memcpy3D;
  CUDA_MEMCPY3D_v1 memcpy3D_v1;

  // CHECK: HIP_RESOURCE_DESC_st res_descr_st;
  // CHECK-NEXT: HIP_RESOURCE_DESC res_descr;
  CUDA_RESOURCE_DESC_st res_descr_st;
  CUDA_RESOURCE_DESC res_descr;

  // CHECK: HIP_RESOURCE_VIEW_DESC_st res_view_descr_st;
  // CHECK-NEXT: HIP_RESOURCE_VIEW_DESC res_view_descr;
  CUDA_RESOURCE_VIEW_DESC_st res_view_descr_st;
  CUDA_RESOURCE_VIEW_DESC res_view_descr;

  // CHECK: HIP_TEXTURE_DESC_st tex_descr_st;
  // CHECK-NEXT: HIP_TEXTURE_DESC tex_descr;
  CUDA_TEXTURE_DESC_st tex_descr_st;
  CUDA_TEXTURE_DESC tex_descr;

  // CHECK: hipIpcMemHandle_st ipc_mem_handle_st;
  // CHECK-NEXT: hipIpcMemHandle_t ipc_mem_handle;
  CUipcMemHandle_st ipc_mem_handle_st;
  CUipcMemHandle ipc_mem_handle;

  // CHECK: hipArray* array_st_ptr;
  // CHECK-NEXT: hipArray_t array_ptr;
  CUarray_st* array_st_ptr;
  CUarray array_ptr;

  // CHECK: ihipCtx_t* ctx_st_ptr;
  // CHECK-NEXT: hipCtx_t ctx;
  CUctx_st* ctx_st_ptr;
  CUcontext ctx;

  // CHECK: ihipEvent_t* evnt_st_ptr;
  // CHECK-NEXT: hipEvent_t evnt;
  CUevent_st* evnt_st_ptr;
  CUevent evnt;

  // CHECK: ihipModuleSymbol_t* func_st_ptr;
  // CHECK-NEXT: hipFunction_t func;
  CUfunc_st* func_st_ptr;
  CUfunction func;

  // CHECK: hipMipmappedArray* mipmapped_array_st_ptr;
  // CHECK-NEXT: hipMipmappedArray_t mipmapped_array;
  CUmipmappedArray_st* mipmapped_array_st_ptr;
  CUmipmappedArray mipmapped_array;

  // CHECK: ihipStream_t* stream_st_ptr;
  // CHECK-NEXT: hipStream_t stream;
  CUstream_st* stream_st_ptr;
  CUstream stream;

  // CHECK: textureReference* tex_ref_st_ptr;
  // CHECK-NEXT: hipTexRef tex_ref;
  CUtexref_st* tex_ref_st_ptr;
  CUtexref tex_ref;

  // CHECK: hipGraphicsResource* graphicsResource_st;
  // CHECK-NEXT: hipGraphicsResource_t graphicsResource;
  CUgraphicsResource_st* graphicsResource_st;
  CUgraphicsResource graphicsResource;

  // CHECK: hipUUID_t uuid_st;
  CUuuid_st uuid_st;

  // CHECK: ihiprtcLinkState* linkState_ptr;
  // CHECK-NEXT: hiprtcLinkState linkState;
  CUlinkState_st* linkState_ptr;
  CUlinkState linkState;

#if CUDA_VERSION >= 9000
  // CHECK: hipFunctionLaunchParams_t LAUNCH_PARAMS_st;
  // CHECK-NEXT: hipFunctionLaunchParams LAUNCH_PARAMS;
  CUDA_LAUNCH_PARAMS_st LAUNCH_PARAMS_st;
  CUDA_LAUNCH_PARAMS LAUNCH_PARAMS;
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipExternalMemoryBufferDesc_st ext_mem_buff_st;
  // CHECK-NEXT: hipExternalMemoryBufferDesc ext_mem_buff;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st ext_mem_buff_st;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC ext_mem_buff;

  // CHECK: hipExternalMemoryHandleDesc_st ext_mem_handle_st;
  // CHECK-NEXT: hipExternalMemoryHandleDesc ext_mem_handle;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st ext_mem_handle_st;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC ext_mem_handle;

  // CHECK: hipExternalSemaphoreHandleDesc_st ext_sema_handle_st;
  // CHECK-NEXT: hipExternalSemaphoreHandleDesc ext_sema_handle;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st ext_sema_handle_st;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC ext_sema_handle;

  // CHECK: hipExternalSemaphoreSignalParams_st ext_sema_params_st;
  // CHECK-NEXT: hipExternalSemaphoreSignalParams ext_sema_params;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st ext_sema_params_st;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_sema_params;

  // CHECK: hipHostNodeParams host_node_params_st;
  // CHECK-NEXT: hipHostNodeParams host_node_params;
  CUDA_HOST_NODE_PARAMS_st host_node_params_st;
  CUDA_HOST_NODE_PARAMS host_node_params;

  // CHECK: hipKernelNodeParams kern_node_params_st;
  // CHECK-NEXT: hipKernelNodeParams kern_node_params;
  CUDA_KERNEL_NODE_PARAMS_st kern_node_params_st;
  CUDA_KERNEL_NODE_PARAMS kern_node_params;

  // CHECK: hipExternalMemory_t ext_mem;
  CUexternalMemory ext_mem;

  // CHECK: hipExternalSemaphore_t ext_sema;
  CUexternalSemaphore ext_sema;

  // CHECK: ihipGraph* graph_st;
  // CHECK-NEXT: hipGraph_t graph;
  CUgraph_st* graph_st;
  CUgraph graph;

  // CHECK: hipGraphExec* graphExec_st;
  // CHECK-NEXT: hipGraphExec_t graphExec;
  CUgraphExec_st* graphExec_st;
  CUgraphExec graphExec;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipMemAccessDesc memAccessDesc_st;
  // CHECK-NEXT: hipMemAccessDesc memAccessDesc;
  CUmemAccessDesc_st memAccessDesc_st;
  CUmemAccessDesc memAccessDesc;

  // CHECK: hipMemAllocationProp memAllocationProp_st;
  // CHECK-NEXT: hipMemAllocationProp memAllocationProp;
  CUmemAllocationProp_st memAllocationProp_st;
  CUmemAllocationProp memAllocationProp;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipAccessPolicyWindow accessPolicyWindow_st;
  // CHECK-NEXT: hipAccessPolicyWindow accessPolicyWindow;
  CUaccessPolicyWindow_st accessPolicyWindow_st;
  CUaccessPolicyWindow accessPolicyWindow;
#endif

#if CUDA_VERSION >= 11010
  // CHECK: hipArrayMapInfo arrayMapInfo_st;
  // CHECK-NEXT: hipArrayMapInfo arrayMapInfo;
  CUarrayMapInfo_st arrayMapInfo_st;
  CUarrayMapInfo arrayMapInfo;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: ihipMemPoolHandle_t* memPoolHandle_st;
  // CHECK-NEXT: hipMemPool_t memPool_t;
  CUmemPoolHandle_st* memPoolHandle_st;
  CUmemoryPool memPool_t;

  // CHECK: hipMemLocation memLocation_st;
  // CHECK-NEXT: hipMemLocation memLocation;
  CUmemLocation_st memLocation_st;
  CUmemLocation memLocation;

  // CHECK: hipMemPoolProps memPoolProps_st;
  // CHECK-NEXT: hipMemPoolProps memPoolProps;
  CUmemPoolProps_st memPoolProps_st;
  CUmemPoolProps memPoolProps;

  // CHECK: hipMemPoolPtrExportData memPoolPtrExportData_st;
  // CHECK-NEXT: hipMemPoolPtrExportData memPoolPtrExportData;
  CUmemPoolPtrExportData_st memPoolPtrExportData_st;
  CUmemPoolPtrExportData memPoolPtrExportData;

  // CHECK: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS_st;
  // CHECK-NEXT: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st EXT_SEM_SIGNAL_NODE_PARAMS_st;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS EXT_SEM_SIGNAL_NODE_PARAMS;

  // CHECK: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS_st;
  // CHECK-NEXT: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_st EXT_SEM_WAIT_NODE_PARAMS_st;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS EXT_SEM_WAIT_NODE_PARAMS;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: HIP_ARRAY3D_DESCRIPTOR array3D_v2;
  CUDA_ARRAY3D_DESCRIPTOR_v2 array3D_v2;

  // CHECK: HIP_ARRAY_DESCRIPTOR array_descr_v2;
  CUDA_ARRAY_DESCRIPTOR_v2 array_descr_v2;

  // CHECK: hipExternalMemoryBufferDesc ext_mem_buff_v1;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 ext_mem_buff_v1;

  // CHECK: hipExternalMemoryHandleDesc ext_mem_handle_v1;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 ext_mem_handle_v1;

  // CHECK: hipExternalSemaphoreHandleDesc ext_sema_handle_v1;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 ext_sema_handle_v1;

  // CHECK: hipExternalSemaphoreSignalParams ext_sema_params_v1;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 ext_sema_params_v1;

  // CHECK: hipHostNodeParams host_node_params_v1;
  CUDA_HOST_NODE_PARAMS_v1 host_node_params_v1;

  // CHECK: hipKernelNodeParams kern_node_params_v1;
  CUDA_KERNEL_NODE_PARAMS_v1 kern_node_params_v1;

  // CHECK: hip_Memcpy2D memcpy2D_v2;
  CUDA_MEMCPY2D_v2 memcpy2D_v2;

  // CHECK: HIP_MEMCPY3D memcpy3D_v2;
  CUDA_MEMCPY3D_v2 memcpy3D_v2;

  // CHECK: HIP_RESOURCE_DESC res_descr_v1;
  CUDA_RESOURCE_DESC_v1 res_descr_v1;

  // CHECK: HIP_RESOURCE_VIEW_DESC res_view_descr_v1;
  CUDA_RESOURCE_VIEW_DESC_v1 res_view_descr_v1;

  // CHECK: HIP_TEXTURE_DESC tex_descr_v1;
  CUDA_TEXTURE_DESC_v1 tex_descr_v1;

  // CHECK: hipIpcMemHandle_t ipc_mem_handle_v1;
  CUipcMemHandle_v1 ipc_mem_handle_v1;

  // CHECK: hipMemLocation memLocation_v1;
  CUmemLocation_v1 memLocation_v1;

  // CHECK: hipUserObject* userObject_st_ptr;
  // CHECK-NEXT: hipUserObject_t userObject;
  CUuserObject_st* userObject_st_ptr;
  CUuserObject userObject;

  // CHECK: hipMemAccessDesc memAccessDesc_v1;
  CUmemAccessDesc_v1 memAccessDesc_v1;

  // CHECK: hipMemPoolProps memPoolProps_v1;
  CUmemPoolProps_v1 memPoolProps_v1;

  // CHECK: hipMemPoolPtrExportData memPoolPtrExportData_v1;
  CUmemPoolPtrExportData_v1 memPoolPtrExportData_v1;

  // CHECK: hipMemAllocationProp memAllocationProp_v1;
  CUmemAllocationProp_v1 memAllocationProp_v1;

  // CHECK: hipArrayMapInfo arrayMapInfo_v1;
  CUarrayMapInfo_v1 arrayMapInfo_v1;

  // CHECK: hipFunctionLaunchParams LAUNCH_PARAMS_v1;
  CUDA_LAUNCH_PARAMS_v1 LAUNCH_PARAMS_v1;

  // CHECK: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS_v1;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 EXT_SEM_SIGNAL_NODE_PARAMS_v1;

  // CHECK: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS_v1;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 EXT_SEM_WAIT_NODE_PARAMS_v1;
#endif

#if CUDA_VERSION >= 11040
  // CHECK: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS;
  CUDA_MEM_ALLOC_NODE_PARAMS MEM_ALLOC_NODE_PARAMS;
#endif

#if CUDA_VERSION >= 11040 && CUDA_VERSION < 12020
  // CHECK: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS_st;
  CUDA_MEM_ALLOC_NODE_PARAMS_st MEM_ALLOC_NODE_PARAMS_st;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: hipGraphInstantiateParams GRAPH_INSTANTIATE_PARAMS_st;
  // CHECK-NEXT: hipGraphInstantiateParams GRAPH_INSTANTIATE_PARAMS;
  CUDA_GRAPH_INSTANTIATE_PARAMS_st GRAPH_INSTANTIATE_PARAMS_st;
  CUDA_GRAPH_INSTANTIATE_PARAMS GRAPH_INSTANTIATE_PARAMS;
#endif

#if CUDA_VERSION >= 12020
  // CHECK: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS_v1_st;
  // CHECK-NEXT: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS_v1;
  CUDA_MEM_ALLOC_NODE_PARAMS_v1_st MEM_ALLOC_NODE_PARAMS_v1_st;
  CUDA_MEM_ALLOC_NODE_PARAMS_v1 MEM_ALLOC_NODE_PARAMS_v1;

  // CHECK: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
  // CHECK-NEXT: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS_v2;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 EXT_SEM_SIGNAL_NODE_PARAMS_v2;

  // CHECK: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS_v2_st;
  // CHECK-NEXT: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS_v2;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st EXT_SEM_WAIT_NODE_PARAMS_v2_st;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 EXT_SEM_WAIT_NODE_PARAMS_v2;

  // CHECK: hipMemcpyNodeParams MEMCPY_NODE_PARAMS_st;
  // CHECK-NEXT: hipMemcpyNodeParams MEMCPY_NODE_PARAMS;
  CUDA_MEMCPY_NODE_PARAMS_st MEMCPY_NODE_PARAMS_st;
  CUDA_MEMCPY_NODE_PARAMS MEMCPY_NODE_PARAMS;

  // CHECK: hipChildGraphNodeParams CHILD_GRAPH_NODE_PARAMS_st;
  // CHECK-NEXT: hipChildGraphNodeParams CHILD_GRAPH_NODE_PARAMS;
  CUDA_CHILD_GRAPH_NODE_PARAMS_st CHILD_GRAPH_NODE_PARAMS_st;
  CUDA_CHILD_GRAPH_NODE_PARAMS CHILD_GRAPH_NODE_PARAMS;

  // CHECK: hipMemFreeNodeParams MEM_FREE_NODE_PARAMS_st;
  // CHECK-NEXT: hipMemFreeNodeParams MEM_FREE_NODE_PARAMS;
  CUDA_MEM_FREE_NODE_PARAMS_st MEM_FREE_NODE_PARAMS_st;
  CUDA_MEM_FREE_NODE_PARAMS MEM_FREE_NODE_PARAMS;

  // CHECK: hipEventRecordNodeParams EVENT_RECORD_NODE_PARAMS_st;
  // CHECK-NEXT: hipEventRecordNodeParams EVENT_RECORD_NODE_PARAMS;
  CUDA_EVENT_RECORD_NODE_PARAMS_st EVENT_RECORD_NODE_PARAMS_st;
  CUDA_EVENT_RECORD_NODE_PARAMS EVENT_RECORD_NODE_PARAMS;

  // CHECK: hipEventWaitNodeParams EVENT_WAIT_NODE_PARAMS_st;
  // CHECK-NEXT: hipEventWaitNodeParams EVENT_WAIT_NODE_PARAMS;
  CUDA_EVENT_WAIT_NODE_PARAMS_st EVENT_WAIT_NODE_PARAMS_st;
  CUDA_EVENT_WAIT_NODE_PARAMS EVENT_WAIT_NODE_PARAMS;

  // CHECK: hipGraphNodeParams graphNodeParams_st;
  // CHECK-NEXT: hipGraphNodeParams graphNodeParams;
  CUgraphNodeParams_st graphNodeParams_st;
  CUgraphNodeParams graphNodeParams;
#endif

  return 0;
}
