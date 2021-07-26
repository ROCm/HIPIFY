// RUN: %run_test hipify "%s" "%t" %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

int main() {
  printf("01. CUDA Driver API Structs synthetic test\n");

  // CHECK: HIP_ARRAY3D_DESCRIPTOR array3D_st;
  // CHECK-NEXT: HIP_ARRAY3D_DESCRIPTOR array3D;
  // CHECK-NEXT: HIP_ARRAY3D_DESCRIPTOR array3D_v2;
  CUDA_ARRAY3D_DESCRIPTOR_st array3D_st;
  CUDA_ARRAY3D_DESCRIPTOR array3D;
  CUDA_ARRAY3D_DESCRIPTOR_v2 array3D_v2;

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
  // CHECK: HIP_ARRAY_DESCRIPTOR array_descr_v2;
  CUDA_ARRAY_DESCRIPTOR_v2 array_descr_v2;

  // CHECK: hipExternalMemoryBufferDesc_st ext_mem_buff_st;
  // CHECK-NEXT: hipExternalMemoryBufferDesc ext_mem_buff;
  // CHECK-NEXT: hipExternalMemoryBufferDesc ext_mem_buff_v1;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st ext_mem_buff_st;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC ext_mem_buff;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 ext_mem_buff_v1;

  // CHECK: hipExternalMemoryHandleDesc_st ext_mem_handle_st;
  // CHECK-NEXT: hipExternalMemoryHandleDesc ext_mem_handle;
  // CHECK-NEXT: hipExternalMemoryHandleDesc ext_mem_handle_v1;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st ext_mem_handle_st;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC ext_mem_handle;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 ext_mem_handle_v1;

  // CHECK: hipExternalSemaphoreHandleDesc_st ext_sema_handle_st;
  // CHECK-NEXT: hipExternalSemaphoreHandleDesc ext_sema_handle;
  // CHECK-NEXT: hipExternalSemaphoreHandleDesc ext_sema_handle_v1;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st ext_sema_handle_st;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC ext_sema_handle;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 ext_sema_handle_v1;

  // CHECK: hipExternalSemaphoreSignalParams_st ext_sema_params_st;
  // CHECK-NEXT: hipExternalSemaphoreSignalParams ext_sema_params;
  // CHECK-NEXT: hipExternalSemaphoreSignalParams ext_sema_params_v1;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st ext_sema_params_st;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_sema_params;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 ext_sema_params_v1;

  // CHECK: hipHostNodeParams host_node_params_st;
  // CHECK-NEXT: hipHostNodeParams host_node_params;
  // CHECK-NEXT: hipHostNodeParams host_node_params_v1;
  CUDA_HOST_NODE_PARAMS_st host_node_params_st;
  CUDA_HOST_NODE_PARAMS host_node_params;
  CUDA_HOST_NODE_PARAMS_v1 host_node_params_v1;

  // CHECK: hipKernelNodeParams kern_node_params_st;
  // CHECK-NEXT: hipKernelNodeParams kern_node_params;
  // CHECK-NEXT: hipKernelNodeParams kern_node_params_v1;
  CUDA_KERNEL_NODE_PARAMS_st kern_node_params_st;
  CUDA_KERNEL_NODE_PARAMS kern_node_params;
  CUDA_KERNEL_NODE_PARAMS_v1 kern_node_params_v1;

  // CHECK: hip_Memcpy2D memcpy2D_st;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D_v1_st;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D_v1;
  // CHECK-NEXT: hip_Memcpy2D memcpy2D_v2;
  CUDA_MEMCPY2D_st memcpy2D_st;
  CUDA_MEMCPY2D_v1_st memcpy2D_v1_st;
  CUDA_MEMCPY2D memcpy2D;
  CUDA_MEMCPY2D_v1 memcpy2D_v1;
  CUDA_MEMCPY2D_v2 memcpy2D_v2;

  // CHECK: HIP_MEMCPY3D memcpy3D_st;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D_v1_st;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D_v1;
  // CHECK-NEXT: HIP_MEMCPY3D memcpy3D_v2;
  CUDA_MEMCPY3D_st memcpy3D_st;
  CUDA_MEMCPY3D_v1_st memcpy3D_v1_st;
  CUDA_MEMCPY3D memcpy3D;
  CUDA_MEMCPY3D_v1 memcpy3D_v1;
  CUDA_MEMCPY3D_v2 memcpy3D_v2;

  // CHECK: HIP_RESOURCE_DESC_st res_descr_st;
  // CHECK-NEXT: HIP_RESOURCE_DESC res_descr;
  // CHECK-NEXT: HIP_RESOURCE_DESC res_descr_v1;
  CUDA_RESOURCE_DESC_st res_descr_st;
  CUDA_RESOURCE_DESC res_descr;
  CUDA_RESOURCE_DESC_v1 res_descr_v1;

  // CHECK: HIP_RESOURCE_VIEW_DESC_st res_view_descr_st;
  // CHECK-NEXT: HIP_RESOURCE_VIEW_DESC res_view_descr;
  // CHECK-NEXT: HIP_RESOURCE_VIEW_DESC res_view_descr_v1;
  CUDA_RESOURCE_VIEW_DESC_st res_view_descr_st;
  CUDA_RESOURCE_VIEW_DESC res_view_descr;
  CUDA_RESOURCE_VIEW_DESC_v1 res_view_descr_v1;

  // CHECK: HIP_TEXTURE_DESC_st tex_descr_st;
  // CHECK-NEXT: HIP_TEXTURE_DESC tex_descr;
  // CHECK-NEXT: HIP_TEXTURE_DESC tex_descr_v1;
  CUDA_TEXTURE_DESC_st tex_descr_st;
  CUDA_TEXTURE_DESC tex_descr;
  CUDA_TEXTURE_DESC_v1 tex_descr_v1;

  // CHECK: hipIpcMemHandle_st ipc_mem_handle_st;
  // CHECK-NEXT: hipIpcMemHandle_t ipc_mem_handle;
  // CHECK-NEXT: hipIpcMemHandle_t ipc_mem_handle_v1;
  CUipcMemHandle_st ipc_mem_handle_st;
  CUipcMemHandle ipc_mem_handle;
  CUipcMemHandle_v1 ipc_mem_handle_v1;

  // CHECK: hipArray* array_st_ptr;
  // CHECK-NEXT: hipArray* array_ptr;
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

  // CHECK: hipExternalMemory_t ext_mem;
  CUexternalMemory ext_mem;

  // CHECK: hipExternalSemaphore_t ext_sema;
  CUexternalSemaphore ext_sema;

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

  // CHECK: hipGraph* graph_st;
  // CHECK-NEXT: hipGraph_t graph;
  CUgraph_st* graph_st;
  CUgraph graph;

  // CHECK: hipGraphExec* graphExec_st;
  // CHECK-NEXT: hipGraphExec_t graphExec;
  CUgraphExec_st* graphExec_st;
  CUgraphExec graphExec;

  // CHECK: hipGraphicsResource* graphicsResource_st;
  // CHECK-NEXT: hipGraphicsResource_t graphicsResource;
  CUgraphicsResource_st* graphicsResource_st;
  CUgraphicsResource graphicsResource;

  return 0;
}
