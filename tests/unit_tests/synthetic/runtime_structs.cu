// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
  printf("05. CUDA Runtime API Structs synthetic test\n");

  // CHECK: hipChannelFormatDesc ChannelFormatDesc;
  cudaChannelFormatDesc ChannelFormatDesc;

  // CHECK: hipDeviceProp_t DeviceProp;
  cudaDeviceProp DeviceProp;

  // CHECK: hipExtent Extent;
  cudaExtent Extent;

  // CHECK: hipFuncAttributes FuncAttributes;
  cudaFuncAttributes FuncAttributes;

  // CHECK: hipIpcEventHandle_st IpcEventHandle_st;
  // CHECK-NEXT: hipIpcEventHandle_t IpcEventHandle_t;
  cudaIpcEventHandle_st IpcEventHandle_st;
  cudaIpcEventHandle_t IpcEventHandle_t;

  // CHECK: hipIpcMemHandle_st IpcMemHandle_st;
  // CHECK-NEXT: hipIpcMemHandle_t IpcMemHandle_t;
  cudaIpcMemHandle_st IpcMemHandle_st;
  cudaIpcMemHandle_t IpcMemHandle_t;

  // CHECK: hipMemcpy3DParms Memcpy3DParms;
  cudaMemcpy3DParms Memcpy3DParms;

  // CHECK: hipPitchedPtr PitchedPtr;
  cudaPitchedPtr PitchedPtr;

  // CHECK: hipPointerAttribute_t PointerAttributes;
  cudaPointerAttributes PointerAttributes;

  // CHECK: hipPos Pos;
  cudaPos Pos;

  // CHECK: hipResourceDesc ResourceDesc;
  cudaResourceDesc ResourceDesc;

  // CHECK: hipResourceViewDesc ResourceViewDesc;
  cudaResourceViewDesc ResourceViewDesc;

  // CHECK: hipTextureDesc TextureDesc;
  cudaTextureDesc TextureDesc;

  // CHECK: ihipEvent_t* event_st;
  // CHECK-NEXT: hipEvent_t Event_t;
  CUevent_st* event_st;
  cudaEvent_t Event_t;

  // CHECK: hipGraphicsResource* GraphicsResource;
  // CHECK-NEXT: hipGraphicsResource_t GraphicsResource_t;
  cudaGraphicsResource* GraphicsResource;
  cudaGraphicsResource_t GraphicsResource_t;

  // CHECK: hipArray* Array;
  // CHECK-NEXT: hipArray_t Array_t;
  // CHECK-NEXT: hipArray_const_t Array_const_t;
  cudaArray* Array;
  cudaArray_t Array_t;
  cudaArray_const_t Array_const_t;

  // CHECK: hipMipmappedArray* MipmappedArray;
  // CHECK-NEXT: hipMipmappedArray_t MipmappedArray_t;
  // CHECK-NEXT: hipMipmappedArray_const_t MipmappedArray_const_t;
  cudaMipmappedArray* MipmappedArray;
  cudaMipmappedArray_t MipmappedArray_t;
  cudaMipmappedArray_const_t MipmappedArray_const_t;

  // CHECK: ihipStream_t* stream_st;
  // CHECK-NEXT: hipStream_t Stream_t;
  CUstream_st* stream_st;
  cudaStream_t Stream_t;

  // CHECK: ihipModuleSymbol_t* func_st_ptr;
  CUfunc_st* func_st_ptr;

  // CHECK: hipUUID_t uuid_st;
  CUuuid_st uuid_st;

#if CUDA_VERSION >= 9000
  // CHECK: hipLaunchParams LaunchParams;
  cudaLaunchParams LaunchParams;
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipExternalMemoryBufferDesc ExternalMemoryBufferDesc;
  cudaExternalMemoryBufferDesc ExternalMemoryBufferDesc;

  // CHECK: hipExternalMemoryHandleDesc ExternalMemoryHandleDesc;
  cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc;

  // CHECK: hipExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;
  cudaExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;

  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
  cudaExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;

  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
  cudaExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;

  // CHECK: hipHostNodeParams HostNodeParams;
  cudaHostNodeParams HostNodeParams;

  // CHECK: hipKernelNodeParams KernelNodeParams;
  cudaKernelNodeParams KernelNodeParams;

  // CHECK: hipMemsetParams MemsetParams;
  cudaMemsetParams MemsetParams;

  // CHECK: hipExternalMemory_t ExternalMemory_t;
  cudaExternalMemory_t ExternalMemory_t;

  // CHECK: hipExternalSemaphore_t ExternalSemaphore_t;
  cudaExternalSemaphore_t ExternalSemaphore_t;

  // CHECK: ihipGraph* graph_st;
  // CHECK-NEXT: hipGraph_t Graph_t;
  CUgraph_st* graph_st;
  cudaGraph_t Graph_t;

  // CHECK: hipGraphExec* graphExec_st;
  // CHECK-NEXT: hipGraphExec_t GraphExec_t;
  CUgraphExec_st* graphExec_st;
  cudaGraphExec_t GraphExec_t;

  // CHECK: hipGraphNode* graphNode_st;
  // CHECK-NEXT: hipGraphNode_t GraphNode_t;
  CUgraphNode_st* graphNode_st;
  cudaGraphNode_t GraphNode_t;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipFunction_t func;
  cudaFunction_t func;

  // CHECK: hipAccessPolicyWindow AccessPolicyWindow;
  cudaAccessPolicyWindow AccessPolicyWindow;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams_v1;
  cudaExternalSemaphoreSignalParams_v1 ExternalSemaphoreSignalParams_v1;

  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams_v1;
  cudaExternalSemaphoreWaitParams_v1 ExternalSemaphoreWaitParams_v1;

  // CHECK: hipMemPool_t memPool_t;
  cudaMemPool_t memPool_t;

  // CHECK: hipMemLocation memLocation;
  cudaMemLocation memLocation;

  // CHECK: hipMemAccessDesc MemAccessDesc;
  cudaMemAccessDesc MemAccessDesc;

  // CHECK: hipMemPoolProps MemPoolProps;
  cudaMemPoolProps MemPoolProps;

  // CHECK: hipExternalSemaphoreSignalNodeParams ExternalSemaphoreSignalNodeParams;
  cudaExternalSemaphoreSignalNodeParams ExternalSemaphoreSignalNodeParams;

  // CHECK: hipExternalSemaphoreWaitNodeParams ExternalSemaphoreWaitNodeParams;
  cudaExternalSemaphoreWaitNodeParams ExternalSemaphoreWaitNodeParams;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipMemPoolPtrExportData memPoolPtrExportData;
  cudaMemPoolPtrExportData memPoolPtrExportData;

  // CHECK: hipUserObject_t userObject;
  cudaUserObject_t userObject;
#endif

#if CUDA_VERSION >= 11040
  // CHECK: hipMemAllocNodeParams MemAllocNodeParams;
  cudaMemAllocNodeParams MemAllocNodeParams;
#endif

#if CUDA_VERSION < 12000
  // CHECK: surfaceReference surfaceRef;
  surfaceReference surfaceRef;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: hipGraphInstantiateParams GRAPH_INSTANTIATE_PARAMS_st;
  // CHECK-NEXT: hipGraphInstantiateParams GRAPH_INSTANTIATE_PARAMS;
  cudaGraphInstantiateParams_st GRAPH_INSTANTIATE_PARAMS_st;
  cudaGraphInstantiateParams GRAPH_INSTANTIATE_PARAMS;
#endif

#if CUDA_VERSION >= 12020
  // CHECK: hipExternalSemaphoreSignalNodeParams ExternalSemaphoreSignalNodeParams_v2;
  cudaExternalSemaphoreSignalNodeParamsV2 ExternalSemaphoreSignalNodeParams_v2;

  // CHECK: hipExternalSemaphoreWaitNodeParams ExternalSemaphoreWaitNodeParams_v2;
  cudaExternalSemaphoreWaitNodeParamsV2 ExternalSemaphoreWaitNodeParams_v2;

  // CHECK: hipMemFreeNodeParams MemFreeNodeParams;
  cudaMemFreeNodeParams MemFreeNodeParams;

  // CHECK: hipChildGraphNodeParams ChildGraphNodeParams;
  cudaChildGraphNodeParams ChildGraphNodeParams;

  // CHECK: hipEventRecordNodeParams EventRecordNodeParams;
  cudaEventRecordNodeParams EventRecordNodeParams;

  // CHECK: hipEventWaitNodeParams EventWaitNodeParams;
  cudaEventWaitNodeParams EventWaitNodeParams;

  // CHECK: hipGraphNodeParams *GraphNodeParams = nullptr
  cudaGraphNodeParams *GraphNodeParams = nullptr;
#endif

  return 0;
}
