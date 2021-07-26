// RUN: %run_test hipify "%s" "%t" %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>

int main() {
  printf("05. CUDA Runtime API Structs synthetic test\n");

  // CHECK: hipChannelFormatDesc ChannelFormatDesc;
  cudaChannelFormatDesc ChannelFormatDesc;

  // CHECK: hipDeviceProp_t DeviceProp;
  cudaDeviceProp DeviceProp;

  // CHECK: hipExtent Extent;
  cudaExtent Extent;

  // CHECK: hipExternalMemoryBufferDesc ExternalMemoryBufferDesc;
  cudaExternalMemoryBufferDesc ExternalMemoryBufferDesc;

  // CHECK: hipExternalMemoryHandleDesc ExternalMemoryHandleDesc;
  cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc;

  // CHECK: hipExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;
  cudaExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;

  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
  // CHECK-NEXT: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams_v1;
  cudaExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
  cudaExternalSemaphoreSignalParams_v1 ExternalSemaphoreSignalParams_v1;

  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
  // CHECK-NEXT: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams_v1;
  cudaExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
  cudaExternalSemaphoreWaitParams_v1 ExternalSemaphoreWaitParams_v1;

  // CHECK: hipFuncAttributes FuncAttributes;
  cudaFuncAttributes FuncAttributes;

  // CHECK: hipHostNodeParams HostNodeParams;
  cudaHostNodeParams HostNodeParams;

  // CHECK: hipIpcEventHandle_st IpcEventHandle_st;
  // CHECK-NEXT: hipIpcEventHandle_t IpcEventHandle_t;
  cudaIpcEventHandle_st IpcEventHandle_st;
  cudaIpcEventHandle_t IpcEventHandle_t;

  // CHECK: hipIpcMemHandle_st IpcMemHandle_st;
  // CHECK-NEXT: hipIpcMemHandle_t IpcMemHandle_t;
  cudaIpcMemHandle_st IpcMemHandle_st;
  cudaIpcMemHandle_t IpcMemHandle_t;

  // CHECK: hipKernelNodeParams KernelNodeParams;
  cudaKernelNodeParams KernelNodeParams;

  // CHECK: hipLaunchParams LaunchParams;
  cudaLaunchParams LaunchParams;

  // CHECK: hipMemcpy3DParms Memcpy3DParms;
  cudaMemcpy3DParms Memcpy3DParms;

  // CHECK: hipMemsetParams MemsetParams;
  cudaMemsetParams MemsetParams;

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

  // CHECK: surfaceReference surfaceRef;
  surfaceReference surfaceRef;

  // CHECK: ihipEvent_t* event_st;
  // CHECK-NEXT: hipEvent_t Event_t;
  CUevent_st* event_st;
  cudaEvent_t Event_t;

  // CHECK: hipExternalMemory_t ExternalMemory_t;
  cudaExternalMemory_t ExternalMemory_t;

  // CHECK: hipExternalSemaphore_t ExternalSemaphore_t;
  cudaExternalSemaphore_t ExternalSemaphore_t;

  // CHECK: hipGraph* graph_st;
  // CHECK-NEXT: hipGraph_t Graph_t;
  CUgraph_st* graph_st;
  cudaGraph_t Graph_t;

  // CHECK: hipGraphExec* graphExec_st;
  // CHECK-NEXT: hipGraphExec_t GraphExec_t;
  CUgraphExec_st* graphExec_st;
  cudaGraphExec_t GraphExec_t;

  // CHECK: hipGraphicsResource* GraphicsResource;
  // CHECK-NEXT: hipGraphicsResource_t GraphicsResource_t;
  cudaGraphicsResource* GraphicsResource;
  cudaGraphicsResource_t GraphicsResource_t;

  // CHECK: hipGraphNode* graphNode_st;
  // CHECK-NEXT: hipGraphNode_t GraphNode_t;
  CUgraphNode_st* graphNode_st;
  cudaGraphNode_t GraphNode_t;

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
  // CHECK-NEXT: hipFunction_t func;
  CUfunc_st* func_st_ptr;
  cudaFunction_t func;

  return 0;
}
