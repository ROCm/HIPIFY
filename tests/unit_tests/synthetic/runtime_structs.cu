// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

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

#if CUDA_VERSION > 9020
  // CHECK: hipExternalMemoryBufferDesc ExternalMemoryBufferDesc;
  cudaExternalMemoryBufferDesc ExternalMemoryBufferDesc;

  // CHECK: hipExternalMemoryHandleDesc ExternalMemoryHandleDesc;
  cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc;

  // CHECK: hipExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;
  cudaExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;

  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
  cudaExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
#endif
#if CUDA_VERSION > 11010
  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams_v1;
  cudaExternalSemaphoreSignalParams_v1 ExternalSemaphoreSignalParams_v1;
#endif

#if CUDA_VERSION > 9020
  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
  cudaExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
#endif
#if CUDA_VERSION > 11010
  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams_v1;
  cudaExternalSemaphoreWaitParams_v1 ExternalSemaphoreWaitParams_v1;
#endif

  // CHECK: hipFuncAttributes FuncAttributes;
  cudaFuncAttributes FuncAttributes;

#if CUDA_VERSION > 9020
  // CHECK: hipHostNodeParams HostNodeParams;
  cudaHostNodeParams HostNodeParams;
#endif

  // CHECK: hipIpcEventHandle_st IpcEventHandle_st;
  // CHECK-NEXT: hipIpcEventHandle_t IpcEventHandle_t;
  cudaIpcEventHandle_st IpcEventHandle_st;
  cudaIpcEventHandle_t IpcEventHandle_t;

  // CHECK: hipIpcMemHandle_st IpcMemHandle_st;
  // CHECK-NEXT: hipIpcMemHandle_t IpcMemHandle_t;
  cudaIpcMemHandle_st IpcMemHandle_st;
  cudaIpcMemHandle_t IpcMemHandle_t;

#if CUDA_VERSION > 9020
  // CHECK: hipKernelNodeParams KernelNodeParams;
  cudaKernelNodeParams KernelNodeParams;
#endif

#if CUDA_VERSION > 8000
  // CHECK: hipLaunchParams LaunchParams;
  cudaLaunchParams LaunchParams;
#endif

  // CHECK: hipMemcpy3DParms Memcpy3DParms;
  cudaMemcpy3DParms Memcpy3DParms;

#if CUDA_VERSION > 9020
  // CHECK: hipMemsetParams MemsetParams;
  cudaMemsetParams MemsetParams;
#endif

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

#if CUDA_VERSION > 9020
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
#endif

  // CHECK: hipGraphicsResource* GraphicsResource;
  // CHECK-NEXT: hipGraphicsResource_t GraphicsResource_t;
  cudaGraphicsResource* GraphicsResource;
  cudaGraphicsResource_t GraphicsResource_t;

#if CUDA_VERSION > 9020
  // CHECK: hipGraphNode* graphNode_st;
  // CHECK-NEXT: hipGraphNode_t GraphNode_t;
  CUgraphNode_st* graphNode_st;
  cudaGraphNode_t GraphNode_t;
#endif

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
#if CUDA_VERSION > 10020
  // CHECK: hipFunction_t func;
  cudaFunction_t func;
#endif

  // CHECK: hipUUID_t uuid_st;
  CUuuid_st uuid_st;

  return 0;
}
