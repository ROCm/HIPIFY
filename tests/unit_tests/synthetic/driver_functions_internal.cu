// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#elif CUDA_VERSION <= 10000
  #include <GL/glew.h>
#endif
#include "cudaGL.h"

int main() {
  printf("13. CUDA Driver API Internal Functions synthetic test\n");

#if defined(_WIN32)
  unsigned long long ull = 0;
#else
  unsigned long ull = 0;
#endif
  size_t bytes = 0;

  // CHECK: hipTexRef texref;
  // CHECK-NEXT: HIP_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  // CHECK-NEXT: hipDeviceptr_t deviceptr;
  // CHECK-NEXT: hipStream_t stream;
  CUtexref texref;
  CUDA_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  CUdeviceptr deviceptr;
  CUstream stream;

  // CUDA: CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr, size_t Pitch);
  // CHECK: hipError_t result = hipTexRefSetAddress2D(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);
  CUresult result = cuTexRefSetAddress2D_v2(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);

  // [TODO][#2062] Rename all DO-NOT-CHECK back
#if CUDA_VERSION >= 10000
  // DO-NOT-CHECK: hipStreamCaptureStatus streamCaptureStatus;
  // DO-NOT-CHECK-NEXT: hipGraph_t graph;
  // DO-NOT-CHECK-NEXT: const hipGraphNode_t *pGraphNode = nullptr;
  CUstreamCaptureStatus streamCaptureStatus;
  CUgraph graph;
  const CUgraphNode *pGraphNode = nullptr;
#endif

  // [TODO][#2062] Rename all DO-NOT-CHECK back
#if CUDA_VERSION >= 11030
  // CUDA < 12000: CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
  // CUDA:         CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
  // HIP: hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out __dparm(0), hipGraph_t* graph_out __dparm(0), const hipGraphNode_t** dependencies_out __dparm(0), size_t* numDependencies_out __dparm(0));
  // DO-NOT-CHECK: result = hipStreamGetCaptureInfo_v2(stream, &streamCaptureStatus, &ull, &graph, &pGraphNode, &bytes);
  result = cuStreamGetCaptureInfo_v2(stream, &streamCaptureStatus, &ull, &graph, &pGraphNode, &bytes);
#endif

  return 0;
}
