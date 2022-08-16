// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>

int main() {
  printf("12.11010. CUDA Runtime API Functions synthetic test for CUDA >= 11010\n");

  size_t bytes = 0;
  size_t width = 0;
  size_t wOffset = 0;
  void* image = nullptr;
  void* src = nullptr;
  void* dst = nullptr;

  // CHECK: hipError_t result = hipSuccess;
  cudaError result = cudaSuccess;

  // CHECK: hipMemcpyKind MemcpyKind;
  cudaMemcpyKind MemcpyKind;

#if CUDA_VERSION >= 10000
  // CHECK: hipGraphNode_t graphNode, graphNode_2;
  cudaGraphNode_t graphNode, graphNode_2;

  // CHECK: hipGraph_t Graph_t, Graph_t_2;
  cudaGraph_t Graph_t, Graph_t_2;

  // CHECK: hipGraphExec_t GraphExec_t;
  cudaGraphExec_t GraphExec_t;
#endif

#if CUDA_VERSION >= 11010
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeToSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphAddMemcpyNodeToSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphAddMemcpyNodeToSymbol(&graphNode, Graph_t, &graphNode_2, width, HIP_SYMBOL(image), src, bytes, wOffset, MemcpyKind);
  result = cudaGraphAddMemcpyNodeToSymbol(&graphNode, Graph_t, &graphNode_2, width, image, src, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNodeFromSymbol(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphAddMemcpyNodeFromSymbol(&graphNode, Graph_t, &graphNode_2, width, dst, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind);
  result = cudaGraphAddMemcpyNodeFromSymbol(&graphNode, Graph_t, &graphNode_2, width, dst, image, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsToSymbol(cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphMemcpyNodeSetParamsToSymbol(hipGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphMemcpyNodeSetParamsToSymbol(graphNode, HIP_SYMBOL(image), src, bytes, wOffset, MemcpyKind);
  result = cudaGraphMemcpyNodeSetParamsToSymbol(graphNode, image, src, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParamsFromSymbol(cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(hipGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphMemcpyNodeSetParamsFromSymbol(graphNode, dst, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind);
  result = cudaGraphMemcpyNodeSetParamsFromSymbol(graphNode, dst, image, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsToSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphExecMemcpyNodeSetParamsToSymbol(GraphExec_t, graphNode, HIP_SYMBOL(image), src, bytes, wOffset, MemcpyKind);
  result = cudaGraphExecMemcpyNodeSetParamsToSymbol(GraphExec_t, graphNode, image, src, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParamsFromSymbol(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind);
  // CHECK: result = hipGraphExecMemcpyNodeSetParamsFromSymbol(GraphExec_t, graphNode, dst, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind);
  result = cudaGraphExecMemcpyNodeSetParamsFromSymbol(GraphExec_t, graphNode, dst, image, bytes, wOffset, MemcpyKind);
#endif

  return 0;
}
