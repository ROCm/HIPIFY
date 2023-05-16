// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>

int main() {
  printf("12.12000. CUDA Runtime API Functions synthetic test for CUDA >= 12000\n");

  // CHECK: hipError_t result = hipSuccess;
  cudaError result = cudaSuccess;

  // CHECK: hipGraphExec_t GraphExec_t;
  cudaGraphExec_t GraphExec_t;

  // CHECK: hipGraph_t Graph_t;
  cudaGraph_t Graph_t;

#if defined(_WIN32)
  unsigned long long ull = 0;
#else
  unsigned long ull = 0;
#endif

#if CUDA_VERSION >= 12000
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0));
  // HIP: hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags);
  // CHECK: result = hipGraphInstantiateWithFlags(&GraphExec_t, Graph_t, ull);
  result = cudaGraphInstantiate(&GraphExec_t, Graph_t, ull);
#endif

  return 0;
}
