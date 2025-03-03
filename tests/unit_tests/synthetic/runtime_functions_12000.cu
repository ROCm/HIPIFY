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

  void *pfn = nullptr;
  std::string symbol = "symbol";

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

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

  // CHECK: hipDriverProcAddressQueryResult driverProcAddressQueryResult;
  cudaDriverEntryPointQueryResult driverProcAddressQueryResult;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, enum cudaDriverEntryPointQueryResult *driverStatus = NULL);
  // CUDA < 12000: extern __host__ cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags);
  // NOTE: cudaGetDriverEntryPoint for CUDA < 12000 is not supported by HIP
  // TODO: detect cudaGetDriverEntryPoint signature and report warning/error for old (before CUDA 12.0) signature
  // HIP: hipError_t hipGetProcAddress(const char* symbol, void** pfn, int hipVersion, uint64_t flags, hipDriverProcAddressQueryResult* symbolStatus);
  // TODO: add an explicit static_cast<uint64_t> for ull
  // CHECK: result = hipGetProcAddress(symbol.c_str(), &pfn, 604, ull, &driverProcAddressQueryResult);
  result = cudaGetDriverEntryPoint(symbol.c_str(), &pfn, ull, &driverProcAddressQueryResult);
#endif

  return 0;
}
