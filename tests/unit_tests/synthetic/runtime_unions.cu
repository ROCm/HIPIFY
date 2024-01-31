// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main() {
  printf("11. CUDA Runtime API Unions synthetic test\n");

#if CUDA_VERSION >= 11000
  // CHECK: hipKernelNodeAttrValue KernelNodeAttrValue;
  cudaKernelNodeAttrValue KernelNodeAttrValue;
#endif

  return 0;
}
