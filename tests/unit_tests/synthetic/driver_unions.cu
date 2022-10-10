// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main() {
  printf("10. CUDA Driver API Unions synthetic test\n");

#if CUDA_VERSION >= 11000 && CUDA_VERSION < 11080
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue;
  // CHECK-NEXT: hipKernelNodeAttrValue kernelNodeAttrValue_union;
  CUkernelNodeAttrValue kernelNodeAttrValue;
  CUkernelNodeAttrValue_union kernelNodeAttrValue_union;
#endif

#if CUDA_VERSION >= 11030 && CUDA_VERSION < 11080
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue_v1;
  CUkernelNodeAttrValue_v1 kernelNodeAttrValue_v1;
#endif

  return 0;
}
