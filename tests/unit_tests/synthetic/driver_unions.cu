// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main() {
  printf("10. CUDA Driver API Unions synthetic test\n");

#if CUDA_VERSION >= 8000
  // CHECK: hipStreamBatchMemOpParams streamBatchMemOpParams;
  // CHECK-NEXT: hipStreamBatchMemOpParams_union streamBatchMemOpParams_union;
  CUstreamBatchMemOpParams streamBatchMemOpParams;
  CUstreamBatchMemOpParams_union streamBatchMemOpParams_union;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue;
  CUkernelNodeAttrValue kernelNodeAttrValue;

  // CHECK: hipLaunchAttributeValue streamAttrValue;
  CUstreamAttrValue streamAttrValue;
#endif

#if CUDA_VERSION >= 11000 && CUDA_VERSION < 11080
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue_union;
  CUkernelNodeAttrValue_union kernelNodeAttrValue_union;

  // CHECK: hipLaunchAttributeValue streamAttrValue_union;
  CUstreamAttrValue_union streamAttrValue_union;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue_v1;
  CUkernelNodeAttrValue_v1 kernelNodeAttrValue_v1;

  // CHECK: hipLaunchAttributeValue streamAttrValue_v1;
  CUstreamAttrValue_v1 streamAttrValue_v1;
#endif

#if CUDA_VERSION >= 11080
  // CHECK: hipLaunchAttributeValue LaunchAttributeValue;
  cudaLaunchAttributeValue LaunchAttributeValue;
#endif

  return 0;
}
