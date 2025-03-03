// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblaslt.h"
#include "cublasLt.h"
// CHECK-NOT: #include "hipblaslt.h"

int main() {
  printf("21.10010.10020. cuBLASLt API to hipBLASLt API synthetic test\n");

#if CUDA_VERSION >= 10010 && CUDA_VERSION <= 10020
  // CHECK: hipblasLtMatrixLayoutOpaque_t blasLtMatrixLayoutOpaque;
  cublasLtMatrixLayoutStruct blasLtMatrixLayoutOpaque;
#endif

  return 0;
}
