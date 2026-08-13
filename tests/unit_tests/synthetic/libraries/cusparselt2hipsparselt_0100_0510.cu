// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "hipsparseLt.h"
#include "cusparseLt.h"
// CHECK-NOT: #include "hipsparseLt.h"

int main() {
  printf("27.1.after_0001_before_0502 cuSPARSELt API to hipSPARSELt API synthetic test\n");

#if CUSPARSELT_VERSION >= 100 && CUSPARSELT_VERSION <= 501
  // CHECK: hipsparseLtComputetype_t SPARSE_COMPUTE_TF32 = HIPSPARSELT_COMPUTE_TF32;
  // CHECK-NEXT: hipsparseLtComputetype_t SPARSE_COMPUTE_TF32_FAST = HIPSPARSELT_COMPUTE_TF32_FAST;
  cusparseComputeType SPARSE_COMPUTE_TF32 = CUSPARSE_COMPUTE_TF32;
  cusparseComputeType SPARSE_COMPUTE_TF32_FAST = CUSPARSE_COMPUTE_TF32_FAST;
#endif

  return 0;
}
