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
  printf("27. cuSPARSELt API to hipSPARSELt API synthetic test\n");

  return 0;
}
