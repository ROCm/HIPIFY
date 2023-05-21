// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "hipsparse.h"

int main() {
  printf("17. cuSPARSE API to hipSPARSE API synthetic test\n");

  // CHECK: hipsparseHandle_t handle_t;
  cusparseHandle_t handle_t;

  // CHECK: hipsparseMatDescr_t matDescr_t;
  cusparseMatDescr_t matDescr_t;

#if CUDA_VERSION >= 10010
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t;
  cusparseDnMatDescr_t dnMatDescr_t;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipsparseSpVecDescr_t spVecDescr_t;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: hipsparseDnVecDescr_t dnVecDescr_t;
  cusparseDnVecDescr_t dnVecDescr_t;
#endif

#if CUDA_VERSION < 11000
  // CHECK: hipsparseHybMat_t hybMat_t;
  cusparseHybMat_t hybMat_t;
#endif

  return 0;
}
