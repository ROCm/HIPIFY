// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

int main() {
  printf("18. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t;

#if CUDA_VERSION >= 10010
  // CHECK: _rocsparse_spmat_descr *spMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_spmat_descr spMatDescr_t;
  cusparseSpMatDescr *spMatDescr = nullptr;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: _rocsparse_dnmat_descr *dnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnmat_descr dnMatDescr_t;
  cusparseDnMatDescr *dnMatDescr = nullptr;
  cusparseDnMatDescr_t dnMatDescr_t;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: _rocsparse_spvec_descr *spVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_spvec_descr spVecDescr_t;
  cusparseSpVecDescr *spVecDescr = nullptr;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: _rocsparse_dnvec_descr *dnVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnvec_descr dnVecDescr_t;
  cusparseDnVecDescr *dnVecDescr = nullptr;
  cusparseDnVecDescr_t dnVecDescr_t;
#endif

#if CUDA_VERSION < 11000
  // CHECK: _rocsparse_hyb_mat *hybMat = nullptr;
  // CHECK-NEXT: rocsparse_hyb_mat hybMat_t;
  cusparseHybMat *hybMat = nullptr;
  cusparseHybMat_t hybMat_t;
#endif

  return 0;
}
