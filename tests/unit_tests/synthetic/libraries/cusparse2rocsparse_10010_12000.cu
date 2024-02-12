// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

int main() {
  printf("18.1. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: rocsparse_status status_t;
  cusparseStatus_t status_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

  // CHECK: rocsparse_action action_t;
  cusparseAction_t action_t;

  // CHECK: rocsparse_index_base indexBase_t;
  cusparseIndexBase_t indexBase_t;

  int m = 0;
  int n = 0;
  int innz = 0;
  int csrRowPtrA = 0;
  int csrRowPtrB = 0;
  int csrRowPtrC = 0;
  int cscRowIndA = 0;
  int csrColIndA = 0;
  int csrColIndB = 0;
  int csrColIndC = 0;
  int cscColPtrA = 0;
  size_t bufferSize = 0;
  void *pcsrVal = nullptr;
  void *pcscVal = nullptr;
  void *alpha = nullptr;
  void *beta = nullptr;
  void *tempBuffer = nullptr;
  void* result = nullptr;

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

#if CUDA_VERSION >= 8000
  // TODO: [#899] There should be rocsparse_datatype instead of hipDataType
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 10010 && CUSPARSE_VERSION >= 10200
  // TODO: cusparseCsr2CscAlg_t has no analogue in rocSPARSE. The deletion of declaration and usage is needed to be implemented
  cusparseCsr2CscAlg_t Csr2CscAlg_t;

#if (CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: rocsparse_spmat_descr spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: rocsparse_dnmat_descr dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;

  // CHECK: rocsparse_spmm_alg spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;

  // CHECK: _rocsparse_spvec_descr *spVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_spvec_descr spVecDescr_t;
  cusparseSpVecDescr *spVecDescr = nullptr;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: _rocsparse_dnvec_descr *dnVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnvec_descr dnVecDescr_t, vecX, vecY;
  cusparseDnVecDescr *dnVecDescr = nullptr;
  cusparseDnVecDescr_t dnVecDescr_t, vecX, vecY;
#endif
#endif

  return 0;
}
