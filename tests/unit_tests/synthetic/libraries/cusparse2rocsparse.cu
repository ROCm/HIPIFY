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

  // CHECK: _rocsparse_color_info *colorInfo = nullptr;
  // CHECK-NEXT: rocsparse_color_info colorInfo_t;
  cusparseColorInfo *colorInfo = nullptr;
  cusparseColorInfo_t colorInfo_t;

  // CHECK: rocsparse_operation sparseOperation_t;
  // CHECK-NEXT: rocsparse_operation OPERATION_NON_TRANSPOSE = rocsparse_operation_none;
  // CHECK-NEXT: rocsparse_operation OPERATION_TRANSPOSE = rocsparse_operation_transpose;
  // CHECK-NEXT: rocsparse_operation OPERATION_CONJUGATE_TRANSPOSE = rocsparse_operation_conjugate_transpose;
  cusparseOperation_t sparseOperation_t;
  cusparseOperation_t OPERATION_NON_TRANSPOSE = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t OPERATION_TRANSPOSE = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t OPERATION_CONJUGATE_TRANSPOSE = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

  // CHECK: rocsparse_index_base indexBase_t;
  // CHECK-NEXT: rocsparse_index_base INDEX_BASE_ZERO = rocsparse_index_base_zero;
  // CHECK-NEXT: rocsparse_index_base INDEX_BASE_ONE = rocsparse_index_base_one;
  cusparseIndexBase_t indexBase_t;
  cusparseIndexBase_t INDEX_BASE_ZERO = CUSPARSE_INDEX_BASE_ZERO;
  cusparseIndexBase_t INDEX_BASE_ONE = CUSPARSE_INDEX_BASE_ONE;

  // CHECK: rocsparse_matrix_type matrixType_t;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_GENERAL = rocsparse_matrix_type_general;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_SYMMETRIC = rocsparse_matrix_type_symmetric;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_HERMITIAN = rocsparse_matrix_type_hermitian;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_TRIANGULAR = rocsparse_matrix_type_triangular;
  cusparseMatrixType_t matrixType_t;
  cusparseMatrixType_t MATRIX_TYPE_GENERAL = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseMatrixType_t MATRIX_TYPE_SYMMETRIC = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
  cusparseMatrixType_t MATRIX_TYPE_HERMITIAN = CUSPARSE_MATRIX_TYPE_HERMITIAN;
  cusparseMatrixType_t MATRIX_TYPE_TRIANGULAR = CUSPARSE_MATRIX_TYPE_TRIANGULAR;

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
