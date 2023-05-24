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

  // CHECK: rocsparse_diag_type diagType_t;
  // CHECK-NEXT: rocsparse_diag_type DIAG_TYPE_NON_UNIT = rocsparse_diag_type_non_unit;
  // CHECK-NEXT: rocsparse_diag_type DIAG_TYPE_UNIT = rocsparse_diag_type_unit;
  cusparseDiagType_t diagType_t;
  cusparseDiagType_t DIAG_TYPE_NON_UNIT = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseDiagType_t DIAG_TYPE_UNIT = CUSPARSE_DIAG_TYPE_UNIT;

  // CHECK: rocsparse_fill_mode fillMode_t;
  // CHECK-NEXT: rocsparse_fill_mode FILL_MODE_LOWER = rocsparse_fill_mode_lower;
  // CHECK-NEXT: rocsparse_fill_mode FILL_MODE_UPPER = rocsparse_fill_mode_upper;
  cusparseFillMode_t fillMode_t;
  cusparseFillMode_t FILL_MODE_LOWER = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t FILL_MODE_UPPER = CUSPARSE_FILL_MODE_UPPER;

  // CHECK: rocsparse_action action_t;
  // CHECK-NEXT: rocsparse_action ACTION_SYMBOLIC = rocsparse_action_symbolic;
  // CHECK-NEXT: rocsparse_action ACTION_NUMERIC = rocsparse_action_numeric;
  cusparseAction_t action_t;
  cusparseAction_t ACTION_SYMBOLIC = CUSPARSE_ACTION_SYMBOLIC;
  cusparseAction_t ACTION_NUMERIC = CUSPARSE_ACTION_NUMERIC;

  // CHECK: rocsparse_direction direction_t;
  // CHECK-NEXT: rocsparse_direction DIRECTION_ROW = rocsparse_direction_row;
  // CHECK-NEXT: rocsparse_direction DIRECTION_COLUMN = rocsparse_direction_column;
  cusparseDirection_t direction_t;
  cusparseDirection_t DIRECTION_ROW = CUSPARSE_DIRECTION_ROW;
  cusparseDirection_t DIRECTION_COLUMN = CUSPARSE_DIRECTION_COLUMN;

  // CHECK: rocsparse_solve_policy solvePolicy_t;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_NO_LEVEL = rocsparse_solve_policy_auto;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_USE_LEVEL = rocsparse_solve_policy_auto;
  cusparseSolvePolicy_t solvePolicy_t;
  cusparseSolvePolicy_t SOLVE_POLICY_NO_LEVEL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseSolvePolicy_t SOLVE_POLICY_USE_LEVEL = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  // CHECK: rocsparse_pointer_mode pointerMode_t;
  // CHECK-NEXT: rocsparse_pointer_mode POINTER_MODE_HOST = rocsparse_pointer_mode_host;
  // CHECK-NEXT: rocsparse_pointer_mode POINTER_MODE_DEVICE = rocsparse_pointer_mode_device;
  cusparsePointerMode_t pointerMode_t;
  cusparsePointerMode_t POINTER_MODE_HOST = CUSPARSE_POINTER_MODE_HOST;
  cusparsePointerMode_t POINTER_MODE_DEVICE = CUSPARSE_POINTER_MODE_DEVICE;

  // CHECK: rocsparse_status status_t;
  // CHECK-NEXT: rocsparse_status STATUS_SUCCESS = rocsparse_status_success;
  // CHECK-NEXT: rocsparse_status STATUS_NOT_INITIALIZED = rocsparse_status_not_initialized;
  // CHECK-NEXT: rocsparse_status STATUS_ALLOC_FAILED = rocsparse_status_memory_error;
  // CHECK-NEXT: rocsparse_status STATUS_INVALID_VALUE = rocsparse_status_invalid_value;
  // CHECK-NEXT: rocsparse_status STATUS_ARCH_MISMATCH = rocsparse_status_arch_mismatch;
  // CHECK-NEXT: rocsparse_status STATUS_INTERNAL_ERROR = rocsparse_status_internal_error;
  // CHECK-NEXT: rocsparse_status STATUS_ZERO_PIVOT = rocsparse_status_zero_pivot;
  cusparseStatus_t status_t;
  cusparseStatus_t STATUS_SUCCESS = CUSPARSE_STATUS_SUCCESS;
  cusparseStatus_t STATUS_NOT_INITIALIZED = CUSPARSE_STATUS_NOT_INITIALIZED;
  cusparseStatus_t STATUS_ALLOC_FAILED = CUSPARSE_STATUS_ALLOC_FAILED;
  cusparseStatus_t STATUS_INVALID_VALUE = CUSPARSE_STATUS_INVALID_VALUE;
  cusparseStatus_t STATUS_ARCH_MISMATCH = CUSPARSE_STATUS_ARCH_MISMATCH;
  cusparseStatus_t STATUS_INTERNAL_ERROR = CUSPARSE_STATUS_INTERNAL_ERROR;
  cusparseStatus_t STATUS_ZERO_PIVOT = CUSPARSE_STATUS_ZERO_PIVOT;

#if CUDA_VERSION >= 10010
  // CHECK: _rocsparse_spmat_descr *spMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_spmat_descr spMatDescr_t;
  cusparseSpMatDescr *spMatDescr = nullptr;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: _rocsparse_dnmat_descr *dnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnmat_descr dnMatDescr_t;
  cusparseDnMatDescr *dnMatDescr = nullptr;
  cusparseDnMatDescr_t dnMatDescr_t;

  // CHECK: rocsparse_indextype indexType_t;
  // CHECK-NEXT: rocsparse_indextype INDEX_16U = rocsparse_indextype_u16;
  // CHECK-NEXT: rocsparse_indextype INDEX_32I = rocsparse_indextype_i32;
  // CHECK-NEXT: rocsparse_indextype INDEX_64I = rocsparse_indextype_i64;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t INDEX_16U = CUSPARSE_INDEX_16U;
  cusparseIndexType_t INDEX_32I = CUSPARSE_INDEX_32I;
  cusparseIndexType_t INDEX_64I = CUSPARSE_INDEX_64I;

  // CHECK: rocsparse_format format_t;
  // CHECK-NEXT: rocsparse_format FORMAT_CSR = rocsparse_format_csr;
  // CHECK-NEXT: rocsparse_format FORMAT_CSC = rocsparse_format_csc;
  // CHECK-NEXT: rocsparse_format FORMAT_CSO = rocsparse_format_coo;
  cusparseFormat_t format_t;
  cusparseFormat_t FORMAT_CSR = CUSPARSE_FORMAT_CSR;
  cusparseFormat_t FORMAT_CSC = CUSPARSE_FORMAT_CSC;
  cusparseFormat_t FORMAT_CSO = CUSPARSE_FORMAT_COO;
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

  // CHECK: rocsparse_status STATUS_NOT_SUPPORTED = rocsparse_status_not_implemented;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;
#endif

#if CUDA_VERSION >= 10020 && CUDA_VERSION < 12000
  // CHECK: rocsparse_format FORMAT_COO_AOS = rocsparse_format_coo_aos;
  cusparseFormat_t FORMAT_COO_AOS = CUSPARSE_FORMAT_COO_AOS;
#endif

#if CUDA_VERSION < 11000
  // CHECK: _rocsparse_hyb_mat *hybMat = nullptr;
  // CHECK-NEXT: rocsparse_hyb_mat hybMat_t;
  cusparseHybMat *hybMat = nullptr;
  cusparseHybMat_t hybMat_t;

  // CHECK: rocsparse_hyb_partition hybPartition_t;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_AUTO = rocsparse_hyb_partition_auto;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_USER = rocsparse_hyb_partition_user;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_MAX = rocsparse_hyb_partition_max;
  cusparseHybPartition_t hybPartition_t;
  cusparseHybPartition_t HYB_PARTITION_AUTO = CUSPARSE_HYB_PARTITION_AUTO;
  cusparseHybPartition_t HYB_PARTITION_USER = CUSPARSE_HYB_PARTITION_USER;
  cusparseHybPartition_t HYB_PARTITION_MAX = CUSPARSE_HYB_PARTITION_MAX;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: rocsparse_format FORMAT_BLOCKED_ELL = rocsparse_format_bell;
  cusparseFormat_t FORMAT_BLOCKED_ELL = CUSPARSE_FORMAT_BLOCKED_ELL;
#endif

#if CUDA_VERSION >= 12010
  // CHECK: rocsparse_format FORMAT_BSR = rocsparse_format_bsr;
  // CHECK-NEXT: rocsparse_format FORMAT_SLICED_ELLPACK = rocsparse_format_ell;
  cusparseFormat_t FORMAT_BSR = CUSPARSE_FORMAT_BSR;
  cusparseFormat_t FORMAT_SLICED_ELLPACK = CUSPARSE_FORMAT_SLICED_ELLPACK;
#endif

  return 0;
}
