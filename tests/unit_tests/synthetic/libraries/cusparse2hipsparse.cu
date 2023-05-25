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

  // CHECK: hipsparseColorInfo_t colorInfo_t;
  cusparseColorInfo_t colorInfo_t;

  // CHECK: hipsparseOperation_t sparseOperation_t;
  // CHECK-NEXT: hipsparseOperation_t OPERATION_NON_TRANSPOSE = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  // CHECK-NEXT: hipsparseOperation_t OPERATION_TRANSPOSE = HIPSPARSE_OPERATION_TRANSPOSE;
  // CHECK-NEXT: hipsparseOperation_t OPERATION_CONJUGATE_TRANSPOSE = HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  cusparseOperation_t sparseOperation_t;
  cusparseOperation_t OPERATION_NON_TRANSPOSE = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t OPERATION_TRANSPOSE = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t OPERATION_CONJUGATE_TRANSPOSE = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

  // CHECK: hipsparseIndexBase_t indexBase_t;
  // CHECK-NEXT: hipsparseIndexBase_t INDEX_BASE_ZERO = HIPSPARSE_INDEX_BASE_ZERO;
  // CHECK-NEXT: hipsparseIndexBase_t INDEX_BASE_ONE = HIPSPARSE_INDEX_BASE_ONE;
  cusparseIndexBase_t indexBase_t;
  cusparseIndexBase_t INDEX_BASE_ZERO = CUSPARSE_INDEX_BASE_ZERO;
  cusparseIndexBase_t INDEX_BASE_ONE = CUSPARSE_INDEX_BASE_ONE;

  // CHECK: hipsparseMatrixType_t matrixType_t;
  // CHECK-NEXT: hipsparseMatrixType_t MATRIX_TYPE_GENERAL = HIPSPARSE_MATRIX_TYPE_GENERAL;
  // CHECK-NEXT: hipsparseMatrixType_t MATRIX_TYPE_SYMMETRIC = HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
  // CHECK-NEXT: hipsparseMatrixType_t MATRIX_TYPE_HERMITIAN = HIPSPARSE_MATRIX_TYPE_HERMITIAN;
  // CHECK-NEXT: hipsparseMatrixType_t MATRIX_TYPE_TRIANGULAR = HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
  cusparseMatrixType_t matrixType_t;
  cusparseMatrixType_t MATRIX_TYPE_GENERAL = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseMatrixType_t MATRIX_TYPE_SYMMETRIC = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
  cusparseMatrixType_t MATRIX_TYPE_HERMITIAN = CUSPARSE_MATRIX_TYPE_HERMITIAN;
  cusparseMatrixType_t MATRIX_TYPE_TRIANGULAR = CUSPARSE_MATRIX_TYPE_TRIANGULAR;

  // CHECK: hipsparseDiagType_t diagType_t;
  // CHECK-NEXT: hipsparseDiagType_t DIAG_TYPE_NON_UNIT = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  // CHECK-NEXT: hipsparseDiagType_t DIAG_TYPE_UNIT = HIPSPARSE_DIAG_TYPE_UNIT;
  cusparseDiagType_t diagType_t;
  cusparseDiagType_t DIAG_TYPE_NON_UNIT = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseDiagType_t DIAG_TYPE_UNIT = CUSPARSE_DIAG_TYPE_UNIT;

  // CHECK: hipsparseFillMode_t fillMode_t;
  // CHECK-NEXT: hipsparseFillMode_t FILL_MODE_LOWER = HIPSPARSE_FILL_MODE_LOWER;
  // CHECK-NEXT: hipsparseFillMode_t FILL_MODE_UPPER = HIPSPARSE_FILL_MODE_UPPER;
  cusparseFillMode_t fillMode_t;
  cusparseFillMode_t FILL_MODE_LOWER = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t FILL_MODE_UPPER = CUSPARSE_FILL_MODE_UPPER;

  // CHECK: hipsparseAction_t action_t;
  // CHECK-NEXT: hipsparseAction_t ACTION_SYMBOLIC = HIPSPARSE_ACTION_SYMBOLIC;
  // CHECK-NEXT: hipsparseAction_t ACTION_NUMERIC = HIPSPARSE_ACTION_NUMERIC;
  cusparseAction_t action_t;
  cusparseAction_t ACTION_SYMBOLIC = CUSPARSE_ACTION_SYMBOLIC;
  cusparseAction_t ACTION_NUMERIC = CUSPARSE_ACTION_NUMERIC;

  // CHECK: hipsparseDirection_t direction_t;
  // CHECK-NEXT: hipsparseDirection_t DIRECTION_ROW = HIPSPARSE_DIRECTION_ROW;
  // CHECK-NEXT: hipsparseDirection_t DIRECTION_COLUMN = HIPSPARSE_DIRECTION_COLUMN;
  cusparseDirection_t direction_t;
  cusparseDirection_t DIRECTION_ROW = CUSPARSE_DIRECTION_ROW;
  cusparseDirection_t DIRECTION_COLUMN = CUSPARSE_DIRECTION_COLUMN;

  // CHECK: hipsparseSolvePolicy_t solvePolicy_t;
  // CHECK-NEXT: hipsparseSolvePolicy_t SOLVE_POLICY_NO_LEVEL = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;
  // CHECK-NEXT: hipsparseSolvePolicy_t SOLVE_POLICY_USE_LEVEL = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  cusparseSolvePolicy_t solvePolicy_t;
  cusparseSolvePolicy_t SOLVE_POLICY_NO_LEVEL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseSolvePolicy_t SOLVE_POLICY_USE_LEVEL = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  // CHECK: hipsparsePointerMode_t pointerMode_t;
  // CHECK-NEXT: hipsparsePointerMode_t POINTER_MODE_HOST = HIPSPARSE_POINTER_MODE_HOST;
  // CHECK-NEXT: hipsparsePointerMode_t POINTER_MODE_DEVICE = HIPSPARSE_POINTER_MODE_DEVICE;
  cusparsePointerMode_t pointerMode_t;
  cusparsePointerMode_t POINTER_MODE_HOST = CUSPARSE_POINTER_MODE_HOST;
  cusparsePointerMode_t POINTER_MODE_DEVICE = CUSPARSE_POINTER_MODE_DEVICE;

  // CHECK: hipsparseStatus_t status_t;
  // CHECK-NEXT: hipsparseStatus_t STATUS_SUCCESS = HIPSPARSE_STATUS_SUCCESS;
  // CHECK-NEXT: hipsparseStatus_t STATUS_NOT_INITIALIZED = HIPSPARSE_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hipsparseStatus_t STATUS_ALLOC_FAILED = HIPSPARSE_STATUS_ALLOC_FAILED;
  // CHECK-NEXT: hipsparseStatus_t STATUS_INVALID_VALUE = HIPSPARSE_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hipsparseStatus_t STATUS_ARCH_MISMATCH = HIPSPARSE_STATUS_ARCH_MISMATCH;
  // CHECK-NEXT: hipsparseStatus_t STATUS_MAPPING_ERROR = HIPSPARSE_STATUS_MAPPING_ERROR;
  // CHECK-NEXT: hipsparseStatus_t STATUS_EXECUTION_FAILED = HIPSPARSE_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT: hipsparseStatus_t STATUS_INTERNAL_ERROR = HIPSPARSE_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hipsparseStatus_t STATUS_MATRIX_TYPE_NOT_SUPPORTED = HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  // CHECK-NEXT: hipsparseStatus_t STATUS_ZERO_PIVOT = HIPSPARSE_STATUS_ZERO_PIVOT;
  cusparseStatus_t status_t;
  cusparseStatus_t STATUS_SUCCESS = CUSPARSE_STATUS_SUCCESS;
  cusparseStatus_t STATUS_NOT_INITIALIZED = CUSPARSE_STATUS_NOT_INITIALIZED;
  cusparseStatus_t STATUS_ALLOC_FAILED = CUSPARSE_STATUS_ALLOC_FAILED;
  cusparseStatus_t STATUS_INVALID_VALUE = CUSPARSE_STATUS_INVALID_VALUE;
  cusparseStatus_t STATUS_ARCH_MISMATCH = CUSPARSE_STATUS_ARCH_MISMATCH;
  cusparseStatus_t STATUS_MAPPING_ERROR = CUSPARSE_STATUS_MAPPING_ERROR;
  cusparseStatus_t STATUS_EXECUTION_FAILED = CUSPARSE_STATUS_EXECUTION_FAILED;
  cusparseStatus_t STATUS_INTERNAL_ERROR = CUSPARSE_STATUS_INTERNAL_ERROR;
  cusparseStatus_t STATUS_MATRIX_TYPE_NOT_SUPPORTED = CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  cusparseStatus_t STATUS_ZERO_PIVOT = CUSPARSE_STATUS_ZERO_PIVOT;

#if CUDA_VERSION >= 10010
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t;
  cusparseDnMatDescr_t dnMatDescr_t;

  // CHECK: hipsparseIndexType_t indexType_t;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_16U = HIPSPARSE_INDEX_16U;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_32I = HIPSPARSE_INDEX_32I;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_64I = HIPSPARSE_INDEX_64I;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t INDEX_16U = CUSPARSE_INDEX_16U;
  cusparseIndexType_t INDEX_32I = CUSPARSE_INDEX_32I;
  cusparseIndexType_t INDEX_64I = CUSPARSE_INDEX_64I;

  // CHECK: hipsparseFormat_t format_t;
  // CHECK-NEXT: hipsparseFormat_t FORMAT_CSR = HIPSPARSE_FORMAT_CSR;
  // CHECK-NEXT: hipsparseFormat_t FORMAT_CSC = HIPSPARSE_FORMAT_CSC;
  // CHECK-NEXT: hipsparseFormat_t FORMAT_CSO = HIPSPARSE_FORMAT_COO;
  cusparseFormat_t format_t;
  cusparseFormat_t FORMAT_CSR = CUSPARSE_FORMAT_CSR;
  cusparseFormat_t FORMAT_CSC = CUSPARSE_FORMAT_CSC;
  cusparseFormat_t FORMAT_CSO = CUSPARSE_FORMAT_COO;

  // CHECK: hipsparseOrder_t order_t;
  // CHECK-NEXT: hipsparseOrder_t ORDER_COL = HIPSPARSE_ORDER_COL;
  // CHECK-NEXT: hipsparseOrder_t ORDER_ROW = HIPSPARSE_ORDER_ROW;
  cusparseOrder_t order_t;
  cusparseOrder_t ORDER_COL = CUSPARSE_ORDER_COL;
  cusparseOrder_t ORDER_ROW = CUSPARSE_ORDER_ROW;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipsparseSpVecDescr_t spVecDescr_t;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: hipsparseDnVecDescr_t dnVecDescr_t;
  cusparseDnVecDescr_t dnVecDescr_t;

  // CHECK: hipsparseStatus_t STATUS_NOT_SUPPORTED = HIPSPARSE_STATUS_NOT_SUPPORTED;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;

  // CHECK: hipsparseSpMVAlg_t spMVAlg_t;
  cusparseSpMVAlg_t spMVAlg_t;
#endif

#if CUDA_VERSION >= 10020 && CUDA_VERSION < 12000
  // CHECK: hipsparseFormat_t FORMAT_COO_AOS = HIPSPARSE_FORMAT_COO_AOS;
  cusparseFormat_t FORMAT_COO_AOS = CUSPARSE_FORMAT_COO_AOS;

  // CHECK: hipsparseSpMVAlg_t MV_ALG_DEFAULT = HIPSPARSE_MV_ALG_DEFAULT;
  cusparseSpMVAlg_t MV_ALG_DEFAULT = CUSPARSE_MV_ALG_DEFAULT;

  // CHECK: hipsparseSpMVAlg_t COOMV_ALG = HIPSPARSE_COOMV_ALG;
  // CHECK-NEXT: hipsparseSpMVAlg_t CSRMV_ALG1 = HIPSPARSE_CSRMV_ALG1;
  // CHECK-NEXT: hipsparseSpMVAlg_t CSRMV_ALG2 = HIPSPARSE_CSRMV_ALG2;
  cusparseSpMVAlg_t COOMV_ALG = CUSPARSE_COOMV_ALG;
  cusparseSpMVAlg_t CSRMV_ALG1 = CUSPARSE_CSRMV_ALG1;
  cusparseSpMVAlg_t CSRMV_ALG2 = CUSPARSE_CSRMV_ALG2;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipsparseStatus_t STATUS_INSUFFICIENT_RESOURCES = HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
  cusparseStatus_t STATUS_INSUFFICIENT_RESOURCES = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
#endif

#if CUDA_VERSION < 11000
  // CHECK: hipsparseHybMat_t hybMat_t;
  cusparseHybMat_t hybMat_t;

  // CHECK: hipsparseHybPartition_t hybPartition_t;
  // CHECK-NEXT: hipsparseHybPartition_t HYB_PARTITION_AUTO = HIPSPARSE_HYB_PARTITION_AUTO;
  // CHECK-NEXT: hipsparseHybPartition_t HYB_PARTITION_USER = HIPSPARSE_HYB_PARTITION_USER;
  // CHECK-NEXT: hipsparseHybPartition_t HYB_PARTITION_MAX = HIPSPARSE_HYB_PARTITION_MAX;
  cusparseHybPartition_t hybPartition_t;
  cusparseHybPartition_t HYB_PARTITION_AUTO = CUSPARSE_HYB_PARTITION_AUTO;
  cusparseHybPartition_t HYB_PARTITION_USER = CUSPARSE_HYB_PARTITION_USER;
  cusparseHybPartition_t HYB_PARTITION_MAX = CUSPARSE_HYB_PARTITION_MAX;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: hipsparseFormat_t FORMAT_BLOCKED_ELL = HIPSPARSE_FORMAT_BLOCKED_ELL;
  cusparseFormat_t FORMAT_BLOCKED_ELL = CUSPARSE_FORMAT_BLOCKED_ELL;

  // CHECK: hipsparseSpMVAlg_t SPMV_ALG_DEFAULT = HIPSPARSE_SPMV_ALG_DEFAULT;
  // CHECK-NEXT: hipsparseSpMVAlg_t SPMV_COO_ALG1 = HIPSPARSE_SPMV_COO_ALG1;
  // CHECK-NEXT: hipsparseSpMVAlg_t SPMV_COO_ALG2 = HIPSPARSE_SPMV_COO_ALG2;
  // CHECK-NEXT: hipsparseSpMVAlg_t SPMV_CSR_ALG1 = HIPSPARSE_SPMV_CSR_ALG1;
  // CHECK-NEXT: hipsparseSpMVAlg_t SPMV_CSR_ALG2 = HIPSPARSE_SPMV_CSR_ALG2;
  cusparseSpMVAlg_t SPMV_ALG_DEFAULT = CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseSpMVAlg_t SPMV_COO_ALG1 = CUSPARSE_SPMV_COO_ALG1;
  cusparseSpMVAlg_t SPMV_COO_ALG2 = CUSPARSE_SPMV_COO_ALG2;
  cusparseSpMVAlg_t SPMV_CSR_ALG1 = CUSPARSE_SPMV_CSR_ALG1;
  cusparseSpMVAlg_t SPMV_CSR_ALG2 = CUSPARSE_SPMV_CSR_ALG2;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipsparseSpMatAttribute_t spMatAttribute_t;
  // CHECK-NEXT: hipsparseSpMatAttribute_t SPMAT_FILL_MODE = HIPSPARSE_SPMAT_FILL_MODE;
  // CHECK-NEXT: hipsparseSpMatAttribute_t SPMAT_DIAG_TYPE = HIPSPARSE_SPMAT_DIAG_TYPE;
  cusparseSpMatAttribute_t spMatAttribute_t;
  cusparseSpMatAttribute_t SPMAT_FILL_MODE = CUSPARSE_SPMAT_FILL_MODE;
  cusparseSpMatAttribute_t SPMAT_DIAG_TYPE = CUSPARSE_SPMAT_DIAG_TYPE;

  // CHECK: hipsparseSpSVAlg_t spSVAlg_t;
  // CHECK-NEXT: hipsparseSpSVAlg_t SPSV_ALG_DEFAULT = HIPSPARSE_SPSV_ALG_DEFAULT;
  cusparseSpSVAlg_t spSVAlg_t;
  cusparseSpSVAlg_t SPSV_ALG_DEFAULT = CUSPARSE_SPSV_ALG_DEFAULT;
#endif

  return 0;
}
