// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --use-hip-data-types %clang_args -ferror-limit=500

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

  // CHECK: hipsparseMatDescr_t matDescr_t, matDescr_t_2;
  cusparseMatDescr_t matDescr_t, matDescr_t_2;

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

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  int iVal = 0;
  int64_t size = 0;
  int64_t nnz = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  void *indices = nullptr;
  void *values = nullptr;
  void *cooRowInd = nullptr;
  void *cscRowInd = nullptr;
  void *csrColInd = nullptr;
  void *cooColInd = nullptr;
  void *cooValues = nullptr;
  void *csrValues = nullptr;
  void *cscValues = nullptr;
  void *csrRowOffsets = nullptr;
  void *cscColOffsets = nullptr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t* handle);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle);
  // CHECK: status_t = hipsparseCreate(&handle_t);
  status_t = cusparseCreate(&handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle);
  // CHECK: status_t = hipsparseDestroy(handle_t);
  status_t = cusparseDestroy(handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId);
  // CHECK: status_t = hipsparseSetStream(handle_t, stream_t);
  status_t = cusparseSetStream(handle_t, stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId);
  // CHECK: status_t = hipsparseGetStream(handle_t, &stream_t);
  status_t = cusparseGetStream(handle_t, &stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode);
  // CHECK: status_t = hipsparseSetPointerMode(handle_t, pointerMode_t);
  status_t = cusparseSetPointerMode(handle_t, pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode);
  // CHECK: status_t = hipsparseGetPointerMode(handle_t, &pointerMode_t);
  status_t = cusparseGetPointerMode(handle_t, &pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int* version);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version);
  // CHECK: status_t = hipsparseGetVersion(handle_t, &iVal);
  status_t = cusparseGetVersion(handle_t, &iVal);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA);
  // CHECK: status_t = hipsparseCreateMatDescr(&matDescr_t);
  status_t = cusparseCreateMatDescr(&matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA);
  // CHECK: status_t = hipsparseDestroyMatDescr(matDescr_t);
  status_t = cusparseDestroyMatDescr(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base);
  // CHECK: status_t = hipsparseSetMatIndexBase(matDescr_t, indexBase_t);
  status_t = cusparseSetMatIndexBase(matDescr_t, indexBase_t);

  // CUDA: cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA);
  // CHECK: indexBase_t = hipsparseGetMatIndexBase(matDescr_t);
  indexBase_t = cusparseGetMatIndexBase(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type);
  // CHECK: status_t = hipsparseSetMatType(matDescr_t, matrixType_t);
  status_t = cusparseSetMatType(matDescr_t, matrixType_t);

  // CUDA: cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA);
  // CHECK: matrixType_t = hipsparseGetMatType(matDescr_t);
  matrixType_t = cusparseGetMatType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode);
  // CHECK: status_t = hipsparseSetMatFillMode(matDescr_t, fillMode_t);
  status_t = cusparseSetMatFillMode(matDescr_t, fillMode_t);

  // CUDA: cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA);
  // CHECK: fillMode_t = hipsparseGetMatFillMode(matDescr_t);
  fillMode_t = cusparseGetMatFillMode(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType);
  // CHECK: status_t = hipsparseSetMatDiagType(matDescr_t, diagType_t);
  status_t = cusparseSetMatDiagType(matDescr_t, diagType_t);

  // CUDA: cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);
  // HIP: HIPSPARSE_EXPORT hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA);
  // CHECK: diagType_t = hipsparseGetMatDiagType(matDescr_t);
  diagType_t = cusparseGetMatDiagType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info);
  // CHECK: status_t = hipsparseCreateColorInfo(&colorInfo_t);
  status_t = cusparseCreateColorInfo(&colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info);
  // CHECK: status_t = hipsparseDestroyColorInfo(colorInfo_t);
  status_t = cusparseDestroyColorInfo(colorInfo_t);

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src);
  // CHECK: status_t = hipsparseCopyMatDescr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t;
  cusparseDnMatDescr_t dnMatDescr_t;

  // CHECK: hipsparseIndexType_t indexType_t;
  // CHECK-NEXT: hipsparseIndexType_t csrRowOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscColOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscRowIndType;
  // CHECK-NEXT: hipsparseIndexType_t csrColIndType;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_16U = HIPSPARSE_INDEX_16U;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_32I = HIPSPARSE_INDEX_32I;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_64I = HIPSPARSE_INDEX_64I;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t cscColOffsetsType;
  cusparseIndexType_t cscRowIndType;
  cusparseIndexType_t csrColIndType;
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

  // CHECK: hipsparseSpMMAlg_t spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t ows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, hipsparseIndexType_t cooIdxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateCoo(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateCoo(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
#endif

#if CUDA_VERSION >= 10010 && CUDA_VERSION < 12000
  // CHECK: hipsparseSpMMAlg_t COOMM_ALG1 = HIPSPARSE_COOMM_ALG1;
  // CHECK-NEXT: hipsparseSpMMAlg_t COOMM_ALG2 = HIPSPARSE_COOMM_ALG2;
  // CHECK-NEXT: hipsparseSpMMAlg_t COOMM_ALG3 = HIPSPARSE_COOMM_ALG3;
  cusparseSpMMAlg_t COOMM_ALG1 = CUSPARSE_COOMM_ALG1;
  cusparseSpMMAlg_t COOMM_ALG2 = CUSPARSE_COOMM_ALG2;
  cusparseSpMMAlg_t COOMM_ALG3 = CUSPARSE_COOMM_ALG3;
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, hipsparseIndexType_t idxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateSpVec(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateSpVec(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr);
  // HIP: hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr);
  // CHECK: status_t = hipsparseDestroySpVec(spVecDescr_t);
  status_t = cusparseDestroySpVec(spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values);
  // CHECK: status_t = hipsparseSpVecGetValues(spVecDescr_t, &values);
  status_t = cusparseSpVecGetValues(spVecDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values);
  // CHECK: status_t = hipsparseSpVecSetValues(spVecDescr_t, values);
  status_t = cusparseSpVecSetValues(spVecDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, hipsparseIndexType_t csrRowOffsetsType, hipsparseIndexType_t csrColIndType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateCsr(&spMatDescr_t, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateCsr(&spMatDescr_t, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);
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

  // CHECK: hipsparseSpMMAlg_t MM_ALG_DEFAULT = HIPSPARSE_MM_ALG_DEFAULT;
  // CHECK: hipsparseSpMMAlg_t CSRMM_ALG1 = HIPSPARSE_CSRMM_ALG1;
  cusparseSpMMAlg_t MM_ALG_DEFAULT = CUSPARSE_MM_ALG_DEFAULT;
  cusparseSpMMAlg_t CSRMM_ALG1 = CUSPARSE_CSRMM_ALG1;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, hipsparseIndexType_t cooIdxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateCooAoS(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateCooAoS(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipsparseStatus_t STATUS_INSUFFICIENT_RESOURCES = HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
  cusparseStatus_t STATUS_INSUFFICIENT_RESOURCES = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;

  // CHECK: hipsparseSpMMAlg_t SPMM_ALG_DEFAULT = HIPSPARSE_SPMM_ALG_DEFAULT;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_COO_ALG1 = HIPSPARSE_SPMM_COO_ALG1;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_COO_ALG2 = HIPSPARSE_SPMM_COO_ALG2;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_COO_ALG3 = HIPSPARSE_SPMM_COO_ALG3;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_CSR_ALG1 = HIPSPARSE_SPMM_CSR_ALG1;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_COO_ALG4 = HIPSPARSE_SPMM_COO_ALG4;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_CSR_ALG2 = HIPSPARSE_SPMM_CSR_ALG2;
  cusparseSpMMAlg_t SPMM_ALG_DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT;
  cusparseSpMMAlg_t SPMM_COO_ALG1 = CUSPARSE_SPMM_COO_ALG1;
  cusparseSpMMAlg_t SPMM_COO_ALG2 = CUSPARSE_SPMM_COO_ALG2;
  cusparseSpMMAlg_t SPMM_COO_ALG3 = CUSPARSE_SPMM_COO_ALG3;
  cusparseSpMMAlg_t SPMM_CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1;
  cusparseSpMMAlg_t SPMM_COO_ALG4 = CUSPARSE_SPMM_COO_ALG4;
  cusparseSpMMAlg_t SPMM_CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2;

  // CHECK: hipsparseSpGEMMAlg_t spGEMMAlg_t;
  // CHECK-NEXT: hipsparseSpGEMMAlg_t SPGEMM_DEFAULT = HIPSPARSE_SPGEMM_DEFAULT;
  cusparseSpGEMMAlg_t spGEMMAlg_t;
  cusparseSpGEMMAlg_t SPGEMM_DEFAULT = CUSPARSE_SPGEMM_DEFAULT;
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t* hybA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA);
  // CHECK: status_t = hipsparseCreateHybMat(&hybMat_t);
  status_t = cusparseCreateHybMat(&hybMat_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA);
  // CHECK: status_t = hipsparseDestroyHybMat(hybMat_t);
  status_t = cusparseDestroyHybMat(hybMat_t);
#endif

#if CUDA_VERSION >= 11010
  // CHECK: hipsparseSparseToDenseAlg_t sparseToDenseAlg_t;
  // CHECK-NEXT: hipsparseSparseToDenseAlg_t SPARSETODENSE_ALG_DEFAULT = HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
  cusparseSparseToDenseAlg_t sparseToDenseAlg_t;
  cusparseSparseToDenseAlg_t SPARSETODENSE_ALG_DEFAULT = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

  // CHECK: hipsparseDenseToSparseAlg_t denseToSparseAlg_t;
  // CHECK-NEXT: hipsparseDenseToSparseAlg_t DENSETOSPARSE_ALG_DEFAULT = HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
  cusparseDenseToSparseAlg_t denseToSparseAlg_t;
  cusparseDenseToSparseAlg_t DENSETOSPARSE_ALG_DEFAULT = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, hipsparseIndexType_t cscColOffsetsType, hipsparseIndexType_t cscRowIndType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateCsc(&spMatDescr_t, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateCsc(&spMatDescr_t, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, csrColIndType, indexBase_t, dataType);
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

  // CHECK: hipsparseSpMMAlg_t SPMM_CSR_ALG3 = HIPSPARSE_SPMM_CSR_ALG3;
  // CHECK-NEXT: hipsparseSpMMAlg_t SPMM_BLOCKED_ELL_ALG1 = HIPSPARSE_SPMM_BLOCKED_ELL_ALG1;
  cusparseSpMMAlg_t SPMM_CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3;
  cusparseSpMMAlg_t SPMM_BLOCKED_ELL_ALG1 = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;

  // CHECK: hipsparseSDDMMAlg_t sDDMMAlg_t;
  // CHECK-NEXT: hipsparseSDDMMAlg_t SDDMM_ALG_DEFAULT = HIPSPARSE_SDDMM_ALG_DEFAULT;
  cusparseSDDMMAlg_t sDDMMAlg_t;
  cusparseSDDMMAlg_t SDDMM_ALG_DEFAULT = CUSPARSE_SDDMM_ALG_DEFAULT;
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

  // CHECK: hipsparseSpSMAlg_t spSMAlg_t;
  // CHECK-NEXT: hipsparseSpSMAlg_t SPSM_ALG_DEFAULT = HIPSPARSE_SPSM_ALG_DEFAULT;
  cusparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t SPSM_ALG_DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT;
#endif

  return 0;
}
