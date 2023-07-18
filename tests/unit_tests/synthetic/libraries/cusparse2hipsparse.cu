// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "hipsparse.h"

int main() {
  printf("17. cuSPARSE API to hipSPARSE API synthetic test\n");

  // CHECK: hipsparseHandle_t handle_t;
  cusparseHandle_t handle_t;

  // CHECK: hipsparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_C;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_C;

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
  int batchCount = 0;
  int m = 0;
  int n = 0;
  int mb = 0;
  int nb = 0;
  int nnzb = 0;
  int innz = 0;
  int blockDim = 0;
  int cscRowIndA = 0;
  int cscColPtrA = 0;
  int csrRowPtrA = 0;
  int csrColIndA = 0;
  int ncolors = 0;
  int coloring = 0;
  int reordering = 0;
  int bsrRowPtrA = 0;
  int bsrRowPtrC = 0;
  int csrRowPtrC = 0;
  int bsrColIndA = 0;
  int bsrColIndC = 0;
  int csrColIndC = 0;
  int rowBlockDimA = 0;
  int colBlockDimA = 0;
  int rowBlockDimC = 0;
  int colBlockDimC = 0;
  int bsrSortedRowPtrC = 0;
  int bsrSortedColIndC = 0;
  int bufferSizeInBytes = 0;
  int nnzTotalDevHostPtr = 0;
  int64_t size = 0;
  int64_t nnz = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t ellCols = 0;
  int64_t ellBlockSize = 0;
  int64_t batchStride = 0;
  int64_t offsetsBatchStride = 0;
  int64_t columnsValuesBatchStride = 0;
  int64_t ld = 0;
  void *indices = nullptr;
  void *values = nullptr;
  void *cooRowInd = nullptr;
  int icooRowInd = 0;
  void *cscRowInd = nullptr;
  void *csrColInd = nullptr;
  void *cooColInd = nullptr;
  void *ellColInd = nullptr;
  void *cooValues = nullptr;
  void *csrValues = nullptr;
  void *cscValues = nullptr;
  void *ellValue = nullptr;
  void *csrRowOffsets = nullptr;
  void *cscColOffsets = nullptr;
  void *cooRows = nullptr;
  int icooRows = 0;
  void *cooColumns = nullptr;
  int icooColumns = 0;
  void *data = nullptr;
  void *alpha = nullptr;
  void *beta = nullptr;
  void *pBuffer = nullptr;
  int *P = nullptr;
  void *tempBuffer = nullptr;
  void *c_coeff = nullptr;
  void *s_coeff = nullptr;
  size_t dataSize = 0;
  size_t bufferSize = 0;
  double dfractionToColor = 0.f;
  float ffractionToColor = 0.f;
  double bsrValA = 0.f;
  double csrValA = 0.f;
  float fcsrValA = 0.f;
  double csrValC = 0.f;
  float csrSortedValA = 0.f;
  double dbsrSortedValA = 0.f;
  double dbsrSortedValC = 0.f;
  float fbsrSortedValA = 0.f;
  float fbsrSortedValC = 0.f;
  float fcsrSortedValC = 0.f;
  double percentage = 0.f;

  pruneInfo_t prune_info;

  // CHECK: hipDoubleComplex dcomplex, dComplexbsrSortedValA, dComplexbsrSortedValC;
  cuDoubleComplex dcomplex, dComplexbsrSortedValA, dComplexbsrSortedValC;

  // CHECK: hipComplex complex, complexbsrValA, complexbsrSortedValC;
  cuComplex complex, complexbsrValA, complexbsrSortedValC;

  // CHECK: hipsparseOperation_t opA, opB;
  cusparseOperation_t opA, opB;

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, hipsparseColorInfo_t info);
  // CHECK: status_t = hipsparseZcsrcolor(handle_t, m, innz, matDescr_t, &dcomplex, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseZcsrcolor(handle_t, m, innz, matDescr_t, &dcomplex, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, hipsparseColorInfo_t info);
  // CHECK: status_t = hipsparseCcsrcolor(handle_t, m, innz, matDescr_t, &complex, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseCcsrcolor(handle_t, m, innz, matDescr_t, &complex, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, hipsparseColorInfo_t info);
  // CHECK: status_t = hipsparseDcsrcolor(handle_t, m, innz, matDescr_t, &csrValA, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseDcsrcolor(handle_t, m, innz, matDescr_t, &csrValA, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrcolor(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, hipsparseColorInfo_t info);
  // CHECK: status_t = hipsparseScsrcolor(handle_t, m, innz, matDescr_t, &csrSortedValA, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseScsrcolor(handle_t, m, innz, matDescr_t, &csrSortedValA, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA:cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const hipsparseMatDescr_t descrC, hipDoubleComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* buffer);
  // CHECK: status_t = hipsparseZgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, bsrRowPtrC, bsrColIndC, tempBuffer);
  status_t = cusparseZgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, bsrRowPtrC, bsrColIndC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC,int colBlockDimC, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* bufferSize);
  // CHECK: status_t = hipsparseZgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseZgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const hipsparseMatDescr_t descrC, hipComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* buffer);
  // CHECK: status_t = hipsparseCgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseCgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* bufferSize);
  // CHECK: status_t = hipsparseCgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseCgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const hipsparseMatDescr_t descrC, double* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* buffer);
  // CHECK: status_t = hipsparseDgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseDgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* bufferSize);
  // CHECK: status_t = hipsparseDgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseDgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const hipsparseMatDescr_t descrC, float* bsrValC, int* bsrRowPtrC, int* bsrColIndC, int rowBlockDimC, int colBlockDimC, void* buffer);
  // CHECK: status_t = hipsparseSgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseSgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* bufferSize);
  // CHECK: status_t = hipsparseSgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseSgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDimC, int colBlockDimC, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, int nnzb, const hipsparseMatDescr_t descrA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDimA, int colBlockDimA, const hipsparseMatDescr_t descrC, int* bsrRowPtrC, int rowBlockDimC, int colBlockDimC, int* nnzTotalDevHostPtr, void* buffer);
  // CHECK: status_t = hipsparseXgebsr2gebsrNnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC,  rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);
  status_t = cusparseXgebsr2gebsrNnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC,  rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgebsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseZgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgebsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const hipsparseMatDescr_t descrC, hipComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseCgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgebsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const hipsparseMatDescr_t descrC, double* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseDgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgebsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int rowBlockDim, int colBlockDim, const hipsparseMatDescr_t descrC, float* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseSgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseSgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseZbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipsparseMatDescr_t descrC, hipComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseCbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipsparseMatDescr_t descrC, double* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseDbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsr2csr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nb, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipsparseMatDescr_t descrC, float* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseSbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseSbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle, int m, int n, int nnz, int* cooRows, int* cooCols, int* P, void* pBuffer);
  // CHECK: status_t = hipsparseXcoosortByColumn(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);
  status_t = cusparseXcoosortByColumn(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle, int m, int n, int nnz, int* cooRows, int* cooCols, int* P, void* pBuffer);
  // CHECK: status_t = hipsparseXcoosortByRow(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);
  status_t = cusparseXcoosortByRow(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowsA, const int* cooColsA, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, const int* cooRows, const int* cooCols, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseXcoosort_bufferSizeExt(handle_t, m, n, innz, &icooRows, &icooColumns, &bufferSize);
  status_t = cusparseXcoosort_bufferSizeExt(handle_t, m, n, innz, &icooRows, &icooColumns, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* cscColPtrA, int* cscRowIndA, int* P, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, const int* cscColPtr, int* cscRowInd, int* P, void* pBuffer);
  // CHECK: status_t = hipsparseXcscsort(handle_t, m, n, innz, matDescr_A, &cscColPtrA, &cscRowIndA, P, pBuffer);
  status_t = cusparseXcscsort(handle_t, m, n, innz, matDescr_A, &cscColPtrA, &cscRowIndA, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cscColPtrA, const int* cscRowIndA, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, const int* cscColPtr, const int* cscRowInd, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseXcscsort_bufferSizeExt(handle_t, m, n, innz, &cscColPtrA, &cscRowIndA, &bufferSize);
  status_t = cusparseXcscsort_bufferSizeExt(handle_t, m, n, innz, &cscColPtrA, &cscRowIndA, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, const int* csrRowPtr, int* csrColInd, int* P, void* pBuffer);
  // CHECK: status_t = hipsparseXcsrsort(handle_t, m, n, innz, matDescr_A, &cscRowIndA, &cscColPtrA, P, pBuffer);
  status_t = cusparseXcsrsort(handle_t, m, n, innz, matDescr_A, &cscRowIndA, &cscColPtrA, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* csrRowPtrA, const int* csrColIndA, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, const int* csrRowPtr, const int* csrColInd, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseXcsrsort_bufferSizeExt(handle_t, m, n, innz, &cscRowIndA, &cscColPtrA, &bufferSize);
  status_t = cusparseXcsrsort_bufferSizeExt(handle_t, m, n, innz, &cscRowIndA, &cscColPtrA, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p);
  // CHECK: status_t = hipsparseCreateIdentityPermutation(handle_t, n, P);
  status_t = cusparseCreateIdentityPermutation(handle_t, n, P);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr, cusparseIndexBase_t idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrRowPtr, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseXcoo2csr(handle_t, &icooRowInd, nnz, m, &csrRowPtrA, indexBase_t);
  status_t = cusparseXcoo2csr(handle_t, &icooRowInd, nnz, m, &csrRowPtrA, indexBase_t);

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 9000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double percentage, const hipsparseMatDescr_t descrC, double* csrValC, const int* csrRowPtrC, int* csrColIndC, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseDpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src);
  // CHECK: status_t = hipsparseCopyMatDescr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);
#endif

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t, matC;
  cusparseSpMatDescr_t spMatDescr_t, matC;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t, matA, matB;
  cusparseDnMatDescr_t dnMatDescr_t, matA, matB;

  // CHECK: hipsparseIndexType_t indexType_t;
  // CHECK-NEXT: hipsparseIndexType_t csrRowOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscColOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscRowIndType;
  // CHECK-NEXT: hipsparseIndexType_t csrColIndType;
  // CHECK-NEXT: hipsparseIndexType_t ellIdxType;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_16U = HIPSPARSE_INDEX_16U;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_32I = HIPSPARSE_INDEX_32I;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_64I = HIPSPARSE_INDEX_64I;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t cscColOffsetsType;
  cusparseIndexType_t cscRowIndType;
  cusparseIndexType_t csrColIndType;
  cusparseIndexType_t ellIdxType;
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr);
  // CHECK: status_t = hipsparseDestroySpMat(spMatDescr_t);
  status_t = cusparseDestroySpMat(spMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr, hipsparseFormat_t* format);
  // CHECK: status_t = hipsparseSpMatGetFormat(spMatDescr_t, &format_t);
  status_t = cusparseSpMatGetFormat(spMatDescr_t, &format_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpMatGetIndexBase(spMatDescr_t, &indexBase_t);
  status_t = cusparseSpMatGetIndexBase(spMatDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, hipDataType valueType, hipsparseOrder_t order);
  // CHECK: status_t = hipsparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr);
  // CHECK: status_t = hipsparseDestroyDnMat(dnMatDescr_t);
  status_t = cusparseDestroyDnMat(dnMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, hipDataType* valueType, hipsparseOrder_t* order);
  // CHECK: status_t = hipsparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);
  status_t = cusparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // CHECK: status_t = hipsparseDnMatGetStridedBatch(dnMatDescr_t, &batchCount, &batchStride);
  status_t = cusparseDnMatGetStridedBatch(dnMatDescr_t, &batchCount, &batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // CHECK: status_t = hipsparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);
  status_t = cusparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipsparseCsr2CscAlg_t Csr2CscAlg_t;
  // CHECK-NEXT: hipsparseCsr2CscAlg_t CSR2CSC_ALG1 = HIPSPARSE_CSR2CSC_ALG1;
  cusparseCsr2CscAlg_t Csr2CscAlg_t;
  cusparseCsr2CscAlg_t CSR2CSC_ALG1 = CUSPARSE_CSR2CSC_ALG1;
#endif

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000)
  // CHECK: hipsparseSpMMAlg_t COOMM_ALG1 = HIPSPARSE_COOMM_ALG1;
  // CHECK-NEXT: hipsparseSpMMAlg_t COOMM_ALG2 = HIPSPARSE_COOMM_ALG2;
  // CHECK-NEXT: hipsparseSpMMAlg_t COOMM_ALG3 = HIPSPARSE_COOMM_ALG3;
  cusparseSpMMAlg_t COOMM_ALG1 = CUSPARSE_COOMM_ALG1;
  cusparseSpMMAlg_t COOMM_ALG2 = CUSPARSE_COOMM_ALG2;
  cusparseSpMMAlg_t COOMM_ALG3 = CUSPARSE_COOMM_ALG3;
#endif

#if CUDA_VERSION >= 10010 && CUDA_VERSION < 12000
  // CHECK: hipsparseCsr2CscAlg_t CSR2CSC_ALG2 = HIPSPARSE_CSR2CSC_ALG2;
  cusparseCsr2CscAlg_t CSR2CSC_ALG2 = CUSPARSE_CSR2CSC_ALG2;
#endif

#if (CUDA_VERSION >= 10020 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: hipsparseSpVecDescr_t spVecDescr_t;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: hipsparseDnVecDescr_t dnVecDescr_t, vecX, vecY;
  cusparseDnVecDescr_t dnVecDescr_t, vecX, vecY;

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsrGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, hipsparseIndexType_t* csrRowOffsetsType, hipsparseIndexType_t* csrColIndType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseCsrGet(spMatDescr_t, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);
  status_t = cusparseCsrGet(spMatDescr_t, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values);
  // CHECK: status_t = hipsparseSpMatGetValues(spMatDescr_t, &values);
  status_t = cusparseSpMatGetValues(spMatDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values);
  // CHECK: status_t = hipsparseSpMatSetValues(spMatDescr_t, values);
  status_t = cusparseSpMatSetValues(spMatDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int* batchCount);
  // CHECK: status_t = hipsparseSpMatGetStridedBatch(spMatDescr_t, &batchCount);
  status_t = cusparseSpMatGetStridedBatch(spMatDescr_t, &batchCount);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateDnVec(&dnVecDescr_t, size, values, dataType);
  status_t = cusparseCreateDnVec(&dnVecDescr_t, size, values, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr);
  // CHECK: status_t = hipsparseDestroyDnVec(dnVecDescr_t);
  status_t = cusparseDestroyDnVec(dnVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, hipDataType* valueType);
  // CHECK: status_t = hipsparseDnVecGet(dnVecDescr_t, &size, &values, &dataType);
  status_t = cusparseDnVecGet(dnVecDescr_t, &size, &values, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values);
  // CHECK: status_t = hipsparseDnVecGetValues(dnVecDescr_t, &values);
  status_t = cusparseDnVecGetValues(dnVecDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values);
  // CHECK: status_t = hipsparseDnVecSetValues(dnVecDescr_t, values);
  status_t = cusparseDnVecSetValues(dnVecDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values);
  // CHECK: status_t = hipsparseDnMatGetValues(dnMatDescr_t, &values);
  status_t = cusparseDnMatGetValues(dnMatDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values);
  // CHECK: status_t = hipsparseDnMatSetValues(dnMatDescr_t, values);
  status_t = cusparseDnMatSetValues(dnMatDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t vecX, const void* beta, const hipsparseDnVecDescr_t vecY, hipDataType computeType, hipsparseSpMVAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMV(handle_t, opA, alpha, spMatDescr_t, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);
  status_t = cusparseSpMV(handle_t, opA, alpha, spMatDescr_t, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipsparseStatus_t STATUS_NOT_SUPPORTED = HIPSPARSE_STATUS_NOT_SUPPORTED;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;
#endif

#if (CUDA_VERSION >= 10020 && CUDA_VERSION < 11000 && !defined(_WIN32)) || (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000)
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
  // CHECK: status_t = hipsparseCreateCooAoS(&spMatDescr_t, rows, cols, nnz, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateCooAoS(&spMatDescr_t, rows, cols, nnz, cooColInd, cooValues, indexType_t, indexBase_t, dataType);

  // CUDA: CUSPARSE_DEPRECATED(cusparseCooGet) cusparseStatus_t CUSPARSEAPI cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd, void** cooValues, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseCooAoSGet(spMatDescr_t, &rows, &cols, &nnz, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooAoSGet(spMatDescr_t, &rows, &cols, &nnz, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount);
  // CHECK: status_t = hipsparseSpMatSetStridedBatch(spMatDescr_t, batchCount);
  status_t = cusparseSpMatSetStridedBatch(spMatDescr_t, batchCount);
#endif

#if CUDA_VERSION >= 11000 && CUSPARSE_VERSION >= 11100
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCooSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride);
  // CHECK: status_t = hipsparseCooSetStridedBatch(spMatDescr_t, batchCount, batchStride);
  status_t = cusparseCooSetStridedBatch(spMatDescr_t, batchCount, batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsrSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride);
  // CHECK: status_t = hipsparseCsrSetStridedBatch(spMatDescr_t, batchCount, offsetsBatchStride, columnsValuesBatchStride);
  status_t = cusparseCsrSetStridedBatch(spMatDescr_t, batchCount, offsetsBatchStride, columnsValuesBatchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseRot(cusparseHandle_t handle, const void* c_coeff, const void* s_coeff, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseRot(hipsparseHandle_t handle, const void* c_coeff, const void* s_coeff, hipsparseSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseRot(handle_t, c_coeff, s_coeff, spVecDescr_t, vecY);
  status_t = cusparseRot(handle_t, c_coeff, s_coeff, spVecDescr_t, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScatter(hipsparseHandle_t handle, hipsparseSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseScatter(handle_t, spVecDescr_t, vecY);
  status_t = cusparseScatter(handle_t, spVecDescr_t, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGather(hipsparseHandle_t handle, hipsparseDnVecDescr_t vecY, hipsparseSpVecDescr_t vecX);
  // CHECK: status_t = hipsparseGather(handle_t, vecY, spVecDescr_t);
  status_t = cusparseGather(handle_t, vecY, spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t handle, const void* alpha, hipsparseSpVecDescr_t vecX, const void* beta, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseAxpby(handle_t, alpha, spVecDescr_t, beta, vecY);
  status_t = cusparseAxpby(handle_t, alpha, spVecDescr_t, beta, vecY);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipsparseStatus_t STATUS_INSUFFICIENT_RESOURCES = HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
  cusparseStatus_t STATUS_INSUFFICIENT_RESOURCES = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;

  // CHECK: hipsparseSpGEMMAlg_t spGEMMAlg_t;
  // CHECK-NEXT: hipsparseSpGEMMAlg_t SPGEMM_DEFAULT = HIPSPARSE_SPGEMM_DEFAULT;
  cusparseSpGEMMAlg_t spGEMMAlg_t;
  cusparseSpGEMMAlg_t SPGEMM_DEFAULT = CUSPARSE_SPGEMM_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);
  // CHECK: status_t = hipsparseCsrSetPointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);
  status_t = cusparseCsrSetPointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // CHECK: status_t = hipsparseSpMatGetSize(spMatDescr_t, &rows, &cols, &nnz);
  status_t = cusparseSpMatGetSize(spMatDescr_t, &rows, &cols, &nnz);
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

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZhyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, cuDoubleComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t handle, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, hipDoubleComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // CHECK: status_t = hipsparseZhyb2csr(handle_t, matDescr_t, hybMat_t, &dComplexbsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseZhyb2csr(handle_t, matDescr_t, hybMat_t, &dComplexbsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseChyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, cuComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t handle, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, hipComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // CHECK: status_t = hipsparseChyb2csr(handle_t, matDescr_t, hybMat_t, &complex, &csrRowPtrA, &csrColIndA);
  status_t = cusparseChyb2csr(handle_t, matDescr_t, hybMat_t, &complex, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDhyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, double* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t handle, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, double* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // CHECK: status_t = hipsparseDhyb2csr(handle_t, matDescr_t, hybMat_t, &csrValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseDhyb2csr(handle_t, matDescr_t, hybMat_t, &csrValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseShyb2csr(cusparseHandle_t handle, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, float* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t handle, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, float* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // CHECK: status_t = hipsparseShyb2csr(handle_t, matDescr_t, hybMat_t, &fcsrValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseShyb2csr(handle_t, matDescr_t, hybMat_t, &fcsrValA, &csrRowPtrA, &csrColIndA);
#endif

#if CUDA_VERSION >= 11010 && CUSPARSE_VERSION >= 11300
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCooSetPointers(hipsparseSpMatDescr_t spMatDescr, void* cooRowInd, void* cooColInd, void* cooValues);
  // CHECK: status_t = hipsparseCooSetPointers(spMatDescr_t, cooRows, cooColumns, cooValues);
  status_t = cusparseCooSetPointers(spMatDescr_t, cooRows, cooColumns, cooValues);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCscSetPointers(hipsparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues);
  // CHECK: status_t = hipsparseCscSetPointers(spMatDescr_t, cscColOffsets, cscRowInd, cscValues);
  status_t = cusparseCscSetPointers(spMatDescr_t, cscColOffsets, cscRowInd, cscValues);
#endif

#if CUDA_VERSION >= 11020 && CUSPARSE_VERSION >= 11400
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateBlockedEll(hipsparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, hipsparseIndexType_t ellIdxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateBlockedEll(&spMatDescr_t, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);
  status_t = cusparseCreateBlockedEll(&spMatDescr_t, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseBlockedEllGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, hipsparseIndexType_t* ellIdxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseBlockedEllGet(spMatDescr_t, &rows, &cols, &ellBlockSize, &ellCols, &ellColInd, &ellValue, &ellIdxType, &indexBase_t, &dataType);
  status_t = cusparseBlockedEllGet(spMatDescr_t, &rows, &cols, &ellBlockSize, &ellCols, &ellColInd, &ellValue, &ellIdxType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM_preprocess(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
 status_t = cusparseSDDMM_preprocess(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSDDMM_bufferSize(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, &bufferSize);
 status_t = cusparseSDDMM_bufferSize(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
 status_t = cusparseSDDMM(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // HIP: hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t spMatDescr, hipsparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // CHECK: status_t = hipsparseSpMatGetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatGetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t spMatDescr, hipsparseSpMatAttribute_t attribute, const void* data, size_t dataSize);
  // CHECK: status_t = hipsparseSpMatSetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatSetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
#endif

#if CUDA_VERSION >= 11030 && CUSPARSE_VERSION >= 11600
  // CHECK: hipsparseSpSMAlg_t spSMAlg_t;
  // CHECK-NEXT: hipsparseSpSMAlg_t SPSM_ALG_DEFAULT = HIPSPARSE_SPSM_ALG_DEFAULT;
  cusparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t SPSM_ALG_DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT;

  // CHECK: hipsparseSpGEMMAlg_t SPGEMM_CSR_ALG_DETERMINITIC = HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC;
  // CHECK-NEXT: hipsparseSpGEMMAlg_t SPGEMM_CSR_ALG_NONDETERMINITIC = HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC;
  cusparseSpGEMMAlg_t SPGEMM_CSR_ALG_DETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
  cusparseSpGEMMAlg_t SPGEMM_CSR_ALG_NONDETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: hipsparseCsr2CscAlg_t CSR2CSC_ALG_DEFAULT = HIPSPARSE_CSR2CSC_ALG_DEFAULT;
  cusparseCsr2CscAlg_t CSR2CSC_ALG_DEFAULT = CUSPARSE_CSR2CSC_ALG_DEFAULT;

  // CHECK: hipsparseSpGEMMAlg_t SPGEMM_ALG1 = HIPSPARSE_SPGEMM_ALG1;
  // CHECK: hipsparseSpGEMMAlg_t SPGEMM_ALG2 = HIPSPARSE_SPGEMM_ALG2;
  // CHECK: hipsparseSpGEMMAlg_t SPGEMM_ALG3 = HIPSPARSE_SPGEMM_ALG3;
  cusparseSpGEMMAlg_t SPGEMM_ALG1 = CUSPARSE_SPGEMM_ALG1;
  cusparseSpGEMMAlg_t SPGEMM_ALG2 = CUSPARSE_SPGEMM_ALG2;
  cusparseSpGEMMAlg_t SPGEMM_ALG3 = CUSPARSE_SPGEMM_ALG3;
#endif

  return 0;
}
