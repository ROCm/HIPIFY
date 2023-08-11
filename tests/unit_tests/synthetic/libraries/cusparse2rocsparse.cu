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
  printf("18. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_C;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_C;

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

  // CHECK: rocsparse_action action_t, copyValues;
  // CHECK-NEXT: rocsparse_action ACTION_SYMBOLIC = rocsparse_action_symbolic;
  // CHECK-NEXT: rocsparse_action ACTION_NUMERIC = rocsparse_action_numeric;
  cusparseAction_t action_t, copyValues;
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

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  int iVal = 0;
  int batchCount = 0;
  int m = 0;
  int n = 0;
  int mb = 0;
  int nb = 0;
  int nnza = 0;
  int nnzb = 0;
  int nnzPerRow = 0;
  int innz = 0;
  int blockDim = 0;
  int csrSortedRowPtr = 0;
  int csrSortedColInd = 0;
  int cscRowIndA = 0;
  int cscColPtrA = 0;
  int csrRowPtrA = 0;
  int csrColIndA = 0;
  int ncolors = 0;
  int coloring = 0;
  int reordering = 0;
  int bscRowInd = 0;
  int bsrRowPtrA = 0;
  int bsrRowPtrC = 0;
  int csrRowPtrC = 0;
  int bscColPtr = 0;
  int bsrColIndA = 0;
  int bsrColIndC = 0;
  int csrColIndC = 0;
  int rowBlockDim = 0;
  int rowBlockDimA = 0;
  int colBlockDimA = 0;
  int rowBlockDimC = 0;
  int colBlockDim = 0;
  int colBlockDimC = 0;
  int bsrSortedRowPtr = 0;
  int bsrSortedRowPtrC = 0;
  int bsrSortedColInd = 0;
  int bsrSortedColIndC = 0;
  int bufferSizeInBytes = 0;
  int nnzTotalDevHostPtr = 0;
  int userEllWidth = 0;
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
  float fcsrValC = 0.f;
  float csrSortedVal = 0.f;
  float cscSortedVal = 0.f;
  float csrSortedValA = 0.f;
  double dcsrSortedVal = 0.f;
  double dcscSortedVal = 0.f;
  double dcsrSortedValA = 0.f;
  double dbsrSortedVal = 0.f;
  double dbsrSortedValA = 0.f;
  double dbsrSortedValC = 0.f;
  float fbsrSortedVal = 0.f;
  float fbsrSortedValA = 0.f;
  float fbsrSortedValC = 0.f;
  float fcsrSortedValC = 0.f;
  double dcsrSortedValC = 0.f;
  double percentage = 0.f;
  double dthreshold = 0.f;
  float fthreshold = 0.f;
  double dtol = 0.f;
  float ftol = 0.f;
  double dbscVal = 0.f;
  float fbscVal = 0.f;

  // CHECK: rocsparse_mat_info prune_info;
  pruneInfo_t prune_info;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal;
  cuDoubleComplex dcomplex, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal;
  cuComplex complex, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal;

  // CHECK: rocsparse_operation opA, opB;
  cusparseOperation_t opA, opB;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t* handle);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_handle(rocsparse_handle* handle);
  // CHECK: status_t = rocsparse_create_handle(&handle_t);
  status_t = cusparseCreate(&handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle);
  // CHECK: status_t = rocsparse_destroy_handle(handle_t);
  status_t = cusparseDestroy(handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream);
  // CHECK: status_t = rocsparse_set_stream(handle_t, stream_t);
  status_t = cusparseSetStream(handle_t, stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream);
  // CHECK: status_t = rocsparse_get_stream(handle_t, &stream_t);
  status_t = cusparseGetStream(handle_t, &stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode pointer_mode);
  // CHECK: status_t = rocsparse_set_pointer_mode(handle_t, pointerMode_t);
  status_t = cusparseSetPointerMode(handle_t, pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode* pointer_mode);
  // CHECK: status_t = rocsparse_get_pointer_mode(handle_t, &pointerMode_t);
  status_t = cusparseGetPointerMode(handle_t, &pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int* version);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version);
  // CHECK: status_t = rocsparse_get_version(handle_t, &iVal);
  status_t = cusparseGetVersion(handle_t, &iVal);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr);
  // CHECK: status_t = rocsparse_create_mat_descr(&matDescr_t);
  status_t = cusparseCreateMatDescr(&matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr);
  // CHECK: status_t = rocsparse_destroy_mat_descr(matDescr_t);
  status_t = cusparseDestroyMatDescr(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base);
  // CHECK: status_t = rocsparse_set_mat_index_base(matDescr_t, indexBase_t);
  status_t = cusparseSetMatIndexBase(matDescr_t, indexBase_t);

  // CUDA: cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr);
  // CHECK: indexBase_t = rocsparse_get_mat_index_base(matDescr_t);
  indexBase_t = cusparseGetMatIndexBase(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type);
  // CHECK: status_t = rocsparse_set_mat_type(matDescr_t, matrixType_t);
  status_t = cusparseSetMatType(matDescr_t, matrixType_t);

  // CUDA: cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr);
  // CHECK: matrixType_t = rocsparse_get_mat_type(matDescr_t);
  matrixType_t = cusparseGetMatType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr, rocsparse_fill_mode fill_mode);
  // CHECK: status_t = rocsparse_set_mat_fill_mode(matDescr_t, fillMode_t);
  status_t = cusparseSetMatFillMode(matDescr_t, fillMode_t);

  // CUDA: cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr);
  // CHECK: fillMode_t = rocsparse_get_mat_fill_mode(matDescr_t);
  fillMode_t = cusparseGetMatFillMode(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr, rocsparse_diag_type diag_type);
  // CHECK: status_t = rocsparse_set_mat_diag_type(matDescr_t, diagType_t);
  status_t = cusparseSetMatDiagType(matDescr_t, diagType_t);

  // CUDA: cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr);
  // CHECK: diagType_t = rocsparse_get_mat_diag_type(matDescr_t);
  diagType_t = cusparseGetMatDiagType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info);
  // CHECK: status_t = rocsparse_create_color_info(&colorInfo_t);
  status_t = cusparseCreateColorInfo(&colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info);
  // CHECK: status_t = rocsparse_destroy_color_info(colorInfo_t);
  status_t = cusparseDestroyColorInfo(colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrcolor(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* fraction_to_color, rocsparse_int* ncolors, rocsparse_int* coloring, rocsparse_int* reordering, rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_zcsrcolor(handle_t, m, innz, matDescr_t, &dcomplex, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseZcsrcolor(handle_t, m, innz, matDescr_t, &dcomplex, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrcolor(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* fraction_to_color, rocsparse_int* ncolors, rocsparse_int* coloring, rocsparse_int* reordering, rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_ccsrcolor(handle_t, m, innz, matDescr_t, &complex, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseCcsrcolor(handle_t, m, innz, matDescr_t, &complex, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrcolor(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* fraction_to_color, rocsparse_int* ncolors, rocsparse_int* coloring, rocsparse_int* reordering, rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_dcsrcolor(handle_t, m, innz, matDescr_t, &csrValA, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseDcsrcolor(handle_t, m, innz, matDescr_t, &csrValA, &csrRowPtrA, &csrColIndA, &dfractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* fractionToColor, int* ncolors, int* coloring, int* reordering, const cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrcolor(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* fraction_to_color, rocsparse_int* ncolors, rocsparse_int* coloring, rocsparse_int* reordering, rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_scsrcolor(handle_t, m, innz, matDescr_t, &csrSortedValA, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);
  status_t = cusparseScsrcolor(handle_t, m, innz, matDescr_t, &csrSortedValA, &csrRowPtrA, &csrColIndA, &ffractionToColor, &ncolors, &coloring, &reordering, colorInfo_t);

  // CUDA:cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgebsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const rocsparse_double_complex* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, const rocsparse_mat_descr descr_C, rocsparse_double_complex* bsr_val_C, rocsparse_int* bsr_row_ptr_C, rocsparse_int* bsr_col_ind_C, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, bsrRowPtrC, bsrColIndC, tempBuffer);
  status_t = cusparseZgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, bsrRowPtrC, bsrColIndC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC,int colBlockDimC, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgebsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const rocsparse_double_complex* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgebsr2gebsr_buffer_size(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseZgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dcomplex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgebsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const rocsparse_float_complex* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, const rocsparse_mat_descr descr_C, rocsparse_float_complex* bsr_val_C, rocsparse_int* bsr_row_ptr_C, rocsparse_int* bsr_col_ind_C, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseCgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgebsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const rocsparse_float_complex* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgebsr2gebsr_buffer_size(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseCgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &complex, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgebsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const double* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, const rocsparse_mat_descr descr_C, double* bsr_val_C, rocsparse_int* bsr_row_ptr_C, rocsparse_int* bsr_col_ind_C, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseDgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgebsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const double* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgebsr2gebsr_buffer_size(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseDgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgebsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const float* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, const rocsparse_mat_descr descr_C, float* bsr_val_C, rocsparse_int* bsr_row_ptr_C, rocsparse_int* bsr_col_ind_C, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);
  status_t = cusparseSgebsr2gebsr(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fbsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimC, colBlockDimC, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgebsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const float* bsr_val_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgebsr2gebsr_buffer_size(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);
  status_t = cusparseSgebsr2gebsr_bufferSize(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXgebsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, int nnzb, const cusparseMatDescr_t descrA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDimC, int colBlockDimC, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_gebsr2gebsr_nnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_mat_descr descr_A, const rocsparse_int* bsr_row_ptr_A, const rocsparse_int* bsr_col_ind_A, rocsparse_int row_block_dim_A, rocsparse_int col_block_dim_A, const rocsparse_mat_descr descr_C, rocsparse_int* bsr_row_ptr_C, rocsparse_int row_block_dim_C, rocsparse_int col_block_dim_C, rocsparse_int* nnz_total_dev_host_ptr, void* temp_buffer);
  // CHECK: status_t = rocsparse_gebsr2gebsr_nnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);
  status_t = cusparseXgebsr2gebsrNnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgebsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, const rocsparse_mat_descr csr_descr, rocsparse_double_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_zgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgebsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, const rocsparse_mat_descr csr_descr, rocsparse_float_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_cgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgebsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, const rocsparse_mat_descr csr_descr, double* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_dgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int rowBlockDim, int colBlockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgebsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, const rocsparse_mat_descr csr_descr, float* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_sgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseSgebsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr csr_descr, rocsparse_double_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_zbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &dComplexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr csr_descr, rocsparse_float_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_cbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &complexbsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr csr_descr, double* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_dbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &bsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsr2csr(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsr2csr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nb, const rocsparse_mat_descr bsr_descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr csr_descr, float* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_sbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseSbsr2csr(handle_t, direction_t, mb, nb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, rocsparse_int* coo_row_ind, rocsparse_int* coo_col_ind, rocsparse_int* perm, void* temp_buffer);
  // CHECK: status_t = rocsparse_coosort_by_column(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);
  status_t = cusparseXcoosortByColumn(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz, int* cooRowsA, int* cooColsA, int* P, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, rocsparse_int* coo_row_ind, rocsparse_int* coo_col_ind, rocsparse_int* perm, void* temp_buffer);
  // CHECK: status_t = rocsparse_coosort_by_row(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);
  status_t = cusparseXcoosortByRow(handle_t, m, n, innz, &icooRows, &icooColumns, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cooRowsA, const int* cooColsA, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_int* coo_row_ind, const rocsparse_int* coo_col_ind, size_t* buffer_size);
  // CHECK: status_t = rocsparse_coosort_buffer_size(handle_t, m, n, innz, &icooRows, &icooColumns, &bufferSize);
  status_t = cusparseXcoosort_bufferSizeExt(handle_t, m, n, innz, &icooRows, &icooColumns, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* cscColPtrA, int* cscRowIndA, int* P, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cscsort(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_int* csc_col_ptr, rocsparse_int* csc_row_ind, rocsparse_int* perm, void* temp_buffer);
  // CHECK: status_t = rocsparse_cscsort(handle_t, m, n, innz, matDescr_A, &cscColPtrA, &cscRowIndA, P, pBuffer);
  status_t = cusparseXcscsort(handle_t, m, n, innz, matDescr_A, &cscColPtrA, &cscRowIndA, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* cscColPtrA, const int* cscRowIndA, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cscsort_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_int* csc_col_ptr, const rocsparse_int* csc_row_ind, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cscsort_buffer_size(handle_t, m, n, innz, &cscColPtrA, &cscRowIndA, &bufferSize);
  status_t = cusparseXcscsort_bufferSizeExt(handle_t, m, n, innz, &cscColPtrA, &cscRowIndA, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, const int* csrRowPtrA, int* csrColIndA, int* P, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrsort(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind, rocsparse_int* perm, void* temp_buffer);
  // CHECK: status_t = rocsparse_csrsort(handle_t, m, n, innz, matDescr_A, &cscRowIndA, &cscColPtrA, P, pBuffer);
  status_t = cusparseXcsrsort(handle_t, m, n, innz, matDescr_A, &cscRowIndA, &cscColPtrA, P, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, const int* csrRowPtrA, const int* csrColIndA, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrsort_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, size_t* buffer_size);
  // CHECK: status_t = rocsparse_csrsort_buffer_size(handle_t, m, n, innz, &cscRowIndA, &cscColPtrA, &bufferSize);
  status_t = cusparseXcsrsort_bufferSizeExt(handle_t, m, n, innz, &cscRowIndA, &cscColPtrA, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_identity_permutation(rocsparse_handle handle, rocsparse_int n, rocsparse_int* p);
  // CHECK: status_t = rocsparse_create_identity_permutation(handle_t, n, P);
  status_t = cusparseCreateIdentityPermutation(handle_t, n, P);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m, int* csrSortedRowPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo2csr(rocsparse_handle handle, const rocsparse_int* coo_row_ind, rocsparse_int nnz, rocsparse_int m, rocsparse_int* csr_row_ptr, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_coo2csr(handle_t, &icooRowInd, nnz, m, &csrRowPtrA, indexBase_t);
  status_t = cusparseXcoo2csr(handle_t, &icooRowInd, nnz, m, &csrRowPtrA, indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_mat_descr bsr_descr, rocsparse_double_complex* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseZcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_mat_descr bsr_descr, rocsparse_float_complex* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseCcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_mat_descr bsr_descr, double* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseDcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2gebsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_mat_descr bsr_descr, float* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseScsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDim, int colBlockDim, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr2gebsr_nnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_mat_descr bsr_descr, rocsparse_int* bsr_row_ptr, rocsparse_int row_block_dim, rocsparse_int col_block_dim, rocsparse_int* bsr_nnz_devhost, void* temp_buffer);
  // CHECK: status_t = rocsparse_csr2gebsr_nnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimA, colBlockDimA, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseXcsr2gebsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimA, colBlockDimA, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsr2gebsr_buffer_size(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseZcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* buffer_size);
 // CHECK: status_t = rocsparse_ccsr2gebsr_buffer_size(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseCcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsr2gebsr_buffer_size(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseDcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2gebsr_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsr2gebsr_buffer_size(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseScsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2bsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr bsr_descr, rocsparse_double_complex* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind);
  // CHECK: status_t = rocsparse_zcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseZcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2bsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind,rocsparse_int block_dim, const rocsparse_mat_descr bsr_descr, rocsparse_float_complex* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind);
  // CHECK: status_t = rocsparse_ccsr2bsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseCcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2bsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr bsr_descr, double* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind);
  // CHECK: status_t = rocsparse_dcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseDcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2bsr(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr bsr_descr, float* bsr_val, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_col_ind);
  // CHECK: status_t = rocsparse_scsr2bsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseScsr2bsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr csr_descr, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_int block_dim, const rocsparse_mat_descr bsr_descr, rocsparse_int* bsr_row_ptr, rocsparse_int* bsr_nnz);
  // CHECK: status_t = rocsparse_csr2bsr_nnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &bsrSortedRowPtrC, &nnzTotalDevHostPtr);
  status_t = cusparseXcsr2bsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &bsrSortedRowPtrC, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgebsr2gebsc(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, rocsparse_double_complex* bsc_val, rocsparse_int* bsc_row_ind, rocsparse_int* bsc_col_ptr, rocsparse_action copy_values, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgebsr2gebsc(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dComplexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseZgebsr2gebsc(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dComplexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgebsr2gebsc(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, rocsparse_float_complex* bsc_val, rocsparse_int* bsc_row_ind, rocsparse_int* bsc_col_ptr, rocsparse_action copy_values, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgebsr2gebsc(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &complexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseCgebsr2gebsc(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &complexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, double* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgebsr2gebsc(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, double* bsc_val, rocsparse_int* bsc_row_ind, rocsparse_int* bsc_col_ptr, rocsparse_action copy_values, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgebsr2gebsc(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseDgebsr2gebsc(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, float* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgebsr2gebsc(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, float* bsc_val, rocsparse_int* bsc_row_ind, rocsparse_int* bsc_col_ptr, rocsparse_action copy_values, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgebsr2gebsc(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &fbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseSgebsr2gebsc(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &fbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgebsr2gebsc_buffer_size(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = rocsparse_zgebsr2gebsc_buffer_size(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseZgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgebsr2gebsc_buffer_size(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = rocsparse_cgebsr2gebsc_buffer_size(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseCgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgebsr2gebsc_buffer_size(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = rocsparse_dgebsr2gebsc_buffer_size(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseDgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgebsr2gebsc_buffer_size(rocsparse_handle handle, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int row_block_dim, rocsparse_int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = rocsparse_sgebsr2gebsc_buffer_size(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseSgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // TODO: [#899] There should be rocsparse_datatype
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuDoubleComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuDoubleComplex tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2csr_compress(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, const rocsparse_double_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, rocsparse_int nnz_A, const rocsparse_int* nnz_per_row, rocsparse_double_complex* csr_val_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, rocsparse_double_complex tol);
  // CHECK: status_t = rocsparse_zcsr2csr_compress(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dComplexcsrSortedValC, &csrColIndC, &csrRowPtrC, dcomplextol);
  status_t = cusparseZcsr2csr_compress(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dComplexcsrSortedValC, &csrColIndC, &csrRowPtrC, dcomplextol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuComplex tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2csr_compress(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, const rocsparse_float_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, rocsparse_int nnz_A, const rocsparse_int* nnz_per_row, rocsparse_float_complex* csr_val_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, rocsparse_float_complex tol);
  // CHECK: status_t = rocsparse_ccsr2csr_compress(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &complexcsrSortedValC, &csrColIndC, &csrRowPtrC, complextol);
  status_t = cusparseCcsr2csr_compress(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &complexcsrSortedValC, &csrColIndC, &csrRowPtrC, complextol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, double* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, double tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2csr_compress(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, rocsparse_int nnz_A, const rocsparse_int* nnz_per_row, double* csr_val_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, double tol);
  // CHECK: status_t = rocsparse_dcsr2csr_compress(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dcsrSortedValC, &csrColIndC, &csrRowPtrC, dtol);
  status_t = cusparseDcsr2csr_compress(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dcsrSortedValC, &csrColIndC, &csrRowPtrC, dtol);

  // cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, float* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, float tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2csr_compress(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, rocsparse_int nnz_A, const rocsparse_int* nnz_per_row, float* csr_val_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, float tol);
  // CHECK: status_t = rocsparse_scsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);
  status_t = cusparseScsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src);
  // CHECK: status_t = rocsparse_copy_mat_descr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);
#endif

#if CUDA_VERSION >= 9000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, double percentage, const rocsparse_mat_descr csr_descr_C, double* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_csr2csr_by_percentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, float percentage, const rocsparse_mat_descr csr_descr_C, float* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_csr2csr_by_percentage(handle_t, m, n, nnz, matDescr_A, &fcsrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fcsrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &fcsrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fcsrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr_nnz_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, double percentage, const rocsparse_mat_descr csr_descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_total_dev_host_ptr, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_csr2csr_nnz_by_percentage(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr_nnz_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, float percentage, const rocsparse_mat_descr csr_descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_total_dev_host_ptr, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_csr2csr_nnz_by_percentage(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr_by_percentage_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, double percentage, const rocsparse_mat_descr csr_descr_C, const double* csr_val_C, const rocsparse_int* csr_row_ptr_C, const rocsparse_int* csr_col_ind_C, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr_by_percentage_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, float percentage, const rocsparse_mat_descr csr_descr_C, const float* csr_val_C, const rocsparse_int* csr_row_ptr_C, const rocsparse_int* csr_col_ind_C, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const double* threshold, const rocsparse_mat_descr csr_descr_C, double* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_csr2csr(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseDpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const float* threshold, const rocsparse_mat_descr csr_descr_C, float* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_csr2csr(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseSpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const double* threshold, const rocsparse_mat_descr csr_descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_total_dev_host_ptr, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_csr2csr_nnz(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseDpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const float* threshold, const rocsparse_mat_descr csr_descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_total_dev_host_ptr, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_csr2csr_nnz(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseSpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_csr2csr_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const double* threshold, const rocsparse_mat_descr csr_descr_C, const double* csr_val_C, const rocsparse_int* csr_row_ptr_C, const rocsparse_int* csr_col_ind_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dprune_csr2csr_buffer_size(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseDpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_csr2csr_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz_A, const rocsparse_mat_descr csr_descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const float* threshold, const rocsparse_mat_descr csr_descr_C, const float* csr_val_C, const rocsparse_int* csr_row_ptr_C, const rocsparse_int* csr_col_ind_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sprune_csr2csr_buffer_size(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseSpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
#endif

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: _rocsparse_spmat_descr *spMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_spmat_descr spMatDescr_t, matC;
  cusparseSpMatDescr *spMatDescr = nullptr;
  cusparseSpMatDescr_t spMatDescr_t, matC;

  // CHECK: _rocsparse_dnmat_descr *dnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnmat_descr dnMatDescr_t, matA, matB;
  cusparseDnMatDescr *dnMatDescr = nullptr;
  cusparseDnMatDescr_t dnMatDescr_t, matA, matB;

  // CHECK: rocsparse_indextype indexType_t;
  // CHECK-NEXT: rocsparse_indextype csrRowOffsetsType;
  // CHECK-NEXT: rocsparse_indextype cscColOffsetsType;
  // CHECK-NEXT: rocsparse_indextype cscRowIndType;
  // CHECK-NEXT: rocsparse_indextype csrColIndType;
  // CHECK-NEXT: rocsparse_indextype ellIdxType;
  // CHECK-NEXT: rocsparse_indextype INDEX_16U = rocsparse_indextype_u16;
  // CHECK-NEXT: rocsparse_indextype INDEX_32I = rocsparse_indextype_i32;
  // CHECK-NEXT: rocsparse_indextype INDEX_64I = rocsparse_indextype_i64;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t cscColOffsetsType;
  cusparseIndexType_t cscRowIndType;
  cusparseIndexType_t csrColIndType;
  cusparseIndexType_t ellIdxType;
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

  // CHECK: rocsparse_order order_t;
  // CHECK-NEXT: rocsparse_order ORDER_COL = rocsparse_order_row;
  // CHECK-NEXT: rocsparse_order ORDER_ROW = rocsparse_order_column;
  cusparseOrder_t order_t;
  cusparseOrder_t ORDER_COL = CUSPARSE_ORDER_COL;
  cusparseOrder_t ORDER_ROW = CUSPARSE_ORDER_ROW;

  // CHECK: rocsparse_spmm_alg spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr, int64_t ows, int64_t cols, int64_t nnz, void* cooRowInd, void* cooColInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, void* coo_row_ind, void* coo_col_ind, void* coo_val, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_coo_descr(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateCoo(&spMatDescr_t, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_spmat_descr descr);
  // CHECK: status_t = rocsparse_destroy_spmat_descr(spMatDescr_t);
  status_t = cusparseDestroySpMat(spMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, void** coo_row_ind, void** coo_col_ind, void** coo_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_coo_get(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_format(const rocsparse_spmat_descr descr, rocsparse_format* format);
  // CHECK: status_t = rocsparse_spmat_get_format(spMatDescr_t, &format_t);
  status_t = cusparseSpMatGetFormat(spMatDescr_t, &format_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_index_base(const rocsparse_spmat_descr descr, rocsparse_index_base* idx_base);
  // CHECK: status_t = rocsparse_spmat_get_index_base(spMatDescr_t, &indexBase_t);
  status_t = cusparseSpMatGetIndexBase(spMatDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr, int64_t rows, int64_t cols, int64_t ld, void* values, rocsparse_datatype data_type, rocsparse_order order);
  // CHECK: status_t = rocsparse_create_dnmat_descr(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_dnmat_descr descr);
  // CHECK: status_t = rocsparse_destroy_dnmat_descr(dnMatDescr_t);
  status_t = cusparseDestroyDnMat(dnMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, rocsparse_datatype* data_type, rocsparse_order* order);
  // CHECK: status_t = rocsparse_dnmat_get(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);
  status_t = cusparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_get_strided_batch(rocsparse_dnmat_descr descr, int* batch_count, int64_t* batch_stride);
  // CHECK: status_t = rocsparse_dnmat_get_strided_batch(dnMatDescr_t, &batchCount, &batchStride);
  status_t = cusparseDnMatGetStridedBatch(dnMatDescr_t, &batchCount, &batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_set_strided_batch(rocsparse_dnmat_descr descr, int batch_count, int64_t batch_stride);
  // CHECK: status_t = rocsparse_dnmat_set_strided_batch(dnMatDescr_t, batchCount, batchStride);
  status_t = cusparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);
#endif

#if CUDA_VERSION >= 10020
  // CHECK: rocsparse_status STATUS_NOT_SUPPORTED = rocsparse_status_not_implemented;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;
#endif

#if (CUDA_VERSION >= 10020 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: _rocsparse_spvec_descr *spVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_spvec_descr spVecDescr_t;
  cusparseSpVecDescr *spVecDescr = nullptr;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: _rocsparse_dnvec_descr *dnVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnvec_descr dnVecDescr_t, vecX, vecY;
  cusparseDnVecDescr *dnVecDescr = nullptr;
  cusparseDnVecDescr_t dnVecDescr_t, vecX, vecY;

  // CHECK: rocsparse_spmv_alg spMVAlg_t;
  cusparseSpMVAlg_t spMVAlg_t;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr, int64_t size, int64_t nnz, void* indices, void* values, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_spvec_descr(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateSpVec(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_spvec_descr descr);
  // CHECK: status_t = rocsparse_destroy_spvec_descr(spVecDescr_t);
  status_t = cusparseDestroySpVec(spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr, int64_t* size, int64_t* nnz, void** indices, void** values, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_spvec_get(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get_index_base(const rocsparse_spvec_descr descr, rocsparse_index_base* idx_base);
  // CHECK: status_t = rocsparse_spvec_get_index_base(spVecDescr_t, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values);
  // CHECK: status_t = rocsparse_spvec_get_values(spVecDescr_t, &values);
  status_t = cusparseSpVecGetValues(spVecDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values);
  // CHECK: status_t = rocsparse_spvec_set_values(spVecDescr_t, values);
  status_t = cusparseSpVecSetValues(spVecDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* csrRowOffsets, void* csrColInd, void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, void* csr_row_ptr, void* csr_col_ind, void* csr_val, rocsparse_indextype row_ptr_type, rocsparse_indextype col_ind_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_csr_descr(&spMatDescr_t, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateCsr(&spMatDescr_t, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csrRowOffsets, void** csrColInd, void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csr_row_ptr, void** csr_col_ind, void** csr_val, rocsparse_indextype* row_ptr_type, rocsparse_indextype* col_ind_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_csr_get(spMatDescr_t, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);
  status_t = cusparseCsrGet(spMatDescr_t, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd, &csrValues, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values);
  // CHECK: status_t = rocsparse_spmat_get_values(spMatDescr_t, &values);
  status_t = cusparseSpMatGetValues(spMatDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values);
  // CHECK: status_t = rocsparse_spmat_set_values(spMatDescr_t, values);
  status_t = cusparseSpMatSetValues(spMatDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_spmat_descr descr, int* batch_count);
  // CHECK: status_t = rocsparse_spmat_get_strided_batch(spMatDescr_t, &batchCount);
  status_t = cusparseSpMatGetStridedBatch(spMatDescr_t, &batchCount);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr, int64_t size, void* values, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_dnvec_descr(&dnVecDescr_t, size, values, dataType);
  status_t = cusparseCreateDnVec(&dnVecDescr_t, size, values, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_dnvec_descr descr);
  // CHECK: status_t = rocsparse_destroy_dnvec_descr(dnVecDescr_t);
  status_t = cusparseDestroyDnVec(dnVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr, int64_t* size, void** values, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr, int64_t* size, void** values, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_dnvec_get(dnVecDescr_t, &size, &values, &dataType);
  status_t = cusparseDnVecGet(dnVecDescr_t, &size, &values, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values);
  // CHECK: status_t = rocsparse_dnvec_get_values(dnVecDescr_t, &values);
  status_t = cusparseDnVecGetValues(dnVecDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values);
  // CHECK: status_t = rocsparse_dnvec_set_values(dnVecDescr_t, values);
  status_t = cusparseDnVecSetValues(dnVecDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values);
  // CHECK: status_t = rocsparse_dnmat_get_values(dnMatDescr_t, &values);
  status_t = cusparseDnMatGetValues(dnMatDescr_t, &values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values);
  // CHECK: status_t = rocsparse_dnmat_set_values(dnMatDescr_t, values);
  status_t = cusparseDnMatSetValues(dnMatDescr_t, values);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmv(rocsparse_handle handle, rocsparse_operation trans, const void* alpha, const rocsparse_spmat_descr mat, const rocsparse_dnvec_descr x, const void* beta, const rocsparse_dnvec_descr y, rocsparse_datatype compute_type, rocsparse_spmv_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmv(handle_t, opA, alpha, spMatDescr_t, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);
  status_t = cusparseSpMV(handle_t, opA, alpha, spMatDescr_t, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);
#endif

#if (CUDA_VERSION >= 10020 && CUDA_VERSION < 11000 && !defined(_WIN32)) || (CUDA_VERSION >= 11000 && CUDA_VERSION < 12000)
  // CHECK: rocsparse_format FORMAT_COO_AOS = rocsparse_format_coo_aos;
  cusparseFormat_t FORMAT_COO_AOS = CUSPARSE_FORMAT_COO_AOS;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCooAoS(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cooInd, void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, void* coo_ind, void* coo_val, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_coo_aos_descr(&spMatDescr_t, rows, cols, nnz, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateCooAoS(&spMatDescr_t, rows, cols, nnz, cooColInd, cooValues, indexType_t, indexBase_t, dataType);

  // CUDA: CUSPARSE_DEPRECATED(cusparseCooGet) cusparseStatus_t CUSPARSEAPI cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, void** coo_ind, void** coo_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_coo_aos_get(spMatDescr_t, &rows, &cols, &nnz, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooAoSGet(spMatDescr_t, &rows, &cols, &nnz, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr, int batch_count);
  // CHECK: status_t = rocsparse_spmat_set_strided_batch(spMatDescr_t, batchCount);
  status_t = cusparseSpMatSetStridedBatch(spMatDescr_t, batchCount);
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t* hybA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb);
  // CHECK: status_t = rocsparse_create_hyb_mat(&hybMat_t);
  status_t = cusparseCreateHybMat(&hybMat_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);
  // CHECK: status_t = rocsparse_destroy_hyb_mat(hybMat_t);
  status_t = cusparseDestroyHybMat(hybMat_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t   partitionType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2hyb(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_hyb_mat hyb, rocsparse_int user_ell_width, rocsparse_hyb_partition partition_type);
  // CHECK: status_t = rocsparse_zcsr2hyb(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseZcsr2hyb(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2hyb(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_hyb_mat hyb, rocsparse_int user_ell_width, rocsparse_hyb_partition partition_type);
  // CHECK: status_t = rocsparse_ccsr2hyb(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseCcsr2hyb(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2hyb(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_hyb_mat hyb, rocsparse_int user_ell_width, rocsparse_hyb_partition partition_type);
  // CHECK: status_t = rocsparse_dcsr2hyb(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseDcsr2hyb(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2hyb(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_hyb_mat hyb, rocsparse_int user_ell_width, rocsparse_hyb_partition partition_type);
  // CHECK: status_t = rocsparse_scsr2hyb(handle_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseScsr2hyb(handle_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: rocsparse_spgemm_alg spGEMMAlg_t;
  // CHECK-NEXT: rocsparse_spgemm_alg SPGEMM_DEFAULT = rocsparse_spgemm_alg_default;
  cusparseSpGEMMAlg_t spGEMMAlg_t;
  cusparseSpGEMMAlg_t SPGEMM_DEFAULT = CUSPARSE_SPGEMM_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr, void* csr_row_ptr, void* csr_col_ind, void* csr_val);
  // CHECK: status_t = rocsparse_csr_set_pointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);
  status_t = cusparseCsrSetPointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_size(rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // CHECK: status_t = rocsparse_spmat_get_size(spMatDescr_t, &rows, &cols, &nnz);
  status_t = cusparseSpMatGetSize(spMatDescr_t, &rows, &cols, &nnz);
#endif

#if CUDA_VERSION >= 11000 && CUSPARSE_VERSION >= 11100
  // CHECK: rocsparse_spmm_alg SPMM_ALG_DEFAULT = rocsparse_spmm_alg_default;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG1 = rocsparse_spmm_alg_coo_segmented;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG2 = rocsparse_spmm_alg_coo_atomic;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG3 = rocsparse_spmm_alg_coo_segmented_atomic;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_CSR_ALG1 = rocsparse_spmm_alg_csr;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_CSR_ALG2 = rocsparse_spmm_alg_csr_row_split;
  cusparseSpMMAlg_t SPMM_ALG_DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT;
  cusparseSpMMAlg_t SPMM_COO_ALG1 = CUSPARSE_SPMM_COO_ALG1;
  cusparseSpMMAlg_t SPMM_COO_ALG2 = CUSPARSE_SPMM_COO_ALG2;
  cusparseSpMMAlg_t SPMM_COO_ALG3 = CUSPARSE_SPMM_COO_ALG3;
  cusparseSpMMAlg_t SPMM_CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1;
  cusparseSpMMAlg_t SPMM_CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr, int batch_count, int64_t batch_stride);
  // CHECK: status_t = rocsparse_coo_set_strided_batch(spMatDescr_t, batchCount, batchStride);
  status_t = cusparseCooSetStridedBatch(spMatDescr_t, batchCount, batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr, int batch_count, int64_t offsets_batch_stride, int64_t columns_values_batch_stride);
  // CHECK: status_t = rocsparse_csr_set_strided_batch(spMatDescr_t, batchCount, offsetsBatchStride, columnsValuesBatchStride);
  status_t = cusparseCsrSetStridedBatch(spMatDescr_t, batchCount, offsetsBatchStride, columnsValuesBatchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseRot(cusparseHandle_t handle, const void* c_coeff, const void* s_coeff, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_rot(rocsparse_handle handle, const void* c, const void* s, rocsparse_spvec_descr x, rocsparse_dnvec_descr y);
  // CHECK: status_t = rocsparse_rot(handle_t, c_coeff, s_coeff, spVecDescr_t, vecY);
  status_t = cusparseRot(handle_t, c_coeff, s_coeff, spVecDescr_t, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scatter(rocsparse_handle handle, const rocsparse_spvec_descr x, rocsparse_dnvec_descr y);
  // CHECK: status_t = rocsparse_scatter(handle_t, spVecDescr_t, vecY);
  status_t = cusparseScatter(handle_t, spVecDescr_t, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_gather(rocsparse_handle handle, const rocsparse_dnvec_descr y, rocsparse_spvec_descr x);
  // CHECK: status_t = rocsparse_gather(handle_t, vecY, spVecDescr_t);
  status_t = cusparseGather(handle_t, vecY, spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_axpby(rocsparse_handle handle, const void* alpha, const rocsparse_spvec_descr x, const void* beta, rocsparse_dnvec_descr y);
  // CHECK: status_t = rocsparse_axpby(handle_t, alpha, spVecDescr_t, beta, vecY);
  status_t = cusparseAxpby(handle_t, alpha, spVecDescr_t, beta, vecY);
#endif

#if CUDA_VERSION >= 11010 && CUSPARSE_VERSION >= 11300
  // CHECK: rocsparse_sparse_to_dense_alg sparseToDenseAlg_t;
  // CHECK-NEXT: rocsparse_sparse_to_dense_alg SPARSETODENSE_ALG_DEFAULT = rocsparse_sparse_to_dense_alg_default;
  cusparseSparseToDenseAlg_t sparseToDenseAlg_t;
  cusparseSparseToDenseAlg_t SPARSETODENSE_ALG_DEFAULT = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

  // CHECK: rocsparse_dense_to_sparse_alg denseToSparseAlg_t;
  // CHECK-NEXT: rocsparse_dense_to_sparse_alg DENSETOSPARSE_ALG_DEFAULT = rocsparse_dense_to_sparse_alg_default;
  cusparseDenseToSparseAlg_t denseToSparseAlg_t;
  cusparseDenseToSparseAlg_t DENSETOSPARSE_ALG_DEFAULT = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void* cscColOffsets, void* cscRowInd, void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, void* csc_col_ptr, void* csc_row_ind, void* csc_val, rocsparse_indextype col_ptr_type, rocsparse_indextype row_ind_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_csc_descr(&spMatDescr_t, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateCsc(&spMatDescr_t, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, csrColIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooSetPointers(cusparseSpMatDescr_t spMatDescr, void* cooRows, void* cooColumns, void* cooValues);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr, void* coo_row_ind, void* coo_col_ind, void* coo_val);
  // CHECK: status_t = rocsparse_coo_set_pointers(spMatDescr_t, cooRows, cooColumns, cooValues);
  status_t = cusparseCooSetPointers(spMatDescr_t, cooRows, cooColumns, cooValues);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCscSetPointers(cusparseSpMatDescr_t spMatDescr, void* cscColOffsets, void* cscRowInd, void* cscValues);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr, void* csc_col_ptr, void* csc_row_ind, void* csc_val);
  // CHECK: status_t = rocsparse_csc_set_pointers(spMatDescr_t, cscColOffsets, cscRowInd, cscValues);
  status_t = cusparseCscSetPointers(spMatDescr_t, cscColOffsets, cscRowInd, cscValues);
#endif

#if CUDA_VERSION >= 11020 && CUSPARSE_VERSION >= 11400
  // CHECK: rocsparse_format FORMAT_BLOCKED_ELL = rocsparse_format_bell;
  cusparseFormat_t FORMAT_BLOCKED_ELL = CUSPARSE_FORMAT_BLOCKED_ELL;

  // CHECK: rocsparse_spmv_alg SPMV_ALG_DEFAULT = rocsparse_spmv_alg_default;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_COO_ALG1 = rocsparse_spmv_alg_coo;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_COO_ALG2 = rocsparse_spmv_alg_coo_atomic;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_CSR_ALG1 = rocsparse_spmv_alg_csr_adaptive;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_CSR_ALG2 = rocsparse_spmv_alg_csr_stream;
  cusparseSpMVAlg_t SPMV_ALG_DEFAULT = CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseSpMVAlg_t SPMV_COO_ALG1 = CUSPARSE_SPMV_COO_ALG1;
  cusparseSpMVAlg_t SPMV_COO_ALG2 = CUSPARSE_SPMV_COO_ALG2;
  cusparseSpMVAlg_t SPMV_CSR_ALG1 = CUSPARSE_SPMV_CSR_ALG1;
  cusparseSpMVAlg_t SPMV_CSR_ALG2 = CUSPARSE_SPMV_CSR_ALG2;

  // CHECK: rocsparse_spmm_alg SPMM_CSR_ALG3 = rocsparse_spmm_alg_csr_merge;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_BLOCKED_ELL_ALG1 = rocsparse_spmm_alg_bell;
  cusparseSpMMAlg_t SPMM_CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3;
  cusparseSpMMAlg_t SPMM_BLOCKED_ELL_ALG1 = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;

  // CHECK: rocsparse_sddmm_alg sDDMMAlg_t;
  // CHECK-NEXT: rocsparse_sddmm_alg SDDMM_ALG_DEFAULT = rocsparse_sddmm_alg_default;
  cusparseSDDMMAlg_t sDDMMAlg_t;
  cusparseSDDMMAlg_t SDDMM_ALG_DEFAULT = CUSPARSE_SDDMM_ALG_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateBlockedEll(cusparseSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr, int64_t rows, int64_t cols, rocsparse_direction ell_block_dir, int64_t ell_block_dim, int64_t ell_cols, void* ell_col_ind, void* ell_val, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_bell_descr(&spMatDescr_t, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);
  status_t = cusparseCreateBlockedEll(&spMatDescr_t, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseBlockedEllGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, void** ellColInd, void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, rocsparse_direction* ell_block_dir, int64_t* ell_block_dim, int64_t* ell_cols, void** ell_col_ind, void** ell_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_bell_get(spMatDescr_t, &rows, &cols, &ellBlockSize, &ellCols, &ellColInd, &ellValue, &ellIdxType, &indexBase_t, &dataType);
  status_t = cusparseBlockedEllGet(spMatDescr_t, &rows, &cols, &ellBlockSize, &ellCols, &ellColInd, &ellValue, &ellIdxType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, const rocsparse_dnmat_descr A, const rocsparse_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, void* temp_buffer);
  // CHECK: status_t = rocsparse_sddmm_preprocess(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM_preprocess(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, const rocsparse_dnmat_descr A, const rocsparse_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sddmm_buffer_size(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, &bufferSize);
  status_t = cusparseSDDMM_bufferSize(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, const rocsparse_dnmat_descr A, const rocsparse_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, void* temp_buffer);
  // CHECK: status_t = rocsparse_sddmm(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM(handle_t, opA, opB, alpha, matA, matB, beta, matC, dataType, sDDMMAlg_t, tempBuffer);
#endif

#if CUDA_VERSION >= 11030
  // CHECK: rocsparse_spmat_attribute spMatAttribute_t;
  // CHECK-NEXT: rocsparse_spmat_attribute SPMAT_FILL_MODE = rocsparse_spmat_fill_mode;
  // CHECK-NEXT: rocsparse_spmat_attribute SPMAT_DIAG_TYPE = rocsparse_spmat_diag_type;
  cusparseSpMatAttribute_t spMatAttribute_t;
  cusparseSpMatAttribute_t SPMAT_FILL_MODE = CUSPARSE_SPMAT_FILL_MODE;
  cusparseSpMatAttribute_t SPMAT_DIAG_TYPE = CUSPARSE_SPMAT_DIAG_TYPE;

  // CHECK: rocsparse_spsv_alg spSVAlg_t;
  // CHECK-NEXT: rocsparse_spsv_alg SPSV_ALG_DEFAULT = rocsparse_spsv_alg_default;
  cusparseSpSVAlg_t spSVAlg_t;
  cusparseSpSVAlg_t SPSV_ALG_DEFAULT = CUSPARSE_SPSV_ALG_DEFAULT;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_attribute(rocsparse_spmat_descr descr, rocsparse_spmat_attribute attribute, void* data, size_t data_size);
  // CHECK: status_t = rocsparse_spmat_get_attribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatGetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr descr, rocsparse_spmat_attribute attribute, const void* data, size_t data_size);
  // CHECK: status_t = rocsparse_spmat_set_attribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatSetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
#endif

#if CUDA_VERSION >= 11030 && CUSPARSE_VERSION >= 11600
  // CHECK: rocsparse_spsm_alg spSMAlg_t;
  // CHECK-NEXT: rocsparse_spsm_alg SPSM_ALG_DEFAULT = rocsparse_spsm_alg_default;
  cusparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t SPSM_ALG_DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT;
#endif

#if CUDA_VERSION >= 12010 && CUSPARSE_VERSION >= 12100
  // CHECK: rocsparse_spmv_alg SPMV_SELL_ALG1 = rocsparse_spmv_alg_ell;
  cusparseSpMVAlg_t SPMV_SELL_ALG1 = CUSPARSE_SPMV_SELL_ALG1;

  // CHECK: rocsparse_format FORMAT_BSR = rocsparse_format_bsr;
  // CHECK-NEXT: rocsparse_format FORMAT_SLICED_ELLPACK = rocsparse_format_ell;
  cusparseFormat_t FORMAT_BSR = CUSPARSE_FORMAT_BSR;
  cusparseFormat_t FORMAT_SLICED_ELLPACK = CUSPARSE_FORMAT_SLICED_ELLPACK;
#endif

  return 0;
}
