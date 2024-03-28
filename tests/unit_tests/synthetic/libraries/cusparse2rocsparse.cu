// RUN: %run_test hipify "%s" "%t" %hipify_args 5 --amap --skip-excluded-preprocessor-conditional-blocks --experimental --roc --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("18. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

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

  // CHECK: rocsparse_status status_t, status_2;
  // CHECK-NEXT: rocsparse_status STATUS_SUCCESS = rocsparse_status_success;
  // CHECK-NEXT: rocsparse_status STATUS_NOT_INITIALIZED = rocsparse_status_not_initialized;
  // CHECK-NEXT: rocsparse_status STATUS_ALLOC_FAILED = rocsparse_status_memory_error;
  // CHECK-NEXT: rocsparse_status STATUS_INVALID_VALUE = rocsparse_status_invalid_value;
  // CHECK-NEXT: rocsparse_status STATUS_ARCH_MISMATCH = rocsparse_status_arch_mismatch;
  // CHECK-NEXT: rocsparse_status STATUS_INTERNAL_ERROR = rocsparse_status_internal_error;
  // CHECK-NEXT: rocsparse_status STATUS_ZERO_PIVOT = rocsparse_status_zero_pivot;
  cusparseStatus_t status_t, status_2;
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
  int k = 0;
  int kb = 0;
  int mb = 0;
  int nb = 0;
  int nnza = 0;
  int nnzb = 0;
  int nnzc = 0;
  int nnzd = 0;
  int nrhs = 0;
  int nnzPerRow = 0;
  int nnzPerCol = 0;
  int innz = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int ldx = 0;
  int blockDim = 0;
  int csrSortedRowPtr = 0;
  int csrSortedColInd = 0;
  int cscRowIndA = 0;
  int cscColPtrA = 0;
  int csrRowPtrA = 0;
  int csrRowPtrB = 0;
  int csrRowPtrD = 0;
  int csrColIndA = 0;
  int csrColIndB = 0;
  int csrColIndD = 0;
  int ncolors = 0;
  int coloring = 0;
  int reordering = 0;
  int bscRowInd = 0;
  int bsrRowPtrA = 0;
  int bsrEndPtrA = 0;
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
  int bsrSortedMaskPtrA = 0;
  int bufferSizeInBytes = 0;
  int nnzTotalDevHostPtr = 0;
  int nnzPerRowCol = 0;
  int userEllWidth = 0;
  int ienable_boost = 0;
  int iposition = 0;
  int xInd = 0;
  int sizeOfMask = 0;
  int64_t size = 0;
  int64_t nnz = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t ellCols = 0;
  int64_t ellBlockSize = 0;
  int64_t batchStride = 0;
  int ibatchStride = 0;
  int64_t offsetsBatchStride = 0;
  int64_t columnsValuesBatchStride = 0;
  int64_t ld = 0;
  void *indices = nullptr;
  const void** const indices_const = const_cast<const void**>(&indices);
  void *values = nullptr;
  const void** const values_const = const_cast<const void**>(&values);
  void *cooRowInd = nullptr;
  const void** const cooRowInd_const = const_cast<const void**>(&cooRowInd);
  int icooRowInd = 0;
  void *cscRowInd = nullptr;
  const void** const cscRowInd_const = const_cast<const void**>(&cscRowInd);
  void *cooColInd = nullptr;
  const void** const cooColInd_const = const_cast<const void**>(&cooColInd);
  void *ellColInd = nullptr;
  const void** const ellColInd_const = const_cast<const void**>(&ellColInd);
  void *csrColInd = nullptr;
  const void** const csrColInd_const = const_cast<const void**>(&csrColInd);
  void *cooValues = nullptr;
  const void** const cooValues_const = const_cast<const void**>(&cooValues);
  void *csrValues = nullptr;
  const void** const csrValues_const = const_cast<const void**>(&csrValues);
  void *cscValues = nullptr;
  const void** const cscValues_const = const_cast<const void**>(&cscValues);
  void *ellValue = nullptr;
  const void** const ellValue_const = const_cast<const void**>(&ellValue);
  void *csrRowOffsets = nullptr;
  const void** const csrRowOffsets_const = const_cast<const void**>(&csrRowOffsets);
  void *cscColOffsets = nullptr;
  const void** const cscColOffsets_const = const_cast<const void**>(&cscColOffsets);
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
  float csrSortedValB = 0.f;
  float csrSortedValC = 0.f;
  double dcsrSortedVal = 0.f;
  double dcscSortedVal = 0.f;
  double dcsrSortedValA = 0.f;
  double dcsrSortedValB = 0.f;
  double dbsrSortedVal = 0.f;
  double dbsrSortedValA = 0.f;
  double dbsrSortedValC = 0.f;
  float fbsrSortedVal = 0.f;
  float fbsrSortedValA = 0.f;
  float fbsrSortedValC = 0.f;
  float fcsrSortedValC = 0.f;
  double dcsrSortedValC = 0.f;
  double d_resultDevHostPtr = 0.f;
  float f_resultDevHostPtr = 0.f;
  double percentage = 0.f;
  float fpercentage = 0.f;
  double dthreshold = 0.f;
  float fthreshold = 0.f;
  double dtol = 0.f;
  float ftol = 0.f;
  double dbscVal = 0.f;
  float fbscVal = 0.f;
  double dA = 0.f;
  double dAlpha = 0.f;
  double dB = 0.f;
  double dBeta = 0.f;
  double dC = 0.f;
  double dF = 0.f;
  double dS = 0.f;
  double dX = 0.f;
  double dY = 0.f;
  float fA = 0.f;
  float fAlpha = 0.f;
  float fB = 0.f;
  float fBeta = 0.f;
  float fC = 0.f;
  float fF = 0.f;
  float fS = 0.f;
  float fX = 0.f;
  float fY = 0.f;
  int algo = 0;
  double dds = 0.f;
  double ddl = 0.f;
  double dd = 0.f;
  double ddu = 0.f;
  double ddw = 0.f;
  double dx = 0.f;
  float fds = 0.f;
  float fdl = 0.f;
  float fd = 0.f;
  float fdu = 0.f;
  float fdw = 0.f;
  float fx = 0.f;
  double dboost_val = 0.f;
  float boost_val = 0.f;
  const char* const_ch = nullptr;

  // CHECK: rocsparse_mat_info csrilu02_info;
  csrilu02Info_t csrilu02_info;

  // CHECK: rocsparse_mat_info csric02_info;
  csric02Info_t csric02_info;

  // CHECK: rocsparse_mat_info bsrilu02_info;
  bsrilu02Info_t bsrilu02_info;

  // CHECK: rocsparse_mat_info bsric02_info;
  bsric02Info_t bsric02_info;

  // CHECK: rocsparse_mat_info bsrsm2_info;
  bsrsm2Info_t bsrsm2_info;

  // CHECK: rocsparse_mat_info bsrsv2_info;
  bsrsv2Info_t bsrsv2_info;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val, dcomplex_resultDevHostPtr;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val, dcomplex_resultDevHostPtr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val, complex_resultDevHostPtr;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val, complex_resultDevHostPtr;

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz, int m, int* cooRowInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr2coo(rocsparse_handle handle, const rocsparse_int* csr_row_ptr, rocsparse_int nnz, rocsparse_int m, rocsparse_int* coo_row_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_csr2coo(handle_t, &csrSortedRowPtr, nnz, m, &icooRowInd, indexBase_t);
  status_t = cusparseXcsr2coo(handle_t, &csrSortedRowPtr, nnz, m, &icooRowInd, indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_znnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* A, rocsparse_int ld, rocsparse_int* nnz_per_row_columns, rocsparse_int* nnz_total_dev_host_ptr);
  // CHECK: status_t = rocsparse_znnz(handle_t, direction_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseZnnz(handle_t, direction_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cnnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* A, rocsparse_int ld, rocsparse_int* nnz_per_row_columns, rocsparse_int* nnz_total_dev_host_ptr);
  // CHECK: status_t = rocsparse_cnnz(handle_t, direction_t, m, n, matDescr_A, &complexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseCnnz(handle_t, direction_t, m, n, matDescr_A, &complexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* A, rocsparse_int ld, rocsparse_int* nnz_per_row_columns, rocsparse_int* nnz_total_dev_host_ptr);
  // CHECK: status_t = rocsparse_dnnz(handle_t, direction_t, m, n, matDescr_A, &dA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseDnnz(handle_t, direction_t, m, n, matDescr_A, &dA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_snnz(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* A, rocsparse_int ld, rocsparse_int* nnz_per_row_columns, rocsparse_int* nnz_total_dev_host_ptr);
  // CHECK: status_t = rocsparse_snnz(handle_t, direction_t, m, n, matDescr_A, &fA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseSnnz(handle_t, direction_t, m, n, matDescr_A, &fA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrilu0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrilu0(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrilu02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrilu0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrilu0(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrilu02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrilu0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrilu0(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrilu02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrilu0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrilu0(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrilu02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrilu0_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrilu0_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrilu02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrilu0_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrilu0_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrilu02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseScsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseZcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseCcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseDcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrilu0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrilu0_buffer_size(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseScsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const rocsparse_double_complex* boost_val);
  // CHECK: status_t = rocsparse_zcsrilu0_numeric_boost(handle_t, csrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);
  status_t = cusparseZcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dccsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const rocsparse_float_complex* boost_val);
  // CHECK: status_t = rocsparse_dccsrilu0_numeric_boost(handle_t, csrilu02_info, ienable_boost, &dtol, &complex_boost_val);
  status_t = cusparseCcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &complex_boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const double* boost_val);
  // CHECK: status_t = rocsparse_dcsrilu0_numeric_boost(handle_t, csrilu02_info, ienable_boost, &dtol, &dboost_val);
  status_t = cusparseDcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dboost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const float* boost_val);
  // CHECK: status_t = rocsparse_dscsrilu0_numeric_boost(handle_t, csrilu02_info, ienable_boost, &dtol, &boost_val);
  status_t = cusparseScsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrilu0_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_csrilu0_zero_pivot(handle_t, csrilu02_info, &iposition);
  status_t = cusparseXcsrilu02_zeroPivot(handle_t, csrilu02_info, &iposition);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsric0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsric0(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsric02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsric0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsric0(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsric02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsric0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsric0(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsric02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsric0(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsric0(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsric02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsric0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsric0_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsric02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsric0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsric0_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsric02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsric0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsric0_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsric02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsric0_analysis(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsric0_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsric02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsric0_buffer_size(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsric0_buffer_size(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCcsric02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsric0_buffer_size(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsric0_buffer_size(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseScsric02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsric0_buffer_size(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseZcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsric0_buffer_size(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseCcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsric0_buffer_size(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseDcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsric0_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsric0_buffer_size(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseScsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csric0_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_csric0_zero_pivot(handle_t, csric02_info, &iposition);
  status_t = cusparseXcsric02_zeroPivot(handle_t, csric02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrilu0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrilu0(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrilu0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrilu0(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrilu0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrilu0(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrilu0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrilu0(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseSbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_bsrilu0_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_bsrilu0_zero_pivot(handle_t, bsrilu02_info, &iposition);
  status_t = cusparseXbsrilu02_zeroPivot(handle_t, bsrilu02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrilu0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrilu0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrilu0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrilu0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrilu0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrilu0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrilu0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrilu0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseSbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const rocsparse_double_complex* boost_val);
  // CHECK: status_t = rocsparse_zbsrilu0_numeric_boost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);
  status_t = cusparseZbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcbsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const rocsparse_float_complex* boost_val);
  // CHECK: status_t = rocsparse_dcbsrilu0_numeric_boost(handle_t, bsrilu02_info, ienable_boost, &dtol, &complex_boost_val);
  status_t = cusparseCbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &complex_boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const double* boost_val);
  // CHECK: status_t = rocsparse_dbsrilu0_numeric_boost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dboost_val);
  status_t = cusparseDbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dboost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dsbsrilu0_numeric_boost(rocsparse_handle handle, rocsparse_mat_info info, int enable_boost, const double* boost_tol, const float* boost_val);
  // CHECK: status_t = rocsparse_dsbsrilu0_numeric_boost(handle_t, bsrilu02_info, ienable_boost, &dtol, &boost_val);
  status_t = cusparseSbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsric0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsric0(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsric0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsric0(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsric0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsric0(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsric0(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsric0(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseSbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsric0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsric0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsric0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsric0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsric0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsric0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsric0_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsric0_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseSbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsric0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zbsric0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsric0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cbsric0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsric0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dbsric0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsric0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sbsric0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseSbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_bsric0_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_bsric0_zero_pivot(handle_t, bsric02_info, &iposition);
  status_t = cusparseXbsric02_zeroPivot(handle_t, bsric02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuDoubleComplex* B, int ldb, cuDoubleComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsm_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const rocsparse_double_complex* B, rocsparse_int ldb, rocsparse_double_complex* X, rocsparse_int ldx, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrsm_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dcomplexA, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dcomplexB, ldb, &dcomplexX, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dcomplexA, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dcomplexB, ldb, &dcomplexX, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuComplex* B, int ldb, cuComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsm_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const rocsparse_float_complex* B, rocsparse_int ldb, rocsparse_float_complex* X, rocsparse_int ldx, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrsm_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &complexA, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &complexB, ldb, &complexX, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &complexA, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &complexB, ldb, &complexX, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const double* B, int ldb, double* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsm_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const double* alpha, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const double* B, rocsparse_int ldb, double* X, rocsparse_int ldx, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrsm_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dA, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dB, ldb, &dx, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dA, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dB, ldb, &dx, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const float* B, int ldb, float* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsm_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb,rocsparse_int nrhs, rocsparse_int nnzb, const float* alpha, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const float* B, rocsparse_int ldb, float* X, rocsparse_int ldx, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrsm_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &fA, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &fB, ldb, &fx, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &fA, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &fB, ldb, &fx, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsm_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrsm_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsm_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrsm_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsm_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrsm_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsm_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrsm_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsm_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zbsrsm_buffer_size(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsm_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cbsrsm_buffer_size(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsm_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dbsrsm_buffer_size(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsm_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_X, rocsparse_int mb, rocsparse_int nrhs, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sbsrsm_buffer_size(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseSbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // TODO: rocsparse_bsrsm_zero_pivot needs explicit synchronization because cusparseXbsrsm2_zeroPivot is blocking
  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_bsrsm_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_bsrsm_zero_pivot(handle_t, bsrsm2_info, &iposition);
  status_t = cusparseXbsrsm2_zeroPivot(handle_t, bsrsm2_info, &iposition);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuDoubleComplex* B, const int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrmm(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int mb, rocsparse_int n, rocsparse_int kb, rocsparse_int nnzb, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_double_complex* B, rocsparse_int ldb, const rocsparse_double_complex* beta, rocsparse_double_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_zbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dcomplexA, matDescr_A, &dcomplex, &bsrRowPtrA, &bsrColIndA, blockDim, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dcomplexA, matDescr_A, &dcomplex, &bsrRowPtrA, &bsrColIndA, blockDim, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuComplex* B, const int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrmm(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int mb, rocsparse_int n, rocsparse_int kb, rocsparse_int nnzb, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_float_complex* B, rocsparse_int ldb, const rocsparse_float_complex* beta, rocsparse_float_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_cbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &complexA, matDescr_A, &complex, &bsrRowPtrA, &bsrColIndA, blockDim, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &complexA, matDescr_A, &complex, &bsrRowPtrA, &bsrColIndA, blockDim, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const double* B, const int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrmm(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int mb, rocsparse_int n, rocsparse_int kb, rocsparse_int nnzb, const double* alpha, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const double* B, rocsparse_int ldb, const double* beta, double* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_dbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dA, matDescr_A, &dbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dA, matDescr_A, &dbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const float* B, const int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrmm(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int mb, rocsparse_int n, rocsparse_int kb, rocsparse_int nnzb, const float* alpha, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const float* B, rocsparse_int ldb, const float* beta, float* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_sbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &fA, matDescr_A, &fbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseSbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &fA, matDescr_A, &fbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &fB, ldb, &fBeta, &fC, ldc);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuDoubleComplex* f, cuDoubleComplex* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsv_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const rocsparse_double_complex* x, rocsparse_double_complex* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrsv_solve(handle_t, direction_t, opA, mb, nnzb, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dcomplexF, &dcomplexX, rocsparse_solve_policy_auto, &pBuffer);
  status_t = cusparseZbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuComplex* f, cuComplex* x,cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsv_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const rocsparse_float_complex* x, rocsparse_float_complex* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrsv_solve(handle_t, direction_t, opA, mb, nnzb, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &complexF, &complexX, rocsparse_solve_policy_auto, &pBuffer);
  status_t = cusparseCbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &complexF, &complexX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const double* f, double* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsv_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const double* alpha, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const double* x, double* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrsv_solve(handle_t, direction_t, opA, mb, nnzb, &dAlpha, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dF, &dX, rocsparse_solve_policy_auto, &pBuffer);
  status_t = cusparseDbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dAlpha, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dF, &dX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const float* f, float* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsv_solve(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const float* alpha, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, const float* x, float* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrsv_solve(handle_t, direction_t, opA, mb, nnzb, &fAlpha, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &fF, &fX, rocsparse_solve_policy_auto, &pBuffer);
  status_t = cusparseSbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &fAlpha, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &fF, &fX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsv_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zbsrsv_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsv_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_cbsrsv_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsv_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dbsrsv_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsv_analysis(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_sbsrsv_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseSbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseZbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseCbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseDbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseSbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrsv_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sbsrsv_buffer_size(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseSbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_bsrsv_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_bsrsv_zero_pivot(handle_t, bsrsv2_info, &iposition);
  status_t = cusparseXbsrsv2_zeroPivot(handle_t, bsrsv2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrxmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int size_of_mask, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_mask_ptr, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_end_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_double_complex* x, const rocsparse_double_complex* beta, rocsparse_double_complex* y);
  // CHECK: status_t = rocsparse_zbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrxmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int size_of_mask, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_mask_ptr, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_end_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_float_complex* x, const rocsparse_float_complex* beta, rocsparse_float_complex* y);
  // CHECK: status_t = rocsparse_cbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &complexX, &complexBeta, &complexY);
  status_t = cusparseCbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrxmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int size_of_mask, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const double* alpha, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_mask_ptr, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_end_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const double* x, const double* beta, double* y);
  // CHECK: status_t = rocsparse_dbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dX, &dBeta, &dY);
  status_t = cusparseDbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrxmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int size_of_mask, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const float* alpha, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_mask_ptr, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_end_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const float* x, const float* beta, float* y);
  // CHECK: status_t = rocsparse_sbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &fX, &fBeta, &fY);
  status_t = cusparseSbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &fX, &fBeta, &fY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // ROC: __attribute__((deprecated("This function is deprecated and will be removed in a future release. " "Use rocsparse_zbsrmv_ex instead."))) ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_double_complex* x, const rocsparse_double_complex* beta, rocsparse_double_complex* y);
  // CHECK: status_t = rocsparse_zbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // ROC: __attribute__((deprecated("This function is deprecated and will be removed in a future release. " "Use rocsparse_cbsrmv_ex instead."))) ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const rocsparse_float_complex* x, const rocsparse_float_complex* beta, rocsparse_float_complex* y);
  // CHECK: status_t = rocsparse_cbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &complexX, &complexBeta, &complexY);
  status_t = cusparseCbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &complexX, &complexBeta, &complexY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y);
  // ROC: __attribute__((deprecated("This function is deprecated and will be removed in a future release. " "Use rocsparse_dbsrmv_ex instead."))) ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const double* alpha, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const double* x, const double* beta, double* y);
  // CHECK: status_t = rocsparse_dbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dX, &dBeta, &dY);
  status_t = cusparseDbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dX, &dBeta, &dY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y);
  // ROC: __attribute__((deprecated("This function is deprecated and will be removed in a future release. " "Use rocsparse_sbsrmv_ex instead."))) ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrmv(rocsparse_handle handle, rocsparse_direction dir, rocsparse_operation trans, rocsparse_int mb, rocsparse_int nb, rocsparse_int nnzb, const float* alpha, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, const float* x, const float* beta, float* y);
  // CHECK: status_t = rocsparse_sbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &fX, &fBeta, &fY);
  status_t = cusparseSbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zbsrilu0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_double_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zbsrilu0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cbsrilu0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const rocsparse_float_complex* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cbsrilu0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED ccusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dbsrilu0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const double* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dbsrilu0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sbsrilu0_buffer_size(rocsparse_handle handle, rocsparse_direction dir, rocsparse_int mb, rocsparse_int nnzb, const rocsparse_mat_descr descr, const float* bsr_val, const rocsparse_int* bsr_row_ptr, const rocsparse_int* bsr_col_ind, rocsparse_int block_dim, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sbsrilu0_buffer_size(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseSbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&csric02_info);
  status_t = cusparseCreateCsric02Info(&csric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(csric02_info);
  status_t = cusparseDestroyCsric02Info(csric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&csrilu02_info);
  status_t = cusparseCreateCsrilu02Info(&csrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(csrilu02_info);
  status_t = cusparseDestroyCsrilu02Info(csrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&bsrsv2_info);
  status_t = cusparseCreateBsrsv2Info(&bsrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(bsrsv2_info);
  status_t = cusparseDestroyBsrsv2Info(bsrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&bsrsm2_info);
  status_t = cusparseCreateBsrsm2Info(&bsrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(bsrsm2_info);
  status_t = cusparseDestroyBsrsm2Info(bsrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&bsric02_info);
  status_t = cusparseCreateBsric02Info(&bsric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(bsric02_info);
  status_t = cusparseDestroyBsric02Info(bsric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&bsrilu02_info);
  status_t = cusparseCreateBsrilu02Info(&bsrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(bsrilu02_info);
  status_t = cusparseDestroyBsrilu02Info(bsrilu02_info);

#if CUDA_VERSION >= 7050
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* beta, cuDoubleComplex* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgemvi(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* alpha, const rocsparse_double_complex* A, rocsparse_int lda, rocsparse_int nnz, const rocsparse_double_complex* x_val, const rocsparse_int* x_ind, const rocsparse_double_complex* beta, rocsparse_double_complex* y, rocsparse_index_base idx_base,void* temp_buffer);
  // CHECK: status_t = rocsparse_zgemvi(handle_t, opA, m, n, &dcomplexAlpha, &dcomplexA, lda, innz, &dcomplexX, &xInd, &dcomplexBeta, &dcomplexY, indexBase_t, pBuffer);
  status_t = cusparseZgemvi(handle_t, opA, m, n, &dcomplexAlpha, &dcomplexA, lda, innz, &dcomplexX, &xInd, &dcomplexBeta, &dcomplexY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* beta, cuComplex* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgemvi(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* alpha, const rocsparse_float_complex* A, rocsparse_int lda, rocsparse_int nnz, const rocsparse_float_complex* x_val, const rocsparse_int* x_ind, const rocsparse_float_complex* beta, rocsparse_float_complex* y, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgemvi(handle_t, opA, m, n, &complexAlpha, &complexA, lda, innz, &complexX, &xInd, &complexBeta, &complexY, indexBase_t, pBuffer);
  status_t = cusparseCgemvi(handle_t, opA, m, n, &complexAlpha, &complexA, lda, innz, &complexX, &xInd, &complexBeta, &complexY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* xVal, const int* xInd, const double* beta, double* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgemvi(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, const double* alpha, const double* A, rocsparse_int lda, rocsparse_int nnz, const double* x_val, const rocsparse_int* x_ind, const double* beta, double* y, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgemvi(handle_t, opA, m, n, &dAlpha, &dA, lda, innz, &dX, &xInd, &dBeta, &dY, indexBase_t, pBuffer);
  status_t = cusparseDgemvi(handle_t, opA, m, n, &dAlpha, &dA, lda, innz, &dX, &xInd, &dBeta, &dY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* xVal, const int* xInd, const float* beta, float* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgemvi(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, const float* alpha, const float* A, rocsparse_int lda, rocsparse_int nnz, const float* x_val, const rocsparse_int* x_ind, const float* beta, float* y, rocsparse_index_base idx_base, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgemvi(handle_t, opA, m, n, &fAlpha, &fA, lda, innz, &fX, &xInd, &fBeta, &fY, indexBase_t, pBuffer);
  status_t = cusparseSgemvi(handle_t, opA, m, n, &fAlpha, &fA, lda, innz, &fX, &xInd, &fBeta, &fY, indexBase_t, pBuffer);
#endif

#if CUDA_VERSION >= 8000
  // TODO: [#899] There should be rocsparse_datatype instead of hipDataType
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, float* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, float tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2csr_compress(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, rocsparse_int nnz_A, const rocsparse_int* nnz_per_row, float* csr_val_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, float tol);
  // CHECK: status_t = rocsparse_scsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);
  status_t = cusparseScsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuDoubleComplex tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_znnz_compress(rocsparse_handle handle, rocsparse_int m, const rocsparse_mat_descr descr_A, const rocsparse_double_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, rocsparse_int* nnz_per_row, rocsparse_int* nnz_C, rocsparse_double_complex tol);
  // CHECK: status_t = rocsparse_znnz_compress(handle_t, m, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dcomplextol);
  status_t = cusparseZnnz_compress(handle_t, m, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dcomplextol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuComplex tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cnnz_compress(rocsparse_handle handle, rocsparse_int m, const rocsparse_mat_descr descr_A, const rocsparse_float_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, rocsparse_int* nnz_per_row, rocsparse_int* nnz_C, rocsparse_float_complex tol);
  // CHECK: status_t = rocsparse_cnnz_compress(handle_t, m, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, complextol);
  status_t = cusparseCnnz_compress(handle_t, m, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, complextol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const double* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, double tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnnz_compress(rocsparse_handle handle, rocsparse_int m, const rocsparse_mat_descr descr_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, rocsparse_int* nnz_per_row, rocsparse_int* nnz_C, double tol);
  // CHECK: status_t = rocsparse_dnnz_compress(handle_t, m, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dtol);
  status_t = cusparseDnnz_compress(handle_t, m, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dtol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const float* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, float tol);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_snnz_compress(rocsparse_handle handle, rocsparse_int m, const rocsparse_mat_descr descr_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, rocsparse_int* nnz_per_row, rocsparse_int* nnz_C, float tol);
  // CHECK: status_t = rocsparse_snnz_compress(handle_t, m, matDescr_A, &csrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, ftol);
  status_t = cusparseSnnz_compress(handle_t, m, matDescr_A, &csrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, ftol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream);
  // CHECK: status_t = rocsparse_get_stream(handle_t, &stream_t);
  status_t = cusparseGetStream(handle_t, &stream_t);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src);
  // CHECK: status_t = rocsparse_copy_mat_descr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);
#endif

#if CUDA_VERSION >= 9000
  // CHECK: rocsparse_mat_info prune_info;
  pruneInfo_t prune_info;

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

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC,int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, double percentage, const rocsparse_mat_descr descr, double* csr_val, const rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_dense2csr_by_percentage(handle_t, m, n, &dA, lda, percentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneDense2csrByPercentage(handle_t, m, n, &dA, lda, percentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, float percentage, const rocsparse_mat_descr descr, float* csr_val, const rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_dense2csr_by_percentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneDense2csrByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, double percentage, const rocsparse_mat_descr descr, rocsparse_int* csr_row_ptr, rocsparse_int* nnz_total_dev_host_ptr, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_dense2csr_nnz_by_percentage(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);
  status_t = cusparseDpruneDense2csrNnzByPercentage(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, float percentage, const rocsparse_mat_descr descr, rocsparse_int* csr_row_ptr, rocsparse_int* nnz_total_dev_host_ptr, rocsparse_mat_info info, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_dense2csr_nnz_by_percentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);
  status_t = cusparseSpruneDense2csrNnzByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr_by_percentage_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, double percentage, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr_by_percentage_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, float percentage, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sprune_dense2csr_by_percentage_buffer_size(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, const double* threshold, const rocsparse_mat_descr descr, double* csr_val, const rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_dense2csr(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseDpruneDense2csr(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, const float* threshold, const rocsparse_mat_descr descr, float* csr_val, const rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_dense2csr(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseSpruneDense2csr(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, const double* threshold, const rocsparse_mat_descr descr, rocsparse_int* csr_row_ptr, rocsparse_int* nnz_total_dev_host_ptr, void* temp_buffer);
  // CHECK: status_t = rocsparse_dprune_dense2csr_nnz(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseDpruneDense2csrNnz(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, const float* threshold, const rocsparse_mat_descr descr, rocsparse_int* csr_row_ptr, rocsparse_int* nnz_total_dev_host_ptr, void* temp_buffer);
  // CHECK: status_t = rocsparse_sprune_dense2csr_nnz(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseSpruneDense2csrNnz(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dprune_dense2csr_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* A, rocsparse_int lda, const double* threshold, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dprune_dense2csr_buffer_size(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseDpruneDense2csr_bufferSizeExt(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sprune_dense2csr_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* A, rocsparse_int lda, const float* threshold, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sprune_dense2csr_buffer_size(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseSpruneDense2csr_bufferSizeExt(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_no_pivot(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, rocsparse_double_complex* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgtsv_no_pivot(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);
  status_t = cusparseZgtsv2_nopivot(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_no_pivot(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, rocsparse_float_complex* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgtsv_no_pivot(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);
  status_t = cusparseCgtsv2_nopivot(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORTrocsparse_status rocsparse_dgtsv_no_pivot(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* dl, const double* d, const double* du, double* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgtsv_no_pivot(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);
  status_t = cusparseDgtsv2_nopivot(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_no_pivot(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* dl, const float* d, const float* du, float* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgtsv_no_pivot(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);
  status_t = cusparseSgtsv2_nopivot(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_no_pivot_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, const rocsparse_double_complex* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgtsv_no_pivot_buffer_size(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);
  status_t = cusparseZgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_no_pivot_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, const rocsparse_float_complex* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgtsv_no_pivot_buffer_size(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);
  status_t = cusparseCgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_no_pivot_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* dl, const double* d, const double* du, const double* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgtsv_no_pivot_buffer_size(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);
  status_t = cusparseDgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_no_pivot_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* dl, const float* d, const float* du, const float* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgtsv_no_pivot_buffer_size(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);
  status_t = cusparseSgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, rocsparse_double_complex* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgtsv(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);
  status_t = cusparseZgtsv2(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, rocsparse_float_complex* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgtsv(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);
  status_t = cusparseCgtsv2(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* dl, const double* d, const double* du, double* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgtsv(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);
  status_t = cusparseDgtsv2(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* dl, const float* d, const float* du, float* B, rocsparse_int ldb, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgtsv(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);
  status_t = cusparseSgtsv2(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, const rocsparse_double_complex* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgtsv_buffer_size(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);
  status_t = cusparseZgtsv2_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, const rocsparse_float_complex* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgtsv_buffer_size(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);
  status_t = cusparseCgtsv2_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* dl, const double* d, const double* du, const double* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgtsv_buffer_size(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);
  status_t = cusparseDgtsv2_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* dl, const float* d, const float* du, const float* B, rocsparse_int ldb, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgtsv_buffer_size(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);
  status_t = cusparseSgtsv2_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, int batchStride, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_no_pivot_strided_batch(rocsparse_handle handle, rocsparse_int m, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgtsv_no_pivot_strided_batch(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseZgtsv2StridedBatch(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* x, int batchCount, int batchStride, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_no_pivot_strided_batch(rocsparse_handle handle, rocsparse_int m, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgtsv_no_pivot_strided_batch(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseCgtsv2StridedBatch(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_no_pivot_strided_batch(rocsparse_handle handle, rocsparse_int m, const double* dl, const double* d, const double* du, double* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgtsv_no_pivot_strided_batch(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseDgtsv2StridedBatch(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_no_pivot_strided_batch(rocsparse_handle handle, rocsparse_int m, const float* dl, const float* d, const float* du, float* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgtsv_no_pivot_strided_batch(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseSgtsv2StridedBatch(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle, rocsparse_int m, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, const rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseZgtsv2StridedBatch_bufferSizeExt(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle, rocsparse_int m, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, const rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseCgtsv2StridedBatch_bufferSizeExt(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle, rocsparse_int m, const double* dl, const double* d, const double* du, const double* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseDgtsv2StridedBatch_bufferSizeExt(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(rocsparse_handle handle, rocsparse_int m, const float* dl, const float* d, const float* du, const float* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseSgtsv2StridedBatch_bufferSizeExt(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&prune_info);
  status_t = cusparseCreatePruneInfo(&prune_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(prune_info);
  status_t = cusparseDestroyPruneInfo(prune_info);
#endif

#if (CUDA_VERSION >= 10010 && CUSPARSE_VERSION >= 10200 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: _rocsparse_spmat_descr *spMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_spmat_descr spMatDescr_t, matC, spmatA, spmatB, spmatC;
  cusparseSpMatDescr *spMatDescr = nullptr;
  cusparseSpMatDescr_t spMatDescr_t, matC, spmatA, spmatB, spmatC;

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
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t cscColOffsetsType;
  cusparseIndexType_t cscRowIndType;
  cusparseIndexType_t csrColIndType;
  cusparseIndexType_t ellIdxType;
  cusparseIndexType_t INDEX_16U = CUSPARSE_INDEX_16U;
  cusparseIndexType_t INDEX_32I = CUSPARSE_INDEX_32I;

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr, int64_t rows, int64_t cols, int64_t ld, void* values, rocsparse_datatype data_type, rocsparse_order order);
  // CHECK: status_t = rocsparse_create_dnmat_descr(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_set_strided_batch(rocsparse_dnmat_descr descr, int batch_count, int64_t batch_stride);
  // CHECK: status_t = rocsparse_dnmat_set_strided_batch(dnMatDescr_t, batchCount, batchStride);
  status_t = cusparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);

#if CUSPARSE_VERSION >= 10200
  // CHECK: rocsparse_indextype INDEX_64I = rocsparse_indextype_i64;
  cusparseIndexType_t INDEX_64I = CUSPARSE_INDEX_64I;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, void** coo_row_ind, void** coo_col_ind, void** coo_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_coo_get(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, rocsparse_datatype* data_type, rocsparse_order* order);
  // CHECK: status_t = rocsparse_dnmat_get(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);
  status_t = cusparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);
#endif
#endif

#if CUDA_VERSION >= 10020
  // CHECK: rocsparse_status STATUS_NOT_SUPPORTED = rocsparse_status_not_implemented;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;
#endif

#if (CUDA_VERSION >= 10020 && CUSPARSE_VERSION >= 10200 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr, int64_t* size, int64_t* nnz, void** indices, void** values, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_spvec_get(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr, int64_t size, void* values, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_dnvec_descr(&dnVecDescr_t, size, values, dataType);
  status_t = cusparseCreateDnVec(&dnVecDescr_t, size, values, dataType);

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

#if CUDA_VERSION < 12000
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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get_index_base(const rocsparse_spvec_descr descr, rocsparse_index_base* idx_base);
  // CHECK: status_t = rocsparse_spvec_get_index_base(spVecDescr_t, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);
#endif
#endif

#if CUDA_VERSION >= 10020 && CUSPARSE_VERSION >= 10301
  // CUDA: const char* CUSPARSEAPI cusparseGetErrorName(cusparseStatus_t status);
  // ROC: ROCSPARSE_EXPORT const char* rocsparse_get_status_name(rocsparse_status status);
  // CHECK: const_ch = rocsparse_get_status_name(status_2);
  const_ch = cusparseGetErrorName(status_2);

  // CUDA: const char* CUSPARSEAPI cusparseGetErrorString(cusparseStatus_t status);
  // ROC: ROCSPARSE_EXPORT const char* rocsparse_get_status_description(rocsparse_status status);
  // CHECK: const_ch = rocsparse_get_status_description(status_2);
  const_ch = cusparseGetErrorString(status_2);
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

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
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

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_double_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_double_complex* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_double_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_double_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_zcsrgeam(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZcsrgeam(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_float_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_float_complex* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_float_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_float_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_ccsrgeam(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCcsrgeam(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const double* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const double* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, double* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_dcsrgeam(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDcsrgeam(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseScsrgeam(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const float* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const float* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, float* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_scsrgeam(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseScsrgeam(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseXcsrgeamNnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_C);
  // CHECK: status_t = rocsparse_csrgeam_nnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr);
  status_t = cusparseXcsrgeamNnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zhybmv(rocsparse_handle handle, rocsparse_operation trans, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_hyb_mat hyb, const rocsparse_double_complex* x, const rocsparse_double_complex* beta, rocsparse_double_complex* y);
  // CHECK: status_t = rocsparse_zhybmv(handle_t, opA, &dcomplexAlpha, matDescr_A, hybMat_t, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZhybmv(handle_t, opA, &dcomplexAlpha, matDescr_A, hybMat_t, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseChybmv(cusparseHandle_t handle, cusparseOperation_t transA, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_chybmv(rocsparse_handle handle, rocsparse_operation trans, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_hyb_mat hyb, const rocsparse_float_complex* x, const rocsparse_float_complex* beta, rocsparse_float_complex* y);
  // CHECK: status_t = rocsparse_chybmv(handle_t, opA, &complexAlpha, matDescr_A, hybMat_t, &complexX, &complexBeta, &complexY);
  status_t = cusparseChybmv(handle_t, opA, &complexAlpha, matDescr_A, hybMat_t, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const double* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const double* x, const double* beta, double* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dhybmv(rocsparse_handle handle, rocsparse_operation trans, const double* alpha, const rocsparse_mat_descr descr, const rocsparse_hyb_mat hyb, const double* x, const double* beta, double* y);
  // CHECK: status_t = rocsparse_dhybmv(handle_t, opA, &dAlpha, matDescr_A, hybMat_t, &dX, &dBeta, &dY);
  status_t = cusparseDhybmv(handle_t, opA, &dAlpha, matDescr_A, hybMat_t, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseShybmv(cusparseHandle_t handle, cusparseOperation_t transA, const float* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const float* x, const float* beta, float* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_shybmv(rocsparse_handle handle, rocsparse_operation trans, const float* alpha, const rocsparse_mat_descr descr, const rocsparse_hyb_mat hyb, const float* x, const float* beta, float* y);
  // CHECK: status_t = rocsparse_shybmv(handle_t, opA, &fAlpha, matDescr_A, hybMat_t, &fX, &fBeta, &fY);
  status_t = cusparseShybmv(handle_t, opA, &fAlpha, matDescr_A, hybMat_t, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseZdotci(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* y, cuDoubleComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zdotci(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_double_complex* x_val, const rocsparse_int* x_ind, const rocsparse_double_complex* y, rocsparse_double_complex* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zdotci(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);
  status_t = cusparseZdotci(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseCdotci(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* y, cuComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cdotci(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_float_complex* x_val, const rocsparse_int* x_ind, const rocsparse_float_complex* y, rocsparse_float_complex* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_cdotci(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);
  status_t = cusparseCdotci(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseZdoti(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* y, cuDoubleComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zdoti(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_double_complex* x_val, const rocsparse_int* x_ind, const rocsparse_double_complex* y, rocsparse_double_complex* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zdoti(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);
  status_t = cusparseZdoti(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseCdoti(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* y, cuComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cdoti(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_float_complex* x_val, const rocsparse_int* x_ind, const rocsparse_float_complex* y, rocsparse_float_complex* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_cdoti(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);
  status_t = cusparseCdoti(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseDdoti(cusparseHandle_t handle, int nnz, const double* xVal, const int* xInd, const double* y, double* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ddoti(rocsparse_handle handle, rocsparse_int nnz, const double* x_val, const rocsparse_int* x_ind, const double* y, double* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_ddoti(handle_t, innz, &dX, &xInd, &dY, &d_resultDevHostPtr, indexBase_t);
  status_t = cusparseDdoti(handle_t, innz, &dX, &xInd, &dY, &d_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseSdoti(cusparseHandle_t handle, int nnz, const float* xVal, const int* xInd, const float* y, float* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sdoti(rocsparse_handle handle, rocsparse_int nnz, const float* x_val, const rocsparse_int* x_ind, const float* y, float* result, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_sdoti(handle_t, innz, &fX, &xInd, &fY, &f_resultDevHostPtr, indexBase_t);
  status_t = cusparseSdoti(handle_t, innz, &fX, &xInd, &fY, &f_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_double_complex* B, rocsparse_int ldb, const rocsparse_double_complex* beta, rocsparse_double_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_zcsrmm(handle_t, opA, opB, m, n, k, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZcsrmm2(handle_t, opA, opB, m, n, k, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_float_complex* B, rocsparse_int ldb, const rocsparse_float_complex* beta, rocsparse_float_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_ccsrmm(handle_t, opA, opB, m, n, k, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCcsrmm2(handle_t, opA, opB, m, n, k, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* B, rocsparse_int ldb, const double* beta, double* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_dcsrmm(handle_t, opA, opB, m, n, k, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDcsrmm2(handle_t, opA, opB, m, n, k, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseScsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* B, rocsparse_int ldb, const float* beta, float* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_scsrmm(handle_t, opA, opB, m, n, k, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseScsrmm2(handle_t, opA, opB, m, n, k, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
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

#if CUDA_VERSION >= 11070
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cscColOffsets, void** cscRowInd, void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, void** csc_col_ptr, void** csc_row_ind, void** csc_val, rocsparse_indextype* col_ptr_type, rocsparse_indextype* row_ind_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_csc_get(spmatA, &rows, &cols, &nnz, &cscColOffsets, &cscRowInd, &cscValues, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
  status_t = cusparseCscGet(spmatA, &rows, &cols, &nnz, &cscColOffsets, &cscRowInd, &cscValues, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
#endif

#if CUDA_VERSION < 12000
  // CHECK: rocsparse_mat_info csrgemm2_info;
  csrgemm2Info_t csrgemm2_info;
  // CHECK: rocsparse_mat_descr csrsv2_info;
  csrsv2Info_t csrsv2_info;

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuDoubleComplex* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsc2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* csc_val, const rocsparse_int* csc_col_ptr, const rocsparse_int* csc_row_ind, rocsparse_double_complex* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_zcsc2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);
  status_t = cusparseZcsc2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuComplex* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsc2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* csc_val, const rocsparse_int* csc_col_ptr, const rocsparse_int* csc_row_ind, rocsparse_float_complex* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_ccsc2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);
  status_t = cusparseCcsc2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, double* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsc2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* csc_val, const rocsparse_int* csc_col_ptr, const rocsparse_int* csc_row_ind, double* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_dcsc2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);
  status_t = cusparseDcsc2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, float* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsc2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* csc_val, const rocsparse_int* csc_col_ptr, const rocsparse_int* csc_row_ind, float* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_scsc2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);
  status_t = cusparseScsc2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsr2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_double_complex* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_zcsr2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);
  status_t = cusparseZcsr2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsr2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind,rocsparse_float_complex* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_ccsr2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);
  status_t = cusparseCcsr2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsr2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, double* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_dcsr2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);
  status_t = cusparseDcsr2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* A, int lda);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsr2dense(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, float* A, rocsparse_int ld);
  // CHECK: status_t = rocsparse_scsr2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);
  status_t = cusparseScsr2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, const int* nnzPerCol, cuDoubleComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zdense2csc(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* A, rocsparse_int ld, const rocsparse_int* nnz_per_columns, rocsparse_double_complex* csc_val, rocsparse_int* csc_col_ptr, rocsparse_int* csc_row_ind);
  // CHECK: status_t = rocsparse_zdense2csc(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerCol, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseZdense2csc(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerCol, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, const int* nnzPerCol, cuComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cdense2csc(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* A, rocsparse_int ld, const rocsparse_int* nnz_per_columns, rocsparse_float_complex* csc_val, rocsparse_int* csc_col_ptr, rocsparse_int* csc_row_ind);
  // CHECK: status_t = rocsparse_cdense2csc(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerCol, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseCdense2csc(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerCol, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, const int* nnzPerCol, double* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ddense2csc(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* A, rocsparse_int ld, const rocsparse_int* nnz_per_columns, double* csc_val, rocsparse_int* csc_col_ptr, rocsparse_int* csc_row_ind);
  // CHECK: status_t = rocsparse_ddense2csc(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerCol, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseDdense2csc(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerCol, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerCol, float* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sdense2csc(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* A, rocsparse_int ld, const rocsparse_int* nnz_per_columns, float* csc_val, rocsparse_int* csc_col_ptr, rocsparse_int* csc_row_ind);
  // CHECK: status_t = rocsparse_sdense2csc(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerCol, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseSdense2csc(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerCol, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, const int* nnzPerRow, cuDoubleComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zdense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_double_complex* A, rocsparse_int ld, const rocsparse_int* nnz_per_rows, rocsparse_double_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_zdense2csr(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRow, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseZdense2csr(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRow, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, const int* nnzPerRow, cuComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cdense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const rocsparse_float_complex* A, rocsparse_int ld, const rocsparse_int* nnz_per_rows, rocsparse_float_complex* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_cdense2csr(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerRow, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseCdense2csr(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerRow, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, const int* nnzPerRow, double* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ddense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const double* A, rocsparse_int ld, const rocsparse_int* nnz_per_rows, double* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_ddense2csr(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerRow, &dcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseDdense2csr(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerRow, &dcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerRow, float* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sdense2csr(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr, const float* A, rocsparse_int ld, const rocsparse_int* nnz_per_rows, float* csr_val, rocsparse_int* csr_row_ptr, rocsparse_int* csr_col_ind);
  // CHECK: status_t = rocsparse_sdense2csr(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerRow, &csrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseSdense2csr(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerRow, &csrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Nnz(cusparseHandle_t handle, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, const csrgemm2Info_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrgemm_nnz(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const rocsparse_mat_descr descr_A,ocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A,const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, const rocsparse_mat_descr descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_C, const rocsparse_mat_info info_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_csrgemm_nnz(handle_t, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, csrgemm2_info, pBuffer);
  status_t = cusparseXcsrgemm2Nnz(handle_t, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrgemm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_double_complex* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, rocsparse_mat_info info_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrgemm_buffer_size(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseZcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrgemm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_float_complex* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, rocsparse_mat_info info_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrgemm_buffer_size(handle_t, m, n, k, &complexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseCcsrgemm2_bufferSizeExt(handle_t, m, n, k, &complexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrgemm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const double* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const double* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, rocsparse_mat_info info_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrgemm_buffer_size(handle_t, m, n, k, &dA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseDcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrgemm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const float* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const float* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, rocsparse_mat_info info_C, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrgemm_buffer_size(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseScsrgemm2_bufferSizeExt(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseZcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseCcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseDcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseScsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle, csrsv2Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrsv_zero_pivot(rocsparse_handle handle, const rocsparse_mat_descr descr, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_csrsv_zero_pivot(handle_t, csrsv2_info, &iposition);
  status_t = cusparseXcsrsv2_zeroPivot(handle_t, csrsv2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zsctr(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_double_complex* x_val, const rocsparse_int* x_ind, rocsparse_double_complex* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zsctr(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, indexBase_t);
  status_t = cusparseZsctr(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csctr(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_float_complex* x_val, const rocsparse_int* x_ind, rocsparse_float_complex* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_csctr(handle_t, innz, &complexX, &xInd, &complexY, indexBase_t);
  status_t = cusparseCsctr(handle_t, innz, &complexX, &xInd, &complexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, int nnz, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dsctr(rocsparse_handle handle, rocsparse_int nnz, const double* x_val, const rocsparse_int* x_ind, double* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_dsctr(handle_t, innz, &dX, &xInd, &dY, indexBase_t);
  status_t = cusparseDsctr(handle_t, innz, &dX, &xInd, &dY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, int nnz, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ssctr(rocsparse_handle handle, rocsparse_int nnz, const float* x_val, const rocsparse_int* x_ind, float* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_ssctr(handle_t, innz, &fX, &xInd, &fY, indexBase_t);
  status_t = cusparseSsctr(handle_t, innz, &fX, &xInd, &fY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseRot) cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, int nnz, double* xVal, const int* xInd, double* y, const double* c, const double* s, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_droti(rocsparse_handle handle, rocsparse_int nnz, double* x_val, const rocsparse_int* x_ind, double* y, const double* c, const double* s, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_droti(handle_t, innz, &dX, &xInd, &dY, &dC, &dS, indexBase_t);
  status_t = cusparseDroti(handle_t, innz, &dX, &xInd, &dY, &dC, &dS, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseRot) cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, int nnz, float* xVal, const int* xInd, float* y, const float* c, const float* s, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sroti(rocsparse_handle handle, rocsparse_int nnz, float* x_val, const rocsparse_int* x_ind, float* y, const float* c, const float* s, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_sroti(handle_t, innz, &fX, &xInd, &fY, &fC, &fS, indexBase_t);
  status_t = cusparseSroti(handle_t, innz, &fX, &xInd, &fY, &fC, &fS, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, int nnz, cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgthrz(rocsparse_handle handle, rocsparse_int nnz, rocsparse_double_complex* y, rocsparse_double_complex* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zgthrz(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);
  status_t = cusparseZgthrz(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgthrz(rocsparse_handle handle, rocsparse_int nnz, rocsparse_float_complex* y, rocsparse_float_complex* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_cgthrz(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);
  status_t = cusparseCgthrz(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, int nnz, double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgthrz(rocsparse_handle handle, rocsparse_int nnz, double* y, double* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_dgthrz(handle_t, innz, &dY, &dX, &xInd, indexBase_t);
  status_t = cusparseDgthrz(handle_t, innz, &dY, &dX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, int nnz, float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgthrz(rocsparse_handle handle, rocsparse_int nnz, float* y, float* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_sgthrz(handle_t, innz, &fY, &fX, &xInd, indexBase_t);
  status_t = cusparseSgthrz(handle_t, innz, &fY, &fX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgthr(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_double_complex* y, rocsparse_double_complex* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zgthr(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);
  status_t = cusparseZgthr(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, int nnz, const cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgthr(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_float_complex* y, rocsparse_float_complex* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_cgthr(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);
  status_t = cusparseCgthr(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, int nnz, const double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgthr(rocsparse_handle handle, rocsparse_int nnz, const double* y, double* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_dgthr(handle_t, innz, &dY, &dX, &xInd, indexBase_t);
  status_t = cusparseDgthr(handle_t, innz, &dY, &dX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, int nnz, const float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgthr(rocsparse_handle handle, rocsparse_int nnz, const float* y, float* x_val, const rocsparse_int* x_ind, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_sgthr(handle_t, innz, &fY, &fX, &xInd, indexBase_t);
  status_t = cusparseSgthr(handle_t, innz, &fY, &fX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, int nnz, const cuDoubleComplex* alpha, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zaxpyi(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_double_complex* x_val, const rocsparse_int* x_ind, rocsparse_double_complex* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_zaxpyi(handle_t, innz, &dcomplexAlpha, &dcomplexX, &xInd, &dcomplexY, indexBase_t);
  status_t = cusparseZaxpyi(handle_t, innz, &dcomplexAlpha, &dcomplexX, &xInd, &dcomplexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, int nnz, const cuComplex* alpha, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_caxpyi(rocsparse_handle handle, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_float_complex* x_val, const rocsparse_int* x_ind, rocsparse_float_complex* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_caxpyi(handle_t, innz, &complexAlpha, &complexX, &xInd, &complexY, indexBase_t);
  status_t = cusparseCaxpyi(handle_t, innz, &complexAlpha, &complexX, &xInd, &complexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double* alpha, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_daxpyi(rocsparse_handle handle, rocsparse_int nnz, const double* alpha, const double* x_val, const rocsparse_int* x_ind, double* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_daxpyi(handle_t, innz, &dAlpha, &dX, &xInd, &dY, indexBase_t);
  status_t = cusparseDaxpyi(handle_t, innz, &dAlpha, &dX, &xInd, &dY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_saxpyi(rocsparse_handle handle, rocsparse_int nnz, const float* alpha, const float* x_val, const rocsparse_int* x_ind, float* y, rocsparse_index_base idx_base);
  // CHECK: status_t = rocsparse_saxpyi(handle_t, innz, &fAlpha, &fX, &xInd, &fY, indexBase_t);
  status_t = cusparseSaxpyi(handle_t, innz, &fAlpha, &fX, &xInd, &fY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&csrsv2_info);
  status_t = cusparseCreateCsrsv2Info(&csrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(csrsv2_info);
  status_t = cusparseDestroyCsrsv2Info(csrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&csrgemm2_info);
  status_t = cusparseCreateCsrgemm2Info(&csrgemm2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(csrgemm2_info);
  status_t = cusparseDestroyCsrgemm2Info(csrgemm2_info);
#endif

#if CUDA_VERSION >= 12000
  // CHECK: rocsparse_const_spvec_descr constSpVecDescr = nullptr;
  cusparseConstSpVecDescr_t constSpVecDescr = nullptr;

  // CHECK: rocsparse_const_spmat_descr constSpMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_const_spmat_descr constSpMatDescrB = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescr = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescrB = nullptr;

  // CHECK: rocsparse_const_dnvec_descr constDnVecDescr = nullptr;
  cusparseConstDnVecDescr_t constDnVecDescr = nullptr;

  // CHECK: rocsparse_const_dnmat_descr constDnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_const_dnmat_descr constDnMatDescrB = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescr = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescrB = nullptr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_spvec_descr(rocsparse_const_spvec_descr* descr, int64_t size, int64_t nnz, const void* indices, const void* values, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_spvec_descr(&constSpVecDescr, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateConstSpVec(&constSpVecDescr, size, nnz, indices, values, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_const_spvec_descr descr);
  // CHECK: status_t = rocsparse_destroy_spvec_descr(constSpVecDescr);
  status_t = cusparseDestroySpVec(constSpVecDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpVecGet(cusparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, const void** indices, const void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_spvec_get(rocsparse_const_spvec_descr descr, int64_t* size, int64_t* nnz, const void** indices, const void** values, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_spvec_get(constSpVecDescr, &size, &nnz, indices_const, values_const, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseConstSpVecGet(constSpVecDescr, &size, &nnz, indices_const, values_const, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvec_get_index_base(rocsparse_const_spvec_descr descr, rocsparse_index_base* idx_base);
  // CHECK: status_t = rocsparse_spvec_get_index_base(constSpVecDescr, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(constSpVecDescr, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t spVecDescr, const void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_spvec_get_values(rocsparse_const_spvec_descr descr, const void** values);
  // CHECK: status_t = rocsparse_const_spvec_get_values(constSpVecDescr, values_const);
  status_t = cusparseConstSpVecGetValues(constSpVecDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCoo(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, const void* coo_row_ind, const void* coo_col_ind, const void* coo_val, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_coo_descr(&constSpMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateConstCoo(&constSpMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, const void* csr_row_ptr, const void* csr_col_ind, const void* csr_val, rocsparse_indextype row_ptr_type, rocsparse_indextype col_ind_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_csr_descr(&constSpMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateConstCsr(&constSpMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr, int64_t rows, int64_t cols, int64_t nnz, const void* csc_col_ptr, const void* csc_row_ind, const void* csc_val, rocsparse_indextype col_ptr_type, rocsparse_indextype row_ind_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_csc_descr(&constSpMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, indexBase_t, dataType);
  status_t = cusparseCreateConstCsc(&constSpMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr, int64_t rows, int64_t cols, rocsparse_direction ell_block_dir, int64_t ell_block_dim, int64_t ell_cols, const void* ell_col_ind, const void* ell_val, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_bell_descr(&constSpMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);
  status_t = cusparseCreateConstBlockedEll(&constSpMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr);
  // CHECK: status_t = rocsparse_destroy_spmat_descr(constSpMatDescr);
  status_t = cusparseDestroySpMat(constSpMatDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCooGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cooRowInd, const void** cooColInd, const void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** coo_row_ind, const void** coo_col_ind, const void** coo_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_coo_get(constSpMatDescr, &rows, &cols, &nnz, cooRowInd_const, cooColInd_const, cooValues_const, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseConstCooGet(constSpMatDescr, &rows, &cols, &nnz, cooRowInd_const, cooColInd_const, cooValues_const, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csrRowOffsets, const void** csrColInd, const void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csr_row_ptr, const void** csr_col_ind, const void** csr_val, rocsparse_indextype* row_ptr_type, rocsparse_indextype* col_ind_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_csr_get(constSpMatDescr, &rows, &cols, &nnz, csrRowOffsets_const, csrColInd_const, csrValues_const, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);
  status_t = cusparseConstCsrGet(constSpMatDescr, &rows, &cols, &nnz, csrRowOffsets_const, csrColInd_const, csrValues_const, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCscGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cscColOffsets, const void** cscRowInd, const void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csc_col_ptr, const void** csc_row_ind, const void** csc_val, rocsparse_indextype* col_ptr_type, rocsparse_indextype* row_ind_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_csc_get(constSpMatDescr, &rows, &cols, &nnz, cscColOffsets_const, cscRowInd_const, cscValues_const, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
  status_t = cusparseConstCscGet(constSpMatDescr, &rows, &cols, &nnz, cscColOffsets_const, cscRowInd_const, cscValues_const, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstBlockedEllGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, const void** ellColInd, const void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr, int64_t* rows, int64_t* cols, rocsparse_direction* ell_block_dir, int64_t* ell_block_dim, int64_t* ell_cols, const void** ell_col_ind, const void** ell_val, rocsparse_indextype* idx_type, rocsparse_index_base* idx_base, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_bell_get(constSpMatDescr, &rows, &cols, &ellBlockSize, &ellCols, ellColInd_const, ellValue_const, &ellIdxType, &indexBase_t, &dataType);
  status_t = cusparseConstBlockedEllGet(constSpMatDescr, &rows, &cols, &ellBlockSize, &ellCols, ellColInd_const, ellValue_const, &ellIdxType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // CHECK: status_t = rocsparse_spmat_get_size(constSpMatDescr, &rows, &cols, &nnz);
  status_t = cusparseSpMatGetSize(constSpMatDescr, &rows, &cols, &nnz);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr, rocsparse_format* format);
  // CHECK: status_t = rocsparse_spmat_get_format(constSpMatDescr, &format_t);
  status_t = cusparseSpMatGetFormat(constSpMatDescr, &format_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr, rocsparse_index_base* idx_base);
  // CHECK: status_t = rocsparse_spmat_get_index_base(constSpMatDescr, &indexBase_t);
  status_t = cusparseSpMatGetIndexBase(constSpMatDescr, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr, const void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr, const void** values);
  // CHECK: status_t = rocsparse_const_spmat_get_values(constSpMatDescr, values_const);
  status_t = cusparseConstSpMatGetValues(constSpMatDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr, int* batch_count);
  // CHECK: status_t = rocsparse_spmat_get_strided_batch(constSpMatDescr, &batchCount);
  status_t = cusparseSpMatGetStridedBatch(constSpMatDescr, &batchCount);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr, rocsparse_spmat_attribute attribute, void* data, size_t data_size);
  // CHECK: status_t = rocsparse_spmat_get_attribute(constSpMatDescr, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatGetAttribute(constSpMatDescr, spMatAttribute_t, &data, dataSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_dnvec_descr(rocsparse_const_dnvec_descr* descr, int64_t size, const void* values, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_const_dnvec_descr(&constDnVecDescr, size, values, dataType);
  status_t = cusparseCreateConstDnVec(&constDnVecDescr, size, values, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_const_dnvec_descr descr);
  // CHECK: status_t = rocsparse_destroy_dnvec_descr(constDnVecDescr);
  status_t = cusparseDestroyDnVec(constDnVecDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnVecGet(cusparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values, cudaDataType* valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_dnvec_get(rocsparse_const_dnvec_descr descr, int64_t* size, const void** values, rocsparse_datatype* data_type);
  // CHECK: status_t = rocsparse_const_dnvec_get(constDnVecDescr, &size, values_const, &dataType);
  status_t = cusparseConstDnVecGet(constDnVecDescr, &size, values_const, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t dnVecDescr, const void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_dnvec_get_values(rocsparse_const_dnvec_descr descr, const void** values);
  // CHECK: status_t = rocsparse_const_dnvec_get_values(constDnVecDescr, values_const);
  status_t = cusparseConstDnVecGetValues(constDnVecDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, cudaDataType valueType, cusparseOrder_t order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_const_dnmat_descr(rocsparse_const_dnmat_descr* descr, int64_t rows, int64_t cols, int64_t ld, const void* values, rocsparse_datatype data_type, rocsparse_order order);
  // CHECK: status_t = rocsparse_create_const_dnmat_descr(&constDnMatDescr, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateConstDnMat(&constDnMatDescr, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_const_dnmat_descr descr);
  // CHECK: status_t = rocsparse_destroy_dnmat_descr(constDnMatDescr);
  status_t = cusparseDestroyDnMat(constDnMatDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnMatGet(cusparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, cudaDataType* type, cusparseOrder_t* order);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_dnmat_get(rocsparse_const_dnmat_descr descr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, rocsparse_datatype* data_type, rocsparse_order* order);
  // CHECK: status_t = rocsparse_const_dnmat_get(constDnMatDescr, &rows, &cols, &ld, values_const, &dataType, &order_t);
  status_t = cusparseConstDnMatGet(constDnMatDescr, &rows, &cols, &ld, values_const, &dataType, &order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t dnMatDescr, const void** values);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_const_dnmat_get_values(rocsparse_const_dnmat_descr descr, const void** values);
  // CHECK: status_t = rocsparse_const_dnmat_get_values(constDnMatDescr, values_const);
  status_t = cusparseConstDnMatGetValues(constDnMatDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dnmat_get_strided_batch(rocsparse_const_dnmat_descr descr, int* batch_count, int64_t* batch_stride);
  // CHECK: status_t = rocsparse_dnmat_get_strided_batch(constDnMatDescr, &batchCount, &batchStride);
  status_t = cusparseDnMatGetStridedBatch(constDnMatDescr, &batchCount, &batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scatter(rocsparse_handle handle, rocsparse_const_spvec_descr x, rocsparse_dnvec_descr y);
  // CHECK: status_t = rocsparse_scatter(handle_t, constSpVecDescr, vecY);
  status_t = cusparseScatter(handle_t, constSpVecDescr, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_gather(rocsparse_handle handle, rocsparse_const_dnvec_descr y, rocsparse_spvec_descr x);
  // CHECK: status_t = rocsparse_gather(handle_t, constDnVecDescr, spVecDescr_t);
  status_t = cusparseGather(handle_t, constDnVecDescr, spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_axpby(rocsparse_handle handle, const void* alpha, rocsparse_const_spvec_descr x, const void* beta, rocsparse_dnvec_descr y);
  // CHECK: status_t = rocsparse_axpby(handle_t, alpha, constSpVecDescr, beta, vecY);
  status_t = cusparseAxpby(handle_t, alpha, constSpVecDescr, beta, vecY);
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
