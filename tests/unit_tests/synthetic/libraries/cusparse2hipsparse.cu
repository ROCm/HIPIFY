// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --skip-excluded-preprocessor-conditional-blocks --experimental --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "hipsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "hipsparse.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("17. cuSPARSE API to hipSPARSE API synthetic test\n");

  // CHECK: hipsparseHandle_t handle_t;
  cusparseHandle_t handle_t;

  // CHECK: hipsparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

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

  // CHECK: hipsparseAction_t action_t, copyValues;
  // CHECK-NEXT: hipsparseAction_t ACTION_SYMBOLIC = HIPSPARSE_ACTION_SYMBOLIC;
  // CHECK-NEXT: hipsparseAction_t ACTION_NUMERIC = HIPSPARSE_ACTION_NUMERIC;
  cusparseAction_t action_t, copyValues;
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

  // CHECK: hipsparseStatus_t status_t, status_2;
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
  cusparseStatus_t status_t, status_2;
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
  int cscRowIndB = 0;
  int cscColPtrA = 0;
  int cscColPtrB = 0;
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
  void *csrColInd = nullptr;
  const void** const csrColInd_const = const_cast<const void**>(&csrColInd);
  void *cooColInd = nullptr;
  const void** const cooColInd_const = const_cast<const void**>(&cooColInd);
  void *ellColInd = nullptr;
  const void** const ellColInd_const = const_cast<const void**>(&ellColInd);
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
  void *pcsrVal = nullptr;
  void *pcscVal = nullptr;
  int *P = nullptr;
  void *tempBuffer = nullptr;
  void *tempBuffer1 = nullptr;
  void *tempBuffer2 = nullptr;
  void *tempBuffer3 = nullptr;
  void *tempBuffer4 = nullptr;
  void *tempBuffer5 = nullptr;
  void *c_coeff = nullptr;
  void *s_coeff = nullptr;
  void *workspace = nullptr;
  size_t dataSize = 0;
  size_t bufferSize = 0;
  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  size_t bufferSize3 = 0;
  size_t bufferSize4 = 0;
  size_t bufferSize5 = 0;
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
  float csrSortedValD = 0.f;
  double dcsrSortedVal = 0.f;
  double dcscSortedVal = 0.f;
  double dcsrSortedValA = 0.f;
  double dcsrSortedValB = 0.f;
  double dcsrSortedValC = 0.f;
  double dcsrSortedValD = 0.f;
  double dbsrSortedVal = 0.f;
  double dbsrSortedValA = 0.f;
  double dbsrSortedValC = 0.f;
  float fbsrSortedVal = 0.f;
  float fbsrSortedValA = 0.f;
  float fbsrSortedValC = 0.f;
  float fcsrSortedValC = 0.f;
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
  void* result = nullptr;
  csrilu02Info_t csrilu02_info;
  csric02Info_t csric02_info;
  bsrilu02Info_t bsrilu02_info;
  bsric02Info_t bsric02_info;
  bsrsm2Info_t bsrsm2_info;
  bsrsv2Info_t bsrsv2_info;
  csru2csrInfo_t csru2_info;

  // CHECK: hipDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val, dcomplex_resultDevHostPtr;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val, dcomplex_resultDevHostPtr;

  // CHECK: hipComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val, complex_resultDevHostPtr;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val, complex_resultDevHostPtr;

  // CHECK: hipsparseOperation_t opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

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
  // CHECK: status_t = hipsparseXgebsr2gebsrNnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);
  status_t = cusparseXgebsr2gebsrNnz(handle_t, direction_t, mb, nb, nnzb, matDescr_t, &bsrRowPtrA, &bsrColIndA, rowBlockDimA, colBlockDimA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, &nnzTotalDevHostPtr, &tempBuffer);

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const hipDoubleComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, const hipsparseMatDescr_t bsr_descr, hipDoubleComplex* bsr_val, int* bsr_row_ptr, int* bsr_col_ind, int row_block_dim, int col_block_dim, void* p_buffer);
  // CHECK: status_t = hipsparseZcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcomplex, &csrRowPtrA, &csrColIndA, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseZcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcomplex, &csrRowPtrA, &csrColIndA, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const hipComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, const hipsparseMatDescr_t bsr_descr, hipComplex* bsr_val, int* bsr_row_ptr, int* bsr_col_ind, int row_block_dim, int col_block_dim, void* p_buffer);
  // CHECK: status_t = hipsparseCcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseCcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const double* csr_val, const int* csr_row_ptr, const int* csr_col_ind, const hipsparseMatDescr_t bsr_descr, double* bsr_val, int* bsr_row_ptr, int* bsr_col_ind, int row_block_dim, int col_block_dim, void* p_buffer);
  // CHECK: status_t = hipsparseDcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseDcsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC, int rowBlockDim, int colBlockDim, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const float* csr_val, const int* csr_row_ptr, const int* csr_col_ind, const hipsparseMatDescr_t bsr_descr, float* bsr_val, int* bsr_row_ptr, int* bsr_col_ind, int row_block_dim, int col_block_dim, void* p_buffer);
  // CHECK: status_t = hipsparseScsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);
  status_t = cusparseScsr2gebsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC, rowBlockDimA, colBlockDimA, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2gebsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int rowBlockDim, int colBlockDim, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const int* csr_row_ptr, const int* csr_col_ind, const hipsparseMatDescr_t bsr_descr, int* bsr_row_ptr, int row_block_dim, int col_block_dim, int* bsr_nnz_devhost, void* p_buffer);
  // CHECK: status_t = hipsparseXcsr2gebsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimA, colBlockDimA, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseXcsr2gebsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, matDescr_C, &bsrSortedRowPtrC, rowBlockDimA, colBlockDimA, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const hipDoubleComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseZcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseZcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const hipComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseCcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseCcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const double* csr_val, const int* csr_row_ptr, const int* csr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseDcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseDcsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dir, int m, int n, const hipsparseMatDescr_t csr_descr, const float* csr_val, const int* csr_row_ptr, const int* csr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseScsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);
  status_t = cusparseScsr2gebsr_bufferSize(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, rowBlockDimA, colBlockDimA, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuDoubleComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2bsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const hipsparseMatDescr_t descrC, hipDoubleComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC);
  // CHECK: status_t = hipsparseZcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseZcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dComplexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, cuComplex* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2bsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const hipComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const hipsparseMatDescr_t descrC, hipComplex* bsrValC, int* bsrRowPtrC, int* bsrColIndC);
  // CHECK: status_t = hipsparseCcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseCcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &complexcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, double* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2bsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n,const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim,const hipsparseMatDescr_t descrC, double* bsrValC, int* bsrRowPtrC, int* bsrColIndC);
  // CHECK: status_t = hipsparseDcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseDcsr2bsr(handle_t, direction_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &dcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2bsr(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, float* bsrSortedValC, int* bsrSortedRowPtrC, int* bsrSortedColIndC);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2bsr(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const hipsparseMatDescr_t descrC, float* bsrValC, int* bsrRowPtrC, int* bsrColIndC);
  // CHECK: status_t = hipsparseScsr2bsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);
  status_t = cusparseScsr2bsr(handle_t, direction_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &fcsrSortedValC, &bsrSortedRowPtrC, &bsrSortedColIndC);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2bsrNnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, int blockDim, const cusparseMatDescr_t descrC, int* bsrSortedRowPtrC, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsr2bsrNnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const int* csrRowPtrA, const int* csrColIndA, int blockDim, const hipsparseMatDescr_t descrC, int* bsrRowPtrC, int* bsrNnzb);
  // CHECK: status_t = hipsparseXcsr2bsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &bsrSortedRowPtrC, &nnzTotalDevHostPtr);
  status_t = cusparseXcsr2bsrNnz(handle_t, direction_t, m, n, matDescr_A, &csrRowPtrA, &csrColIndA, blockDim, matDescr_C, &bsrSortedRowPtrC, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t handle, int mb, int nb, int nnzb, const hipDoubleComplex* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, hipDoubleComplex* bsc_val, int* bsc_row_ind, int* bsc_col_ptr, hipsparseAction_t copy_values, hipsparseIndexBase_t idx_base, void* temp_buffer);
  // CHECK: status_t = hipsparseZgebsr2gebsc(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dComplexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseZgebsr2gebsc(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dComplexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t handle, int mb, int nb, int nnzb, const hipComplex* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, hipComplex* bsc_val, int* bsc_row_ind, int* bsc_col_ptr,hipsparseAction_t copy_values, hipsparseIndexBase_t idx_base, void* temp_buffer);
  // CHECK: status_t = hipsparseCgebsr2gebsc(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &complexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseCgebsr2gebsc(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &complexbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, double* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t handle, int mb, int nb, int nnzb, const double* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, double* bsc_val, int* bsc_row_ind, int* bsc_col_ptr, hipsparseAction_t copy_values, hipsparseIndexBase_t idx_base, void* temp_buffer);
  // CHECK: status_t = hipsparseDgebsr2gebsc(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseDgebsr2gebsc(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &dbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, float* bscVal, int* bscRowInd, int* bscColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t handle, int mb, int nb, int nnzb, const float* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, float* bsc_val, int* bsc_row_ind, int* bsc_col_ptr, hipsparseAction_t copy_values, hipsparseIndexBase_t idx_base, void* temp_buffer);
  // CHECK: status_t = hipsparseSgebsr2gebsc(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &fbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);
  status_t = cusparseSgebsr2gebsc(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &fbscVal, &bscRowInd, &bscColPtr, copyValues, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t handle, int mb, int nb, int nnzb, const hipDoubleComplex* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseZgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseZgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle, int mb, int nb, int nnzb, const hipComplex* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseCgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseCgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle, int mb, int nb, int nnzb, const double* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseDgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseDgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle, int mb, int nb, int nnzb, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int rowBlockDim, int colBlockDim, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle, int mb, int nb, int nnzb, const float* bsr_val, const int* bsr_row_ptr, const int* bsr_col_ind, int row_block_dim, int col_block_dim, size_t* p_buffer_size);
  // CHECK: status_t = hipsparseSgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);
  status_t = cusparseSgebsr2gebsc_bufferSize(handle_t, mb, nb, nnzb, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, rowBlockDim, colBlockDim, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz, int m, int* cooRowInd, cusparseIndexBase_t idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle, const int* csrRowPtr, int nnz, int m, int* cooRowInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseXcsr2coo(handle_t, &csrSortedRowPtr, nnz, m, &icooRowInd, indexBase_t);
  status_t = cusparseXcsr2coo(handle_t, &csrSortedRowPtr, nnz, m, &icooRowInd, indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const hipDoubleComplex* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseZnnz(handle_t, direction_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseZnnz(handle_t, direction_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const hipComplex* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseCnnz(handle_t, direction_t, m, n, matDescr_A, &complexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseCnnz(handle_t, direction_t, m, n, matDescr_A, &complexA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const double* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseDnnz(handle_t, direction_t, m, n, matDescr_A, &dA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseDnnz(handle_t, direction_t, m, n, matDescr_A, &dA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSnnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowCol, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const float* A, int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseSnnz(handle_t, direction_t, m, n, matDescr_A, &fA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);
  status_t = cusparseSnnz(handle_t, direction_t, m, n, matDescr_A, &fA, lda, &nnzPerRowCol, &nnzTotalDevHostPtr);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrilu02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrilu02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrilu02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrilu02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrilu02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrilu02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrilu02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsrilu02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrilu02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrilu02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrilu02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrilu02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsrilu02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);
  status_t = cusparseZcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);
  status_t = cusparseCcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);
  status_t = cusparseDcsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);
  status_t = cusparseScsrilu02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseZcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseCcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseDcsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csrilu02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrilu02Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);
  status_t = cusparseScsrilu02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrilu02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, hipDoubleComplex* boost_val);
  // CHECK: status_t = hipsparseZcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);
  status_t = cusparseZcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, hipComplex* boost_val);
  // CHECK: status_t = hipsparseCcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &complex_boost_val);
  status_t = cusparseCcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &complex_boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrilu02_numericBoost(hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // CHECK: status_t = hipsparseDcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dboost_val);
  status_t = cusparseDcsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &dboost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrilu02_numericBoost(cusparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrilu02_numericBoost(hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // CHECK: status_t = hipsparseScsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &boost_val);
  status_t = cusparseScsrilu02_numericBoost(handle_t, csrilu02_info, ienable_boost, &dtol, &boost_val);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position);
  // CHECK: status_t = hipsparseXcsrilu02_zeroPivot(handle_t, csrilu02_info, &iposition);
  status_t = cusparseXcsrilu02_zeroPivot(handle_t, csrilu02_info, &iposition);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsric02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsric02(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsric02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsric02(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsric02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsric02(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsric02(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsric02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsric02(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsric02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsric02_analysis(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsric02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsric02_analysis(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsric02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsric02_analysis(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsric02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsric02_analysis(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);
  status_t = cusparseZcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsric02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);
  status_t = cusparseCcsric02_bufferSize(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);
  status_t = cusparseDcsric02_bufferSize(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSize(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsric02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);
  status_t = cusparseScsric02_bufferSize(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseZcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseZcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseCcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseCcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseDcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseDcsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsric02_bufferSizeExt(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, csric02Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t handle, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csric02Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseScsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);
  status_t = cusparseScsric02_bufferSizeExt(handle_t, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csric02_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position);
  // CHECK: status_t = hipsparseXcsric02_zeroPivot(handle_t, csric02_info, &iposition);
  status_t = cusparseXcsric02_zeroPivot(handle_t, csric02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA_valM, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA_valM, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA_valM, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA_valM, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrilu02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position);
  // CHECK: status_t = hipsparseXbsrilu02_zeroPivot(handle_t, bsrilu02_info, &iposition);
  status_t = cusparseXbsrilu02_zeroPivot(handle_t, bsrilu02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrilu02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuDoubleComplex* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, hipDoubleComplex* boost_val);
  // CHECK: status_t = hipsparseZbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);
  status_t = cusparseZbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dcomplex_boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, cuComplex* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, hipComplex* boost_val);
  // CHECK: status_t = hipsparseCbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &complex_boost_val);
  status_t = cusparseCbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &complex_boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrilu02_numericBoost(hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val);
  // CHECK: status_t = hipsparseDbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dboost_val);
  status_t = cusparseDbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &dboost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_numericBoost(cusparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrilu02_numericBoost(hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);
  // CHECK: status_t = hipsparseSbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &boost_val);
  status_t = cusparseSbsrilu02_numericBoost(handle_t, bsrilu02_info, ienable_boost, &dtol, &boost_val);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsric02(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, cusparseSolvePolicy_t policy, void* pInputBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsric02_analysis(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);
  status_t = cusparseZbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);
  status_t = cusparseCbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);
  status_t = cusparseDbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsric02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, bsric02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);
  status_t = cusparseSbsric02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsric02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position);
  // CHECK: status_t = hipsparseXbsric02_zeroPivot(handle_t, bsric02_info, &iposition);
  status_t = cusparseXbsric02_zeroPivot(handle_t, bsric02_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuDoubleComplex* B, int ldb, cuDoubleComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, const hipDoubleComplex* B, int ldb, hipDoubleComplex* X, int ldx, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dcomplexA, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dcomplexB, ldb, &dcomplexX, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dcomplexA, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dcomplexB, ldb, &dcomplexX, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuComplex* B, int ldb, cuComplex* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, const hipComplex* B, int ldb, hipComplex* X, int ldx, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &complexA, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &complexB, ldb, &complexX, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &complexA, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &complexB, ldb, &complexX, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const double* B, int ldb, double* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const double* alpha, const hipsparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, const double* B, int ldb, double* X, int ldx, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dA, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dB, ldb, &dx, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &dA, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &dB, ldb, &dx, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, const float* B, int ldb, float* X, int ldx, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const float* alpha, const hipsparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, const float* B, int ldb, float* X, int ldx, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &fA, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &fB, ldb, &fx, ldx, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrsm2_solve(handle_t, direction_t, opA, opX, mb, nrhs, nnzb, &fA, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &fB, ldb, &fx, ldx, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, const hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrsm2_analysis(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);
  status_t = cusparseZbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);
  status_t = cusparseCbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);
  status_t = cusparseDbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsm2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transXY, int mb, int n, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockSize, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transX, int mb, int nrhs, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsm2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);
  status_t = cusparseSbsrsm2_bufferSize(handle_t, direction_t, opA, opX, mb, n, nnzb, matDescr_A, &fbsrSortedVal, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsm2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle, bsrsm2Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position);
  // CHECK: status_t = hipsparseXbsrsm2_zeroPivot(handle_t, bsrsm2_info, &iposition);
  status_t = cusparseXbsrsm2_zeroPivot(handle_t, bsrsm2_info, &iposition);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuDoubleComplex* B, const int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n, int kb, int nnzb, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipDoubleComplex* B, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, int ldc);
  // CHECK: status_t = hipsparseZbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dcomplexA, matDescr_A, &dcomplex, &bsrRowPtrA, &bsrColIndA, blockDim, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dcomplexA, matDescr_A, &dcomplex, &bsrRowPtrA, &bsrColIndA, blockDim, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const cuComplex* B, const int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n, int kb, int nnzb, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const hipComplex* B, int ldb, const hipComplex* beta, hipComplex* C, int ldc);
  // CHECK: status_t = hipsparseCbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &complexA, matDescr_A, &complex, &bsrRowPtrA, &bsrColIndA, blockDim, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &complexA, matDescr_A, &complex, &bsrRowPtrA, &bsrColIndA, blockDim, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const double* B, const int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double* alpha, const hipsparseMatDescr_t descrA, const double* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const double* B, int ldb, const double* beta, double* C, int ldc);
  // CHECK: status_t = hipsparseDbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dA, matDescr_A, &dbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &dA, matDescr_A, &dbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, const int blockSize, const float* B, const int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float* alpha, const hipsparseMatDescr_t descrA, const float* bsrValA, const int* bsrRowPtrA, const int* bsrColIndA, int blockDim, const float* B, int ldb, const float* beta, float* C, int ldc);
  // CHECK: status_t = hipsparseSbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &fA, matDescr_A, &fbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseSbsrmm(handle_t, direction_t, opA, opB, mb, n, kb, nnzb, &fA, matDescr_A, &fbscVal, &bsrRowPtrA, &bsrColIndA, blockDim, &fB, ldb, &fBeta, &fC, ldc);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuDoubleComplex* f, cuDoubleComplex* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const hipDoubleComplex* f, hipDoubleComplex* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, &pBuffer);
  status_t = cusparseZbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuComplex* f, cuComplex* x,cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const hipComplex* f, hipComplex* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &complexF, &complexX, solvePolicy_t, &pBuffer);
  status_t = cusparseCbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &complexF, &complexX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const double* f, double* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const double* alpha, const hipsparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const double* f, double* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dAlpha, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dF, &dX, solvePolicy_t, &pBuffer);
  status_t = cusparseDbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &dAlpha, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &dF, &dX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_solve(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const float* f, float* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const float* alpha, const hipsparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const float* f, float* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &fAlpha, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &fF, &fX, solvePolicy_t, &pBuffer);
  status_t = cusparseSbsrsv2_solve(handle_t, direction_t, opA, mb, nnzb, &fAlpha, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, blockDim, bsrsv2_info, &fF, &fX, solvePolicy_t, &pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_analysis(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseSbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseSbsrsv2_analysis(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseZbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseZbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseCbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseCbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseDbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseDbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseSbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);
  status_t = cusparseSbsrsv2_bufferSizeExt(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);
  status_t = cusparseZbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dComplexbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);
  status_t = cusparseCbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &complexbsrValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);
  status_t = cusparseDbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrsv2_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);
  status_t = cusparseSbsrsv2_bufferSize(handle_t, direction_t, opA, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrRowPtrA, &bsrColIndA, blockDim, bsrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle, bsrsv2Info_t info, int* position);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position);
  // CHECK: status_t = hipsparseXbsrsv2_zeroPivot(handle_t, bsrsv2_info, &iposition);
  status_t = cusparseXbsrsv2_zeroPivot(handle_t, bsrsv2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrxmv(hipsparseHandle_t handle, hipsparseDirection_t dir, hipsparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descr, const hipDoubleComplex* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrEndPtr, const int* bsrColInd, int blockDim, const hipDoubleComplex* x, const hipDoubleComplex* beta, hipDoubleComplex* y);
  // CHECK: status_t = hipsparseZbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrxmv(hipsparseHandle_t handle, hipsparseDirection_t dir, hipsparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const hipComplex* alpha, const hipsparseMatDescr_t descr, const hipComplex* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrEndPtr, const int* bsrColInd, int blockDim, const hipComplex* x, const hipComplex* beta, hipComplex* y);
  // CHECK: status_t = hipsparseCbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &complexX, &complexBeta, &complexY);
  status_t = cusparseCbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrxmv(hipsparseHandle_t handle, hipsparseDirection_t dir, hipsparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const double* alpha, const hipsparseMatDescr_t descr, const double* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrEndPtr, const int* bsrColInd, int blockDim, const double* x, const double* beta, double* y);
  // CHECK: status_t = hipsparseDbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dX, &dBeta, &dY);
  status_t = cusparseDbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrxmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedMaskPtrA, const int* bsrSortedRowPtrA, const int* bsrSortedEndPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrxmv(hipsparseHandle_t handle, hipsparseDirection_t dir, hipsparseOperation_t trans, int sizeOfMask, int mb, int nb, int nnzb, const float* alpha, const hipsparseMatDescr_t descr, const float* bsrVal, const int* bsrMaskPtr, const int* bsrRowPtr, const int* bsrEndPtr, const int* bsrColInd, int blockDim, const float* x, const float* beta, float* y);
  // CHECK: status_t = hipsparseSbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &fX, &fBeta, &fY);
  status_t = cusparseSbsrxmv(handle_t, direction_t, opA, sizeOfMask, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, &bsrEndPtrA, &bsrColIndA, blockDim, &fX, &fBeta, &fY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nb, int nnzb, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const hipDoubleComplex* x, const hipDoubleComplex* beta, hipDoubleComplex* y);
  // CHECK: status_t = hipsparseZbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dcomplexAlpha, matDescr_t, &dComplexbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nb, int nnzb, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const hipComplex* x, const hipComplex* beta, hipComplex* y);
  // CHECK: status_t = hipsparseCbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &complexX, &complexBeta, &complexY);
  status_t = cusparseCbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &complexAlpha, matDescr_t, &complexbsrValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &complexX, &complexBeta, &complexY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const double* alpha, const cusparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nb, int nnzb, const double* alpha, const hipsparseMatDescr_t descrA, const double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const double* x, const double* beta, double* y);
  // CHECK: status_t = hipsparseDbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dX, &dBeta, &dY);
  status_t = cusparseDbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &dAlpha, matDescr_t, &dbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &dX, &dBeta, &dY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dirA, cusparseOperation_t transA, int mb, int nb, int nnzb, const float* alpha, const cusparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t handle, hipsparseDirection_t dirA, hipsparseOperation_t transA, int mb, int nb, int nnzb, const float* alpha, const hipsparseMatDescr_t descrA, const float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, const float* x, const float* beta, float* y);
  // CHECK: status_t = hipsparseSbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &fX, &fBeta, &fY);
  status_t = cusparseSbsrmv(handle_t, direction_t, opA, mb, nb, nnzb, &fAlpha, matDescr_t, &fbsrSortedValA, &bsrSortedMaskPtrA, &bsrRowPtrA, blockDim, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuDoubleComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipDoubleComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);
  status_t = cusparseZbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, cuComplex* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, hipComplex* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);
  status_t = cusparseCbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &complexbsrValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED ccusparseStatus_t CUSPARSEAPI cusparseDbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, double* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, double* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);
  status_t = cusparseDbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &dbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSbsrilu02_bufferSize(cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb, const cusparseMatDescr_t descrA, float* bsrSortedVal, const int* bsrSortedRowPtr, const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t handle, hipsparseDirection_t dirA, int mb, int nnzb, const hipsparseMatDescr_t descrA, float* bsrSortedValA, const int* bsrSortedRowPtrA, const int* bsrSortedColIndA, int blockDim, bsrilu02Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);
  status_t = cusparseSbsrilu02_bufferSize(handle_t, direction_t, mb, nnzb, matDescr_A, &fbsrSortedValA, &bsrSortedRowPtr, &bsrSortedColInd, blockDim, bsrilu02_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateCsric02Info(csric02Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info);
  // CHECK: status_t = hipsparseCreateCsric02Info(&csric02_info);
  status_t = cusparseCreateCsric02Info(&csric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyCsric02Info(csric02Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info);
  // CHECK: status_t = hipsparseDestroyCsric02Info(csric02_info);
  status_t = cusparseDestroyCsric02Info(csric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02Info(csrilu02Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info);
  // CHECK: status_t = hipsparseCreateCsrilu02Info(&csrilu02_info);
  status_t = cusparseCreateCsrilu02Info(&csrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02Info(csrilu02Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info);
  // CHECK: status_t = hipsparseDestroyCsrilu02Info(csrilu02_info);
  status_t = cusparseDestroyCsrilu02Info(csrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsv2Info(bsrsv2Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info);
  // CHECK: status_t = hipsparseCreateBsrsv2Info(&bsrsv2_info);
  status_t = cusparseCreateBsrsv2Info(&bsrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsv2Info(bsrsv2Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info);
  // CHECK: status_t = hipsparseDestroyBsrsv2Info(bsrsv2_info);
  status_t = cusparseDestroyBsrsv2Info(bsrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrsm2Info(bsrsm2Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info);
  // CHECK: status_t = hipsparseCreateBsrsm2Info(&bsrsm2_info);
  status_t = cusparseCreateBsrsm2Info(&bsrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrsm2Info(bsrsm2Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info);
  // CHECK: status_t = hipsparseDestroyBsrsm2Info(bsrsm2_info);
  status_t = cusparseDestroyBsrsm2Info(bsrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsric02Info(bsric02Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info);
  // CHECK: status_t = hipsparseCreateBsric02Info(&bsric02_info);
  status_t = cusparseCreateBsric02Info(&bsric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsric02Info(bsric02Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info);
  // CHECK: status_t = hipsparseDestroyBsric02Info(bsric02_info);
  status_t = cusparseDestroyBsric02Info(bsric02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateBsrilu02Info(bsrilu02Info_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info);
  // CHECK: status_t = hipsparseCreateBsrilu02Info(&bsrilu02_info);
  status_t = cusparseCreateBsrilu02Info(&bsrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyBsrilu02Info(bsrilu02Info_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info);
  // CHECK: status_t = hipsparseDestroyBsrilu02Info(bsrilu02_info);
  status_t = cusparseDestroyBsrilu02Info(bsrilu02_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreateCsru2csrInfo(csru2csrInfo_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info);
  // CHECK: status_t = hipsparseCreateCsru2csrInfo(&csru2_info);
  status_t = cusparseCreateCsru2csrInfo(&csru2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyCsru2csrInfo(csru2csrInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info);
  // CHECK: status_t = hipsparseDestroyCsru2csrInfo(csru2_info);
  status_t = cusparseDestroyCsru2csrInfo(csru2_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseZcsru2csr(handle_t, m, n, innz, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseZcsru2csr(handle_t, m, n, innz, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseCcsru2csr(handle_t, m, n, innz, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseCcsru2csr(handle_t, m, n, innz, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseDcsru2csr(handle_t, m, n, innz, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseDcsru2csr(handle_t, m, n, innz, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsru2csr(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseScsru2csr(handle_t, m, n, innz, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseScsru2csr(handle_t, m, n, innz, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, cuDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, hipDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsru2csr_bufferSizeExt(handle_t, m, n, innz, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);
  status_t = cusparseZcsru2csr_bufferSizeExt(handle_t, m, n, innz, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, cuComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, hipComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsru2csr_bufferSizeExt(handle_t, m, n, innz, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);
  status_t = cusparseCcsru2csr_bufferSizeExt(handle_t, m, n, innz, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsru2csr_bufferSizeExt(handle_t, m, n, innz, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);
  status_t = cusparseDcsru2csr_bufferSizeExt(handle_t, m, n, innz, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnz, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsru2csr_bufferSizeExt(handle_t, m, n, innz, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);
  status_t = cusparseScsru2csr_bufferSizeExt(handle_t, m, n, innz, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsr2csru(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseZcsr2csru(handle_t, m, n, innz, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseZcsr2csru(handle_t, m, n, innz, matDescr_A, &dComplexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsr2csru(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseCcsr2csru(handle_t, m, n, innz, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseCcsr2csru(handle_t, m, n, innz, matDescr_A, &complexbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsr2csru(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, double* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseDcsr2csru(handle_t, m, n, innz, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseDcsr2csru(handle_t, m, n, innz, matDescr_A, &dbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsr2csru(cusparseHandle_t handle, int m, int n, int nnz, const cusparseMatDescr_t descrA, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t handle, int m, int n, int nnz, const hipsparseMatDescr_t descrA, float* csrVal, const int* csrRowPtr, int* csrColInd, csru2csrInfo_t info, void* pBuffer);
  // CHECK: status_t = hipsparseScsr2csru(handle_t, m, n, innz, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);
  status_t = cusparseScsr2csru(handle_t, m, n, innz, matDescr_A, &fbsrSortedVal, &bsrSortedRowPtr, &bsrSortedColInd, csru2_info, pBuffer);

#if CUDA_VERSION >= 7050
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* beta, cuDoubleComplex* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda, int nnz, const hipDoubleComplex* x, const int* xInd, const hipDoubleComplex* beta, hipDoubleComplex* y, hipsparseIndexBase_t idxBase, void* pBuffer);
  // CHECK: status_t = hipsparseZgemvi(handle_t, opA, m, n, &dcomplexAlpha, &dcomplexA, lda, innz, &dcomplexX, &xInd, &dcomplexBeta, &dcomplexY, indexBase_t, pBuffer);
  status_t = cusparseZgemvi(handle_t, opA, m, n, &dcomplexAlpha, &dcomplexA, lda, innz, &dcomplexX, &xInd, &dcomplexBeta, &dcomplexY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* beta, cuComplex* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, const hipComplex* alpha, const hipComplex* A, int lda, int nnz, const hipComplex* x, const int* xInd, const hipComplex* beta, hipComplex* y, hipsparseIndexBase_t idxBase, void* pBuffer);
  // CHECK: status_t = hipsparseCgemvi(handle_t, opA, m, n, &complexAlpha, &complexA, lda, innz, &complexX, &xInd, &complexBeta, &complexY, indexBase_t, pBuffer);
  status_t = cusparseCgemvi(handle_t, opA, m, n, &complexAlpha, &complexA, lda, innz, &complexX, &xInd, &complexBeta, &complexY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* xVal, const int* xInd, const double* beta, double* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, const double* alpha, const double* A, int lda, int nnz, const double* x, const int* xInd, const double* beta, double* y, hipsparseIndexBase_t idxBase, void* pBuffer);
  // CHECK: status_t = hipsparseDgemvi(handle_t, opA, m, n, &dAlpha, &dA, lda, innz, &dX, &xInd, &dBeta, &dY, indexBase_t, pBuffer);
  status_t = cusparseDgemvi(handle_t, opA, m, n, &dAlpha, &dA, lda, innz, &dX, &xInd, &dBeta, &dY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgemvi(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* xVal, const int* xInd, const float* beta, float* y, cusparseIndexBase_t idxBase, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, const float* alpha, const float* A, int lda, int nnz, const float* x, const int* xInd, const float* beta, float* y, hipsparseIndexBase_t idxBase, void* pBuffer);
  // CHECK: status_t = hipsparseSgemvi(handle_t, opA, m, n, &fAlpha, &fA, lda, innz, &fX, &xInd, &fBeta, &fY, indexBase_t, pBuffer);
  status_t = cusparseSgemvi(handle_t, opA, m, n, &fAlpha, &fA, lda, innz, &fX, &xInd, &fBeta, &fY, indexBase_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgemvi_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // CHECK: status_t = hipsparseZgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
  status_t = cusparseZgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgemvi_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // CHECK: status_t = hipsparseCgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
  status_t = cusparseCgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgemvi_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // CHECK: status_t = hipsparseDgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
  status_t = cusparseDgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgemvi_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // CHECK: status_t = hipsparseSgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
  status_t = cusparseSgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
#endif

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuDoubleComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuDoubleComplex tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrValA, const int* csrColIndA, const int* csrRowPtrA, int nnzA, const int* nnzPerRow, hipDoubleComplex* csrValC, int* csrColIndC, int* csrRowPtrC, hipDoubleComplex tol);
  // CHECK: status_t = hipsparseZcsr2csr_compress(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dComplexcsrSortedValC, &csrColIndC, &csrRowPtrC, dcomplextol);
  status_t = cusparseZcsr2csr_compress(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dComplexcsrSortedValC, &csrColIndC, &csrRowPtrC, dcomplextol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, cuComplex* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, cuComplex tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const hipComplex* csrValA, const int* csrColIndA, const int* csrRowPtrA, int nnzA, const int* nnzPerRow, hipComplex* csrValC, int* csrColIndC, int* csrRowPtrC, hipComplex tol);
  // CHECK: status_t = hipsparseCcsr2csr_compress(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &complexcsrSortedValC, &csrColIndC, &csrRowPtrC, complextol);
  status_t = cusparseCcsr2csr_compress(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &complexcsrSortedValC, &csrColIndC, &csrRowPtrC, complextol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, double* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, double tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrColIndA, const int* csrRowPtrA, int nnzA, const int* nnzPerRow, double* csrValC, int* csrColIndC, int* csrRowPtrC, double tol);
  // CHECK: status_t = hipsparseDcsr2csr_compress(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dcsrSortedValC, &csrColIndC, &csrRowPtrC, dtol);
  status_t = cusparseDcsr2csr_compress(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &dcsrSortedValC, &csrColIndC, &csrRowPtrC, dtol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsr2csr_compress(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA, int nnzA, const int* nnzPerRow, float* csrSortedValC, int* csrSortedColIndC, int* csrSortedRowPtrC, float tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrColIndA, const int* csrRowPtrA, int nnzA, const int* nnzPerRow, float* csrValC, int* csrColIndC, int* csrRowPtrC, float tol);
  // CHECK: status_t = hipsparseScsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);
  status_t = cusparseScsr2csr_compress(handle_t, m, n, matDescr_A, &csrSortedValA, &csrColIndA, &csrRowPtrA, nnza, &nnzPerRow, &fcsrSortedValC, &csrColIndC, &csrRowPtrC, ftol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuDoubleComplex tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t handle, int m, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, int* nnzPerRow, int* nnzC, hipDoubleComplex tol);
  // CHECK: status_t = hipsparseZnnz_compress(handle_t, m, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dcomplextol);
  status_t = cusparseZnnz_compress(handle_t, m, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dcomplextol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, cuComplex tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t handle, int m, const hipsparseMatDescr_t descrA, const hipComplex* csrValA, const int* csrRowPtrA, int* nnzPerRow, int* nnzC, hipComplex tol);
  // CHECK: status_t = hipsparseCnnz_compress(handle_t, m, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, complextol);
  status_t = cusparseCnnz_compress(handle_t, m, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, complextol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const double* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, double tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t handle, int m, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, int* nnzPerRow, int* nnzC, double tol);
  // CHECK: status_t = hipsparseDnnz_compress(handle_t, m, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dtol);
  status_t = cusparseDnnz_compress(handle_t, m, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, dtol);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSnnz_compress(cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const float* csrSortedValA, const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, float tol);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t handle, int m, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, int* nnzPerRow, int* nnzC, float tol);
  // CHECK: status_t = hipsparseSnnz_compress(handle_t, m, matDescr_A, &csrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, ftol);
  status_t = cusparseSnnz_compress(handle_t, m, matDescr_A, &csrSortedValA, &csrRowPtrA, &nnzPerRow, &nnzc, ftol);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId);
  // CHECK: status_t = hipsparseGetStream(handle_t, &stream_t);
  status_t = cusparseGetStream(handle_t, &stream_t);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src);
  // CHECK: status_t = hipsparseCopyMatDescr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseZgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* cscValB, const int* cscColPtrB, const int* cscRowIndB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t handle, int m, int n, int k, int nnz, const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda, const hipDoubleComplex* cscValB, const int* cscColPtrB, const int* cscRowIndB, const hipDoubleComplex* beta, hipDoubleComplex* C, int ldc);
  // CHECK: status_t = hipsparseZgemmi(handle_t, m, n, k, innz, &dcomplexAlpha, &dcomplexA, lda, &dcomplexB, &cscColPtrB, &cscRowIndB, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZgemmi(handle_t, m, n, k, innz, &dcomplexAlpha, &dcomplexA, lda, &dcomplexB, &cscColPtrB, &cscRowIndB, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseCgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* cscValB, const int* cscColPtrB, const int* cscRowIndB, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle, int m, int n, int k, int nnz, const hipComplex* alpha, const hipComplex* A, int lda, const hipComplex* cscValB, const int* cscColPtrB, const int* cscRowIndB, const hipComplex* beta, hipComplex* C, int ldc);
  // CHECK: status_t = hipsparseCgemmi(handle_t, m, n, k, innz, &complexAlpha, &complexA, lda, &complexB, &cscColPtrB, &cscRowIndB, &complexBeta, &complexC, ldc);
  status_t = cusparseCgemmi(handle_t, m, n, k, innz, &complexAlpha, &complexA, lda, &complexB, &cscColPtrB, &cscRowIndB, &complexBeta, &complexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseDgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha, const double* A, int lda, const double* cscValB, const int* cscColPtrB, const int* cscRowIndB, const double* beta, double* C, int ldc);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha, const double* A, int lda, const double* cscValB, const int* cscColPtrB, const int* cscRowIndB, const double* beta, double* C, int ldc);
  // CHECK: status_t = hipsparseDgemmi(handle_t, m, n, k, innz, &dAlpha, &dA, lda, &dB, &cscColPtrB, &cscRowIndB, &dBeta, &dC, ldc);
  status_t = cusparseDgemmi(handle_t, m, n, k, innz, &dAlpha, &dA, lda, &dB, &cscColPtrB, &cscRowIndB, &dBeta, &dC, ldc);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseSgemmi(cusparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha, const float* A, int lda, const float* cscValB, const int* cscColPtrB, const int* cscRowIndB, const float* beta, float* C, int ldc);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha, const float* A, int lda, const float* cscValB, const int* cscColPtrB, const int* cscRowIndB, const float* beta, float* C, int ldc);
  // CHECK: status_t = hipsparseSgemmi(handle_t, m, n, k, innz, &fAlpha, &fA, lda, &fB, &cscColPtrB, &cscRowIndB, &fBeta, &fC, ldc);
  status_t = cusparseSgemmi(handle_t, m, n, k, innz, &fAlpha, &fA, lda, &fB, &cscColPtrB, &cscRowIndB, &fBeta, &fC, ldc);
#endif

#if CUDA_VERSION >= 9000
  pruneInfo_t prune_info;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double percentage, const hipsparseMatDescr_t descrC, double* csrValC, const int* csrRowPtrC, int* csrColIndC, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseDpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &csrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, float percentage, const hipsparseMatDescr_t descrC, float* csrValC, const int* csrRowPtrC, int* csrColIndC, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseSpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &fcsrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fcsrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneCsr2csrByPercentage(handle_t, m, n, nnz, matDescr_A, &fcsrValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fcsrValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double percentage, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseDpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, float percentage, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseSpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneCsr2csrNnzByPercentage(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, double percentage, const hipsparseMatDescr_t descrC, const double* csrValC, const int* csrRowPtrC, const int* csrColIndC, pruneInfo_t info, size_t* bufferSize);
  // CHECK: status_t = hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float percentage, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, float percentage, const hipsparseMatDescr_t descrC, const float* csrValC, const int* csrRowPtrC, const int* csrColIndC, pruneInfo_t info, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, percentage, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* threshold, const hipsparseMatDescr_t descrC, double* csrValC, const int* csrRowPtrC, int* csrColIndC, void* buffer);
  // CHECK: status_t = hipsparseDpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseDpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* threshold, const hipsparseMatDescr_t descrC, float* csrValC, const int* csrRowPtrC, int* csrColIndC, void* buffer);
  // CHECK: status_t = hipsparseSpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseSpruneCsr2csr(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* threshold, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, void* buffer);
  // CHECK: status_t = hipsparseDpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseDpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csrNnz(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* threshold, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, void* buffer);
  // CHECK: status_t = hipsparseSpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseSpruneCsr2csrNnz(handle_t, m, n, nnz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* threshold, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* threshold, const hipsparseMatDescr_t descrC, const double* csrValC, const int* csrRowPtrC, const int* csrColIndC, size_t* bufferSize);
  // CHECK: status_t = hipsparseDpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseDpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &dbsrSortedValA, &csrRowPtrA, &csrColIndA, &dthreshold, matDescr_C, &dbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnzA, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* threshold, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* threshold, const hipsparseMatDescr_t descrC, const float* csrValC, const int* csrRowPtrC, const int* csrColIndC, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseSpruneCsr2csr_bufferSizeExt(handle_t, m, n, nnz, matDescr_A, &fbsrSortedValA, &csrRowPtrA, &csrColIndA, &fthreshold, matDescr_C, &fbsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC,int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t handle, int m, int n, const double* A, int lda, double percentage, const hipsparseMatDescr_t descr, double* csrVal, const int* csrRowPtr, int* csrColInd, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseDpruneDense2csrByPercentage(handle_t, m, n, &dA, lda, percentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseDpruneDense2csrByPercentage(handle_t, m, n, &dA, lda, percentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const hipsparseMatDescr_t descr, float* csrVal, const int* csrRowPtr, int* csrColInd, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseSpruneDense2csrByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);
  status_t = cusparseSpruneDense2csrByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t handle, int m, int n, const double* A, int lda, double percentage, const hipsparseMatDescr_t descr, int* csrRowPtr, int* nnzTotalDevHostPtr, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseDpruneDense2csrNnzByPercentage(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);
  status_t = cusparseDpruneDense2csrNnzByPercentage(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnzByPercentage(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, pruneInfo_t info, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const hipsparseMatDescr_t descr, int* csrRowPtr, int* nnzTotalDevHostPtr, pruneInfo_t info, void* buffer);
  // CHECK: status_t = hipsparseSpruneDense2csrNnzByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);
  status_t = cusparseSpruneDense2csrNnzByPercentage(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, prune_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* A, int lda, float percentage, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const double* A, int lda, double percentage, const hipsparseMatDescr_t descr, const double* csrVal, const int* csrRowPtr, const int* csrColInd, pruneInfo_t info, size_t* bufferSize);
  // CHECK: status_t = hipsparseDpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &dA, lda, fpercentage, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrByPercentage_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, pruneInfo_t info, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const float* A, int lda, float percentage, const hipsparseMatDescr_t descr, const float* csrVal, const int* csrRowPtr, const int* csrColInd, pruneInfo_t info, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);
  status_t = cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle_t, m, n, &fA, lda, fpercentage, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, prune_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const hipsparseMatDescr_t descr, double* csrVal, const int* csrRowPtr, int* csrColInd, void* buffer);
  // CHECK: status_t = hipsparseDpruneDense2csr(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseDpruneDense2csr(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const hipsparseMatDescr_t descr, float* csrVal, const int* csrRowPtr, int* csrColInd, void* buffer);
  // CHECK: status_t = hipsparseSpruneDense2csr(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseSpruneDense2csr(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const hipsparseMatDescr_t descr, int* csrRowPtr, int* nnzTotalDevHostPtr, void* buffer);
  // CHECK: status_t = hipsparseDpruneDense2csrNnz(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseDpruneDense2csrNnz(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csrNnz(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const hipsparseMatDescr_t descr, int* csrRowPtr, int* nnzTotalDevHostPtr, void* buffer);
  // CHECK: status_t = hipsparseSpruneDense2csrNnz(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);
  status_t = cusparseSpruneDense2csrNnz(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDpruneDense2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const double* A, int lda, const double* threshold, const hipsparseMatDescr_t descr, const double* csrVal, const int* csrRowPtr, const int* csrColInd, size_t* bufferSize);
  // CHECK: status_t = hipsparseDpruneDense2csr_bufferSizeExt(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseDpruneDense2csr_bufferSizeExt(handle_t, m, n, &dA, lda, &dthreshold, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseSpruneDense2csr_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const float* A, int lda, const float* threshold, const hipsparseMatDescr_t descr,const float* csrVal, const int* csrRowPtr, const int* csrColInd, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpruneDense2csr_bufferSizeExt(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseSpruneDense2csr_bufferSizeExt(handle_t, m, n, &fA, lda, &fthreshold, matDescr_C, &fcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, hipDoubleComplex* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseZgtsv2_nopivot(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);
  status_t = cusparseZgtsv2_nopivot(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle, int m, int n, const hipComplex* dl, const hipComplex* d, const hipComplex* du, hipComplex* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseCgtsv2_nopivot(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);
  status_t = cusparseCgtsv2_nopivot(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseDgtsv2_nopivot(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);
  status_t = cusparseDgtsv2_nopivot(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseSgtsv2_nopivot(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);
  status_t = cusparseSgtsv2_nopivot(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, const hipDoubleComplex* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);
  status_t = cusparseZgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipComplex* dl, const hipComplex* d, const hipComplex* du, const hipComplex* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);
  status_t = cusparseCgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int db, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);
  status_t = cusparseDgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);
  status_t = cusparseSgtsv2_nopivot_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, hipDoubleComplex* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseZgtsv2(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);
  status_t = cusparseZgtsv2(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle, int m, int n, const hipComplex* dl, const hipComplex* d, const hipComplex* du, hipComplex* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseCgtsv2(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);
  status_t = cusparseCgtsv2(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, double* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseDgtsv2(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);
  status_t = cusparseDgtsv2(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, float* B, int ldb, void* pBuffer);
  // CHECK: status_t = hipsparseSgtsv2(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);
  status_t = cusparseSgtsv2(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, const hipDoubleComplex* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZgtsv2_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);
  status_t = cusparseZgtsv2_bufferSizeExt(handle_t, m, n, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipComplex* dl, const hipComplex* d, const hipComplex* du, const hipComplex* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCgtsv2_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);
  status_t = cusparseCgtsv2_bufferSizeExt(handle_t, m, n, &complexdl, &complexd, &complexdu, &complexB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const double* dl, const double* d, const double* du, const double* B, int db, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDgtsv2_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);
  status_t = cusparseDgtsv2_bufferSizeExt(handle_t, m, n, &ddl, &dd, &ddu, &dB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const float* dl, const float* d, const float* du, const float* B, int ldb, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSgtsv2_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);
  status_t = cusparseSgtsv2_bufferSizeExt(handle_t, m, n, &fdl, &fd, &fdu, &fB, ldb, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, int batchStride, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t handle, int m, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, hipDoubleComplex* x, int batchCount, int batchStride, void* pBuffer);
  // CHECK: status_t = hipsparseZgtsv2StridedBatch(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseZgtsv2StridedBatch(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, cuComplex* x, int batchCount, int batchStride, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle, int m, const hipComplex* dl, const hipComplex* d, const hipComplex* du, hipComplex* x, int batchCount, int batchStride, void* pBuffer);
  // CHECK: status_t = hipsparseCgtsv2StridedBatch(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseCgtsv2StridedBatch(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle, int m, const double* dl, const double* d, const double* du, double* x, int batchCount, int batchStride, void* pBuffer);
  // CHECK: status_t = hipsparseDgtsv2StridedBatch(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseDgtsv2StridedBatch(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle, int m, const float* dl, const float* d, const float* du, float* x, int batchCount, int batchStride, void* pBuffer);
  // CHECK: status_t = hipsparseSgtsv2StridedBatch(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, pBuffer);
  status_t = cusparseSgtsv2StridedBatch(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle, int m, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, const hipDoubleComplex* x, int batchCount, int batchStride, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZgtsv2StridedBatch_bufferSizeExt(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseZgtsv2StridedBatch_bufferSizeExt(handle_t, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle, int m, const hipComplex* dl, const hipComplex* d, const hipComplex* du, const hipComplex* x, int batchCount, int batchStride, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCgtsv2StridedBatch_bufferSizeExt(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseCgtsv2StridedBatch_bufferSizeExt(handle_t, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, int batchStride, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDgtsv2StridedBatch_bufferSizeExt(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseDgtsv2StridedBatch_bufferSizeExt(handle_t, m, &ddl, &dd, &ddu, &dx, batchCount, ibatchStride, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* bufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, int batchStride, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSgtsv2StridedBatch_bufferSizeExt(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, &bufferSize);
  status_t = cusparseSgtsv2StridedBatch_bufferSizeExt(handle_t, m, &fdl, &fd, &fdu, &fx, batchCount, ibatchStride, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCreatePruneInfo(pruneInfo_t* info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info);
  // CHECK: status_t = hipsparseCreatePruneInfo(&prune_info);
  status_t = cusparseCreatePruneInfo(&prune_info);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDestroyPruneInfo(pruneInfo_t info);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info);
  // CHECK: status_t = hipsparseDestroyPruneInfo(prune_info);
  status_t = cusparseDestroyPruneInfo(prune_info);
#endif

#if CUDA_VERSION >= 9020
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* ds, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* dw, cuDoubleComplex* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgpsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, hipDoubleComplex* ds, hipDoubleComplex* dl, hipDoubleComplex* d, hipDoubleComplex* du, hipDoubleComplex* dw, hipDoubleComplex* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseZgpsvInterleavedBatch(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, pBuffer);
  status_t = cusparseZgpsvInterleavedBatch(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* ds, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* dw, cuComplex* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgpsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, hipComplex* ds, hipComplex* dl, hipComplex* d, hipComplex* du, hipComplex* dw, hipComplex* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseCgpsvInterleavedBatch(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, pBuffer);
  status_t = cusparseCgpsvInterleavedBatch(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgpsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseDgpsvInterleavedBatch(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, pBuffer);
  status_t = cusparseDgpsvInterleavedBatch(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgpsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseSgpsvInterleavedBatch(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, pBuffer);
  status_t = cusparseSgpsvInterleavedBatch(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* ds, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* dw, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const hipDoubleComplex* ds, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, const hipDoubleComplex* dw, const hipDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, &bufferSize);
  status_t = cusparseZgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* ds, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* dw, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const hipComplex* ds, const hipComplex* dl, const hipComplex* d, const hipComplex* du, const hipComplex* dw, const hipComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, &bufferSize);
  status_t = cusparseCgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, &bufferSize);
  status_t = cusparseDgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, &bufferSize);
  status_t = cusparseSgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, hipDoubleComplex* dl, hipDoubleComplex* d, hipDoubleComplex* du, hipDoubleComplex* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseZgtsvInterleavedBatch(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, pBuffer);
  status_t = cusparseZgtsvInterleavedBatch(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, hipComplex* dl, hipComplex* d, hipComplex* du, hipComplex* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseCgtsvInterleavedBatch(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, pBuffer);
  status_t = cusparseCgtsvInterleavedBatch(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseDgtsvInterleavedBatch(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, pBuffer);
  status_t = cusparseDgtsvInterleavedBatch(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsvInterleavedBatch(hipsparseHandle_t handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer);
  // CHECK: status_t = hipsparseSgtsvInterleavedBatch(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, pBuffer);
  status_t = cusparseSgtsvInterleavedBatch(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const hipDoubleComplex* dl, const hipDoubleComplex* d, const hipDoubleComplex* du, const hipDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, &bufferSize);
  status_t = cusparseZgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const hipComplex* dl, const hipComplex* d, const hipComplex* du, const hipComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, &bufferSize);
  status_t = cusparseCgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, &bufferSize);
  status_t = cusparseDgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseSgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, &bufferSize);
  status_t = cusparseSgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, &bufferSize);

#if CUDA_VERSION < 12000
  csrsm2Info_t csrsm2_info;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipDoubleComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, size_t* pBufferSize);
  // CHECK: status_t = hipsparseZcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);
  status_t = cusparseZcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, size_t* pBufferSize);
  // CHECK: status_t = hipsparseCcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);
  status_t = cusparseCcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, size_t* pBufferSize);
  // CHECK: status_t = hipsparseDcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);
  status_t = cusparseDcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t handle, int algo, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, hipsparseSolvePolicy_t policy, size_t* pBufferSize);
  // CHECK: status_t = hipsparseScsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);
  status_t = cusparseScsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle, csrsm2Info_t info, int* position);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position);
  // CHECK: status_t = hipsparseXcsrsm2_zeroPivot(handle_t, csrsm2_info, &iposition);
  status_t = cusparseXcsrsm2_zeroPivot(handle_t, csrsm2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsm2Info(csrsm2Info_t* info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info);
  // CHECK: status_t = hipsparseCreateCsrsm2Info(&csrsm2_info);
  status_t = cusparseCreateCsrsm2Info(&csrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsm2Info(csrsm2Info_t info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info);
  // CHECK: status_t = hipsparseDestroyCsrsm2Info(csrsm2_info);
  status_t = cusparseDestroyCsrsm2Info(csrsm2_info);
#endif
#endif

#if CUDA_VERSION >= 10000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrgeam2(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseZcsrgeam2(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t handle, int m, int n, const hipComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, hipComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrgeam2(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseCcsrgeam2(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t handle, int m, int n, const double* alpha, const hipsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const hipsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrgeam2(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseDcsrgeam2(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t handle, int m, int n, const float* alpha, const hipsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const hipsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // CHECK: status_t = hipsparseScsrgeam2(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);
  status_t = cusparseScsrgeam2(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace);
  // CHECK: status_t = hipsparseXcsrgeam2Nnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, workspace);
  status_t = cusparseXcsrgeam2Nnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, workspace);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, const hipDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsrgeam2_bufferSizeExt(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseZcsrgeam2_bufferSizeExt(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const cuComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORThipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const hipComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, const hipComplex* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsrgeam2_bufferSizeExt(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseCcsrgeam2_bufferSizeExt(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const double* alpha, const hipsparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const hipsparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, const double* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsrgeam2_bufferSizeExt(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseDcsrgeam2_bufferSizeExt(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, const float* alpha, const hipsparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const hipsparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const hipsparseMatDescr_t descrC, const float* csrSortedValC, const int* csrSortedRowPtrC, const int* csrSortedColIndC, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsrgeam2_bufferSizeExt(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
  status_t = cusparseScsrgeam2_bufferSizeExt(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, &bufferSize);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipsparseCsr2CscAlg_t Csr2CscAlg_t;
  // CHECK-NEXT: hipsparseCsr2CscAlg_t CSR2CSC_ALG1 = HIPSPARSE_CSR2CSC_ALG1;
  cusparseCsr2CscAlg_t Csr2CscAlg_t;
  cusparseCsr2CscAlg_t CSR2CSC_ALG1 = CUSPARSE_CSR2CSC_ALG1;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void* buffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, hipDataType valType, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase, hipsparseCsr2CscAlg_t alg, void* buffer);
  // CHECK: status_t = hipsparseCsr2cscEx2(handle_t, m, n, innz, pcsrVal, &csrRowPtrA, &csrColIndA, pcscVal, &cscColPtrA, &cscRowIndA, dataType, action_t, indexBase_t, Csr2CscAlg_t, pBuffer);
  status_t = cusparseCsr2cscEx2(handle_t, m, n, innz, pcsrVal, &csrRowPtrA, &csrColIndA, pcscVal, &cscColPtrA, &cscRowIndA, dataType, action_t, indexBase_t, Csr2CscAlg_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, hipDataType valType, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase, hipsparseCsr2CscAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseCsr2cscEx2_bufferSize(handle_t, m, n, innz, pcsrVal, &csrRowPtrA, &csrColIndA, pcscVal, &cscColPtrA, &cscRowIndA, dataType, action_t, indexBase_t, Csr2CscAlg_t, &bufferSize);
  status_t = cusparseCsr2cscEx2_bufferSize(handle_t, m, n, innz, pcsrVal, &csrRowPtrA, &csrColIndA, pcscVal, &cscColPtrA, &cscRowIndA, dataType, action_t, indexBase_t, Csr2CscAlg_t, &bufferSize);
#endif

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;

  // CHECK: hipsparseIndexType_t indexType_t;
  // CHECK-NEXT: hipsparseIndexType_t csrRowOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscColOffsetsType;
  // CHECK-NEXT: hipsparseIndexType_t cscRowIndType;
  // CHECK-NEXT: hipsparseIndexType_t csrColIndType;
  // CHECK-NEXT: hipsparseIndexType_t ellIdxType;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_16U = HIPSPARSE_INDEX_16U;
  // CHECK-NEXT: hipsparseIndexType_t INDEX_32I = HIPSPARSE_INDEX_32I;

  cusparseIndexType_t indexType_t;
  cusparseIndexType_t csrRowOffsetsType;
  cusparseIndexType_t cscColOffsetsType;
  cusparseIndexType_t cscRowIndType;
  cusparseIndexType_t csrColIndType;
  cusparseIndexType_t ellIdxType;
  cusparseIndexType_t INDEX_16U = CUSPARSE_INDEX_16U;
  cusparseIndexType_t INDEX_32I = CUSPARSE_INDEX_32I;

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, cudaDataType valueType, cusparseOrder_t order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void* values, hipDataType valueType, hipsparseOrder_t order);
  // CHECK: status_t = hipsparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateDnMat(&dnMatDescr_t, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride);
  // CHECK: status_t = hipsparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);
  status_t = cusparseDnMatSetStridedBatch(dnMatDescr_t, batchCount, batchStride);

#if CUSPARSE_VERSION >= 10200
  // CHECK: hipsparseIndexType_t INDEX_64I = HIPSPARSE_INDEX_64I;
  cusparseIndexType_t INDEX_64I = CUSPARSE_INDEX_64I;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cooRowInd, void** cooColInd, void** cooValues, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseCooGet(spMatDescr_t, &rows, &cols, &nnz, &cooRowInd, &cooColInd, &cooValues, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, cudaDataType* type, cusparseOrder_t* order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, void** values, hipDataType* valueType, hipsparseOrder_t* order);
  // CHECK: status_t = hipsparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);
  status_t = cusparseDnMatGet(dnMatDescr_t, &rows, &cols, &ld, &values, &dataType, &order_t);

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // CHECK: status_t = hipsparseDnMatGetStridedBatch(dnMatDescr_t, &batchCount, &batchStride);
  status_t = cusparseDnMatGetStridedBatch(dnMatDescr_t, &batchCount, &batchStride);
#endif
#endif

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr);
  // CHECK: status_t = hipsparseDestroySpMat(spMatDescr_t);
  status_t = cusparseDestroySpMat(spMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr, cusparseFormat_t* format);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr, hipsparseFormat_t* format);
  // CHECK: status_t = hipsparseSpMatGetFormat(spMatDescr_t, &format_t);
  status_t = cusparseSpMatGetFormat(spMatDescr_t, &format_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpMatGetIndexBase(spMatDescr_t, &indexBase_t);
  status_t = cusparseSpMatGetIndexBase(spMatDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr);
  // CHECK: status_t = hipsparseDestroyDnMat(dnMatDescr_t);
  status_t = cusparseDestroyDnMat(dnMatDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize);
  // HIP: hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpMM_bufferSize(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);
  status_t = cusparseSpMM_bufferSize(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMM(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
  status_t = cusparseSpMM(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
#endif
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

#if CUDA_VERSION < 12000
  csrgemm2Info_t csrgemm2_info;
  csrsv2Info_t csrsv2_info;

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsv2Info(csrsv2Info_t* info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info);
  // CHECK: status_t = hipsparseCreateCsrsv2Info(&csrsv2_info);
  status_t = cusparseCreateCsrsv2Info(&csrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsv2Info(csrsv2Info_t info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info);
  // CHECK: status_t = hipsparseDestroyCsrsv2Info(csrsv2_info);
  status_t = cusparseDestroyCsrsv2Info(csrsv2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrgemm2Info(csrgemm2Info_t* info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info);
  // CHECK: status_t = hipsparseCreateCsrgemm2Info(&csrgemm2_info);
  status_t = cusparseCreateCsrgemm2Info(&csrgemm2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrgemm2Info(csrgemm2Info_t info);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info);
  // CHECK: status_t = hipsparseDestroyCsrgemm2Info(csrgemm2_info);
  status_t = cusparseDestroyCsrgemm2Info(csrgemm2_info);
#endif

#if (CUDA_VERSION >= 10010 && CUSPARSE_VERSION >= 10200 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
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

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr);
  // HIP: hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr);
  // CHECK: status_t = hipsparseDestroySpVec(spVecDescr_t);
  status_t = cusparseDestroySpVec(spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(spVecDescr_t, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr, int* batchCount);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int* batchCount);
  // CHECK: status_t = hipsparseSpMatGetStridedBatch(spMatDescr_t, &batchCount);
  status_t = cusparseSpMatGetStridedBatch(spMatDescr_t, &batchCount);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr);
  // CHECK: status_t = hipsparseDestroyDnVec(dnVecDescr_t);
  status_t = cusparseDestroyDnVec(dnVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, const cusparseSpVecDescr_t vecX, const cusparseDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opX, hipsparseSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY, void* result, hipDataType computeType, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpVV_bufferSize(handle_t, opX, spVecDescr_t, vecY, result, dataType, &bufferSize);
  status_t = cusparseSpVV_bufferSize(handle_t, opX, spVecDescr_t, vecY, result, dataType, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, const cusparseSpVecDescr_t vecX, const cusparseDnVecDescr_t vecY, void* result, cudaDataType computeType, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t handle, hipsparseOperation_t opX, hipsparseSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY, void* result, hipDataType computeType, void* externalBuffer);
  // CHECK: status_t = hipsparseSpVV(handle_t, opX, spVecDescr_t, vecY, result, dataType, tempBuffer);
  status_t = cusparseSpVV(handle_t, opX, spVecDescr_t, vecY, result, dataType, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX, const void* beta, const cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize);
  // HIP: hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t vecX, const void* beta, const hipsparseDnVecDescr_t vecY, hipDataType computeType, hipsparseSpMVAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpMV_bufferSize(handle_t, opA, alpha, spmatA, vecX, beta, vecY, dataType, spMVAlg_t, &bufferSize);
  status_t = cusparseSpMV_bufferSize(handle_t, opA, alpha, spmatA, vecX, beta, vecY, dataType, spMVAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnVecDescr_t vecX, const void* beta, const cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t vecX, const void* beta, const hipsparseDnVecDescr_t vecY, hipDataType computeType, hipsparseSpMVAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMV(handle_t, opA, alpha, spmatA, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);
  status_t = cusparseSpMV(handle_t, opA, alpha, spmatA, vecX, beta, vecY, dataType, spMVAlg_t, tempBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex* f, cuDoubleComplex* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const hipDoubleComplex* f, hipDoubleComplex* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrsv2_solve(handle_t, opA, m, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrsv2_solve(handle_t, opA, m, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuComplex* f, cuComplex* x,cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const hipComplex* f, hipComplex* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrsv2_solve(handle_t, opA, m, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &complexF, &complexX, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrsv2_solve(handle_t, opA, m, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &complexF, &complexX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const double* f, double* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const double* f, double* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrsv2_solve(handle_t, opA, m, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dF, &dX, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrsv2_solve(handle_t, opA, m, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dF, &dX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const float* f, float* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const float* f, float* x, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrsv2_solve(handle_t, opA, m, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &fF, &fX, solvePolicy_t, pBuffer);
  status_t = cusparseScsrsv2_solve(handle_t, opA, m, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &fF, &fX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseZcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseCcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseDcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, hipsparseSolvePolicy_t policy, void* pBuffer);
  // CHECK: status_t = hipsparseScsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);
  status_t = cusparseScsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);
#endif

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, void** indices, void** values, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseSpVecGet(spVecDescr_t, &size, &nnz, &indices, &values, &indexType_t, &indexBase_t, &dataType);

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

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr, int64_t size, void* values, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateDnVec(&dnVecDescr_t, size, values, dataType);
  status_t = cusparseCreateDnVec(&dnVecDescr_t, size, values, dataType);

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
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipsparseStatus_t STATUS_NOT_SUPPORTED = HIPSPARSE_STATUS_NOT_SUPPORTED;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;
#endif

#if CUDA_VERSION >= 10020 && CUSPARSE_VERSION >= 10301
  // CUDA: const char* CUSPARSEAPI cusparseGetErrorName(cusparseStatus_t status);
  // HIP: HIPSPARSE_EXPORT const char* hipsparseGetErrorName(hipsparseStatus_t status);
  // CHECK: const_ch = hipsparseGetErrorName(status_2);
  const_ch = cusparseGetErrorName(status_2);

  // CUDA: const char* CUSPARSEAPI cusparseGetErrorString(cusparseStatus_t status);
  // HIP: HIPSPARSE_EXPORT const char* hipsparseGetErrorString(hipsparseStatus_t status);
  // CHECK: const_ch = hipsparseGetErrorString(status_2);
  const_ch = cusparseGetErrorString(status_2);
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

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipsparseHybMat_t hybA, int userEllWidth, hipsparseHybPartition_t partitionType);
  // CHECK: status_t = hipsparseZcsr2hyb(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseZcsr2hyb(handle_t, m, n, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseCcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipsparseHybMat_t hybA, int userEllWidth, hipsparseHybPartition_t partitionType);
  // CHECK: status_t = hipsparseCcsr2hyb(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseCcsr2hyb(handle_t, m, n, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDcsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipsparseHybMat_t hybA, int userEllWidth, hipsparseHybPartition_t partitionType);
  // CHECK: status_t = hipsparseDcsr2hyb(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseDcsr2hyb(handle_t, m, n, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseScsr2hyb(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cusparseHybMat_t hybA, int userEllWidth, cusparseHybPartition_t partitionType);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, hipsparseHybMat_t hybA, int userEllWidth, hipsparseHybPartition_t partitionType);
  // CHECK: status_t = hipsparseScsr2hyb(handle_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);
  status_t = cusparseScsr2hyb(handle_t, m, n, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, hybMat_t, userEllWidth, hybPartition_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2) cusparseStatus_t CUSPARSEAPI cusparseZcsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const cuDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, cuDoubleComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t handle, int m, int n, int nnz, const hipDoubleComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, hipDoubleComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZcsr2csc(handle_t, m, n, nnz, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);
  status_t = cusparseZcsr2csc(handle_t, m, n, nnz, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2) cusparseStatus_t CUSPARSEAPI cusparseCcsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const cuComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, cuComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr,cusparseAction_t copyValues, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t handle, int m, int n, int nnz, const hipComplex* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, hipComplex* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCcsr2csc(handle_t, m, n, nnz, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);
  status_t = cusparseCcsr2csc(handle_t, m, n, nnz, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2) cusparseStatus_t CUSPARSEAPI cusparseDcsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, double* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t handle, int m, int n, int nnz, const double* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, double* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDcsr2csc(handle_t, m, n, nnz, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);
  status_t = cusparseDcsr2csc(handle_t, m, n, nnz, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseCsr2cscEx2) cusparseStatus_t CUSPARSEAPI cusparseScsr2csc(cusparseHandle_t handle, int m, int n, int nnz, const float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, float* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t handle, int m, int n, int nnz, const float* csrSortedVal, const int* csrSortedRowPtr, const int* csrSortedColInd, float* cscSortedVal, int* cscSortedRowInd, int* cscSortedColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseScsr2csc(handle_t, m, n, nnz, &csrSortedVal, &csrRowPtrA, &csrColIndA, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);
  status_t = cusparseScsr2csc(handle_t, m, n, nnz, &csrSortedVal, &csrRowPtrA, &csrColIndA, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, copyValues, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t handle, int m, int n, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipDoubleComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipDoubleComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseZcsrgeam(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZcsrgeam(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t handle, int m, int n, const hipComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipComplex* beta, const hipsparseMatDescr_t descrB, int nnzB, const hipComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, hipComplex* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseCcsrgeam(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCcsrgeam(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t handle, int m, int n, const double* alpha, const hipsparseMatDescr_t descrA, int nnzA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const double* beta, const hipsparseMatDescr_t descrB, int nnzB, const double* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, double* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseDcsrgeam(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDcsrgeam(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseScsrgeam(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t handle, int m, int n, const float* alpha, const hipsparseMatDescr_t descrA, int nnzA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const float* beta, const hipsparseMatDescr_t descrB, int nnzB, const float* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, float* csrValC, int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseScsrgeam(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseScsrgeam(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgeam2) cusparseStatus_t CUSPARSEAPI cusparseXcsrgeamNnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseXcsrgeamNnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr);
  status_t = cusparseXcsrgeamNnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrmm(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* B, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, int ldc);
  // CHECK: status_t = hipsparseZcsrmm(handle_t, opA, m, n, k, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZcsrmm(handle_t, opA, m, n, k, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrmm(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* B, int ldb, const hipComplex* beta, hipComplex* C, int ldc);
  // CHECK: status_t = hipsparseCcsrmm(handle_t, opA, m, n, k, innz, &complexA, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCcsrmm(handle_t, opA, m, n, k, innz, &complexA, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // CHECK: status_t = hipsparseDcsrmm(handle_t, opA, m, n, k, innz, &dA, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDcsrmm(handle_t, opA, m, n, k, innz, &dA, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseScsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // CHECK: status_t = hipsparseScsrmm(handle_t, opA, m, n, k, innz, &fA, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseScsrmm(handle_t, opA, m, n, k, innz, &fA, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseZhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t handle, hipsparseOperation_t transA, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, const hipDoubleComplex* x, const hipDoubleComplex* beta, hipDoubleComplex* y);
  // CHECK: status_t = hipsparseZhybmv(handle_t, opA, &dcomplexAlpha, matDescr_A, hybMat_t, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZhybmv(handle_t, opA, &dcomplexAlpha, matDescr_A, hybMat_t, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseChybmv(cusparseHandle_t handle, cusparseOperation_t transA, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t handle, hipsparseOperation_t transA, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, const hipComplex* x, const hipComplex* beta, hipComplex* y);
  // CHECK: status_t = hipsparseChybmv(handle_t, opA, &complexAlpha, matDescr_A, hybMat_t, &complexX, &complexBeta, &complexY);
  status_t = cusparseChybmv(handle_t, opA, &complexAlpha, matDescr_A, hybMat_t, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseDhybmv(cusparseHandle_t handle, cusparseOperation_t transA, const double* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const double* x, const double* beta, double* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t handle, hipsparseOperation_t transA, const double* alpha, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, const double* x, const double* beta, double* y);
  // CHECK: status_t = hipsparseDhybmv(handle_t, opA, &dAlpha, matDescr_A, hybMat_t, &dX, &dBeta, &dY);
  status_t = cusparseDhybmv(handle_t, opA, &dAlpha, matDescr_A, hybMat_t, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED cusparseStatus_t CUSPARSEAPI cusparseShybmv(cusparseHandle_t handle, cusparseOperation_t transA, const float* alpha, const cusparseMatDescr_t descrA, const cusparseHybMat_t hybA, const float* x, const float* beta, float* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t handle, hipsparseOperation_t transA, const float* alpha, const hipsparseMatDescr_t descrA, const hipsparseHybMat_t hybA, const float* x, const float* beta, float* y);
  // CHECK: status_t = hipsparseShybmv(handle_t, opA, &fAlpha, matDescr_A, hybMat_t, &fX, &fBeta, &fY);
  status_t = cusparseShybmv(handle_t, opA, &fAlpha, matDescr_A, hybMat_t, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseZcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* x, const hipDoubleComplex* beta, hipDoubleComplex* y);
  // CHECK: status_t = hipsparseZcsrmv(handle_t, opA, m, n, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZcsrmv(handle_t, opA, m, n, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseCcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* x, const hipComplex* beta, hipComplex* y);
  // CHECK: status_t = hipsparseCcsrmv(handle_t, opA, m, n, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexX, &complexBeta, &complexY);
  status_t = cusparseCcsrmv(handle_t, opA, m, n, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* x, const double* beta, double* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* x, const double* beta, double* y);
  // CHECK: status_t = hipsparseDcsrmv(handle_t, opA, m, n, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dX, &dBeta, &dY);
  status_t = cusparseDcsrmv(handle_t, opA, m, n, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* x, const float* beta, float* y);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* x, const float* beta, float* y);
  // CHECK: status_t = hipsparseScsrmv(handle_t, opA, m, n, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fX, &fBeta, &fY);
  status_t = cusparseScsrmv(handle_t, opA, m, n, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseZdotci(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* y, cuDoubleComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZdotci(hipsparseHandle_t handle, int nnz, const hipDoubleComplex* xVal, const int* xInd, const hipDoubleComplex* y, hipDoubleComplex* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZdotci(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);
  status_t = cusparseZdotci(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseCdotci(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* y, cuComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCdotci(hipsparseHandle_t handle, int nnz, const hipComplex* xVal, const int* xInd, const hipComplex* y, hipComplex* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCdotci(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);
  status_t = cusparseCdotci(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseZdoti(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, const cuDoubleComplex* y, cuDoubleComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t handle, int nnz, const hipDoubleComplex* xVal, const int* xInd, const hipDoubleComplex* y, hipDoubleComplex* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZdoti(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);
  status_t = cusparseZdoti(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, &dcomplex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseCdoti(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, const cuComplex* y, cuComplex* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t handle, int nnz, const hipComplex* xVal, const int* xInd, const hipComplex* y, hipComplex* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCdoti(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);
  status_t = cusparseCdoti(handle_t, innz, &complexX, &xInd, &complexY, &complex_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseDdoti(cusparseHandle_t handle, int nnz, const double* xVal, const int* xInd, const double* y, double* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t handle, int nnz, const double* xVal, const int* xInd, const double* y, double* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDdoti(handle_t, innz, &dX, &xInd, &dY, &d_resultDevHostPtr, indexBase_t);
  status_t = cusparseDdoti(handle_t, innz, &dX, &xInd, &dY, &d_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpVV) cusparseStatus_t CUSPARSEAPI cusparseSdoti(cusparseHandle_t handle, int nnz, const float* xVal, const int* xInd, const float* y, float* resultDevHostPtr, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle, int nnz, const float* xVal, const int* xInd, const float* y, float* result, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSdoti(handle_t, innz, &fX, &xInd, &fY, &f_resultDevHostPtr, indexBase_t);
  status_t = cusparseSdoti(handle_t, innz, &fX, &xInd, &fY, &f_resultDevHostPtr, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrmm2(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, int nnz, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, const hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipDoubleComplex* B, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* C, int ldc);
  // CHECK: status_t = hipsparseZcsrmm2(handle_t, opA, opB, m, n, k, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZcsrmm2(handle_t, opA, opB, m, n, k, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrmm2(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, int nnz, const hipComplex* alpha, const hipsparseMatDescr_t descrA, const hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const hipComplex* B, int ldb, const hipComplex* beta, hipComplex* C, int ldc);
  // CHECK: status_t = hipsparseCcsrmm2(handle_t, opA, opB, m, n, k, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCcsrmm2(handle_t, opA, opB, m, n, k, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, int nnz, const double* alpha, const hipsparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // CHECK: status_t = hipsparseDcsrmm2(handle_t, opA, opB, m, n, k, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDcsrmm2(handle_t, opA, opB, m, n, k, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseScsrmm2(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, int nnz, const float* alpha, const hipsparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // CHECK: status_t = hipsparseScsrmm2(handle_t, opA, opB, m, n, k, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseScsrmm2(handle_t, opA, opB, m, n, k, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipsparseStatus_t STATUS_INSUFFICIENT_RESOURCES = HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
  cusparseStatus_t STATUS_INSUFFICIENT_RESOURCES = CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;

  // CHECK: hipsparseSpGEMMAlg_t spGEMMAlg_t;
  // CHECK-NEXT: hipsparseSpGEMMAlg_t SPGEMM_DEFAULT = HIPSPARSE_SPGEMM_DEFAULT;
  cusparseSpGEMMAlg_t spGEMMAlg_t;
  cusparseSpGEMMAlg_t SPGEMM_DEFAULT = CUSPARSE_SPGEMM_DEFAULT;

  // CHECK: hipsparseSpGEMMDescr_t spGEMMDescr;
  cusparseSpGEMMDescr_t spGEMMDescr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr, void* csrRowOffsets, void* csrColInd, void* csrValues);
  // CHECK: status_t = hipsparseCsrSetPointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);
  status_t = cusparseCsrSetPointers(spMatDescr_t, csrRowOffsets, csrColInd, csrValues);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
  // CHECK: status_t = hipsparseSpGEMM_createDescr(&spGEMMDescr);
  status_t = cusparseSpGEMM_createDescr(&spGEMMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
  // CHECK: status_t = hipsparseSpGEMM_destroyDescr(spGEMMDescr);
  status_t = cusparseSpGEMM_destroyDescr(spGEMMDescr);

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // CHECK: status_t = hipsparseSpMatGetSize(spMatDescr_t, &rows, &cols, &nnz);
  status_t = cusparseSpMatGetSize(spMatDescr_t, &rows, &cols, &nnz);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // CHECK: status_t = hipsparseSpGEMM_workEstimation(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMM_workEstimation(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);
  // CHECK: status_t = hipsparseSpGEMM_compute(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMM_compute(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr);
  // CHECK: status_t = hipsparseSpGEMM_copy(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);
  status_t = cusparseSpGEMM_copy(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);
#endif
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

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScatter(hipsparseHandle_t handle, hipsparseSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseScatter(handle_t, spVecDescr_t, vecY);
  status_t = cusparseScatter(handle_t, spVecDescr_t, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseDnVecDescr_t vecY, cusparseSpVecDescr_t vecX);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGather(hipsparseHandle_t handle, hipsparseDnVecDescr_t vecY, hipsparseSpVecDescr_t vecX);
  // CHECK: status_t = hipsparseGather(handle_t, vecY, spVecDescr_t);
  status_t = cusparseGather(handle_t, vecY, spVecDescr_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t handle, const void* alpha, hipsparseSpVecDescr_t vecX, const void* beta, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseAxpby(handle_t, alpha, spVecDescr_t, beta, vecY);
  status_t = cusparseAxpby(handle_t, alpha, spVecDescr_t, beta, vecY);
#endif
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

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t handle, hipsparseSpMatDescr_t matA, hipsparseDnMatDescr_t matB, hipsparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSparseToDense_bufferSize(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, &bufferSize);
  status_t = cusparseSparseToDense_bufferSize(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* buffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t handle, hipsparseSpMatDescr_t matA, hipsparseDnMatDescr_t matB, hipsparseSparseToDenseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSparseToDense(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, tempBuffer);
  status_t = cusparseSparseToDense(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t handle, hipsparseDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseDenseToSparse_bufferSize(handle_t, dnmatA, spmatB, denseToSparseAlg_t, &bufferSize);
  status_t = cusparseDenseToSparse_bufferSize(handle_t, dnmatA, spmatB, denseToSparseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* buffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t handle, hipsparseDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseDenseToSparse_analysis(handle_t, dnmatA, spmatB, denseToSparseAlg_t, tempBuffer);
  status_t = cusparseDenseToSparse_analysis(handle_t, dnmatA, spmatB, denseToSparseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* buffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t handle, hipsparseDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseDenseToSparse_convert(handle_t, dnmatA, spmatB, denseToSparseAlg_t, tempBuffer);
  status_t = cusparseDenseToSparse_convert(handle_t, dnmatA, spmatB, denseToSparseAlg_t, tempBuffer);
#endif
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

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMM_preprocess(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
  status_t = cusparseSpMM_preprocess(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSDDMM_bufferSize(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);
  status_t = cusparseSDDMM_bufferSize(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM_preprocess(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM_preprocess(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseDnMatDescr_t matA, cusparseDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseDnMatDescr_t A, const hipsparseDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM(handle_t, opA, opB, alpha, dnmatA, dnmatB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
#endif
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

  // CHECK: hipsparseSpSVDescr_t spSVDescr;
  cusparseSpSVDescr_t spSVDescr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr);
  // CHECK: status_t = hipsparseSpSV_createDescr(&spSVDescr);
  status_t = cusparseSpSV_createDescr(&spSVDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr);
  // CHECK: status_t = hipsparseSpSV_destroyDescr(spSVDescr);
  status_t = cusparseSpSV_destroyDescr(spSVDescr);

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // HIP: hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t spMatDescr, hipsparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // CHECK: status_t = hipsparseSpMatGetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatGetAttribute(spMatDescr_t, spMatAttribute_t, &data, dataSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, size_t* bufferSize);
  // HIP: hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpSV_bufferSize(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr, &bufferSize);
  status_t = cusparseSpSV_bufferSize(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, void* externalBuffer);
  // HIP: hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr, void* externalBuffer);
  // CHECK: status_t = hipsparseSpSV_analysis(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr, tempBuffer);
  status_t = cusparseSpSV_analysis(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr);
  // CHECK: status_t = hipsparseSpSV_solve(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr);
  status_t = cusparseSpSV_solve(handle_t, opA, alpha, spmatA, vecX, vecY, dataType, spSVAlg_t, spSVDescr);
#endif

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

  // CHECK: hipsparseSpSMDescr_t spSMDescr;
  cusparseSpSMDescr_t spSMDescr;

  // CHECK: hipsparseSpGEMMAlg_t SPGEMM_CSR_ALG_DETERMINITIC = HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC;
  // CHECK-NEXT: hipsparseSpGEMMAlg_t SPGEMM_CSR_ALG_NONDETERMINITIC = HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC;
  cusparseSpGEMMAlg_t SPGEMM_CSR_ALG_DETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
  cusparseSpGEMMAlg_t SPGEMM_CSR_ALG_NONDETERMINITIC = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
  // CHECK: status_t = hipsparseSpSM_createDescr(&spSMDescr);
  status_t = cusparseSpSM_createDescr(&spSMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
  // CHECK: status_t = hipsparseSpSM_destroyDescr(spSMDescr);
  status_t = cusparseSpSM_destroyDescr(spSMDescr);

#if CUDA_VERSION < 11000
  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2) cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const hipDoubleComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrValC, const int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseZcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &dcomplexA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndD);
  status_t = cusparseZcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &dcomplexA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndD);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2) cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const hipComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const hipComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, hipComplex* csrValC, const int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseCcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &complexA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndD);
  status_t = cusparseCcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &complexA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndD);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2) cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const double* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, double* csrValC, const int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseDcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &dA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dB, &csrRowPtrB, &csrColIndB, matDescr_C, &dC, &csrRowPtrC, &csrColIndD);
  status_t = cusparseDcsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &dA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dB, &csrRowPtrB, &csrColIndB, matDescr_C, &dC, &csrRowPtrC, &csrColIndD);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2) cusparseStatus_t CUSPARSEAPI cusparseScsrgemm(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, const int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const float* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, float* csrValC, const int* csrRowPtrC, int* csrColIndC);
  // CHECK: status_t = hipsparseScsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &fA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &fB, &csrRowPtrB, &csrColIndB, matDescr_C, &fC, &csrRowPtrC, &csrColIndD);
  status_t = cusparseScsrgemm(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &fA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &fB, &csrRowPtrB, &csrColIndB, matDescr_C, &fC, &csrRowPtrC, &csrColIndD);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseXcsrgemm2) cusparseStatus_t CUSPARSEAPI cusparseXcsrgemmNnz(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, int m, int n, int k, const cusparseMatDescr_t descrA, const int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, const int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr);
  // HIP: DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr);
  // CHECK: status_t = hipsparseXcsrgemmNnz(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &csrColIndD, &nnzTotalDevHostPtr);
  status_t = cusparseXcsrgemmNnz(handle_t, opA, opB, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &csrColIndD, &nnzTotalDevHostPtr);
#endif

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // CHECK: status_t = hipsparseSpGEMMreuse_workEstimation(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMMreuse_workEstimation(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_nnz(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4);
  // CHECK: status_t = hipsparseSpGEMMreuse_nnz(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize2, tempBuffer2, &bufferSize3, tempBuffer3, &bufferSize4, tempBuffer4);
  status_t = cusparseSpGEMMreuse_nnz(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize2, tempBuffer2, &bufferSize3, tempBuffer3, &bufferSize4, tempBuffer4);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr);
  // CHECK: status_t = hipsparseSpGEMMreuse_compute(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);
  status_t = cusparseSpGEMMreuse_compute(handle_t, opA, opB, alpha, spmatA, spmatB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseSpMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5);
  // CHECK: status_t = hipsparseSpGEMMreuse_copy(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize5, tempBuffer5);
  status_t = cusparseSpGEMMreuse_copy(handle_t, opA, opB, spmatA, spmatB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize5, tempBuffer5);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpSMAlg_t alg, hipsparseSpSMDescr_t spsmDescr, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpSM_bufferSize(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr, &bufferSize);
  status_t = cusparseSpSM_bufferSize(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpSMAlg_t alg, hipsparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // CHECK: status_t = hipsparseSpSM_analysis(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr, tempBuffer);
  status_t = cusparseSpSM_analysis(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr, tempBuffer);
#endif
#endif

#if CUDA_VERSION >= 11070
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCscGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cscColOffsets, void** cscRowInd, void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCscGet(const hipsparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, void** cscColOffsets, void** cscRowInd, void** cscValues, hipsparseIndexType_t* cscColOffsetsType, hipsparseIndexType_t* cscRowIndType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseCscGet(spmatA, &rows, &cols, &nnz, &cscColOffsets, &cscRowInd, &cscValues, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
  status_t = cusparseCscGet(spmatA, &rows, &cols, &nnz, &cscColOffsets, &cscRowInd, &cscValues, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
#endif

#if CUDA_VERSION < 12000
  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseZcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuDoubleComplex* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipDoubleComplex* csc_val, const int* csc_row_ind, const int* csc_col_ptr, hipDoubleComplex* A, int ld);
  // CHECK: status_t = hipsparseZcsc2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);
  status_t = cusparseZcsc2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseCcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, cuComplex* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipComplex* csc_val, const int* csc_row_ind, const int* csc_col_ptr, hipComplex* A, int ld);
  // CHECK: status_t = hipsparseCcsc2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);
  status_t = cusparseCcsc2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseDcsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, double* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const double* csc_val, const int* csc_row_ind, const int* csc_col_ptr, double* A, int ld);
  // CHECK: status_t = hipsparseDcsc2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);
  status_t = cusparseDcsc2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseScsc2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* cscSortedValA, const int* cscSortedRowIndA, const int* cscSortedColPtrA, float* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const float* csc_val, const int* csc_row_ind, const int* csc_col_ptr, float* A, int ld);
  // CHECK: status_t = hipsparseScsc2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);
  status_t = cusparseScsc2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseZcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipDoubleComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, hipDoubleComplex* A, int ld);
  // CHECK: status_t = hipsparseZcsr2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);
  status_t = cusparseZcsr2dense(handle_t, m, n, matDescr_A, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dcomplexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseCcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipComplex* csr_val, const int* csr_row_ptr, const int* csr_col_ind, hipComplex* A, int ld);
  // CHECK: status_t = hipsparseCcsr2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);
  status_t = cusparseCcsr2dense(handle_t, m, n, matDescr_A, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &complexA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseDcsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const double* csr_val, const int* csr_row_ptr, const int* csr_col_ind, double* A, int ld);
  // CHECK: status_t = hipsparseDcsr2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);
  status_t = cusparseDcsr2dense(handle_t, m, n, matDescr_A, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &dA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSparseToDense) cusparseStatus_t CUSPARSEAPI cusparseScsr2dense(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* A, int lda);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const float* csr_val, const int* csr_row_ptr, const int* csr_col_ind, float* A, int ld);
  // CHECK: status_t = hipsparseScsr2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);
  status_t = cusparseScsr2dense(handle_t, m, n, matDescr_A, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd, &fA, lda);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseZdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, const int* nnzPerCol, cuDoubleComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipDoubleComplex* A, int ld, const int* nnz_per_columns, hipDoubleComplex* csc_val, int* csc_row_ind, int* csc_col_ptr);
  // CHECK: status_t = hipsparseZdense2csc(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerCol, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseZdense2csc(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerCol, &dComplexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseCdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, const int* nnzPerCol, cuComplex* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipComplex* A, int ld, const int* nnz_per_columns, hipComplex* csc_val, int* csc_row_ind, int* csc_col_ptr);
  // CHECK: status_t = hipsparseCdense2csc(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerCol, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseCdense2csc(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerCol, &complexcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseDdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, const int* nnzPerCol, double* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const double* A, int ld, const int* nnz_per_columns, double* csc_val, int* csc_row_ind, int* csc_col_ptr);
  // CHECK: status_t = hipsparseDdense2csc(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerCol, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseDdense2csc(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerCol, &dcscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseSdense2csc(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerCol, float* cscSortedValA, int* cscSortedRowIndA, int* cscSortedColPtrA);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const float* A, int ld, const int* nnz_per_columns, float* csc_val, int* csc_row_ind, int* csc_col_ptr);
  // CHECK: status_t = hipsparseSdense2csc(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerCol, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd);
  status_t = cusparseSdense2csc(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerCol, &cscSortedVal, &csrSortedRowPtr, &csrSortedColInd);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseZdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuDoubleComplex* A, int lda, const int* nnzPerRow, cuDoubleComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipDoubleComplex* A, int ld, const int* nnz_per_rows, hipDoubleComplex* csr_val, int* csr_row_ptr, int* csr_col_ind);
  // CHECK: status_t = hipsparseZdense2csr(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRow, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseZdense2csr(handle_t, m, n, matDescr_A, &dcomplexA, lda, &nnzPerRow, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseCdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const cuComplex* A, int lda, const int* nnzPerRow, cuComplex* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const hipComplex* A, int ld, const int* nnz_per_rows, hipComplex* csr_val, int* csr_row_ptr, int* csr_col_ind);
  // CHECK: status_t = hipsparseCdense2csr(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerRow, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseCdense2csr(handle_t, m, n, matDescr_A, &complexA, lda, &nnzPerRow, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseDdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const double* A, int lda, const int* nnzPerRow, double* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const double* A, int ld, const int* nnz_per_rows, double* csr_val, int* csr_row_ptr, int* csr_col_ind);
  // CHECK: status_t = hipsparseDdense2csr(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerRow, &dcsrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseDdense2csr(handle_t, m, n, matDescr_A, &dA, lda, &nnzPerRow, &dcsrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseDenseToSparse) cusparseStatus_t CUSPARSEAPI cusparseSdense2csr(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, const float* A, int lda, const int* nnzPerRow, float* csrSortedValA, int* csrSortedRowPtrA, int* csrSortedColIndA);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descr, const float* A, int ld, const int* nnz_per_rows, float* csr_val, int* csr_row_ptr, int* csr_col_ind);
  // CHECK: status_t = hipsparseSdense2csr(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerRow, &csrSortedValA, &csrRowPtrA, &csrColIndA);
  status_t = cusparseSdense2csr(handle_t, m, n, matDescr_A, &fA, lda, &nnzPerRow, &csrSortedValA, &csrRowPtrA, &csrColIndA);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Nnz(cusparseHandle_t handle, int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, const csrgemm2Info_t info, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t handle, int m, int n, int k, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const hipsparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, const hipsparseMatDescr_t descrC, int* csrRowPtrC, int* nnzTotalDevHostPtr, const csrgemm2Info_t info, void* pBuffer);
  // CHECK: status_t = hipsparseXcsrgemm2Nnz(handle_t, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, csrgemm2_info, pBuffer);
  status_t = cusparseXcsrgemm2Nnz(handle_t, m, n, k, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const cuDoubleComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t handle, int m, int n, int k, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipDoubleComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const hipDoubleComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipDoubleComplex* beta, const hipsparseMatDescr_t descrD, int nnzD, const hipDoubleComplex* csrValD, const int* csrRowPtrD, const int* csrColIndD, const hipsparseMatDescr_t descrC, hipDoubleComplex* csrValC, const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer);
  // CHECK: status_t = hipsparseZcsrgemm2(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &dComplexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseZcsrgemm2(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &dComplexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const cuComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t handle, int m, int n, int k, const hipComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const hipComplex* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const hipComplex* csrValB, const int* csrRowPtrB, const int* csrColIndB, const hipComplex* beta, const hipsparseMatDescr_t descrD, int nnzD, const hipComplex* csrValD, const int* csrRowPtrD, const int* csrColIndD, const hipsparseMatDescr_t descrC, hipComplex* csrValC, const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer);
  // CHECK: status_t = hipsparseCcsrgemm2(handle_t, m, n, k, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &complexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseCcsrgemm2(handle_t, m, n, k, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &complexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const cusparseMatDescr_t descrD, int nnzD, const double* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t handle, int m, int n, int k, const double* alpha, const hipsparseMatDescr_t descrA, int nnzA, const double* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const double* csrValB, const int* csrRowPtrB, const int* csrColIndB, const double* beta, const hipsparseMatDescr_t descrD, int nnzD, const double* csrValD, const int* csrRowPtrD, const int* csrColIndD, const hipsparseMatDescr_t descrC, double* csrValC, const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer);
  // CHECK: status_t = hipsparseDcsrgemm2(handle_t, m, n, k, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &dcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseDcsrgemm2(handle_t, m, n, k, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &dcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2(cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const cusparseMatDescr_t descrD, int nnzD, const float* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t handle, int m, int n, int k, const float* alpha, const hipsparseMatDescr_t descrA, int nnzA, const float* csrValA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const float* csrValB, const int* csrRowPtrB, const int* csrColIndB, const float* beta, const hipsparseMatDescr_t descrD, int nnzD, const float* csrValD, const int* csrRowPtrD, const int* csrColIndD, const hipsparseMatDescr_t descrC, float* csrValC, const int* csrRowPtrC, int* csrColIndC, const csrgemm2Info_t info, void* pBuffer);
  // CHECK: status_t = hipsparseScsrgemm2(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseScsrgemm2(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int k, const hipDoubleComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const hipDoubleComplex* beta, const hipsparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseZcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int k, const hipComplex* alpha, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const hipComplex* beta, const hipsparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsrgemm2_bufferSizeExt(handle_t, m, n, k, &complexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseCcsrgemm2_bufferSizeExt(handle_t, m, n, k, &complexA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int k, const double* alpha, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const double* beta, const hipsparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseDcsrgemm2_bufferSizeExt(handle_t, m, n, k, &dA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const cusparseMatDescr_t descrD, int nnzD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t handle, int m, int n, int k, const float* alpha, const hipsparseMatDescr_t descrA, int nnzA, const int* csrRowPtrA, const int* csrColIndA, const hipsparseMatDescr_t descrB, int nnzB, const int* csrRowPtrB, const int* csrColIndB, const float* beta, const hipsparseMatDescr_t descrD, int nnzD, const int* csrRowPtrD, const int* csrColIndD, csrgemm2Info_t info, size_t* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsrgemm2_bufferSizeExt(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);
  status_t = cusparseScsrgemm2_bufferSizeExt(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrRowPtrD, &csrColIndD, csrgemm2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseZcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseZcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORThipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseCcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseCcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseDcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseDcsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, size_t* pBufferSize);
  // CHECK: status_t = hipsparseScsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);
  status_t = cusparseScsrsv2_bufferSizeExt(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, hipDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseZcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);
  status_t = cusparseZcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, hipComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseCcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);
  status_t = cusparseCcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseDcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);
  status_t = cusparseDcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int nnz, const hipsparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // CHECK: status_t = hipsparseScsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);
  status_t = cusparseScsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle, csrsv2Info_t info, int* position);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position);
  // CHECK: status_t = hipsparseXcsrsv2_zeroPivot(handle_t, csrsv2_info, &iposition);
  status_t = cusparseXcsrsv2_zeroPivot(handle_t, csrsv2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseZsctr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZsctr(hipsparseHandle_t handle, int nnz, const hipDoubleComplex* xVal, const int* xInd, hipDoubleComplex* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZsctr(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, indexBase_t);
  status_t = cusparseZsctr(handle_t, innz, &dcomplexX, &xInd, &dcomplexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseCsctr(cusparseHandle_t handle, int nnz, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCsctr(hipsparseHandle_t handle, int nnz, const hipComplex* xVal, const int* xInd, hipComplex* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCsctr(handle_t, innz, &complexX, &xInd, &complexY, indexBase_t);
  status_t = cusparseCsctr(handle_t, innz, &complexX, &xInd, &complexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseDsctr(cusparseHandle_t handle, int nnz, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t handle, int nnz, const double* xVal, const int* xInd, double* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDsctr(handle_t, innz, &dX, &xInd, &dY, indexBase_t);
  status_t = cusparseDsctr(handle_t, innz, &dX, &xInd, &dY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseScatter) cusparseStatus_t CUSPARSEAPI cusparseSsctr(cusparseHandle_t handle, int nnz, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t handle, int nnz, const float* xVal, const int* xInd, float* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSsctr(handle_t, innz, &fX, &xInd, &fY, indexBase_t);
  status_t = cusparseSsctr(handle_t, innz, &fX, &xInd, &fY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseRot) cusparseStatus_t CUSPARSEAPI cusparseDroti(cusparseHandle_t handle, int nnz, double* xVal, const int* xInd, double* y, const double* c, const double* s, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDroti(hipsparseHandle_t handle, int nnz, double* xVal, const int* xInd, double* y, const double* c, const double* s, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDroti(handle_t, innz, &dX, &xInd, &dY, &dC, &dS, indexBase_t);
  status_t = cusparseDroti(handle_t, innz, &dX, &xInd, &dY, &dC, &dS, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseRot) cusparseStatus_t CUSPARSEAPI cusparseSroti(cusparseHandle_t handle, int nnz, float* xVal, const int* xInd, float* y, const float* c, const float* s, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSroti(hipsparseHandle_t handle, int nnz, float* xVal, const int* xInd, float* y, const float* c, const float* s, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSroti(handle_t, innz, &fX, &xInd, &fY, &fC, &fS, indexBase_t);
  status_t = cusparseSroti(handle_t, innz, &fX, &xInd, &fY, &fC, &fS, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseZgthrz(cusparseHandle_t handle, int nnz, cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgthrz(hipsparseHandle_t handle, int nnz, hipDoubleComplex* y, hipDoubleComplex* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZgthrz(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);
  status_t = cusparseZgthrz(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseCgthrz(cusparseHandle_t handle, int nnz, cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgthrz(hipsparseHandle_t handle, int nnz, hipComplex* y, hipComplex* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCgthrz(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);
  status_t = cusparseCgthrz(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseDgthrz(cusparseHandle_t handle, int nnz, double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgthrz(hipsparseHandle_t handle, int nnz, double* y, double* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDgthrz(handle_t, innz, &dY, &dX, &xInd, indexBase_t);
  status_t = cusparseDgthrz(handle_t, innz, &dY, &dX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseSgthrz(cusparseHandle_t handle, int nnz, float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgthrz(hipsparseHandle_t handle, int nnz, float* y, float* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSgthrz(handle_t, innz, &fY, &fX, &xInd, indexBase_t);
  status_t = cusparseSgthrz(handle_t, innz, &fY, &fX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseZgthr(cusparseHandle_t handle, int nnz, const cuDoubleComplex* y, cuDoubleComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZgthr(hipsparseHandle_t handle, int nnz, const hipDoubleComplex* y, hipDoubleComplex* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZgthr(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);
  status_t = cusparseZgthr(handle_t, innz, &dcomplexY, &dcomplexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseCgthr(cusparseHandle_t handle, int nnz, const cuComplex* y, cuComplex* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCgthr(hipsparseHandle_t handle, int nnz, const hipComplex* y, hipComplex* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCgthr(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);
  status_t = cusparseCgthr(handle_t, innz, &complexY, &complexX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseDgthr(cusparseHandle_t handle, int nnz, const double* y, double* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t handle, int nnz, const double* y, double* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDgthr(handle_t, innz, &dY, &dX, &xInd, indexBase_t);
  status_t = cusparseDgthr(handle_t, innz, &dY, &dX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseGather) cusparseStatus_t CUSPARSEAPI cusparseSgthr(cusparseHandle_t handle, int nnz, const float* y, float* xVal, const int* xInd, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t handle, int nnz, const float* y, float* xVal, const int* xInd, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSgthr(handle_t, innz, &fY, &fX, &xInd, indexBase_t);
  status_t = cusparseSgthr(handle_t, innz, &fY, &fX, &xInd, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseZaxpyi(cusparseHandle_t handle, int nnz, const cuDoubleComplex* alpha, const cuDoubleComplex* xVal, const int* xInd, cuDoubleComplex* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseZaxpyi(hipsparseHandle_t handle, int nnz, const hipDoubleComplex* alpha, const hipDoubleComplex* xVal, const int* xInd, hipDoubleComplex* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseZaxpyi(handle_t, innz, &dcomplexAlpha, &dcomplexX, &xInd, &dcomplexY, indexBase_t);
  status_t = cusparseZaxpyi(handle_t, innz, &dcomplexAlpha, &dcomplexX, &xInd, &dcomplexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseCaxpyi(cusparseHandle_t handle, int nnz, const cuComplex* alpha, const cuComplex* xVal, const int* xInd, cuComplex* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCaxpyi(hipsparseHandle_t handle, int nnz, const hipComplex* alpha, const hipComplex* xVal, const int* xInd, hipComplex* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseCaxpyi(handle_t, innz, &complexAlpha, &complexX, &xInd, &complexY, indexBase_t);
  status_t = cusparseCaxpyi(handle_t, innz, &complexAlpha, &complexX, &xInd, &complexY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double* alpha, const double* xVal, const int* xInd, double* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t handle, int nnz, const double* alpha, const double* xVal, const int* xInd, double* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseDaxpyi(handle_t, innz, &dAlpha, &dX, &xInd, &dY, indexBase_t);
  status_t = cusparseDaxpyi(handle_t, innz, &dAlpha, &dX, &xInd, &dY, indexBase_t);

  // CUDA: CUSPARSE_DEPRECATED(cusparseAxpby) cusparseStatus_t CUSPARSEAPI cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, cusparseIndexBase_t idxBase);
  // HIP: DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12") HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t handle, int nnz, const float* alpha, const float* xVal, const int* xInd, float* y, hipsparseIndexBase_t idxBase);
  // CHECK: status_t = hipsparseSaxpyi(handle_t, innz, &fAlpha, &fX, &xInd, &fY, indexBase_t);
  status_t = cusparseSaxpyi(handle_t, innz, &fAlpha, &fX, &xInd, &fY, indexBase_t);
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

  // CHECK: hipsparseConstSpVecDescr_t constSpVecDescr = nullptr;
  cusparseConstSpVecDescr_t constSpVecDescr = nullptr;

  // CHECK: hipsparseConstSpMatDescr_t constSpMatDescr = nullptr;
  // CHECK-NEXT: hipsparseConstSpMatDescr_t constSpMatDescrB = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescr = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescrB = nullptr;

  // CHECK: hipsparseConstDnVecDescr_t constDnVecDescr = nullptr;
  cusparseConstDnVecDescr_t constDnVecDescr = nullptr;

  // CHECK: hipsparseConstDnMatDescr_t constDnMatDescr = nullptr;
  // CHECK-NEXT: hipsparseConstDnMatDescr_t constDnMatDescrB = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescr = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescrB = nullptr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: hipsparseStatus_t hipsparseCreateConstSpVec(hipsparseConstSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, const void* indices, const void* values, hipsparseIndexType_t idxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstSpVec(&constSpVecDescr, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateConstSpVec(&constSpVecDescr, size, nnz, indices, values, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpVec(cusparseConstSpVecDescr_t spVecDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroySpVec(hipsparseConstSpVecDescr_t spVecDescr);
  // CHECK: status_t = hipsparseDestroySpVec(constSpVecDescr);
  status_t = cusparseDestroySpVec(constSpVecDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpVecGet(cusparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, const void** indices, const void** values, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstSpVecGet(hipsparseConstSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz, const void** indices, const void** values, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstSpVecGet(constSpVecDescr, &size, &nnz, indices_const, values_const, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseConstSpVecGet(constSpVecDescr, &size, &nnz, indices_const, values_const, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseConstSpVecDescr_t spVecDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpVecGetIndexBase(constSpVecDescr, &indexBase_t);
  status_t = cusparseSpVecGetIndexBase(constSpVecDescr, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t spVecDescr, const void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstSpVecGetValues(hipsparseConstSpVecDescr_t spVecDescr, const void** values);
  // CHECK: status_t = hipsparseConstSpVecGetValues(constSpVecDescr, values_const);
  status_t = cusparseConstSpVecGetValues(constSpVecDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCoo(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, cusparseIndexType_t cooIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstCoo(hipsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cooRowInd, const void* cooColInd, const void* cooValues, hipsparseIndexType_t cooIdxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstCoo(&constSpMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateConstCoo(&constSpMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, indexType_t, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCsr(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, cusparseIndexType_t csrRowOffsetsType, cusparseIndexType_t csrColIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstCsr(hipsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* csrRowOffsets, const void* csrColInd, const void* csrValues, hipsparseIndexType_t csrRowOffsetsType, hipsparseIndexType_t csrColIndType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstCsr(&constSpMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);
  status_t = cusparseCreateConstCsr(&constSpMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType, csrColIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstCsc(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, cusparseIndexType_t cscColOffsetsType, cusparseIndexType_t cscRowIndType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstCsc(hipsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t nnz, const void* cscColOffsets, const void* cscRowInd, const void* cscValues, hipsparseIndexType_t cscColOffsetsType, hipsparseIndexType_t cscRowIndType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstCsc(&constSpMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, indexBase_t, dataType);
  status_t = cusparseCreateConstCsc(&constSpMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, cscColOffsetsType, cscRowIndType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, cusparseIndexType_t ellIdxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstBlockedEll(hipsparseConstSpMatDescr_t* spMatDescr, int64_t rows, int64_t cols, int64_t ellBlockSize, int64_t ellCols, const void* ellColInd, const void* ellValue, hipsparseIndexType_t ellIdxType, hipsparseIndexBase_t idxBase, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstBlockedEll(&constSpMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);
  status_t = cusparseCreateConstBlockedEll(&constSpMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, ellIdxType, indexBase_t, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroySpMat(cusparseConstSpMatDescr_t spMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroySpMat(hipsparseConstSpMatDescr_t spMatDescr);
  // CHECK: status_t = hipsparseDestroySpMat(constSpMatDescr);
  status_t = cusparseDestroySpMat(constSpMatDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCooGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cooRowInd, const void** cooColInd, const void** cooValues, cusparseIndexType_t* idxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstCooGet(hipsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cooRowInd, const void** cooColInd, const void** cooValues, hipsparseIndexType_t* idxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstCooGet(constSpMatDescr, &rows, &cols, &nnz, cooRowInd_const, cooColInd_const, cooValues_const, &indexType_t, &indexBase_t, &dataType);
  status_t = cusparseConstCooGet(constSpMatDescr, &rows, &cols, &nnz, cooRowInd_const, cooColInd_const, cooValues_const, &indexType_t, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCsrGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csrRowOffsets, const void** csrColInd, const void** csrValues, cusparseIndexType_t* csrRowOffsetsType, cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstCsrGet(hipsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** csrRowOffsets, const void** csrColInd, const void** csrValues, hipsparseIndexType_t* csrRowOffsetsType, hipsparseIndexType_t* csrColIndType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstCsrGet(constSpMatDescr, &rows, &cols, &nnz, csrRowOffsets_const, csrColInd_const, csrValues_const, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);
  status_t = cusparseConstCsrGet(constSpMatDescr, &rows, &cols, &nnz, csrRowOffsets_const, csrColInd_const, csrValues_const, &csrRowOffsetsType, &csrColIndType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstBlockedEllGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, const void** ellColInd, const void** ellValue, cusparseIndexType_t* ellIdxType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstBlockedEllGet(hipsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* ellBlockSize, int64_t* ellCols, const void** ellColInd, const void** ellValue, hipsparseIndexType_t* ellIdxType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstBlockedEllGet(constSpMatDescr, &rows, &cols, &ellBlockSize, &ellCols, ellColInd_const, ellValue_const, &ellIdxType, &indexBase_t, &dataType);
  status_t = cusparseConstBlockedEllGet(constSpMatDescr, &rows, &cols, &ellBlockSize, &ellCols, ellColInd_const, ellValue_const, &ellIdxType, &indexBase_t, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetSize(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetSize(hipsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz);
  // CHECK: status_t = hipsparseSpMatGetSize(constSpMatDescr, &rows, &cols, &nnz);
  status_t = cusparseSpMatGetSize(constSpMatDescr, &rows, &cols, &nnz);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetFormat(cusparseConstSpMatDescr_t spMatDescr, cusparseFormat_t* format);
  // HIP: hipsparseStatus_t hipsparseSpMatGetFormat(hipsparseConstSpMatDescr_t spMatDescr, hipsparseFormat_t* format);
  // CHECK: status_t = hipsparseSpMatGetFormat(constSpMatDescr, &format_t);
  status_t = cusparseSpMatGetFormat(constSpMatDescr, &format_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t spMatDescr, cusparseIndexBase_t* idxBase);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetIndexBase(hipsparseConstSpMatDescr_t spMatDescr, hipsparseIndexBase_t* idxBase);
  // CHECK: status_t = hipsparseSpMatGetIndexBase(constSpMatDescr, &indexBase_t);
  status_t = cusparseSpMatGetIndexBase(constSpMatDescr, &indexBase_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t spMatDescr, const void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstSpMatGetValues(hipsparseConstSpMatDescr_t spMatDescr, const void** values);
  // CHECK: status_t = hipsparseConstSpMatGetValues(constSpMatDescr, values_const);
  status_t = cusparseConstSpMatGetValues(constSpMatDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t spMatDescr, int* batchCount);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseConstSpMatDescr_t spMatDescr, int* batchCount);
  // CHECK: status_t = hipsparseSpMatGetStridedBatch(constSpMatDescr, &batchCount);
  status_t = cusparseSpMatGetStridedBatch(constSpMatDescr, &batchCount);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t spMatDescr, cusparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseConstSpMatDescr_t spMatDescr, hipsparseSpMatAttribute_t attribute, void* data, size_t dataSize);
  // CHECK: status_t = hipsparseSpMatGetAttribute(constSpMatDescr, spMatAttribute_t, &data, dataSize);
  status_t = cusparseSpMatGetAttribute(constSpMatDescr, spMatAttribute_t, &data, dataSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, cudaDataType valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstDnVec(hipsparseConstDnVecDescr_t* dnVecDescr, int64_t size, const void* values, hipDataType valueType);
  // CHECK: status_t = hipsparseCreateConstDnVec(&constDnVecDescr, size, values, dataType);
  status_t = cusparseCreateConstDnVec(&constDnVecDescr, size, values, dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnVec(cusparseConstDnVecDescr_t dnVecDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnVec(hipsparseConstDnVecDescr_t dnVecDescr);
  // CHECK: status_t = hipsparseDestroyDnVec(constDnVecDescr);
  status_t = cusparseDestroyDnVec(constDnVecDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnVecGet(cusparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstDnVecGet(hipsparseConstDnVecDescr_t dnVecDescr, int64_t* size, const void** values, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstDnVecGet(constDnVecDescr, &size, values_const, &dataType);
  status_t = cusparseConstDnVecGet(constDnVecDescr, &size, values_const, &dataType);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, cudaDataType valueType, cusparseOrder_t order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseCreateConstDnMat(hipsparseConstDnMatDescr_t* dnMatDescr, int64_t rows, int64_t cols, int64_t ld, const void* values, hipDataType valueType, hipsparseOrder_t order);
  // CHECK: status_t = hipsparseCreateConstDnMat(&constDnMatDescr, rows, cols, ld, values, dataType, order_t);
  status_t = cusparseCreateConstDnMat(&constDnMatDescr, rows, cols, ld, values, dataType, order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyDnMat(cusparseConstDnMatDescr_t dnMatDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDestroyDnMat(hipsparseConstDnMatDescr_t dnMatDescr);
  // CHECK: status_t = hipsparseDestroyDnMat(constDnMatDescr);
  status_t = cusparseDestroyDnMat(constDnMatDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnMatGet(cusparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, cudaDataType* type, cusparseOrder_t* order);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstDnMatGet(hipsparseConstDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols, int64_t* ld, const void** values, hipDataType* valueType, hipsparseOrder_t* order);
  // CHECK: status_t = hipsparseConstDnMatGet(constDnMatDescr, &rows, &cols, &ld, values_const, &dataType, &order_t);
  status_t = cusparseConstDnMatGet(constDnMatDescr, &rows, &cols, &ld, values_const, &dataType, &order_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t dnMatDescr, const void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstDnMatGetValues(hipsparseConstDnMatDescr_t dnMatDescr, const void** values);
  // CHECK: status_t = hipsparseConstDnMatGetValues(constDnMatDescr, values_const);
  status_t = cusparseConstDnMatGetValues(constDnMatDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseConstDnMatDescr_t dnMatDescr, int* batchCount, int64_t* batchStride);
  // CHECK: status_t = hipsparseDnMatGetStridedBatch(constDnMatDescr, &batchCount, &batchStride);
  status_t = cusparseDnMatGetStridedBatch(constDnMatDescr, &batchCount, &batchStride);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseAxpby(cusparseHandle_t handle, const void* alpha, cusparseConstSpVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t handle, const void* alpha, hipsparseConstSpVecDescr_t vecX, const void* beta, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseAxpby(handle_t, alpha, constSpVecDescr, beta, vecY);
  status_t = cusparseAxpby(handle_t, alpha, constSpVecDescr, beta, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGather(cusparseHandle_t handle, cusparseConstDnVecDescr_t vecY, cusparseSpVecDescr_t vecX);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseGather(hipsparseHandle_t handle, hipsparseConstDnVecDescr_t vecY, hipsparseSpVecDescr_t vecX);
  // CHECK: status_t = hipsparseGather(handle_t, constDnVecDescr, spVecDescr_t);
  status_t = cusparseGather(handle_t, constDnVecDescr, spVecDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScatter(cusparseHandle_t handle, cusparseConstSpVecDescr_t vecX, cusparseDnVecDescr_t vecY);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseScatter(hipsparseHandle_t handle, hipsparseConstSpVecDescr_t vecX, hipsparseDnVecDescr_t vecY);
  // CHECK: status_t = hipsparseScatter(handle_t, constSpVecDescr, vecY);
  status_t = cusparseScatter(handle_t, constSpVecDescr, vecY);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t handle, hipsparseConstSpMatDescr_t matA, hipsparseDnMatDescr_t matB, hipsparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSparseToDense_bufferSize(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, &bufferSize);
  status_t = cusparseSparseToDense_bufferSize(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t handle, hipsparseConstSpMatDescr_t matA, hipsparseDnMatDescr_t matB, hipsparseSparseToDenseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSparseToDense(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, tempBuffer);
  status_t = cusparseSparseToDense(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t handle, hipsparseConstDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseDenseToSparse_bufferSize(handle_t, dnmatB, spMatDescr_t, denseToSparseAlg_t, &bufferSize);
  status_t = cusparseDenseToSparse_bufferSize(handle_t, dnmatB, spMatDescr_t, denseToSparseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t handle, hipsparseConstDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseDenseToSparse_analysis(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, tempBuffer);
  status_t = cusparseDenseToSparse_analysis(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t handle, hipsparseConstDnMatDescr_t matA, hipsparseSpMatDescr_t matB, hipsparseDenseToSparseAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseDenseToSparse_convert(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, tempBuffer);
  status_t = cusparseDenseToSparse_convert(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opX, hipsparseConstSpVecDescr_t vecX, hipsparseConstDnVecDescr_t vecY, void* result, hipDataType computeType, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpVV_bufferSize(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, &bufferSize);
  status_t = cusparseSpVV_bufferSize(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, void* result, cudaDataType computeType, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t handle, hipsparseOperation_t opX, hipsparseConstSpVecDescr_t vecX, hipsparseConstDnVecDescr_t vecY, void* result, hipDataType computeType, void* externalBuffer);
  // CHECK: status_t = hipsparseSpVV(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, tempBuffer);
  status_t = cusparseSpVV(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t vecX, const void* beta, const hipsparseDnVecDescr_t vecY, hipDataType computeType, hipsparseSpMVAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpMV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, &bufferSize);
  status_t = cusparseSpMV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t vecX, const void* beta, const hipsparseDnVecDescr_t vecY, hipDataType computeType, hipsparseSpMVAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMV(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, tempBuffer);
  status_t = cusparseSpMV(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpMM_bufferSize(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);
  status_t = cusparseSpMM_bufferSize(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMM_preprocess(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
  status_t = cusparseSpMM_preprocess(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnMatDescr_t matB, const void* beta, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpMMAlg_t alg, void* externalBuffer);
  // CHECK: status_t = hipsparseSpMM(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
  status_t = cusparseSpMM(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // CHECK: status_t = hipsparseSpGEMM_workEstimation(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMM_workEstimation(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);
  // CHECK: status_t = hipsparseSpGEMM_compute(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMM_compute(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMM_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr);
  // CHECK: status_t = hipsparseSpGEMM_copy(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);
  status_t = cusparseSpGEMM_copy(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_workEstimation(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
  // CHECK: status_t = hipsparseSpGEMMreuse_workEstimation(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);
  status_t = cusparseSpGEMMreuse_workEstimation(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_nnz(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2, size_t* bufferSize3, void* externalBuffer3, size_t* bufferSize4, void* externalBuffer4);
  // CHECK: status_t = hipsparseSpGEMMreuse_nnz(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize2, tempBuffer2, &bufferSize3, tempBuffer3, &bufferSize4, tempBuffer4);
  status_t = cusparseSpGEMMreuse_nnz(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize2, tempBuffer2, &bufferSize3, tempBuffer3, &bufferSize4, tempBuffer4);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_compute(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, const void* beta, hipsparseSpMatDescr_t matC, hipDataType computeType, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr);
  // CHECK: status_t = hipsparseSpGEMMreuse_compute(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);
  status_t = cusparseSpGEMMreuse_compute(handle_t, opA, opB, alpha, constSpMatDescr, constSpMatDescrB, beta, spmatC, dataType, spGEMMAlg_t, spGEMMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpGEMMreuse_copy(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, hipsparseConstSpMatDescr_t matA, hipsparseConstSpMatDescr_t matB, hipsparseSpMatDescr_t matC, hipsparseSpGEMMAlg_t alg, hipsparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize5, void* externalBuffer5);
  // CHECK: status_t = hipsparseSpGEMMreuse_copy(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize5, tempBuffer5);
  status_t = cusparseSpGEMMreuse_copy(handle_t, opA, opB, constSpMatDescr, constSpMatDescrB, spmatC, spGEMMAlg_t, spGEMMDescr, &bufferSize5, tempBuffer5);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstDnMatDescr_t A, hipsparseConstDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, size_t* bufferSize);
  // CHECK: status_t = hipsparseSDDMM_bufferSize(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);
  status_t = cusparseSDDMM_bufferSize(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstDnMatDescr_t A, hipsparseConstDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM_preprocess(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM_preprocess(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstDnMatDescr_t A, hipsparseConstDnMatDescr_t B, const void* beta, hipsparseSpMatDescr_t C, hipDataType computeType, hipsparseSDDMMAlg_t alg, void* tempBuffer);
  // CHECK: status_t = hipsparseSDDMM(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpSV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr, &bufferSize);
  status_t = cusparseSpSV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_analysis(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr, void* externalBuffer);
  // CHECK: status_t = hipsparseSpSV_analysis(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr, tempBuffer);
  status_t = cusparseSpSV_analysis(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_solve(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t handle, hipsparseOperation_t opA, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnVecDescr_t x, const hipsparseDnVecDescr_t y, hipDataType computeType, hipsparseSpSVAlg_t alg, hipsparseSpSVDescr_t spsvDescr);
  // CHECK: status_t = hipsparseSpSV_solve(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr);
  status_t = cusparseSpSV_solve(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, size_t* bufferSize);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnMatDescr_t matB, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpSMAlg_t alg, hipsparseSpSMDescr_t spsmDescr, size_t* bufferSize);
  // CHECK: status_t = hipsparseSpSM_bufferSize(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr, &bufferSize);
  status_t = cusparseSpSM_bufferSize(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, hipsparseConstSpMatDescr_t matA, hipsparseConstDnMatDescr_t matB, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpSMAlg_t alg, hipsparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // CHECK: status_t = hipsparseSpSM_analysis(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr, tempBuffer);
  status_t = cusparseSpSM_analysis(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t dnVecDescr, const void** values);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstDnVecGetValues(hipsparseConstDnVecDescr_t dnVecDescr, const void** values);
  // CHECK: status_t = hipsparseConstDnVecGetValues(constDnVecDescr, values_const);
  status_t = cusparseConstDnVecGetValues(constDnVecDescr, values_const);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseConstCscGet(cusparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cscColOffsets, const void** cscRowInd, const void** cscValues, cusparseIndexType_t* cscColOffsetsType, cusparseIndexType_t* cscRowIndType, cusparseIndexBase_t* idxBase, cudaDataType* valueType);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseConstCscGet(hipsparseConstSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols, int64_t* nnz, const void** cscColOffsets, const void** cscRowInd, const void** cscValues, hipsparseIndexType_t* cscColOffsetsType, hipsparseIndexType_t* cscRowIndType, hipsparseIndexBase_t* idxBase, hipDataType* valueType);
  // CHECK: status_t = hipsparseConstCscGet(constSpMatDescr, &rows, &cols, &nnz, cscColOffsets_const, cscRowInd_const, cscValues_const, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
  status_t = cusparseConstCscGet(constSpMatDescr, &rows, &cols, &nnz, cscColOffsets_const, cscRowInd_const, cscValues_const, &cscColOffsetsType, &cscRowIndType, &indexBase_t, &dataType);
#endif

  return 0;
}
