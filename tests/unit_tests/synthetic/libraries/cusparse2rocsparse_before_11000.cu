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

  // CHECK: rocsparse_status status_t;
  cusparseStatus_t status_t;

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

  // CHECK: rocsparse_solve_policy solvePolicy_t;
  cusparseSolvePolicy_t solvePolicy_t;

  int m = 0;
  int n = 0;
  int k = 0;
  int innz = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int csrRowPtrA = 0;
  int csrColIndA = 0;
  double dAlpha = 0.f;
  double dBeta = 0.f;
  double dA = 0.f;
  double dB = 0.f;
  double dC = 0.f;
  double dF = 0.f;
  double dX = 0.f;
  double dY = 0.f;
  double dcsrSortedValA = 0.f;
  float fAlpha = 0.f;
  float fBeta = 0.f;
  float fA = 0.f;
  float fB = 0.f;
  float fC = 0.f;
  float fF = 0.f;
  float fX = 0.f;
  float fY = 0.f;
  float csrSortedValA = 0.f;
  void *pBuffer = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION < 11000
  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseZcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* x, const cuDoubleComplex* beta, cuDoubleComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrmv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const rocsparse_double_complex* x, const rocsparse_double_complex* beta, rocsparse_double_complex* y);
  // CHECK: status_t = rocsparse_zcsrmv(handle_t, opA, m, n, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, nullptr, &dcomplexX, &dcomplexBeta, &dcomplexY);
  status_t = cusparseZcsrmv(handle_t, opA, m, n, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexX, &dcomplexBeta, &dcomplexY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseCcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* x, const cuComplex* beta, cuComplex* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrmv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const rocsparse_float_complex* x, const rocsparse_float_complex* beta, rocsparse_float_complex* y);
  // CHECK: status_t = rocsparse_ccsrmv(handle_t, opA, m, n, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, nullptr, &complexX, &complexBeta, &complexY);
  status_t = cusparseCcsrmv(handle_t, opA, m, n, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexX, &complexBeta, &complexY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* x, const double* beta, double* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrmv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const double* x, const double* beta, double* y);
  // CHECK: status_t = rocsparse_dcsrmv(handle_t, opA, m, n, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, nullptr, &dX, &dBeta, &dY);
  status_t = cusparseDcsrmv(handle_t, opA, m, n, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dX, &dBeta, &dY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMV) cusparseStatus_t CUSPARSEAPI cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* x, const float* beta, float* y);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrmv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const float* x, const float* beta, float* y);
  // CHECK: status_t = rocsparse_scsrmv(handle_t, opA, m, n, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, nullptr, &fX, &fBeta, &fY);
  status_t = cusparseScsrmv(handle_t, opA, m, n, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fX, &fBeta, &fY);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_double_complex* B, rocsparse_int ldb, const rocsparse_double_complex* beta, rocsparse_double_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_zcsrmm(handle_t, opA, rocsparse_operation_none, m, n, k, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);
  status_t = cusparseZcsrmm(handle_t, opA, m, n, k, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, &dcomplexBeta, &dcomplexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_float_complex* B, rocsparse_int ldb, const rocsparse_float_complex* beta, rocsparse_float_complex* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_ccsrmm(handle_t, opA, rocsparse_operation_none, m, n, k, innz, &complexA, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);
  status_t = cusparseCcsrmm(handle_t, opA, m, n, k, innz, &complexA, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, ldb, &complexBeta, &complexC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* B, rocsparse_int ldb, const double* beta, double* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_dcsrmm(handle_t, opA, rocsparse_operation_none, m, n, k, innz, &dA, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);
  status_t = cusparseDcsrmm(handle_t, opA, m, n, k, innz, &dA, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, ldb, &dBeta, &dC, ldc);

  // CUDA: CUSPARSE_DEPRECATED_HINT(cusparseSpMM) cusparseStatus_t CUSPARSEAPI cusparseScsrmm(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int k, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* B, rocsparse_int ldb, const float* beta, float* C, rocsparse_int ldc);
  // CHECK: status_t = rocsparse_scsrmm(handle_t, opA, rocsparse_operation_none, m, n, k, innz, &fA, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
  status_t = cusparseScsrmm(handle_t, opA, m, n, k, innz, &fA, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, ldb, &fBeta, &fC, ldc);
#endif

  return 0;
}
