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

  int m = 0;
  int n = 0;
  int k = 0;
  int nnza = 0;
  int nnzb = 0;
  int nnzc = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int csrRowPtrA = 0;
  int csrRowPtrB = 0;
  int csrRowPtrC = 0;
  int csrColIndA = 0;
  int csrColIndB = 0;
  int csrColIndC = 0;
  int nnzTotalDevHostPtr = 0;
  double dAlpha = 0.f;
  double dBeta = 0.f;
  double dA = 0.f;
  double dB = 0.f;
  double dC = 0.f;
  double dcsrSortedValA = 0.f;
  double dcsrSortedValB = 0.f;
  double dcsrSortedValC = 0.f;
  float fA = 0.f;
  float fB = 0.f;
  float fC = 0.f;
  float csrSortedValA = 0.f;
  float csrSortedValB = 0.f;
  float csrSortedValC = 0.f;
  void *pBuffer = nullptr;
  void *workspace = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION >= 10000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrgeam2(cusparseHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_double_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_double_complex* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_double_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_double_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_zcsrgeam(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseZcsrgeam2(handle_t, m, n, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrgeam2(cusparseHandle_t handle, int m, int n, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* beta, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_float_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_float_complex* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_float_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_float_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_ccsrgeam(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseCcsrgeam2(handle_t, m, n, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, &complexB, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrgeam2(cusparseHandle_t handle, int m, int n, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* beta, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, double* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const double* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const double* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const double* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, double* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_dcsrgeam(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseDcsrgeam2(handle_t, m, n, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, &dB, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrgeam2(cusparseHandle_t handle, int m, int n, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* beta, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, float* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrgeam(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const float* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const float* beta, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const float* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, float* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C);
  // CHECK: status_t = rocsparse_scsrgeam(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC);
  status_t = cusparseScsrgeam2(handle_t, m, n, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, &fB, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA, int nnzA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cusparseMatDescr_t descrC, int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrgeam_nnz(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_mat_descr descr_C, rocsparse_int* csr_row_ptr_C, rocsparse_int* nnz_C);
  // CHECK: status_t = rocsparse_csrgeam_nnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr);
  status_t = cusparseXcsrgeam2Nnz(handle_t, m, n, matDescr_A, nnza, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrRowPtrB, &csrColIndB, matDescr_C, &csrRowPtrC, &nnzTotalDevHostPtr, workspace);
#endif

  return 0;
}
