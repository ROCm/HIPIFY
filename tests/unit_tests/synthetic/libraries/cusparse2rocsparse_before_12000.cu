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
  int innz = 0;
  int csrRowPtrA = 0;
  int csrColIndA = 0;
  double dAlpha = 0.f;
  double dF = 0.f;
  double dX = 0.f;
  double dcsrSortedValA = 0.f;
  float fAlpha = 0.f;
  float fF = 0.f;
  float fX = 0.f;
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

#if CUDA_VERSION < 12000
  // CHECK: rocsparse_mat_descr csrsv2_info;
  csrsv2Info_t csrsv2_info;

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuDoubleComplex* f, cuDoubleComplex* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsv_solve(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const rocsparse_double_complex* x, rocsparse_double_complex* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrsv_solve(handle_t, opA, m, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dcomplexF, &dcomplexX, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrsv2_solve(handle_t, opA, m, innz, &dcomplexAlpha, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dcomplexF, &dcomplexX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const cuComplex* f, cuComplex* x,cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsv_solve(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const rocsparse_float_complex* x, rocsparse_float_complex* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrsv_solve(handle_t, opA, m, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &complexF, &complexX, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrsv2_solve(handle_t, opA, m, innz, &complexAlpha, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &complexF, &complexX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const double* f, double* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsv_solve(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const double* x, double* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrsv_solve(handle_t, opA, m, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dF, &dX, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrsv2_solve(handle_t, opA, m, innz, &dAlpha, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &dF, &dX, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_solve(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, const float* f, float* x, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsv_solve(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, const float* x, float* y, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrsv_solve(handle_t, opA, m, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &fF, &fX, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrsv2_solve(handle_t, opA, m, innz, &fAlpha, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &fF, &fX, solvePolicy_t, pBuffer);
#endif

  return 0;
}
