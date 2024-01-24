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
  int nnza = 0;
  int nnzb = 0;
  int nnzc = 0;
  int nnzd = 0;
  int csrRowPtrA = 0;
  int csrRowPtrB = 0;
  int csrRowPtrC = 0;
  int csrRowPtrD = 0;
  int csrColIndA = 0;
  int csrColIndB = 0;
  int csrColIndC = 0;
  int csrColIndD = 0;
  int bufferSizeInBytes = 0;
  size_t bufferSize = 0;
  double dA = 0.f;
  double dB = 0.f;
  double dAlpha = 0.f;
  double dF = 0.f;
  double dX = 0.f;
  double dcsrSortedValA = 0.f;
  double dcsrSortedValB = 0.f;
  double dcsrSortedValC = 0.f;
  double dcsrSortedValD = 0.f;
  float fAlpha = 0.f;
  float fA = 0.f;
  float fB = 0.f;
  float fF = 0.f;
  float fX = 0.f;
  float csrSortedValA = 0.f;
  float csrSortedValB = 0.f;
  float csrSortedValC = 0.f;
  float csrSortedValD = 0.f;
  void *pBuffer = nullptr;
  void *tempBuffer = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION < 12000
  // CHECK: rocsparse_mat_descr csrsv2_info;
  csrsv2Info_t csrsv2_info;
  // CHECK: rocsparse_mat_info csrgemm2_info;
  csrgemm2Info_t csrgemm2_info;

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

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsv_analysis(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrsv_analysis(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsv_analysis(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrsv_analysis(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsv_analysis(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrsv_analysis(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_analysis(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsv_analysis(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrsv_analysis(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrsv2_analysis(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDcsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSV) cusparseStatus_t CUSPARSEAPI cusparseScsrsv2_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int nnz, const cusparseMatDescr_t descrA, float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, csrsv2Info_t info, int* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsv_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int nnz, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_mat_info info, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrsv_buffer_size(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseScsrsv2_bufferSize(handle_t, opA, m, innz, matDescr_A, &csrSortedValA, &csrRowPtrA, &csrColIndA, csrsv2_info, &bufferSizeInBytes);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuDoubleComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuDoubleComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const cuDoubleComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, cuDoubleComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrgemm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_double_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_double_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_double_complex* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_double_complex* csr_val_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, const rocsparse_mat_descr descr_C, rocsparse_double_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, const rocsparse_mat_info info_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrgemm(handle_t, rocsparse_operation_none, rocsparse_operation_none, m, n, k, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &dComplexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseZcsrgemm2(handle_t, m, n, k, &dcomplexA, matDescr_A, nnza, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dComplexcsrSortedValB, &csrRowPtrB, &csrColIndB, &dcomplexB, matDescr_D, nnzd, &dComplexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dComplexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const cuComplex* alpha, const cusparseMatDescr_t descrA, int nnzA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const cuComplex* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const cuComplex* beta, const cusparseMatDescr_t descrD, int nnzD, const cuComplex* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, cuComplex* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrgemm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const rocsparse_float_complex* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const rocsparse_float_complex* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const rocsparse_float_complex* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const rocsparse_float_complex* csr_val_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, const rocsparse_mat_descr descr_C, rocsparse_float_complex* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, const rocsparse_mat_info info_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrgemm(handle_t, rocsparse_operation_none, rocsparse_operation_none, m, n, k, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &complexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseCcsrgemm2(handle_t, m, n, k, &complexA, matDescr_A, nnza, &complexcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &complexcsrSortedValB, &csrRowPtrB, &csrColIndB, &complexB, matDescr_D, nnzd, &complexcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &complexcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2(cusparseHandle_t handle, int m, int n, int k, const double* alpha, const cusparseMatDescr_t descrA, int nnzA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const double* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const double* beta, const cusparseMatDescr_t descrD, int nnzD, const double* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, double* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrgemm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const double* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const double* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const double* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const double* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const double* csr_val_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, const rocsparse_mat_descr descr_C, double* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, const rocsparse_mat_info info_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrgemm(handle_t, rocsparse_operation_none, rocsparse_operation_none, m, n, k, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &dcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseDcsrgemm2(handle_t, m, n, k, &dA, matDescr_A, nnza, &dcsrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &dcsrSortedValB, &csrRowPtrB, &csrColIndB, &dB, matDescr_D, nnzd, &dcsrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &dcsrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpGEMM) cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2(cusparseHandle_t handle, int m, int n, int k, const float* alpha, const cusparseMatDescr_t descrA, int nnzA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cusparseMatDescr_t descrB, int nnzB, const float* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB, const float* beta, const cusparseMatDescr_t descrD, int nnzD, const float* csrSortedValD, const int* csrSortedRowPtrD, const int* csrSortedColIndD, const cusparseMatDescr_t descrC, float* csrSortedValC, const int* csrSortedRowPtrC, int* csrSortedColIndC, const csrgemm2Info_t info, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrgemm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n, rocsparse_int k, const float* alpha, const rocsparse_mat_descr descr_A, rocsparse_int nnz_A, const float* csr_val_A, const rocsparse_int* csr_row_ptr_A, const rocsparse_int* csr_col_ind_A, const rocsparse_mat_descr descr_B, rocsparse_int nnz_B, const float* csr_val_B, const rocsparse_int* csr_row_ptr_B, const rocsparse_int* csr_col_ind_B, const float* beta, const rocsparse_mat_descr descr_D, rocsparse_int nnz_D, const float* csr_val_D, const rocsparse_int* csr_row_ptr_D, const rocsparse_int* csr_col_ind_D, const rocsparse_mat_descr descr_C, float* csr_val_C, const rocsparse_int* csr_row_ptr_C, rocsparse_int* csr_col_ind_C, const rocsparse_mat_info info_C, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrgemm(handle_t, rocsparse_operation_none, rocsparse_operation_none, m, n, k, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
  status_t = cusparseScsrgemm2(handle_t, m, n, k, &fA, matDescr_A, nnza, &csrSortedValA, &csrRowPtrA, &csrColIndA, matDescr_B, nnzb, &csrSortedValB, &csrRowPtrB, &csrColIndB, &fB, matDescr_D, nnzd, &csrSortedValD, &csrRowPtrD, &csrColIndD, matDescr_C, &csrSortedValC, &csrRowPtrC, &csrColIndC, csrgemm2_info, pBuffer);
#endif

  return 0;
}
