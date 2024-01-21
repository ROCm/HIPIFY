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

  int batchCount = 0;
  int m = 0;
  int algo = 0;
  int nrhs = 0;
  int innz = 0;
  int ldb = 0;
  int csrRowPtrA = 0;
  int csrColIndA = 0;
  int iposition = 0;
  double dds = 0.f;
  double ddl = 0.f;
  double dd = 0.f;
  double ddu = 0.f;
  double ddw = 0.f;
  double dx = 0.f;
  double dA = 0.f;
  double dB = 0.f;
  double dcsrSortedVal = 0.f;
  float fA = 0.f;
  float fB = 0.f;
  float fds = 0.f;
  float fdl = 0.f;
  float fd = 0.f;
  float fdu = 0.f;
  float fdw = 0.f;
  float fx = 0.f;
  float csrSortedVal = 0.f;
  size_t bufferSize = 0;
  void *pBuffer = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

  // CHECK: rocsparse_solve_policy solvePolicy_t;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_NO_LEVEL = rocsparse_solve_policy_auto;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_USE_LEVEL = rocsparse_solve_policy_auto;
  cusparseSolvePolicy_t solvePolicy_t;
  cusparseSolvePolicy_t SOLVE_POLICY_NO_LEVEL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseSolvePolicy_t SOLVE_POLICY_USE_LEVEL = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

#if CUDA_VERSION >= 9020 && CUDA_VERSION < 12000
  // CHECK: rocsparse_mat_info csrsm2_info;
  csrsm2Info_t csrsm2_info;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsm_solve(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_double_complex* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrsm_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsm_solve(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_float_complex* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrsm_solve(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsm_solve(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, double* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrsm_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_solve(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsm_solve(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, float* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrsm_solve(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrsm2_solve(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsm_analysis(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_double_complex* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_zcsrsm_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseZcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsm_analysis(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_float_complex * alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex * csr_val, const rocsparse_int * csr_row_ptr, const rocsparse_int * csr_col_ind, const rocsparse_float_complex * B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_ccsrsm_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseCcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsm_analysis(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_dcsrsm_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseDcsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_analysis(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsm_analysis(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_analysis_policy analysis, rocsparse_solve_policy solve, void* temp_buffer);
  // CHECK: status_t = rocsparse_scsrsm_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, rocsparse_analysis_policy_force, rocsparse_solve_policy_auto, pBuffer);
  status_t = cusparseScsrsm2_analysis(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, pBuffer);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuDoubleComplex* alpha, const cusparseMatDescr_t descrA, const cuDoubleComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuDoubleComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zcsrsm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_double_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_double_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_double_complex* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zcsrsm_buffer_size(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, rocsparse_solve_policy_auto, &bufferSize);
  status_t = cusparseZcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dcomplexA, matDescr_A, &dComplexcsrSortedValA, &csrRowPtrA, &csrColIndA, &dcomplexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const cuComplex* alpha, const cusparseMatDescr_t descrA, const cuComplex* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const cuComplex* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_ccsrsm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const rocsparse_float_complex* alpha, const rocsparse_mat_descr descr, const rocsparse_float_complex* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const rocsparse_float_complex* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, size_t* buffer_size);
  // CHECK: status_t = rocsparse_ccsrsm_buffer_size(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, rocsparse_solve_policy_auto, &bufferSize);
  status_t = cusparseCcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &complexA, matDescr_A, &complex, &csrRowPtrA, &csrColIndA, &complexB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const double* alpha, const cusparseMatDescr_t descrA, const double* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const double* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dcsrsm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const double* alpha, const rocsparse_mat_descr descr, const double* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const double* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dcsrsm_buffer_size(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, rocsparse_solve_policy_auto, &bufferSize);
  status_t = cusparseDcsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &dA, matDescr_A, &dcsrSortedVal, &csrRowPtrA, &csrColIndA, &dB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseScsrsm2_bufferSizeExt(cusparseHandle_t handle, int algo, cusparseOperation_t transA, cusparseOperation_t transB, int m, int nrhs, int nnz, const float* alpha, const cusparseMatDescr_t descrA, const float* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA, const float* B, int ldb, csrsm2Info_t info, cusparseSolvePolicy_t policy, size_t* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_scsrsm_buffer_size(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, rocsparse_int m, rocsparse_int nrhs, rocsparse_int nnz, const float* alpha, const rocsparse_mat_descr descr, const float* csr_val, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, const float* B, rocsparse_int ldb, rocsparse_mat_info info, rocsparse_solve_policy policy, size_t* buffer_size);
  // CHECK: status_t = rocsparse_scsrsm_buffer_size(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, rocsparse_solve_policy_auto, &bufferSize);
  status_t = cusparseScsrsm2_bufferSizeExt(handle_t, algo, opA, opB, m, nrhs, innz, &fA, matDescr_A, &csrSortedVal, &csrRowPtrA, &csrColIndA, &fB, ldb, csrsm2_info, solvePolicy_t, &bufferSize);

  // TODO: rocsparse_csrsm_zero_pivot needs explicit synchronization because cusparseXcsrsm2_zeroPivot is blocking
  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle, csrsm2Info_t info, int* position);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csrsm_zero_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position);
  // CHECK: status_t = rocsparse_csrsm_zero_pivot(handle_t, csrsm2_info, &iposition);
  status_t = cusparseXcsrsm2_zeroPivot(handle_t, csrsm2_info, &iposition);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseCreateCsrsm2Info(csrsm2Info_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info);
  // CHECK: status_t = rocsparse_create_mat_info(&csrsm2_info);
  status_t = cusparseCreateCsrsm2Info(&csrsm2_info);

  // CUDA: CUSPARSE_DEPRECATED(cusparseSpSM) cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrsm2Info(csrsm2Info_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info);
  // CHECK: status_t = rocsparse_destroy_mat_info(csrsm2_info);
  status_t = cusparseDestroyCsrsm2Info(csrsm2_info);
#endif

  return 0;
}
