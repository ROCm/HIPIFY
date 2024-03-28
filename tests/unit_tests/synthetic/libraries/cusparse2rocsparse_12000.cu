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
  void *alpha = nullptr;
  void *beta = nullptr;
  void* result = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION >= 8000
  // TODO: [#899] There should be rocsparse_datatype instead of hipDataType
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if (CUDA_VERSION >= 10010 && CUSPARSE_VERSION >= 10200 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: rocsparse_spmat_descr spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: rocsparse_dnmat_descr dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;

  // CHECK: rocsparse_spmm_alg spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;

  // CHECK: _rocsparse_dnvec_descr *dnVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnvec_descr dnVecDescr_t, vecX, vecY;
  cusparseDnVecDescr *dnVecDescr = nullptr;
  cusparseDnVecDescr_t dnVecDescr_t, vecX, vecY;

  // CHECK: rocsparse_spmv_alg spMVAlg_t;
  cusparseSpMVAlg_t spMVAlg_t;
#endif

#if CUDA_VERSION >= 11010 && CUSPARSE_VERSION >= 11300
  // CHECK: rocsparse_sparse_to_dense_alg sparseToDenseAlg_t;
  cusparseSparseToDenseAlg_t sparseToDenseAlg_t;

  // CHECK: rocsparse_dense_to_sparse_alg denseToSparseAlg_t;
  cusparseDenseToSparseAlg_t denseToSparseAlg_t;
#endif

#if CUDA_VERSION >= 11020 && CUSPARSE_VERSION >= 11400
  // CHECK: rocsparse_sddmm_alg sDDMMAlg_t;
  // CHECK-NEXT: rocsparse_sddmm_alg SDDMM_ALG_DEFAULT = rocsparse_sddmm_alg_default;
  cusparseSDDMMAlg_t sDDMMAlg_t;
  cusparseSDDMMAlg_t SDDMM_ALG_DEFAULT = CUSPARSE_SDDMM_ALG_DEFAULT;
#endif

#if CUDA_VERSION >= 11030 && CUSPARSE_VERSION >= 11600
  // CHECK: rocsparse_spsm_alg spSMAlg_t;
  // CHECK-NEXT: rocsparse_spsm_alg SPSM_ALG_DEFAULT = rocsparse_spsm_alg_default;
  cusparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t SPSM_ALG_DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT;

  // NOTE:cusparseSpSMDescr_t doesn't have a correspondence in rocSPARSE, the corresponding function argument is removed in the hipified call of the rocsparse_spsm function
  cusparseSpSMDescr_t spSMDescr;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: rocsparse_spsv_alg spSVAlg_t;
  // CHECK-NEXT: rocsparse_spsv_alg SPSV_ALG_DEFAULT = rocsparse_spsv_alg_default;
  cusparseSpSVAlg_t spSVAlg_t;
  cusparseSpSVAlg_t SPSV_ALG_DEFAULT = CUSPARSE_SPSV_ALG_DEFAULT;

  // TODO: remove decalration of cusparseSpSVDescr_t, as it is not mirroved and not used in rocSPARSE
  cusparseSpSVDescr_t spSVDescr;

  // CHECK: rocsparse_const_spmat_descr constSpMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_const_spmat_descr constSpMatDescrB = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescr = nullptr;
  cusparseConstSpMatDescr_t constSpMatDescrB = nullptr;

  // CHECK: rocsparse_const_dnmat_descr constDnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_const_dnmat_descr constDnMatDescrB = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescr = nullptr;
  cusparseConstDnMatDescr_t constDnMatDescrB = nullptr;

  // CHECK: rocsparse_const_spvec_descr constSpVecDescr = nullptr;
  cusparseConstSpVecDescr_t constSpVecDescr = nullptr;

  // CHECK: rocsparse_const_dnvec_descr constDnVecDescr = nullptr;
  cusparseConstDnVecDescr_t constDnVecDescr = nullptr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle handle, rocsparse_const_spmat_descr mat_A, rocsparse_dnmat_descr mat_B, rocsparse_sparse_to_dense_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_sparse_to_dense(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, nullptr, tempBuffer);
  status_t = cusparseSparseToDense(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseConstSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle handle, rocsparse_const_spmat_descr mat_A, rocsparse_dnmat_descr mat_B, rocsparse_sparse_to_dense_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_sparse_to_dense(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, &bufferSize, nullptr);
  status_t = cusparseSparseToDense_bufferSize(handle_t, constSpMatDescr, dnmatB, sparseToDenseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle handle, rocsparse_const_dnmat_descr mat_A, rocsparse_spmat_descr mat_B, rocsparse_dense_to_sparse_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_dense_to_sparse(handle_t, dnmatB, spMatDescr_t, denseToSparseAlg_t, &bufferSize, nullptr);
  status_t = cusparseDenseToSparse_bufferSize(handle_t, dnmatB, spMatDescr_t, denseToSparseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseConstDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle handle, rocsparse_const_dnmat_descr mat_A, rocsparse_spmat_descr mat_B, rocsparse_dense_to_sparse_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_dense_to_sparse(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, nullptr, tempBuffer);
  status_t = cusparseDenseToSparse_analysis(handle_t, constDnMatDescr, spmatB, denseToSparseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, rocsparse_const_spmat_descr mat_A, rocsparse_const_dnmat_descr mat_B, const void* beta, const rocsparse_dnmat_descr mat_C, rocsparse_datatype compute_type, rocsparse_spmm_alg alg, rocsparse_spmm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmm(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, rocsparse_spmm_stage_compute, &bufferSize, nullptr);
  status_t = cusparseSpMM_bufferSize(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spsm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, rocsparse_const_spmat_descr matA, rocsparse_const_dnmat_descr matB, const rocsparse_dnmat_descr matC, rocsparse_datatype compute_type, rocsparse_spsm_alg alg, rocsparse_spsm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spsm(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, rocsparse_spsm_stage_compute, nullptr, tempBuffer);
  status_t = cusparseSpSM_analysis(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spsm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, rocsparse_const_spmat_descr matA, rocsparse_const_dnmat_descr matB, const rocsparse_dnmat_descr matC, rocsparse_datatype compute_type, rocsparse_spsm_alg alg, rocsparse_spsm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spsm(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, rocsparse_spsm_stage_compute, nullptr, nullptr);
  status_t = cusparseSpSM_solve(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescrB, dnmatC, dataType, spSMAlg_t, spSMDescr);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, rocsparse_const_spmat_descr mat_A, rocsparse_const_dnmat_descr mat_B, const void* beta, const rocsparse_dnmat_descr mat_C, rocsparse_datatype compute_type, rocsparse_spmm_alg alg, rocsparse_spmm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmm(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, rocsparse_spmm_stage_compute, nullptr, tempBuffer);
  status_t = cusparseSpMM(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, void* result, cudaDataType computeType, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_const_spvec_descr x, rocsparse_const_dnvec_descr y, void* result, rocsparse_datatype compute_type, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spvv(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, nullptr, tempBuffer);
  status_t = cusparseSpVV(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX, cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY, const void* result, cudaDataType computeType, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spvv(rocsparse_handle handle, rocsparse_operation trans, rocsparse_const_spvec_descr x, rocsparse_const_dnvec_descr y, void* result, rocsparse_datatype compute_type, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spvv(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, &bufferSize, nullptr);
  status_t = cusparseSpVV_bufferSize(handle_t, opX, constSpVecDescr, constDnVecDescr, result, dataType, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmv(rocsparse_handle handle, rocsparse_operation trans, const void* alpha, rocsparse_const_spmat_descr mat, rocsparse_const_dnvec_descr x, const void* beta, const rocsparse_dnvec_descr y, rocsparse_datatype compute_type, rocsparse_spmv_alg alg, rocsparse_spmv_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmv(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, rocsparse_spmv_stage_compute, tempBuffer);
  status_t = cusparseSpMV(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, const void* beta, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpMVAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmv(rocsparse_handle handle, rocsparse_operation trans, const void* alpha, rocsparse_const_spmat_descr mat, rocsparse_const_dnvec_descr x, const void* beta, const rocsparse_dnvec_descr y, rocsparse_datatype compute_type, rocsparse_spmv_alg alg, rocsparse_spmv_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmv(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, rocsparse_spmv_stage_buffer_size, &bufferSize, nullptr);
  status_t = cusparseSpMV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, beta, vecY, dataType, spMVAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, const rocsparse_spmat_descr mat_A, const rocsparse_dnmat_descr mat_B, const void* beta, const rocsparse_dnmat_descr mat_C, rocsparse_datatype compute_type, rocsparse_spmm_alg alg, rocsparse_spmm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmm(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, rocsparse_spmm_stage_preprocess, nullptr, tempBuffer);
  status_t = cusparseSpMM_preprocess(handle_t, opA, opB, alpha, constSpMatDescr, constDnMatDescr, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm_buffer_size(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, const rocsparse_dnmat_descr A, const rocsparse_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sddmm_buffer_size(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);
  status_t = cusparseSDDMM_bufferSize(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM_preprocess(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm_preprocess(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, rocsparse_const_dnmat_descr A, rocsparse_const_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, void* temp_buffer);
  // CHECK: status_t = rocsparse_sddmm_preprocess(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM_preprocess(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSDDMM(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB, const void* beta, cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSDDMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sddmm(rocsparse_handle handle, rocsparse_operation opA, rocsparse_operation opB, const void* alpha, rocsparse_const_dnmat_descr A, rocsparse_const_dnmat_descr B, const void* beta, rocsparse_spmat_descr C, rocsparse_datatype compute_type, rocsparse_sddmm_alg alg, void* temp_buffer);
  // CHECK: status_t = rocsparse_sddmm(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);
  status_t = cusparseSDDMM(handle_t, opA, opB, alpha, constDnMatDescr, constDnMatDescrB, beta, spmatC, dataType, sDDMMAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, const void* alpha, cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX, cusparseDnVecDescr_t vecY, cudaDataType computeType, cusparseSpSVAlg_t alg, cusparseSpSVDescr_t spsvDescr, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spsv(rocsparse_handle handle, rocsparse_operation trans, const void* alpha, rocsparse_const_spmat_descr mat, rocsparse_const_dnvec_descr x, const rocsparse_dnvec_descr y, rocsparse_datatype compute_type, rocsparse_spsv_alg alg, rocsparse_spsv_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spsv(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, rocsparse_spsv_stage_buffer_size, &bufferSize, nullptr);
  status_t = cusparseSpSV_bufferSize(handle_t, opA, alpha, constSpMatDescr, constDnVecDescr, vecY, dataType, spSVAlg_t, spSVDescr, &bufferSize);
#endif

  return 0;
}
