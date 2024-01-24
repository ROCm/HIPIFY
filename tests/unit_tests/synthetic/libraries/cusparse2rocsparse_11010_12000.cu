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

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: rocsparse_spmat_descr spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: rocsparse_dnmat_descr dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;
#endif

#if CUDA_VERSION >= 11010 && CUSPARSE_VERSION >= 11300
  // CHECK: rocsparse_sparse_to_dense_alg sparseToDenseAlg_t;
  cusparseSparseToDenseAlg_t sparseToDenseAlg_t;

  // CHECK: rocsparse_dense_to_sparse_alg denseToSparseAlg_t;
  cusparseDenseToSparseAlg_t denseToSparseAlg_t;

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, void* buffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle handle, const rocsparse_spmat_descr mat_A, rocsparse_dnmat_descr mat_B, rocsparse_sparse_to_dense_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_sparse_to_dense(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, nullptr, tempBuffer);
  status_t = cusparseSparseToDense(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, tempBuffer);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSparseToDense_bufferSize(cusparseHandle_t handle, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle handle, const rocsparse_spmat_descr mat_A, rocsparse_dnmat_descr mat_B, rocsparse_sparse_to_dense_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_sparse_to_dense(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, &bufferSize, nullptr);
  status_t = cusparseSparseToDense_bufferSize(handle_t, spmatA, dnmatB, sparseToDenseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_bufferSize(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle handle, const rocsparse_dnmat_descr mat_A, rocsparse_spmat_descr mat_B, rocsparse_dense_to_sparse_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_dense_to_sparse(handle_t, dnmatA, spmatB, denseToSparseAlg_t, &bufferSize, nullptr);
  status_t = cusparseDenseToSparse_bufferSize(handle_t, dnmatA, spmatB, denseToSparseAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDenseToSparse_analysis(cusparseHandle_t handle, cusparseDnMatDescr_t matA, cusparseSpMatDescr_t matB, cusparseDenseToSparseAlg_t alg, void* buffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle handle, const rocsparse_dnmat_descr mat_A, rocsparse_spmat_descr mat_B, rocsparse_dense_to_sparse_alg alg, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_dense_to_sparse(handle_t, dnmatA, spmatB, denseToSparseAlg_t, nullptr, tempBuffer);
  status_t = cusparseDenseToSparse_analysis(handle_t, dnmatA, spmatB, denseToSparseAlg_t, tempBuffer);
#endif
#endif

  return 0;
}
