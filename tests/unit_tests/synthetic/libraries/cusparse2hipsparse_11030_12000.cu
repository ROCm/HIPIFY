// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --use-hip-data-types %clang_args -ferror-limit=500

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
  printf("17.1. cuSPARSE API to hipSPARSE API synthetic test\n");

  // CHECK: hipsparseStatus_t status_t;
  cusparseStatus_t status_t;

  // CHECK: hipsparseHandle_t handle_t;
  cusparseHandle_t handle_t;

  // CHECK: hipsparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

  // CHECK: hipsparseOperation_t opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

  // CHECK: hipsparseSolvePolicy_t solvePolicy_t;
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
  void *alpha = nullptr;
  void *pBuffer = nullptr;
  void *tempBuffer = nullptr;

  // CHECK: hipDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexAlpha, dcomplexB, dcomplexBeta, dcomplexC, dcomplexF, dcomplexX, dcomplexY, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dComplexcsrSortedValD, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // CHECK: hipComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexAlpha, complexB, complexBeta, complexC, complexF, complexX, complexY, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complexcsrSortedValD, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if (CUDA_VERSION >= 10010 && CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: hipsparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: hipsparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;
#endif

#if CUDA_VERSION >= 11030 && CUSPARSE_VERSION >= 11600
  // CHECK: hipsparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t spSMAlg_t;

  // CHECK: hipsparseSpSMDescr_t spSMDescr;
  cusparseSpSMDescr_t spSMDescr;

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr);
  // HIP: HIPSPARSE_EXPORT hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t handle, hipsparseOperation_t opA, hipsparseOperation_t opB, const void* alpha, const hipsparseSpMatDescr_t matA, const hipsparseDnMatDescr_t matB, const hipsparseDnMatDescr_t matC, hipDataType computeType, hipsparseSpSMAlg_t alg, hipsparseSpSMDescr_t spsmDescr, void* externalBuffer);
  // CHECK: status_t = hipsparseSpSM_solve(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr, nullptr);
  status_t = cusparseSpSM_solve(handle_t, opA, opB, alpha, spmatA, dnmatB, dnmatC, dataType, spSMAlg_t, spSMDescr);
#endif
#endif

  return 0;
}
