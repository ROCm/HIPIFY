// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipsolver.h"
#include "cusolverDn.h"
#include "cusolverRf.h"
#include "cusolverSp.h"

int main() {
  printf("19. cuSOLVER API to hipSOLVER API synthetic test\n");

  int ldt = 0;
  int ldxf = 0;
  int m = 0;
  int n = 0;
  int nnzA = 0;
  int nnzL = 0;
  int nnzM = 0;
  int nnzU = 0;
  int k = 0;
  int il = 0;
  int iu = 0;
  int imeig = 0;
  int irank = 0;
  int nrhs = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int ldu = 0;
  int ldv = 0;
  int ldvt = 0;
  int Lwork = 0;
  int devIpiv = 0;
  int devInfo = 0;
  int info = 0;
  int id_info = 0;
  int infoArray = 0;
  int batchSize = 0;
  int imax_sweeps = 0;
  int isort_eig = 0;
  int iexecuted_sweeps = 0;
  int iecon = 0;
  int icsrRowPtr = 0;
  int icsrColInd = 0;
  int ih_csrRowPtrA = 0;
  int ih_csrColIndA = 0;
  int ih_csrRowPtrL = 0;
  int *pih_csrRowPtrL = nullptr;
  int ih_csrColIndL = 0;
  int *pih_csrColIndL = nullptr;
  int ih_csrRowPtrU = 0;
  int *pih_csrRowPtrU = nullptr;
  int ih_csrColIndU = 0;
  int *pih_csrColIndU = nullptr;
  int ih_nnzL = 0;
  int ih_nnzU = 0;
  int ih_P = 0;
  int ih_Q = 0;
  int iposition = 0;
  int ireorder = 0;
  int isingularity = 0;
  float fA = 0.f;
  float fd_A = 0.f;
  float fB = 0.f;
  float fC = 0.f;
  float fD = 0.f;
  float fE = 0.f;
  float fS = 0.f;
  float fd_S = 0.f;
  float fU = 0.f;
  float fd_U = 0.f;
  float fvl = 0.f;
  float fvu = 0.f;
  float fV = 0.f;
  float fd_V = 0.f;
  float fVT = 0.f;
  float fX = 0.f;
  float fW = 0.f;
  float fTAU = 0.f;
  float fTAUQ = 0.f;
  float fTAUP = 0.f;
  float fcsrVal = 0.f;
  float ftol = 0.f;
  double dA = 0.f;
  double dd_A = 0.f;
  double dB = 0.f;
  double dC = 0.f;
  double dD = 0.f;
  double dE = 0.f;
  double dS = 0.f;
  double dd_S = 0.f;
  double dU = 0.f;
  double dd_U = 0.f;
  double dvl = 0.f;
  double dvu = 0.f;
  double dV = 0.f;
  double dd_V = 0.f;
  double dVT = 0.f;
  double dX = 0.f;
  double dXF = 0.f;
  double dW = 0.f;
  double dTAU = 0.f;
  double dTAUQ = 0.f;
  double dTAUP = 0.f;
  double dtolerance = 0.f;
  double dresidual = 0.f;
  double dcsrVal = 0.f;
  double dtol = 0.f;
  float fWorkspace = 0.f;
  float fd_Workspace = 0.f;
  float frWork = 0.f;
  double dWorkspace = 0.f;
  double dd_Workspace = 0.f;
  double drWork = 0.f;
  double dh_R_nrmF = 0.f;
  double dzero = 0.f;
  double dboost = 0.f;
  double dh_csrValA = 0.f;
  double dh_csrValL = 0.f;
  double *pdh_csrValL = nullptr;
  double dh_csrValU = 0.f;
  double *pdh_csrValU = nullptr;
  double dTemp = 0.f;
  void *Workspace = nullptr;
  int *piMp = nullptr;
  int *piMi = nullptr;
  double *pdMx = nullptr;
  size_t lwork_bytes = 0;

  signed char jobu = 0;
  signed char jobvt = 0;

  long long int strideA = 0;
  long long int strideS = 0;
  long long int strideU = 0;
  long long int strideV = 0;

  float** fAarray = 0;
  float** fBarray = 0;
  double** dAarray = 0;
  double** dBarray = 0;

  // CHECK: hipDoubleComplex dComplexA, dComplexd_A, dComplexB, dComplexC, dComplexD, dComplexE, dComplexS, dComplexU, dComplexd_U, dComplexV, dComplexd_V, dComplexVT, dComplexX, dComplexWorkspace, dComplexd_Workspace, dComplexrWork, dComplexTAU, dComplexTAUQ, dComplexTAUP;
  cuDoubleComplex dComplexA, dComplexd_A, dComplexB, dComplexC, dComplexD, dComplexE, dComplexS, dComplexU, dComplexd_U, dComplexV, dComplexd_V, dComplexVT, dComplexX, dComplexWorkspace, dComplexd_Workspace, dComplexrWork, dComplexTAU, dComplexTAUQ, dComplexTAUP;

  // CHECK: hipComplex complexA, complexd_A, complexB, complexC, complexD, complexE, complexS, complexU, complexd_U, complexV, complexd_V, complexVT, complexX, complexWorkspace, complexd_Workspace, complexrWork, complexTAU, complexTAUQ, complexTAUP;
  cuComplex complexA, complexd_A, complexB, complexC, complexD, complexE, complexS, complexU, complexd_U, complexV, complexd_V, complexVT, complexX, complexWorkspace, complexd_Workspace, complexrWork, complexTAU, complexTAUQ, complexTAUP;

  // CHECK: hipDoubleComplex** dcomplexAarray = 0;
  // CHECK-NEXT: hipDoubleComplex** dcomplexBarray = 0;
  cuDoubleComplex** dcomplexAarray = 0;
  cuDoubleComplex** dcomplexBarray = 0;

  // CHECK: hipComplex** complexAarray = 0;
  // CHECK-NEXT: hipComplex** complexBarray = 0;
  cuComplex** complexAarray = 0;
  cuComplex** complexBarray = 0;

  // CHECK: hipsolverHandle_t handle;
  cusolverDnHandle_t handle;

  // CHECK: hipblasFillMode_t fillMode;
  cublasFillMode_t fillMode;

  // CHECK: hipsolverStatus_t status;
  // CHECK-NEXT: hipsolverStatus_t STATUS_SUCCESS = HIPSOLVER_STATUS_SUCCESS;
  // CHECK-NEXT: hipsolverStatus_t STATUS_NOT_INITIALIZED = HIPSOLVER_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hipsolverStatus_t STATUS_ALLOC_FAILED = HIPSOLVER_STATUS_ALLOC_FAILED;
  // CHECK-NEXT: hipsolverStatus_t STATUS_INVALID_VALUE = HIPSOLVER_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hipsolverStatus_t STATUS_ARCH_MISMATCH = HIPSOLVER_STATUS_ARCH_MISMATCH;
  // CHECK-NEXT: hipsolverStatus_t STATUS_MAPPING_ERROR = HIPSOLVER_STATUS_MAPPING_ERROR;
  // CHECK-NEXT: hipsolverStatus_t STATUS_EXECUTION_FAILED = HIPSOLVER_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT: hipsolverStatus_t STATUS_INTERNAL_ERROR = HIPSOLVER_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hipsolverStatus_t STATUS_MATRIX_TYPE_NOT_SUPPORTED = HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  // CHECK-NEXT: hipsolverStatus_t STATUS_NOT_SUPPORTED = HIPSOLVER_STATUS_NOT_SUPPORTED;
  // CHECK-NEXT: hipsolverStatus_t STATUS_ZERO_PIVOT = HIPSOLVER_STATUS_ZERO_PIVOT;
  cusolverStatus_t status;
  cusolverStatus_t STATUS_SUCCESS = CUSOLVER_STATUS_SUCCESS;
  cusolverStatus_t STATUS_NOT_INITIALIZED = CUSOLVER_STATUS_NOT_INITIALIZED;
  cusolverStatus_t STATUS_ALLOC_FAILED = CUSOLVER_STATUS_ALLOC_FAILED;
  cusolverStatus_t STATUS_INVALID_VALUE = CUSOLVER_STATUS_INVALID_VALUE;
  cusolverStatus_t STATUS_ARCH_MISMATCH = CUSOLVER_STATUS_ARCH_MISMATCH;
  cusolverStatus_t STATUS_MAPPING_ERROR = CUSOLVER_STATUS_MAPPING_ERROR;
  cusolverStatus_t STATUS_EXECUTION_FAILED = CUSOLVER_STATUS_EXECUTION_FAILED;
  cusolverStatus_t STATUS_INTERNAL_ERROR = CUSOLVER_STATUS_INTERNAL_ERROR;
  cusolverStatus_t STATUS_MATRIX_TYPE_NOT_SUPPORTED = CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  cusolverStatus_t STATUS_NOT_SUPPORTED = CUSOLVER_STATUS_NOT_SUPPORTED;
  cusolverStatus_t STATUS_ZERO_PIVOT = CUSOLVER_STATUS_ZERO_PIVOT;

  // CHECK: hipblasOperation_t blasOperation;
  cublasOperation_t blasOperation;

  // CHECK: hipblasSideMode_t blasSideMode;
  cublasSideMode_t blasSideMode;

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  // CHECK: hipsolverRfResetValuesFastMode_t RfResetValuesFastMode;
  // CHECK-NEXT: hipsolverRfResetValuesFastMode_t RESET_VALUES_FAST_MODE_OFF = HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
  // CHECK-NEXT: hipsolverRfResetValuesFastMode_t RESET_VALUES_FAST_MODE_ON = HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON;
  cusolverRfResetValuesFastMode_t RfResetValuesFastMode;
  cusolverRfResetValuesFastMode_t RESET_VALUES_FAST_MODE_OFF = CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
  cusolverRfResetValuesFastMode_t RESET_VALUES_FAST_MODE_ON = CUSOLVERRF_RESET_VALUES_FAST_MODE_ON;

  // CHECK: hipsolverRfMatrixFormat_t RfMatrixFormat;
  // CHECK-NEXT: hipsolverRfMatrixFormat_t MATRIX_FORMAT_CSR = HIPSOLVERRF_MATRIX_FORMAT_CSR;
  // CHECK-NEXT: hipsolverRfMatrixFormat_t MATRIX_FORMAT_CSC = HIPSOLVERRF_MATRIX_FORMAT_CSC;
  cusolverRfMatrixFormat_t RfMatrixFormat;
  cusolverRfMatrixFormat_t MATRIX_FORMAT_CSR = CUSOLVERRF_MATRIX_FORMAT_CSR;
  cusolverRfMatrixFormat_t MATRIX_FORMAT_CSC = CUSOLVERRF_MATRIX_FORMAT_CSC;

  // CHECK: hipsolverRfUnitDiagonal_t RfUnitDiagonal;
  // CHECK-NEXT: hipsolverRfUnitDiagonal_t UNIT_DIAGONAL_STORED_L = HIPSOLVERRF_UNIT_DIAGONAL_STORED_L;
  // CHECK-NEXT: hipsolverRfUnitDiagonal_t UNIT_DIAGONAL_STORED_U = HIPSOLVERRF_UNIT_DIAGONAL_STORED_U;
  // CHECK-NEXT: hipsolverRfUnitDiagonal_t UNIT_DIAGONAL_ASSUMED_L = HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
  // CHECK-NEXT: hipsolverRfUnitDiagonal_t UNIT_DIAGONAL_ASSUMED_U = HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U;
  cusolverRfUnitDiagonal_t RfUnitDiagonal;
  cusolverRfUnitDiagonal_t UNIT_DIAGONAL_STORED_L = CUSOLVERRF_UNIT_DIAGONAL_STORED_L;
  cusolverRfUnitDiagonal_t UNIT_DIAGONAL_STORED_U = CUSOLVERRF_UNIT_DIAGONAL_STORED_U;
  cusolverRfUnitDiagonal_t UNIT_DIAGONAL_ASSUMED_L = CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
  cusolverRfUnitDiagonal_t UNIT_DIAGONAL_ASSUMED_U = CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U;

  // CHECK: hipsolverRfFactorization_t RfFactorization;
  // CHECK-NEXT: hipsolverRfFactorization_t FACTORIZATION_ALG0 = HIPSOLVERRF_FACTORIZATION_ALG0;
  // CHECK-NEXT: hipsolverRfFactorization_t FACTORIZATION_ALG1 = HIPSOLVERRF_FACTORIZATION_ALG1;
  // CHECK-NEXT: hipsolverRfFactorization_t FACTORIZATION_ALG2 = HIPSOLVERRF_FACTORIZATION_ALG2;
  cusolverRfFactorization_t RfFactorization;
  cusolverRfFactorization_t FACTORIZATION_ALG0 = CUSOLVERRF_FACTORIZATION_ALG0;
  cusolverRfFactorization_t FACTORIZATION_ALG1 = CUSOLVERRF_FACTORIZATION_ALG1;
  cusolverRfFactorization_t FACTORIZATION_ALG2 = CUSOLVERRF_FACTORIZATION_ALG2;

  // CHECK: hipsolverRfTriangularSolve_t RfTriangularSolve;
  // CHECK-NEXT: hipsolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG1 = HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1;
  // CHECK-NEXT: hipsolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG2 = HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2;
  // CHECK-NEXT: hipsolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG3 = HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3;
  cusolverRfTriangularSolve_t RfTriangularSolve;
  cusolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG1 = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;
  cusolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG2 = CUSOLVERRF_TRIANGULAR_SOLVE_ALG2;
  cusolverRfTriangularSolve_t TRIANGULAR_SOLVE_ALG3 = CUSOLVERRF_TRIANGULAR_SOLVE_ALG3;

  // CHECK: hipsolverRfNumericBoostReport_t RfNumericBoostReport;
  // CHECK-NEXT: hipsolverRfNumericBoostReport_t NUMERIC_BOOST_NOT_USED = HIPSOLVERRF_NUMERIC_BOOST_NOT_USED;
  // CHECK-NEXT: hipsolverRfNumericBoostReport_t NUMERIC_BOOST_USED = HIPSOLVERRF_NUMERIC_BOOST_USED;
  cusolverRfNumericBoostReport_t RfNumericBoostReport;
  cusolverRfNumericBoostReport_t NUMERIC_BOOST_NOT_USED = CUSOLVERRF_NUMERIC_BOOST_NOT_USED;
  cusolverRfNumericBoostReport_t NUMERIC_BOOST_USED = CUSOLVERRF_NUMERIC_BOOST_USED;

  // CHECK: hipsolverRfHandle_t RfHandle;
  cusolverRfHandle_t RfHandle;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle);
  // CHECK: status = hipsolverDnCreate(&handle);
  status = cusolverDnCreate(&handle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle);
  // CHECK: status = hipsolverDnDestroy(handle);
  status = cusolverDnDestroy(handle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle, int m, int n, float* A, int lda, float* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnSgetrf(handle, m, n, &fA, lda, &fWorkspace, &devIpiv, &devInfo);
  status = cusolverDnSgetrf(handle, m, n, &fA, lda, &fWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle, int m, int n,double* A, int lda, double* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnDgetrf(handle, m, n, &dA, lda, &dWorkspace, &devIpiv, &devInfo);
  status = cusolverDnDgetrf(handle, m, n, &dA, lda, &dWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n, cuComplex * A, int lda, cuComplex * Workspace, int * devIpiv, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, hipFloatComplex* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnCgetrf(handle, m, n, &complexA, lda, &complexWorkspace, &devIpiv, &devInfo);
  status = cusolverDnCgetrf(handle, m, n, &complexA, lda, &complexWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * Workspace, int * devIpiv, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnZgetrf(handle, m, n, &dComplexA, lda, &dComplexWorkspace, &devIpiv, &devInfo);
  status = cusolverDnZgetrf(handle, m, n, &dComplexA, lda, &dComplexWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSgetrf_bufferSize(handle, m, n, &fA, lda, &Lwork);
  status = cusolverDnSgetrf_bufferSize(handle, m, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDgetrf_bufferSize(handle, m, n, &dA, lda, &Lwork);
  status = cusolverDnDgetrf_bufferSize(handle, m, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnCgetrf_bufferSize(handle, m, n, &complexA, lda, &Lwork);
  status = cusolverDnCgetrf_bufferSize(handle, m, n, &complexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnZgetrf_bufferSize(handle, m, n, &dComplexA, lda, &Lwork);
  status = cusolverDnZgetrf_bufferSize(handle, m, n, &dComplexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnSgetrs(handle, blasOperation, n, nrhs , &fA, lda, &devIpiv, &fB, ldb, &devInfo);
  status = cusolverDnSgetrs(handle, blasOperation, n, nrhs , &fA, lda, &devIpiv, &fB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnDgetrs(handle, blasOperation, n, nrhs , &dA, lda, &devIpiv, &dB, ldb, &devInfo);
  status = cusolverDnDgetrs(handle, blasOperation, n, nrhs , &dA, lda, &devIpiv, &dB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex * A, int lda, const int * devIpiv, cuComplex * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipFloatComplex* A, int lda, const int* devIpiv, hipFloatComplex* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnCgetrs(handle, blasOperation, n, nrhs , &complexA, lda, &devIpiv, &complexB, ldb, &devInfo);
  status = cusolverDnCgetrs(handle, blasOperation, n, nrhs , &complexA, lda, &devIpiv, &complexB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *A, int lda, const int * devIpiv, cuDoubleComplex * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrs(hipsolverHandle_t handle, hipblasOperation_t trans, int n, int nrhs, const hipDoubleComplex* A, int lda, const int* devIpiv, hipDoubleComplex* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnZgetrs(handle, blasOperation, n, nrhs , &dComplexA, lda, &devIpiv, &dComplexB, ldb, &devInfo);
  status = cusolverDnZgetrs(handle, blasOperation, n, nrhs , &dComplexA, lda, &devIpiv, &dComplexB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId);
  // CHECK: status = hipsolverSetStream(handle, stream_t);
  status = cusolverDnSetStream(handle, stream_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId);
  // CHECK: status = hipsolverGetStream(handle, &stream_t);
  status = cusolverDnGetStream(handle, &stream_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSpotrf_bufferSize(handle, fillMode, n, &fA, lda, &Lwork);
  status = cusolverDnSpotrf_bufferSize(handle, fillMode, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDpotrf_bufferSize(handle, fillMode, n, &dA, lda, &Lwork);
  status = cusolverDnDpotrf_bufferSize(handle, fillMode, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnCpotrf_bufferSize(handle, fillMode, n, &complexA, lda, &Lwork);
  status = cusolverDnCpotrf_bufferSize(handle, fillMode, n, &complexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnZpotrf_bufferSize(handle, fillMode, n, &dComplexA, lda, &Lwork);
  status = cusolverDnZpotrf_bufferSize(handle, fillMode, n, &dComplexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, float * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSpotrf(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSpotrf(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, double * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDpotrf(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDpotrf(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCpotrf(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCpotrf(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZpotrf(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZpotrf(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float * A, int lda, float * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const float* A, int lda, float* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnSpotrs(handle, fillMode, n, nrhs, &fA, lda, &fB, ldb, &devInfo);
  status = cusolverDnSpotrs(handle, fillMode, n, nrhs, &fA, lda, &fB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double * A, int lda, double * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const double* A, int lda, double* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnDpotrs(handle, fillMode, n, nrhs, &dA, lda, &dB, ldb, &devInfo);
  status = cusolverDnDpotrs(handle, fillMode, n, nrhs, &dA, lda, &dB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex * A, int lda, cuComplex * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnCpotrs(handle, fillMode, n, nrhs, &complexA, lda, &complexB, ldb, &devInfo);
  status = cusolverDnCpotrs(handle, fillMode, n, nrhs, &complexA, lda, &complexB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex *A, int lda, cuDoubleComplex * B, int ldb, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrs(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, const hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnZpotrs(handle, fillMode, n, nrhs, &dComplexA, lda, &dComplexB, ldb, &devInfo);
  status = cusolverDnZpotrs(handle, fillMode, n, nrhs, &dComplexA, lda, &dComplexB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSgeqrf_bufferSize(handle, m, n, &fA, lda, &Lwork);
  status = cusolverDnSgeqrf_bufferSize(handle, m, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDgeqrf_bufferSize(handle, m, n, &dA, lda, &Lwork);
  status = cusolverDnDgeqrf_bufferSize(handle, m, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnCgeqrf_bufferSize(handle, m, n, &complexA, lda, &Lwork);
  status = cusolverDnCgeqrf_bufferSize(handle, m, n, &complexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnZgeqrf_bufferSize(handle, m, n, &dComplexA, lda, &Lwork);
  status = cusolverDnZgeqrf_bufferSize(handle, m, n, &dComplexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n, float * A, int lda, float * TAU, float * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf(hipsolverHandle_t handle, int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSgeqrf(handle, m, n, &fA, lda, &fTAU, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSgeqrf(handle, m, n, &fA, lda, &fTAU, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n, double * A, int lda, double * TAU, double * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf(hipsolverHandle_t handle, int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDgeqrf(handle, m, n, &dA, lda, &dTAU, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDgeqrf(handle, m, n, &dA, lda, &dTAU, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgeqrf(cusolverDnHandle_t handle, int m, int n, cuComplex * A, int lda, cuComplex * TAU, cuComplex * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf(hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, hipFloatComplex* tau, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCgeqrf(handle, m, n, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCgeqrf(handle, m, n, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgeqrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * TAU, cuDoubleComplex * Workspace, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf(hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* tau, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZgeqrf(handle, m, n, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZgeqrf(handle, m, n, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf_bufferSize(cusolverDnHandle_t handle, int n, float * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrf_bufferSize(hipsolverHandle_t handle, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSsytrf_bufferSize(handle, n, &fA, lda, &Lwork);
  status = cusolverDnSsytrf_bufferSize(handle, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf_bufferSize(cusolverDnHandle_t handle, int n, double * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDsytrf_bufferSize(handle, n, &dA, lda, &Lwork);
  status = cusolverDnDsytrf_bufferSize(handle, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf_bufferSize(hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnCsytrf_bufferSize(handle, n, &complexA, lda, &Lwork);
  status = cusolverDnCsytrf_bufferSize(handle, n, &complexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf_bufferSize(cusolverDnHandle_t handle, int n, cuDoubleComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf_bufferSize(hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnZsytrf_bufferSize(handle, n, &dComplexA, lda, &Lwork);
  status = cusolverDnZsytrf_bufferSize(handle, n, &dComplexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, int * ipiv, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, int* ipiv, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsytrf(handle, fillMode, n, &fA, lda, &devIpiv, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSsytrf(handle, fillMode, n, &fA, lda, &devIpiv, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, int * ipiv, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, int* ipiv, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsytrf(handle, fillMode, n, &dA, lda, &devIpiv, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDsytrf(handle, fillMode, n, &dA, lda, &devIpiv, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, int * ipiv, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, int* ipiv, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCsytrf(handle, fillMode, n, &complexA, lda, &devIpiv, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCsytrf(handle, fillMode, n, &complexA, lda, &devIpiv, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZsytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, int * ipiv, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, int* ipiv, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZsytrf(handle, fillMode, n, &dComplexA, lda, &devIpiv, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZsytrf(handle, fillMode, n, &dComplexA, lda, &devIpiv, &dComplexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnSgebrd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnSgebrd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnDgebrd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnDgebrd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnCgebrd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnCgebrd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnZgebrd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnZgebrd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgebrd(cusolverDnHandle_t handle, int m, int n, float * A, int lda, float * D, float * E, float * TAUQ, float * TAUP, float * Work, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd(hipsolverHandle_t handle, int m, int n, float* A, int lda, float* D, float* E, float* tauq, float* taup, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSgebrd(handle, m, n, &fA, lda, &fD, &fE, &fTAUQ, &fTAUP, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSgebrd(handle, m, n, &fA, lda, &fD, &fE, &fTAUQ, &fTAUP, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgebrd(cusolverDnHandle_t handle, int m, int n, double * A, int lda, double * D, double * E, double * TAUQ, double * TAUP, double * Work, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd(hipsolverHandle_t handle, int m, int n, double* A, int lda, double* D, double* E, double* tauq, double* taup, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDgebrd(handle, m, n, &dA, lda, &dD, &dE, &dTAUQ, &dTAUP, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDgebrd(handle, m, n, &dA, lda, &dD, &dE, &dTAUQ, &dTAUP, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgebrd(cusolverDnHandle_t handle, int m, int n, cuComplex * A, int lda, float * D, float * E, cuComplex * TAUQ, cuComplex * TAUP, cuComplex * Work, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd(hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, float* D, float* E, hipFloatComplex* tauq, hipFloatComplex* taup, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCgebrd(handle, m, n, &complexA, lda, &fD, &fE, &complexTAUQ, &complexTAUP, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCgebrd(handle, m, n, &complexA, lda, &fD, &fE, &complexTAUQ, &complexTAUP, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgebrd(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex * A, int lda, double * D, double * E, cuDoubleComplex * TAUQ, cuDoubleComplex * TAUP, cuDoubleComplex * Work, int Lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd(hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, double* D, double* E, hipDoubleComplex* tauq, hipDoubleComplex* taup, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZgebrd(handle, m, n, &dComplexA, lda, &dD, &dE, &dComplexTAUQ, &dComplexTAUP, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZgebrd(handle, m, n, &dComplexA, lda, &dD, &dE, &dComplexTAUQ, &dComplexTAUP, &dComplexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, float * d, float * e, float * tau, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, float* D, float* E, float* tau, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsytrd(handle, fillMode, n, &fA, lda, &fD, &fE, &fTAU, &fWorkspace, Lwork, &info);
  status = cusolverDnSsytrd(handle, fillMode, n, &fA, lda, &fD, &fE, &fTAU, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A,int lda, double * d, double * e, double * tau, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, double* D, double* E, double* tau, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsytrd(handle, fillMode, n, &dA, lda, &dD, &dE, &dTAU, &dWorkspace, Lwork, &info);
  status = cusolverDnDsytrd(handle, fillMode, n, &dA, lda, &dD, &dE, &dTAU, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnSgesvd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnSgesvd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnDgesvd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnDgesvd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnCgesvd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnCgesvd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork);
  // CHECK: status = hipsolverDnZgesvd_bufferSize(handle, m, n, &Lwork);
  status = cusolverDnZgesvd_bufferSize(handle, m, n, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * VT, int ldvt, float * work, int lwork, float * rwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, float* rwork, int* devInfo);
  // CHECK: status = hipsolverDnSgesvd(handle, jobu, jobvt, m, n, &fA, lda, &fS, &fU, ldu, &fVT, ldvt, &fWorkspace, Lwork, &frWork, &info);
  status = cusolverDnSgesvd(handle, jobu, jobvt, m, n, &fA, lda, &fS, &fU, ldu, &fVT, ldvt, &fWorkspace, Lwork, &frWork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * VT, int ldvt, double * work, int lwork, double * rwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, double* rwork, int* devInfo);
  // CHECK: status = hipsolverDnDgesvd(handle, jobu, jobvt, m, n, &dA, lda, &dS, &dU, ldu, &dVT, ldvt, &dWorkspace, Lwork, &drWork, &info);
  status = cusolverDnDgesvd(handle, jobu, jobvt, m, n, &dA, lda, &dS, &dU, ldu, &dVT, ldvt, &dWorkspace, Lwork, &drWork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex * A, int lda, float * S, cuComplex * U, int ldu, cuComplex * VT, int ldvt, cuComplex * work, int lwork, float * rwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipFloatComplex* A, int lda, float* S, hipFloatComplex* U, int ldu, hipFloatComplex* V, int ldv, hipFloatComplex* work, int lwork, float* rwork, int* devInfo);
  // CHECK: status = hipsolverDnCgesvd(handle, jobu, jobvt, m, n, &complexA, lda, &fS, &complexU, ldu, &complexVT, ldvt, &complexWorkspace, Lwork, &frWork, &info);
  status = cusolverDnCgesvd(handle, jobu, jobvt, m, n, &complexA, lda, &fS, &complexU, ldu, &complexVT, ldvt, &complexWorkspace, Lwork, &frWork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex * A, int lda, double * S, cuDoubleComplex * U, int ldu, cuDoubleComplex * VT, int ldvt, cuDoubleComplex * work, int lwork, double * rwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd(hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, hipDoubleComplex* A, int lda, double* S, hipDoubleComplex* U, int ldu, hipDoubleComplex* V, int ldv, hipDoubleComplex* work, int lwork, double* rwork, int* devInfo);
  // CHECK: status = hipsolverDnZgesvd(handle, jobu, jobvt, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexVT, ldvt, &dComplexWorkspace, Lwork, &drWork, &info);
  status = cusolverDnZgesvd(handle, jobu, jobvt, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexVT, ldvt, &dComplexWorkspace, Lwork, &drWork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfCreate(cusolverRfHandle_t* handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle);
  // CHECK: status = hipsolverRfCreate(&RfHandle);
  status = cusolverRfCreate(&RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfDestroy(cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfDestroy(RfHandle);
  status = cusolverRfDestroy(RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfGetMatrixFormat(cusolverRfHandle_t handle, cusolverRfMatrixFormat_t* format, cusolverRfUnitDiagonal_t* diag);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t handle, hipsolverRfMatrixFormat_t* format, hipsolverRfUnitDiagonal_t* diag);
  // CHECK: status = hipsolverRfGetMatrixFormat(RfHandle, &RfMatrixFormat, &RfUnitDiagonal);
  status = cusolverRfGetMatrixFormat(RfHandle, &RfMatrixFormat, &RfUnitDiagonal);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetMatrixFormat(cusolverRfHandle_t handle, cusolverRfMatrixFormat_t format, cusolverRfUnitDiagonal_t diag);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t handle, hipsolverRfMatrixFormat_t format, hipsolverRfUnitDiagonal_t diag);
  // CHECK: status = hipsolverRfSetMatrixFormat(RfHandle, RfMatrixFormat, RfUnitDiagonal);
  status = cusolverRfSetMatrixFormat(RfHandle, RfMatrixFormat, RfUnitDiagonal);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetNumericProperties(cusolverRfHandle_t handle, double zero, double boost);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle, double effective_zero, double boost_val);
  // CHECK: status = hipsolverRfSetNumericProperties(RfHandle, dzero, dboost);
  status = cusolverRfSetNumericProperties(RfHandle, dzero, dboost);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericProperties(cusolverRfHandle_t handle, double* zero, double* boost);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double* zero, double* boost);
  // CHECK: status = hipsolverRfGetNumericProperties(RfHandle, &dzero, &dboost);
  status = cusolverRfGetNumericProperties(RfHandle, &dzero, &dboost);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfGetNumericBoostReport(cusolverRfHandle_t handle, cusolverRfNumericBoostReport_t* report);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t handle, hipsolverRfNumericBoostReport_t* report);
  // CHECK: status = hipsolverRfGetNumericBoostReport(RfHandle, &RfNumericBoostReport);
  status = cusolverRfGetNumericBoostReport(RfHandle, &RfNumericBoostReport);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetAlgs(cusolverRfHandle_t handle, cusolverRfFactorization_t factAlg, cusolverRfTriangularSolve_t solveAlg);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t handle, hipsolverRfFactorization_t fact_alg, hipsolverRfTriangularSolve_t solve_alg);
  // CHECK: status = hipsolverRfSetAlgs(RfHandle, RfFactorization, RfTriangularSolve);
  status = cusolverRfSetAlgs(RfHandle, RfFactorization, RfTriangularSolve);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t* fastMode);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t* fastMode);
  // CHECK: status = hipsolverRfGetResetValuesFastMode(RfHandle, &RfResetValuesFastMode);
  status = cusolverRfGetResetValuesFastMode(RfHandle, &RfResetValuesFastMode);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfGetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t* fastMode);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t* fastMode);
  // CHECK: status = hipsolverRfGetResetValuesFastMode(RfHandle, &RfResetValuesFastMode);
  status = cusolverRfGetResetValuesFastMode(RfHandle, &RfResetValuesFastMode);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetResetValuesFastMode(cusolverRfHandle_t handle, cusolverRfResetValuesFastMode_t fastMode);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t handle, hipsolverRfResetValuesFastMode_t fastMode);
  // CHECK: status = hipsolverRfSetResetValuesFastMode(RfHandle, RfResetValuesFastMode);
  status = cusolverRfSetResetValuesFastMode(RfHandle, RfResetValuesFastMode);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetupHost(int n, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA, int nnzL, int* h_csrRowPtrL, int* h_csrColIndL, double* h_csrValL, int nnzU, int* h_csrRowPtrU, int* h_csrColIndU, double* h_csrValU, int* h_P, int* h_Q, cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetupHost(int n, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA, int nnzL, int* h_csrRowPtrL, int* h_csrColIndL, double* h_csrValL, int nnzU, int* h_csrRowPtrU, int* h_csrColIndU, double* h_csrValU, int* h_P, int* h_Q, hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfSetupHost(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);
  status = cusolverRfSetupHost(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSetupDevice(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSetupDevice(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int nnzL, int* csrRowPtrL, int* csrColIndL, double* csrValL, int nnzU, int* csrRowPtrU, int* csrColIndU, double* csrValU, int* P, int* Q, hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfSetupDevice(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);
  status = cusolverRfSetupDevice(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfResetValues(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int* P, int* Q, cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfResetValues(int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA, int* P, int* Q, hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfResetValues(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, &ih_P, &ih_Q, RfHandle);
  status = cusolverRfResetValues(n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, &dh_csrValA, &ih_P, &ih_Q, RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfAnalyze(cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfAnalyze(RfHandle);
  status = cusolverRfAnalyze(RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfRefactor(cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfRefactor(RfHandle);
  status = cusolverRfRefactor(RfHandle);

  // CUDA: cusolverRfAccessBundledFactorsDevice(cusolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(hipsolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx);
  // CHECK: status = hipsolverRfAccessBundledFactorsDevice(RfHandle, &nnzM, &piMp, &piMi, &pdMx);
  status = cusolverRfAccessBundledFactorsDevice(RfHandle, &nnzM, &piMp, &piMi, &pdMx);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(hipsolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx);
  // CHECK: status = hipsolverRfExtractBundledFactorsHost(RfHandle, &nnzM, &piMp, &piMi, &pdMx);
  status = cusolverRfExtractBundledFactorsHost(RfHandle, &nnzM, &piMp, &piMi, &pdMx);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfExtractSplitFactorsHost(cusolverRfHandle_t handle, int* h_nnzL, int** h_csrRowPtrL, int** h_csrColIndL, double** h_csrValL, int* h_nnzU, int** h_csrRowPtrU, int** h_csrColIndU, double** h_csrValU);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle, int* h_nnzL, int** h_Lp, int** h_Li, double** h_Lx, int* h_nnzU, int** h_Up, int** h_Ui, double** h_Ux);
  // CHECK: status = hipsolverRfExtractSplitFactorsHost(RfHandle, &ih_nnzL, &pih_csrRowPtrL, &pih_csrColIndL, &pdh_csrValL, &ih_nnzU, &pih_csrRowPtrU, &pih_csrColIndU, &pdh_csrValU);
  status = cusolverRfExtractSplitFactorsHost(RfHandle, &ih_nnzL, &pih_csrRowPtrL, &pih_csrColIndL, &pdh_csrValL, &ih_nnzU, &pih_csrRowPtrU, &pih_csrColIndU, &pdh_csrValU);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfSolve( cusolverRfHandle_t handle, int* P, int* Q, int nrhs, double* Temp, int ldt, double* XF, int ldxf);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle, int* P, int* Q, int nrhs, double* Temp, int ldt, double* XF, int ldxf);
  // CHECK: status = hipsolverRfSolve(RfHandle, &ih_P, &ih_Q, nrhs, &dTemp, ldt, &dXF, ldxf);
  status = cusolverRfSolve(RfHandle, &ih_P, &ih_Q, nrhs, &dTemp, ldt, &dXF, ldxf);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchSetupHost( int batchSize, int n, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA_array[], int nnzL, int* h_csrRowPtrL, int* h_csrColIndL, double* h_csrValL, int nnzU, int* h_csrRowPtrU, int* h_csrColIndU, double* h_csrValU, int* h_P, int* h_Q, cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchSetupHost(int batchSize, int n, int nnzA, int* h_csrRowPtrA, int* h_csrColIndA, double* h_csrValA_array[], int nnzL, int* h_csrRowPtrL, int* h_csrColIndL, double* h_csrValL, int nnzU, int* h_csrRowPtrU, int* h_csrColIndU, double* h_csrValU, int* h_P, int* h_Q, hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfBatchSetupHost(batchSize, n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, dAarray, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);
  status = cusolverRfBatchSetupHost(batchSize, n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, dAarray, nnzL, &ih_csrRowPtrL, &ih_csrColIndL, &dh_csrValL, nnzU, &ih_csrRowPtrU, &ih_csrColIndU, &dh_csrValU, &ih_P, &ih_Q, RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchResetValues(int batchSize, int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA_array[], int* P, int* Q, cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchResetValues(int batchSize, int n, int nnzA, int* csrRowPtrA, int* csrColIndA, double* csrValA_array[], int* P, int* Q, hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfBatchResetValues(batchSize, n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, dAarray, &ih_P, &ih_Q, RfHandle);
  status = cusolverRfBatchResetValues(batchSize, n, nnzA, &ih_csrRowPtrA, &ih_csrColIndA, dAarray, &ih_P, &ih_Q, RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchAnalyze(cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfBatchAnalyze(RfHandle);
  status = cusolverRfBatchAnalyze(RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchRefactor(cusolverRfHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle);
  // CHECK: status = hipsolverRfBatchRefactor(RfHandle);
  status = cusolverRfBatchRefactor(RfHandle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchSolve(cusolverRfHandle_t handle, int* P, int* Q, int nrhs, double* Temp, int ldt, double* XF_array[], int ldxf);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle, int* P, int* Q, int nrhs, double* Temp, int ldt, double* XF_array[], int ldxf);
  // CHECK: status = hipsolverRfBatchSolve(RfHandle, &ih_P, &ih_Q, nrhs, &dTemp, ldt, dAarray, ldxf);
  status = cusolverRfBatchSolve(RfHandle, &ih_P, &ih_Q, nrhs, &dTemp, ldt, dAarray, ldxf);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverRfBatchZeroPivot(cusolverRfHandle_t handle, int* position);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle, int* position);
  // CHECK: status = hipsolverRfBatchZeroPivot(RfHandle, &iposition);
  status = cusolverRfBatchZeroPivot(RfHandle, &iposition);

  // CHECK: hipsolverSpHandle_t SpHandle_t;
  cusolverSpHandle_t SpHandle_t;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpCreate(cusolverSpHandle_t *handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle);
  // CHECK: status = hipsolverSpCreate(&SpHandle_t);
  status = cusolverSpCreate(&SpHandle_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpDestroy(cusolverSpHandle_t handle);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle);
  // CHECK: status = hipsolverSpDestroy(SpHandle_t);
  status = cusolverSpDestroy(SpHandle_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId);
  // CHECK: status = hipsolverSpSetStream(SpHandle_t, stream_t);
  status = cusolverSpSetStream(SpHandle_t, stream_t);

  // CHECK: hipsparseMatDescr_t MatDescr_t;
  cusparseMatDescr_t MatDescr_t;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvchol(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float * csrVal, const int * csrRowPtr, const int * csrColInd, const float * b, float tol, int reorder, float *x, int * singularity);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrVal, const int* csrRowPtr, const int* csrColInd, const float* b, float tolerance, int reorder, float* x, int* singularity);
  // CHECK: status = hipsolverSpScsrlsvchol(SpHandle_t, m, nnzA, MatDescr_t, &fcsrVal, &icsrRowPtr, &icsrColInd, &fB, ftol, ireorder, &fX, &isingularity);
  status = cusolverSpScsrlsvchol(SpHandle_t, m, nnzA, MatDescr_t, &fcsrVal, &icsrRowPtr, &icsrColInd, &fB, ftol, ireorder, &fX, &isingularity);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double * csrVal, const int * csrRowPtr, const int * csrColInd, const double * b, double tol, int reorder, double *x, int * singularity);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrVal, const int* csrRowPtr, const int* csrColInd, const double* b, double tolerance, int reorder, double* x, int* singularity);
  // CHECK: status = hipsolverSpDcsrlsvchol(SpHandle_t, m, nnzA, MatDescr_t, &dcsrVal, &icsrRowPtr, &icsrColInd, &dB, dtol, ireorder, &dX, &isingularity);
  status = cusolverSpDcsrlsvchol(SpHandle_t, m, nnzA, MatDescr_t, &dcsrVal, &icsrRowPtr, &icsrColInd, &dB, dtol, ireorder, &dX, &isingularity);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpScsrlsvcholHost(cusolverSpHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float * csrVal, const int * csrRowPtr, const int * csrColInd, const float * b, float tol, int reorder, float * x, int * singularity);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const float* csrVal, const int* csrRowPtr, const int* csrColInd, const float* b, float tolerance, int reorder, float* x, int* singularity);
  // CHECK: status = hipsolverSpScsrlsvcholHost(SpHandle_t, m, nnzA, MatDescr_t, &fcsrVal, &icsrRowPtr, &icsrColInd, &fB, ftol, ireorder, &fX, &isingularity);
  status = cusolverSpScsrlsvcholHost(SpHandle_t, m, nnzA, MatDescr_t, &fcsrVal, &icsrRowPtr, &icsrColInd, &fB, ftol, ireorder, &fX, &isingularity);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverSpDcsrlsvcholHost(cusolverSpHandle_t handle, int m,int nnz, const cusparseMatDescr_t descrA, const double * csrVal, const int * csrRowPtr, const int * csrColInd, const double * b, double tol, int reorder, double * x, int * singularity);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t handle, int n, int nnzA, const hipsparseMatDescr_t descrA, const double* csrVal, const int* csrRowPtr, const int* csrColInd, const double* b, double tolerance, int reorder, double* x, int* singularity);
  // CHECK: status = hipsolverSpDcsrlsvcholHost(SpHandle_t, m, nnzA, MatDescr_t, &dcsrVal, &icsrRowPtr, &icsrColInd, &dB, dtol, ireorder, &dX, &isingularity);
  status = cusolverSpDcsrlsvcholHost(SpHandle_t, m, nnzA, MatDescr_t, &dcsrVal, &icsrRowPtr, &icsrColInd, &dB, dtol, ireorder, &dX, &isingularity);

#if CUDA_VERSION >= 8000
  // CHECK: hipsolverEigType_t eigType;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_1 = HIPSOLVER_EIG_TYPE_1;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_2 = HIPSOLVER_EIG_TYPE_2;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_3 = HIPSOLVER_EIG_TYPE_3;
  cusolverEigType_t eigType;
  cusolverEigType_t EIG_TYPE_1 = CUSOLVER_EIG_TYPE_1;
  cusolverEigType_t EIG_TYPE_2 = CUSOLVER_EIG_TYPE_2;
  cusolverEigType_t EIG_TYPE_3 = CUSOLVER_EIG_TYPE_3;

  // CHECK: hipsolverEigMode_t eigMode, jobz;
  // CHECK-NEXT: hipsolverEigMode_t SOLVER_EIG_MODE_NOVECTOR = HIPSOLVER_EIG_MODE_NOVECTOR;
  // CHECK-NEXT: hipsolverEigMode_t SOLVER_EIG_MODE_VECTOR = HIPSOLVER_EIG_MODE_VECTOR;
  cusolverEigMode_t eigMode, jobz;
  cusolverEigMode_t SOLVER_EIG_MODE_NOVECTOR = CUSOLVER_EIG_MODE_NOVECTOR;
  cusolverEigMode_t SOLVER_EIG_MODE_VECTOR = CUSOLVER_EIG_MODE_VECTOR;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const float * A, int lda, const float * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork);
  // CHECK: status = hipsolverDnSorgqr_bufferSize(handle, m, n, k, &fA, lda, &fTAU, &Lwork);
  status = cusolverDnSorgqr_bufferSize(handle, m, n, k, &fA, lda, &fTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const double * A, int lda, const double * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork);
  // CHECK: status = hipsolverDnDorgqr_bufferSize(handle, m, n, k, &dA, lda, &dTAU, &Lwork);
  status = cusolverDnDorgqr_bufferSize(handle, m, n, k, &dA, lda, &dTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuComplex * A, int lda, const cuComplex * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnCungqr_bufferSize(handle, m, n, k, &complexA, lda, &complexTAU, &Lwork);
  status = cusolverDnCungqr_bufferSize(handle, m, n, k, &complexA, lda, &complexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungqr_bufferSize(cusolverDnHandle_t handle, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverHandle_t handle, int m, int n, int k, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnZungqr_bufferSize(handle, m, n, k, &dComplexA, lda, &dComplexTAU, &Lwork);
  status = cusolverDnZungqr_bufferSize(handle, m, n, k, &dComplexA, lda, &dComplexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgqr(cusolverDnHandle_t handle, int m, int n, int k, float * A, int lda, const float * tau, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr(hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSorgqr(handle, m, n, k, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);
  status = cusolverDnSorgqr(handle, m, n, k, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgqr(cusolverDnHandle_t handle, int m, int n, int k, double * A, int lda, const double * tau, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr(hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDorgqr(handle, m, n, k, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);
  status = cusolverDnDorgqr(handle, m, n, k, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungqr(cusolverDnHandle_t handle, int m, int n, int k, cuComplex * A, int lda, const cuComplex * tau, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr(hipsolverHandle_t handle, int m, int n, int k, hipFloatComplex* A, int lda, const hipFloatComplex* tau, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCungqr(handle, m, n, k, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);
  status = cusolverDnCungqr(handle, m, n, k, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungqr(cusolverDnHandle_t handle, int m, int n, int k, cuDoubleComplex * A, int lda, const cuDoubleComplex *tau, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr(hipsolverHandle_t handle, int m, int n, int k, hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZungqr(handle, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZungqr(handle, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float * A, int lda, const float * tau, const float * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnSormqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &fA, lda, &fTAU, &fC, ldc, &Lwork);
  status = cusolverDnSormqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &fA, lda, &fTAU, &fC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double * A, int lda, const double * tau, const double * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnDormqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &dA, lda, &dTAU, &dC, ldc, &Lwork);
  status = cusolverDnDormqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &dA, lda, &dTAU, &dC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex * A, int lda, const cuComplex * tau, const cuComplex * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, const hipFloatComplex* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnCunmqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &complexA, lda, &complexTAU, &complexC, ldc, &Lwork);
  status = cusolverDnCunmqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &complexA, lda, &complexTAU, &complexC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, const hipDoubleComplex* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnZunmqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &Lwork);
  status = cusolverDnZunmqr_bufferSize(handle, blasSideMode, blasOperation, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float * A, int lda, const float * tau, float * C, int ldc, float * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, float* C, int ldc, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSormqr(handle, blasSideMode, blasOperation, m, n, k, &fA, lda, &fTAU, &fC, ldc, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSormqr(handle, blasSideMode, blasOperation, m, n, k, &fA, lda, &fTAU, &fC, ldc, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double * A, int lda, const double * tau, double * C, int ldc, double * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const double* A, int lda, const double* tau, double* C, int ldc, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDormqr(handle, blasSideMode, blasOperation, m, n, k, &dA, lda, &dTAU, &dC, ldc, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDormqr(handle, blasSideMode, blasOperation, m, n, k, &dA, lda, &dTAU, &dC, ldc, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex * A, int lda, const cuComplex * tau, cuComplex * C, int ldc, cuComplex * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, hipFloatComplex* C, int ldc, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCunmqr(handle, blasSideMode, blasOperation, m, n, k, &complexA, lda, &complexTAU, &complexC, ldc, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCunmqr(handle, blasSideMode, blasOperation, m, n, k, &complexA, lda, &complexTAU, &complexC, ldc, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZunmqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex * C, int ldc, cuDoubleComplex * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, hipDoubleComplex* C, int ldc, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZunmqr(handle, blasSideMode, blasOperation, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZunmqr(handle, blasSideMode, blasOperation, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &dComplexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const float * A, int lda, const float * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const float* A, int lda, const float* tau, int* lwork);
  // CHECK: status = hipsolverDnSorgbr_bufferSize(handle, blasSideMode, m, n, k, &fA, lda, &fTAU, &Lwork);
  status = cusolverDnSorgbr_bufferSize(handle, blasSideMode, m, n, k, &fA, lda, &fTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const double * A, int lda, const double * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const double* A, int lda, const double* tau, int* lwork);
  // CHECK: status = hipsolverDnDorgbr_bufferSize(handle, blasSideMode, m, n, k, &dA, lda, &dTAU, &Lwork);
  status = cusolverDnDorgbr_bufferSize(handle, blasSideMode, m, n, k, &dA, lda, &dTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuComplex * A, int lda, const cuComplex * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnCungbr_bufferSize(handle, blasSideMode, m, n, k, &complexA, lda, &complexTAU, &Lwork);
  status = cusolverDnCungbr_bufferSize(handle, blasSideMode, m, n, k, &complexA, lda, &complexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungbr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnZungbr_bufferSize(handle, blasSideMode, m, n, k, &dComplexA, lda, &dComplexTAU, &Lwork);
  status = cusolverDnZungbr_bufferSize(handle, blasSideMode, m, n, k, &dComplexA, lda, &dComplexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, float * A, int lda, const float * tau, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSorgbr(handle, blasSideMode, m, n, k, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);
  status = cusolverDnSorgbr(handle, blasSideMode, m, n, k, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, double * A, int lda, const double * tau, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDorgbr(handle, blasSideMode, m, n, k, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);
  status = cusolverDnDorgbr(handle, blasSideMode, m, n, k, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuComplex * A, int lda, const cuComplex * tau, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, hipFloatComplex* A, int lda, const hipFloatComplex* tau, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCungbr(handle, blasSideMode, m, n, k, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);
  status = cusolverDnCungbr(handle, blasSideMode, m, n, k, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungbr(cusolverDnHandle_t handle, cublasSideMode_t side, int m, int n, int k, cuDoubleComplex * A, int lda, const cuDoubleComplex *tau, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr(hipsolverHandle_t handle, hipblasSideMode_t side, int m, int n, int k, hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZungbr(handle, blasSideMode, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZungbr(handle, blasSideMode, m, n, k, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float * A, int lda, const float * d, const float * e, const float * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* D, const float* E, const float* tau, int* lwork);
  // CHECK: status = hipsolverDnSsytrd_bufferSize(handle, fillMode, n, &fA, lda, &fD, &fE, &fTAU, &Lwork);
  status = cusolverDnSsytrd_bufferSize(handle, fillMode, n, &fA, lda, &fD, &fE, &fTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double * A, int lda, const double * d, const double * e, const double * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* D, const double* E, const double* tau, int* lwork);
  // CHECK: status = hipsolverDnDsytrd_bufferSize(handle, fillMode, n, &dA, lda, &dD, &dE, &dTAU, &Lwork);
  status = cusolverDnDsytrd_bufferSize(handle, fillMode, n, &dA, lda, &dD, &dE, &dTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const float * d, const float * e, const cuComplex * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const float* D, const float* E, const hipFloatComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnChetrd_bufferSize(handle, fillMode, n, &complexA, lda, &fD, &fE, &complexTAU, &Lwork);
  status = cusolverDnChetrd_bufferSize(handle, fillMode, n, &complexA, lda, &fD, &fE, &complexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double * d, const double * e, const cuDoubleComplex *tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const double* D, const double* E, const hipDoubleComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnZhetrd_bufferSize(handle, fillMode, n, &dComplexA, lda, &dD, &dE, &dComplexTAU, &Lwork);
  status = cusolverDnZhetrd_bufferSize(handle, fillMode, n, &dComplexA, lda, &dD, &dE, &dComplexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, float * d, float * e, cuComplex * tau, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, float* D, float* E, hipFloatComplex* tau, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnChetrd(handle, fillMode, n, &complexA, lda, &fD, &fE, &complexTAU, &complexWorkspace, Lwork, &info);
  status = cusolverDnChetrd(handle, fillMode, n, &complexA, lda, &fD, &fE, &complexTAU, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhetrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, double * d, double * e, cuDoubleComplex * tau, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, double* D, double* E, hipDoubleComplex* tau, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZhetrd(handle, fillMode, n, &dComplexA, lda, &dD, &dE, &dComplexTAU, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZhetrd(handle, fillMode, n, &dComplexA, lda, &dD, &dE, &dComplexTAU, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const float * A, int lda, const float * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* tau, int* lwork);
  // CHECK: status = hipsolverDnSorgtr_bufferSize(handle, fillMode, n, &fA, lda, &fTAU, &Lwork);
  status = cusolverDnSorgtr_bufferSize(handle, fillMode, n, &fA, lda, &fTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const double * A, int lda, const double * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* tau, int* lwork);
  // CHECK: status = hipsolverDnDorgtr_bufferSize(handle, fillMode, n, &dA, lda, &dTAU, &Lwork);
  status = cusolverDnDorgtr_bufferSize(handle, fillMode, n, &dA, lda, &dTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const cuComplex * tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnCungtr_bufferSize(handle, fillMode, n, &complexA, lda, &complexTAU, &Lwork);
  status = cusolverDnCungtr_bufferSize(handle, fillMode, n, &complexA, lda, &complexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungtr_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, int* lwork);
  // CHECK: status = hipsolverDnZungtr_bufferSize(handle, fillMode, n, &dComplexA, lda, &dComplexTAU, &Lwork);
  status = cusolverDnZungtr_bufferSize(handle, fillMode, n, &dComplexA, lda, &dComplexTAU, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, const float * tau, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, const float* tau, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSorgtr(handle, fillMode, n, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);
  status = cusolverDnSorgtr(handle, fillMode, n, &fA, lda, &fTAU, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDorgtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, const double * tau, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, const double* tau, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDorgtr(handle, fillMode, n, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);
  status = cusolverDnDorgtr(handle, fillMode, n, &dA, lda, &dTAU, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, const cuComplex * tau, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, const hipFloatComplex* tau, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCungtr(handle, fillMode, n, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);
  status = cusolverDnCungtr(handle, fillMode, n, &complexA, lda, &complexTAU, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZungtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, const cuDoubleComplex *tau, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZungtr(handle, fillMode, n, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZungtr(handle, fillMode, n, &dComplexA, lda, &dComplexTAU, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const float * A, int lda, const float * tau, const float * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnSormtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &fA, lda, &fTAU, &fC, ldc, &Lwork);
  status = cusolverDnSormtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &fA, lda, &fTAU, &fC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const double * A, int lda, const double * tau, const double * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const double* A, int lda, const double* tau, const double* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnDormtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &dA, lda, &dTAU, &dC, ldc, &Lwork);
  status = cusolverDnDormtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &dA, lda, &dTAU, &dC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuComplex * A, int lda, const cuComplex * tau, const cuComplex * C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const hipFloatComplex* A, int lda, const hipFloatComplex* tau, const hipFloatComplex* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnCunmtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &complexA, lda, &complexTAU, &complexC, ldc, &Lwork);
  status = cusolverDnCunmtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &complexA, lda, &complexTAU, &complexC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, const cuDoubleComplex *C, int ldc, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr_bufferSize(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, const hipDoubleComplex* A, int lda, const hipDoubleComplex* tau, const hipDoubleComplex* C, int ldc, int* lwork);
  // CHECK: status = hipsolverDnZunmtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &Lwork);
  status = cusolverDnZunmtr_bufferSize(handle, blasSideMode, fillMode, blasOperation, m, n, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, float * A, int lda, float * tau, float * C, int ldc, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, float* A, int lda, float* tau, float* C, int ldc, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSormtr(handle, blasSideMode, fillMode, blasOperation, m, n, &fA, lda, &fTAU, &fC, ldc, &fWorkspace, Lwork, &info);
  status = cusolverDnSormtr(handle, blasSideMode, fillMode, blasOperation, m, n, &fA, lda, &fTAU, &fC, ldc, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, double * A, int lda, double * tau, double * C, int ldc, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, double* A, int lda, double* tau, double* C, int ldc, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDormtr(handle, blasSideMode, fillMode, blasOperation, m, n, &dA, lda, &dTAU, &dC, ldc, &dWorkspace, Lwork, &info);
  status = cusolverDnDormtr(handle, blasSideMode, fillMode, blasOperation, m, n, &dA, lda, &dTAU, &dC, ldc, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuComplex * A, int lda, cuComplex * tau, cuComplex * C, int ldc, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, hipFloatComplex* A, int lda, hipFloatComplex* tau, hipFloatComplex* C, int ldc, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCunmtr(handle, blasSideMode, fillMode, blasOperation, m, n, &complexA, lda, &complexTAU, &complexC, ldc, &complexWorkspace, Lwork, &info);
  status = cusolverDnCunmtr(handle, blasSideMode, fillMode, blasOperation, m, n, &complexA, lda, &complexTAU, &complexC, ldc, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZunmtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * tau, cuDoubleComplex * C, int ldc, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr(hipsolverHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t trans, int m, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* tau, hipDoubleComplex* C, int ldc, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZunmtr(handle, blasSideMode, fillMode, blasOperation, m, n, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZunmtr(handle, blasSideMode, fillMode, blasOperation, m, n, &dComplexA, lda, &dComplexTAU, &dComplexC, ldc, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork);
  // CHECK: status = hipsolverDnSsyevd_bufferSize(handle, eigMode, fillMode, n, &fA, lda, &fW, &Lwork);
  status = cusolverDnSsyevd_bufferSize(handle, eigMode, fillMode, n, &fA, lda, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork);
  // CHECK: status = hipsolverDnDsyevd_bufferSize(handle, eigMode, fillMode, n, &dA, lda, &dW, &Lwork);
  status = cusolverDnDsyevd_bufferSize(handle, eigMode, fillMode, n, &dA, lda, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const float* W, int* lwork);
  // CHECK: status = hipsolverDnCheevd_bufferSize(handle, eigMode, fillMode, n, &complexA, lda, &fW, &Lwork);
  status = cusolverDnCheevd_bufferSize(handle, eigMode, fillMode, n, &complexA, lda, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const double* W, int* lwork);
  // CHECK: status = hipsolverDnZheevd_bufferSize(handle, eigMode, fillMode, n, &dComplexA, lda, &dW, &Lwork);
  status = cusolverDnZheevd_bufferSize(handle, eigMode, fillMode, n, &dComplexA, lda, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsyevd(handle, eigMode, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info);
  status = cusolverDnSsyevd(handle, eigMode, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsyevd(handle, eigMode, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info);
  status = cusolverDnDsyevd(handle, eigMode, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex * A, int lda, float * W, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, float* W, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCheevd(handle, eigMode, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info);
  status = cusolverDnCheevd(handle, eigMode, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, double * W, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, double* W, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZheevd(handle, eigMode, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZheevd(handle, eigMode, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork);
  // CHECK: status = hipsolverDnSsygvd_bufferSize(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &Lwork);
  status = cusolverDnSsygvd_bufferSize(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork);
  // CHECK: status = hipsolverDnDsygvd_bufferSize(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &Lwork);
  status = cusolverDnDsygvd_bufferSize(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const cuComplex * B, int ldb, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const hipFloatComplex* B, int ldb, const float* W, int* lwork);
  // CHECK: status = hipsolverDnChegvd_bufferSize(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &Lwork);
  status = cusolverDnChegvd_bufferSize(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const hipDoubleComplex* B, int ldb, const double* W, int* lwork);
  // CHECK: status = hipsolverDnZhegvd_bufferSize(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &Lwork);
  status = cusolverDnZhegvd_bufferSize(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsygvd(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &fWorkspace, Lwork, &info);
  status = cusolverDnSsygvd(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsygvd(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &dWorkspace, Lwork, &info);
  status = cusolverDnDsygvd(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * B, int ldb, float * W, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, float* W, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnChegvd(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &complexWorkspace, Lwork, &info);
  status = cusolverDnChegvd(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvd(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * B, int ldb, double * W, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, double* W, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZhegvd(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZhegvd(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &dComplexWorkspace, Lwork, &info);
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipsolverSyevjInfo_t syevj_info;
  syevjInfo_t syevj_info;

  // CHECK: hipsolverGesvdjInfo_t gesvdj_info;
  gesvdjInfo_t gesvdj_info;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(syevjInfo_t *info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateSyevjInfo(hipsolverSyevjInfo_t* info);
  // CHECK: status = hipsolverDnCreateSyevjInfo(&syevj_info);
  status = cusolverDnCreateSyevjInfo(&syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(syevjInfo_t info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroySyevjInfo(hipsolverSyevjInfo_t info);
  // CHECK: status = hipsolverDnDestroySyevjInfo(syevj_info);
  status = cusolverDnDestroySyevjInfo(syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetTolerance(hipsolverSyevjInfo_t info, double tolerance);
  // CHECK: status = hipsolverDnXsyevjSetTolerance(syevj_info, dtolerance);
  status = cusolverDnXsyevjSetTolerance(syevj_info, dtolerance);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info, int max_sweeps);
  // CHECK: status = hipsolverDnXsyevjSetMaxSweeps(syevj_info, imax_sweeps);
  status = cusolverDnXsyevjSetMaxSweeps(syevj_info, imax_sweeps);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetSortEig(hipsolverSyevjInfo_t info, int sort_eig);
  // CHECK: status = hipsolverDnXsyevjSetSortEig(syevj_info, isort_eig);
  status = cusolverDnXsyevjSetSortEig(syevj_info, isort_eig);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info, double * residual);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetResidual(hipsolverDnHandle_t handle, hipsolverSyevjInfo_t info, double* residual);
  // CHECK: status = hipsolverDnXsyevjGetResidual(handle, syevj_info, &dresidual);
  status = cusolverDnXsyevjGetResidual(handle, syevj_info, &dresidual);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info, int * executed_sweeps);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetSweeps(hipsolverDnHandle_t handle, hipsolverSyevjInfo_t info, int* executed_sweeps);
  // CHECK: status = hipsolverDnXsyevjGetSweeps(handle, syevj_info, &iexecuted_sweeps);
  status = cusolverDnXsyevjGetSweeps(handle, syevj_info, &iexecuted_sweeps);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnSsyevjBatched_bufferSize(handle, jobz, fillMode, n, &fA, lda, &fW, &Lwork, syevj_info, batchSize);
  status = cusolverDnSsyevjBatched_bufferSize(handle, jobz, fillMode, n, &fA, lda, &fW, &Lwork, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnDsyevjBatched_bufferSize(handle, jobz, fillMode, n, &dA, lda, &dW, &Lwork, syevj_info, batchSize);
  status = cusolverDnDsyevjBatched_bufferSize(handle, jobz, fillMode, n, &dA, lda, &dW, &Lwork, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const float * W, int * lwork, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const float* W, int* lwork, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnCheevjBatched_bufferSize(handle, jobz, fillMode, n, &complexA, lda, &fW, &Lwork, syevj_info, batchSize);
  status = cusolverDnCheevjBatched_bufferSize(handle, jobz, fillMode, n, &complexA, lda, &fW, &Lwork, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double * W, int * lwork, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const double* W, int* lwork, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnZheevjBatched_bufferSize(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &Lwork, syevj_info, batchSize);
  status = cusolverDnZheevjBatched_bufferSize(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &Lwork, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * info, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnSsyevjBatched(handle, jobz, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info, syevj_info, batchSize);
  status = cusolverDnSsyevjBatched(handle, jobz, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * info, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnDsyevjBatched(handle, jobz, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info, syevj_info, batchSize);
  status = cusolverDnDsyevjBatched(handle, jobz, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex * A, int lda, float * W, cuComplex * work, int lwork, int * info, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, float* W, hipFloatComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnCheevjBatched(handle, jobz, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info, syevj_info, batchSize);
  status = cusolverDnCheevjBatched(handle, jobz, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, double * W, cuDoubleComplex * work, int lwork, int * info, syevjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, double* W, hipDoubleComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnZheevjBatched(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info, syevj_info, batchSize);
  status = cusolverDnZheevjBatched(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info, syevj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float * A, int lda, const float * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnSsyevj_bufferSize(handle, jobz, fillMode, n, &fA, lda, &fW, &Lwork, syevj_info);
  status = cusolverDnSsyevj_bufferSize(handle, jobz, fillMode, n, &fA, lda, &fW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double * A, int lda, const double * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnDsyevj_bufferSize(handle, jobz, fillMode, n, &dA, lda, &dW, &Lwork, syevj_info);
  status = cusolverDnDsyevj_bufferSize(handle, jobz, fillMode, n, &dA, lda, &dW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const float * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const float* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnCheevj_bufferSize(handle, jobz, fillMode, n, &complexA, lda, &fW, &Lwork, syevj_info);
  status = cusolverDnCheevj_bufferSize(handle, jobz, fillMode, n, &complexA, lda, &fW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const double * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const double* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnZheevj_bufferSize(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &Lwork, syevj_info);
  status = cusolverDnZheevj_bufferSize(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float * A, int lda, float * W, float * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float* A, int lda, float* W, float* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnSsyevj(handle, jobz, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnSsyevj(handle, jobz, fillMode, n, &fA, lda, &fW, &fWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double * A, int lda, double * W, double * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double* A, int lda, double* W, double* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);B
  // CHECK: status = hipsolverDnDsyevj(handle, jobz, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnDsyevj(handle, jobz, fillMode, n, &dA, lda, &dW, &dWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex * A, int lda, float * W, cuComplex * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, float* W, hipFloatComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnCheevj(handle, jobz, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnCheevj(handle, jobz, fillMode, n, &complexA, lda, &fW, &complexWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, double * W, cuDoubleComplex * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, double* W, hipDoubleComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnZheevj(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnZheevj(handle, jobz, fillMode, n, &dComplexA, lda, &dW, &dComplexWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, const float * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, const float* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnSsygvj_bufferSize(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &Lwork, syevj_info);
  status = cusolverDnSsygvj_bufferSize(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, const double * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, const double* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnDsygvj_bufferSize(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &Lwork, syevj_info);
  status = cusolverDnDsygvj_bufferSize(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const cuComplex * B, int ldb, const float * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const hipFloatComplex* B, int ldb, const float* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnChegvj_bufferSize(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &Lwork, syevj_info);
  status = cusolverDnChegvj_bufferSize(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, const double * W, int * lwork, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const hipDoubleComplex* B, int ldb, const double* W, int* lwork, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnZhegvj_bufferSize(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &Lwork, syevj_info);
  status = cusolverDnZhegvj_bufferSize(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &Lwork, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float * W, float * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float* W, float* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnSsygvj(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &fWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnSsygvj(handle, eigType, jobz, fillMode, n, &fA, lda, &fB, ldb, &fW, &fWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double * W, double * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double* W, double* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnDsygvj(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &dWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnDsygvj(handle, eigType, jobz, fillMode, n, &dA, lda, &dB, ldb, &dW, &dWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * B, int ldb, float * W, cuComplex * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, float* W, hipFloatComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnChegvj(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &complexWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnChegvj(handle, eigType, jobz, fillMode, n, &complexA, lda, &complexB, ldb, &fW, &complexWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvj(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * B, int ldb, double * W, cuDoubleComplex * work, int lwork, int * info, syevjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvj(hipsolverDnHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, double* W, hipDoubleComplex* work, int lwork, int* devInfo, hipsolverSyevjInfo_t params);
  // CHECK: status = hipsolverDnZhegvj(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &dComplexWorkspace, Lwork, &info, syevj_info);
  status = cusolverDnZhegvj(handle, eigType, jobz, fillMode, n, &dComplexA, lda, &dComplexB, ldb, &dW, &dComplexWorkspace, Lwork, &info, syevj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(gesvdjInfo_t *info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateGesvdjInfo(hipsolverGesvdjInfo_t* info);
  // CHECK: status = hipsolverDnCreateGesvdjInfo(&gesvdj_info);
  status = cusolverDnCreateGesvdjInfo(&gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroyGesvdjInfo(hipsolverGesvdjInfo_t info);
  // CHECK: status = hipsolverDnDestroyGesvdjInfo(gesvdj_info);
  status = cusolverDnDestroyGesvdjInfo(gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetTolerance(hipsolverGesvdjInfo_t info, double tolerance);
  // CHECK: status = hipsolverDnXgesvdjSetTolerance(gesvdj_info, dtolerance);
  status = cusolverDnXgesvdjSetTolerance(gesvdj_info, dtolerance);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info, int max_sweeps);
  // CHECK: status = hipsolverDnXgesvdjSetMaxSweeps(gesvdj_info, imax_sweeps);
  status = cusolverDnXgesvdjSetMaxSweeps(gesvdj_info, imax_sweeps);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetSortEig(hipsolverGesvdjInfo_t info, int sort_eig);
  // CHECK: status = hipsolverDnXgesvdjSetSortEig(gesvdj_info, isort_eig);
  status = cusolverDnXgesvdjSetSortEig(gesvdj_info, isort_eig);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info, double * residual);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjGetResidual(hipsolverDnHandle_t handle, hipsolverGesvdjInfo_t info, double* residual);
  // CHECK: status = hipsolverDnXgesvdjGetResidual(handle, gesvdj_info, &dresidual);
  status = cusolverDnXgesvdjGetResidual(handle, gesvdj_info, &dresidual);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info, int * executed_sweeps);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjGetSweeps(hipsolverDnHandle_t handle, hipsolverGesvdjInfo_t info, int* executed_sweeps);
  // CHECK: status = hipsolverDnXgesvdjGetSweeps(handle, gesvdj_info, &iexecuted_sweeps);
  status = cusolverDnXgesvdjGetSweeps(handle, gesvdj_info, &iexecuted_sweeps);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &Lwork, gesvdj_info, batchSize);
  status = cusolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &Lwork, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &Lwork, gesvdj_info, batchSize);
  status = cusolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &Lwork, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuComplex * A, int lda, const float * S, const cuComplex * U, int ldu, const cuComplex * V, int ldv, int * lwork, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipFloatComplex* A, int lda, const float* S, const hipFloatComplex* U, int ldu, const hipFloatComplex* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &Lwork, gesvdj_info, batchSize);
  status = cusolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &Lwork, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const cuDoubleComplex *A, int lda, const double * S, const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv, int * lwork, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdjBatched_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, const hipDoubleComplex* A, int lda, const double* S, const hipDoubleComplex* U, int ldu, const hipDoubleComplex* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &Lwork, gesvdj_info, batchSize);
  status = cusolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &Lwork, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * info, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnSgesvdjBatched(handle, jobz, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &fWorkspace, Lwork, &info, gesvdj_info, batchSize);
  status = cusolverDnSgesvdjBatched(handle, jobz, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &fWorkspace, Lwork, &info, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * info, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnDgesvdjBatched(handle, jobz, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &dWorkspace, Lwork, &info, gesvdj_info, batchSize);
  status = cusolverDnDgesvdjBatched(handle, jobz, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &dWorkspace, Lwork, &info, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuComplex * A, int lda, float * S, cuComplex * U, int ldu, cuComplex * V, int ldv, cuComplex * work, int lwork, int * info, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipFloatComplex* A, int lda, float* S, hipFloatComplex* U, int ldu, hipFloatComplex* V, int ldv, hipFloatComplex* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnCgesvdjBatched(handle, jobz, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &complexWorkspace, Lwork, &info, gesvdj_info, batchSize);
  status = cusolverDnCgesvdjBatched(handle, jobz, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &complexWorkspace, Lwork, &info, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, cuDoubleComplex * A, int lda, double * S, cuDoubleComplex * U, int ldu, cuDoubleComplex * V, int ldv, cuDoubleComplex * work, int lwork, int * info, gesvdjInfo_t params, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdjBatched(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int m, int n, hipDoubleComplex* A, int lda, double* S, hipDoubleComplex* U, int ldu, hipDoubleComplex* V, int ldv, hipDoubleComplex* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params, int batch_count);
  // CHECK: status = hipsolverDnZgesvdjBatched(handle, jobz, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &dComplexWorkspace, Lwork, &info, gesvdj_info, batchSize);
  status = cusolverDnZgesvdjBatched(handle, jobz, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &dComplexWorkspace, Lwork, &info, gesvdj_info, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const float * A, int lda, const float * S, const float * U, int ldu, const float * V, int ldv, int * lwork, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const float* A, int lda, const float* S, const float* U, int ldu, const float* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnSgesvdj_bufferSize(handle, jobz, iecon, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &Lwork, gesvdj_info);
  status = cusolverDnSgesvdj_bufferSize(handle, jobz, iecon, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &Lwork, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const double * A, int lda, const double * S, const double * U, int ldu, const double * V, int ldv, int * lwork, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const double* A, int lda, const double* S, const double* U, int ldu, const double* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnDgesvdj_bufferSize(handle, jobz, iecon, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &Lwork, gesvdj_info);
  status = cusolverDnDgesvdj_bufferSize(handle, jobz, iecon, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &Lwork, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuComplex * A, int lda, const float * S, const cuComplex * U, int ldu, const cuComplex * V, int ldv, int * lwork, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipFloatComplex* A, int lda, const float* S, const hipFloatComplex* U, int ldu, const hipFloatComplex* V, int ldv, int* lwork, hipsolverGesvdjInfo_t  params);
  // CHECK: status = hipsolverDnCgesvdj_bufferSize(handle, jobz, iecon, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &Lwork, gesvdj_info);
  status = cusolverDnCgesvdj_bufferSize(handle, jobz, iecon, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &Lwork, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, const cuDoubleComplex *A, int lda, const double * S, const cuDoubleComplex *U, int ldu, const cuDoubleComplex *V, int ldv, int * lwork, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdj_bufferSize(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, const hipDoubleComplex* A, int lda, const double* S, const hipDoubleComplex* U, int ldu, const hipDoubleComplex* V, int ldv, int* lwork, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnZgesvdj_bufferSize(handle, jobz, iecon, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &Lwork, gesvdj_info);
  status = cusolverDnZgesvdj_bufferSize(handle, jobz, iecon, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &Lwork, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, float * A, int lda, float * S, float * U, int ldu, float * V, int ldv, float * work, int lwork, int * info, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, float* A, int lda, float* S, float* U, int ldu, float* V, int ldv, float* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnSgesvdj(handle, jobz, iecon, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &fWorkspace, Lwork, &info, gesvdj_info);
  status = cusolverDnSgesvdj(handle, jobz, iecon, m, n, &fA, lda, &fS, &fU, ldu, &fV, ldv, &fWorkspace, Lwork, &info, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, double * A, int lda, double * S, double * U, int ldu, double * V, int ldv, double * work, int lwork, int * info, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, double* A, int lda, double* S, double* U, int ldu, double* V, int ldv, double* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnDgesvdj(handle, jobz, iecon, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &dWorkspace, Lwork, &info, gesvdj_info);
  status = cusolverDnDgesvdj(handle, jobz, iecon, m, n, &dA, lda, &dS, &dU, ldu, &dV, ldv, &dWorkspace, Lwork, &info, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuComplex * A, int lda, float * S, cuComplex * U, int ldu, cuComplex * V, int ldv, cuComplex * work, int lwork, int * info, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipFloatComplex* A, int lda, float* S, hipFloatComplex* U, int ldu, hipFloatComplex* V, int ldv, hipFloatComplex* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnCgesvdj(handle, jobz, iecon, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &complexWorkspace, Lwork, &info, gesvdj_info);
  status = cusolverDnCgesvdj(handle, jobz, iecon, m, n, &complexA, lda, &fS, &complexU, ldu, &complexV, ldv, &complexWorkspace, Lwork, &info, gesvdj_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, cuDoubleComplex * A, int lda, double * S, cuDoubleComplex * U, int ldu, cuDoubleComplex * V, int ldv, cuDoubleComplex * work, int lwork, int * info, gesvdjInfo_t params);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdj(hipsolverDnHandle_t handle, hipsolverEigMode_t jobz, int econ, int m, int n, hipDoubleComplex* A, int lda, double* S, hipDoubleComplex* U, int ldu, hipDoubleComplex* V, int ldv, hipDoubleComplex* work, int lwork, int* devInfo, hipsolverGesvdjInfo_t params);
  // CHECK: status = hipsolverDnZgesvdj(handle, jobz, iecon, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &dComplexWorkspace, Lwork, &info, gesvdj_info);
  status = cusolverDnZgesvdj(handle, jobz, iecon, m, n, &dComplexA, lda, &dS, &dComplexU, ldu, &dComplexV, ldv, &dComplexWorkspace, Lwork, &info, gesvdj_info);
#endif

#if CUDA_VERSION >= 9010
  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * Aarray[], int lda, int * infoArray, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A[], int lda, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnSpotrfBatched(handle, fillMode, n, fAarray, lda, &infoArray, batchSize);
  status = cusolverDnSpotrfBatched(handle, fillMode, n, fAarray, lda, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * Aarray[], int lda, int * infoArray, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A[], int lda, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnDpotrfBatched(handle, fillMode, n, dAarray, lda, &infoArray, batchSize);
  status = cusolverDnDpotrfBatched(handle, fillMode, n, dAarray, lda, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * Aarray[], int lda, int * infoArray, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A[], int lda, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnCpotrfBatched(handle, fillMode, n, complexAarray, lda, &infoArray, batchSize);
  status = cusolverDnCpotrfBatched(handle, fillMode, n, complexAarray, lda, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * Aarray[], int lda, int * infoArray, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A[], int lda, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnZpotrfBatched(handle, fillMode, n, dcomplexAarray, lda, &infoArray, batchSize);
  status = cusolverDnZpotrfBatched(handle, fillMode, n, dcomplexAarray, lda, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, float * A[], int lda, float * B[], int ldb, int * d_info, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, float* A[], int lda, float* B[], int ldb, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnSpotrsBatched(handle, fillMode, n, nrhs, fAarray, lda, fBarray, ldb, &infoArray, batchSize);
  status = cusolverDnSpotrsBatched(handle, fillMode, n, nrhs, fAarray, lda, fBarray, ldb, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, double * A[], int lda, double * B[], int ldb, int * d_info, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, double* A[], int lda, double* B[], int ldb, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnDpotrsBatched(handle, fillMode, n, nrhs, dAarray, lda, dBarray, ldb, &infoArray, batchSize);
  status = cusolverDnDpotrsBatched(handle, fillMode, n, nrhs, dAarray, lda, dBarray, ldb, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuComplex * A[], int lda, cuComplex * B[], int ldb, int * d_info, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, hipFloatComplex* A[], int lda, hipFloatComplex* B[], int ldb, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnCpotrsBatched(handle, fillMode, n, nrhs, complexAarray, lda, complexBarray, ldb, &infoArray, batchSize);
  status = cusolverDnCpotrsBatched(handle, fillMode, n, nrhs, complexAarray, lda, complexBarray, ldb, &infoArray, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrsBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, cuDoubleComplex * A[], int lda, cuDoubleComplex * B[], int ldb, int * d_info, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, int nrhs, hipDoubleComplex* A[], int lda, hipDoubleComplex* B[], int ldb, int* devInfo, int batch_count);
  // CHECK: status = hipsolverDnZpotrsBatched(handle, fillMode, n, nrhs, dcomplexAarray, lda, dcomplexBarray, ldb, &infoArray, batchSize);
  status = cusolverDnZpotrsBatched(handle, fillMode, n, nrhs, dcomplexAarray, lda, dcomplexBarray, ldb, &infoArray, batchSize);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipsolverEigRange_t eigRange;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_ALL = HIPSOLVER_EIG_RANGE_ALL;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_I = HIPSOLVER_EIG_RANGE_I;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_V = HIPSOLVER_EIG_RANGE_V;
  cusolverEigRange_t eigRange;
  cusolverEigRange_t EIG_RANGE_ALL = CUSOLVER_EIG_RANGE_ALL;
  cusolverEigRange_t EIG_RANGE_I = CUSOLVER_EIG_RANGE_I;
  cusolverEigRange_t EIG_RANGE_V = CUSOLVER_EIG_RANGE_V;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSpotri_bufferSize(handle, fillMode, n, &fA, lda, &Lwork);
  status = cusolverDnSpotri_bufferSize(handle, fillMode, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDpotri_bufferSize(handle, fillMode, n, &dA, lda, &Lwork);
  status = cusolverDnDpotri_bufferSize(handle, fillMode, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnCpotri_bufferSize(handle, fillMode, n, &complexA, lda, &Lwork);
  status = cusolverDnCpotri_bufferSize(handle, fillMode, n, &complexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotri_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnZpotri_bufferSize(handle, fillMode, n, &dComplexA, lda, &Lwork);
  status = cusolverDnZpotri_bufferSize(handle, fillMode, n, &dComplexA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, float * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSpotri(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &infoArray);
  status = cusolverDnSpotri(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &infoArray);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, double * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDpotri(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &infoArray);
  status = cusolverDnDpotri(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &infoArray);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCpotri(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &infoArray);
  status = cusolverDnCpotri(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &infoArray);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * work, int lwork, int * devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZpotri(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &infoArray);
  status = cusolverDnZpotri(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &infoArray);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float * A, int lda, float vl, float vu, int il, int iu, int * meig, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float* A, int lda, float vl, float vu, int il, int iu, int* nev, const float* W, int* lwork);
  // CHECK: status = hipsolverDnSsyevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &fA, lda, fvl, fvu, il, iu, &imeig, &fW, &Lwork);
  status = cusolverDnSsyevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &fA, lda, fvl, fvu, il, iu, &imeig, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double * A, int lda, double vl, double vu, int il, int iu, int * meig, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double* A, int lda, double vl, double vu, int il, int iu, int* nev, const double* W, int* lwork);
  // CHECK: status = hipsolverDnDsyevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &dA, lda, dvl, dvu, il, iu, &imeig, &dW, &Lwork);
  status = cusolverDnDsyevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &dA, lda, dvl, dvu, il, iu, &imeig, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, float vl, float vu, int il, int iu, int * meig, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, float vl, float vu, int il, int iu, int* nev, const float* W, int* lwork);
  // CHECK: status = hipsolverDnCheevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &complexA, lda, fvl, fvu, il, iu, &imeig, &fW, &Lwork);
  status = cusolverDnCheevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &complexA, lda, fvl, fvu, il, iu, &imeig, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, double vl, double vu, int il, int iu, int * meig, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevdx_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* nev, const double* W, int* lwork);
  // CHECK: status = hipsolverDnZheevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &dComplexA, lda, dvl, dvu, il, iu, &imeig, &dW, &Lwork);
  status = cusolverDnZheevdx_bufferSize(handle, jobz, eigRange, fillMode, n, &dComplexA, lda, dvl, dvu, il, iu, &imeig, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float * A, int lda, float vl, float vu, int il, int iu, int * meig, float * W, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float* A, int lda, float vl, float vu, int il, int iu, int* nev, float* W, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsyevdx(handle, jobz, eigRange, fillMode, n, &fA, lda, fvl, fvu, il, iu, &imeig, &fW, &fWorkspace, Lwork, &info);
  status = cusolverDnSsyevdx(handle, jobz, eigRange, fillMode, n, &fA, lda, fvl, fvu, il, iu, &imeig, &fW, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsyevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double * A, int lda, double vl, double vu, int il, int iu, int * meig, double * W, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double* A, int lda, double vl, double vu, int il, int iu, int* nev, double* W, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsyevdx(handle, jobz, eigRange, fillMode, n, &dA, lda, dvl, dvu, il, iu, &imeig, &dW, &dWorkspace, Lwork, &info);
  status = cusolverDnDsyevdx(handle, jobz, eigRange, fillMode, n, &dA, lda, dvl, dvu, il, iu, &imeig, &dW, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex * A, int lda, float vl, float vu, int il, int iu, int * meig, float * W, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, float vl, float vu, int il, int iu, int* nev, float* W, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnCheevdx(handle, jobz, eigRange, fillMode, n, &complexA, lda, fvl, fvu, il, iu, &imeig, &fW, &complexWorkspace, Lwork, &info);
  status = cusolverDnCheevdx(handle, jobz, eigRange, fillMode, n, &complexA, lda, fvl, fvu, il, iu, &imeig, &fW, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZheevdx(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, double vl, double vu, int il, int iu, int * meig, double * W, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevdx(hipsolverHandle_t handle, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, double vl, double vu, int il, int iu, int* nev, double* W, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZheevdx(handle, jobz, eigRange, fillMode, n, &dComplexA, lda, dvl, dvu, il, iu, &imeig, &dW, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZheevdx(handle, jobz, eigRange, fillMode, n, &dComplexA, lda, dvl, dvu, il, iu, &imeig, &dW, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const float * A, int lda, const float * B, int ldb, float vl, float vu, int il, int iu, int * meig, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const float* A, int lda, const float* B, int ldb, float vl, float vu, int il, int iu, int* nev, const float* W, int* lwork);
  // CHECK: status = hipsolverDnSsygvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &fA, lda, &fB, ldb, fvl, fvu, il, iu, &imeig, &fW, &Lwork);
  status = cusolverDnSsygvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &fA, lda, &fB, ldb, fvl, fvu, il, iu, &imeig, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const double * A, int lda, const double * B, int ldb, double vl, double vu, int il, int iu, int * meig, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const double* A, int lda, const double* B, int ldb, double vl, double vu, int il, int iu, int* nev, const double* W, int* lwork);
  // CHECK: status = hipsolverDnDsygvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &dA, lda, &dB, ldb, dvl, dvu, il, iu, &imeig, &dW, &Lwork);
  status = cusolverDnDsygvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &dA, lda, &dB, ldb, dvl, dvu, il, iu, &imeig, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuComplex * A, int lda, const cuComplex * B, int ldb, float vl, float vu, int il, int iu, int * meig, const float * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipFloatComplex* A, int lda, const hipFloatComplex* B, int ldb, float vl, float vu, int il, int iu, int* nev, const float* W, int* lwork);
  // CHECK: status = hipsolverDnChegvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &complexA, lda, &complexB, ldb, fvl, fvu, il, iu, &imeig, &fW, &Lwork);
  status = cusolverDnChegvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &complexA, lda, &complexB, ldb, fvl, fvu, il, iu, &imeig, &fW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx_bufferSize(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, const cuDoubleComplex *A, int lda, const cuDoubleComplex *B, int ldb, double vl, double vu, int il, int iu, int * meig, const double * W, int * lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvdx_bufferSize(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, const hipDoubleComplex* A, int lda, const hipDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* nev, const double* W, int* lwork);
  // CHECK: status = hipsolverDnZhegvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &dComplexA, lda, &dComplexB, ldb, dvl, dvu, il, iu, &imeig, &dW, &Lwork);
  status = cusolverDnZhegvdx_bufferSize(handle, eigType, jobz, eigRange, fillMode, n, &dComplexA, lda, &dComplexB, ldb, dvl, dvu, il, iu, &imeig, &dW, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, float * A, int lda, float * B, int ldb, float vl, float vu, int il, int iu, int * meig, float * W, float * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, float* A, int lda, float* B, int ldb, float vl, float vu, int il, int iu, int* nev, float* W, float* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnSsygvdx(handle, eigType, jobz, eigRange, fillMode, n, &fA, lda, &fB, ldb, fvl, fvu, il, iu, &imeig, &fW, &fWorkspace, Lwork, &info);
  status = cusolverDnSsygvdx(handle, eigType, jobz, eigRange, fillMode, n, &fA, lda, &fB, ldb, fvl, fvu, il, iu, &imeig, &fW, &fWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDsygvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, double * A, int lda, double * B, int ldb, double vl, double vu, int il, int iu, int * meig, double * W, double * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, double* A, int lda, double* B, int ldb, double vl, double vu, int il, int iu, int* nev, double* W, double* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnDsygvdx(handle, eigType, jobz, eigRange, fillMode, n, &dA, lda, &dB, ldb, dvl, dvu, il, iu, &imeig, &dW, &dWorkspace, Lwork, &info);
  status = cusolverDnDsygvdx(handle, eigType, jobz, eigRange, fillMode, n, &dA, lda, &dB, ldb, dvl, dvu, il, iu, &imeig, &dW, &dWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnChegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * B, int ldb, float vl, float vu, int il, int iu, int * meig, float * W, cuComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, float vl, float vu, int il, int iu, int* nev, float* W, hipFloatComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnChegvdx(handle, eigType, jobz, eigRange, fillMode, n, &complexA, lda, &complexB, ldb, fvl, fvu, il, iu, &imeig, &fW, &complexWorkspace, Lwork, &info);
  status = cusolverDnChegvdx(handle, eigType, jobz, eigRange, fillMode, n, &complexA, lda, &complexB, ldb, fvl, fvu, il, iu, &imeig, &fW, &complexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZhegvdx(cusolverDnHandle_t handle, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * B, int ldb, double vl, double vu, int il, int iu, int * meig, double * W, cuDoubleComplex * work, int lwork, int * info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvdx(hipsolverHandle_t handle, hipsolverEigType_t itype, hipsolverEigMode_t jobz, hipsolverEigRange_t range, hipblasFillMode_t uplo, int n, hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, double vl, double vu, int il, int iu, int* nev, double* W, hipDoubleComplex* work, int lwork, int* devInfo);
  // CHECK: status = hipsolverDnZhegvdx(handle, eigType, jobz, eigRange, fillMode, n, &dComplexA, lda, &dComplexB, ldb, dvl, dvu, il, iu, &imeig, &dW, &dComplexWorkspace, Lwork, &info);
  status = cusolverDnZhegvdx(handle, eigType, jobz, eigRange, fillMode, n, &dComplexA, lda, &dComplexB, ldb, dvl, dvu, il, iu, &imeig, &dW, &dComplexWorkspace, Lwork, &info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float * d_A, int lda, long long int strideA, const float * d_S, long long int strideS, const float * d_U, int ldu, long long int strideU, const float * d_V, int ldv, long long int strideV, int * lwork, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const float* A, int lda, long long int strideA, const float* S, long long int strideS, const float* U, int ldu, long long int strideU, const float* V, int ldv, long long int strideV, int* lwork, int batch_count);
  // CHECK: status = hipsolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &fd_A, lda, strideA, &fd_S, strideS, &fd_U, ldu, strideU, &fd_V, ldv, strideV, &Lwork, batchSize);
  status = cusolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &fd_A, lda, strideA, &fd_S, strideS, &fd_U, ldu, strideU, &fd_V, ldv, strideV, &Lwork, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double * d_A, int lda, long long int strideA, const double * d_S, long long int strideS, const double * d_U, int ldu, long long int strideU, const double * d_V, int ldv, long long int strideV, int * lwork, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const double* A, int lda, long long int strideA, const double* S, long long int strideS, const double* U, int ldu, long long int strideU, const double* V, int ldv, long long int strideV, int* lwork, int batch_count);
  // CHECK: status = hipsolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &dd_A, lda, strideA, &dd_S, strideS, &dd_U, ldu, strideU, &dd_V, ldv, strideV, &Lwork, batchSize);
  status = cusolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &dd_A, lda, strideA, &dd_S, strideS, &dd_U, ldu, strideU, &dd_V, ldv, strideV, &Lwork, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex * d_A, int lda, long long int strideA, const float * d_S, long long int strideS, const cuComplex * d_U, int ldu, long long int strideU, const cuComplex * d_V, int ldv, long long int strideV, int * lwork, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipFloatComplex* A, int lda, long long int strideA, const float* S, long long int strideS, const hipFloatComplex* U, int ldu, long long int strideU, const hipFloatComplex* V, int ldv, long long int strideV, int* lwork, int batch_count);
  // CHECK: status = hipsolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &complexd_A, lda, strideA, &fd_S, strideS, &complexd_U, ldu, strideU, &complexd_V, ldv, strideV, &Lwork, batchSize);
  status = cusolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &complexd_A, lda, strideA, &fd_S, strideS, &complexd_U, ldu, strideU, &complexd_V, ldv, strideV, &Lwork, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *d_A, int lda, long long int strideA, const double * d_S, long long int strideS, const cuDoubleComplex *d_U, int ldu, long long int strideU, const cuDoubleComplex *d_V, int ldv, long long int strideV, int * lwork, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdaStridedBatched_bufferSize(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipDoubleComplex* A, int lda, long long int strideA, const double* S, long long int strideS, const hipDoubleComplex* U, int ldu, long long int strideU, const hipDoubleComplex* V, int ldv, long long int strideV, int* lwork, int batch_count);
  // CHECK: status = hipsolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &dComplexd_A, lda, strideA, &dd_S, strideS, &dComplexd_U, ldu, strideU, &dComplexd_V, ldv, strideV, &Lwork, batchSize);
  status = cusolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, irank, m, n, &dComplexd_A, lda, strideA, &dd_S, strideS, &dComplexd_U, ldu, strideU, &dComplexd_V, ldv, strideV, &Lwork, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const float * d_A, int lda, long long int strideA, float * d_S, long long int strideS, float * d_U, int ldu, long long int strideU, float * d_V, int ldv, long long int strideV, float * d_work, int lwork, int * d_info, double * h_R_nrmF, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const float* A, int lda, long long int strideA, float* S, long long int strideS, float* U, int ldu, long long int strideU, float* V, int ldv, long long int strideV, float* work, int lwork, int* devInfo, double* hRnrmF, int batch_count);
  // CHECK: status = hipsolverDnSgesvdaStridedBatched(handle, jobz, irank, m, n, &fd_A, lda, strideA, &fd_S, strideS, &fd_U, ldu, strideU, &fd_V, ldv, strideV, &fd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);
  status = cusolverDnSgesvdaStridedBatched(handle, jobz, irank, m, n, &fd_A, lda, strideA, &fd_S, strideS, &fd_U, ldu, strideU, &fd_V, ldv, strideV, &fd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const double * d_A, int lda, long long int strideA, double * d_S, long long int strideS, double * d_U, int ldu, long long int strideU, double * d_V, int ldv, long long int strideV, double * d_work, int lwork, int * d_info, double * h_R_nrmF, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const double* A, int lda, long long int strideA, double* S, long long int strideS, double* U, int ldu, long long int strideU, double* V, int ldv, long long int strideV, double* work, int lwork, int* devInfo, double* hRnrmF, int batch_count);
  // CHECK: status = hipsolverDnDgesvdaStridedBatched(handle, jobz, irank, m, n, &dd_A, lda, strideA, &dd_S, strideS, &dd_U, ldu, strideU, &dd_V, ldv, strideV, &dd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);
  status = cusolverDnDgesvdaStridedBatched(handle, jobz, irank, m, n, &dd_A, lda, strideA, &dd_S, strideS, &dd_U, ldu, strideU, &dd_V, ldv, strideV, &dd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuComplex * d_A, int lda, long long int strideA, float * d_S, long long int strideS, cuComplex * d_U, int ldu, long long int strideU, cuComplex * d_V, int ldv, long long int strideV, cuComplex * d_work, int lwork, int * d_info, double * h_R_nrmF, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipFloatComplex* A, int lda, long long int strideA, float* S, long long int strideS, hipFloatComplex* U, int ldu, long long int strideU, hipFloatComplex* V, int ldv, long long int strideV, hipFloatComplex* work, int lwork, int* devInfo, double* hRnrmF, int batch_count);
  // CHECK: status = hipsolverDnCgesvdaStridedBatched(handle, jobz, irank, m, n, &complexd_A, lda, strideA, &fd_S, strideS, &complexd_U, ldu, strideU, &complexd_V, ldv, strideV, &complexd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);
  status = cusolverDnCgesvdaStridedBatched(handle, jobz, irank, m, n, &complexd_A, lda, strideA, &fd_S, strideS, &complexd_U, ldu, strideU, &complexd_V, ldv, strideV, &complexd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZgesvdaStridedBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n, const cuDoubleComplex *d_A, int lda, long long int strideA, double * d_S, long long int strideS, cuDoubleComplex * d_U, int ldu, long long int strideU, cuDoubleComplex * d_V, int ldv, long long int strideV, cuDoubleComplex * d_work, int lwork, int * d_info, double * h_R_nrmF, int batchSize);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdaStridedBatched(hipsolverHandle_t handle, hipsolverEigMode_t jobz, int rank, int m, int n, const hipDoubleComplex* A, int lda, long long int strideA, double* S, long long int strideS, hipDoubleComplex* U, int ldu, long long int strideU, hipDoubleComplex* V, int ldv, long long int strideV, hipDoubleComplex* work, int lwork, int* devInfo, double* hRnrmF, int batch_count);
  // CHECK: status = hipsolverDnZgesvdaStridedBatched(handle, jobz, irank, m, n, &dComplexd_A, lda, strideA, &dd_S, strideS, &dComplexd_U, ldu, strideU, &dComplexd_V, ldv, strideV, &dComplexd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);
  status = cusolverDnZgesvdaStridedBatched(handle, jobz, irank, m, n, &dComplexd_A, lda, strideA, &dd_S, strideS, &dComplexd_U, ldu, strideU, &dComplexd_V, ldv, strideV, &dComplexd_Workspace, Lwork, &id_info, &dh_R_nrmF, batchSize);
#endif

#if CUDA_VERSION >= 10020
  // CHECK: int solver_int = 0;
  // CHECK: int lm = 0;
  // CHECK: int ln = 0;
  // CHECK: int lnrhs = 0;
  // CHECK: int ldda = 0;
  // CHECK: int lddb = 0;
  // CHECK: int lddx = 0;
  // CHECK: int dipiv = 0;
  // CHECK: int iter = 0;
  // CHECK: int d_info = 0;
  cusolver_int_t solver_int = 0;
  cusolver_int_t lm = 0;
  cusolver_int_t ln = 0;
  cusolver_int_t lnrhs = 0;
  cusolver_int_t ldda = 0;
  cusolver_int_t lddb = 0;
  cusolver_int_t lddx = 0;
  cusolver_int_t dipiv = 0;
  cusolver_int_t iter = 0;
  cusolver_int_t d_info = 0;

  // CHECK: hipsolverStatus_t STATUS_IRS_PARAMS_INVALID = HIPSOLVER_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hipsolverStatus_t STATUS_IRS_INTERNAL_ERROR = HIPSOLVER_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hipsolverStatus_t STATUS_IRS_NOT_SUPPORTED = HIPSOLVER_STATUS_NOT_SUPPORTED;
  cusolverStatus_t STATUS_IRS_PARAMS_INVALID = CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  cusolverStatus_t STATUS_IRS_INTERNAL_ERROR = CUSOLVER_STATUS_IRS_INTERNAL_ERROR;
  cusolverStatus_t STATUS_IRS_NOT_SUPPORTED = CUSOLVER_STATUS_IRS_NOT_SUPPORTED;

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, cuDoubleComplex * dB, cusolver_int_t lddb, cuDoubleComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex* A, int lda, int* devIpiv, hipDoubleComplex* B, int ldb, hipDoubleComplex* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnZZgesv(handle, ln, lnrhs, &dComplexA, ldda, &dipiv, &dComplexB, lddb, &dComplexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnZZgesv(handle, ln, lnrhs, &dComplexA, ldda, &dipiv, &dComplexB, lddb, &dComplexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, cuComplex * dB, cusolver_int_t lddb, cuComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex* A, int lda, int* devIpiv, hipFloatComplex* B, int ldb, hipFloatComplex* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnCCgesv(handle, ln, lnrhs, &complexA, ldda, &dipiv, &complexB, lddb, &complexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnCCgesv(handle, ln, lnrhs, &complexA, ldda, &dipiv, &complexB, lddb, &complexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, double * dB, cusolver_int_t lddb, double * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv(hipsolverHandle_t handle, int n, int nrhs, double* A, int lda, int* devIpiv, double* B, int ldb, double* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnDDgesv(handle, ln, lnrhs, &dA, ldda, &dipiv, &dB, lddb, &dX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnDDgesv(handle, ln, lnrhs, &dA, ldda, &dipiv, &dB, lddb, &dX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, float * dB, cusolver_int_t lddb, float * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv(hipsolverHandle_t handle, int n, int nrhs, float* A, int lda, int* devIpiv, float* B, int ldb, float* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnSSgesv(handle, ln, lnrhs, &fA, ldda, &dipiv, &fB, lddb, &fX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnSSgesv(handle, ln, lnrhs, &fA, ldda, &dipiv, &fB, lddb, &fX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZZgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, cuDoubleComplex * dB, cusolver_int_t lddb, cuDoubleComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipDoubleComplex* A, int lda, int* devIpiv, hipDoubleComplex* B, int ldb, hipDoubleComplex* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnZZgesv_bufferSize(handle, ln, lnrhs, &dComplexA, ldda, &dipiv, &dComplexB, lddb, &dComplexX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnZZgesv_bufferSize(handle, ln, lnrhs, &dComplexA, ldda, &dipiv, &dComplexB, lddb, &dComplexX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCCgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, cuComplex * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, cuComplex * dB, cusolver_int_t lddb, cuComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, hipFloatComplex* A, int lda, int* devIpiv, hipFloatComplex* B, int ldb, hipFloatComplex* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnCCgesv_bufferSize(handle, ln, lnrhs, &complexA, ldda, &dipiv, &complexB, lddb, &complexX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnCCgesv_bufferSize(handle, ln, lnrhs, &complexA, ldda, &dipiv, &complexB, lddb, &complexX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDDgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, double * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, double * dB, cusolver_int_t lddb, double * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, double* A, int lda, int* devIpiv, double* B, int ldb, double* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnDDgesv_bufferSize(handle, ln, lnrhs, &dA, ldda, &dipiv, &dB, lddb, &dX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnDDgesv_bufferSize(handle, ln, lnrhs, &dA, ldda, &dipiv, &dB, lddb, &dX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSSgesv_bufferSize(cusolverDnHandle_t handle, cusolver_int_t n, cusolver_int_t nrhs, float * dA, cusolver_int_t ldda, cusolver_int_t * dipiv, float * dB, cusolver_int_t lddb, float * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv_bufferSize(hipsolverHandle_t handle, int n, int nrhs, float* A, int lda, int* devIpiv, float* B, int ldb, float* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnSSgesv_bufferSize(handle, ln, lnrhs, &fA, ldda, &dipiv, &fB, lddb, &fX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnSSgesv_bufferSize(handle, ln, lnrhs, &fA, ldda, &dipiv, &fB, lddb, &fX, lddx, &Workspace, &lwork_bytes);
#endif

#if CUDA_VERSION >= 11000
  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZZgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex * dA, cusolver_int_t ldda, cuDoubleComplex * dB, cusolver_int_t lddb, cuDoubleComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, hipDoubleComplex* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnZZgels(handle, lm, ln, lnrhs, &dComplexA, ldda, &dComplexB, lddb, &dComplexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnZZgels(handle, lm, ln, lnrhs, &dComplexA, ldda, &dComplexB, lddb, &dComplexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCCgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex * dA, cusolver_int_t ldda, cuComplex * dB, cusolver_int_t lddb, cuComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, hipFloatComplex* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnCCgels(handle, lm, ln, lnrhs, &complexA, ldda, &complexB, lddb, &complexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnCCgels(handle, lm, ln, lnrhs, &complexA, ldda, &complexB, lddb, &complexX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDDgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double * dA, cusolver_int_t ldda, double * dB, cusolver_int_t lddb, double * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels(hipsolverHandle_t handle, int m, int n, int nrhs, double* A, int lda, double* B, int ldb, double* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnDDgels(handle, lm, ln, lnrhs, &dA, ldda, &dB, lddb, &dX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnDDgels(handle, lm, ln, lnrhs, &dA, ldda, &dB, lddb, &dX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSSgels(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float * dA, cusolver_int_t ldda, float * dB, cusolver_int_t lddb, float * dX, cusolver_int_t lddx, void * dWorkspace, size_t lwork_bytes, cusolver_int_t * iter, cusolver_int_t * d_info);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels(hipsolverHandle_t handle, int m, int n, int nrhs, float* A, int lda, float* B, int ldb, float* X, int ldx, void* work, size_t lwork, int* niters, int* devInfo);
  // CHECK: status = hipsolverDnSSgels(handle, lm, ln, lnrhs, &fA, ldda, &fB, lddb, &fX, lddx, &Workspace, lwork_bytes, &iter, &d_info);
  status = cusolverDnSSgels(handle, lm, ln, lnrhs, &fA, ldda, &fB, lddb, &fX, lddx, &Workspace, lwork_bytes, &iter, &d_info);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZZgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuDoubleComplex * dA, cusolver_int_t ldda, cuDoubleComplex * dB, cusolver_int_t lddb, cuDoubleComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipDoubleComplex* A, int lda, hipDoubleComplex* B, int ldb, hipDoubleComplex* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnZZgels_bufferSize(handle, lm, ln, lnrhs, &dComplexA, ldda, &dComplexB, lddb, &dComplexX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnZZgels_bufferSize(handle, lm, ln, lnrhs, &dComplexA, ldda, &dComplexB, lddb, &dComplexX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCCgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, cuComplex * dA, cusolver_int_t ldda, cuComplex * dB, cusolver_int_t lddb, cuComplex * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, hipFloatComplex* A, int lda, hipFloatComplex* B, int ldb, hipFloatComplex* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnCCgels_bufferSize(handle, lm, ln, lnrhs, &complexA, ldda, &complexB, lddb, &complexX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnCCgels_bufferSize(handle, lm, ln, lnrhs, &complexA, ldda, &complexB, lddb, &complexX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDDgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, double * dA, cusolver_int_t ldda, double * dB, cusolver_int_t lddb, double * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, double* A, int lda, double* B, int ldb, double* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnDDgels_bufferSize(handle, lm, ln, lnrhs, &dA, ldda, &dB, lddb, &dX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnDDgels_bufferSize(handle, lm, ln, lnrhs, &dA, ldda, &dB, lddb, &dX, lddx, &Workspace, &lwork_bytes);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSSgels_bufferSize(cusolverDnHandle_t handle, cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, float * dA, cusolver_int_t ldda, float * dB, cusolver_int_t lddb, float * dX, cusolver_int_t lddx, void * dWorkspace, size_t * lwork_bytes);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels_bufferSize(hipsolverHandle_t handle, int m, int n, int nrhs, float* A, int lda, float* B, int ldb, float* X, int ldx, void* work, size_t* lwork);
  // CHECK: status = hipsolverDnSSgels_bufferSize(handle, lm, ln, lnrhs, &fA, ldda, &fB, lddb, &fX, lddx, &Workspace, &lwork_bytes);
  status = cusolverDnSSgels_bufferSize(handle, lm, ln, lnrhs, &fA, ldda, &fB, lddb, &fX, lddx, &Workspace, &lwork_bytes);
#endif

  return 0;
}
