// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipsolver.h"
#include "cusolverDn.h"

int main() {
  printf("19. cuSOLVER API to hipSOLVER API synthetic test\n");

  int m = 0;
  int n = 0;
  int k = 0;
  int nrhs = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int Lwork = 0;
  int devIpiv = 0;
  int devInfo = 0;
  int info = 0;
  int infoArray = 0;
  int batchSize = 0;
  float fA = 0.f;
  float fB = 0.f;
  float fC = 0.f;
  float fD = 0.f;
  float fE = 0.f;
  float fX = 0.f;
  float fTAU = 0.f;
  float fTAUQ = 0.f;
  float fTAUP = 0.f;
  double dA = 0.f;
  double dB = 0.f;
  double dC = 0.f;
  double dD = 0.f;
  double dE = 0.f;
  double dX = 0.f;
  double dTAU = 0.f;
  double dTAUQ = 0.f;
  double dTAUP = 0.f;
  float fWorkspace = 0.f;
  double dWorkspace = 0.f;
  void *Workspace = nullptr;
  size_t lwork_bytes = 0;

  float** fAarray = 0;
  float** fBarray = 0;
  double** dAarray = 0;
  double** dBarray = 0;

  // CHECK: hipDoubleComplex dComplexA, dComplexB, dComplexC, dComplexD, dComplexE, dComplexX, dComplexWorkspace, dComplexTAU, dComplexTAUQ, dComplexTAUP;
  cuDoubleComplex dComplexA, dComplexB, dComplexC, dComplexD, dComplexE, dComplexX, dComplexWorkspace, dComplexTAU, dComplexTAUQ, dComplexTAUP;

  // CHECK: hipComplex complexA, complexB, complexC, complexD, complexE, complexX, complexWorkspace, complexTAU, complexTAUQ, complexTAUP;
  cuComplex complexA, complexB, complexC, complexD, complexE, complexX, complexWorkspace, complexTAU, complexTAUQ, complexTAUP;

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
  cusolverStatus_t STATUS_NOT_SUPPORTED = CUSOLVER_STATUS_NOT_SUPPORTED;
  cusolverStatus_t STATUS_ZERO_PIVOT = CUSOLVER_STATUS_ZERO_PIVOT;

  // CHECK: hipblasOperation_t blasOperation;
  cublasOperation_t blasOperation;

  // CHECK: hipblasSideMode_t blasSideMode;
  cublasSideMode_t blasSideMode;

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

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

#if CUDA_VERSION >= 8000
  // CHECK: hipsolverEigType_t eigType;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_1 = HIPSOLVER_EIG_TYPE_1;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_2 = HIPSOLVER_EIG_TYPE_2;
  // CHECK-NEXT: hipsolverEigType_t EIG_TYPE_3 = HIPSOLVER_EIG_TYPE_3;
  cusolverEigType_t eigType;
  cusolverEigType_t EIG_TYPE_1 = CUSOLVER_EIG_TYPE_1;
  cusolverEigType_t EIG_TYPE_2 = CUSOLVER_EIG_TYPE_2;
  cusolverEigType_t EIG_TYPE_3 = CUSOLVER_EIG_TYPE_3;

  // CHECK: hipsolverEigMode_t eigMode;
  // CHECK-NEXT: hipsolverEigMode_t SOLVER_EIG_MODE_NOVECTOR = HIPSOLVER_EIG_MODE_NOVECTOR;
  // CHECK-NEXT: hipsolverEigMode_t SOLVER_EIG_MODE_VECTOR = HIPSOLVER_EIG_MODE_VECTOR;
  cusolverEigMode_t eigMode;
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
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverHandle_t  handle, hipblasSideMode_t side, hipblasOperation_t trans, int m, int n, int k, const float* A, int lda, const float* tau, const float* C, int ldc, int* lwork);
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
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipsolverSyevjInfo_t syevj_info;
  syevjInfo_t syevj_info;

  // CHECK: hipsolverGesvdjInfo_t gesvdj_info;
  gesvdjInfo_t gesvdj_info;
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
