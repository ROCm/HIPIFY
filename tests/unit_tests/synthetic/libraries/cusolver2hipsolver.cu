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
  int nrhs = 0;
  int lda = 0;
  int ldb = 0;
  int Lwork = 0;
  int devIpiv = 0;
  int devInfo = 0;
  float fA = 0.f;
  float fB = 0.f;
  float fX = 0.f;
  double dA = 0.f;
  double dB = 0.f;
  double dX = 0.f;
  float fWorkspace = 0.f;
  double dWorkspace = 0.f;
  void *Workspace = nullptr;
  size_t lwork_bytes = 0;

  // CHECK: hipDoubleComplex dComplexA, dComplexB, dComplexX, dComplexWorkspace;
  cuDoubleComplex dComplexA, dComplexB, dComplexX, dComplexWorkspace;

  // CHECK: hipComplex complexA, complexB, complexX, complexWorkspace;
  cuComplex complexA, complexB, complexX, complexWorkspace;

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

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle, int m, int n,double* A, int lda, double* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnDgetrf(handle, m, n, &dA, lda, &dWorkspace, &devIpiv, &devInfo);
  status = cusolverDnDgetrf(handle, m, n, &dA, lda, &dWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnDgetrf_bufferSize(handle, m, n, &dA, lda, &Lwork);
  status = cusolverDnDgetrf_bufferSize(handle, m, n, &dA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle, int m, int n, float* A, int lda, float* work, int* devIpiv, int* devInfo);
  // CHECK: status = hipsolverDnSgetrf(handle, m, n, &fA, lda, &fWorkspace, &devIpiv, &devInfo);
  status = cusolverDnSgetrf(handle, m, n, &fA, lda, &fWorkspace, &devIpiv, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf_bufferSize(hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);
  // CHECK: status = hipsolverDnSgetrf_bufferSize(handle, m, n, &fA, lda, &Lwork);
  status = cusolverDnSgetrf_bufferSize(handle, m, n, &fA, lda, &Lwork);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDgetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnDgetrs(handle, blasOperation, n, nrhs , &dA, lda, &devIpiv, &dB, ldb, &devInfo);
  status = cusolverDnDgetrs(handle, blasOperation, n, nrhs , &dA, lda, &devIpiv, &dB, ldb, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSgetrs(cusolverDnHandle_t handle, cublasOperation_t  trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrs(hipsolverHandle_t handle, hipsolverOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
  // CHECK: status = hipsolverDnSgetrs(handle, blasOperation, n, nrhs , &fA, lda, &devIpiv, &fB, ldb, &devInfo);
  status = cusolverDnSgetrs(handle, blasOperation, n, nrhs , &fA, lda, &devIpiv, &fB, ldb, &devInfo);

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
  // HIP: HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, hipFloatComplex*  A, int lda, int* lwork);
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
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipsolverSyevjInfo_t syevj_info;
  syevjInfo_t syevj_info;

  // CHECK: hipsolverGesvdjInfo_t gesvdj_info;
  gesvdjInfo_t gesvdj_info;
#endif

#if CUDA_VERSION >= 10010
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

  // CHECK: hipsolverEigRange_t eigRange;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_ALL = HIPSOLVER_EIG_RANGE_ALL;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_I = HIPSOLVER_EIG_RANGE_I;
  // CHECK-NEXT: hipsolverEigRange_t EIG_RANGE_V = HIPSOLVER_EIG_RANGE_V;
  cusolverEigRange_t eigRange;
  cusolverEigRange_t EIG_RANGE_ALL = CUSOLVER_EIG_RANGE_ALL;
  cusolverEigRange_t EIG_RANGE_I = CUSOLVER_EIG_RANGE_I;
  cusolverEigRange_t EIG_RANGE_V = CUSOLVER_EIG_RANGE_V;
#endif

#if CUDA_VERSION >= 10020
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
