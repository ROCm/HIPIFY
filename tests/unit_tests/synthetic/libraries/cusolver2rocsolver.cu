// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocsolver/rocsolver.h"
#include "cusolverDn.h"

int main() {
  printf("20. cuSOLVER API to rocSOLVER API synthetic test\n");

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

  // CHECK: rocblas_double_complex dComplexA, dComplexB, dComplexX, dComplexWorkspace;
  cuDoubleComplex dComplexA, dComplexB, dComplexX, dComplexWorkspace;

  // CHECK: rocblas_float_complex complexA, complexB, complexX, complexWorkspace;
  cuComplex complexA, complexB, complexX, complexWorkspace;

  // CHECK: rocblas_handle handle;
  cusolverDnHandle_t handle;

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  // CHECK: rocblas_fill fillMode;
  cublasFillMode_t fillMode;

  // CHECK: rocblas_status status;
  // CHECK-NEXT: rocblas_status STATUS_SUCCESS = rocblas_status_success;
  // CHECK-NEXT: rocblas_status STATUS_NOT_INITIALIZED = rocblas_status_invalid_handle;
  // CHECK-NEXT: rocblas_status STATUS_ALLOC_FAILED = rocblas_status_memory_error;
  // CHECK-NEXT: rocblas_status STATUS_INVALID_VALUE = rocblas_status_invalid_value;
  // CHECK-NEXT: rocblas_status STATUS_ARCH_MISMATCH = rocblas_status_arch_mismatch;
  // CHECK-NEXT: rocblas_status STATUS_MAPPING_ERROR = rocblas_status_not_implemented;
  // CHECK-NEXT: rocblas_status STATUS_EXECUTION_FAILED = rocblas_status_not_implemented;
  // CHECK-NEXT: rocblas_status STATUS_INTERNAL_ERROR = rocblas_status_internal_error;
  // CHECK-NEXT: rocblas_status STATUS_NOT_SUPPORTED = rocblas_status_not_implemented;
  // CHECK-NEXT: rocblas_status STATUS_ZERO_PIVOT = rocblas_status_not_implemented;
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

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle);
  // CHECK: status = rocblas_create_handle(&handle);
  status = cusolverDnCreate(&handle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle);
  // CHECK: status = rocblas_destroy_handle(handle);
  status = cusolverDnDestroy(handle);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);
  // CHECK: status = rocblas_set_stream(handle, stream_t);
  status = cusolverDnSetStream(handle, stream_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream);
  // CHECK: status = rocblas_get_stream(handle, &stream_t);
  status = cusolverDnGetStream(handle, &stream_t);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float * A, int lda, float * Workspace, int Lwork, int * devInfo);
  // ROC: ROCSOLVER_EXPORT rocblas_status rocsolver_spotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, float* A, const rocblas_int lda, rocblas_int* info);
  // CHECK: status = rocsolver_spotrf(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &devInfo);
  status = cusolverDnSpotrf(handle, fillMode, n, &fA, lda, &fWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double * A, int lda, double * Workspace, int Lwork, int * devInfo);
  // ROC: ROCSOLVER_EXPORT rocblas_status rocsolver_dpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, double* A, const rocblas_int lda, rocblas_int* info);
  // CHECK: status = rocsolver_dpotrf(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &devInfo);
  status = cusolverDnDpotrf(handle, fillMode, n, &dA, lda, &dWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnCpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex * A, int lda, cuComplex * Workspace, int Lwork, int * devInfo);
  // ROC: ROCSOLVER_EXPORT rocblas_status rocsolver_cpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, rocblas_float_complex* A, const rocblas_int lda, rocblas_int* info);
  // CHECK: status = rocsolver_cpotrf(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &devInfo);
  status = cusolverDnCpotrf(handle, fillMode, n, &complexA, lda, &complexWorkspace, Lwork, &devInfo);

  // CUDA: cusolverStatus_t CUSOLVERAPI cusolverDnZpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex * A, int lda, cuDoubleComplex * Workspace, int Lwork, int * devInfo);
  // ROC: ROCSOLVER_EXPORT rocblas_status rocsolver_zpotrf(rocblas_handle handle, const rocblas_fill uplo, const rocblas_int n, rocblas_double_complex* A, const rocblas_int lda, rocblas_int* info);
  // CHECK: status = rocsolver_zpotrf(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &devInfo);
  status = cusolverDnZpotrf(handle, fillMode, n, &dComplexA, lda, &dComplexWorkspace, Lwork, &devInfo);

#if CUDA_VERSION >= 8000
  // CHECK: rocblas_eform eigType;
  // CHECK-NEXT: rocblas_eform EIG_TYPE_1 = rocblas_eform_ax;
  // CHECK-NEXT: rocblas_eform EIG_TYPE_2 = rocblas_eform_abx;
  // CHECK-NEXT: rocblas_eform EIG_TYPE_3 = rocblas_eform_bax;
  cusolverEigType_t eigType;
  cusolverEigType_t EIG_TYPE_1 = CUSOLVER_EIG_TYPE_1;
  cusolverEigType_t EIG_TYPE_2 = CUSOLVER_EIG_TYPE_2;
  cusolverEigType_t EIG_TYPE_3 = CUSOLVER_EIG_TYPE_3;

  // CHECK: rocblas_evect eigMode;
  // CHECK-NEXT: rocblas_evect SOLVER_EIG_MODE_NOVECTOR = rocblas_evect_none;
  // CHECK-NEXT: rocblas_evect SOLVER_EIG_MODE_VECTOR = rocblas_evect_original;
  cusolverEigMode_t eigMode;
  cusolverEigMode_t SOLVER_EIG_MODE_NOVECTOR = CUSOLVER_EIG_MODE_NOVECTOR;
  cusolverEigMode_t SOLVER_EIG_MODE_VECTOR = CUSOLVER_EIG_MODE_VECTOR;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: rocblas_erange eigRange;
  // CHECK-NEXT: rocblas_erange EIG_RANGE_ALL = rocblas_erange_all;
  // CHECK-NEXT: rocblas_erange EIG_RANGE_I = rocblas_erange_index;
  // CHECK-NEXT: rocblas_erange EIG_RANGE_V = rocblas_erange_value;
  cusolverEigRange_t eigRange;
  cusolverEigRange_t EIG_RANGE_ALL = CUSOLVER_EIG_RANGE_ALL;
  cusolverEigRange_t EIG_RANGE_I = CUSOLVER_EIG_RANGE_I;
  cusolverEigRange_t EIG_RANGE_V = CUSOLVER_EIG_RANGE_V;
#endif

  return 0;
}
