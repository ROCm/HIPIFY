// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocsolver.h"
#include "cusolverDn.h"

int main() {
  printf("20. cuSOLVER API to rocSOLVER API synthetic test\n");

  // CHECK: rocblas_handle handle;
  cusolverDnHandle_t handle;

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

  return 0;
}
