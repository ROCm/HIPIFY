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
  cusolverStatus_t status;

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
