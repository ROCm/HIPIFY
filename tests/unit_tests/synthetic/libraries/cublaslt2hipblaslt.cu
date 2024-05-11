// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblaslt.h"
#include "cublasLt.h"
// CHECK-NOT: #include "hipblaslt.h"

int main() {
  printf("20. cuBLASLt API to hipBLASLt API synthetic test\n");

  // CHECK: hipblasLtHandle_t blasLtHandle;
  cublasLtHandle_t blasLtHandle;

  // CHECK: hipblasStatus_t status;
  cublasStatus_t status;

  const char *const_ch = nullptr;

#if CUDA_VERSION >= 10010
  // CHECK: hipblasLtMatmulAlgo_t blasLtMatmulAlgo;
  cublasLtMatmulAlgo_t blasLtMatmulAlgo;

  // CHECK: hipblasLtMatmulDesc_t blasLtMatmulDesc;
  cublasLtMatmulDesc_t blasLtMatmulDesc;

  // CHECK: hipblasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;
  cublasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;

  // CHECK: hipblasLtMatmulPreference_t blasLtMatmulPreference;
  cublasLtMatmulPreference_t blasLtMatmulPreference;

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
  // CHECK: status = hipblasLtCreate(&blasLtHandle);
  status = cublasLtCreate(&blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
  // CHECK: status = hipblasLtDestroy(blasLtHandle);
  status = cublasLtDestroy(blasLtHandle);

#if CUBLAS_VERSION >= 10200
  // CHECK: hipblasLtPointerMode_t blasLtPointerMode;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_HOST = HIPBLASLT_POINTER_MODE_HOST;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = HIPBLASLT_POINTER_MODE_DEVICE;
  cublasLtPointerMode_t blasLtPointerMode;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_HOST = CUBLASLT_POINTER_MODE_HOST;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = CUBLASLT_POINTER_MODE_DEVICE;
#endif
#endif

#if CUDA_VERSION >= 11000 && CUBLAS_VERSION >= 11000
  // CHECK: hipblasLtMatrixLayoutOpaque_t blasLtMatrixLayoutOpaque;
  cublasLtMatrixLayoutOpaque_t blasLtMatrixLayoutOpaque;

  // CHECK: hipblasLtMatmulDescOpaque_t blasLtMatmulDescOpaque;
  cublasLtMatmulDescOpaque_t blasLtMatmulDescOpaque;

  // CHECK: hipblasLtMatrixTransformDescOpaque_t blasLtMatrixTransformDescOpaque;
  cublasLtMatrixTransformDescOpaque_t blasLtMatrixTransformDescOpaque;

  // CHECK: hipblasLtMatmulPreferenceOpaque_t blasLtMatmulPreferenceOpaque;
  cublasLtMatmulPreferenceOpaque_t blasLtMatmulPreferenceOpaque;
#endif

#if CUDA_VERSION >= 11040 && CUBLAS_VERSION >= 11601
  // CHECK: hipblasLtPointerMode_t BLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
#endif
  return 0;
}
