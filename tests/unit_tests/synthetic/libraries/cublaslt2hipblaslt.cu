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

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  void *A = nullptr;
  void *B = nullptr;
  void *C = nullptr;
  void *D = nullptr;
  void *alpha = nullptr;
  void *beta = nullptr;
  void *workspace = nullptr;
  const char *const_ch = nullptr;

  size_t workspaceSizeInBytes = 0;

#if CUDA_VERSION >= 10010
  // CHECK: hipblasLtMatmulAlgo_t blasLtMatmulAlgo;
  cublasLtMatmulAlgo_t blasLtMatmulAlgo;

  // CHECK: hipblasLtMatmulDesc_t blasLtMatmulDesc;
  cublasLtMatmulDesc_t blasLtMatmulDesc;

  // CHECK: hipblasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;
  cublasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;

  // CHECK: hipblasLtMatmulPreference_t blasLtMatmulPreference;
  cublasLtMatmulPreference_t blasLtMatmulPreference;

  // CHECK: hipblasLtMatrixLayout_t blasLtMatrixLayout, Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatrixLayout_t blasLtMatrixLayout, Adesc, Bdesc, Cdesc, Ddesc;

  // CHECK: hipblasLtOrder_t blasLtOrder;
  // CHECK-NEXT: hipblasLtOrder_t BLASLT_ORDER_COL = HIPBLASLT_ORDER_COL;
  // CHECK-NEXT: hipblasLtOrder_t BLASLT_ORDER_ROW = HIPBLASLT_ORDER_ROW;
  cublasLtOrder_t blasLtOrder;
  cublasLtOrder_t BLASLT_ORDER_COL = CUBLASLT_ORDER_COL;
  cublasLtOrder_t BLASLT_ORDER_ROW = CUBLASLT_ORDER_ROW;

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
  // CHECK: status = hipblasLtCreate(&blasLtHandle);
  status = cublasLtCreate(&blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
  // CHECK: status = hipblasLtDestroy(blasLtHandle);
  status = cublasLtDestroy(blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc, const void* B, hipblasLtMatrixLayout_t Bdesc, const void* beta, const void* C, hipblasLtMatrixLayout_t Cdesc, void* D, hipblasLtMatrixLayout_t Ddesc, const hipblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, hipStream_t stream);
  // CHECK: status = hipblasLtMatmul(blasLtHandle, blasLtMatmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, &blasLtMatmulAlgo, workspace, workspaceSizeInBytes, stream);
  status = cublasLtMatmul(blasLtHandle, blasLtMatmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, &blasLtMatmulAlgo, workspace, workspaceSizeInBytes, stream);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t lightHandle, hipblasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc, const void* beta, const void* B, hipblasLtMatrixLayout_t Bdesc, void* C, hipblasLtMatrixLayout_t Cdesc, hipStream_t stream);
  // CHECK: status = hipblasLtMatrixTransform(blasLtHandle, blasLtMatrixTransformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
  status = cublasLtMatrixTransform(blasLtHandle, blasLtMatrixTransformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
#endif

#if CUBLAS_VERSION >= 10200
  // CHECK: hipblasLtPointerMode_t blasLtPointerMode;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_HOST = HIPBLASLT_POINTER_MODE_HOST;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = HIPBLASLT_POINTER_MODE_DEVICE;
  cublasLtPointerMode_t blasLtPointerMode;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_HOST = CUBLASLT_POINTER_MODE_HOST;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = CUBLASLT_POINTER_MODE_DEVICE;
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
