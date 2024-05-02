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
  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
  // CHECK: status = hipblasLtCreate(&blasLtHandle);
  status = cublasLtCreate(&blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
  // CHECK: status = hipblasLtDestroy(blasLtHandle);
  status = cublasLtDestroy(blasLtHandle);
#endif

  return 0;
}
