// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblas.h"
#include "cublas.h"
// CHECK-NOT: #include "hipblas.h"

int main() {
  printf("14.8000 cuBLAS API to hipBLAS API synthetic test\n");

  int n = 0;
  int k = 0;
  int lda = 0;
  int ldc = 0;
  void* Aptr = nullptr;
  void* Cptr = nullptr;
  float fa = 0;
  float fb = 0;

  // CHECK: hipblasStatus_t blasStatus;
  cublasStatus blasStatus;

  // CHECK: hipblasHandle_t blasHandle;
  cublasHandle_t blasHandle;

  // CHECK: hipblasFillMode_t blasFillMode;
  cublasFillMode_t blasFillMode;

  // CHECK: hipblasOperation_t blasOperation;
  cublasOperation_t blasOperation;

  // CHECK: hipComplex complexa, complexb;
  cuComplex complexa, complexb;

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType Atype, Ctype;
  cudaDataType Atype, Ctype;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSyrkEx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, const void* beta, void* C, hipDataType cType, int ldc, hipDataType computeType);
  // CHECK: blasStatus = hipblasSyrkEx(blasHandle, blasFillMode, blasOperation, n, k, &complexa, Aptr, Atype, lda, &complexb, Cptr, Ctype, ldc, HIP_C_32F);
  blasStatus = cublasCsyrkEx(blasHandle, blasFillMode, blasOperation, n, k, &complexa, Aptr, Atype, lda, &complexb, Cptr, Ctype, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHerkEx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, const void* beta, void* C, hipDataType cType, int ldc, hipDataType computeType);
  // CHECK: blasStatus = hipblasHerkEx(blasHandle, blasFillMode, blasOperation, n, k, &fa, Aptr, Atype, lda, &fb, Cptr, Ctype, ldc, HIP_C_32F);
  blasStatus = cublasCherkEx(blasHandle, blasFillMode, blasOperation, n, k, &fa, Aptr, Atype, lda, &fb, Cptr, Ctype, ldc);
#endif

  return 0;
}
