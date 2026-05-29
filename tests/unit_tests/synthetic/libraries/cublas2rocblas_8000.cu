// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocblas.h"
#include "cublas.h"
// CHECK-NOT: #include "rocblas.h"

int main() {
  printf("16.8000 cuBLAS API to rocBLAS API synthetic test\n");

  int n = 0;
  int k = 0;
  int lda = 0;
  int ldc = 0;
  void* Aptr = nullptr;
  void* Cptr = nullptr;
  float fa = 0;
  float fb = 0;

  // CHECK: rocblas_status blasStatus;
  cublasStatus blasStatus;

  // CHECK: rocblas_float_complex complexa, complexb;
  cuComplex complexa, complexb;

  // CHECK: rocblas_handle blasHandle;
  cublasHandle_t blasHandle;

  // CHECK: rocblas_fill blasFillMode;
  cublasFillMode_t blasFillMode;

  // CHECK: rocblas_operation blasOperation;
  cublasOperation_t blasOperation;

#if CUDA_VERSION >= 8000
  // CHECK: rocblas_datatype Atype, Ctype;
  cudaDataType Atype, Ctype;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_syrk_ex(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const void* alpha, const void* A, rocblas_datatype a_type, rocblas_int lda, const void* beta, void* C, rocblas_datatype c_type, rocblas_int ldc, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_syrk_ex(blasHandle, blasFillMode, blasOperation, n, k, &complexa, Aptr, Atype, lda, &complexb, Cptr, Ctype, ldc, rocblas_datatype_f32_c);
  blasStatus = cublasCsyrkEx(blasHandle, blasFillMode, blasOperation, n, k, &complexa, Aptr, Atype, lda, &complexb, Cptr, Ctype, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const void* A, cudaDataType Atype, int lda, const float* beta, void* C, cudaDataType Ctype, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_herk_ex(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const void* alpha, const void* A, rocblas_datatype a_type, rocblas_int lda, const void* beta, void* C, rocblas_datatype c_type, rocblas_int ldc, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_herk_ex(blasHandle, blasFillMode, blasOperation, n, k, &fa, Aptr, Atype, lda, &fb, Cptr, Ctype, ldc, rocblas_datatype_f32_c);
  blasStatus = cublasCherkEx(blasHandle, blasFillMode, blasOperation, n, k, &fa, Aptr, Atype, lda, &fb, Cptr, Ctype, ldc);
#endif

  return 0;
}
