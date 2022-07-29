// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hipblas.h"
#include "cublas.h"

int main() {
  printf("14. cuBLAS API to hipBLAS API synthetic test\n");

  // CHECK: hipblasOperation_t blasOperation;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_N = HIPBLAS_OP_N;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_T = HIPBLAS_OP_T;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_C = HIPBLAS_OP_C;
  cublasOperation_t blasOperation;
  cublasOperation_t BLAS_OP_N = CUBLAS_OP_N;
  cublasOperation_t BLAS_OP_T = CUBLAS_OP_T;
  cublasOperation_t BLAS_OP_C = CUBLAS_OP_C;

#if CUDA_VERSION >= 10010
  // CHECK: hipblasOperation_t BLAS_OP_HERMITAN = HIPBLAS_OP_C;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;
#endif

  // CHECK: hipblasStatus_t blasStatus;
  // CHECK-NEXT: hipblasStatus_t blasStatus_t;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_ALLOC_FAILED = HIPBLAS_STATUS_ALLOC_FAILED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_INVALID_VALUE = HIPBLAS_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_MAPPING_ERROR = HIPBLAS_STATUS_MAPPING_ERROR;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_EXECUTION_FAILED = HIPBLAS_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_INTERNAL_ERROR = HIPBLAS_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_NOT_SUPPORTED = HIPBLAS_STATUS_NOT_SUPPORTED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_ARCH_MISMATCH = HIPBLAS_STATUS_ARCH_MISMATCH;
  cublasStatus blasStatus;
  cublasStatus_t blasStatus_t;
  cublasStatus_t BLAS_STATUS_SUCCESS = CUBLAS_STATUS_SUCCESS;
  cublasStatus_t BLAS_STATUS_NOT_INITIALIZED = CUBLAS_STATUS_NOT_INITIALIZED;
  cublasStatus_t BLAS_STATUS_ALLOC_FAILED = CUBLAS_STATUS_ALLOC_FAILED;
  cublasStatus_t BLAS_STATUS_INVALID_VALUE = CUBLAS_STATUS_INVALID_VALUE;
  cublasStatus_t BLAS_STATUS_MAPPING_ERROR = CUBLAS_STATUS_MAPPING_ERROR;
  cublasStatus_t BLAS_STATUS_EXECUTION_FAILED = CUBLAS_STATUS_EXECUTION_FAILED;
  cublasStatus_t BLAS_STATUS_INTERNAL_ERROR = CUBLAS_STATUS_INTERNAL_ERROR;
  cublasStatus_t BLAS_STATUS_NOT_SUPPORTED = CUBLAS_STATUS_NOT_SUPPORTED;
  cublasStatus_t BLAS_STATUS_ARCH_MISMATCH = CUBLAS_STATUS_ARCH_MISMATCH;

  // CHECK: hipblasFillMode_t blasFillMode;
  // CHECK-NEXT: hipblasFillMode_t BLAS_FILL_MODE_LOWER = HIPBLAS_FILL_MODE_LOWER;
  // CHECK-NEXT: hipblasFillMode_t BLAS_FILL_MODE_UPPER = HIPBLAS_FILL_MODE_UPPER;
  cublasFillMode_t blasFillMode;
  cublasFillMode_t BLAS_FILL_MODE_LOWER = CUBLAS_FILL_MODE_LOWER;
  cublasFillMode_t BLAS_FILL_MODE_UPPER = CUBLAS_FILL_MODE_UPPER;

#if CUDA_VERSION >= 10010
  // CHECK: hipblasFillMode_t BLAS_FILL_MODE_FULL = HIPBLAS_FILL_MODE_FULL;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;
#endif

  // CHECK: hipblasDiagType_t blasDiagType;
  // CHECK-NEXT: hipblasDiagType_t BLAS_DIAG_NON_UNIT = HIPBLAS_DIAG_NON_UNIT;
  // CHECK-NEXT: hipblasDiagType_t BLAS_DIAG_UNIT = HIPBLAS_DIAG_UNIT;
  cublasDiagType_t blasDiagType;
  cublasDiagType_t BLAS_DIAG_NON_UNIT = CUBLAS_DIAG_NON_UNIT;
  cublasDiagType_t BLAS_DIAG_UNIT = CUBLAS_DIAG_UNIT;

  // CHECK: hipblasSideMode_t blasSideMode;
  // CHECK-NEXT: hipblasSideMode_t BLAS_SIDE_LEFT = HIPBLAS_SIDE_LEFT;
  // CHECK-NEXT: hipblasSideMode_t BLAS_SIDE_RIGHT = HIPBLAS_SIDE_RIGHT;
  cublasSideMode_t blasSideMode;
  cublasSideMode_t BLAS_SIDE_LEFT = CUBLAS_SIDE_LEFT;
  cublasSideMode_t BLAS_SIDE_RIGHT = CUBLAS_SIDE_RIGHT;

  // CHECK: hipblasPointerMode_t blasPointerMode;
  // CHECK-NEXT: hipblasPointerMode_t BLAS_POINTER_MODE_HOST = HIPBLAS_POINTER_MODE_HOST;
  // CHECK-NEXT: hipblasPointerMode_t BLAS_POINTER_MODE_DEVICE = HIPBLAS_POINTER_MODE_DEVICE;
  cublasPointerMode_t blasPointerMode;
  cublasPointerMode_t BLAS_POINTER_MODE_HOST = CUBLAS_POINTER_MODE_HOST;
  cublasPointerMode_t BLAS_POINTER_MODE_DEVICE = CUBLAS_POINTER_MODE_DEVICE;

  // CHECK: hipblasAtomicsMode_t blasAtomicsMode;
  // CHECK-NEXT: hipblasAtomicsMode_t BLAS_ATOMICS_NOT_ALLOWED = HIPBLAS_ATOMICS_NOT_ALLOWED;
  // CHECK-NEXT: hipblasAtomicsMode_t BLAS_ATOMICS_ALLOWED = HIPBLAS_ATOMICS_ALLOWED;
  cublasAtomicsMode_t blasAtomicsMode;
  cublasAtomicsMode_t BLAS_ATOMICS_NOT_ALLOWED = CUBLAS_ATOMICS_NOT_ALLOWED;
  cublasAtomicsMode_t BLAS_ATOMICS_ALLOWED = CUBLAS_ATOMICS_ALLOWED;

#if CUDA_VERSION >= 8000
  // CHECK: hipblasDatatype_t DataType;
  // CHECK-NEXT: hipblasDatatype_t DataType_t;
  // CHECK-NEXT: hipblasDatatype_t blasDataType;
  // CHECK-NEXT: hipblasDatatype_t R_16F = HIPBLAS_R_16F;
  // CHECK-NEXT: hipblasDatatype_t C_16F = HIPBLAS_C_16F;
  // CHECK-NEXT: hipblasDatatype_t R_32F = HIPBLAS_R_32F;
  // CHECK-NEXT: hipblasDatatype_t C_32F = HIPBLAS_C_32F;
  // CHECK-NEXT: hipblasDatatype_t R_64F = HIPBLAS_R_64F;
  // CHECK-NEXT: hipblasDatatype_t C_64F = HIPBLAS_C_64F;
  // CHECK-NEXT: hipblasDatatype_t R_8I = HIPBLAS_R_8I;
  // CHECK-NEXT: hipblasDatatype_t C_8I = HIPBLAS_C_8I;
  // CHECK-NEXT: hipblasDatatype_t R_8U = HIPBLAS_R_8U;
  // CHECK-NEXT: hipblasDatatype_t C_8U = HIPBLAS_C_8U;
  // CHECK-NEXT: hipblasDatatype_t R_32I = HIPBLAS_R_32I;
  // CHECK-NEXT: hipblasDatatype_t C_32I = HIPBLAS_C_32I;
  // CHECK-NEXT: hipblasDatatype_t R_32U = HIPBLAS_R_32U;
  // CHECK-NEXT: hipblasDatatype_t C_32U = HIPBLAS_C_32U;
  cudaDataType DataType;
  cudaDataType_t DataType_t;
  cublasDataType_t blasDataType;
  cublasDataType_t R_16F = CUDA_R_16F;
  cublasDataType_t C_16F = CUDA_C_16F;
  cublasDataType_t R_32F = CUDA_R_32F;
  cublasDataType_t C_32F = CUDA_C_32F;
  cublasDataType_t R_64F = CUDA_R_64F;
  cublasDataType_t C_64F = CUDA_C_64F;
  cublasDataType_t R_8I = CUDA_R_8I;
  cublasDataType_t C_8I = CUDA_C_8I;
  cublasDataType_t R_8U = CUDA_R_8U;
  cublasDataType_t C_8U = CUDA_C_8U;
  cublasDataType_t R_32I = CUDA_R_32I;
  cublasDataType_t C_32I = CUDA_C_32I;
  cublasDataType_t R_32U = CUDA_R_32U;
  cublasDataType_t C_32U = CUDA_C_32U;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipblasDatatype_t R_16BF = HIPBLAS_R_16B;
  // CHECK-NEXT: hipblasDatatype_t C_16BF = HIPBLAS_C_16B;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;
#endif

#if CUDA_VERSION >= 8000
  // CHECK: hipblasGemmAlgo_t blasGemmAlgo;
  // CHECK-NEXT: hipblasGemmAlgo_t BLAS_GEMM_DFALT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t blasGemmAlgo;
  cublasGemmAlgo_t BLAS_GEMM_DFALT = CUBLAS_GEMM_DFALT;
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipblasGemmAlgo_t BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
#endif

  // CHECK: hipblasHandle_t blasHandle;
  cublasHandle_t blasHandle;

  return 0;
}
