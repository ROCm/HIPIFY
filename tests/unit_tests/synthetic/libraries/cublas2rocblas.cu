// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocblas.h"
// CHECK-NOT: #include "cublas_v2.h"
#include "cublas.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "rocblas.h"

int main() {
  printf("16. cuBLAS API to hipBLAS API synthetic test\n");

  // CHECK: rocblas_operation blasOperation;
  // CHECK-NEXT: rocblas_operation BLAS_OP_N = rocblas_operation_none;
  // CHECK-NEXT: rocblas_operation BLAS_OP_T = rocblas_operation_transpose;
  // CHECK-NEXT: rocblas_operation BLAS_OP_C = rocblas_operation_conjugate_transpose;
  cublasOperation_t blasOperation;
  cublasOperation_t BLAS_OP_N = CUBLAS_OP_N;
  cublasOperation_t BLAS_OP_T = CUBLAS_OP_T;
  cublasOperation_t BLAS_OP_C = CUBLAS_OP_C;

  // CHECK: rocblas_status blasStatus;
  // CHECK-NEXT: rocblas_status blasStatus_t;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_SUCCESS = rocblas_status_success;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_NOT_INITIALIZED = rocblas_status_invalid_handle;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_ALLOC_FAILED = rocblas_status_not_implemented;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_INVALID_VALUE = rocblas_status_invalid_pointer;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_MAPPING_ERROR = rocblas_status_invalid_size;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_EXECUTION_FAILED = rocblas_status_memory_error;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_INTERNAL_ERROR = rocblas_status_internal_error;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_NOT_SUPPORTED = rocblas_status_perf_degraded;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_ARCH_MISMATCH = rocblas_status_size_query_mismatch;
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

  // CHECK: rocblas_fill blasFillMode;
  // CHECK-NEXT: rocblas_fill BLAS_FILL_MODE_LOWER = rocblas_fill_lower;
  // CHECK-NEXT: rocblas_fill BLAS_FILL_MODE_UPPER = rocblas_fill_upper;
  cublasFillMode_t blasFillMode;
  cublasFillMode_t BLAS_FILL_MODE_LOWER = CUBLAS_FILL_MODE_LOWER;
  cublasFillMode_t BLAS_FILL_MODE_UPPER = CUBLAS_FILL_MODE_UPPER;

  // CHECK: rocblas_diagonal blasDiagType;
  // CHECK-NEXT: rocblas_diagonal BLAS_DIAG_NON_UNIT = rocblas_diagonal_non_unit;
  // CHECK-NEXT: rocblas_diagonal BLAS_DIAG_UNIT = rocblas_diagonal_unit;
  cublasDiagType_t blasDiagType;
  cublasDiagType_t BLAS_DIAG_NON_UNIT = CUBLAS_DIAG_NON_UNIT;
  cublasDiagType_t BLAS_DIAG_UNIT = CUBLAS_DIAG_UNIT;

  // CHECK: rocblas_side blasSideMode;
  // CHECK-NEXT: rocblas_side BLAS_SIDE_LEFT = rocblas_side_left;
  // CHECK-NEXT: rocblas_side BLAS_SIDE_RIGHT = rocblas_side_right;
  cublasSideMode_t blasSideMode;
  cublasSideMode_t BLAS_SIDE_LEFT = CUBLAS_SIDE_LEFT;
  cublasSideMode_t BLAS_SIDE_RIGHT = CUBLAS_SIDE_RIGHT;

  // CHECK: rocblas_pointer_mode blasPointerMode;
  // CHECK-NEXT: rocblas_pointer_mode BLAS_POINTER_MODE_HOST = rocblas_pointer_mode_host;
  // CHECK-NEXT: rocblas_pointer_mode BLAS_POINTER_MODE_DEVICE = rocblas_pointer_mode_device;
  cublasPointerMode_t blasPointerMode;
  cublasPointerMode_t BLAS_POINTER_MODE_HOST = CUBLAS_POINTER_MODE_HOST;
  cublasPointerMode_t BLAS_POINTER_MODE_DEVICE = CUBLAS_POINTER_MODE_DEVICE;

  // CHECK: rocblas_atomics_mode blasAtomicsMode;
  // CHECK-NEXT: rocblas_atomics_mode BLAS_ATOMICS_NOT_ALLOWED = rocblas_atomics_not_allowed;
  // CHECK-NEXT: rocblas_atomics_mode BLAS_ATOMICS_ALLOWED = rocblas_atomics_allowed;
  cublasAtomicsMode_t blasAtomicsMode;
  cublasAtomicsMode_t BLAS_ATOMICS_NOT_ALLOWED = CUBLAS_ATOMICS_NOT_ALLOWED;
  cublasAtomicsMode_t BLAS_ATOMICS_ALLOWED = CUBLAS_ATOMICS_ALLOWED;

  // CHECK: rocblas_gemm_algo blasGemmAlgo;
  // CHECK-NEXT: rocblas_gemm_algo BLAS_GEMM_DFALT = rocblas_gemm_algo_standard;
  cublasGemmAlgo_t blasGemmAlgo;
  cublasGemmAlgo_t BLAS_GEMM_DFALT = CUBLAS_GEMM_DFALT;

#if CUDA_VERSION >= 9000
  // CHECK: rocblas_gemm_algo BLAS_GEMM_DEFAULT = rocblas_gemm_algo_standard;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: rocblas_operation BLAS_OP_HERMITAN = rocblas_operation_conjugate_transpose;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;

  // CHECK: rocblas_fill BLAS_FILL_MODE_FULL = rocblas_fill_full;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;
#endif

  return 0;
}
