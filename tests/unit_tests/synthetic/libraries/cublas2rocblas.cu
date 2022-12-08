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

  // CHECK: rocblas_handle blasHandle;
  cublasHandle_t blasHandle;

  // CUDA: cublasStatus CUBLASWINAPI cublasInit(void);
  // ROC: ROCBLAS_EXPORT void rocblas_initialize(void);
  // CHECK: blasStatus = rocblas_initialize();
  blasStatus = cublasInit();

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_atomics_mode(rocblas_handle handle, rocblas_atomics_mode* atomics_mode);
  // CHECK: blasStatus = rocblas_get_atomics_mode(blasHandle, &blasAtomicsMode);
  blasStatus = cublasGetAtomicsMode(blasHandle, &blasAtomicsMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_atomics_mode(rocblas_handle handle, rocblas_atomics_mode atomics_mode);
  // CHECK: blasStatus = rocblas_set_atomics_mode(blasHandle, blasAtomicsMode);
  blasStatus = cublasSetAtomicsMode(blasHandle, blasAtomicsMode);

  const char* const_ch = nullptr;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle);
  // CUDA: #define cublasCreate cublasCreate_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_create_handle(rocblas_handle* handle);
  // CHECK: blasStatus = rocblas_create_handle(&blasHandle);
  // CHECK-NEXT: blasStatus = rocblas_create_handle(&blasHandle);
  blasStatus = cublasCreate(&blasHandle);
  blasStatus = cublasCreate_v2(&blasHandle);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle);
  // CUDA: #define cublasDestroy cublasDestroy_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_destroy_handle(rocblas_handle handle);
  // CHECK: blasStatus = rocblas_destroy_handle(blasHandle);
  // CHECK-NEXT: blasStatus = rocblas_destroy_handle(blasHandle);
  blasStatus = cublasDestroy(blasHandle);
  blasStatus = cublasDestroy_v2(blasHandle);

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
  // CUDA: #define cublasSetStream cublasSetStream_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);
  // CHECK: blasStatus = rocblas_set_stream(blasHandle, stream);
  // CHECK-NEXT: blasStatus = rocblas_set_stream(blasHandle, stream);
  blasStatus = cublasSetStream(blasHandle, stream);
  blasStatus = cublasSetStream_v2(blasHandle, stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
  // CUDA: #define cublasGetStream cublasGetStream_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_stream(rocblas_handle handle, hipStream_t* stream);
  // CHECK: blasStatus = rocblas_get_stream(blasHandle, &stream);
  // CHECK-NEXT: blasStatus = rocblas_get_stream(blasHandle, &stream);
  blasStatus = cublasGetStream(blasHandle, &stream);
  blasStatus = cublasGetStream_v2(blasHandle, &stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
  // CUDA: #define cublasSetPointerMode cublasSetPointerMode_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_pointer_mode(rocblas_handle handle, rocblas_pointer_mode pointer_mode);
  // CHECK: blasStatus = rocblas_set_pointer_mode(blasHandle, blasPointerMode);
  // CHECK-NEXT: blasStatus = rocblas_set_pointer_mode(blasHandle, blasPointerMode);
  blasStatus = cublasSetPointerMode(blasHandle, blasPointerMode);
  blasStatus = cublasSetPointerMode_v2(blasHandle, blasPointerMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
  // CUDA: #define cublasGetPointerMode cublasGetPointerMode_v2
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_pointer_mode(rocblas_handle handle, rocblas_pointer_mode* pointer_mode);
  // CHECK: blasStatus = rocblas_get_pointer_mode(blasHandle, &blasPointerMode);
  // CHECK-NEXT: blasStatus = rocblas_get_pointer_mode(blasHandle, &blasPointerMode);
  blasStatus = cublasGetPointerMode(blasHandle, &blasPointerMode);
  blasStatus = cublasGetPointerMode_v2(blasHandle, &blasPointerMode);

  int n = 0;
  int num = 0;
  int incx = 0;
  int incy = 0;
  void* image = nullptr;
  void* image_2 = nullptr;

  // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1281
  // TODO: Apply the chosen typecasting of int to rocblas_int arguments

  /*
  #if defined(rocblas_ILP64)
    typedef int64_t rocblas_int;
  #else
    typedef int32_t rocblas_int;
  #endif
  */

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_vector(rocblas_int n, rocblas_int elem_size, const void* x, rocblas_int incx, void* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_set_vector(n, num, image, incx, image_2, incy);
  blasStatus = cublasSetVector(n, num, image, incx, image_2, incy);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_vector(rocblas_int n, rocblas_int elem_size, const void* x, rocblas_int incx, void* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_get_vector(n, num, image, incx, image_2, incy);
  blasStatus = cublasGetVector(n, num, image, incx, image_2, incy);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_vector_async(rocblas_int n, rocblas_int elem_size, const void* x, rocblas_int incx, void* y, rocblas_int incy, hipStream_t stream);
  // CHECK: blasStatus = rocblas_set_vector_async(n, num, image, incx, image_2, incy, stream);
  blasStatus = cublasSetVectorAsync(n, num, image, incx, image_2, incy, stream);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_vector_async(rocblas_int n, rocblas_int elem_size, const void* x, rocblas_int incx, void* y, rocblas_int incy, hipStream_t stream);
  // CHECK: blasStatus = rocblas_get_vector_async(n, num, image, incx, image_2, incy, stream);
  blasStatus = cublasGetVectorAsync(n, num, image, incx, image_2, incy, stream);

  int rows = 0;
  int cols = 0;

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void* a, rocblas_int lda, void* b, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_set_matrix(rows, cols, num, image, incx, image_2, incy);
  blasStatus = cublasSetMatrix(rows, cols, num, image, incx, image_2, incy);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_matrix(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void* a, rocblas_int lda, void* b, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_get_matrix(rows, cols, num, image, incx, image_2, incy);
  blasStatus = cublasGetMatrix(rows, cols, num, image, incx, image_2, incy);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_set_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void* a, rocblas_int lda, void* b, rocblas_int ldb, hipStream_t stream);
  // CHECK: blasStatus = rocblas_set_matrix_async(rows, cols, num, image, incx, image_2, incy, stream);
  blasStatus = cublasSetMatrixAsync(rows, cols, num, image, incx, image_2, incy, stream);

  // TODO: #1281
  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_get_matrix_async(rocblas_int rows, rocblas_int cols, rocblas_int elem_size, const void* a, rocblas_int lda, void* b, rocblas_int ldb, hipStream_t stream);
  // CHECK: blasStatus = rocblas_get_matrix_async(rows, cols, num, image, incx, image_2, incy, stream);
  blasStatus = cublasGetMatrixAsync(rows, cols, num, image, incx, image_2, incy, stream);

  float fa = 0;
  float fA = 0;
  float fb = 0;
  float fB = 0;
  float fx = 0;
  float fx1 = 0;
  float fy = 0;
  float fy1 = 0;
  float fc = 0;
  float fC = 0;
  float fs = 0;
  float fd1 = 0;
  float fd2 = 0;
  float fresult = 0;

  float** fAarray = 0;
  float** fBarray = 0;
  float** fCarray = 0;
  float** fTauarray = 0;

  double da = 0;
  double dA = 0;
  double db = 0;
  double dB = 0;
  double dx = 0;
  double dx1 = 0;
  double dy = 0;
  double dy1 = 0;
  double dc = 0;
  double dC = 0;
  double ds = 0;
  double dd1 = 0;
  double dd2 = 0;
  double dresult = 0;

  double** dAarray = 0;
  double** dBarray = 0;
  double** dCarray = 0;
  double** dTauarray = 0;

  void** voidAarray = nullptr;
  void** voidBarray = nullptr;
  void** voidCarray = nullptr;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_snrm2(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);
  // CHECK: blasStatus = rocblas_snrm2(blasHandle, n, &fx, incx, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_snrm2(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSnrm2(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSnrm2_v2(blasHandle, n, &fx, incx, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dnrm2(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);
  // CHECK: blasStatus = rocblas_dnrm2(blasHandle, n, &dx, incx, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dnrm2(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDnrm2(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDnrm2_v2(blasHandle, n, &dx, incx, &dresult);

  // CHECK: rocblas_float_complex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb;
  cuComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb;
  // CHECK: rocblas_double_complex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb;
  cuDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb;

  // CHECK: rocblas_float_complex** complexAarray = 0;
  // CHECK-NEXT: rocblas_float_complex** complexBarray = 0;
  // CHECK-NEXT: rocblas_float_complex** complexCarray = 0;
  // CHECK-NEXT: rocblas_float_complex** complexTauarray = 0;
  cuComplex** complexAarray = 0;
  cuComplex** complexBarray = 0;
  cuComplex** complexCarray = 0;
  cuComplex** complexTauarray = 0;

  // CHECK: rocblas_double_complex** dcomplexAarray = 0;
  // CHECK-NEXT: rocblas_double_complex** dcomplexBarray = 0;
  // CHECK-NEXT: rocblas_double_complex** dcomplexCarray = 0;
  // CHECK-NEXT: rocblas_double_complex** dcomplexTauarray = 0;
  cuDoubleComplex** dcomplexAarray = 0;
  cuDoubleComplex** dcomplexBarray = 0;
  cuDoubleComplex** dcomplexCarray = 0;
  cuDoubleComplex** dcomplexTauarray = 0;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scnrm2(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, float* result);
  // CHECK: blasStatus = rocblas_scnrm2(blasHandle, n, &complex, incx, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_scnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dznrm2(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, double* result);
  // CHECK: blasStatus = rocblas_dznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  blasStatus = cublasDznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  blasStatus = cublasDznrm2_v2(blasHandle, n, &dcomplex, incx, &dresult);

#if CUDA_VERSION >= 8000
  // CHECK: rocblas_datatype DataType;
  // CHECK-NEXT: rocblas_datatype_ DataType_t;
  // CHECK-NEXT: rocblas_datatype blasDataType;
  // CHECK-NEXT: rocblas_datatype R_16F = rocblas_datatype_f16_r;
  // CHECK-NEXT: rocblas_datatype C_16F = rocblas_datatype_f16_c;
  // CHECK-NEXT: rocblas_datatype R_32F = rocblas_datatype_f32_r;
  // CHECK-NEXT: rocblas_datatype C_32F = rocblas_datatype_f32_c;
  // CHECK-NEXT: rocblas_datatype R_64F = rocblas_datatype_f64_r;
  // CHECK-NEXT: rocblas_datatype C_64F = rocblas_datatype_f64_c;
  // CHECK-NEXT: rocblas_datatype R_8I = rocblas_datatype_i8_r;
  // CHECK-NEXT: rocblas_datatype C_8I = rocblas_datatype_i8_c;
  // CHECK-NEXT: rocblas_datatype R_8U = rocblas_datatype_u8_r;
  // CHECK-NEXT: rocblas_datatype C_8U = rocblas_datatype_u8_c;
  // CHECK-NEXT: rocblas_datatype R_32I = rocblas_datatype_i32_r;
  // CHECK-NEXT: rocblas_datatype C_32I = rocblas_datatype_i32_c;
  // CHECK-NEXT: rocblas_datatype R_32U = rocblas_datatype_u32_r;
  // CHECK-NEXT: rocblas_datatype C_32U = rocblas_datatype_u32_c;
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

#if CUDA_VERSION >= 11000
  // CHECK: rocblas_datatype R_16BF = rocblas_datatype_bf16_r;
  // CHECK-NEXT: rocblas_datatype C_16BF = rocblas_datatype_bf16_c;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;
#endif

#if CUDA_VERSION >= 11040
  // CUDA: CUBLASAPI const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status);
  // ROC: ROCBLAS_EXPORT const char* rocblas_status_to_string(rocblas_status status);
  // CHECK: const_ch = rocblas_status_to_string(blasStatus);
  const_ch = cublasGetStatusString(blasStatus);
#endif

  return 0;
}
