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
  int res = 0;
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

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sdot(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* result);
  // CHECK: blasStatus = rocblas_sdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_sdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSdot_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ddot(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* result);
  // CHECK: blasStatus = rocblas_ddot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_ddot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDdot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDdot_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cdotu(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* result);
  // CHECK: blasStatus = rocblas_cdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = rocblas_cdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cdotc(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* result);
  // CHECK: blasStatus = rocblas_cdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = rocblas_cdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdotu(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* result);
  // CHECK: blasStatus = rocblas_zdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = rocblas_zdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdotc(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* result);
  // CHECK: blasStatus = rocblas_zdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = rocblas_zdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotc_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sscal(rocblas_handle handle, rocblas_int n, const float* alpha, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_sscal(blasHandle, n, &fy, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_sscal(blasHandle, n, &fy, &fx, incx);
  blasStatus = cublasSscal(blasHandle, n, &fy, &fx, incx);
  blasStatus = cublasSscal_v2(blasHandle, n, &fy, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dscal(rocblas_handle handle, rocblas_int n, const double* alpha, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dscal(blasHandle, n, &dx, &dy, incx);
  // CHECK-NEXT: blasStatus = rocblas_dscal(blasHandle, n, &dx, &dy, incx);
  blasStatus = cublasDscal(blasHandle, n, &dx, &dy, incx);
  blasStatus = cublasDscal_v2(blasHandle, n, &dx, &dy, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cscal(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* alpha, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_cscal(blasHandle, n, &complexa, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_cscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal_v2(blasHandle, n, &complexa, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csscal(rocblas_handle handle, rocblas_int n, const float* alpha, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_csscal(blasHandle, n, &fx, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_csscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal_v2(blasHandle, n, &fx, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zscal(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* alpha, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_zscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_zscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdscal(rocblas_handle handle, rocblas_int n, const double* alpha, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_zdscal(blasHandle, n, &dx, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_zdscal(blasHandle, n, &dx, &dcomplexx, incx);
  blasStatus = cublasZdscal(blasHandle, n, &dx, &dcomplexx, incx);
  blasStatus = cublasZdscal_v2(blasHandle, n, &dx, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_saxpy(rocblas_handle handle, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_saxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_saxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  blasStatus = cublasSaxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  blasStatus = cublasSaxpy_v2(blasHandle, n, &fa, &fx, incx, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_daxpy(rocblas_handle handle, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_daxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_daxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  blasStatus = cublasDaxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  blasStatus = cublasDaxpy_v2(blasHandle, n, &da, &dx, incx, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_caxpy(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_caxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_caxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zaxpy(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZaxpy_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scopy(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_scopy(blasHandle, n, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_scopy(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasScopy(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasScopy_v2(blasHandle, n, &fx, incx, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dcopy(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dcopy(blasHandle, n, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dcopy(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDcopy(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDcopy_v2(blasHandle, n, &dx, incx, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ccopy(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_ccopy(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_ccopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zcopy(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZcopy_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sswap(rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_sswap(blasHandle, n, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_sswap(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasSswap(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasSswap_v2(blasHandle, n, &fx, incx, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dswap(rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dswap(blasHandle, n, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dswap(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDswap(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDswap_v2(blasHandle, n, &dx, incx, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cswap(rocblas_handle handle, rocblas_int n, rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_cswap(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_cswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zswap(rocblas_handle handle, rocblas_int n, rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZswap_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_isamax(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_isamax(blasHandle, n, &fx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_isamax(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamax(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamax_v2(blasHandle, n, &fx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_idamax(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_idamax(blasHandle, n, &dx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_idamax(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamax(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamax_v2(blasHandle, n, &dx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_icamax(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_icamax(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_icamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax_v2(blasHandle, n, &complexx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_izamax(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_izamax(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_izamax(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamax(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamax_v2(blasHandle, n, &dcomplexx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_isamin(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_isamin(blasHandle, n, &fx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_isamin(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamin(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamin_v2(blasHandle, n, &fx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_idamin(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_idamin(blasHandle, n, &dx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_idamin(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamin(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamin_v2(blasHandle, n, &dx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_icamin(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_icamin(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_icamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin_v2(blasHandle, n, &complexx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_izamin(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, rocblas_int* result);
  // CHECK: blasStatus = rocblas_izamin(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = rocblas_izamin(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamin(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamin_v2(blasHandle, n, &dcomplexx, incx, &res);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sasum(rocblas_handle handle, rocblas_int n, const float* x, rocblas_int incx, float* result);
  // CHECK: blasStatus = rocblas_sasum(blasHandle, n, &fx, incx, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_sasum(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSasum(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSasum_v2(blasHandle, n, &fx, incx, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dasum(rocblas_handle handle, rocblas_int n, const double* x, rocblas_int incx, double* result);
  // CHECK: blasStatus = rocblas_dasum(blasHandle, n, &dx, incx, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dasum(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDasum(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDasum_v2(blasHandle, n, &dx, incx, &dresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scasum(rocblas_handle handle, rocblas_int n, const rocblas_float_complex* x, rocblas_int incx, float* result);
  // CHECK: blasStatus = rocblas_scasum(blasHandle, n, &complexx, incx, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_scasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum_v2(blasHandle, n, &complexx, incx, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dzasum(rocblas_handle handle, rocblas_int n, const rocblas_double_complex* x, rocblas_int incx, double* result);
  // CHECK: blasStatus = rocblas_dzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  blasStatus = cublasDzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  blasStatus = cublasDzasum_v2(blasHandle, n, &dcomplexx, incx, &dresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srot(rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy, const float* c, const float* s);
  // CHECK: blasStatus = rocblas_srot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  // CHECK-NEXT: blasStatus = rocblas_srot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  blasStatus = cublasSrot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  blasStatus = cublasSrot_v2(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drot(rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy, const double* c, const double* s);
  // CHECK: blasStatus = rocblas_drot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  // CHECK-NEXT: blasStatus = rocblas_drot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  blasStatus = cublasDrot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  blasStatus = cublasDrot_v2(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);

  // TODO: #1281
  // CUDA: CUBLASAPI CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_crot(rocblas_handle handle, rocblas_int n, rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* y, rocblas_int incy, const float* c, const rocblas_float_complex* s);
  // CHECK: blasStatus = rocblas_crot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  // CHECK-NEXT: blasStatus = rocblas_crot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csrot(rocblas_handle handle, rocblas_int n, rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* y, rocblas_int incy, const float* c, const float* s);
  // CHECK: blasStatus = rocblas_csrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  // CHECK-NEXT: blasStatus = rocblas_csrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zrot(rocblas_handle handle, rocblas_int n, rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* y, rocblas_int incy, const double* c, const rocblas_double_complex* s);
  // CHECK: blasStatus = rocblas_zrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = rocblas_zrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdrot(rocblas_handle handle, rocblas_int n, rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* y, rocblas_int incy, const double* c, const double* s);
  // CHECK: blasStatus = rocblas_zdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  // CHECK-NEXT: blasStatus = rocblas_zdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  blasStatus = cublasZdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  blasStatus = cublasZdrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, float* a, float* b, float* c, float* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srotg(rocblas_handle handle, float* a, float* b, float* c, float* s);
  // CHECK: blasStatus = rocblas_srotg(blasHandle, &fa, &fb, &fc, &fs);
  // CHECK-NEXT: blasStatus = rocblas_srotg(blasHandle, &fa, &fb, &fc, &fs);
  blasStatus = cublasSrotg(blasHandle, &fa, &fb, &fc, &fs);
  blasStatus = cublasSrotg_v2(blasHandle, &fa, &fb, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, double* a, double* b, double* c, double* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drotg(rocblas_handle handle, double* a, double* b, double* c, double* s);
  // CHECK: blasStatus = rocblas_drotg(blasHandle, &da, &db, &dc, &ds);
  // CHECK-NEXT: blasStatus = rocblas_drotg(blasHandle, &da, &db, &dc, &ds);
  blasStatus = cublasDrotg(blasHandle, &da, &db, &dc, &ds);
  blasStatus = cublasDrotg_v2(blasHandle, &da, &db, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_crotg(rocblas_handle handle, rocblas_float_complex* a, rocblas_float_complex* b, float* c, rocblas_float_complex* s);
  // CHECK: blasStatus = rocblas_crotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  // CHECK-NEXT: blasStatus = rocblas_crotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zrotg(rocblas_handle handle, rocblas_double_complex* a, rocblas_double_complex* b, double* c, rocblas_double_complex* s);
  // CHECK: blasStatus = rocblas_zrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = rocblas_zrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  blasStatus = cublasZrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  blasStatus = cublasZrotg_v2(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srotm(rocblas_handle handle, rocblas_int n, float* x, rocblas_int incx, float* y, rocblas_int incy, const float* param);
  // CHECK: blasStatus = rocblas_srotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_srotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSrotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSrotm_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drotm(rocblas_handle handle, rocblas_int n, double* x, rocblas_int incx, double* y, rocblas_int incy, const double* param);
  // CHECK: blasStatus = rocblas_drotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_drotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDrotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDrotm_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srotmg(rocblas_handle handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // CHECK: blasStatus = rocblas_srotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_srotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  blasStatus = cublasSrotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  blasStatus = cublasSrotmg_v2(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drotmg(rocblas_handle handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // CHECK: blasStatus = rocblas_drotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_drotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  blasStatus = cublasDrotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  blasStatus = cublasDrotmg_v2(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);

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

  // CHECK: rocblas_datatype DataType_2, DataType_3;
  cudaDataType DataType_2, DataType_3;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_nrm2_ex(rocblas_handle handle, rocblas_int n, const void* x, rocblas_datatype x_type, rocblas_int incx, void* results, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_nrm2_ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);
  blasStatus = cublasNrm2Ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);
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
