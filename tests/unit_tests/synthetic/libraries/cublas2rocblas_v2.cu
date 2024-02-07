// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --skip-excluded-preprocessor-conditional-blocks --experimental --roc %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocblas.h"
// CHECK-NOT: #include "cublas_v2.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "rocblas.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("16.v2. cuBLAS API to hipBLAS API synthetic test\n");

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
  // CHECK-NEXT: rocblas_status BLAS_STATUS_INVALID_VALUE = rocblas_status_invalid_value;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_MAPPING_ERROR = rocblas_status_invalid_size;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_EXECUTION_FAILED = rocblas_status_memory_error;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_INTERNAL_ERROR = rocblas_status_internal_error;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_NOT_SUPPORTED = rocblas_status_perf_degraded;
  // CHECK-NEXT: rocblas_status BLAS_STATUS_ARCH_MISMATCH = rocblas_status_arch_mismatch;
  cublasStatus_t blasStatus;
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

  // CHECK: rocblas_handle blasHandle;
  cublasHandle_t blasHandle;

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
  int64_t n_64 = 0;
  int nrhs = 0;
  int m = 0;
  int num = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int res = 0;
  int64_t res_64 = 0;
  int incx = 0;
  int64_t incx_64 = 0;
  int incy = 0;
  int64_t incy_64 = 0;
  int k = 0;
  int kl = 0;
  int ku = 0;
  int batchCount = 0;
  void *image = nullptr;
  void *image_2 = nullptr;
  void *valpha = nullptr;
  void *vc = nullptr;
  void *vs = nullptr;
  void *vx = nullptr;
  void *vy = nullptr;
  void *vresult = nullptr;

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
  float fparam = 0;

  float** fAarray = 0;
  const float** const fAarray_const = const_cast<const float**>(fAarray);
  float** fBarray = 0;
  const float** const fBarray_const = const_cast<const float**>(fBarray);
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
  double dparam = 0;

  double** dAarray = 0;
  const double** const dAarray_const = const_cast<const double**>(dAarray);
  double** dBarray = 0;
  const double** const dBarray_const = const_cast<const double**>(dBarray);
  double** dCarray = 0;
  double** dTauarray = 0;

  void** voidAarray = nullptr;
  const void** const voidAarray_const = const_cast<const void**>(voidAarray);
  void** voidBarray = nullptr;
  const void** const voidBarray_const = const_cast<const void**>(voidBarray);
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

  // CHECK: rocblas_float_complex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb, complexresult;
  cuComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb, complexresult;
  // CHECK: rocblas_double_complex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb, dcomplexresult;
  cuDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb, dcomplexresult;

  // CHECK: rocblas_float_complex** complexAarray = 0;
  // CHECK: const rocblas_float_complex** const complexAarray_const = const_cast<const rocblas_float_complex**>(complexAarray);
  // CHECK-NEXT: rocblas_float_complex** complexBarray = 0;
  // CHECK: const rocblas_float_complex** const complexBarray_const = const_cast<const rocblas_float_complex**>(complexBarray);
  // CHECK-NEXT: rocblas_float_complex** complexCarray = 0;
  // CHECK-NEXT: rocblas_float_complex** complexTauarray = 0;
  cuComplex** complexAarray = 0;
  const cuComplex** const complexAarray_const = const_cast<const cuComplex**>(complexAarray);
  cuComplex** complexBarray = 0;
  const cuComplex** const complexBarray_const = const_cast<const cuComplex**>(complexBarray);
  cuComplex** complexCarray = 0;
  cuComplex** complexTauarray = 0;

  // CHECK: rocblas_double_complex** dcomplexAarray = 0;
  // CHECK: const rocblas_double_complex** const dcomplexAarray_const = const_cast<const rocblas_double_complex**>(dcomplexAarray);
  // CHECK-NEXT: rocblas_double_complex** dcomplexBarray = 0;
  // CHECK: const rocblas_double_complex** const dcomplexBarray_const = const_cast<const rocblas_double_complex**>(dcomplexBarray);
  // CHECK-NEXT: rocblas_double_complex** dcomplexCarray = 0;
  // CHECK-NEXT: rocblas_double_complex** dcomplexTauarray = 0;
  cuDoubleComplex** dcomplexAarray = 0;
  const cuDoubleComplex** const dcomplexAarray_const = const_cast<const cuDoubleComplex**>(dcomplexAarray);
  cuDoubleComplex** dcomplexBarray = 0;
  const cuDoubleComplex** const dcomplexBarray_const = const_cast<const cuDoubleComplex**>(dcomplexBarray);
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

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, const float* x, rocblas_int incx, const float* beta, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_sgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_sgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgemv_v2(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, const double* x, rocblas_int incx, const double* beta, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgemv_v2(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_cgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_cgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgemv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgemv_v2(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, const float* alpha, const float* A, rocblas_int lda, const float* x, rocblas_int incx, const float* beta, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_sgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_sgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, const double* alpha, const double* A, rocblas_int lda, const double* x, rocblas_int incx, const double* beta, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_cgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_cgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgbmv(rocblas_handle handle, rocblas_operation trans, rocblas_int m, rocblas_int n, rocblas_int kl, rocblas_int ku, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_strmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const float* A, rocblas_int lda, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_strmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_strmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const double* A, rocblas_int lda, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztrmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_stbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int k, const float* A, rocblas_int lda, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_stbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_stbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int k, const double* A, rocblas_int lda, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int k, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, rocblas_int m, rocblas_int k, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_stpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const float* A, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_stpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_stpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const double* A, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex* A, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex* A, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_strsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const float* A, rocblas_int lda, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_strsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_strsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const double* A, rocblas_int lda, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztrsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_stpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, const float* AP, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_stpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_stpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, const double* AP, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, const rocblas_float_complex* AP, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztpsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, const rocblas_double_complex* AP, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_stbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, const float* A, rocblas_int lda, float* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_stbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = rocblas_stbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, const double* A, rocblas_int lda, double* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_dtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = rocblas_dtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ctbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ctbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztbsv(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int n, rocblas_int k, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* x, rocblas_int incx);
  // CHECK: blasStatus = rocblas_ztbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = rocblas_ztbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, const float* x, rocblas_int incx, const float* beta, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_ssymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_ssymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsymv_v2(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, const double* x, rocblas_int incx, const double* beta, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsymv_v2(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_csymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_csymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsymv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_chemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_chemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhemv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhemv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, const float* x, rocblas_int incx, const float* beta, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_ssbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_ssbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsbmv_v2(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, const double* x, rocblas_int incx, const double* beta, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsbmv_v2(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_chbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_chbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhbmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhbmv_v2(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* A, const float* x, rocblas_int incx, const float* beta, float* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_sspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = rocblas_sspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSspmv_v2(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dspmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* A, const double* x, rocblas_int incx, const double* beta, double* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_dspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = rocblas_dspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDspmv_v2(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* AP, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* beta, rocblas_float_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_chpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_chpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhpmv(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* AP, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* beta, rocblas_double_complex* y, rocblas_int incy);
  // CHECK: blasStatus = rocblas_zhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = rocblas_zhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sger(rocblas_handle handle, rocblas_int m, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_sger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  // CHECK-NEXT: blasStatus = rocblas_sger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSger_v2(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dger(rocblas_handle handle, rocblas_int m, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_dger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  // CHECK-NEXT: blasStatus = rocblas_dger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDger_v2(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_cgeru(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_cgeru(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgeru(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgeru_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_cgerc(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_cgerc(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgerc(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgerc_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgeru(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zgeru(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zgeru(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgeru(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgeru_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgerc(rocblas_handle handle, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zgerc(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zgerc(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgerc(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgerc_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, float* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_ssyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  // CHECK-NEXT: blasStatus = rocblas_ssyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  blasStatus = cublasSsyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  blasStatus = cublasSsyr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, double* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_dsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  // CHECK-NEXT: blasStatus = rocblas_dsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  blasStatus = cublasDsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  blasStatus = cublasDsyr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_csyr(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_csyr(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCsyr(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCsyr_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsyr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zsyr(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zsyr(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZsyr(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZsyr_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_cher(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_cher(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCher(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCher_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zher(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zher(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zher(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZher(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZher_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, float* AP);
  // CHECK: blasStatus = rocblas_sspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  // CHECK-NEXT: blasStatus = rocblas_sspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  blasStatus = cublasSspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  blasStatus = cublasSspr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dspr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, double* AP);
  // CHECK: blasStatus = rocblas_dspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  // CHECK-NEXT: blasStatus = rocblas_dspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  blasStatus = cublasDspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  blasStatus = cublasDspr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* AP);
  // CHECK: blasStatus = rocblas_chpr(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  // CHECK-NEXT: blasStatus = rocblas_chpr(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  blasStatus = cublasChpr(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  blasStatus = cublasChpr_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhpr(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* AP);
  // CHECK: blasStatus = rocblas_zhpr(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  // CHECK-NEXT: blasStatus = rocblas_zhpr(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  blasStatus = cublasZhpr(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  blasStatus = cublasZhpr_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_ssyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  // CHECK-NEXT: blasStatus = rocblas_ssyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSsyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSsyr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_dsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  // CHECK-NEXT: blasStatus = rocblas_dsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDsyr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_csyr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_csyr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCsyr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCsyr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsyr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zsyr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zsyr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZsyr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZsyr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_cher2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_cher2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCher2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCher2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zher2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* A, rocblas_int lda);
  // CHECK: blasStatus = rocblas_zher2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = rocblas_zher2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZher2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZher2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const float* alpha, const float* x, rocblas_int incx, const float* y, rocblas_int incy, float* AP);
  // CHECK: blasStatus = rocblas_sspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  // CHECK-NEXT: blasStatus = rocblas_sspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  blasStatus = cublasSspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  blasStatus = cublasSspr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dspr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const double* alpha, const double* x, rocblas_int incx, const double* y, rocblas_int incy, double* AP);
  // CHECK: blasStatus = rocblas_dspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  // CHECK-NEXT: blasStatus = rocblas_dspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  blasStatus = cublasDspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  blasStatus = cublasDspr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, rocblas_int incx, const rocblas_float_complex* y, rocblas_int incy, rocblas_float_complex* AP);
  // CHECK: blasStatus = rocblas_chpr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  // CHECK-NEXT: blasStatus = rocblas_chpr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  blasStatus = cublasChpr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  blasStatus = cublasChpr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhpr2(rocblas_handle handle, rocblas_fill uplo, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, rocblas_int incx, const rocblas_double_complex* y, rocblas_int incy, rocblas_double_complex* AP);
  // CHECK: blasStatus = rocblas_zhpr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  // CHECK-NEXT: blasStatus = rocblas_zhpr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  blasStatus = cublasZhpr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  blasStatus = cublasZhpr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);

  // CHECK rocblas_operation transa, transb;
  cublasOperation_t transa, transb;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_sgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_sgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSgemm_v2(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_dgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDgemm_v2(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cgemm(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_cgemm(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCgemm(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCgemm_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zgemm(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zgemm(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZgemm(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZgemm_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha, const float* const A[], rocblas_int lda, const float* const B[], rocblas_int ldb, const float* beta, float* const C[], rocblas_int ldc, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_sgemm_batched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);
  blasStatus = cublasSgemmBatched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* const A[], rocblas_int lda, const double* const B[], rocblas_int ldb, const double* beta, double* const C[], rocblas_int ldc, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_dgemm_batched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);
  blasStatus = cublasDgemmBatched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* const A[], rocblas_int lda, const rocblas_float_complex* const B[], rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* const C[], rocblas_int ldc, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_cgemm_batched(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);
  blasStatus = cublasCgemmBatched(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* const A[], rocblas_int lda, const rocblas_double_complex* const B[], rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* const C[], rocblas_int ldc, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_zgemm_batched(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);
  blasStatus = cublasZgemmBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, const float* beta, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ssyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_ssyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  blasStatus = cublasSsyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  blasStatus = cublasSsyrk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, const double* beta, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_dsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  blasStatus = cublasDsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  blasStatus = cublasDsyrk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_csyrk(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_csyrk(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrk(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrk_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsyrk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zsyrk(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zsyrk(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrk(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrk_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const float* alpha, const rocblas_float_complex* A, rocblas_int lda, const float* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cherk(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_cherk(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  blasStatus = cublasCherk(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  blasStatus = cublasCherk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zherk(rocblas_handle handle, rocblas_fill uplo, rocblas_operation transA, rocblas_int n, rocblas_int k, const double* alpha, const rocblas_double_complex* A, rocblas_int lda, const double* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zherk(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zherk(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  blasStatus = cublasZherk(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  blasStatus = cublasZherk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ssyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_ssyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_dsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_csyr2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_csyr2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyr2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsyr2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zsyr2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zsyr2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyr2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ssyrkx(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyrkx(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dsyrkx(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyrkx(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_csyrkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsyrkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zsyrkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const float* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cher2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_cher2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCher2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCher2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zher2k(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const double* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zher2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zher2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZher2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZher2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const float* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cherkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCherkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zherkx(rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const double* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zherkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZherkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ssymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, const float* B, rocblas_int ldb, const float* beta, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ssymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_ssymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, const double* B, rocblas_int ldb, const double* beta, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_dsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_csymm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_csymm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsymm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zsymm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zsymm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zsymm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsymm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_chemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_chemm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_chemm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasChemm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasChemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zhemm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zhemm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_zhemm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZhemm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZhemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_strsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, float* B, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_strsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  // CHECK-NEXT: blasStatus = rocblas_strsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  blasStatus = cublasStrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  blasStatus = cublasStrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, double* B, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_dtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  // CHECK-NEXT: blasStatus = rocblas_dtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  blasStatus = cublasDtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  blasStatus = cublasDtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, rocblas_float_complex* B, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_ctrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  // CHECK-NEXT: blasStatus = rocblas_ctrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  blasStatus = cublasCtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  blasStatus = cublasCtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztrsm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, rocblas_double_complex* B, rocblas_int ldb);
  // CHECK: blasStatus = rocblas_ztrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  // CHECK-NEXT: blasStatus = rocblas_ztrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  blasStatus = cublasZtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  blasStatus = cublasZtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_strmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, const float* B, rocblas_int ldb, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_strmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_strmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  blasStatus = cublasStrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  blasStatus = cublasStrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, const double* B, rocblas_int ldb, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_dtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  blasStatus = cublasDtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  blasStatus = cublasDtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* B, rocblas_int ldb, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ctrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_ctrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztrmm(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* B, rocblas_int ldb, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ztrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = rocblas_ztrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgeam(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, const float* alpha, const float* A, rocblas_int lda, const float* beta, const float* B, rocblas_int ldb, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_sgeam(blasHandle, transa, transb, m, n, &fa, &fA, lda, &fb, &fB, ldb, &fC, ldc);
  blasStatus = cublasSgeam(blasHandle, transa, transb, m, n, &fa, &fA, lda, &fb, &fB, ldb, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgeam(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, const double* alpha, const double* A, rocblas_int lda, const double* beta, const double* B, rocblas_int ldb, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_dgeam(blasHandle, transa, transb, m, n, &da, &dA, lda, &db, &dB, ldb, &dC, ldc);
  blasStatus = cublasDgeam(blasHandle, transa, transb, m, n, &da, &dA, lda, &db, &dB, ldb, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgeam(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* beta, const rocblas_float_complex* B, rocblas_int ldb, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cgeam(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCgeam(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgeam(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* beta, const rocblas_double_complex* B, rocblas_int ldb, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zgeam(blasHandle, transa, transb, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZgeam(blasHandle, transa, transb, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexB, ldb, &dcomplexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_strsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const float* alpha, const float* const A[], rocblas_int lda, float* const B[], rocblas_int ldb, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_strsm_batched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, fAarray_const, lda, fBarray, ldb, batchCount);
  blasStatus = cublasStrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, fAarray_const, lda, fBarray, ldb, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dtrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const double* alpha, const double* const A[], rocblas_int lda, double* const B[], rocblas_int ldb, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_dtrsm_batched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, dAarray_const, lda, dBarray, ldb, batchCount);
  blasStatus = cublasDtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, dAarray_const, lda, dBarray, ldb, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ctrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_float_complex* alpha, const rocblas_float_complex* const A[], rocblas_int lda, rocblas_float_complex* const B[], rocblas_int ldb, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_ctrsm_batched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);
  blasStatus = cublasCtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ztrsm_batched(rocblas_handle handle, rocblas_side side, rocblas_fill uplo, rocblas_operation transA, rocblas_diagonal diag, rocblas_int m, rocblas_int n, const rocblas_double_complex* alpha, const rocblas_double_complex* const A[], rocblas_int lda, rocblas_double_complex* const B[], rocblas_int ldb, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_ztrsm_batched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray, ldb, batchCount);
  blasStatus = cublasZtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray, ldb, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sdgmm(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, const float* A, rocblas_int lda, const float* x, rocblas_int incx, float* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_sdgmm(blasHandle, blasSideMode, m, n, &fa, lda, &fx, incx, &fC, ldc);
  blasStatus = cublasSdgmm(blasHandle, blasSideMode, m, n, &fa, lda, &fx, incx, &fC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ddgmm(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, const double* A, rocblas_int lda, const double* x, rocblas_int incx, double* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_ddgmm(blasHandle, blasSideMode, m, n, &da, lda, &dx, incx, &dC, ldc);
  blasStatus = cublasDdgmm(blasHandle, blasSideMode, m, n, &da, lda, &dx, incx, &dC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cdgmm(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, const rocblas_float_complex* A, rocblas_int lda, const rocblas_float_complex* x, rocblas_int incx, rocblas_float_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_cdgmm(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);
  blasStatus = cublasCdgmm(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdgmm(rocblas_handle handle, rocblas_side side, rocblas_int m, rocblas_int n, const rocblas_double_complex* A, rocblas_int lda, const rocblas_double_complex* x, rocblas_int incx, rocblas_double_complex* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_zdgmm(blasHandle, blasSideMode, m, n, &dcomplexa, lda, &dcomplexx, incx, &dcomplexC, ldc);
  blasStatus = cublasZdgmm(blasHandle, blasSideMode, m, n, &dcomplexa, lda, &dcomplexx, incx, &dcomplexC, ldc);

  long long int strideA = 0;
  long long int strideB = 0;
  long long int strideC = 0;

#if CUDA_VERSION >= 7050
  // CHECK: rocblas_half* ha = 0;
  __half* ha = 0;
  // CHECK: rocblas_half* hA = 0;
  __half* hA = 0;
  // CHECK: rocblas_half* hb = 0;
  __half* hb = 0;
  // CHECK: rocblas_half* hB = 0;
  __half* hB = 0;
  // CHECK: rocblas_half* hc = 0;
  __half* hc = 0;
  // CHECK: rocblas_half* hC = 0;
  __half* hC = 0;

  // CHECK: rocblas_half** hAarray = 0;
  __half** hAarray = 0;
  // CHECK: const rocblas_half** const hAarray_const = const_cast<const rocblas_half**>(hAarray);
  const __half** const hAarray_const = const_cast<const __half**>(hAarray);
  // CHECK: rocblas_half** hBarray = 0;
  __half** hBarray = 0;
  // CHECK: const rocblas_half** const hBarray_const = const_cast<const rocblas_half**>(hBarray);
  const __half** const hBarray_const = const_cast<const __half**>(hBarray);
  // CHECK: rocblas_half** hCarray = 0;
  __half** hCarray = 0;
  // CHECK: const rocblas_half** const hCarray_const = const_cast<const rocblas_half**>(hCarray);
  const __half** const hCarray_const = const_cast<const __half**>(hCarray);
  // CHECK: rocblas_half** hxarray = 0;
  __half** hxarray = 0;
  // CHECK: const rocblas_half** const hxarray_const = const_cast<const rocblas_half**>(hxarray_const);
  const __half** const hxarray_const = const_cast<const __half**>(hxarray_const);
  // CHECK: rocblas_half** hyarray = 0;
  __half** hyarray = 0;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, const __half* B, int ldb, const __half* beta, __half* C, int ldc);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_hgemm(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_half* alpha, const rocblas_half* A, rocblas_int lda, const rocblas_half* B, rocblas_int ldb, const rocblas_half* beta, rocblas_half* C, rocblas_int ldc);
  // CHECK: blasStatus = rocblas_hgemm(blasHandle, transa, transb, m, n, k, ha, hA, lda, hB, ldb, hb, hC, ldc);
  blasStatus = cublasHgemm(blasHandle, transa, transb, m, n, k, ha, hA, lda, hB, ldb, hb, hC, ldc);
#endif

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

  // CHECK: rocblas_datatype DataType_2, DataType_3, alpha_type, cs_type, x_type, y_type, execution_type, result_type;
  cudaDataType DataType_2, DataType_3, alpha_type, cs_type, x_type, y_type, execution_type, result_type;

  // CHECK: rocblas_gemm_algo blasGemmAlgo;
  // CHECK-NEXT: rocblas_gemm_algo BLAS_GEMM_DFALT = rocblas_gemm_algo_standard;
  cublasGemmAlgo_t blasGemmAlgo;
  cublasGemmAlgo_t BLAS_GEMM_DFALT = CUBLAS_GEMM_DFALT;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_nrm2_ex(rocblas_handle handle, rocblas_int n, const void* x, rocblas_datatype x_type, rocblas_int incx, void* results, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_nrm2_ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);
  blasStatus = cublasNrm2Ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const float* alpha, const float* A, rocblas_int lda, rocblas_stride stride_a, const float* B, rocblas_int ldb, rocblas_stride stride_b, const float* beta, float* C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_sgemm_strided_batched(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, strideA, &fB, ldb, strideB, &fb, &fC, ldc, strideC, batchCount);
  blasStatus = cublasSgemmStridedBatched(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, strideA, &fB, ldb, strideB, &fb, &fC, ldc, strideC, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const double* alpha, const double* A, rocblas_int lda, rocblas_stride stride_a, const double* B, rocblas_int ldb, rocblas_stride stride_b, const double* beta, double* C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_dgemm_strided_batched(blasHandle, transa, transb, m, n, k, &da, &dA, lda, strideA, &dB, ldb, strideB, &db, &dC, ldc, strideC, batchCount);
  blasStatus = cublasDgemmStridedBatched(blasHandle, transa, transb, m, n, k, &da, &dA, lda, strideA, &dB, ldb, strideB, &db, &dC, ldc, strideC, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_float_complex* alpha, const rocblas_float_complex* A, rocblas_int lda, rocblas_stride stride_a, const rocblas_float_complex* B, rocblas_int ldb, rocblas_stride stride_b, const rocblas_float_complex* beta, rocblas_float_complex* C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_cgemm_strided_batched(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);
  blasStatus = cublasCgemmStridedBatched(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_double_complex* alpha, const rocblas_double_complex* A, rocblas_int lda, rocblas_stride stride_a, const rocblas_double_complex* B, rocblas_int ldb, rocblas_stride stride_b, const rocblas_double_complex* beta, rocblas_double_complex* C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_zgemm_strided_batched(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);
  blasStatus = cublasZgemmStridedBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half* beta, __half* C, int ldc, long long int strideC, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_hgemm_strided_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_half* alpha, const rocblas_half* A, rocblas_int lda, rocblas_stride stride_a, const rocblas_half* B, rocblas_int ldb, rocblas_stride stride_b, const rocblas_half* beta, rocblas_half* C, rocblas_int ldc, rocblas_stride stride_c, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_hgemm_strided_batched(blasHandle, transa, transb, m, n, k, ha, hA, lda, strideA, hB, ldb, strideB, hb, hC, ldc, strideC, batchCount);
  blasStatus = cublasHgemmStridedBatched(blasHandle, transa, transb, m, n, k, ha, hA, lda, strideA, hB, ldb, strideB, hb, hC, ldc, strideC, batchCount);

  void* aptr = nullptr;
  void* Aptr = nullptr;
  void* bptr = nullptr;
  void* Bptr = nullptr;
  void* cptr = nullptr;
  void* Cptr = nullptr;
  void* xptr = nullptr;
  void* yptr = nullptr;
  void* sptr = nullptr;

  // CHECK: rocblas_datatype Atype;
  // CHECK-NEXT: rocblas_datatype Btype;
  // CHECK-NEXT: rocblas_datatype Ctype;
  // CHECK-NEXT: rocblas_datatype Xtype;
  // CHECK-NEXT: rocblas_datatype Ytype;
  // CHECK-NEXT: rocblas_datatype CStype;
  // CHECK-NEXT: rocblas_datatype Executiontype;
  cudaDataType Atype;
  cudaDataType Btype;
  cudaDataType Ctype;
  cudaDataType Xtype;
  cudaDataType Ytype;
  cudaDataType CStype;
  cudaDataType Executiontype;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scal_ex(rocblas_handle handle, rocblas_int n, const void* alpha, rocblas_datatype alpha_type, void* x, rocblas_datatype x_type, rocblas_int incx, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_scal_ex(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);
  blasStatus = cublasScalEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_axpy_ex(rocblas_handle handle, rocblas_int n, const void* alpha, rocblas_datatype alpha_type, const void* x, rocblas_datatype x_type, rocblas_int incx, void* y, rocblas_datatype y_type, rocblas_int incy, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_axpy_ex(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);
  blasStatus = cublasAxpyEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dot_ex(rocblas_handle handle, rocblas_int n, const void* x, rocblas_datatype x_type, rocblas_int incx, const void* y, rocblas_datatype y_type, rocblas_int incy, void* result, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_dot_ex(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dotc_ex(rocblas_handle handle, rocblas_int n, const void* x, rocblas_datatype x_type, rocblas_int incx, const void* y, rocblas_datatype y_type, rocblas_int incy, void* result, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_dotc_ex(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotcEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 11000
  // CHECK: rocblas_datatype computeType;
  cudaDataType computeType;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_gemm_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha, const void* a, rocblas_datatype a_type, rocblas_int lda, const void* b, rocblas_datatype b_type, rocblas_int ldb, const void* beta, const void* c, rocblas_datatype c_type, rocblas_int ldc, void* d, rocblas_datatype d_type, rocblas_int ldd, rocblas_datatype compute_type, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags);
  // CHECK: blasStatus = rocblas_gemm_ex(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
  blasStatus = cublasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 9000
  // CHECK: rocblas_gemm_algo BLAS_GEMM_DEFAULT = rocblas_gemm_algo_standard;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* const Aarray[], int lda, const __half* const Barray[], int ldb, const __half* beta, __half* const Carray[], int ldc, int batchCount);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_hgemm_batched(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const rocblas_half* alpha, const rocblas_half* const A[], rocblas_int lda, const rocblas_half* const B[], rocblas_int ldb, const rocblas_half* beta, rocblas_half* const C[], rocblas_int ldc, rocblas_int batch_count);
  // CHECK: blasStatus = rocblas_hgemm_batched(blasHandle, transa, transb, m, n, k, ha, hAarray_const, lda, hBarray_const, ldb, hb, hCarray, ldc, batchCount);
  blasStatus = cublasHgemmBatched(blasHandle, transa, transb, m, n, k, ha, hAarray_const, lda, hBarray_const, ldb, hb, hCarray, ldc, batchCount);
#endif

#if CUDA_VERSION >= 9010 && CUDA_VERSION < 11000
  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_gemm_batched_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha, const void* a, rocblas_datatype a_type, rocblas_int lda, const void* b, rocblas_datatype b_type, rocblas_int ldb, const void* beta, const void* c, rocblas_datatype c_type, rocblas_int ldc, void* d, rocblas_datatype d_type, rocblas_int ldd, rocblas_int batch_count, rocblas_datatype compute_type, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags);
  // CHECK: blasStatus = rocblas_gemm_batched_ex(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_gemm_strided_batched_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB, rocblas_int m, rocblas_int n, rocblas_int k, const void* alpha, const void* a, rocblas_datatype a_type, rocblas_int lda, rocblas_stride stride_a, const void* b, rocblas_datatype b_type, rocblas_int ldb, rocblas_stride stride_b, const void* beta, const void* c, rocblas_datatype c_type, rocblas_int ldc, rocblas_stride stride_c, void* d, rocblas_datatype d_type, rocblas_int ldd, rocblas_stride stride_d, rocblas_int batch_count, rocblas_datatype compute_type, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags);
  // CHECK: blasStatus = rocblas_gemm_strided_batched_ex(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: rocblas_operation BLAS_OP_HERMITAN = rocblas_operation_conjugate_transpose;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;

  // CHECK: rocblas_fill BLAS_FILL_MODE_FULL = rocblas_fill_full;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;

  // TODO: #1281
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_rot_ex(rocblas_handle handle, rocblas_int n, void* x, rocblas_datatype x_type, rocblas_int incx, void* y, rocblas_datatype y_type, rocblas_int incy, const void* c, const void* s, rocblas_datatype cs_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_rot_ex(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
  blasStatus = cublasRotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: rocblas_datatype R_16BF = rocblas_datatype_bf16_r;
  // CHECK-NEXT: rocblas_datatype C_16BF = rocblas_datatype_bf16_c;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;
#endif

#if CUDA_VERSION >= 11040 && CUBLAS_VERSION >= 11600
  // CUDA: CUBLASAPI const char* CUBLASWINAPI cublasGetStatusString(cublasStatus_t status);
  // ROC: ROCBLAS_EXPORT const char* rocblas_status_to_string(rocblas_status status);
  // CHECK: const_ch = rocblas_status_to_string(blasStatus);
  const_ch = cublasGetStatusString(blasStatus);
#endif

#if CUDA_VERSION >= 12000
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_isamax_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_isamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_isamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamax_v2_64(blasHandle, n_64, &fx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_idamax_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_idamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_idamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamax_v2_64(blasHandle, n_64, &dx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_icamax_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_icamax_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_icamax_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamax_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamax_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_izamax_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_izamax_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_izamax_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamax_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamax_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_isamin_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_isamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_isamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamin_v2_64(blasHandle, n_64, &fx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_idamin_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_idamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_idamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamin_v2_64(blasHandle, n_64, &dx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_icamin_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_icamin_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_icamin_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamin_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamin_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_izamin_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = rocblas_izamin_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = rocblas_izamin_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamin_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamin_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sasum_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, float* result);
  // CHECK: blasStatus = rocblas_sasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_sasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSasum_v2_64(blasHandle, n_64, &fx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dasum_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, double* result);
  // CHECK: blasStatus = rocblas_dasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDasum_v2_64(blasHandle, n_64, &dx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scasum_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, float* result);
  // CHECK: blasStatus = rocblas_scasum_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_scasum_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScasum_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScasum_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dzasum_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, double* result);
  // CHECK: blasStatus = rocblas_dzasum_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dzasum_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDzasum_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDzasum_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_saxpy_64(rocblas_handle handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = rocblas_saxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_saxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSaxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSaxpy_v2_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_daxpy_64(rocblas_handle handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = rocblas_daxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_daxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDaxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDaxpy_v2_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_caxpy_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* alpha, const rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_caxpy_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_caxpy_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCaxpy_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCaxpy_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zaxpy_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* alpha, const rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_zaxpy_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_zaxpy_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZaxpy_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZaxpy_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scopy_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = rocblas_scopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_scopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasScopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasScopy_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dcopy_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = rocblas_dcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_dcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDcopy_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ccopy_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_ccopy_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_ccopy_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCcopy_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCcopy_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zcopy_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_zcopy_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_zcopy_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZcopy_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZcopy_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sdot_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);
  // CHECK: blasStatus = rocblas_sdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_sdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  blasStatus = cublasSdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  blasStatus = cublasSdot_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_ddot_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);
  // CHECK: blasStatus = rocblas_ddot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_ddot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  blasStatus = cublasDdot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  blasStatus = cublasDdot_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cdotc_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, const rocblas_float_complex* y, int64_t incy, rocblas_float_complex* result);
  // CHECK: blasStatus = rocblas_cdotc_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  // CHECK-NEXT: blasStatus = rocblas_cdotc_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotc_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotc_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cdotu_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, const rocblas_float_complex* y, int64_t incy, rocblas_float_complex* result);
  // CHECK: blasStatus = rocblas_cdotu_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  // CHECK-NEXT: blasStatus = rocblas_cdotu_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotu_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotu_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdotc_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, const rocblas_double_complex* y, int64_t incy, rocblas_double_complex* result);
  // CHECK: blasStatus = rocblas_zdotc_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  // CHECK-NEXT: blasStatus = rocblas_zdotc_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotc_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotc_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdotu_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, const rocblas_double_complex* y, int64_t incy, rocblas_double_complex* result);
  // CHECK: blasStatus = rocblas_zdotu_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  // CHECK-NEXT: blasStatus = rocblas_zdotu_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotu_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotu_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_snrm2_64(rocblas_handle handle, int64_t n, const float* x, int64_t incx, float* result);
  // CHECK: blasStatus = rocblas_snrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_snrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSnrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSnrm2_v2_64(blasHandle, n_64, &fx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dnrm2_64(rocblas_handle handle, int64_t n, const double* x, int64_t incx, double* result);
  // CHECK: blasStatus = rocblas_dnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDnrm2_v2_64(blasHandle, n_64, &dx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scnrm2_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* x, int64_t incx, float* result);
  // CHECK: blasStatus = rocblas_scnrm2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = rocblas_scnrm2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScnrm2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScnrm2_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dznrm2_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* x, int64_t incx, double* result);
  // CHECK: blasStatus = rocblas_dznrm2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = rocblas_dznrm2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDznrm2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDznrm2_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srot_64(rocblas_handle handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s);
  // CHECK: blasStatus = rocblas_srot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  // CHECK-NEXT: blasStatus = rocblas_srot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  blasStatus = cublasSrot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  blasStatus = cublasSrot_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drot_64(rocblas_handle handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s);
  // CHECK: blasStatus = rocblas_drot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  // CHECK-NEXT: blasStatus = rocblas_drot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  blasStatus = cublasDrot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  blasStatus = cublasDrot_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_crot_64(rocblas_handle handle, int64_t n, rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy, const float* c, const rocblas_float_complex* s);
  // CHECK: blasStatus = rocblas_crot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  // CHECK-NEXT: blasStatus = rocblas_crot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  blasStatus = cublasCrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  blasStatus = cublasCrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csrot_64(rocblas_handle handle, int64_t n, rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy, const float* c, const float* s);
  // CHECK: blasStatus = rocblas_csrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  // CHECK-NEXT: blasStatus = rocblas_csrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  blasStatus = cublasCsrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  blasStatus = cublasCsrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zrot_64(rocblas_handle handle, int64_t n, rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy, const double* c, const rocblas_double_complex* s);
  // CHECK: blasStatus = rocblas_zrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = rocblas_zrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  blasStatus = cublasZrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdrot_64(rocblas_handle handle, int64_t n, rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy, const double* c, const double* s);
  // CHECK: blasStatus = rocblas_zdrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  // CHECK-NEXT: blasStatus = rocblas_zdrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  blasStatus = cublasZdrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  blasStatus = cublasZdrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_srotm_64(rocblas_handle handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param);
  // CHECK: blasStatus = rocblas_srotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  // CHECK-NEXT: blasStatus = rocblas_srotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  blasStatus = cublasSrotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  blasStatus = cublasSrotm_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_drotm_64(rocblas_handle handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param);
  // CHECK: blasStatus = rocblas_drotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  // CHECK-NEXT: blasStatus = rocblas_drotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  blasStatus = cublasDrotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  blasStatus = cublasDrotm_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sscal_64(rocblas_handle handle, int64_t n, const float* alpha, float* x, int64_t incx);
  // CHECK: blasStatus = rocblas_sscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_sscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  blasStatus = cublasSscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  blasStatus = cublasSscal_v2_64(blasHandle, n_64, &fa, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dscal_64(rocblas_handle handle, int64_t n, const double* alpha, double* x, int64_t incx);
  // CHECK: blasStatus = rocblas_dscal_64(blasHandle, n_64, &da, &dx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_dscal_64(blasHandle, n_64, &da, &dx, incx_64);
  blasStatus = cublasDscal_64(blasHandle, n_64, &da, &dx, incx_64);
  blasStatus = cublasDscal_v2_64(blasHandle, n_64, &da, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cscal_64(rocblas_handle handle, int64_t n, const rocblas_float_complex* alpha, rocblas_float_complex* x, int64_t incx);
  // CHECK: blasStatus = rocblas_cscal_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_cscal_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  blasStatus = cublasCscal_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  blasStatus = cublasCscal_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_csscal_64(rocblas_handle handle, int64_t n, const float* alpha, rocblas_float_complex* x, int64_t incx);
  // CHECK: blasStatus = rocblas_csscal_64(blasHandle, n_64, &fa, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_csscal_64(blasHandle, n_64, &fa, &complexx, incx_64);
  blasStatus = cublasCsscal_64(blasHandle, n_64, &fa, &complexx, incx_64);
  blasStatus = cublasCsscal_v2_64(blasHandle, n_64, &fa, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zscal_64(rocblas_handle handle, int64_t n, const rocblas_double_complex* alpha, rocblas_double_complex* x, int64_t incx);
  // CHECK: blasStatus = rocblas_zscal_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_zscal_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  blasStatus = cublasZscal_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  blasStatus = cublasZscal_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zdscal_64(rocblas_handle handle, int64_t n, const double* alpha, rocblas_double_complex* x, int64_t incx);
  // CHECK: blasStatus = rocblas_zdscal_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = rocblas_zdscal_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  blasStatus = cublasZdscal_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  blasStatus = cublasZdscal_v2_64(blasHandle, n_64, &da, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_sswap_64(rocblas_handle handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = rocblas_sswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_sswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSswap_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dswap_64(rocblas_handle handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = rocblas_dswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_dswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDswap_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_cswap_64(rocblas_handle handle, int64_t n, rocblas_float_complex* x, int64_t incx, rocblas_float_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_cswap_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_cswap_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCswap_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCswap_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_zswap_64(rocblas_handle handle, int64_t n, rocblas_double_complex* x, int64_t incx, rocblas_double_complex* y, int64_t incy);
  // CHECK: blasStatus = rocblas_zswap_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = rocblas_zswap_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZswap_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZswap_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_axpy_ex_64(rocblas_handle handle, int64_t n, const void* alpha, rocblas_datatype alpha_type, const void* x, rocblas_datatype x_type, int64_t incx, void* y, rocblas_datatype y_type, int64_t incy, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_axpy_ex_64(blasHandle, n_64, valpha, alpha_type, vx, x_type, incx_64, vy, y_type, incy_64, execution_type);
  blasStatus = cublasAxpyEx_64(blasHandle, n_64, valpha, alpha_type, vx, x_type, incx_64, vy, y_type, incy_64, execution_type);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dot_ex_64(rocblas_handle handle, int64_t n, const void* x, rocblas_datatype x_type, int64_t incx, const void* y, rocblas_datatype y_type, int64_t incy, void* result, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_dot_ex_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vresult, result_type, execution_type);
  blasStatus = cublasDotEx_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vresult, result_type, execution_type);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_dotc_ex_64(rocblas_handle handle, int64_t n, const void* x, rocblas_datatype x_type, int64_t incx, const void* y, rocblas_datatype y_type, int64_t incy, void* result, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_dotc_ex_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vresult, result_type, execution_type);
  blasStatus = cublasDotcEx_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vresult, result_type, execution_type);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_nrm2_ex_64(rocblas_handle handle, int64_t n, const void* x, rocblas_datatype x_type, int64_t incx, void* results, rocblas_datatype result_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_nrm2_ex_64(blasHandle, n_64, vx, x_type, incx_64, vresult, result_type, execution_type);
  blasStatus = cublasNrm2Ex_64(blasHandle, n_64, vx, x_type, incx_64, vresult, result_type, execution_type);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_rot_ex_64(rocblas_handle handle, int64_t n, void* x, rocblas_datatype x_type, int64_t incx, void* y, rocblas_datatype y_type, int64_t incy, const void* c, const void* s, rocblas_datatype cs_type, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_rot_ex_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vc, vs, cs_type, execution_type);
  blasStatus = cublasRotEx_64(blasHandle, n_64, vx, x_type, incx_64, vy, y_type, incy_64, vc, vs, cs_type, execution_type);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType);
  // ROC: ROCBLAS_EXPORT rocblas_status rocblas_scal_ex_64(rocblas_handle handle, int64_t n, const void* alpha, rocblas_datatype alpha_type, void* x, rocblas_datatype x_type, int64_t incx, rocblas_datatype execution_type);
  // CHECK: blasStatus = rocblas_scal_ex_64(blasHandle, n_64, valpha, alpha_type, vx, x_type, incx_64, execution_type);
  blasStatus = cublasScalEx_64(blasHandle, n_64, valpha, alpha_type, vx, x_type, incx_64, execution_type);
#endif

  return 0;
}
