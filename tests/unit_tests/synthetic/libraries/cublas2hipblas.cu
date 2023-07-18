// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblas.h"
#include "cublas.h"
// CHECK-NOT: #include "hipblas.h"

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

  // CHECK: hipblasHandle_t blasHandle;
  cublasHandle_t blasHandle;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t* mode);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t* atomics_mode);
  // CHECK: blasStatus = hipblasGetAtomicsMode(blasHandle, &blasAtomicsMode);
  blasStatus = cublasGetAtomicsMode(blasHandle, &blasAtomicsMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetAtomicsMode(hipblasHandle_t handle, hipblasAtomicsMode_t atomics_mode);
  // CHECK: blasStatus = hipblasSetAtomicsMode(blasHandle, blasAtomicsMode);
  blasStatus = cublasSetAtomicsMode(blasHandle, blasAtomicsMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t* handle);
  // CUDA: #define cublasCreate cublasCreate_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);
  // CHECK: blasStatus = hipblasCreate(&blasHandle);
  blasStatus = cublasCreate_v2(&blasHandle);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle);
  // CUDA: #define cublasDestroy cublasDestroy_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
  // CHECK: blasStatus = hipblasDestroy(blasHandle);
  blasStatus = cublasDestroy_v2(blasHandle);

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
  // CUDA: #define cublasSetStream cublasSetStream_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);
  // CHECK: blasStatus = hipblasSetStream(blasHandle, stream);
  blasStatus = cublasSetStream_v2(blasHandle, stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
  // CUDA: #define cublasGetStream cublasGetStream_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId);
  // CHECK: blasStatus = hipblasGetStream(blasHandle, &stream);
  blasStatus = cublasGetStream_v2(blasHandle, &stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
  // CUDA: #define cublasSetPointerMode cublasSetPointerMode_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode);
  // CHECK: blasStatus = hipblasSetPointerMode(blasHandle, blasPointerMode);
  blasStatus = cublasSetPointerMode_v2(blasHandle, blasPointerMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
  // CUDA: #define cublasGetPointerMode cublasGetPointerMode_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode);
  // CHECK: blasStatus = hipblasGetPointerMode(blasHandle, &blasPointerMode);
  blasStatus = cublasGetPointerMode_v2(blasHandle, &blasPointerMode);

  int n = 0;
  int nrhs = 0;
  int m = 0;
  int num = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  int res = 0;
  int incx = 0;
  int incy = 0;
  int k = 0;
  int kl = 0;
  int ku = 0;
  int batchCount = 0;
  int P = 0;
  int info = 0;
  void* image = nullptr;
  void* image_2 = nullptr;
  void* deviceptr = nullptr;

  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetVector(int n, int elemSize, const void* x, int incx, void* devicePtr, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
  // CHECK: blasStatus = hipblasSetVector(n, num, image, incx, image_2, incy);
  blasStatus = cublasSetVector(n, num, image, incx, image_2, incy);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy);
  // CHECK: blasStatus = hipblasGetVector(n, num, image, incx, image_2, incy);
  blasStatus = cublasGetVector(n, num, image, incx, image_2, incy);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetVectorAsync(int n, int elemSize, const void* hostPtr, int incx, void* devicePtr, int incy, cudaStream_t stream);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetVectorAsync(int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);
  // CHECK: blasStatus = hipblasSetVectorAsync(n, num, image, incx, image_2, incy, stream);
  blasStatus = cublasSetVectorAsync(n, num, image, incx, image_2, incy, stream);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetVectorAsync(int n, int elemSize, const void* devicePtr, int incx, void* hostPtr, int incy, cudaStream_t stream);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetVectorAsync(int n, int elemSize, const void* x, int incx, void* y, int incy, hipStream_t stream);
  // CHECK: blasStatus = hipblasGetVectorAsync(n, num, image, incx, image_2, incy, stream);
  blasStatus = cublasGetVectorAsync(n, num, image, incx, image_2, incy, stream);

  int rows = 0;
  int cols = 0;

  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetMatrix(int rows, int cols, int elemSize, const void* AP, int lda, void* BP, int ldb);
  // CHECK: blasStatus = hipblasSetMatrix(rows, cols, num, image, incx, image_2, incy);
  blasStatus = cublasSetMatrix(rows, cols, num, image, incx, image_2, incy);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetMatrix(int rows, int cols, int elemSize, const void* AP, int lda, void* BP, int ldb);
  // CHECK: blasStatus = hipblasGetMatrix(rows, cols, num, image, incx, image_2, incy);
  blasStatus = cublasGetMatrix(rows, cols, num, image, incx, image_2, incy);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetMatrixAsync(int rows, int cols, int elemSize, const void* AP, int lda, void* BP, int ldb, hipStream_t stream);
  // CHECK: blasStatus = hipblasSetMatrixAsync(rows, cols, num, image, incx, image_2, incy, stream);
  blasStatus = cublasSetMatrixAsync(rows, cols, num, image, incx, image_2, incy, stream);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda, void* B, int ldb, cudaStream_t stream);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetMatrixAsync(int rows, int cols, int elemSize, const void* AP, int lda, void* BP, int ldb, hipStream_t stream);
  // CHECK: blasStatus = hipblasGetMatrixAsync(rows, cols, num, image, incx, image_2, incy, stream);
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

  // NOTE: float CUBLASWINAPI cublasSnrm2(int n, const float* x, int incx) is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result);
  // CHECK: blasStatus = hipblasSnrm2(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSnrm2_v2(blasHandle, n, &fx, incx, &fresult);

  // NOTE: double CUBLASWINAPI cublasDnrm2(int n, const double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDnrm2(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDnrm2_v2(blasHandle, n, &dx, incx, &dresult);

  // CHECK: hipComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb;
  cuComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb;
  // CHECK: hipDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb;
  cuDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb;

  // CHECK: hipComplex** complexAarray = 0;
  // CHECK: const hipComplex** const complexAarray_const = const_cast<const hipComplex**>(complexAarray);
  // CHECK-NEXT: hipComplex** complexBarray = 0;
  // CHECK: const hipComplex** const complexBarray_const = const_cast<const hipComplex**>(complexBarray);
  // CHECK-NEXT: hipComplex** complexCarray = 0;
  // CHECK-NEXT: hipComplex** complexTauarray = 0;
  cuComplex** complexAarray = 0;
  const cuComplex** const complexAarray_const = const_cast<const cuComplex**>(complexAarray);
  cuComplex** complexBarray = 0;
  const cuComplex** const complexBarray_const = const_cast<const cuComplex**>(complexBarray);
  cuComplex** complexCarray = 0;
  cuComplex** complexTauarray = 0;

  // CHECK: hipDoubleComplex** dcomplexAarray = 0;
  // CHECK: const hipDoubleComplex** const dcomplexAarray_const = const_cast<const hipDoubleComplex**>(dcomplexAarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexBarray = 0;
  // CHECK: const hipDoubleComplex** const dcomplexBarray_const = const_cast<const hipDoubleComplex**>(dcomplexBarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexCarray = 0;
  // CHECK-NEXT: hipDoubleComplex** dcomplexTauarray = 0;
  cuDoubleComplex** dcomplexAarray = 0;
  const cuDoubleComplex** const dcomplexAarray_const = const_cast<const cuDoubleComplex**>(dcomplexAarray);
  cuDoubleComplex** dcomplexBarray = 0;
  const cuDoubleComplex** const dcomplexBarray_const = const_cast<const cuDoubleComplex**>(dcomplexBarray);
  cuDoubleComplex** dcomplexCarray = 0;
  cuDoubleComplex** dcomplexTauarray = 0;

  // NOTE: float CUBLASWINAPI cublasScnrm2(int n, const cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);

  // NOTE: double CUBLASWINAPI cublasDznrm2(int n, const cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  blasStatus = cublasDznrm2_v2(blasHandle, n, &dcomplex, incx, &dresult);

  // NOTE: float CUBLASWINAPI cublasSdot(int n, const float* x, int incx, const float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
  // CHECK: blasStatus = hipblasSdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSdot_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // NOTE: double CUBLASWINAPI cublasDdot(int n, const double* x, int incx, const double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  // CHECK: blasStatus = hipblasDdot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDdot_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // NOTE: cuComplex CUBLASWINAPI cublasCdotu(int n, const cuComplex* x, int incx, const cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* result);
  // CHECK: blasStatus = hipblasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // NOTE: cuComplex CUBLASWINAPI cublasCdotc(int n, const cuComplex* x, int incx, const cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* result);
  // CHECK: blasStatus = hipblasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // NOTE: cuDoubleComplex CUBLASWINAPI cublasZdotu(int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // NOTE: cuDoubleComplex CUBLASWINAPI cublasZdotc(int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotc_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // NOTE: void CUBLASWINAPI cublasSscal(int n, float alpha, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx);
  // CHECK: blasStatus = hipblasSscal(blasHandle, n, &fy, &fx, incx);
  blasStatus = cublasSscal_v2(blasHandle, n, &fy, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDscal(int n, double alpha, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx);
  // CHECK: blasStatus = hipblasDscal(blasHandle, n, &dx, &dy, incx);
  blasStatus = cublasDscal_v2(blasHandle, n, &dx, &dy, incx);

  // NOTE: void CUBLASWINAPI cublasCscal(int n, cuComplex alpha, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCscal(hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal_v2(blasHandle, n, &complexa, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasCsscal(int n, float alpha, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCsscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal_v2(blasHandle, n, &fx, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZscal(int n, cuDoubleComplex alpha, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZscal(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasZdscal(int n, double alpha, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdscal(hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZdscal(blasHandle, n, &dx, &dcomplexx, incx);
  blasStatus = cublasZdscal_v2(blasHandle, n, &dx, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasSaxpy(int n, float alpha, const float* x, int incx, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasSaxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  blasStatus = cublasSaxpy_v2(blasHandle, n, &fa, &fx, incx, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDaxpy(int n, double alpha, const double* x, int incx, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDaxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  blasStatus = cublasDaxpy_v2(blasHandle, n, &da, &dx, incx, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasCaxpy(int n, cuComplex alpha, const cuComplex* x, int incx, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZaxpy(int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZaxpy_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasScopy(int n, const float* x, int incx, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasScopy(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasScopy_v2(blasHandle, n, &fx, incx, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDcopy(int n, const double* x, int incx, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDcopy(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDcopy_v2(blasHandle, n, &dx, incx, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasCcopy(int n, const cuComplex* x, int incx, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCcopy(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZcopy(int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZcopy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZcopy_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasSswap(int n, float* x, int incx, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasSswap(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasSswap_v2(blasHandle, n, &fx, incx, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDswap(int n, double* x, int incx, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDswap(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDswap_v2(blasHandle, n, &dx, incx, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasCswap(int n, cuComplex* x, int incx, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZswap(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZswap_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // NOTE: int CUBLASWINAPI cublasIsamax(int n, const float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIsamax(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamax_v2(blasHandle, n, &fx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIdamax(int n, const double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIdamax(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamax_v2(blasHandle, n, &dx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIcamax(int n, const cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax_v2(blasHandle, n, &complexx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIzamax(int n, const cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamax(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamax_v2(blasHandle, n, &dcomplexx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIsamin(int n, const float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIsamin(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamin_v2(blasHandle, n, &fx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIdamin(int n, const double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIdamin(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamin_v2(blasHandle, n, &dx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIcamin(int n, const cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin_v2(blasHandle, n, &complexx, incx, &res);

  // NOTE: int CUBLASWINAPI cublasIzamin(int n, const cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamin(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamin_v2(blasHandle, n, &dcomplexx, incx, &res);

  // NOTE: float CUBLASWINAPI cublasSasum(int n, const float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result);
  // CHECK: blasStatus = hipblasSasum(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSasum_v2(blasHandle, n, &fx, incx, &fresult);

  // NOTE: double CUBLASWINAPI cublasDasum(int n, const double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDasum(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDasum_v2(blasHandle, n, &dx, incx, &dresult);

  // NOTE: float CUBLASWINAPI cublasScasum(int n, const cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum_v2(blasHandle, n, &complexx, incx, &fresult);

  // NOTE: double CUBLASWINAPI cublasDzasum(int n, const cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  blasStatus = cublasDzasum_v2(blasHandle, n, &dcomplexx, incx, &dresult);

  // NOTE: void CUBLASWINAPI cublasSrot(int n, float* x, int incx, float* y, int incy, float sc, float ss); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrot(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasSrot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  blasStatus = cublasSrot_v2(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);

  // NOTE: void CUBLASWINAPI cublasDrot(int n, double* x, int incx, double* y, int incy, double sc, double ss); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrot(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasDrot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  blasStatus = cublasDrot_v2(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);

  // NOTE: void CUBLASWINAPI cublasCrot(int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, cuComplex s); is not supported by HIP
  // CUDA: CUBLASAPI CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrot(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, const float* c, const hipblasComplex* s);
  // CHECK: blasStatus = hipblasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);

  // NOTE: void CUBLASWINAPI cublasCsrot(int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, float s); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsrot(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);

  // NOTE: void CUBLASWINAPI cublasZrot(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double sc, cuDoubleComplex cs); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrot(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy, const double* c, const hipblasDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);

  // NOTE: void CUBLASWINAPI cublasZdrot(int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double c, double s); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdrot(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasZdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  blasStatus = cublasZdrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);

  // NOTE: void CUBLASWINAPI cublasSrotg(float* sa, float* sb, float* sc, float* ss); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, float* a, float* b, float* c, float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s);
  // CHECK: blasStatus = hipblasSrotg(blasHandle, &fa, &fb, &fc, &fs);
  blasStatus = cublasSrotg_v2(blasHandle, &fa, &fb, &fc, &fs);

  // NOTE: void CUBLASWINAPI cublasDrotg(double* sa, double* sb, double* sc, double* ss); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, double* a, double* b, double* c, double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s);
  // CHECK: blasStatus = hipblasDrotg(blasHandle, &da, &db, &dc, &ds);
  blasStatus = cublasDrotg_v2(blasHandle, &da, &db, &dc, &ds);

  // NOTE: void CUBLASWINAPI cublasCrotg(cuComplex* ca, cuComplex cb, float* sc, cuComplex* cs); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrotg(hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);
  // CHECK: blasStatus = hipblasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);

  // NOTE: void CUBLASWINAPI cublasZrotg(cuDoubleComplex* ca, cuDoubleComplex cb, double* sc, cuDoubleComplex* cs); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrotg(hipblasHandle_t handle, hipblasDoubleComplex* a, hipblasDoubleComplex* b, double* c, hipblasDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  blasStatus = cublasZrotg_v2(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);

  // NOTE: void CUBLASWINAPI cublasSrotm(int n, float* x, int incx, float* y, int incy, const float* sparam); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotm(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);
  // CHECK: blasStatus = hipblasSrotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSrotm_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // NOTE: void CUBLASWINAPI cublasDrotm(int n, double* x, int incx, double* y, int incy, const double* sparam); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotm(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);
  // CHECK: blasStatus = hipblasDrotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDrotm_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // NOTE: void CUBLASWINAPI cublasSrotmg(float* sd1, float* sd2, float* sx1, const float* sy1, float* sparam); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmg(hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // CHECK: blasStatus = hipblasSrotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  blasStatus = cublasSrotmg_v2(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);

  // NOTE: void CUBLASWINAPI cublasDrotmg(double* sd1, double* sd2, double* sx1, const double* sy1, double* sparam); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // CHECK: blasStatus = hipblasDrotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  blasStatus = cublasDrotmg_v2(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);

  // NOTE: void CUBLASWINAPI cublasSgemv(char trans, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgemv_v2(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDgemv(char trans, int m, int n, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgemv_v2(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasCgemv(char trans, int m, int n, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex beta, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZgemv(char trans, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgemv_v2(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasSgbmv(char trans, int m, int n, int kl, int ku, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDgbmv(char trans, int m, int n, int kl, int ku, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasCgbmv(char trans, int m, int n, int kl, int ku, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex beta, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZgbmv(char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasStrmv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtrmv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtrmv(char uplo, char trans, char diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtrmv(char uplo, char trans, char diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasStbmv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtbmv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtbmv(char uplo, char trans, char diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtbmv(char uplo, char trans, char diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasStpmv(char uplo, char trans, char diag, int n, const float* AP, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, float* x, int incx);
  // CHECK: blasStatus = hipblasStpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtpmv(char uplo, char trans, char diag, int n, const double* AP, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, double* x, int incx);
  // CHECK: blasStatus = hipblasDtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtpmv(char uplo, char trans, char diag, int n, const cuComplex* AP, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtpmv(char uplo, char trans, char diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasStrsv(char uplo, char trans, char diag, int n, const float* A, int lda, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtrsv(char uplo, char trans, char diag, int n, const double* A, int lda, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtrsv(char uplo, char trans, char diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtrsv(char uplo, char trans, char diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasStpsv(char uplo, char trans, char diag, int n, const float* AP, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, float* x, int incx);
  // CHECK: blasStatus = hipblasStpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtpsv(char uplo, char trans, char diag, int n, const double* AP, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, double* x, int incx);
  // CHECK: blasStatus = hipblasDtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtpsv(char uplo, char trans, char diag, int n, const cuComplex* AP, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtpsv(char uplo, char trans, char diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasStbsv(char uplo, char trans, char diag, int n, int k, const float* A, int lda, float* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // NOTE: void CUBLASWINAPI cublasDtbsv(char uplo, char trans, char diag, int n, int k, const double* A, int lda, double* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // NOTE: void CUBLASWINAPI cublasCtbsv(char uplo, char trans, char diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // NOTE: void CUBLASWINAPI cublasZtbsv(char uplo, char trans, char diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // NOTE: void CUBLASWINAPI cublasSsymv(char uplo, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSsymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsymv_v2(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDsymv(char uplo, int n, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsymv_v2(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasChemv(char uplo, int n, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex beta, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZhemv(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhemv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasSsbmv(char uplo, int n, int k, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSsbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsbmv_v2(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDsbmv(char uplo, int n, int k, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsbmv_v2(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasChbmv(char uplo, int n, int k, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex beta, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZhbmv(char uplo, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhbmv_v2(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasSspmv(char uplo, int n, float alpha, const float* AP, const float* x, int incx, float beta, float* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSspmv_v2(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);

  // NOTE: void CUBLASWINAPI cublasDspmv(char uplo, int n, double alpha, const double* AP, const double* x, int incx, double beta, double* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDspmv_v2(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);

  // NOTE: void CUBLASWINAPI cublasChpmv(char uplo, int n, cuComplex alpha, const cuComplex* AP, const cuComplex* x, int incx, cuComplex beta, cuComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);

  // NOTE: void CUBLASWINAPI cublasZhpmv(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // NOTE: void CUBLASWINAPI cublasSger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP, int lda);
  // CHECK: blasStatus = hipblasSger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSger_v2(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // NOTE: void CUBLASWINAPI cublasDger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP, int lda);
  // CHECK: blasStatus = hipblasDger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDger_v2(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // NOTE: void CUBLASWINAPI cublasCgeru(int m, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeru(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCgeru(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgeru_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // NOTE: void CUBLASWINAPI cublasCgerc(int m, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgerc(hipblasHandle_t handle, int m, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCgerc(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgerc_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // NOTE: void CUBLASWINAPI cublasZgeru(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeru(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZgeru(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgeru_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasZgerc(int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgerc(hipblasHandle_t handle, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZgerc(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgerc_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasSsyr(char uplo, int n, float alpha, const float* x, int incx, float* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP, int lda);
  // CHECK: blasStatus = hipblasSsyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  blasStatus = cublasSsyr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);

  // NOTE: void CUBLASWINAPI cublasDsyr(char uplo, int n, double alpha, const double* x, int incx, double* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP, int lda);
  // CHECK: blasStatus = hipblasDsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  blasStatus = cublasDsyr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCsyr(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCsyr_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZsyr(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZsyr_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasCher(char uplo, int n, float alpha, const cuComplex* x, int incx, cuComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const hipblasComplex* x, int incx, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCher(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCher_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);

  // NOTE: void CUBLASWINAPI cublasZher(char uplo, int n, double alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZher(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZher_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasSspr(char uplo, int n, float alpha, const float* x, int incx, float* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);
  // CHECK: blasStatus = hipblasSspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  blasStatus = cublasSspr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);

  // NOTE: void CUBLASWINAPI cublasDspr(char uplo, int n, double alpha, const double* x, int incx, double* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);
  // CHECK: blasStatus = hipblasDspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  blasStatus = cublasDspr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);

  // NOTE: void CUBLASWINAPI cublasChpr(char uplo, int n, float alpha, const cuComplex* x, int incx, cuComplex* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const hipblasComplex* x, int incx, hipblasComplex* AP);
  // CHECK: blasStatus = hipblasChpr(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  blasStatus = cublasChpr_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);

  // NOTE: void CUBLASWINAPI cublasZhpr(char uplo, int n, double alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  blasStatus = cublasZhpr_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);

  // NOTE: void CUBLASWINAPI cublasSsyr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP, int lda);
  // CHECK: blasStatus = hipblasSsyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSsyr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // NOTE: void CUBLASWINAPI cublasDsyr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP, int lda);
  // CHECK: blasStatus = hipblasDsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDsyr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCsyr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCsyr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZsyr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZsyr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasCher2(char uplo, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCher2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCher2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // NOTE: void CUBLASWINAPI cublasZher2(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZher2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZher2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // NOTE: void CUBLASWINAPI cublasSspr2(char uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);
  // CHECK: blasStatus = hipblasSspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  blasStatus = cublasSspr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);

  // NOTE: void CUBLASWINAPI cublasDspr2(char uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);
  // CHECK: blasStatus = hipblasDspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  blasStatus = cublasDspr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);

  // NOTE: void CUBLASWINAPI cublasChpr2(char uplo, int n, cuComplex alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* AP);
  // CHECK: blasStatus = hipblasChpr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  blasStatus = cublasChpr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);

  // NOTE: void CUBLASWINAPI cublasZhpr2(char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  blasStatus = cublasZhpr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);

  cublasOperation_t transa, transb;

  // NOTE: void CUBLASWINAPI cublasSgemm(char transa, char transb, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSgemm_v2(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // NOTE: void CUBLASWINAPI cublasDgemm(char transa, char transb, int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDgemm_v2(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // NOTE: void CUBLASWINAPI cublasCgemm(char transa, char transb, int m, int n, int k, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCgemm(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCgemm_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZgemm(char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZgemm(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZgemm_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // TODO: __half -> hipblasHalf
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, const __half* B, int ldb, const __half* beta, __half* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* AP, int lda, const hipblasHalf* BP, int ldb, const hipblasHalf* beta, hipblasHalf* CP, int ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* const AP[], int lda, const float* const BP[], int ldb, const float* beta, float* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasSgemmBatched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);
  blasStatus = cublasSgemmBatched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double* alpha, const double* const AP[], int lda, const double* const BP[], int ldb, const double* beta, double* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasDgemmBatched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);
  blasStatus = cublasDgemmBatched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);

  // TODO: __half -> hipblasHalf
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* const Aarray[], int lda, const __half* const Barray[], int ldb, const __half* beta, __half* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* const AP[], int lda, const hipblasHalf* const BP[], int ldb, const hipblasHalf* beta, hipblasHalf* const CP[], int ldc, int batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasComplex* alpha, const hipblasComplex* const AP[], int lda, const hipblasComplex* const BP[], int ldb, const hipblasComplex* beta, hipblasComplex* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasCgemmBatched(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);
  blasStatus = cublasCgemmBatched(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* const AP[], int lda, const hipblasDoubleComplex* const BP[], int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasZgemmBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);
  blasStatus = cublasZgemmBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);

  // NOTE: void CUBLASWINAPI cublasSsyrk(char uplo, char trans, int n, int k, float alpha, const float* A, int lda, float beta, float* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const float* AP, int lda, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  blasStatus = cublasSsyrk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);

  // NOTE: void CUBLASWINAPI cublasDsyrk(char uplo, char trans, int n, int k, double alpha, const double* A, int lda, double beta, double* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const double* AP, int lda, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  blasStatus = cublasDsyrk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);

  // NOTE: void CUBLASWINAPI cublasCsyrk(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex* A, int lda, cuComplex beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyrk(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrk_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZsyrk(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyrk(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrk_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasCherk(char uplo, char trans, int n, int k, float alpha, const cuComplex* A, int lda, float beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const hipblasComplex* AP, int lda, const float* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCherk(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  blasStatus = cublasCherk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZherk(char uplo, char trans, int n, int k, double alpha, const cuDoubleComplex* A, int lda, double beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const hipblasDoubleComplex* AP, int lda, const double* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZherk(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  blasStatus = cublasZherk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasSsyr2k(char uplo, char trans, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);

  // NOTE: void CUBLASWINAPI cublasDsyr2k(char uplo, char trans, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);

  // NOTE: void CUBLASWINAPI cublasCsyr2k(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyr2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZsyr2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyr2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsyrkx(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyrkx(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsyrkx(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyrkx(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyrkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyrkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasCher2k(char uplo, char trans, int n, int k, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, float beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const float* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCher2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCher2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZher2k(char uplo, char trans, int n, int k, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, double beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const double* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZher2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZher2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const float* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCherkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCherkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherkx(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const double* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZherkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZherkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasSsymm(char side, char uplo, int m, int n, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // NOTE: void CUBLASWINAPI cublasDsymm(char side, char uplo, int m, int n, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // NOTE: void CUBLASWINAPI cublasCsymm(char side, char uplo, int m, int n, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsymm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZsymm(char side, char uplo, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsymm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasChemm(char side, char uplo, int m, int n, cuComplex alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex beta, cuComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* BP, int ldb, const hipblasComplex* beta, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasChemm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasChemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // NOTE: void CUBLASWINAPI cublasZhemm(char side, char uplo, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* BP, int ldb, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZhemm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZhemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // NOTE: void CUBLASWINAPI cublasStrsm(char side, char uplo, char transa, char diag, int m, int n, float alpha, const float* A, int lda, float* B, int ldb); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const float* alpha, float* AP, int lda, float* BP, int ldb);
  // CHECK: blasStatus = hipblasStrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  blasStatus = cublasStrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);

  // NOTE: void CUBLASWINAPI cublasDtrsm(char side, char uplo, char transa, char diag, int m, int n, double alpha, const double* A, int lda, double* B, int ldb); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const double* alpha, double* AP, int lda, double* BP, int ldb);
  // CHECK: blasStatus = hipblasDtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  blasStatus = cublasDtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);

  // NOTE: void CUBLASWINAPI cublasCtrsm(char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const cuComplex* A, int lda, cuComplex* B, int ldb); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipblasComplex* alpha, hipblasComplex* AP, int lda, hipblasComplex* BP, int ldb);
  // CHECK: blasStatus = hipblasCtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  blasStatus = cublasCtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);

  // NOTE: void CUBLASWINAPI cublasZtrsm(char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb); is not supported by HIP
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipblasDoubleComplex* alpha, hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* BP, int ldb);
  // CHECK: blasStatus = hipblasZtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  blasStatus = cublasZtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const float* alpha, const float* AP, int lda, const float* beta, const float* BP, int ldb, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSgeam(blasHandle, transa, transb, m, n, &fa, &fA, lda, &fb, &fB, ldb, &fC, ldc);
  blasStatus = cublasSgeam(blasHandle, transa, transb, m, n, &fa, &fA, lda, &fb, &fB, ldb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const double* alpha, const double* AP, int lda, const double* beta, const double* BP, int ldb, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDgeam(blasHandle, transa, transb, m, n, &da, &dA, lda, &db, &dB, ldb, &dC, ldc);
  blasStatus = cublasDgeam(blasHandle, transa, transb, m, n, &da, &dA, lda, &db, &dB, ldb, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, const cuComplex* B, int ldb, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* beta, const hipblasComplex* BP, int ldb, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCgeam(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCgeam(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeam(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* beta, const hipblasDoubleComplex* BP, int ldb, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZgeam(blasHandle, transa, transb, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZgeam(blasHandle, transa, transb, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexB, ldb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrfBatched(cublasHandle_t handle, int n, float* const A[], int lda, int* P, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrfBatched(hipblasHandle_t handle, const int n, float* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasSgetrfBatched(blasHandle, n, fAarray, lda, &P, &info, batchCount);
  blasStatus = cublasSgetrfBatched(blasHandle, n, fAarray, lda, &P, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrfBatched(cublasHandle_t handle, int n, double* const A[], int lda, int* P, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrfBatched(hipblasHandle_t handle, const int n, double* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasDgetrfBatched(blasHandle, n, dAarray, lda, &P, &info, batchCount);
  blasStatus = cublasDgetrfBatched(blasHandle, n, dAarray, lda, &P, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrfBatched(cublasHandle_t handle, int n, cuComplex* const A[], int lda, int* P, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrfBatched(hipblasHandle_t handle, const int n, hipblasComplex* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetrfBatched(blasHandle, n, complexAarray, lda, &P, &info, batchCount);
  blasStatus = cublasCgetrfBatched(blasHandle, n, complexAarray, lda, &P, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrfBatched(hipblasHandle_t handle, const int n, hipblasDoubleComplex* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetrfBatched(blasHandle, n, dcomplexAarray, lda, &P, &info, batchCount);
  blasStatus = cublasZgetrfBatched(blasHandle, n, dcomplexAarray, lda, &P, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetriBatched(cublasHandle_t handle, int n, const float* const A[], int lda, const int* P, float* const C[], int ldc, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgetriBatched(hipblasHandle_t handle, const int n, float* const A[], const int lda, int* ipiv, float* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasSgetriBatched(blasHandle, n, fAarray_const, lda, &P, fCarray, ldc, &info, batchCount);
  blasStatus = cublasSgetriBatched(blasHandle, n, fAarray_const, lda, &P, fCarray, ldc, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetriBatched(cublasHandle_t handle, int n, const double* const A[], int lda, const int* P, double* const C[], int ldc, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgetriBatched(hipblasHandle_t handle, const int n, double* const A[], const int lda, int* ipiv, double* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasDgetriBatched(blasHandle, n, dAarray_const, lda, &P, dCarray, ldc, &info, batchCount);
  blasStatus = cublasDgetriBatched(blasHandle, n, dAarray_const, lda, &P, dCarray, ldc, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetriBatched(cublasHandle_t handle, int n, const cuComplex* const A[], int lda, const int* P, cuComplex* const C[], int ldc, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetriBatched(hipblasHandle_t handle, const int n, hipblasComplex* const A[], const int lda, int* ipiv, hipblasComplex* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetriBatched(blasHandle, n, complexAarray_const, lda, &P, complexCarray, ldc, &info, batchCount);
  blasStatus = cublasCgetriBatched(blasHandle, n, complexAarray_const, lda, &P, complexCarray, ldc, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetriBatched(hipblasHandle_t handle, const int n, hipblasDoubleComplex* const A[], const int lda, int* ipiv, hipblasDoubleComplex* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetriBatched(blasHandle, n, dcomplexAarray_const, lda, &P, dcomplexCarray, ldc, &info, batchCount);
  blasStatus = cublasZgetriBatched(blasHandle, n, dcomplexAarray_const, lda, &P, dcomplexCarray, ldc, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* const Aarray[], int lda, const int* devIpiv, float* const Barray[], int ldb, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, float* const A[], const int lda, const int* ipiv, float* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasSgetrsBatched(blasHandle, transa, n, nrhs, fAarray_const, lda, &P, fBarray, ldb, &info, batchCount);
  blasStatus = cublasSgetrsBatched(blasHandle, transa, n, nrhs, fAarray_const, lda, &P, fBarray, ldb, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* const Aarray[], int lda, const int* devIpiv, double* const Barray[], int ldb, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, double* const A[], const int lda, const int* ipiv, double* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasDgetrsBatched(blasHandle, transa, n, nrhs, dAarray_const, lda, &P, dBarray, ldb, &info, batchCount);
  blasStatus = cublasDgetrsBatched(blasHandle, transa, n, nrhs, dAarray_const, lda, &P, dBarray, ldb, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* const Aarray[], int lda, const int* devIpiv, cuComplex* const Barray[], int ldb, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, hipblasComplex* const A[], const int lda, const int* ipiv, hipblasComplex* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetrsBatched(blasHandle, transa, n, nrhs, complexAarray_const, lda, &P, complexBarray, ldb, &info, batchCount);
  blasStatus = cublasCgetrsBatched(blasHandle, transa, n, nrhs, complexAarray_const, lda, &P, complexBarray, ldb, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrsBatched(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, hipblasDoubleComplex* const A[], const int lda, const int* ipiv, hipblasDoubleComplex* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetrsBatched(blasHandle, transa, n, nrhs, dcomplexAarray_const, lda, &P, dcomplexBarray, ldb, &info, batchCount);
  blasStatus = cublasZgetrsBatched(blasHandle, transa, n, nrhs, dcomplexAarray_const, lda, &P, dcomplexBarray, ldb, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* const A[], int lda, float* const B[], int ldb, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const float* alpha, float* const AP[], int lda, float* BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasStrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, fAarray_const, lda, fBarray, ldb, batchCount);
  blasStatus = cublasStrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, fAarray_const, lda, fBarray, ldb, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* const A[], int lda, double* const B[], int ldb, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const double* alpha, double* const AP[], int lda, double* BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasDtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, dAarray_const, lda, dBarray, ldb, batchCount);
  blasStatus = cublasDtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, dAarray_const, lda, dBarray, ldb, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* const A[], int lda, cuComplex* const B[], int ldb, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipblasComplex* alpha, hipblasComplex* const AP[], int lda, hipblasComplex* BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasCtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);
  blasStatus = cublasCtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsmBatched(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipblasDoubleComplex* alpha, hipblasDoubleComplex* const AP[], int lda, hipblasDoubleComplex* BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasZtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray, ldb, batchCount);
  blasStatus = cublasZtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray, ldb, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeqrfBatched(cublasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgeqrfBatched(hipblasHandle_t handle, const int m, const int n, float* const A[], const int lda, float* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasSgeqrfBatched(blasHandle, m, n, fAarray, lda, fTauarray, &info, batchCount);
  blasStatus = cublasSgeqrfBatched(blasHandle, m, n, fAarray, lda, fTauarray, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeqrfBatched(cublasHandle_t handle, int m, int n, double* const Aarray[], int lda, double* const TauArray[], int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgeqrfBatched(hipblasHandle_t handle, const int m, const int n, double* const A[], const int lda, double* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasDgeqrfBatched(blasHandle, m, n, dAarray, lda, dTauarray, &info, batchCount);
  blasStatus = cublasDgeqrfBatched(blasHandle, m, n, dAarray, lda, dTauarray, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeqrfBatched(cublasHandle_t handle, int m, int n, cuComplex* const Aarray[], int lda, cuComplex* const TauArray[], int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeqrfBatched(hipblasHandle_t handle, const int m, const int n, hipblasComplex* const A[], const int lda, hipblasComplex* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgeqrfBatched(blasHandle, m, n, complexAarray, lda, complexTauarray, &info, batchCount);
  blasStatus = cublasCgeqrfBatched(blasHandle, m, n, complexAarray, lda, complexTauarray, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeqrfBatched(hipblasHandle_t handle, const int m, const int n, hipblasDoubleComplex* const A[], const int lda, hipblasDoubleComplex* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgeqrfBatched(blasHandle, m, n, dcomplexAarray, lda, dcomplexTauarray, &info, batchCount);
  blasStatus = cublasZgeqrfBatched(blasHandle, m, n, dcomplexAarray, lda, dcomplexTauarray, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const float* A, int lda, const float* x, int incx, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSdgmm(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const float* AP, int lda, const float* x, int incx, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSdgmm(blasHandle, blasSideMode, m, n, &fa, lda, &fx, incx, &fC, ldc);
  blasStatus = cublasSdgmm(blasHandle, blasSideMode, m, n, &fa, lda, &fx, incx, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const double* A, int lda, const double* x, int incx, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDdgmm(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const double* AP, int lda, const double* x, int incx, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDdgmm(blasHandle, blasSideMode, m, n, &da, lda, &dx, incx, &dC, ldc);
  blasStatus = cublasDdgmm(blasHandle, blasSideMode, m, n, &da, lda, &dx, incx, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuComplex* A, int lda, const cuComplex* x, int incx, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmm(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, hipblasComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCdgmm(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);
  blasStatus = cublasCdgmm(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmm(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZdgmm(blasHandle, blasSideMode, m, n, &dcomplexa, lda, &dcomplexx, incx, &dcomplexC, ldc);
  blasStatus = cublasZdgmm(blasHandle, blasSideMode, m, n, &dcomplexa, lda, &dcomplexx, incx, &dcomplexC, ldc);

  int deviceInfo = 0;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda, float* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, float* const A[], const int lda, float* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasSgelsBatched(blasHandle, blasOperation, m, n, nrhs, fAarray, lda, fCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasSgelsBatched(blasHandle, blasOperation, m, n, nrhs, fAarray, lda, fCarray, ldc, &info, &deviceInfo, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, double* const Aarray[], int lda, double* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, double* const A[], const int lda, double* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasDgelsBatched(blasHandle, blasOperation, m, n, nrhs, dAarray, lda, dCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasDgelsBatched(blasHandle, blasOperation, m, n, nrhs, dAarray, lda, dCarray, ldc, &info, &deviceInfo, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuComplex* const Aarray[], int lda, cuComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, hipblasComplex* const A[], const int lda, hipblasComplex* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasCgelsBatched(blasHandle, blasOperation, m, n, nrhs, complexAarray, lda, complexCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasCgelsBatched(blasHandle, blasOperation, m, n, nrhs, complexAarray, lda, complexCarray, ldc, &info, &deviceInfo, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgelsBatched(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, hipblasDoubleComplex* const A[], const int lda, hipblasDoubleComplex* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasZgelsBatched(blasHandle, blasOperation, m, n, nrhs, dcomplexAarray, lda, dcomplexCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasZgelsBatched(blasHandle, blasOperation, m, n, nrhs, dcomplexAarray, lda, dcomplexCarray, ldc, &info, &deviceInfo, batchCount);

  long long int strideA = 0;
  long long int strideB = 0;
  long long int strideC = 0;

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

  // CHECK: hipblasDatatype_t DataType_2, DataType_3;
  cudaDataType DataType_2, DataType_3;

  // CHECK: hipblasGemmAlgo_t blasGemmAlgo;
  // CHECK-NEXT: hipblasGemmAlgo_t BLAS_GEMM_DFALT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t blasGemmAlgo;
  cublasGemmAlgo_t BLAS_GEMM_DFALT = CUBLAS_GEMM_DFALT;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2Ex(hipblasHandle_t handle, int n, const void* x, hipblasDatatype_t xType, int incx, void* result, hipblasDatatype_t resultType, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasNrm2Ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);
  blasStatus = cublasNrm2Ex(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, long long int strideA, const float* B, int ldb, long long int strideB, const float* beta, float* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* AP, int lda, long long strideA, const float* BP, int ldb, long long strideB, const float* beta, float* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasSgemmStridedBatched(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, strideA, &fB, ldb, strideB, &fb, &fC, ldc, strideC, batchCount);
  blasStatus = cublasSgemmStridedBatched(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, strideA, &fB, ldb, strideB, &fb, &fC, ldc, strideC, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, long long int strideA, const double* B, int ldb, long long int strideB, const double* beta, double* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double* alpha, const double* AP, int lda, long long strideA, const double* BP, int ldb, long long strideB, const double* beta, double* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasDgemmStridedBatched(blasHandle, transa, transb, m, n, k, &da, &dA, lda, strideA, &dB, ldb, strideB, &db, &dC, ldc, strideC, batchCount);
  blasStatus = cublasDgemmStridedBatched(blasHandle, transa, transb, m, n, k, &da, &dA, lda, strideA, &dB, ldb, strideB, &db, &dC, ldc, strideC, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, long long int strideA, const cuComplex* B, int ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, long long strideA, const hipblasComplex* BP, int ldb, long long strideB, const hipblasComplex* beta, hipblasComplex* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasCgemmStridedBatched(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);
  blasStatus = cublasCgemmStridedBatched(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, long long strideA, const hipblasDoubleComplex* BP, int ldb, long long strideB, const hipblasDoubleComplex* beta, hipblasDoubleComplex* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasZgemmStridedBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);
  blasStatus = cublasZgemmStridedBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);

  // TODO: __half -> hipblasHalf
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half* beta, __half* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* AP, int lda, long long strideA, const hipblasHalf* BP, int ldb, long long strideB, const hipblasHalf* beta, hipblasHalf* CP, int ldc, long long strideC, int batchCount);

  void* aptr = nullptr;
  void* Aptr = nullptr;
  void* bptr = nullptr;
  void* Bptr = nullptr;
  void* cptr = nullptr;
  void* Cptr = nullptr;
  void* xptr = nullptr;
  void* yptr = nullptr;
  void* sptr = nullptr;

  // CHECK: hipblasDatatype_t Atype;
  // CHECK-NEXT: hipblasDatatype_t Btype;
  // CHECK-NEXT: hipblasDatatype_t Ctype;
  // CHECK-NEXT: hipblasDatatype_t Xtype;
  // CHECK-NEXT: hipblasDatatype_t Ytype;
  // CHECK-NEXT: hipblasDatatype_t CStype;
  // CHECK-NEXT: hipblasDatatype_t Executiontype;
  cudaDataType Atype;
  cudaDataType Btype;
  cudaDataType Ctype;
  cudaDataType Xtype;
  cudaDataType Ytype;
  cudaDataType CStype;
  cudaDataType Executiontype;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScalEx(hipblasHandle_t handle, int n, const void* alpha, hipblasDatatype_t alphaType, void* x, hipblasDatatype_t xType, int incx, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasScalEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);
  blasStatus = cublasScalEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyEx(hipblasHandle_t handle, int n, const void* alpha, hipblasDatatype_t alphaType, const void* x, hipblasDatatype_t xType, int incx, void* y, hipblasDatatype_t yType, int incy, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasAxpyEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);
  blasStatus = cublasAxpyEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotEx(hipblasHandle_t handle, int n, const void* x, hipblasDatatype_t xType, int incx, const void* y, hipblasDatatype_t yType, int incy, void* result, hipblasDatatype_t resultType, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasDotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotcEx(hipblasHandle_t handle, int n, const void* x, hipblasDatatype_t xType, int incx, const void* y, hipblasDatatype_t yType, int incy, void* result, hipblasDatatype_t resultType, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasDotcEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotcEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 11000
  // CHECK: hipblasDatatype_t computeType;
  cudaDataType computeType;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType, int lda, const void* B, hipblasDatatype_t bType, int ldb, const void* beta, void* C, hipblasDatatype_t cType, int ldc, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
  blasStatus = cublasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipblasGemmAlgo_t BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
#endif

#if CUDA_VERSION >= 9010 && CUDA_VERSION < 11000
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A[], hipblasDatatype_t aType, int lda, const void* B[], hipblasDatatype_t bType, int ldb, const void* beta, void* C[], hipblasDatatype_t cType, int ldc, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType, int lda, hipblasStride strideA, const void* B, hipblasDatatype_t bType, int ldb, hipblasStride strideB, const void* beta, void* C, hipblasDatatype_t cType, int ldc, hipblasStride strideC, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipblasOperation_t BLAS_OP_HERMITAN = HIPBLAS_OP_C;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;

  // CHECK: hipblasFillMode_t BLAS_FILL_MODE_FULL = HIPBLAS_FILL_MODE_FULL;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasRotEx(hipblasHandle_t handle, int n, void* x, hipblasDatatype_t xType, int incx, void* y, hipblasDatatype_t yType, int incy, const void* c, const void* s, hipblasDatatype_t csType, hipblasDatatype_t executionType);
  // CHECK: blasStatus = hipblasRotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
  blasStatus = cublasRotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipblasDatatype_t R_16BF = HIPBLAS_R_16B;
  // CHECK-NEXT: hipblasDatatype_t C_16BF = HIPBLAS_C_16B;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;

  // NOTE: WORKAROUND: cublasComputeType_t is not actually supported by hipBLAS
  // TODO: Fix it after fixing https://github.com/ROCmSoftwarePlatform/hipBLAS/issues/529
  // CHECK: hipblasDatatype_t blasComputeType;
  cublasComputeType_t blasComputeType;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType, int lda, const void* B, hipblasDatatype_t bType, int ldb, const void* beta, void* C, hipblasDatatype_t cType, int ldc, hipblasDatatype_t computeType, ipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, blasComputeType, blasGemmAlgo);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A[], hipblasDatatype_t aType, int lda, const void* B[], hipblasDatatype_t bType, int ldb, const void* beta, void* C[], hipblasDatatype_t cType, int ldc, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray, Atype, lda, voidBarray, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray, Atype, lda, voidBarray, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, blasComputeType, blasGemmAlgo);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType, int lda, hipblasStride strideA, const void* B, hipblasDatatype_t bType, int ldb, hipblasStride strideB, const void* beta, void* C, hipblasDatatype_t cType, int ldc, hipblasStride strideC, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, blasComputeType, blasGemmAlgo);
#endif

  return 0;
}
