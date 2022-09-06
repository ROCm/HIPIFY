// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas_v2.h"
#include "cublas.h"
#include "cublas_v2.h"
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
  // CHECK-NEXT: blasStatus = hipblasCreate(&blasHandle);
  blasStatus = cublasCreate(&blasHandle);
  blasStatus = cublasCreate_v2(&blasHandle);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t handle);
  // CUDA: #define cublasDestroy cublasDestroy_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
  // CHECK: blasStatus = hipblasDestroy(blasHandle);
  // CHECK-NEXT: blasStatus = hipblasDestroy(blasHandle);
  blasStatus = cublasDestroy(blasHandle);
  blasStatus = cublasDestroy_v2(blasHandle);

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
  // CUDA: #define cublasSetStream cublasSetStream_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetStream(hipblasHandle_t handle, hipStream_t streamId);
  // CHECK: blasStatus = hipblasSetStream(blasHandle, stream);
  // CHECK-NEXT: blasStatus = hipblasSetStream(blasHandle, stream);
  blasStatus = cublasSetStream(blasHandle, stream);
  blasStatus = cublasSetStream_v2(blasHandle, stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
  // CUDA: #define cublasGetStream cublasGetStream_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetStream(hipblasHandle_t handle, hipStream_t* streamId);
  // CHECK: blasStatus = hipblasGetStream(blasHandle, &stream);
  // CHECK-NEXT: blasStatus = hipblasGetStream(blasHandle, &stream);
  blasStatus = cublasGetStream(blasHandle, &stream);
  blasStatus = cublasGetStream_v2(blasHandle, &stream);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
  // CUDA: #define cublasSetPointerMode cublasSetPointerMode_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t mode);
  // CHECK: blasStatus = hipblasSetPointerMode(blasHandle, blasPointerMode);
  // CHECK-NEXT: blasStatus = hipblasSetPointerMode(blasHandle, blasPointerMode);
  blasStatus = cublasSetPointerMode(blasHandle, blasPointerMode);
  blasStatus = cublasSetPointerMode_v2(blasHandle, blasPointerMode);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
  // CUDA: #define cublasGetPointerMode cublasGetPointerMode_v2
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGetPointerMode(hipblasHandle_t handle, hipblasPointerMode_t* mode);
  // CHECK: blasStatus = hipblasGetPointerMode(blasHandle, &blasPointerMode);
  // CHECK-NEXT: blasStatus = hipblasGetPointerMode(blasHandle, &blasPointerMode);
  blasStatus = cublasGetPointerMode(blasHandle, &blasPointerMode);
  blasStatus = cublasGetPointerMode_v2(blasHandle, &blasPointerMode);

  int n = 0;
  int m = 0;
  int num = 0;
  int lda = 0;
  int res = 0;
  int incx = 0;
  int incy = 0;
  int k = 0;
  int kl = 0;
  int ku = 0;
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
  float fx = 0;
  float fx1 = 0;
  float fy = 0;
  float fy1 = 0;
  float fc = 0;
  float fs = 0;
  float fd1 = 0;
  float fd2 = 0;
  float fresult = 0;

  double da = 0;
  double dA = 0;
  double db = 0;
  double dx = 0;
  double dx1 = 0;
  double dy = 0;
  double dy1 = 0;
  double dc = 0;
  double ds = 0;
  double dd1 = 0;
  double dd2 = 0;
  double dresult = 0;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSnrm2(hipblasHandle_t handle, int n, const float* x, int incx, float* result);
  // CHECK: blasStatus = hipblasSnrm2(blasHandle, n, &fx, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSnrm2(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSnrm2(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSnrm2_v2(blasHandle, n, &fx, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2(hipblasHandle_t handle, int n, const double* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDnrm2(blasHandle, n, &dx, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDnrm2(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDnrm2(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDnrm2_v2(blasHandle, n, &dx, incx, &dresult);

  // CHECK: hipComplex complex, complexa, complexA, complexx, complexy, complexs, complexb;
  cuComplex complex, complexa, complexA, complexx, complexy, complexs, complexb;
  // CHECK: hipDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexx, dcomplexy, dcomplexs, dcomplexb;
  cuDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexx, dcomplexy, dcomplexs, dcomplexb;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScnrm2(blasHandle, n, &complex, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  blasStatus = cublasDznrm2(blasHandle, n, &dcomplex, incx, &dresult);
  blasStatus = cublasDznrm2_v2(blasHandle, n, &dcomplex, incx, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSdot(hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
  // CHECK: blasStatus = hipblasSdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSdot(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSdot_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDdot(hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result);
  // CHECK: blasStatus = hipblasDdot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDdot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDdot(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDdot_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotu(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* result);
  // CHECK: blasStatus = hipblasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = hipblasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotc(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, const hipblasComplex* y, int incy, hipblasComplex* result);
  // CHECK: blasStatus = hipblasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = hipblasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotu(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = hipblasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotc(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* y, int incy, hipblasDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = hipblasZdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotc(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotc_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSscal(hipblasHandle_t handle, int n, const float* alpha, float* x, int incx);
  // CHECK: blasStatus = hipblasSscal(blasHandle, n, &fy, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasSscal(blasHandle, n, &fy, &fx, incx);
  blasStatus = cublasSscal(blasHandle, n, &fy, &fx, incx);
  blasStatus = cublasSscal_v2(blasHandle, n, &fy, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDscal(hipblasHandle_t handle, int n, const double* alpha, double* x, int incx);
  // CHECK: blasStatus = hipblasDscal(blasHandle, n, &dx, &dy, incx);
  // CHECK-NEXT: blasStatus = hipblasDscal(blasHandle, n, &dx, &dy, incx);
  blasStatus = cublasDscal(blasHandle, n, &dx, &dy, incx);
  blasStatus = cublasDscal_v2(blasHandle, n, &dx, &dy, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCscal(hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCscal(blasHandle, n, &complexa, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal_v2(blasHandle, n, &complexa, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsscal(hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCsscal(blasHandle, n, &fx, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCsscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal_v2(blasHandle, n, &fx, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZscal(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdscal(hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZdscal(blasHandle, n, &dx, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZdscal(blasHandle, n, &dx, &dcomplexx, incx);
  blasStatus = cublasZdscal(blasHandle, n, &dx, &dcomplexx, incx);
  blasStatus = cublasZdscal_v2(blasHandle, n, &dx, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpy(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasSaxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSaxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  blasStatus = cublasSaxpy(blasHandle, n, &fa, &fx, incx, &fy, incy);
  blasStatus = cublasSaxpy_v2(blasHandle, n, &fa, &fx, incx, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpy(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDaxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDaxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  blasStatus = cublasDaxpy(blasHandle, n, &da, &dx, incx, &dy, incy);
  blasStatus = cublasDaxpy_v2(blasHandle, n, &da, &dx, incx, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpy(hipblasHandle_t handle, int n, const hipblasComplex* alpha, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZaxpy(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZaxpy_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2(cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScopy(hipblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasScopy(blasHandle, n, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasScopy(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasScopy(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasScopy_v2(blasHandle, n, &fx, incx, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2(cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDcopy(hipblasHandle_t handle, int n, const double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDcopy(blasHandle, n, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDcopy(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDcopy(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDcopy_v2(blasHandle, n, &dx, incx, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCcopy(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZcopy(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZcopy(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZcopy_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSswap(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy);
  // CHECK: blasStatus = hipblasSswap(blasHandle, n, &fx, incx, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSswap(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasSswap(blasHandle, n, &fx, incx, &fy, incy);
  blasStatus = cublasSswap_v2(blasHandle, n, &fx, incx, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDswap(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy);
  // CHECK: blasStatus = hipblasDswap(blasHandle, n, &dx, incx, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDswap(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDswap(blasHandle, n, &dx, incx, &dy, incy);
  blasStatus = cublasDswap_v2(blasHandle, n, &dx, incx, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCswap(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZswap(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZswap(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  blasStatus = cublasZswap_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamax(hipblasHandle_t handle, int n, const float* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIsamax(blasHandle, n, &fx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIsamax(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamax(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamax_v2(blasHandle, n, &fx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamax(hipblasHandle_t handle, int n, const double* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIdamax(blasHandle, n, &dx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIdamax(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamax(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamax_v2(blasHandle, n, &dx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamax(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamax(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIcamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax_v2(blasHandle, n, &complexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamax(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamax(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIzamax(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamax(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamax_v2(blasHandle, n, &dcomplexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, int n, const float* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamin(hipblasHandle_t handle, int n, const float* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIsamin(blasHandle, n, &fx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIsamin(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamin(blasHandle, n, &fx, incx, &res);
  blasStatus = cublasIsamin_v2(blasHandle, n, &fx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, int n, const double* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamin(hipblasHandle_t handle, int n, const double* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIdamin(blasHandle, n, &dx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIdamin(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamin(blasHandle, n, &dx, incx, &res);
  blasStatus = cublasIdamin_v2(blasHandle, n, &dx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamin(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamin(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIcamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin_v2(blasHandle, n, &complexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamin(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamin(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIzamin(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamin(blasHandle, n, &dcomplexx, incx, &res);
  blasStatus = cublasIzamin_v2(blasHandle, n, &dcomplexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSasum(hipblasHandle_t handle, int n, const float* x, int incx, float* result);
  // CHECK: blasStatus = hipblasSasum(blasHandle, n, &fx, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSasum(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSasum(blasHandle, n, &fx, incx, &fresult);
  blasStatus = cublasSasum_v2(blasHandle, n, &fx, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDasum(hipblasHandle_t handle, int n, const double* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDasum(blasHandle, n, &dx, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDasum(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDasum(blasHandle, n, &dx, incx, &dresult);
  blasStatus = cublasDasum_v2(blasHandle, n, &dx, incx, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScasum(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScasum(blasHandle, n, &complexx, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum_v2(blasHandle, n, &complexx, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDzasum(hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  blasStatus = cublasDzasum(blasHandle, n, &dcomplexx, incx, &dresult);
  blasStatus = cublasDzasum_v2(blasHandle, n, &dcomplexx, incx, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrot(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasSrot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasSrot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  blasStatus = cublasSrot(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);
  blasStatus = cublasSrot_v2(blasHandle, n, &fx, incx, &fy, incy, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrot(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasDrot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasDrot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  blasStatus = cublasDrot(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);
  blasStatus = cublasDrot_v2(blasHandle, n, &dx, incx, &dy, incy, &dc, &ds);

  // CUDA: CUBLASAPI CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrot(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, const float* c, const hipblasComplex* s);
  // CHECK: blasStatus = hipblasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  // CHECK-NEXT: blasStatus = hipblasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsrot(hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrot(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy, const double* c, const hipblasDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = hipblasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdrot(hipblasHandle_t handle, int n, hipblasDoubleComplex* x, int incx, hipblasDoubleComplex* y, int incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasZdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasZdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  blasStatus = cublasZdrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  blasStatus = cublasZdrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, float* a, float* b, float* c, float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotg(hipblasHandle_t handle, float* a, float* b, float* c, float* s);
  // CHECK: blasStatus = hipblasSrotg(blasHandle, &fa, &fb, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasSrotg(blasHandle, &fa, &fb, &fc, &fs);
  blasStatus = cublasSrotg(blasHandle, &fa, &fb, &fc, &fs);
  blasStatus = cublasSrotg_v2(blasHandle, &fa, &fb, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, double* a, double* b, double* c, double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotg(hipblasHandle_t handle, double* a, double* b, double* c, double* s);
  // CHECK: blasStatus = hipblasDrotg(blasHandle, &da, &db, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasDrotg(blasHandle, &da, &db, &dc, &ds);
  blasStatus = cublasDrotg(blasHandle, &da, &db, &dc, &ds);
  blasStatus = cublasDrotg_v2(blasHandle, &da, &db, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, cuComplex* a, cuComplex* b, float* c, cuComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrotg(hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);
  // CHECK: blasStatus = hipblasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  // CHECK-NEXT: blasStatus = hipblasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrotg(hipblasHandle_t handle, hipblasDoubleComplex* a, hipblasDoubleComplex* b, double* c, hipblasDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = hipblasZrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  blasStatus = cublasZrotg(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  blasStatus = cublasZrotg_v2(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotm(hipblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* param);
  // CHECK: blasStatus = hipblasSrotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSrotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSrotm(blasHandle, n, &fx, incx, &fy, incy, &fresult);
  blasStatus = cublasSrotm_v2(blasHandle, n, &fx, incx, &fy, incy, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotm(hipblasHandle_t handle, int n, double* x, int incx, double* y, int incy, const double* param);
  // CHECK: blasStatus = hipblasDrotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDrotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDrotm(blasHandle, n, &dx, incx, &dy, incy, &dresult);
  blasStatus = cublasDrotm_v2(blasHandle, n, &dx, incx, &dy, incy, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotmg(hipblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param);
  // CHECK: blasStatus = hipblasSrotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSrotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  blasStatus = cublasSrotmg(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);
  blasStatus = cublasSrotmg_v2(blasHandle, &fd1, &fd2, &fx1, &fy1, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotmg(hipblasHandle_t handle, double* d1, double* d2, double* x1, const double* y1, double* param);
  // CHECK: blasStatus = hipblasDrotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDrotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  blasStatus = cublasDrotmg(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);
  blasStatus = cublasDrotmg_v2(blasHandle, &dd1, &dd2, &dx1, &dy1, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgemv(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgemv_v2(blasHandle, blasOperation, m, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgemv(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgemv_v2(blasHandle, blasOperation, m, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgemv(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgemv_v2(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgbmv(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgbmv(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmv(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgbmv(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, float* x, int incx);
  // CHECK: blasStatus = hipblasStpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, double* x, int incx);
  // CHECK: blasStatus = hipblasDtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);
  blasStatus = cublasStrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, lda, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);
  blasStatus = cublasDtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, lda, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float* AP, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const float* AP, float* x, int incx);
  // CHECK: blasStatus = hipblasStpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);
  blasStatus = cublasStpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &fA, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double* AP, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const double* AP, double* x, int incx);
  // CHECK: blasStatus = hipblasDtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);
  blasStatus = cublasDtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dA, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex* AP, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasComplex* AP, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipblasDoubleComplex* AP, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  blasStatus = cublasZtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float* A, int lda, float* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const float* AP, int lda, float* x, int incx);
  // CHECK: blasStatus = hipblasStbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  // CHECK-NEXT: blasStatus = hipblasStbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);
  blasStatus = cublasStbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &fA, lda, &fx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double* A, int lda, double* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const double* AP, int lda, double* x, int incx);
  // CHECK: blasStatus = hipblasDtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  // CHECK-NEXT: blasStatus = hipblasDtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);
  blasStatus = cublasDtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dA, lda, &dx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex* A, int lda, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipblasComplex* AP, int lda, hipblasComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsv(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipblasDoubleComplex* AP, int lda, hipblasDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  blasStatus = cublasZtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSsymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSsymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsymv(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsymv_v2(blasHandle, blasFillMode, n, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsymv(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsymv_v2(blasHandle, blasFillMode, n, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhemv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhemv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const float* alpha, const float* AP, int lda, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSsbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSsbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsbmv(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSsbmv_v2(blasHandle, blasFillMode, n, k, &fa, &fA, lda, &fx, incx, &fb, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double* alpha, const double* A, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const double* alpha, const double* AP, int lda, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsbmv(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDsbmv_v2(blasHandle, blasFillMode, n, k, &da, &dA, lda, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipblasComplex* alpha, const hipblasComplex* AP, int lda, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, int lda, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhbmv(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhbmv_v2(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy);
  // CHECK: blasStatus = hipblasSspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  // CHECK-NEXT: blasStatus = hipblasSspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSspmv(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);
  blasStatus = cublasSspmv_v2(blasHandle, blasFillMode, n, &fa, &fA, &fx, incx, &fb, &fy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* AP, const double* x, int incx, const double* beta, double* y, int incy);
  // CHECK: blasStatus = hipblasDspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  // CHECK-NEXT: blasStatus = hipblasDspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDspmv(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);
  blasStatus = cublasDspmv_v2(blasHandle, blasFillMode, n, &da, &dA, &dx, incx, &db, &dy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasComplex* alpha, const hipblasComplex* AP, const hipblasComplex* x, int incx, const hipblasComplex* beta, hipblasComplex* y, int incy);
  // CHECK: blasStatus = hipblasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmv(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipblasDoubleComplex* alpha, const hipblasDoubleComplex* AP, const hipblasDoubleComplex* x, int incx, const hipblasDoubleComplex* beta, hipblasDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

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
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipblasGemmAlgo_t BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipblasOperation_t BLAS_OP_HERMITAN = HIPBLAS_OP_C;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;

  // CHECK: hipblasFillMode_t BLAS_FILL_MODE_FULL = HIPBLAS_FILL_MODE_FULL;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipblasDatatype_t R_16BF = HIPBLAS_R_16B;
  // CHECK-NEXT: hipblasDatatype_t C_16BF = HIPBLAS_C_16B;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;
#endif

  return 0;
}
