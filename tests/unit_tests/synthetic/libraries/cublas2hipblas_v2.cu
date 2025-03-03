// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas_v2.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "hipblas.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("14.v2. cuBLAS API to hipBLAS API synthetic test\n");

  // CHECK: hipblasOperation_t blasOperation;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_N = HIPBLAS_OP_N;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_T = HIPBLAS_OP_T;
  // CHECK-NEXT: hipblasOperation_t BLAS_OP_C = HIPBLAS_OP_C;
  cublasOperation_t blasOperation;
  cublasOperation_t BLAS_OP_N = CUBLAS_OP_N;
  cublasOperation_t BLAS_OP_T = CUBLAS_OP_T;
  cublasOperation_t BLAS_OP_C = CUBLAS_OP_C;

  // CHECK: hipblasStatus_t blasStatus;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_SUCCESS = HIPBLAS_STATUS_SUCCESS;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_NOT_INITIALIZED = HIPBLAS_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_ALLOC_FAILED = HIPBLAS_STATUS_ALLOC_FAILED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_INVALID_VALUE = HIPBLAS_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_MAPPING_ERROR = HIPBLAS_STATUS_MAPPING_ERROR;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_EXECUTION_FAILED = HIPBLAS_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_INTERNAL_ERROR = HIPBLAS_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_NOT_SUPPORTED = HIPBLAS_STATUS_NOT_SUPPORTED;
  // CHECK-NEXT: hipblasStatus_t BLAS_STATUS_ARCH_MISMATCH = HIPBLAS_STATUS_ARCH_MISMATCH;
  cublasStatus_t blasStatus;
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
  int64_t n_64 = 0;
  int nrhs = 0;
  int m = 0;
  int64_t m_64 = 0;
  int num = 0;
  int lda = 0;
  int64_t lda_64 = 0;
  int ldb = 0;
  int64_t ldb_64 = 0;
  int ldc = 0;
  int64_t ldc_64 = 0;
  int res = 0;
  int64_t res_64 = 0;
  int incx = 0;
  int64_t incx_64 = 0;
  int incy = 0;
  int64_t incy_64 = 0;
  int k = 0;
  int64_t k_64 = 0;
  int kl = 0;
  int64_t kl_64 = 0;
  int ku = 0;
  int64_t ku_64 = 0;
  int batchCount = 0;
  int64_t batchCount_64 = 0;
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

  float fa = 0.0f;
  float fA = 0.0f;
  float fAP = 0.0f;
  float fb = 0.0f;
  float fB = 0.0f;
  float fx = 0.0f;
  float fx1 = 0.0f;
  float fy = 0.0f;
  float fy1 = 0.0f;
  float fc = 0.0f;
  float fC = 0.0f;
  float fCP = 0.0f;
  float fs = 0.0f;
  float fd1 = 0.0f;
  float fd2 = 0.0f;
  float fresult = 0.0f;
  float fparam = 0.0f;

  float** fAarray = nullptr;
  const float** const fAarray_const = const_cast<const float**>(fAarray);
  float** fBarray = nullptr;
  const float** const fBarray_const = const_cast<const float**>(fBarray);
  float** fXarray = nullptr;
  const float** const fXarray_const = const_cast<const float**>(fXarray);
  float** fYarray = nullptr;
  const float** const fYarray_const = const_cast<const float**>(fYarray);
  float** fCarray = nullptr;
  const float** const fCarray_const = const_cast<const float**>(fCarray);
  float** fTauarray = nullptr;

  double da = 0.0f;
  double dA = 0.0f;
  double dAP = 0.0f;
  double db = 0.0f;
  double dB = 0.0f;
  double dx = 0.0f;
  double dx1 = 0.0f;
  double dy = 0.0f;
  double dy1 = 0.0f;
  double dc = 0.0f;
  double dC = 0.0f;
  double dCP = 0.0f;
  double ds = 0.0f;
  double dd1 = 0.0f;
  double dd2 = 0.0f;
  double dresult = 0.0f;
  double dparam = 0.0f;

  double** dAarray = nullptr;
  const double** const dAarray_const = const_cast<const double**>(dAarray);
  double** dBarray = nullptr;
  const double** const dBarray_const = const_cast<const double**>(dBarray);
  double** dXarray = nullptr;
  const double** const dXarray_const = const_cast<const double**>(dXarray);
  double** dYarray = nullptr;
  const double** const dYarray_const = const_cast<const double**>(dYarray);
  double** dCarray = nullptr;
  double** dTauarray = nullptr;

  void** voidAarray = nullptr;
  const void** const voidAarray_const = const_cast<const void**>(voidAarray);
  void** voidBarray = nullptr;
  const void** const voidBarray_const = const_cast<const void**>(voidBarray);
  void** voidCarray = nullptr;
  const void** const voidCarray_const = const_cast<const void**>(voidCarray);

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

  // CHECK: hipComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb, complexresult;
  cuComplex complex, complexa, complexA, complexB, complexC, complexx, complexy, complexs, complexb, complexresult;
  // CHECK: hipDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb, dcomplexresult;
  cuDoubleComplex dcomplex, dcomplexa, dcomplexA, dcomplexB, dcomplexC, dcomplexx, dcomplexy, dcomplexs, dcomplexb, dcomplexresult;

  // CHECK: hipComplex** complexAarray = 0;
  // CHECK: const hipComplex** const complexAarray_const = const_cast<const hipComplex**>(complexAarray);
  // CHECK-NEXT: hipComplex** complexBarray = 0;
  // CHECK-NEXT: const hipComplex** const complexBarray_const = const_cast<const hipComplex**>(complexBarray);
  // CHECK-NEXT: hipComplex** complexXarray = 0;
  // CHECK-NEXT: const hipComplex** const complexXarray_const = const_cast<const hipComplex**>(complexXarray);
  // CHECK-NEXT: hipComplex** complexYarray = 0;
  // CHECK-NEXT: const hipComplex** const complexYarray_const = const_cast<const hipComplex**>(complexYarray);
  // CHECK-NEXT: hipComplex** complexCarray = 0;
  // CHECK-NEXT: hipComplex** complexTauarray = 0;
  cuComplex** complexAarray = 0;
  const cuComplex** const complexAarray_const = const_cast<const cuComplex**>(complexAarray);
  cuComplex** complexBarray = 0;
  const cuComplex** const complexBarray_const = const_cast<const cuComplex**>(complexBarray);
  cuComplex** complexXarray = 0;
  const cuComplex** const complexXarray_const = const_cast<const cuComplex**>(complexXarray);
  cuComplex** complexYarray = 0;
  const cuComplex** const complexYarray_const = const_cast<const cuComplex**>(complexYarray);
  cuComplex** complexCarray = 0;
  cuComplex** complexTauarray = 0;

  // CHECK: hipDoubleComplex** dcomplexAarray = 0;
  // CHECK: const hipDoubleComplex** const dcomplexAarray_const = const_cast<const hipDoubleComplex**>(dcomplexAarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexBarray = 0;
  // CHECK-NEXT: const hipDoubleComplex** const dcomplexBarray_const = const_cast<const hipDoubleComplex**>(dcomplexBarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexXarray = 0;
  // CHECK-NEXT: const hipDoubleComplex** const dcomplexXarray_const = const_cast<const hipDoubleComplex**>(dcomplexXarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexYarray = 0;
  // CHECK-NEXT: const hipDoubleComplex** const dcomplexYarray_const = const_cast<const hipDoubleComplex**>(dcomplexYarray);
  // CHECK-NEXT: hipDoubleComplex** dcomplexCarray = 0;
  // CHECK-NEXT: hipDoubleComplex** dcomplexTauarray = 0;
  cuDoubleComplex** dcomplexAarray = 0;
  const cuDoubleComplex** const dcomplexAarray_const = const_cast<const cuDoubleComplex**>(dcomplexAarray);
  cuDoubleComplex** dcomplexBarray = 0;
  const cuDoubleComplex** const dcomplexBarray_const = const_cast<const cuDoubleComplex**>(dcomplexBarray);
  cuDoubleComplex** dcomplexXarray = 0;
  const cuDoubleComplex** const dcomplexXarray_const = const_cast<const cuDoubleComplex**>(dcomplexXarray);
  cuDoubleComplex** dcomplexYarray = 0;
  const cuDoubleComplex** const dcomplexYarray_const = const_cast<const cuDoubleComplex**>(dcomplexYarray);
  cuDoubleComplex** dcomplexCarray = 0;
  cuDoubleComplex** dcomplexTauarray = 0;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2(blasHandle, n, &complex, incx, &fresult);
  blasStatus = cublasScnrm2_v2(blasHandle, n, &complex, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDznrm2_v2(blasHandle, n, &dcomplex, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDznrm2_v2(blasHandle, n, &dcomplex, incx, &dresult);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotu_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* result);
  // CHECK: blasStatus = hipblasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = hipblasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotu_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotc_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* result);
  // CHECK: blasStatus = hipblasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  // CHECK-NEXT: blasStatus = hipblasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc(blasHandle, n, &complexx, incx, &complexy, incy, &complex);
  blasStatus = cublasCdotc_v2(blasHandle, n, &complexx, incx, &complexy, incy, &complex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotu_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = hipblasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  blasStatus = cublasZdotu_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotc_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotc_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
  // CHECK-NEXT: blasStatus = hipblasZdotc_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dcomplex);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCscal_v2(hipblasHandle_t handle, int n, const hipComplex* alpha, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCscal_v2(blasHandle, n, &complexa, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCscal_v2(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal(blasHandle, n, &complexa, &complexx, incx);
  blasStatus = cublasCscal_v2(blasHandle, n, &complexa, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle, int n, const float* alpha, cuComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsscal_v2(hipblasHandle_t handle, int n, const float* alpha, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCsscal_v2(blasHandle, n, &fx, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCsscal_v2(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal(blasHandle, n, &fx, &complexx, incx);
  blasStatus = cublasCsscal_v2(blasHandle, n, &fx, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZscal_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal(blasHandle, n, &dcomplexa, &dcomplexx, incx);
  blasStatus = cublasZscal_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle, int n, const double* alpha, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdscal_v2(hipblasHandle_t handle, int n, const double* alpha, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZdscal_v2(blasHandle, n, &dx, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZdscal_v2(blasHandle, n, &dx, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpy_v2(hipblasHandle_t handle, int n, const hipComplex* alpha, const hipComplex* x, int incx, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);
  blasStatus = cublasCaxpy_v2(blasHandle, n, &complexa, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpy_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZaxpy_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZaxpy_v2(blasHandle, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCcopy_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCcopy_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZcopy_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZcopy_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZcopy_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCswap_v2(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap(blasHandle, n, &complexx, incx, &complexy, incy);
  blasStatus = cublasCswap_v2(blasHandle, n, &complexx, incx, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZswap_v2(hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZswap_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZswap_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamax_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamax_v2(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIcamax_v2(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamax_v2(blasHandle, n, &complexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamax_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamax_v2(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIzamax_v2(blasHandle, n, &dcomplexx, incx, &res);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamin_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIcamin_v2(blasHandle, n, &complexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIcamin_v2(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin(blasHandle, n, &complexx, incx, &res);
  blasStatus = cublasIcamin_v2(blasHandle, n, &complexx, incx, &res);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamin_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, int* result);
  // CHECK: blasStatus = hipblasIzamin_v2(blasHandle, n, &dcomplexx, incx, &res);
  // CHECK-NEXT: blasStatus = hipblasIzamin_v2(blasHandle, n, &dcomplexx, incx, &res);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScasum_v2(hipblasHandle_t handle, int n, const hipComplex* x, int incx, float* result);
  // CHECK: blasStatus = hipblasScasum_v2(blasHandle, n, &complexx, incx, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScasum_v2(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum(blasHandle, n, &complexx, incx, &fresult);
  blasStatus = cublasScasum_v2(blasHandle, n, &complexx, incx, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDzasum_v2(hipblasHandle_t handle, int n, const hipDoubleComplex* x, int incx, double* result);
  // CHECK: blasStatus = hipblasDzasum_v2(blasHandle, n, &dcomplexx, incx, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDzasum_v2(blasHandle, n, &dcomplexx, incx, &dresult);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrot_v2(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy, const float* c, const hipComplex* s);
  // CHECK: blasStatus = hipblasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  // CHECK-NEXT: blasStatus = hipblasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);
  blasStatus = cublasCrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsrot_v2(hipblasHandle_t handle, int n, hipComplex* x, int incx, hipComplex* y, int incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);
  blasStatus = cublasCsrot_v2(blasHandle, n, &complexx, incx, &complexy, incy, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrot_v2(hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy, const double* c, const hipDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = hipblasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &dcomplexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdrot_v2(hipblasHandle_t handle, int n, hipDoubleComplex* x, int incx, hipDoubleComplex* y, int incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasZdrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasZdrot_v2(blasHandle, n, &dcomplexx, incx, &dcomplexy, incy, &dc, &ds);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrotg_v2(hipblasHandle_t handle, hipComplex* a, hipComplex* b, float* c, hipComplex* s);
  // CHECK: blasStatus = hipblasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);
  // CHECK-NEXT: blasStatus = hipblasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg(blasHandle, &complexa, &complexb, &fc, &complexs);
  blasStatus = cublasCrotg_v2(blasHandle, &complexa, &complexb, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrotg_v2(hipblasHandle_t handle, hipDoubleComplex* a, hipDoubleComplex* b, double* c, hipDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrotg_v2(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = hipblasZrotg_v2(blasHandle, &dcomplexa, &dcomplexb, &dc, &dcomplexs);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgemv_v2(blasHandle, blasOperation, m, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgemv_v2(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZgemv_v2(blasHandle, blasOperation, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmv_v2(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, int kl, int ku, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZgbmv_v2(blasHandle, blasOperation, m, n, kl, ku, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipComplex* AP, int lda, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipDoubleComplex* AP, int lda, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtrmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipComplex* AP, int lda, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int k, const hipDoubleComplex* AP, int lda, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtbmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipComplex* AP, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipDoubleComplex* AP, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtpmv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipComplex* AP, int lda, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipDoubleComplex* AP, int lda, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtrsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, lda, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipComplex* AP, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);
  blasStatus = cublasCtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &complexA, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex* AP, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, const hipDoubleComplex* AP, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtpsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, &dcomplexA, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipComplex* AP, int lda, hipComplex* x, int incx);
  // CHECK: blasStatus = hipblasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  // CHECK-NEXT: blasStatus = hipblasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);
  blasStatus = cublasCtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &complexA, lda, &complexx, incx);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int n, int k, const hipDoubleComplex* AP, int lda, hipDoubleComplex* x, int incx);
  // CHECK: blasStatus = hipblasZtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
  // CHECK-NEXT: blasStatus = hipblasZtbsv_v2(blasHandle, blasFillMode, blasOperation, blasDiagType, n, k, &dcomplexA, lda, &dcomplexx, incx);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasCsymv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZsymv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* x, int incx, const cuComplex* beta, cuComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChemv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhemv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhemv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChbmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChbmv_v2(blasHandle, blasFillMode, n, k, &complexa, &complexA, lda, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhbmv_v2(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhbmv_v2(blasHandle, blasFillMode, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* AP, const hipComplex* x, int incx, const hipComplex* beta, hipComplex* y, int incy);
  // CHECK: blasStatus = hipblasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  // CHECK-NEXT: blasStatus = hipblasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);
  blasStatus = cublasChpmv_v2(blasHandle, blasFillMode, n, &complexa, &complexA, &complexx, incx, &complexb, &complexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmv_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, const hipDoubleComplex* x, int incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int incy);
  // CHECK: blasStatus = hipblasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  // CHECK-NEXT: blasStatus = hipblasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);
  blasStatus = cublasZhpmv_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexA, &dcomplexx, incx, &dcomplexb, &dcomplexy, incy);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSger(hipblasHandle_t handle, int m, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP, int lda);
  // CHECK: blasStatus = hipblasSger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  // CHECK-NEXT: blasStatus = hipblasSger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSger(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSger_v2(blasHandle, m, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDger(hipblasHandle_t handle, int m, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP, int lda);
  // CHECK: blasStatus = hipblasDger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  // CHECK-NEXT: blasStatus = hipblasDger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDger(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDger_v2(blasHandle, m, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeru_v2(hipblasHandle_t handle, int m, int n, const hipComplex* alpha, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCgeru_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCgeru_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgeru(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgeru_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgerc_v2(hipblasHandle_t handle, int m, int n, const hipComplex* alpha, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCgerc_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCgerc_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgerc(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCgerc_v2(blasHandle, m, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeru_v2(hipblasHandle_t handle, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZgeru_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZgeru_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgeru(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgeru_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgerc_v2(hipblasHandle_t handle, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZgerc_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZgerc_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgerc(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZgerc_v2(blasHandle, m, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP, int lda);
  // CHECK: blasStatus = hipblasSsyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  // CHECK-NEXT: blasStatus = hipblasSsyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  blasStatus = cublasSsyr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);
  blasStatus = cublasSsyr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP, int lda);
  // CHECK: blasStatus = hipblasDsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  // CHECK-NEXT: blasStatus = hipblasDsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  blasStatus = cublasDsyr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);
  blasStatus = cublasDsyr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* x, int incx, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCsyr_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCsyr_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCsyr(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCsyr_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZsyr_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZsyr_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZsyr(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZsyr_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const hipComplex* x, int incx, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCher_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCher_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCher(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);
  blasStatus = cublasCher_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const hipDoubleComplex* x, int incx, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZher_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZher_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZher(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);
  blasStatus = cublasZher_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* AP);
  // CHECK: blasStatus = hipblasSspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  // CHECK-NEXT: blasStatus = hipblasSspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  blasStatus = cublasSspr(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);
  blasStatus = cublasSspr_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, double* AP);
  // CHECK: blasStatus = hipblasDspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  // CHECK-NEXT: blasStatus = hipblasDspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  blasStatus = cublasDspr(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);
  blasStatus = cublasDspr_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const cuComplex* x, int incx, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const hipComplex* x, int incx, hipComplex* AP);
  // CHECK: blasStatus = hipblasChpr_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  // CHECK-NEXT: blasStatus = hipblasChpr_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  blasStatus = cublasChpr(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);
  blasStatus = cublasChpr_v2(blasHandle, blasFillMode, n, &fa, &complexx, incx, &complexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const hipDoubleComplex* x, int incx, hipDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  // CHECK-NEXT: blasStatus = hipblasZhpr_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  blasStatus = cublasZhpr(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);
  blasStatus = cublasZhpr_v2(blasHandle, blasFillMode, n, &da, &dcomplexx, incx, &dcomplexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP, int lda);
  // CHECK: blasStatus = hipblasSsyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  // CHECK-NEXT: blasStatus = hipblasSsyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSsyr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);
  blasStatus = cublasSsyr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP, int lda);
  // CHECK: blasStatus = hipblasDsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  // CHECK-NEXT: blasStatus = hipblasDsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDsyr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);
  blasStatus = cublasDsyr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCsyr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCsyr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCsyr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCsyr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZsyr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZsyr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZsyr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZsyr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* AP, int lda);
  // CHECK: blasStatus = hipblasCher2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  // CHECK-NEXT: blasStatus = hipblasCher2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCher2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);
  blasStatus = cublasCher2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* AP, int lda);
  // CHECK: blasStatus = hipblasZher2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  // CHECK-NEXT: blasStatus = hipblasZher2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZher2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);
  blasStatus = cublasZher2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA, lda);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, const float* y, int incy, float* AP);
  // CHECK: blasStatus = hipblasSspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  // CHECK-NEXT: blasStatus = hipblasSspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  blasStatus = cublasSspr2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);
  blasStatus = cublasSspr2_v2(blasHandle, blasFillMode, n, &fa, &fx, incx, &fy, incy, &fA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const double* alpha, const double* x, int incx, const double* y, int incy, double* AP);
  // CHECK: blasStatus = hipblasDspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  // CHECK-NEXT: blasStatus = hipblasDspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  blasStatus = cublasDspr2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);
  blasStatus = cublasDspr2_v2(blasHandle, blasFillMode, n, &da, &dx, incx, &dy, incy, &dA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex* alpha, const cuComplex* x, int incx, const cuComplex* y, int incy, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipComplex* alpha, const hipComplex* x, int incx, const hipComplex* y, int incy, hipComplex* AP);
  // CHECK: blasStatus = hipblasChpr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  // CHECK-NEXT: blasStatus = hipblasChpr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  blasStatus = cublasChpr2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);
  blasStatus = cublasChpr2_v2(blasHandle, blasFillMode, n, &complexa, &complexx, incx, &complexy, incy, &complexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int incx, const hipDoubleComplex* y, int incy, hipDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  // CHECK-NEXT: blasStatus = hipblasZhpr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  blasStatus = cublasZhpr2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);
  blasStatus = cublasZhpr2_v2(blasHandle, blasFillMode, n, &dcomplexa, &dcomplexx, incx, &dcomplexy, incy, &dcomplexA);

  cublasOperation_t transa, transb;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = hipblasSgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSgemm(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSgemm_v2(blasHandle, transa, transb, m, n, k, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = hipblasDgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDgemm(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDgemm_v2(blasHandle, transa, transb, m, n, k, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemm_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCgemm_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCgemm_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCgemm(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCgemm_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemm_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZgemm_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZgemm_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZgemm(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZgemm_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const float* alpha, const float* const AP[], int lda, const float* const BP[], int ldb, const float* beta, float* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasSgemmBatched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);
  blasStatus = cublasSgemmBatched(blasHandle, transa, transb, m, n, k, &fa, fAarray_const, lda, fBarray_const, ldb, &fb, fCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha, const double* const Aarray[], int lda, const double* const Barray[], int ldb, const double* beta, double* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const double* alpha, const double* const AP[], int lda, const double* const BP[], int ldb, const double* beta, double* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasDgemmBatched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);
  blasStatus = cublasDgemmBatched(blasHandle, transa, transb, m, n, k, &da, dAarray_const, lda, dBarray_const, ldb, &db, dCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha, const cuComplex* const Aarray[], int lda, const cuComplex* const Barray[], int ldb, const cuComplex* beta, cuComplex* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipComplex* alpha, const hipComplex* const AP[], int lda, const hipComplex* const BP[], int ldb, const hipComplex* beta, hipComplex* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasCgemmBatched_v2(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);
  blasStatus = cublasCgemmBatched(blasHandle, transa, transb, m, n, k, &complexa, complexAarray_const, lda, complexBarray_const, ldb, &complexb, complexCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int lda, const cuDoubleComplex* const Barray[], int ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int lda, const hipDoubleComplex* const BP[], int ldb, const hipDoubleComplex* beta, hipDoubleComplex* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasZgemmBatched_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);
  blasStatus = cublasZgemmBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray_const, ldb, &dcomplexb, dcomplexCarray, ldc, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const float* AP, int lda, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = hipblasSsyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  blasStatus = cublasSsyrk(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);
  blasStatus = cublasSsyrk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrk(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const double* AP, int lda, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = hipblasDsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  blasStatus = cublasDsyrk(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);
  blasStatus = cublasDsyrk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrk_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyrk_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCsyrk_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrk(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrk_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrk_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyrk_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZsyrk_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrk(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrk_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const cuComplex* A, int lda, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherk_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const hipComplex* AP, int lda, const float* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCherk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCherk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  blasStatus = cublasCherk(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);
  blasStatus = cublasCherk_v2(blasHandle, blasFillMode, transa, n, k, &fa, &complexA, lda, &fb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const cuDoubleComplex* A, int lda, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherk_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const hipDoubleComplex* AP, int lda, const double* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZherk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZherk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  blasStatus = cublasZherk(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);
  blasStatus = cublasZherk_v2(blasHandle, blasFillMode, transa, n, k, &da, &dcomplexA, lda, &db, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = hipblasSsyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyr2k(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &fa, &fA, lda, &fb, ldb, &fb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2k(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = hipblasDsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyr2k(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);
  blasStatus = cublasDsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &da, &dA, lda, &db, ldb, &db, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2k_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyr2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2k_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZsyr2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyr2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &dcomplexb, &dcomplexC, ldc);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkx_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsyrkx_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsyrkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkx_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsyrkx_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsyrkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2k_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const float* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCher2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCher2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCher2k(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCher2k_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexb, ldb, &fb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2k_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const double* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZher2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZher2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZher2k(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZher2k_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexb, ldb, &db, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const float* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherkx_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const float* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCherkx_v2(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);
  blasStatus = cublasCherkx(blasHandle, blasFillMode, transa, n, k, &complexa, &complexA, lda, &complexB, ldb, &fb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherkx_v2(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const double* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZherkx_v2(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);
  blasStatus = cublasZherkx(blasHandle, blasFillMode, transa, n, k, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &db, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const float* alpha, const float* AP, int lda, const float* BP, int ldb, const float* beta, float* CP, int ldc);
  // CHECK: blasStatus = hipblasSsymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  // CHECK-NEXT: blasStatus = hipblasSsymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsymm(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);
  blasStatus = cublasSsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &fa, &fA, lda, &fB, ldb, &fb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const double* alpha, const double* AP, int lda, const double* BP, int ldb, const double* beta, double* CP, int ldc);
  // CHECK: blasStatus = hipblasDsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  // CHECK-NEXT: blasStatus = hipblasDsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsymm(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);
  blasStatus = cublasDsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &da, &dA, lda, &dB, ldb, &db, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsymm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasCsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsymm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZsymm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* BP, int ldb, const hipComplex* beta, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasChemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasChemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasChemm(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);
  blasStatus = cublasChemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* BP, int ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZhemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZhemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZhemm(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);
  blasStatus = cublasZhemm_v2(blasHandle, blasSideMode, blasFillMode, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexb, &dcomplexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const float* alpha, float* AP, int lda, float* BP, int ldb);
  // CHECK: blasStatus = hipblasStrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  // CHECK-NEXT: blasStatus = hipblasStrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  blasStatus = cublasStrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);
  blasStatus = cublasStrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, double* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const double* alpha, double* AP, int lda, double* BP, int ldb);
  // CHECK: blasStatus = hipblasDtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  // CHECK-NEXT: blasStatus = hipblasDtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  blasStatus = cublasDtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);
  blasStatus = cublasDtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, cuComplex* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipComplex* alpha, const hipComplex* AP, int lda, hipComplex* BP, int ldb);
  // CHECK: blasStatus = hipblasCtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  // CHECK-NEXT: blasStatus = hipblasCtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  blasStatus = cublasCtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);
  blasStatus = cublasCtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, hipDoubleComplex* BP, int ldb);
  // CHECK: blasStatus = hipblasZtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  // CHECK-NEXT: blasStatus = hipblasZtrsm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
  blasStatus = cublasZtrsm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeam_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipComplex* alpha, const hipComplex* AP, int lda, const hipComplex* beta, const hipComplex* BP, int ldb, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCgeam_v2(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCgeam(blasHandle, transa, transb, m, n, &complexa, &complexA, lda, &complexb, &complexB, ldb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeam_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* beta, const hipDoubleComplex* BP, int ldb, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZgeam_v2(blasHandle, transa, transb, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexb, &dcomplexB, ldb, &dcomplexC, ldc);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrfBatched_v2(hipblasHandle_t handle, const int n, hipComplex* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetrfBatched_v2(blasHandle, n, complexAarray, lda, &P, &info, batchCount);
  blasStatus = cublasCgetrfBatched(blasHandle, n, complexAarray, lda, &P, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrfBatched(cublasHandle_t handle, int n, cuDoubleComplex* const A[], int lda, int* P, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrfBatched_v2(hipblasHandle_t handle, const int n, hipDoubleComplex* const A[], const int lda, int* ipiv, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetrfBatched_v2(blasHandle, n, dcomplexAarray, lda, &P, &info, batchCount);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetriBatched_v2(hipblasHandle_t handle, const int n, hipComplex* const A[], const int lda, int* ipiv, hipComplex* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetriBatched_v2(blasHandle, n, complexAarray_const, lda, &P, complexCarray, ldc, &info, batchCount);
  blasStatus = cublasCgetriBatched(blasHandle, n, complexAarray_const, lda, &P, complexCarray, ldc, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetriBatched(cublasHandle_t handle, int n, const cuDoubleComplex* const A[], int lda, const int* P, cuDoubleComplex* const C[], int ldc, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetriBatched_v2(hipblasHandle_t handle, const int n, hipDoubleComplex* const A[], const int lda, int* ipiv, hipDoubleComplex* const C[], const int ldc, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetriBatched_v2(blasHandle, n, dcomplexAarray_const, lda, &P, dcomplexCarray, ldc, &info, batchCount);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgetrsBatched_v2(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, hipComplex* const A[], const int lda, const int* ipiv, hipComplex* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgetrsBatched_v2(blasHandle, transa, n, nrhs, complexAarray_const, lda, &P, complexBarray, ldb, &info, batchCount);
  blasStatus = cublasCgetrsBatched(blasHandle, transa, n, nrhs, complexAarray_const, lda, &P, complexBarray, ldb, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* const Aarray[], int lda, const int* devIpiv, cuDoubleComplex* const Barray[], int ldb, int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgetrsBatched_v2(hipblasHandle_t handle, const hipblasOperation_t trans, const int n, const int nrhs, hipDoubleComplex* const A[], const int lda, const int* ipiv, hipDoubleComplex* const B[], const int ldb, int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgetrsBatched_v2(blasHandle, transa, n, nrhs, dcomplexAarray_const, lda, &P, dcomplexBarray, ldb, &info, batchCount);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsmBatched_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA,hipblasDiagType_t diag, int m, int n, const hipComplex* alpha, const hipComplex* const AP[], int lda, hipComplex* const BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasCtrsmBatched_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);
  blasStatus = cublasCtrsmBatched(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, complexAarray_const, lda, complexBarray, ldb, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int lda, cuDoubleComplex* const B[], int ldb, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsmBatched_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int lda, hipDoubleComplex* const BP[], int ldb, int batchCount);
  // CHECK: blasStatus = hipblasZtrsmBatched_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, dcomplexAarray_const, lda, dcomplexBarray, ldb, batchCount);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeqrfBatched_v2(hipblasHandle_t handle, const int m, const int n, hipComplex* const A[], const int lda, hipComplex* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasCgeqrfBatched_v2(blasHandle, m, n, complexAarray, lda, complexTauarray, &info, batchCount);
  blasStatus = cublasCgeqrfBatched(blasHandle, m, n, complexAarray, lda, complexTauarray, &info, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeqrfBatched(cublasHandle_t handle, int m, int n, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const TauArray[], int* info, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeqrfBatched_v2(hipblasHandle_t handle, const int m, const int n, hipDoubleComplex* const A[], const int lda, hipDoubleComplex* const ipiv[], int* info, const int batchCount);
  // CHECK: blasStatus = hipblasZgeqrfBatched_v2(blasHandle, m, n, dcomplexAarray, lda, dcomplexTauarray, &info, batchCount);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmm_v2(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const hipComplex* AP, int lda, const hipComplex* x, int incx, hipComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasCdgmm_v2(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);
  blasStatus = cublasCdgmm(blasHandle, blasSideMode, m, n, &complexa, lda, &complexx, incx, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmm_v2(hipblasHandle_t handle, hipblasSideMode_t side, int m, int n, const hipDoubleComplex* AP, int lda, const hipDoubleComplex* x, int incx, hipDoubleComplex* CP, int ldc);
  // CHECK: blasStatus = hipblasZdgmm_v2(blasHandle, blasSideMode, m, n, &dcomplexa, lda, &dcomplexx, incx, &dcomplexC, ldc);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgelsBatched_v2(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, hipComplex* const A[], const int lda, hipComplex* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasCgelsBatched_v2(blasHandle, blasOperation, m, n, nrhs, complexAarray, lda, complexCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasCgelsBatched(blasHandle, blasOperation, m, n, nrhs, complexAarray, lda, complexCarray, ldc, &info, &deviceInfo, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgelsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int nrhs, cuDoubleComplex* const Aarray[], int lda, cuDoubleComplex* const Carray[], int ldc, int* info, int* devInfoArray, int batchSize);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgelsBatched_v2(hipblasHandle_t handle, hipblasOperation_t trans, const int m, const int n, const int nrhs, hipDoubleComplex* const A[], const int lda, hipDoubleComplex* const B[], const int ldb, int* info, int* deviceInfo, const int batchCount);
  // CHECK: blasStatus = hipblasZgelsBatched_v2(blasHandle, blasOperation, m, n, nrhs, dcomplexAarray, lda, dcomplexCarray, ldc, &info, &deviceInfo, batchCount);
  blasStatus = cublasZgelsBatched(blasHandle, blasOperation, m, n, nrhs, dcomplexAarray, lda, dcomplexCarray, ldc, &info, &deviceInfo, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc);
  // CHECK: blasStatus = hipblasStrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  // CHECK-NEXT: blasStatus = hipblasStrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  blasStatus = cublasStrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);
  blasStatus = cublasStrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &fa, &fA, lda, &fB, ldb, &fC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmm(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const double* alpha, const double* A, int lda, const double* B, int ldb, double* C, int ldc);
  // CHECK: blasStatus = hipblasDtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  // CHECK-NEXT: blasStatus = hipblasDtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  blasStatus = cublasDtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);
  blasStatus = cublasDtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &da, &dA, lda, &dB, ldb, &dC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B, int ldb, cuComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipComplex* alpha, const hipComplex* A, int lda, const hipComplex* B, int ldb, hipComplex* C, int ldc);
  // CHECK: blasStatus = hipblasCtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasCtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);
  blasStatus = cublasCtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &complexa, &complexA, lda, &complexB, ldb, &complexC, ldc);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, cuDoubleComplex* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmm_v2(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int m, int n, const hipDoubleComplex* alpha, const hipDoubleComplex* A, int lda, const hipDoubleComplex* B, int ldb, hipDoubleComplex* C, int ldc);
  // CHECK: blasStatus = hipblasZtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  // CHECK-NEXT: blasStatus = hipblasZtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZtrmm(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);
  blasStatus = cublasZtrmm_v2(blasHandle, blasSideMode, blasFillMode, transa, blasDiagType, m, n, &dcomplexa, &dcomplexA, lda, &dcomplexB, ldb, &dcomplexC, ldc);

  long long int strideA = 0;
  long long int strideB = 0;
  long long int strideC = 0;
  long long int strideX = 0;
  long long int strideY = 0;

#if CUDA_VERSION >= 7050
  // CHECK: __half* ha = 0;
  __half* ha = 0;
  // CHECK: __half* hA = 0;
  __half* hA = 0;
  // CHECK: __half* hb = 0;
  __half* hb = 0;
  // CHECK: __half* hB = 0;
  __half* hB = 0;
  // CHECK: __half* hc = 0;
  __half* hc = 0;
  // CHECK: __half* hC = 0;
  __half* hC = 0;
  // CHECK: __half* hx = 0;
  __half* hx = 0;
  // CHECK: __half* hy = 0;
  __half* hy = 0;

  // CHECK: __half** hAarray = 0;
  __half** hAarray = 0;
  // CHECK: const __half** const hAarray_const = const_cast<const __half**>(hAarray);
  const __half** const hAarray_const = const_cast<const __half**>(hAarray);
  // CHECK: __half** hBarray = 0;
  __half** hBarray = 0;
  // CHECK: const __half** const hBarray_const = const_cast<const __half**>(hBarray);
  const __half** const hBarray_const = const_cast<const __half**>(hBarray);
  // CHECK: __half** hCarray = 0;
  __half** hCarray = 0;
  // CHECK: const __half** const hCarray_const = const_cast<const __half**>(hCarray);
  const __half** const hCarray_const = const_cast<const __half**>(hCarray);
  // CHECK: __half** hxarray = 0;
  __half** hxarray = 0;
  // CHECK: const __half** const hxarray_const = const_cast<const __half**>(hxarray_const);
  const __half** const hxarray_const = const_cast<const __half**>(hxarray_const);
  // CHECK: __half** hyarray = 0;
  __half** hyarray = 0;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, const __half* B, int ldb, const __half* beta, __half* C, int ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemm(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* AP, int lda, const hipblasHalf* BP, int ldb, const hipblasHalf* beta, hipblasHalf* CP, int ldc);
  // CHECK: blasStatus = hipblasHgemm(blasHandle, transa, transb, m, n, k, ha, hA, lda, hB, ldb, hb, hC, ldc);
  blasStatus = cublasHgemm(blasHandle, transa, transb, m, n, k, ha, hA, lda, hB, ldb, hb, hC, ldc);
#endif

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType DataType;
  // CHECK-NEXT: hipDataType DataType_t;
  // CHECK-NEXT: hipDataType blasDataType;
  // CHECK-NEXT: hipDataType R_16F = HIP_R_16F;
  // CHECK-NEXT: hipDataType C_16F = HIP_C_16F;
  // CHECK-NEXT: hipDataType R_32F = HIP_R_32F;
  // CHECK-NEXT: hipDataType C_32F = HIP_C_32F;
  // CHECK-NEXT: hipDataType R_64F = HIP_R_64F;
  // CHECK-NEXT: hipDataType C_64F = HIP_C_64F;
  // CHECK-NEXT: hipDataType R_8I = HIP_R_8I;
  // CHECK-NEXT: hipDataType C_8I = HIP_C_8I;
  // CHECK-NEXT: hipDataType R_8U = HIP_R_8U;
  // CHECK-NEXT: hipDataType C_8U = HIP_C_8U;
  // CHECK-NEXT: hipDataType R_32I = HIP_R_32I;
  // CHECK-NEXT: hipDataType C_32I = HIP_C_32I;
  // CHECK-NEXT: hipDataType R_32U = HIP_R_32U;
  // CHECK-NEXT: hipDataType C_32U = HIP_C_32U;
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

  // CHECK: hipDataType DataType_2, DataType_3;
  cudaDataType DataType_2, DataType_3;

  // CHECK: hipDataType computeType;
  cudaDataType computeType;

  // CHECK: hipblasGemmAlgo_t blasGemmAlgo;
  // CHECK-NEXT: hipblasGemmAlgo_t BLAS_GEMM_DFALT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t blasGemmAlgo;
  cublasGemmAlgo_t BLAS_GEMM_DFALT = CUBLAS_GEMM_DFALT;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2Ex_v2(hipblasHandle_t handle, int n, const void* x, hipDataType xType, int incx, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasNrm2Ex_v2(blasHandle, n, image, DataType, incx, image_2, DataType_2, DataType_3);
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
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmStridedBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipComplex* alpha, const hipComplex* AP, int lda, long long strideA, const hipComplex* BP, int ldb, long long strideB, const hipComplex* beta, hipComplex* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasCgemmStridedBatched_v2(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);
  blasStatus = cublasCgemmStridedBatched(blasHandle, transa, transb, m, n, k, &complexa, &complexA, lda, strideA, &complexB, ldb, strideB, &complexb, &complexC, ldc, strideC, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda, long long int strideA, const cuDoubleComplex* B, int ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmStridedBatched_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int lda, long long strideA, const hipDoubleComplex* BP, int ldb, long long strideB, const hipDoubleComplex* beta, hipDoubleComplex* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasZgemmStridedBatched_v2(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);
  blasStatus = cublasZgemmStridedBatched(blasHandle, transa, transb, m, n, k, &dcomplexa, &dcomplexA, lda, strideA, &dcomplexB, ldb, strideB, &dcomplexb, &dcomplexC, ldc, strideC, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* A, int lda, long long int strideA, const __half* B, int ldb, long long int strideB, const __half* beta, __half* C, int ldc, long long int strideC, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* AP, int lda, long long strideA, const hipblasHalf* BP, int ldb, long long strideB, const hipblasHalf* beta, hipblasHalf* CP, int ldc, long long strideC, int batchCount);
  // CHECK: blasStatus = hipblasHgemmStridedBatched(blasHandle, transa, transb, m, n, k, ha, hA, lda, strideA, hB, ldb, strideB, hb, hC, ldc, strideC, batchCount);
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

  // CHECK: hipDataType Atype;
  // CHECK-NEXT: hipDataType Btype;
  // CHECK-NEXT: hipDataType Ctype;
  // CHECK-NEXT: hipDataType Xtype;
  // CHECK-NEXT: hipDataType Ytype;
  // CHECK-NEXT: hipDataType CStype;
  // CHECK-NEXT: hipDataType Executiontype;
  cudaDataType Atype;
  cudaDataType Btype;
  cudaDataType Ctype;
  cudaDataType Xtype;
  cudaDataType Ytype;
  cudaDataType CStype;
  cudaDataType Executiontype;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int incx, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScalEx_v2(hipblasHandle_t handle, int n, const void* alpha, hipDataType alphaType, void* x, hipDataType xType, int incx, hipDataType executionType);
  // CHECK: blasStatus = hipblasScalEx_v2(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);
  blasStatus = cublasScalEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx(cublasHandle_t handle, int n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyEx_v2(hipblasHandle_t handle, int n, const void* alpha, hipDataType alphaType, const void* x, hipDataType xType, int incx, void* y, hipDataType yType, int incy, hipDataType executionType);
  // CHECK: blasStatus = hipblasAxpyEx_v2(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);
  blasStatus = cublasAxpyEx(blasHandle, n, aptr, Atype, xptr, Xtype, incx, yptr, Ytype, incy, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotEx_v2(hipblasHandle_t handle, int n, const void* x, hipDataType xType, int incx, const void* y, hipDataType yType, int incy, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasDotEx_v2(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx(cublasHandle_t handle, int n, const void* x, cudaDataType xType, int incx, const void* y, cudaDataType yType, int incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotcEx_v2(hipblasHandle_t handle, int n, const void* x, hipDataType xType, int incx, const void* y, hipDataType yType, int incy, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasDotcEx_v2(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
  blasStatus = cublasDotcEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, image, DataType, Executiontype);
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 11000
  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, const void* B, hipDataType bType, int ldb, const void* beta, void* C, hipDataType cType, int ldc, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmEx_v2(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
  blasStatus = cublasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipblasGemmAlgo_t BLAS_GEMM_DEFAULT = HIPBLAS_GEMM_DEFAULT;
  cublasGemmAlgo_t BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half* alpha, const __half* const Aarray[], int lda, const __half* const Barray[], int ldb, const __half* beta, __half* const Carray[], int ldc, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmBatched(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const hipblasHalf* alpha, const hipblasHalf* const AP[], int lda, const hipblasHalf* const BP[], int ldb, const hipblasHalf* beta, hipblasHalf* const CP[], int ldc, int batchCount);
  // CHECK: blasStatus = hipblasHgemmBatched(blasHandle, transa, transb, m, n, k, ha, hAarray_const, lda, hBarray_const, ldb, hb, hCarray, ldc, batchCount);
  blasStatus = cublasHgemmBatched(blasHandle, transa, transb, m, n, k, ha, hAarray_const, lda, hBarray_const, ldb, hb, hCarray, ldc, batchCount);
#endif

#if CUDA_VERSION >= 9010 && CUDA_VERSION < 11000
  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,  int m, int n, int k, const void *alpha, const void *const Aarray[], cudaDataType Atype, int lda, const void *const Barray[], cudaDataType Btype, int ldb, const void *beta, void *const Carray[], cudaDataType Ctype, int ldc, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo);
  // HIP [Actual] HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A[], hipblasDatatype_t aType, int lda, const void* B[], hipblasDatatype_t bType, int ldb, const void* beta, void* C[], hipblasDatatype_t cType, int ldc, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A[], hipDataType aType, int lda, const void* B[], hipDataType bType, int ldb, const void* beta, void* C[], hipDataType cType, int ldc, int batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmBatchedEx_v2(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray_const, Atype, lda, voidBarray_const, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, computeType, blasGemmAlgo);

  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype, int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb, long long int strideB, const void *beta, void *C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo);
  // HIP [Actual] HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipblasDatatype_t aType, int lda, hipblasStride strideA, const void* B, hipblasDatatype_t bType, int ldb, hipblasStride strideB, const void* beta, void* C, hipblasDatatype_t cType, int ldc, hipblasStride strideC, int batchCount, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, hipblasStride strideA, const void* B, hipDataType bType, int ldb, hipblasStride strideB, const void* beta, void* C, hipDataType cType, int ldc, hipblasStride strideC, int batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmStridedBatchedEx_v2(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, computeType, blasGemmAlgo);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipblasOperation_t BLAS_OP_HERMITAN = HIPBLAS_OP_C;
  cublasOperation_t BLAS_OP_HERMITAN = CUBLAS_OP_HERMITAN;

  // CHECK: hipblasFillMode_t BLAS_FILL_MODE_FULL = HIPBLAS_FILL_MODE_FULL;
  cublasFillMode_t BLAS_FILL_MODE_FULL = CUBLAS_FILL_MODE_FULL;

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx(cublasHandle_t handle, int n, void* x, cudaDataType xType, int incx, void* y, cudaDataType yType, int incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasRotEx_v2(hipblasHandle_t handle, int n, void* x, hipDataType xType, int incx, void* y, hipDataType yType, int incy, const void* c, const void* s, hipDataType csType, hipDataType executionType);
  // CHECK: blasStatus = hipblasRotEx_v2(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
  blasStatus = cublasRotEx(blasHandle, n, xptr, Xtype, incx, yptr, Ytype, incy, cptr, sptr, CStype, Executiontype);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipDataType R_16BF = HIP_R_16BF;
  // CHECK-NEXT: hipDataType C_16BF = HIP_C_16BF;
  cublasDataType_t R_16BF = CUDA_R_16BF;
  cublasDataType_t C_16BF = CUDA_C_16BF;

  // CHECK: __hip_bfloat16* bf16A = nullptr;
  __nv_bfloat16* bf16A = nullptr;
  // CHECK: __hip_bfloat16** bf16Aarray = 0;
  __nv_bfloat16** bf16Aarray = 0;
  // CHECK: const __hip_bfloat16** const bf16Aarray_const = const_cast<const __hip_bfloat16**>(bf16Aarray);
  const __nv_bfloat16** const bf16Aarray_const = const_cast<const __nv_bfloat16**>(bf16Aarray);
  // CHECK: __hip_bfloat16* bf16X = nullptr;
  __nv_bfloat16* bf16X = nullptr;
  // CHECK: __hip_bfloat16** bf16xarray = 0;
  __nv_bfloat16** bf16xarray = 0;
  // CHECK: const __hip_bfloat16** const bf16xarray_const = const_cast<const __hip_bfloat16**>(bf16xarray);
  const __nv_bfloat16** const bf16xarray_const = const_cast<const __nv_bfloat16**>(bf16xarray);
  // CHECK: __hip_bfloat16* bf16Y = nullptr;
  __nv_bfloat16* bf16Y = nullptr;
  // CHECK: __hip_bfloat16** bf16yarray = 0;
  __nv_bfloat16** bf16yarray = 0;
  // CHECK: const __hip_bfloat16** const bf16yarray_const = const_cast<const __hip_bfloat16**>(bf16yarray);
  const __nv_bfloat16** const bf16yarray_const = const_cast<const __nv_bfloat16**>(bf16yarray);

  // CHECK: hipblasComputeType_t blasComputeType;
  cublasComputeType_t blasComputeType;

  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, const void* B, hipDataType bType, int ldb, const void* beta, void* C, hipDataType cType, int ldc, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmEx_v2(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, Bptr, Btype, ldb, bptr, Cptr, Ctype, ldc, blasComputeType, blasGemmAlgo);

  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda, const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A[], hipDataType aType, int lda, const void* B[], hipDataType bType, int ldb, const void* beta, void* C[], hipDataType cType, int ldc, int batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmBatchedEx_v2(blasHandle, transa, transb, m, n, k, aptr, voidAarray, Atype, lda, voidBarray, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx(blasHandle, transa, transb, m, n, k, aptr, voidAarray, Atype, lda, voidBarray, Btype, ldb, bptr, voidCarray, Ctype, ldc, batchCount, blasComputeType, blasGemmAlgo);

  // TODO: [HIPIFY] CUDA VERSION detection by HIPIFY itself
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx_v2(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, hipDataType aType, int lda, hipblasStride strideA, const void* B, hipDataType bType, int ldb, hipblasStride strideB, const void* beta, void* C, hipDataType cType, int ldc, hipblasStride strideC, int batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmStridedBatchedEx_v2(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx(blasHandle, transa, transb, m, n, k, aptr, Aptr, Atype, lda, strideA, Bptr, Btype, ldb, strideB, bptr, Cptr, Ctype, ldc, strideC, batchCount, blasComputeType, blasGemmAlgo);
#endif

#if CUDA_VERSION > 11060 && CUBLAS_VERSION >= 110902 // CUDA 11.6.2
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda, long long int strideA, const float* x, int incx, long long int stridex, const float* beta, float* y, int incy, long long int stridey, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, const float* alpha, const float* AP, int lda, hipblasStride strideA, const float* x, int incx, hipblasStride stridex, const float* beta, float* y, int incy, hipblasStride stridey, int batchCount);
  // CHECK: blasStatus = hipblasSgemvStridedBatched(blasHandle, blasOperation, m, n, &fa, &fA, lda, strideA, &fx, incx, strideX, &fb, &fy, incy, strideY, batchCount);
  blasStatus = cublasSgemvStridedBatched(blasHandle, blasOperation, m, n, &fa, &fA, lda, strideA, &fx, incx, strideX, &fb, &fy, incy, strideY, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvStridedBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* A, int lda, long long int strideA, const double* x, int incx, long long int stridex, const double* beta, double* y, int incy, long long int stridey, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvStridedBatched(hipblasHandle_t handle, hipblasOperation_t transA, int m, int n, const double* alpha, const double* AP, int lda, hipblasStride strideA, const double* x, int incx, hipblasStride stridex, const double* beta, double* y, int incy, hipblasStride stridey, int batchCount);
  // CHECK: blasStatus = hipblasDgemvStridedBatched(blasHandle, blasOperation, m, n, &da, &dA, lda, strideA, &dx, incx, strideX, &db, &dy, incy, strideY, batchCount);
  blasStatus = cublasDgemvStridedBatched(blasHandle, blasOperation, m, n, &da, &dA, lda, strideA, &dx, incx, strideX, &db, &dy, incy, strideY, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float* alpha, const float* const Aarray[], int lda, const float* const xarray[], int incx, const float* beta, float* const yarray[], int incy, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const float* alpha, const float* const AP[], int lda, const float* const x[], int incx, const float* beta, float* const y[], int incy, int batchCount);
  // CHECK: blasStatus = hipblasSgemvBatched(blasHandle, blasOperation, m, n, &fa, fAarray_const, lda, fXarray_const, incx, &fb, fYarray, incy, batchCount);
  blasStatus = cublasSgemvBatched(blasHandle, blasOperation, m, n, &fa, fAarray_const, lda, fXarray_const, incx, &fb, fYarray, incy, batchCount);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double* alpha, const double* const Aarray[], int lda, const double* const xarray[], int incx, const double* beta, double* const yarray[], int incy, int batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvBatched(hipblasHandle_t handle, hipblasOperation_t trans, int m, int n, const double* alpha, const double* const AP[], int lda, const double* const x[], int incx, const double* beta, double* const y[], int incy, int batchCount);
  // CHECK: blasStatus = hipblasDgemvBatched(blasHandle, blasOperation, m, n, &da, dAarray_const, lda, dXarray_const, incx, &db, dYarray, incy, batchCount);
  blasStatus = cublasDgemvBatched(blasHandle, blasOperation, m, n, &da, dAarray_const, lda, dXarray_const, incx, &db, dYarray, incy, batchCount);
#endif

#if CUDA_VERSION >= 12000
  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamax_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIsamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIsamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamax_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamax_v2_64(blasHandle, n_64, &fx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamax_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIdamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIdamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamax_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamax_v2_64(blasHandle, n_64, &dx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamax_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIcamax_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIcamax_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamax_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamax_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamax_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIzamax_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIzamax_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamax_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamax_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIsamin_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIsamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIsamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamin_64(blasHandle, n_64, &fx, incx_64, &res_64);
  blasStatus = cublasIsamin_v2_64(blasHandle, n_64, &fx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIdamin_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIdamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIdamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamin_64(blasHandle, n_64, &dx, incx_64, &res_64);
  blasStatus = cublasIdamin_v2_64(blasHandle, n_64, &dx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIcamin_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIcamin_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIcamin_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamin_64(blasHandle, n_64, &complexx, incx_64, &res_64);
  blasStatus = cublasIcamin_v2_64(blasHandle, n_64, &complexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, int64_t* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasIzamin_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, int64_t* result);
  // CHECK: blasStatus = hipblasIzamin_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  // CHECK-NEXT: blasStatus = hipblasIzamin_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamin_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);
  blasStatus = cublasIzamin_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &res_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSasum_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // CHECK: blasStatus = hipblasSasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSasum_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSasum_v2_64(blasHandle, n_64, &fx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDasum_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // CHECK: blasStatus = hipblasDasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDasum_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDasum_v2_64(blasHandle, n_64, &dx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScasum_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, float* result);
  // CHECK: blasStatus = hipblasScasum_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScasum_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScasum_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScasum_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDzasum_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, double* result);
  // CHECK: blasStatus = hipblasDzasum_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDzasum_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDzasum_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDzasum_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSaxpy_64(hipblasHandle_t handle, int64_t n, const float* alpha, const float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSaxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSaxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSaxpy_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSaxpy_v2_64(blasHandle, n_64, &fa, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDaxpy_64(hipblasHandle_t handle, int64_t n, const double* alpha, const double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDaxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDaxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDaxpy_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDaxpy_v2_64(blasHandle, n_64, &da, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCaxpy_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCaxpy_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCaxpy_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCaxpy_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCaxpy_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZaxpy_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZaxpy_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZaxpy_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZaxpy_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZaxpy_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScopy_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasScopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasScopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasScopy_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasScopy_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDcopy_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDcopy_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDcopy_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCcopy_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCcopy_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCcopy_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCcopy_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCcopy_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZcopy_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZcopy_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZcopy_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZcopy_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZcopy_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSdot_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);
  // CHECK: blasStatus = hipblasSdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  blasStatus = cublasSdot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);
  blasStatus = cublasSdot_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDdot_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, const double* y, int64_t incy, double* result);
  // CHECK: blasStatus = hipblasDdot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDdot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  blasStatus = cublasDdot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);
  blasStatus = cublasDdot_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotc_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* result);
  // CHECK: blasStatus = hipblasCdotc_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  // CHECK-NEXT: blasStatus = hipblasCdotc_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotc_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotc_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdotu_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* result);
  // CHECK: blasStatus = hipblasCdotu_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  // CHECK-NEXT: blasStatus = hipblasCdotu_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotu_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);
  blasStatus = cublasCdotu_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &complexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotc_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotc_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  // CHECK-NEXT: blasStatus = hipblasZdotc_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotc_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotc_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdotu_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* result);
  // CHECK: blasStatus = hipblasZdotu_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  // CHECK-NEXT: blasStatus = hipblasZdotu_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotu_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);
  blasStatus = cublasZdotu_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSnrm2_v2_64(cublasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSnrm2_64(hipblasHandle_t handle, int64_t n, const float* x, int64_t incx, float* result);
  // CHECK: blasStatus = hipblasSnrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = hipblasSnrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSnrm2_64(blasHandle, n_64, &fx, incx_64, &fresult);
  blasStatus = cublasSnrm2_v2_64(blasHandle, n_64, &fx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDnrm2_v2_64(cublasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDnrm2_64(hipblasHandle_t handle, int64_t n, const double* x, int64_t incx, double* result);
  // CHECK: blasStatus = hipblasDnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDnrm2_64(blasHandle, n_64, &dx, incx_64, &dresult);
  blasStatus = cublasDnrm2_v2_64(blasHandle, n_64, &dx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScnrm2_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* x, int64_t incx, float* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScnrm2_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* x, int64_t incx, float* result);
  // CHECK: blasStatus = hipblasScnrm2_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  // CHECK-NEXT: blasStatus = hipblasScnrm2_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScnrm2_64(blasHandle, n_64, &complexx, incx_64, &fresult);
  blasStatus = cublasScnrm2_v2_64(blasHandle, n_64, &complexx, incx_64, &fresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDznrm2_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* x, int64_t incx, double* result);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDznrm2_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* x, int64_t incx, double* result);
  // CHECK: blasStatus = hipblasDznrm2_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  // CHECK-NEXT: blasStatus = hipblasDznrm2_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDznrm2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);
  blasStatus = cublasDznrm2_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dresult);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrot_64(hipblasHandle_t handle, int64_t n,float* x, int64_t incx, float* y, int64_t incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasSrot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasSrot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  blasStatus = cublasSrot_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);
  blasStatus = cublasSrot_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrot_64(hipblasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasDrot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasDrot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  blasStatus = cublasDrot_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);
  blasStatus = cublasDrot_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const cuComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCrot_v2_64(hipblasHandle_t handle, int64_t n, hipComplex* x, int64_t incx, hipComplex* y, int64_t incy, const float* c, const hipComplex* s);
  // CHECK: blasStatus = hipblasCrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  // CHECK-NEXT: blasStatus = hipblasCrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  blasStatus = cublasCrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);
  blasStatus = cublasCrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &complexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy, const float* c, const float* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsrot_v2_64(hipblasHandle_t handle, int64_t n, hipComplex* x, int64_t incx, hipComplex* y, int64_t incy, const float* c, const float* s);
  // CHECK: blasStatus = hipblasCsrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  // CHECK-NEXT: blasStatus = hipblasCsrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  blasStatus = cublasCsrot_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);
  blasStatus = cublasCsrot_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64, &fc, &fs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const cuDoubleComplex* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZrot_v2_64(hipblasHandle_t handle, int64_t n, hipDoubleComplex* x, int64_t incx, hipDoubleComplex* y, int64_t incy, const double* c, const hipDoubleComplex* s);
  // CHECK: blasStatus = hipblasZrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  // CHECK-NEXT: blasStatus = hipblasZrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  blasStatus = cublasZrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);
  blasStatus = cublasZrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &dcomplexs);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy, const double* c, const double* s);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdrot_v2_64(hipblasHandle_t handle, int64_t n, hipDoubleComplex* x, int64_t incx, hipDoubleComplex* y, int64_t incy, const double* c, const double* s);
  // CHECK: blasStatus = hipblasZdrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  // CHECK-NEXT: blasStatus = hipblasZdrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  blasStatus = cublasZdrot_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);
  blasStatus = cublasZdrot_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64, &dc, &ds);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSrotm_64(hipblasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy, const float* param);
  // CHECK: blasStatus = hipblasSrotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  // CHECK-NEXT: blasStatus = hipblasSrotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  blasStatus = cublasSrotm_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);
  blasStatus = cublasSrotm_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64, &fparam);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDrotm_64(hipblasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy, const double* param);
  // CHECK: blasStatus = hipblasDrotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  // CHECK-NEXT: blasStatus = hipblasDrotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  blasStatus = cublasDrotm_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);
  blasStatus = cublasDrotm_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64, &dparam);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSscal_64(hipblasHandle_t handle, int64_t n, const float* alpha, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasSscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasSscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  blasStatus = cublasSscal_64(blasHandle, n_64, &fa, &fx, incx_64);
  blasStatus = cublasSscal_v2_64(blasHandle, n_64, &fa, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDscal_64(hipblasHandle_t handle, int64_t n, const double* alpha, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDscal_64(blasHandle, n_64, &da, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDscal_64(blasHandle, n_64, &da, &dx, incx_64);
  blasStatus = cublasDscal_64(blasHandle, n_64, &da, &dx, incx_64);
  blasStatus = cublasDscal_v2_64(blasHandle, n_64, &da, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2_64(cublasHandle_t handle, int64_t n, const cuComplex* alpha, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCscal_v2_64(hipblasHandle_t handle, int64_t n, const hipComplex* alpha, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCscal_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCscal_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  blasStatus = cublasCscal_64(blasHandle, n_64, &complexa, &complexx, incx_64);
  blasStatus = cublasCscal_v2_64(blasHandle, n_64, &complexa, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2_64(cublasHandle_t handle, int64_t n, const float* alpha, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsscal_v2_64(hipblasHandle_t handle, int64_t n, const float* alpha, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCsscal_v2_64(blasHandle, n_64, &fa, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCsscal_v2_64(blasHandle, n_64, &fa, &complexx, incx_64);
  blasStatus = cublasCsscal_64(blasHandle, n_64, &fa, &complexx, incx_64);
  blasStatus = cublasCsscal_v2_64(blasHandle, n_64, &fa, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2_64(cublasHandle_t handle, int64_t n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZscal_v2_64(hipblasHandle_t handle, int64_t n, const hipDoubleComplex* alpha, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZscal_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZscal_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  blasStatus = cublasZscal_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);
  blasStatus = cublasZscal_v2_64(blasHandle, n_64, &dcomplexa, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2_64(cublasHandle_t handle, int64_t n, const double* alpha, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdscal_v2_64(hipblasHandle_t handle, int64_t n, const double* alpha, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZdscal_v2_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZdscal_v2_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  blasStatus = cublasZdscal_64(blasHandle, n_64, &da, &dcomplexx, incx_64);
  blasStatus = cublasZdscal_v2_64(blasHandle, n_64, &da, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2_64(cublasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSswap_64(hipblasHandle_t handle, int64_t n, float* x, int64_t incx, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSswap_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);
  blasStatus = cublasSswap_v2_64(blasHandle, n_64, &fx, incx_64, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2_64(cublasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDswap_64(hipblasHandle_t handle, int64_t n, double* x, int64_t incx, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDswap_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);
  blasStatus = cublasDswap_v2_64(blasHandle, n_64, &dx, incx_64, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2_64(cublasHandle_t handle, int64_t n, cuComplex* x, int64_t incx, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCswap_v2_64(hipblasHandle_t handle, int64_t n, hipComplex* x, int64_t incx, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCswap_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCswap_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCswap_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);
  blasStatus = cublasCswap_v2_64(blasHandle, n_64, &complexx, incx_64, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2_64(cublasHandle_t handle, int64_t n, cuDoubleComplex* x, int64_t incx, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZswap_v2_64(hipblasHandle_t handle, int64_t n, hipDoubleComplex* x, int64_t incx, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZswap_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZswap_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZswap_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);
  blasStatus = cublasZswap_v2_64(blasHandle, n_64, &dcomplexx, incx_64, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgbmv_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const float* alpha, const float* AP, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgbmv_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const double* alpha, const double* AP, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgbmv_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgbmv_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, int64_t kl, int64_t ku, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZgbmv_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZgbmv_v2_64(blasHandle, blasOperation, m_64, n_64, kl_64, ku_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemv_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* AP, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSgemv_64(blasHandle, blasOperation, m_64, n_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSgemv_64(blasHandle, blasOperation, m_64, n_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSgemv_64(blasHandle, blasOperation, m_64, n_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &fa, &fAP, lda_64, &fx, incx_64, &fb, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemv_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* AP, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDgemv_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDgemv_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDgemv_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemv_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCgemv_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemv_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZgemv_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZgemv_v2_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const Aarray[], int64_t lda, const float* const xarray[], int64_t incx, const float* beta, float* const yarray[], int64_t incy, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvBatched_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* const AP[], int64_t lda, const float* const x[], int64_t incx, const float* beta, float* const y[], int64_t incy, int64_t batchCount);
  // CHECK: blasStatus = hipblasSgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &fa, fAarray_const, lda_64, fXarray_const, incx_64, &fb, fYarray, incy_64, batchCount_64);
  blasStatus = cublasSgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &fa, fAarray_const, lda_64, fXarray_const, incx_64, &fb, fYarray, incy_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const Aarray[], int64_t lda, const double* const xarray[], int64_t incx, const double* beta, double* const yarray[], int64_t incy, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvBatched_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* const AP[], int64_t lda, const double* const x[], int64_t incx, const double* beta, double* const y[], int64_t incy, int64_t batchCount);
  // CHECK: blasStatus = hipblasDgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &da, dAarray_const, lda_64, dXarray_const, incx_64, &db, dYarray, incy_64, batchCount_64);
  blasStatus = cublasDgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &da, dAarray_const, lda_64, dXarray_const, incx_64, &db, dYarray, incy_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const xarray[], int64_t incx, const cuComplex* beta, cuComplex* const yarray[], int64_t incy, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemvBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* const AP[], int64_t lda, const hipComplex* const x[], int64_t incx, const hipComplex* beta, hipComplex* const y[], int64_t incy, int64_t batchCount);
  // CHECK: blasStatus = hipblasCgemvBatched_v2_64(blasHandle, blasOperation, m_64, n_64, &complexa, complexAarray_const, lda_64, complexXarray_const, incx_64, &complexb, complexYarray, incy_64, batchCount_64);
  blasStatus = cublasCgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &complexa, complexAarray_const, lda_64, complexXarray_const, incx_64, &complexb, complexYarray, incy_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemvBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const xarray[], int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* const yarray[], int64_t incy, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemvBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t trans, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int64_t lda, const hipDoubleComplex* const x[], int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* const y[], int64_t incy, int64_t batchCount);
  // CHECK: blasStatus = hipblasZgemvBatched_v2_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexXarray_const, incx_64, &dcomplexb, dcomplexYarray, incy_64, batchCount_64);
  blasStatus = cublasZgemvBatched_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexXarray_const, incx_64, &dcomplexb, dcomplexYarray, incy_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* x, int64_t incx, long long int stridex, const float* beta, float* y, int64_t incy, long long int stridey, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemvStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, int64_t m, int64_t n, const float* alpha, const float* AP, int64_t lda, hipblasStride strideA, const float* x, int64_t incx, hipblasStride stridex, const float* beta, float* y, int64_t incy, hipblasStride stridey, int64_t batchCount);
  // CHECK: blasStatus = hipblasSgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &fa, &fA, lda_64, strideA, &fx, incx_64, strideX, &fb, &fy, incy_64, strideY, batchCount_64);
  blasStatus = cublasSgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &fa, &fA, lda_64, strideA, &fx, incx_64, strideX, &fb, &fy, incy_64, strideY, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* x, int64_t incx, long long int stridex, const double* beta, double* y, int64_t incy, long long int stridey, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemvStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, int64_t m, int64_t n, const double* alpha, const double* AP, int64_t lda, hipblasStride strideA, const double* x, int64_t incx, hipblasStride stridex, const double* beta, double* y, int64_t incy, hipblasStride stridey, int64_t batchCount);
  // CHECK: blasStatus = hipblasDgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, strideA, &dx, incx_64, strideX, &db, &dy, incy_64, strideY, batchCount_64);
  blasStatus = cublasDgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &da, &dA, lda_64, strideA, &dx, incx_64, strideX, &db, &dy, incy_64, strideY, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* x, int64_t incx, long long int stridex, const cuComplex* beta, cuComplex* y, int64_t incy, long long int stridey, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemvStridedBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, int64_t m, int64_t n,const hipComplex* alpha, const hipComplex* AP, int64_t lda,hipblasStride strideA, const hipComplex* x, int64_t incx, hipblasStride stridex, const hipComplex* beta, hipComplex* y, int64_t incy, hipblasStride stridey, int64_t batchCount);
  // CHECK: blasStatus = hipblasCgemvStridedBatched_v2_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, strideA, &complexx, incx_64, strideX, &complexb, &complexy, incy_64, strideY, batchCount_64);
  blasStatus = cublasCgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &complexa, &complexA, lda_64, strideA, &complexx, incx_64, strideX, &complexb, &complexy, incy_64, strideY, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemvStridedBatched_64(cublasHandle_t handle, cublasOperation_t trans, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* x, int64_t incx, long long int stridex, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy, long long int stridey, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemvStridedBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, hipblasStride strideA, const hipDoubleComplex* x, int64_t incx, hipblasStride stridex, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy, hipblasStride stridey, int64_t batchCount);
  // CHECK: blasStatus = hipblasZgemvStridedBatched_v2_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, strideA, &dcomplexx, incx_64, strideX, &dcomplexb, &dcomplexy, incy_64, strideY, batchCount_64);
  blasStatus = cublasZgemvStridedBatched_64(blasHandle, blasOperation, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, strideA, &dcomplexx, incx_64, strideX, &dcomplexb, &dcomplexy, incy_64, strideY, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSger_64(hipblasHandle_t handle, int64_t m, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP, int64_t lda);
  // CHECK: blasStatus = hipblasSger_64(blasHandle, m_64, n_64, &fa, &fx, incx_64, &fy, incy_64, &fAP, lda_64);
  // CHECK-NEXT: blasStatus = hipblasSger_64(blasHandle, m_64, n_64, &fa, &fx, incx_64, &fy, incy_64, &fAP, lda_64);
  blasStatus = cublasSger_64(blasHandle, m_64, n_64, &fa, &fx, incx_64, &fy, incy_64, &fAP, lda_64);
  blasStatus = cublasSger_v2_64(blasHandle, m_64, n_64, &fa, &fx, incx_64, &fy, incy_64, &fAP, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDger_64(hipblasHandle_t handle, int64_t m, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP, int64_t lda);
  // CHECK: blasStatus = hipblasDger_64(blasHandle, m_64, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasDger_64(blasHandle, m_64, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  blasStatus = cublasDger_64(blasHandle, m_64, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  blasStatus = cublasDger_v2_64(blasHandle, m_64, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeru_v2_64(hipblasHandle_t handle, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCgeru_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCgeru_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCgeru_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCgeru_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgerc_v2_64(hipblasHandle_t handle, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCgerc_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCgerc_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCgerc_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCgerc_v2_64(blasHandle, m_64, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeru_v2_64(hipblasHandle_t handle, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZgeru_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZgeru_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZgeru_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZgeru_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2_64(cublasHandle_t handle, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgerc_v2_64(hipblasHandle_t handle, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZgerc_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZgerc_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZgerc_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZgerc_v2_64(blasHandle, m_64, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChbmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasChbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasChbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChbmv_64(blasHandle, blasFillMode, n_64, k_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhbmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZhbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZhbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhbmv_64(blasHandle, blasFillMode, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasChemv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasChemv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChemv_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChemv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZhemv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZhemv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhemv_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhemv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const hipComplex* x, int64_t incx, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCher_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCher_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA, lda_64);
  blasStatus = cublasCher_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA, lda_64);
  blasStatus = cublasCher_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZher_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZher_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA, lda_64);
  blasStatus = cublasZher_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA, lda_64);
  blasStatus = cublasZher_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCher2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCher2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCher2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCher2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZher2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZher2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZher2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZher2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* AP, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* AP, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasChpmv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasChpmv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChpmv_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasChpmv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* AP, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZhpmv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZhpmv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhpmv_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZhpmv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const cuComplex* x, int64_t incx, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const hipComplex* x, int64_t incx, hipComplex* AP);
  // CHECK: blasStatus = hipblasChpr_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA);
  // CHECK-NEXT: blasStatus = hipblasChpr_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA);
  blasStatus = cublasChpr_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA);
  blasStatus = cublasChpr_v2_64(blasHandle, blasFillMode, n_64, &fa, &complexx, incx_64, &complexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA);
  // CHECK-NEXT: blasStatus = hipblasZhpr_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA);
  blasStatus = cublasZhpr_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA);
  blasStatus = cublasZhpr_v2_64(blasHandle, blasFillMode, n_64, &da, &dcomplexx, incx_64, &dcomplexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChpr2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* AP);
  // CHECK: blasStatus = hipblasChpr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA);
  // CHECK-NEXT: blasStatus = hipblasChpr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA);
  blasStatus = cublasChpr2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA);
  blasStatus = cublasChpr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhpr2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* AP);
  // CHECK: blasStatus = hipblasZhpr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA);
  // CHECK-NEXT: blasStatus = hipblasZhpr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA);
  blasStatus = cublasZhpr2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA);
  blasStatus = cublasZhpr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsbmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSsbmv_64(blasHandle, blasFillMode, n_64, k_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSsbmv_64(blasHandle, blasFillMode, n_64, k_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSsbmv_64(blasHandle, blasFillMode, n_64, k_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSsbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsbmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDsbmv_64(blasHandle, blasFillMode, n_64, k_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDsbmv_64(blasHandle, blasFillMode, n_64, k_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDsbmv_64(blasHandle, blasFillMode, n_64, k_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDsbmv_v2_64(blasHandle, blasFillMode, n_64, k_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSspmv_64(blasHandle, blasFillMode, n_64, &fa, &fA, &fx, incx_64, &fb, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSspmv_64(blasHandle, blasFillMode, n_64, &fa, &fA, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSspmv_64(blasHandle, blasFillMode, n_64, &fa, &fA, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSspmv_v2_64(blasHandle, blasFillMode, n_64, &fa, &fA, &fx, incx_64, &fb, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDspmv_64(blasHandle, blasFillMode, n_64, &da, &dA, &dx, incx_64, &db, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDspmv_64(blasHandle, blasFillMode, n_64, &da, &dA, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDspmv_64(blasHandle, blasFillMode, n_64, &da, &dA, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDspmv_v2_64(blasHandle, blasFillMode, n_64, &da, &dA, &dx, incx_64, &db, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP);
  // CHECK: blasStatus = hipblasSspr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA);
  // CHECK-NEXT: blasStatus = hipblasSspr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA);
  blasStatus = cublasSspr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA);
  blasStatus = cublasSspr_v2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP);
  // CHECK: blasStatus = hipblasDspr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA);
  // CHECK-NEXT: blasStatus = hipblasDspr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA);
  blasStatus = cublasDspr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA);
  blasStatus = cublasDspr_v2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSspr2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP);
  // CHECK: blasStatus = hipblasSspr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA);
  // CHECK-NEXT: blasStatus = hipblasSspr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA);
  blasStatus = cublasSspr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA);
  blasStatus = cublasSspr2_v2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDspr2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP);
  // CHECK: blasStatus = hipblasDspr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA);
  // CHECK-NEXT: blasStatus = hipblasDspr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA);
  blasStatus = cublasDspr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA);
  blasStatus = cublasDspr2_v2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* A, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* AP, int64_t lda, const float* x, int64_t incx, const float* beta, float* y, int64_t incy);
  // CHECK: blasStatus = hipblasSsymv_64(blasHandle, blasFillMode, n_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasSsymv_64(blasHandle, blasFillMode, n_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSsymv_64(blasHandle, blasFillMode, n_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);
  blasStatus = cublasSsymv_v2_64(blasHandle, blasFillMode, n_64, &fa, &fA, lda_64, &fx, incx_64, &fb, &fy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* A, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* AP, int64_t lda, const double* x, int64_t incx, const double* beta, double* y, int64_t incy);
  // CHECK: blasStatus = hipblasDsymv_64(blasHandle, blasFillMode, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasDsymv_64(blasHandle, blasFillMode, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDsymv_64(blasHandle, blasFillMode, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);
  blasStatus = cublasDsymv_v2_64(blasHandle, blasFillMode, n_64, &da, &dA, lda_64, &dx, incx_64, &db, &dy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, const cuComplex* beta, cuComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, const hipComplex* beta, hipComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasCsymv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasCsymv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCsymv_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);
  blasStatus = cublasCsymv_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexA, lda_64, &complexx, incx_64, &complexb, &complexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* beta, cuDoubleComplex* y, int64_t incy);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* beta, hipDoubleComplex* y, int64_t incy);
  // CHECK: blasStatus = hipblasZsymv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  // CHECK-NEXT: blasStatus = hipblasZsymv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZsymv_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);
  blasStatus = cublasZsymv_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexb, &dcomplexy, incy_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, float* AP, int64_t lda);
  // CHECK: blasStatus = hipblasSsyr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasSsyr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA, lda_64);
  blasStatus = cublasSsyr_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA, lda_64);
  blasStatus = cublasSsyr_v2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, double* AP, int64_t lda);
  // CHECK: blasStatus = hipblasDsyr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasDsyr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA, lda_64);
  blasStatus = cublasDsyr_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA, lda_64);
  blasStatus = cublasDsyr_v2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCsyr_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCsyr_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexA, lda_64);
  blasStatus = cublasCsyr_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexA, lda_64);
  blasStatus = cublasCsyr_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZsyr_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZsyr_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexA, lda_64);
  blasStatus = cublasZsyr_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexA, lda_64);
  blasStatus = cublasZsyr_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const float* alpha, const float* x, int64_t incx, const float* y, int64_t incy, float* AP, int64_t lda);
  // CHECK: blasStatus = hipblasSsyr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasSsyr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA, lda_64);
  blasStatus = cublasSsyr2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA, lda_64);
  blasStatus = cublasSsyr2_v2_64(blasHandle, blasFillMode, n_64, &fa, &fx, incx_64, &fy, incy_64, &fA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const double* alpha, const double* x, int64_t incx, const double* y, int64_t incy, double* AP, int64_t lda);
  // CHECK: blasStatus = hipblasDsyr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasDsyr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  blasStatus = cublasDsyr2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);
  blasStatus = cublasDsyr2_v2_64(blasHandle, blasFillMode, n_64, &da, &dx, incx_64, &dy, incy_64, &dA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuComplex* alpha, const cuComplex* x, int64_t incx, const cuComplex* y, int64_t incy, cuComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipComplex* alpha, const hipComplex* x, int64_t incx, const hipComplex* y, int64_t incy, hipComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasCsyr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasCsyr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCsyr2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);
  blasStatus = cublasCsyr2_v2_64(blasHandle, blasFillMode, n_64, &complexa, &complexx, incx_64, &complexy, incy_64, &complexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int64_t incx, const cuDoubleComplex* y, int64_t incy, cuDoubleComplex* A, int64_t lda);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* x, int64_t incx, const hipDoubleComplex* y, int64_t incy, hipDoubleComplex* AP, int64_t lda);
  // CHECK: blasStatus = hipblasZsyr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  // CHECK-NEXT: blasStatus = hipblasZsyr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZsyr2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);
  blasStatus = cublasZsyr2_v2_64(blasHandle, blasFillMode, n_64, &dcomplexa, &dcomplexx, incx_64, &dcomplexy, incy_64, &dcomplexA, lda_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const float* AP, int64_t lda, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const double* AP, int64_t lda, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const hipComplex* AP, int64_t lda, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const hipDoubleComplex* AP, int64_t lda, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtbmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtbmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const float* A, int64_t lda, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStbsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const float* AP, int64_t lda, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &fA, lda_64, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const double* A, int64_t lda, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtbsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const double* AP, int64_t lda, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dA, lda_64, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtbsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const hipComplex* AP, int64_t lda, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &complexA, lda_64, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, int64_t k, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtbsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, int64_t k, const hipDoubleComplex* AP, int64_t lda, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtbsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtbsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, k_64, &dcomplexA, lda_64, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  blasStatus = cublasStpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  blasStatus = cublasStpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  blasStatus = cublasDtpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  blasStatus = cublasDtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipComplex* AP, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  blasStatus = cublasCtpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  blasStatus = cublasCtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipDoubleComplex* AP, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  blasStatus = cublasZtpmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  blasStatus = cublasZtpmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStpsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const float* AP, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  blasStatus = cublasStpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);
  blasStatus = cublasStpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtpsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const double* AP, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  blasStatus = cublasDtpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);
  blasStatus = cublasDtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* AP, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtpsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipComplex* AP, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  blasStatus = cublasCtpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);
  blasStatus = cublasCtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* AP, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtpsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipDoubleComplex* AP, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  blasStatus = cublasZtpsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);
  blasStatus = cublasZtpsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const float* AP, int64_t lda, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const double* AP, int64_t lda, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipComplex* AP, int64_t lda, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipDoubleComplex* AP, int64_t lda, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtrmv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtrmv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const float* A, int64_t lda, float* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const float* AP, int64_t lda, float* x, int64_t incx);
  // CHECK: blasStatus = hipblasStrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasStrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);
  blasStatus = cublasStrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &fA, lda_64, &fx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const double* A, int64_t lda, double* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsv_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const double* AP, int64_t lda, double* x, int64_t incx);
  // CHECK: blasStatus = hipblasDtrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasDtrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);
  blasStatus = cublasDtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dA, lda_64, &dx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuComplex* A, int64_t lda, cuComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipComplex* AP, int64_t lda, hipComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasCtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasCtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);
  blasStatus = cublasCtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &complexA, lda_64, &complexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t n, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* x, int64_t incx);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsv_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t n, const hipDoubleComplex* AP, int64_t lda, hipDoubleComplex* x, int64_t incx);
  // CHECK: blasStatus = hipblasZtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  // CHECK-NEXT: blasStatus = hipblasZtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtrsv_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);
  blasStatus = cublasZtrsv_v2_64(blasHandle, blasFillMode, blasOperation, blasDiagType, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasAxpyEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, const void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasAxpyEx_v2_64(hipblasHandle_t handle, int64_t n, const void* alpha, hipDataType alphaType, const void* x, hipDataType xType, int64_t incx, void* y, hipDataType yType, int64_t incy, hipDataType executionType);
  // CHECK: blasStatus = hipblasAxpyEx_v2_64(blasHandle, n_64, aptr, Atype, xptr, Xtype, incx_64, yptr, Ytype, incy_64, Executiontype);
  blasStatus = cublasAxpyEx_64(blasHandle, n_64, aptr, Atype, xptr, Xtype, incx_64, yptr, Ytype, incy_64, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotEx_v2_64(hipblasHandle_t handle, int64_t n, const void* x, hipDataType xType, int64_t incx, const void* y, hipDataType yType, int64_t incy, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasDotEx_v2_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, image, DataType, Executiontype);
  blasStatus = cublasDotEx_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, image, DataType, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDotcEx_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, const void* y, cudaDataType yType, int64_t incy, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDotcEx_v2_64(hipblasHandle_t handle, int64_t n, const void* x, hipDataType xType, int64_t incx, const void* y, hipDataType yType, int64_t incy, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasDotcEx_v2_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, image, DataType, Executiontype);
  blasStatus = cublasDotcEx_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, image, DataType, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasNrm2Ex_64(cublasHandle_t handle, int64_t n, const void* x, cudaDataType xType, int64_t incx, void* result, cudaDataType resultType, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasNrm2Ex_v2_64(hipblasHandle_t handle, int64_t n, const void* x, hipDataType xType, int64_t incx, void* result, hipDataType resultType, hipDataType executionType);
  // CHECK: blasStatus = hipblasNrm2Ex_v2_64(blasHandle, n_64, xptr, Xtype, incx_64, image, DataType, Executiontype);
  blasStatus = cublasNrm2Ex_64(blasHandle, n_64, xptr, Xtype, incx_64, image, DataType, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasRotEx_64(cublasHandle_t handle, int64_t n, void* x, cudaDataType xType, int64_t incx, void* y, cudaDataType yType, int64_t incy, const void* c, const void* s, cudaDataType csType, cudaDataType executiontype);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasRotEx_v2_64(hipblasHandle_t handle, int64_t n, void* x, hipDataType xType, int64_t incx, void* y, hipDataType yType, int64_t incy, const void* c, const void* s, hipDataType csType, hipDataType executionType);
  // CHECK: blasStatus = hipblasRotEx_v2_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, cptr, sptr, CStype, Executiontype);
  blasStatus = cublasRotEx_64(blasHandle, n_64, xptr, Xtype, incx_64, yptr, Ytype, incy_64, cptr, sptr, CStype, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScalEx_64(cublasHandle_t handle, int64_t n, const void* alpha, cudaDataType alphaType, void* x, cudaDataType xType, int64_t incx, cudaDataType executionType);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasScalEx_v2_64(hipblasHandle_t handle, int64_t n, const void* alpha, hipDataType alphaType, void* x, hipDataType xType, int64_t incx, hipDataType executionType);
  // CHECK: blasStatus = hipblasScalEx_v2_64(blasHandle, n_64, aptr, Atype, xptr, Xtype, incx_64, Executiontype);
  blasStatus = cublasScalEx_64(blasHandle, n_64, aptr, Atype, xptr, Xtype, incx_64, Executiontype);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemm_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, const float* BP, int64_t ldb, const float* beta, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasSgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemm_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, const double* BP, int64_t ldb, const double* beta, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasDgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemm_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemm_v2_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemm_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZgemm_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemm_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* A, int64_t lda, const __half* B, int64_t ldb, const __half* beta, __half* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemm_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipblasHalf* alpha, const hipblasHalf* AP, int64_t lda, const hipblasHalf* BP, int64_t ldb, const hipblasHalf* beta, hipblasHalf* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasHgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hA, lda_64, hB, ldb_64, hb, hC, ldc_64);
  blasStatus = cublasHgemm_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hA, lda_64, hB, ldb_64, hb, hC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const Aarray[], int64_t lda, const float* const Barray[], int64_t ldb, const float* beta, float* const Carray[], int64_t ldc, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const float* alpha, const float* const AP[], int64_t lda, const float* const BP[], int64_t ldb, const float* beta, float* const CP[], int64_t ldc, int64_t batchCount);
  // CHECK: blasStatus = hipblasSgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, fAarray_const, lda_64, fBarray_const, ldb_64, &fb, fCarray, ldc_64, batchCount_64);
  blasStatus = cublasSgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, fAarray_const, lda_64, fBarray_const, ldb_64, &fb, fCarray, ldc_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const Aarray[], int64_t lda, const double* const Barray[], int64_t ldb, const double* beta, double* const Carray[], int64_t ldc, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const double* alpha, const double* const AP[], int64_t lda, const double* const BP[], int64_t ldb, const double* beta, double* const CP[], int64_t ldc, int64_t batchCount);
  // CHECK: blasStatus = hipblasDgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, dAarray_const, lda_64, dBarray_const, ldb_64, &db, dCarray, ldc_64, batchCount_64);
  blasStatus = cublasDgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, dAarray_const, lda_64, dBarray_const, ldb_64, &db, dCarray, ldc_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* const Aarray[], int64_t lda, const cuComplex* const Barray[], int64_t ldb, const cuComplex* beta, cuComplex* const Carray[], int64_t ldc, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* const AP[], int64_t lda, const hipComplex* const BP[], int64_t ldb, const hipComplex* beta, hipComplex* const CP[], int64_t ldc, int64_t batchCount);
  // CHECK: blasStatus = hipblasCgemmBatched_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, complexAarray_const, lda_64, complexBarray_const, ldb_64, &complexb, complexCarray, ldc_64, batchCount_64);
  blasStatus = cublasCgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, complexAarray_const, lda_64, complexBarray_const, ldb_64, &complexb, complexCarray, ldc_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* const Aarray[], int64_t lda, const cuDoubleComplex* const Barray[], int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* const Carray[], int64_t ldc, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int64_t lda, const hipDoubleComplex* const BP[], int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* const CP[], int64_t ldc, int64_t batchCount);
  // CHECK: blasStatus = hipblasZgemmBatched_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexBarray_const, ldb_64, &dcomplexb, dcomplexCarray, ldc_64, batchCount_64);
  blasStatus = cublasZgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexBarray_const, ldb_64, &dcomplexb, dcomplexCarray, ldc_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* const Aarray[], int64_t lda, const __half* const Barray[], int64_t ldb, const __half* beta, __half* const Carray[], int64_t ldc, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipblasHalf* alpha, const hipblasHalf* const AP[], int64_t lda, const hipblasHalf* const BP[], int64_t ldb, const hipblasHalf* beta, hipblasHalf* const CP[], int64_t ldc, int64_t batchCount);
  // CHECK: blasStatus = hipblasHgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hAarray_const, lda_64, hBarray_const, ldb_64, hb, hCarray, ldc_64, batchCount_64);
  blasStatus = cublasHgemmBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hAarray_const, lda_64, hBarray_const, ldb_64, hb, hCarray, ldc_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, long long int strideA, const float* B, int64_t ldb, long long int strideB, const float* beta, float* C, int64_t ldc, long long int strideC, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgemmStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, long long strideA, const float* BP, int64_t ldb, long long strideB, const float* beta, float* CP, int64_t ldc, long long strideC, int64_t batchCount);
  // CHECK: blasStatus = hipblasSgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, strideA, &fB, ldb_64, strideB, &fb, &fC, ldc_64, strideC, batchCount_64);
  blasStatus = cublasSgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &fa, &fA, lda_64, strideA, &fB, ldb_64, strideB, &fb, &fC, ldc_64, strideC, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, long long int strideA, const double* B, int64_t ldb, long long int strideB, const double* beta, double* C, int64_t ldc, long long int strideC, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgemmStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, long long strideA, const double* BP, int64_t ldb, long long strideB, const double* beta, double* CP, int64_t ldc, long long strideC, int64_t batchCount);
  // CHECK: blasStatus = hipblasDgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, strideA, &dB, ldb_64, strideB, &db, &dC, ldc_64, strideC, batchCount_64);
  blasStatus = cublasDgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &da, &dA, lda_64, strideA, &dB, ldb_64, strideB, &db, &dC, ldc_64, strideC, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, long long int strideA, const cuComplex* B, int64_t ldb, long long int strideB, const cuComplex* beta, cuComplex* C, int64_t ldc, long long int strideC, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgemmStridedBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, long long strideA, const hipComplex* BP, int64_t ldb, long long strideB, const hipComplex* beta, hipComplex* CP, int64_t ldc, long long strideC, int64_t batchCount);
  // CHECK: blasStatus = hipblasCgemmStridedBatched_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, strideA, &complexB, ldb_64, strideB, &complexb, &complexC, ldc_64, strideC, batchCount_64);
  blasStatus = cublasCgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &complexa, &complexA, lda_64, strideA, &complexB, ldb_64, strideB, &complexb, &complexC, ldc_64, strideC, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, long long int strideA, const cuDoubleComplex* B, int64_t ldb, long long int strideB, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc, long long int strideC, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgemmStridedBatched_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, long long strideA, const hipDoubleComplex* BP, int64_t ldb, long long strideB, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc, long long strideC, int64_t batchCount);
  // CHECK: blasStatus = hipblasZgemmStridedBatched_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, strideA, &dcomplexB, ldb_64, strideB, &dcomplexb, &dcomplexC, ldc_64, strideC, batchCount_64);
  blasStatus = cublasZgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, strideA, &dcomplexB, ldb_64, strideB, &dcomplexb, &dcomplexC, ldc_64, strideC, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasHgemmStridedBatched_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const __half* alpha, const __half* A, int64_t lda, long long int strideA, const __half* B, int64_t ldb, long long int strideB, const __half* beta, __half* C, int64_t ldc, long long int strideC, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasHgemmStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const hipblasHalf* alpha, const hipblasHalf* AP, int64_t lda, long long strideA, const hipblasHalf* BP, int64_t ldb, long long strideB, const hipblasHalf* beta, hipblasHalf* CP, int64_t ldc, long long strideC, int64_t batchCount);
  // CHECK: blasStatus = hipblasHgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hA, lda_64, strideA, hB, ldb_64, strideB, hb, hC, ldc_64, strideC, batchCount_64);
  blasStatus = cublasHgemmStridedBatched_64(blasHandle, transa, transb, m_64, n_64, k_64, ha, hA, lda_64, strideA, hB, ldb_64, strideB, hb, hC, ldc_64, strideC, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const cuComplex* A, int64_t lda, const float* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherk_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const float* alpha, const hipComplex* A, int64_t lda, const float* beta, hipComplex* C, int64_t ldc);
  // CHECK: blasStatus = hipblasCherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &complexA, lda_64, &fb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &complexA, lda_64, &fb, &complexC, ldc_64);
  blasStatus = cublasCherk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &complexA, lda_64, &fb, &complexC, ldc_64);
  blasStatus = cublasCherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &complexA, lda_64, &fb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const cuDoubleComplex* A, int64_t lda, const double* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherk_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const double* alpha, const hipDoubleComplex* AP, int64_t lda, const double* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dcomplexA, lda_64, &db, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dcomplexA, lda_64, &db, &dcomplexC, ldc_64);
  blasStatus = cublasZherk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dcomplexA, lda_64, &db, &dcomplexC, ldc_64);
  blasStatus = cublasZherk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dcomplexA, lda_64, &db, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCherkx_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const float* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCherkx_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);
  blasStatus = cublasCherkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZherkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZherkx_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const double* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZherkx_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);
  blasStatus = cublasZherkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const float* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCher2k_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const float* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);
  blasStatus = cublasCher2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);
  blasStatus = cublasCher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &fb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const double* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZher2k_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const double* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);
  blasStatus = cublasZher2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);
  blasStatus = cublasZher2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &db, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsymm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t m, int64_t n, const float* alpha, const float* AP, int64_t lda, const float* BP, int64_t ldb, const float* beta, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasSsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsymm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t m, int64_t n, const double* alpha, const double* AP, int64_t lda, const double* BP, int64_t ldb, const double* beta, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasDsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsymm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsymm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsymm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsymm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* beta, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrk_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, const float* beta, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fb, &fC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasSsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fb, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* beta, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrk_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, const double* beta, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &db, &dC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasDsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &db, &dC, ldc_64);
  blasStatus = cublasDsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &db, &dC, ldc_64);
  blasStatus = cublasDsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &db, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrk_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrk_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrk_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsyrk_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsyrk_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyr2k_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, const float* BP, int64_t ldb, const float* beta, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasSsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyr2k_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, const double* BP, int64_t ldb, const double* beta, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasDsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyr2k_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2k_v2_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyr2k_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsyr2k_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsyr2k_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSsyrkx_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const float* alpha, const float* AP, int64_t lda, const float* BP, int64_t ldb, const float* beta, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);
  blasStatus = cublasSsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &fa, &fA, lda_64, &fB, ldb_64, &fb, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, const double* beta, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDsyrkx_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const double* alpha, const double* AP, int64_t lda, const double* BP, int64_t ldb, const double* beta, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);
  blasStatus = cublasDsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &da, &dA, lda_64, &dB, ldb_64, &db, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCsyrkx_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCsyrkx_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasCsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyrkx_64(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZsyrkx_v2_64(hipblasHandle_t handle, hipblasFillMode_t uplo, hipblasOperation_t transA, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZsyrkx_v2_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZsyrkx_64(blasHandle, blasFillMode, blasOperation, n_64, k_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* beta, const float* B, int64_t ldb, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSgeam_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, const float* alpha, const float* AP, int64_t lda, const float* beta, const float* BP, int64_t ldb, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSgeam_64(blasHandle, transa, transb, m_64, n_64, &fa, &fA, lda_64, &fb, &fB, ldb_64, &fC, ldc_64);
  blasStatus = cublasSgeam_64(blasHandle, transa, transb, m_64, n_64, &fa, &fA, lda_64, &fb, &fB, ldb_64, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* beta, const double* B, int64_t ldb, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDgeam_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, const double* alpha, const double* AP, int64_t lda, const double* beta, const double* BP, int64_t ldb, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDgeam_64(blasHandle, transa, transb, m_64, n_64, &da, &dA, lda_64, &db, &dB, ldb_64, &dC, ldc_64);
  blasStatus = cublasDgeam_64(blasHandle, transa, transb, m_64, n_64, &da, &dA, lda_64, &db, &dB, ldb_64, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* beta, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCgeam_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* beta, const hipComplex* BP, int64_t ldb, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCgeam_v2_64(blasHandle, transa, transb, m_64, n_64, &complexa, &complexA, lda_64, &complexb, &complexB, ldb_64, &complexC, ldc_64);
  blasStatus = cublasCgeam_64(blasHandle, transa, transb, m_64, n_64, &complexa, &complexA, lda_64, &complexb, &complexB, ldb_64, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeam_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* beta, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZgeam_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* beta, const hipDoubleComplex* BP, int64_t ldb, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZgeam_v2_64(blasHandle, transa, transb, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexB, ldb_64, &dcomplexC, ldc_64);
  blasStatus = cublasZgeam_64(blasHandle, transa, transb, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexb, &dcomplexB, ldb_64, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, const cuComplex* beta, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasChemm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t n, int64_t k, const hipComplex* alpha, const hipComplex* AP, int64_t lda, const hipComplex* BP, int64_t ldb, const hipComplex* beta, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasChemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasChemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasChemm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);
  blasStatus = cublasChemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexb, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZhemm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, int64_t n, int64_t k, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* BP, int64_t ldb, const hipDoubleComplex* beta, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZhemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZhemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZhemm_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);
  blasStatus = cublasZhemm_v2_64(blasHandle, blasSideMode, blasFillMode, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexb, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrmm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float* C, int64_t ldc);
  // CHECK: blasStatus = hipblasStrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasStrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fC, ldc_64);
  blasStatus = cublasStrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fC, ldc_64);
  blasStatus = cublasStrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64, &fC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrmm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, const double* B, int64_t ldb, double* C, int64_t ldc);
  // CHECK: blasStatus = hipblasDtrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &dC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasDtrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &dC, ldc_64);
  blasStatus = cublasDtrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &dC, ldc_64);
  blasStatus = cublasDtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64, &dC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, const cuComplex* B, int64_t ldb, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrmm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* A, int64_t lda, const hipComplex* B, int64_t ldb, hipComplex* C, int64_t ldc);
  // CHECK: blasStatus = hipblasCtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasCtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexC, ldc_64);
  blasStatus = cublasCtrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexC, ldc_64);
  blasStatus = cublasCtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* B, int64_t ldb, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrmm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* A, int64_t lda, const hipDoubleComplex* B, int64_t ldb, hipDoubleComplex* C, int64_t ldc);
  // CHECK: blasStatus = hipblasZtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexC, ldc_64);
  // CHECK-NEXT: blasStatus = hipblasZtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexC, ldc_64);
  blasStatus = cublasZtrmm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexC, ldc_64);
  blasStatus = cublasZtrmm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* A, int64_t lda, float* B, int64_t ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* AP, int64_t lda, float* BP, int64_t ldb);
  // CHECK: blasStatus = hipblasStrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64);
  // CHECK-NEXT: blasStatus = hipblasStrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64);
  blasStatus = cublasStrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64);
  blasStatus = cublasStrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, &fA, lda_64, &fB, ldb_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,int64_t m, int64_t n, const double* alpha, const double* A, int64_t lda, double* B, int64_t ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsm_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* AP, int64_t lda, double* BP, int64_t ldb);
  // CHECK: blasStatus = hipblasDtrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64);
  // CHECK-NEXT: blasStatus = hipblasDtrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64);
  blasStatus = cublasDtrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64);
  blasStatus = cublasDtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, &dA, lda_64, &dB, ldb_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* A, int64_t lda, cuComplex* B, int64_t ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* AP, int64_t lda, hipComplex* BP, int64_t ldb);
  // CHECK: blasStatus = hipblasCtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64);
  // CHECK-NEXT: blasStatus = hipblasCtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64);
  blasStatus = cublasCtrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64);
  blasStatus = cublasCtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, &complexA, lda_64, &complexB, ldb_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsm_v2_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int64_t lda, cuDoubleComplex* B, int64_t ldb);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* AP, int64_t lda, hipDoubleComplex* BP, int64_t ldb);
  // CHECK: blasStatus = hipblasZtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64);
  // CHECK-NEXT: blasStatus = hipblasZtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64);
  blasStatus = cublasZtrsm_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64);
  blasStatus = cublasZtrsm_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, &dcomplexA, lda_64, &dcomplexB, ldb_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const A[], int64_t lda, float* const B[], int64_t ldb, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasStrsmBatched_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const float* alpha, const float* const AP[], int64_t lda, float* const BP[], int64_t ldb, int64_t batchCount);
  // CHECK: blasStatus = hipblasStrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, fAarray_const, lda_64, fBarray, ldb_64, batchCount_64);
  blasStatus = cublasStrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &fa, fAarray_const, lda_64, fBarray, ldb_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const A[], int64_t lda, double* const B[], int64_t ldb, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDtrsmBatched_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const double* alpha, const double* const AP[], int64_t lda, double* const BP[], int64_t ldb, int64_t batchCount);
  // CHECK: blasStatus = hipblasDtrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, dAarray_const, lda_64, dBarray, ldb_64, batchCount_64);
  blasStatus = cublasDtrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &da, dAarray_const, lda_64, dBarray, ldb_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuComplex* alpha, const cuComplex* const A[], int64_t lda, cuComplex* const B[], int64_t ldb, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCtrsmBatched_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipComplex* alpha, const hipComplex* const AP[], int64_t lda, hipComplex* const BP[], int64_t ldb, int64_t batchCount);
  // CHECK: blasStatus = hipblasCtrsmBatched_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, complexAarray_const, lda_64, complexBarray, ldb_64, batchCount_64);
  blasStatus = cublasCtrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &complexa, complexAarray_const, lda_64, complexBarray, ldb_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsmBatched_64(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int64_t m, int64_t n, const cuDoubleComplex* alpha, const cuDoubleComplex* const A[], int64_t lda, cuDoubleComplex* const B[], int64_t ldb, int64_t batchCount);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZtrsmBatched_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, hipblasFillMode_t uplo, hipblasOperation_t transA, hipblasDiagType_t diag, int64_t m, int64_t n, const hipDoubleComplex* alpha, const hipDoubleComplex* const AP[], int64_t lda, hipDoubleComplex* const BP[], int64_t ldb, int64_t batchCount);
  // CHECK: blasStatus = hipblasZtrsmBatched_v2_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexBarray, ldb_64, batchCount_64);
  blasStatus = cublasZtrsmBatched_64(blasHandle, blasSideMode, blasFillMode, blasOperation, blasDiagType, m_64, n_64, &dcomplexa, dcomplexAarray_const, lda_64, dcomplexBarray, ldb_64, batchCount_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const float* A, int64_t lda, const float* x, int64_t incx, float* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasSdgmm_64(hipblasHandle_t handle, hipblasSideMode_t side, int64_t m, int64_t n, const float* AP, int64_t lda, const float* x, int64_t incx, float* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasSdgmm_64(blasHandle, blasSideMode, m_64, n_64, &fAP, lda_64, &fx, incx_64, &fCP, ldc_64);
  blasStatus = cublasSdgmm_64(blasHandle, blasSideMode, m_64, n_64, &fAP, lda_64, &fx, incx_64, &fCP, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const double* A, int64_t lda, const double* x, int64_t incx, double* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasDdgmm_64(hipblasHandle_t handle, hipblasSideMode_t side, int64_t m, int64_t n, const double* AP, int64_t lda, const double* x, int64_t incx, double* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasDdgmm_64(blasHandle, blasSideMode, m_64, n_64, &dAP, lda_64, &dx, incx_64, &dCP, ldc_64);
  blasStatus = cublasDdgmm_64(blasHandle, blasSideMode, m_64, n_64, &dAP, lda_64, &dx, incx_64, &dCP, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuComplex* A, int64_t lda, const cuComplex* x, int64_t incx, cuComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasCdgmm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, int64_t m, int64_t n, const hipComplex* AP, int64_t lda, const hipComplex* x, int64_t incx, hipComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasCdgmm_v2_64(blasHandle, blasSideMode, m_64, n_64, &complexA, lda_64, &complexx, incx_64, &complexC, ldc_64);
  blasStatus = cublasCdgmm_64(blasHandle, blasSideMode, m_64, n_64, &complexA, lda_64, &complexx, incx_64, &complexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdgmm_64(cublasHandle_t handle, cublasSideMode_t mode, int64_t m, int64_t n, const cuDoubleComplex* A, int64_t lda, const cuDoubleComplex* x, int64_t incx, cuDoubleComplex* C, int64_t ldc);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasZdgmm_v2_64(hipblasHandle_t handle, hipblasSideMode_t side, int64_t m, int64_t n, const hipDoubleComplex* AP, int64_t lda, const hipDoubleComplex* x, int64_t incx, hipDoubleComplex* CP, int64_t ldc);
  // CHECK: blasStatus = hipblasZdgmm_v2_64(blasHandle, blasSideMode, m_64, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexC, ldc_64);
  blasStatus = cublasZdgmm_64(blasHandle, blasSideMode, m_64, n_64, &dcomplexA, lda_64, &dcomplexx, incx_64, &dcomplexC, ldc_64);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype, int64_t ldb, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmEx_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, hipDataType aType, int64_t lda, const void* B, hipDataType bType, int64_t ldb, const void* beta, void* C, hipDataType cType, int64_t ldc, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmEx_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, Aptr, Atype, lda_64, Bptr, Btype, ldb_64, bptr, Cptr, Ctype, ldc_64, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmEx_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, Aptr, Atype, lda_64, Bptr, Btype, ldb_64, bptr, Cptr, Ctype, ldc_64, blasComputeType, blasGemmAlgo);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* const Aarray[], cudaDataType Atype, int64_t lda, const void* const Barray[], cudaDataType Btype, int64_t ldb, const void* beta, void* const Carray[], cudaDataType Ctype, int64_t ldc, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmBatchedEx_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A[], hipDataType aType, int64_t lda, const void* B[], hipDataType bType, int64_t ldb, const void* beta, void* C[], hipDataType cType, int64_t ldc, int64_t batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmBatchedEx_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, voidAarray_const, Atype, lda_64, voidBarray_const, Btype, ldb_64, bptr, voidCarray, Ctype, ldc_64, batchCount_64, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmBatchedEx_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, voidAarray_const, Atype, lda_64, voidBarray_const, Btype, ldb_64, bptr, voidCarray, Ctype, ldc_64, batchCount_64, blasComputeType, blasGemmAlgo);

  // CUDA: CUBLASAPI cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx_64(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype, int64_t lda, long long int strideA, const void* B, cudaDataType Btype, int64_t ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long int strideC, int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
  // HIP: HIPBLAS_EXPORT hipblasStatus_t hipblasGemmStridedBatchedEx_v2_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int64_t m, int64_t n, int64_t k, const void* alpha, const void* A, hipDataType aType, int64_t lda, hipblasStride strideA, const void* B, hipDataType bType, int64_t ldb, hipblasStride strideB, const void* beta, void* C, hipDataType cType, int64_t ldc, hipblasStride strideC, int64_t batchCount, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
  // CHECK: blasStatus = hipblasGemmStridedBatchedEx_v2_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, Aptr, Atype, lda_64, strideA, Bptr, Btype, ldb_64, strideB, bptr, Cptr, Ctype, ldc_64, strideC, batchCount_64, blasComputeType, blasGemmAlgo);
  blasStatus = cublasGemmStridedBatchedEx_64(blasHandle, transa, transb, m_64, n_64, k_64, aptr, Aptr, Atype, lda_64, strideA, Bptr, Btype, ldb_64, strideB, bptr, Cptr, Ctype, ldc_64, strideC, batchCount_64, blasComputeType, blasGemmAlgo);
#endif

  return 0;
}
