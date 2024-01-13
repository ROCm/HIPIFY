// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

int main() {
  printf("18.1. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: rocsparse_status status_t;
  cusparseStatus_t status_t;

  int batchCount = 0;
  int m = 0;
  int algo = 0;
  double dds = 0.f;
  double ddl = 0.f;
  double dd = 0.f;
  double ddu = 0.f;
  double ddw = 0.f;
  double dx = 0.f;
  float fds = 0.f;
  float fdl = 0.f;
  float fd = 0.f;
  float fdu = 0.f;
  float fdw = 0.f;
  float fx = 0.f;
  size_t bufferSize = 0;
  void *pBuffer = nullptr;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuDoubleComplex -> rocsparse_double_complex under a new option --sparse
  // CHECK: rocblas_double_complex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;
  cuDoubleComplex dcomplex, dcomplexA, dcomplexB, dComplexbsrSortedValA, dComplexbsrSortedValC, dComplexcsrSortedValA, dComplexcsrSortedValB, dComplexcsrSortedValC, dcomplextol, dComplexbsrSortedVal, dComplexbscVal, dComplexcscSortedVal, dcomplexds, dcomplexdl, dcomplexd, dcomplexdu, dcomplexdw, dcomplexx, dcomplex_boost_val;

  // TODO: should be rocsparse_double_complex
  // TODO: add to TypeOverloads cuComplex -> rocsparse_float_complex under a new option --sparse
  // CHECK: rocblas_float_complex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;
  cuComplex complex, complexA, complexB, complexbsrValA, complexbsrSortedValC, complexcsrSortedValA, complexcsrSortedValB, complexcsrSortedValC, complextol, complexbsrSortedVal, complexbscVal, complexcscSortedVal, complexds, complexdl, complexd, complexdu, complexdw, complexx, complex_boost_val;

#if CUDA_VERSION >= 9020
  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_zgpsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseZgpsvInterleavedBatch calls rocsparse_zgpsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* ds, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* dw, cuDoubleComplex* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgpsv_interleaved_batch(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, rocsparse_double_complex* ds, rocsparse_double_complex* dl, rocsparse_double_complex* d, rocsparse_double_complex* du, rocsparse_double_complex* dw, rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgpsv_interleaved_batch(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, batchCount, pBuffer);
  status_t = cusparseZgpsvInterleavedBatch(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_cgpsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseCgpsvInterleavedBatch calls rocsparse_cgpsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* ds, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* dw, cuComplex* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgpsv_interleaved_batch(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, rocsparse_float_complex* ds, rocsparse_float_complex* dl, rocsparse_float_complex* d, rocsparse_float_complex* du, rocsparse_float_complex* dw, rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgpsv_interleaved_batch(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, batchCount, pBuffer);
  status_t = cusparseCgpsvInterleavedBatch(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_dgpsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseDgpsvInterleavedBatch calls rocsparse_dgpsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgpsv_interleaved_batch(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, double* ds, double* dl, double* d, double* du, double* dw, double* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgpsv_interleaved_batch(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, batchCount, pBuffer);
  status_t = cusparseDgpsvInterleavedBatch(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_sgpsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseSgpsvInterleavedBatch calls rocsparse_sgpsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgpsv_interleaved_batch(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, float* ds, float* dl, float* d, float* du, float* dw, float* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgpsv_interleaved_batch(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, batchCount, pBuffer);
  status_t = cusparseSgpsvInterleavedBatch(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_zgpsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseSgpsvInterleavedBatch calls rocsparse_zgpsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* ds, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* dw, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgpsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, const rocsparse_double_complex* ds, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, const rocsparse_double_complex* dw, const rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgpsv_interleaved_batch_buffer_size(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, batchCount, &bufferSize);
  status_t = cusparseZgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexds, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexdw, &dcomplexx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_cgpsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseCgpsvInterleavedBatch_bufferSizeExt calls rocsparse_cgpsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* ds, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* dw, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgpsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, const rocsparse_float_complex* ds, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, const rocsparse_float_complex* dw, const rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgpsv_interleaved_batch_buffer_size(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, batchCount, &bufferSize);
  status_t = cusparseCgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexds, &complexdl, &complexd, &complexdu, &complexdw, &complexx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_dgpsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseDgpsvInterleavedBatch_bufferSizeExt calls rocsparse_dgpsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgpsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, const double* ds, const double* dl, const double* d, const double* du, const double* dw, const double* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgpsv_interleaved_batch_buffer_size(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, batchCount, &bufferSize);
  status_t = cusparseDgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dds, &ddl, &dd, &ddu, &ddw, &dx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_dgpsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseSgpsvInterleavedBatch_bufferSizeExt calls rocsparse_dgpsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgpsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gpsv_interleaved_alg alg, rocsparse_int m, const float* ds, const float* dl, const float* d, const float* du, const float* dw, const float* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgpsv_interleaved_batch_buffer_size(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, batchCount, &bufferSize);
  status_t = cusparseSgpsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fds, &fdl, &fd, &fdu, &fdw, &fx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_zgtsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseZgtsvInterleavedBatch calls rocsparse_zgtsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuDoubleComplex* dl, cuDoubleComplex* d, cuDoubleComplex* du, cuDoubleComplex* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_interleaved_batch(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, rocsparse_double_complex* dl, rocsparse_double_complex* d, rocsparse_double_complex* du, rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_zgtsv_interleaved_batch(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, batchCount, pBuffer);
  status_t = cusparseZgtsvInterleavedBatch(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_zgtsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseCgtsvInterleavedBatch calls rocsparse_zgtsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_interleaved_batch(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, rocsparse_float_complex* dl, rocsparse_float_complex* d, rocsparse_float_complex* du, rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_cgtsv_interleaved_batch(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, batchCount, pBuffer);
  status_t = cusparseCgtsvInterleavedBatch(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_dgtsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseDgtsvInterleavedBatch calls rocsparse_dgtsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, double* dl, double* d, double* du, double* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_interleaved_batch(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, double* dl, double* d, double* du, double* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_dgtsv_interleaved_batch(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, batchCount, pBuffer);
  status_t = cusparseDgtsvInterleavedBatch(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_sgtsv_interleaved_batch function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseSgtsvInterleavedBatch calls rocsparse_sgtsv_interleaved_batch in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch(cusparseHandle_t handle, int algo, int m, float* dl, float* d, float* du, float* x, int batchCount, void* pBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_interleaved_batch(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, float* dl, float* d, float* du, float* x, rocsparse_int batch_count, rocsparse_int batch_stride, void* temp_buffer);
  // CHECK: status_t = rocsparse_sgtsv_interleaved_batch(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, batchCount, pBuffer);
  status_t = cusparseSgtsvInterleavedBatch(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, pBuffer);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_zgtsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseZgtsvInterleavedBatch_bufferSizeExt calls rocsparse_zgtsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuDoubleComplex* dl, const cuDoubleComplex* d, const cuDoubleComplex* du, const cuDoubleComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgtsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, const rocsparse_double_complex* dl, const rocsparse_double_complex* d, const rocsparse_double_complex* du, const rocsparse_double_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgtsv_interleaved_batch_buffer_size(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, batchCount, &bufferSize);
  status_t = cusparseZgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &dcomplexdl, &dcomplexd, &dcomplexdu, &dcomplexx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_cgtsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseCgtsvInterleavedBatch_bufferSizeExt calls rocsparse_cgtsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const cuComplex* dl, const cuComplex* d, const cuComplex* du, const cuComplex* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgtsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, const rocsparse_float_complex* dl, const rocsparse_float_complex* d, const rocsparse_float_complex* du, const rocsparse_float_complex* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgtsv_interleaved_batch_buffer_size(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, batchCount, &bufferSize);
  status_t = cusparseCgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &complexdl, &complexd, &complexdu, &complexx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_cgtsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseDgtsvInterleavedBatch_bufferSizeExt calls rocsparse_cgtsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const double* dl, const double* d, const double* du, const double* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgtsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, const double* dl, const double* d, const double* du, const double* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgtsv_interleaved_batch_buffer_size(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, batchCount, &bufferSize);
  status_t = cusparseDgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &ddl, &dd, &ddu, &dx, batchCount, &bufferSize);

  // NOTE: An additional argument rocsparse_int batch_stride is added for the rocsparse_sgtsv_interleaved_batch_buffer_size function call: the argument is copied from the previous one: rocsparse_int batch_count. It is how hipsparseSgtsvInterleavedBatch_bufferSizeExt calls rocsparse_sgtsv_interleaved_batch_buffer_size in its implementation.
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle, int algo, int m, const float* dl, const float* d, const float* du, const float* x, int batchCount, size_t* pBufferSizeInBytes);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgtsv_interleaved_batch_buffer_size(rocsparse_handle handle, rocsparse_gtsv_interleaved_alg alg, rocsparse_int m, const float* dl, const float* d, const float* du, const float* x, rocsparse_int batch_count, rocsparse_int batch_stride, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgtsv_interleaved_batch_buffer_size(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, batchCount, &bufferSize);
  status_t = cusparseSgtsvInterleavedBatch_bufferSizeExt(handle_t, algo, m, &fdl, &fd, &fdu, &fx, batchCount, &bufferSize);
#endif

  return 0;
}
