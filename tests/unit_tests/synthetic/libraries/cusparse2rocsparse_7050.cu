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

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

  int batchCount = 0;
  int m = 0;
  int n = 0;
  int innz = 0;
  int algo = 0;
  int bufferSizeInBytes = 0;
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

#if CUDA_VERSION >= 7050
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseZgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_zgemvi_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, size_t* buffer_size);
  // CHECK: status_t = rocsparse_zgemvi_buffer_size(handle_t, opA, m, n, innz, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseZgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_cgemvi_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, size_t* buffer_size);
  // CHECK: status_t = rocsparse_cgemvi_buffer_size(handle_t, opA, m, n, innz, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseCgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_dgemvi_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, size_t* buffer_size);
  // CHECK: status_t = rocsparse_dgemvi_buffer_size(handle_t, opA, m, n, innz, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseDgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSgemvi_bufferSize(cusparseHandle_t handle, cusparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_sgemvi_buffer_size(rocsparse_handle handle, rocsparse_operation trans, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, size_t* buffer_size);
  // CHECK: status_t = rocsparse_sgemvi_buffer_size(handle_t, opA, m, n, innz, reinterpret_cast<size_t*>(&bufferSizeInBytes));
  status_t = cusparseSgemvi_bufferSize(handle_t, opA, m, n, innz, &bufferSizeInBytes);
#endif

  return 0;
}
