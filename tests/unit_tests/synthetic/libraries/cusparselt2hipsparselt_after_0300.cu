// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "hipsparseLt.h"
#include "cusparseLt.h"
// CHECK-NOT: #include "hipsparseLt.h"

int main() {
  printf("27.1.after_0300 cuSPARSELt API to hipSPARSELt API synthetic test\n");

  size_t compressedSize = 0;
  size_t compressBufferSize = 0;
  int isSparseA = 1;
  const void *d_dense = nullptr;
  void *d_compressed = nullptr;
  void* d_compressBuffer = nullptr;

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  // CHECK: hipsparseStatus_t status;
  cusparseStatus_t status;

  // CHECK: hipsparseLtHandle_t handle;
  cusparseLtHandle_t handle;

  // CHECK: hipsparseLtMatmulPlan_t plan;
  cusparseLtMatmulPlan_t plan;

  // CHECK: hipsparseLtMatmulDescriptor_t matmulDescr;
  cusparseLtMatmulDescriptor_t matmulDescr;

  // CHECK: hipsparseLtMatmulAlgSelection_t algSelection;
  cusparseLtMatmulAlgSelection_t algSelection;

  // CHECK: hipsparseLtMatDescriptor_t matDescr, matA, matB, matC, matD;
  cusparseLtMatDescriptor_t matDescr, matA, matB, matC, matD;

  // CHECK: hipsparseOperation_t opA, opB, op;
  cusparseOperation_t opA, opB, op;

#if CUSPARSELT_VERSION >= 400
  // CUDA: cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection);
  // HIP: hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t* handle, hipsparseLtMatmulPlan_t* plan, const hipsparseLtMatmulDescriptor_t* matmulDescr, const hipsparseLtMatmulAlgSelection_t* algSelection);
  // CHECK: status = hipsparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSelection);
  status = cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSelection);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressBuffer, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressBuffer, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMACompress(&handle, &plan, d_dense, d_compressed, d_compressBuffer, stream);
  status = cusparseLtSpMMACompress(&handle, &plan, d_dense, d_compressed, d_compressBuffer, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressBufferSize);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressBufferSize);
  // CHECK: status = hipsparseLtSpMMACompressedSize(&handle, &plan, &compressedSize, &compressBufferSize);
  status = cusparseLtSpMMACompressedSize(&handle, &plan, &compressedSize, &compressBufferSize);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressBuffer, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressBuffer, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMACompress2(&handle, &matA, isSparseA, op, d_dense, d_compressed, d_compressBuffer, stream);
  status = cusparseLtSpMMACompress2(&handle, &matA, isSparseA, op, d_dense, d_compressed, d_compressBuffer, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressBufferSize);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressBufferSize);
  // CHECK: status = hipsparseLtSpMMACompressedSize2(&handle, &matA, &compressedSize, &compressBufferSize);
  status = cusparseLtSpMMACompressedSize2(&handle, &matA, &compressedSize, &compressBufferSize);
#endif

  return 0;
}
