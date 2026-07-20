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
  printf("27. cuSPARSELt API to hipSPARSELt API synthetic test\n");

  // CHECK: hipsparseStatus_t status;
  cusparseStatus_t status;

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  // CHECK: hipsparseOperation_t opA, opB, op;
  cusparseOperation_t opA, opB, op;

  // CHECK: hipsparseOrder_t order;
  cusparseOrder_t order;

  // CHECK: hipDataType valueType;
  cudaDataType valueType;

  int64_t rows = 0;
  int64_t cols = 0;
  int64_t ld = 0;
  uint32_t alignment = 16;
  int isSparseA = 1;
  int valid = 0;
  int32_t numStreams = 0;
  size_t workspaceSize = 0;
  size_t compressedSize = 0;
  size_t compressBufferSize = 0;
  size_t dataSize = 0;
  const void *alpha = nullptr;
  const void *beta = nullptr;
  const void *d_A = nullptr;
  const void *d_B = nullptr;
  const void *d_C = nullptr;
  const void *d_in = nullptr;
  const void *d_dense = nullptr;
  void *d_D = nullptr;
  void *d_out = nullptr;
  void *d_compressed = nullptr;
  void *d_compressBuffer = nullptr;
  void *workspace = nullptr;
  void *data = nullptr;

  // CHECK: hipsparseLtHandle_t handle;
  cusparseLtHandle_t handle;

  // CHECK: hipsparseLtMatDescriptor_t matDescr, matA, matB, matC, matD;
  cusparseLtMatDescriptor_t matDescr, matA, matB, matC, matD;

  // CHECK: hipsparseLtMatmulDescriptor_t matmulDescr;
  cusparseLtMatmulDescriptor_t matmulDescr;

  // CHECK: hipsparseLtMatmulAlgSelection_t algSelection;
  cusparseLtMatmulAlgSelection_t algSelection;

  // CHECK: hipsparseLtMatmulPlan_t plan;
  cusparseLtMatmulPlan_t plan;

  // cuSPARSELt data types (Added: cuSPARSELt 0.0.1)
  // CHECK: hipsparseLtSparsity_t sparsity;
  // CHECK-NEXT: hipsparseLtSparsity_t SPARSITY_50_PERCENT = HIPSPARSELT_SPARSITY_50_PERCENT;
  cusparseLtSparsity_t sparsity;
  cusparseLtSparsity_t SPARSITY_50_PERCENT = CUSPARSELT_SPARSITY_50_PERCENT;

  // CHECK: hipsparseLtComputetype_t computeType;
  // CHECK-NEXT: hipsparseLtComputetype_t COMPUTE_16F = HIPSPARSELT_COMPUTE_16F;
  // CHECK-NEXT: hipsparseLtComputetype_t COMPUTE_32I = HIPSPARSELT_COMPUTE_32I;
  cusparseComputeType computeType;
  cusparseComputeType COMPUTE_16F = CUSPARSE_COMPUTE_16F;
  cusparseComputeType COMPUTE_32I = CUSPARSE_COMPUTE_32I;

  // CHECK: hipsparseLtMatmulAlg_t alg;
  // CHECK-NEXT: hipsparseLtMatmulAlg_t MATMUL_ALG_DEFAULT = HIPSPARSELT_MATMUL_ALG_DEFAULT;
  cusparseLtMatmulAlg_t alg;
  cusparseLtMatmulAlg_t MATMUL_ALG_DEFAULT = CUSPARSELT_MATMUL_ALG_DEFAULT;

  // CHECK: hipsparseLtMatmulAlgAttribute_t algAttribute;
  // CHECK-NEXT: hipsparseLtMatmulAlgAttribute_t MATMUL_ALG_CONFIG_ID = HIPSPARSELT_MATMUL_ALG_CONFIG_ID;
  // CHECK-NEXT: hipsparseLtMatmulAlgAttribute_t MATMUL_ALG_CONFIG_MAX_ID = HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
  // CHECK-NEXT: hipsparseLtMatmulAlgAttribute_t MATMUL_SEARCH_ITERATIONS = HIPSPARSELT_MATMUL_SEARCH_ITERATIONS;
  cusparseLtMatmulAlgAttribute_t algAttribute;
  cusparseLtMatmulAlgAttribute_t MATMUL_ALG_CONFIG_ID = CUSPARSELT_MATMUL_ALG_CONFIG_ID;
  cusparseLtMatmulAlgAttribute_t MATMUL_ALG_CONFIG_MAX_ID = CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID;
  cusparseLtMatmulAlgAttribute_t MATMUL_SEARCH_ITERATIONS = CUSPARSELT_MATMUL_SEARCH_ITERATIONS;

  // CHECK: hipsparseLtPruneAlg_t pruneAlg;
  // CHECK-NEXT: hipsparseLtPruneAlg_t PRUNE_SPMMA_TILE = HIPSPARSELT_PRUNE_SPMMA_TILE;
  // CHECK-NEXT: hipsparseLtPruneAlg_t PRUNE_SPMMA_STRIP = HIPSPARSELT_PRUNE_SPMMA_STRIP;
  cusparseLtPruneAlg_t pruneAlg;
  cusparseLtPruneAlg_t PRUNE_SPMMA_TILE = CUSPARSELT_PRUNE_SPMMA_TILE;
  cusparseLtPruneAlg_t PRUNE_SPMMA_STRIP = CUSPARSELT_PRUNE_SPMMA_STRIP;

  // NOTE: CUSPARSE_COMPUTE_TF32 / CUSPARSE_COMPUTE_TF32_FAST (cuSPARSELt 0.1.0) were removed
  // from cusparseComputeType in later cuSPARSELt (replaced by CUSPARSE_COMPUTE_32F).
  // [ToDo]: Add the CUSPARSE_COMPUTE_32F -> HIPSPARSELT_COMPUTE_32F mapping.

  // cuSPARSELt function reference (Added: cuSPARSELt 0.0.1)
  // CUDA: cusparseStatus_t cusparseLtInit(cusparseLtHandle_t* handle);
  // HIP: hipsparseStatus_t hipsparseLtInit(hipsparseLtHandle_t* handle);
  // CHECK: status = hipsparseLtInit(&handle);
  status = cusparseLtInit(&handle);

  // CUDA: cusparseStatus_t cusparseLtDestroy(const cusparseLtHandle_t* handle);
  // HIP: hipsparseStatus_t hipsparseLtDestroy(const hipsparseLtHandle_t* handle);
  // CHECK: status = hipsparseLtDestroy(&handle);
  status = cusparseLtDestroy(&handle);

  // CUDA: cusparseStatus_t cusparseLtDenseDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order);
  // HIP: hipsparseStatus_t hipsparseLtDenseDescriptorInit(const hipsparseLtHandle_t* handle, hipsparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, hipDataType valueType, hipsparseOrder_t order);
  // CHECK: status = hipsparseLtDenseDescriptorInit(&handle, &matDescr, rows, cols, ld, alignment, valueType, order);
  status = cusparseLtDenseDescriptorInit(&handle, &matDescr, rows, cols, ld, alignment, valueType, order);

  // CUDA: cusparseStatus_t cusparseLtStructuredDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, cudaDataType valueType, cusparseOrder_t order, cusparseLtSparsity_t sparsity);
  // HIP: hipsparseStatus_t hipsparseLtStructuredDescriptorInit(const hipsparseLtHandle_t* handle, hipsparseLtMatDescriptor_t* matDescr, int64_t rows, int64_t cols, int64_t ld, uint32_t alignment, hipDataType valueType, hipsparseOrder_t order, hipsparseLtSparsity_t sparsity);
  // CHECK: status = hipsparseLtStructuredDescriptorInit(&handle, &matA, rows, cols, ld, alignment, valueType, order, sparsity);
  status = cusparseLtStructuredDescriptorInit(&handle, &matA, rows, cols, ld, alignment, valueType, order, sparsity);

  // CUDA: cusparseStatus_t cusparseLtMatmulDescriptorInit(const cusparseLtHandle_t* handle, cusparseLtMatmulDescriptor_t* matmulDescr, cusparseOperation_t opA, cusparseOperation_t opB, const cusparseLtMatDescriptor_t* matA, const cusparseLtMatDescriptor_t* matB, const cusparseLtMatDescriptor_t* matC, const cusparseLtMatDescriptor_t* matD, cusparseComputeType computeType);
  // HIP: hipsparseStatus_t hipsparseLtMatmulDescriptorInit(const hipsparseLtHandle_t* handle, hipsparseLtMatmulDescriptor_t* matmulDescr, hipsparseOperation_t opA, hipsparseOperation_t opB, const hipsparseLtMatDescriptor_t* matA, const hipsparseLtMatDescriptor_t* matB, const hipsparseLtMatDescriptor_t* matC, const hipsparseLtMatDescriptor_t* matD, hipsparseLtComputetype_t computeType);
  // CHECK: status = hipsparseLtMatmulDescriptorInit(&handle, &matmulDescr, opA, opB, &matA, &matB, &matC, &matD, computeType);
  status = cusparseLtMatmulDescriptorInit(&handle, &matmulDescr, opA, opB, &matA, &matB, &matC, &matD, computeType);

  // CUDA: cusparseStatus_t cusparseLtMatmulAlgSelectionInit(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, const cusparseLtMatmulDescriptor_t* matmulDescr, cusparseLtMatmulAlg_t alg);
  // HIP: hipsparseStatus_t hipsparseLtMatmulAlgSelectionInit(const hipsparseLtHandle_t* handle, hipsparseLtMatmulAlgSelection_t* algSelection, const hipsparseLtMatmulDescriptor_t* matmulDescr, hipsparseLtMatmulAlg_t alg);
  // CHECK: status = hipsparseLtMatmulAlgSelectionInit(&handle, &algSelection, &matmulDescr, alg);
  status = cusparseLtMatmulAlgSelectionInit(&handle, &algSelection, &matmulDescr, alg);

  // CUDA: cusparseStatus_t cusparseLtMatmulAlgSetAttribute(const cusparseLtHandle_t* handle, cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize);
  // HIP: hipsparseStatus_t hipsparseLtMatmulAlgSetAttribute(const hipsparseLtHandle_t* handle, hipsparseLtMatmulAlgSelection_t* algSelection, hipsparseLtMatmulAlgAttribute_t attribute, const void* data, size_t dataSize);
  // CHECK: status = hipsparseLtMatmulAlgSetAttribute(&handle, &algSelection, algAttribute, data, dataSize);
  status = cusparseLtMatmulAlgSetAttribute(&handle, &algSelection, algAttribute, data, dataSize);

  // CUDA: cusparseStatus_t cusparseLtMatmulAlgGetAttribute(const cusparseLtHandle_t* handle, const cusparseLtMatmulAlgSelection_t* algSelection, cusparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize);
  // HIP: hipsparseStatus_t hipsparseLtMatmulAlgGetAttribute(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulAlgSelection_t* algSelection, hipsparseLtMatmulAlgAttribute_t attribute, void* data, size_t dataSize);
  // CHECK: status = hipsparseLtMatmulAlgGetAttribute(&handle, &algSelection, algAttribute, data, dataSize);
  status = cusparseLtMatmulAlgGetAttribute(&handle, &algSelection, algAttribute, data, dataSize);

  // CUDA: cusparseStatus_t cusparseLtMatmulGetWorkspace(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* workspaceSize);
  // HIP: hipsparseStatus_t hipsparseLtMatmulGetWorkspace(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, size_t* workspaceSize);
  // CHECK: status = hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);
  status = cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize);

  // CUDA: cusparseStatus_t cusparseLtMatmulPlanInit(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const cusparseLtMatmulDescriptor_t* matmulDescr, const cusparseLtMatmulAlgSelection_t* algSelection);
  // HIP: hipsparseStatus_t hipsparseLtMatmulPlanInit(const hipsparseLtHandle_t* handle, hipsparseLtMatmulPlan_t* plan, const hipsparseLtMatmulDescriptor_t* matmulDescr, const hipsparseLtMatmulAlgSelection_t* algSelection);
  // CHECK: status = hipsparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSelection);
  status = cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescr, &algSelection);

  // CUDA: cusparseStatus_t cusparseLtMatmulPlanDestroy(const cusparseLtMatmulPlan_t* plan);
  // HIP: hipsparseStatus_t hipsparseLtMatmulPlanDestroy(const hipsparseLtMatmulPlan_t* plan);
  // CHECK: status = hipsparseLtMatmulPlanDestroy(&plan);
  status = cusparseLtMatmulPlanDestroy(&plan);

  // CUDA: cusparseStatus_t cusparseLtMatmul(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams);
  // HIP: hipsparseStatus_t hipsparseLtMatmul(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, hipStream_t* streams, int32_t numStreams);
  // CHECK: status = hipsparseLtMatmul(&handle, &plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, &stream, numStreams);
  status = cusparseLtMatmul(&handle, &plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, &stream, numStreams);

  // CUDA: cusparseStatus_t cusparseLtMatmulSearch(const cusparseLtHandle_t* handle, cusparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, cudaStream_t* streams, int32_t numStreams);
  // HIP: hipsparseStatus_t hipsparseLtMatmulSearch(const hipsparseLtHandle_t* handle, hipsparseLtMatmulPlan_t* plan, const void* alpha, const void* d_A, const void* d_B, const void* beta, const void* d_C, void* d_D, void* workspace, hipStream_t* streams, int32_t numStreams);
  // CHECK: status = hipsparseLtMatmulSearch(&handle, &plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, &stream, numStreams);
  status = cusparseLtMatmulSearch(&handle, &plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, &stream, numStreams);

  // CUDA: cusparseStatus_t cusparseLtSpMMAPrune(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMAPrune(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, void* d_out, hipsparseLtPruneAlg_t pruneAlg, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMAPrune(&handle, &matmulDescr, d_in, d_out, pruneAlg, stream);
  status = cusparseLtSpMMAPrune(&handle, &matmulDescr, d_in, d_out, pruneAlg, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMAPruneCheck(const cusparseLtHandle_t* handle, const cusparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMAPruneCheck(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulDescriptor_t* matmulDescr, const void* d_in, int* valid, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMAPruneCheck(&handle, &matmulDescr, d_in, &valid, stream);
  status = cusparseLtSpMMAPruneCheck(&handle, &matmulDescr, d_in, &valid, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompressedSize(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressBufferSize);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompressedSize(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, size_t* compressedSize, size_t* compressBufferSize);
  // CHECK: status = hipsparseLtSpMMACompressedSize(&handle, &plan, &compressedSize, &compressBufferSize);
  status = cusparseLtSpMMACompressedSize(&handle, &plan, &compressedSize, &compressBufferSize);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompress(const cusparseLtHandle_t* handle, const cusparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressBuffer, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompress(const hipsparseLtHandle_t* handle, const hipsparseLtMatmulPlan_t* plan, const void* d_dense, void* d_compressed, void* d_compressBuffer, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMACompress(&handle, &plan, d_dense, d_compressed, d_compressBuffer, stream);
  status = cusparseLtSpMMACompress(&handle, &plan, d_dense, d_compressed, d_compressBuffer, stream);

  // cuSPARSELt function reference (Added: cuSPARSELt 0.1.0)
  // CUDA: cusparseStatus_t cusparseLtMatDescriptorDestroy(const cusparseLtMatDescriptor_t* matDescr);
  // HIP: hipsparseStatus_t hipsparseLtMatDescriptorDestroy(const hipsparseLtMatDescriptor_t* matDescr);
  // CHECK: status = hipsparseLtMatDescriptorDestroy(&matDescr);
  status = cusparseLtMatDescriptorDestroy(&matDescr);

  // CUDA: cusparseStatus_t cusparseLtSpMMAPrune2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, void* d_out, cusparseLtPruneAlg_t pruneAlg, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMAPrune2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void* d_in, void* d_out, hipsparseLtPruneAlg_t pruneAlg, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMAPrune2(&handle, &matA, isSparseA, op, d_in, d_out, pruneAlg, stream);
  status = cusparseLtSpMMAPrune2(&handle, &matA, isSparseA, op, d_in, d_out, pruneAlg, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMAPruneCheck2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_in, int* d_valid, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMAPruneCheck2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void* d_in, int* d_valid, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMAPruneCheck2(&handle, &matA, isSparseA, op, d_in, &valid, stream);
  status = cusparseLtSpMMAPruneCheck2(&handle, &matA, isSparseA, op, d_in, &valid, stream);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompressedSize2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressBufferSize);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompressedSize2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, size_t* compressedSize, size_t* compressBufferSize);
  // CHECK: status = hipsparseLtSpMMACompressedSize2(&handle, &matA, &compressedSize, &compressBufferSize);
  status = cusparseLtSpMMACompressedSize2(&handle, &matA, &compressedSize, &compressBufferSize);

  // CUDA: cusparseStatus_t cusparseLtSpMMACompress2(const cusparseLtHandle_t* handle, const cusparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, cusparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressBuffer, cudaStream_t stream);
  // HIP: hipsparseStatus_t hipsparseLtSpMMACompress2(const hipsparseLtHandle_t* handle, const hipsparseLtMatDescriptor_t* sparseMatDescr, int isSparseA, hipsparseOperation_t op, const void* d_dense, void* d_compressed, void* d_compressBuffer, hipStream_t stream);
  // CHECK: status = hipsparseLtSpMMACompress2(&handle, &matA, isSparseA, op, d_dense, d_compressed, d_compressBuffer, stream);
  status = cusparseLtSpMMACompress2(&handle, &matA, isSparseA, op, d_dense, d_compressed, d_compressBuffer, stream);

  return 0;
}
