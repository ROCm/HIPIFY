// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipblaslt.h"
#include "cublasLt.h"
// CHECK-NOT: #include "hipblaslt.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("21. cuBLASLt API to hipBLASLt API synthetic test\n");

  // CHECK: hipblasLtHandle_t blasLtHandle;
  cublasLtHandle_t blasLtHandle;

  // CHECK: hipblasStatus_t status;
  cublasStatus_t status;

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  void *A = nullptr;
  void *B = nullptr;
  void *C = nullptr;
  void *D = nullptr;
  void *alpha = nullptr;
  void *beta = nullptr;
  void *workspace = nullptr;
  void *buf = nullptr;
  const char *const_ch = nullptr;

  size_t workspaceSizeInBytes = 0;
  size_t sizeWritten = 0;
  uint64_t rows = 0;
  uint64_t cols = 0;
  int64_t ld = 0;
  int requestedAlgoCount = 0;
  int returnAlgoCount = 0;

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType, dataTypeA, dataTypeB, computeType;
  cudaDataType dataType, dataTypeA, dataTypeB, computeType;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipblasLtMatmulAlgo_t blasLtMatmulAlgo;
  cublasLtMatmulAlgo_t blasLtMatmulAlgo;

  // CHECK: hipblasLtMatmulDesc_t blasLtMatmulDesc;
  cublasLtMatmulDesc_t blasLtMatmulDesc;

  // CHECK: hipblasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;
  cublasLtMatrixTransformDesc_t blasLtMatrixTransformDesc;

  // CHECK: hipblasLtMatmulPreference_t blasLtMatmulPreference;
  cublasLtMatmulPreference_t blasLtMatmulPreference;

  // CHECK: hipblasLtMatrixLayout_t blasLtMatrixLayout, Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatrixLayout_t blasLtMatrixLayout, Adesc, Bdesc, Cdesc, Ddesc;

  // CHECK: hipblasLtMatmulHeuristicResult_t blasLtMatmulHeuristicResult;
  // CHECK-NEXT: hipblasLtMatmulHeuristicResult_t *heuristicResultsArray = nullptr;
  cublasLtMatmulHeuristicResult_t blasLtMatmulHeuristicResult;
  cublasLtMatmulHeuristicResult_t *heuristicResultsArray = nullptr;

  // CHECK: hipblasLtOrder_t blasLtOrder;
  // CHECK-NEXT: hipblasLtOrder_t BLASLT_ORDER_COL = HIPBLASLT_ORDER_COL;
  // CHECK-NEXT: hipblasLtOrder_t BLASLT_ORDER_ROW = HIPBLASLT_ORDER_ROW;
  cublasLtOrder_t blasLtOrder;
  cublasLtOrder_t BLASLT_ORDER_COL = CUBLASLT_ORDER_COL;
  cublasLtOrder_t BLASLT_ORDER_ROW = CUBLASLT_ORDER_ROW;

  // CHECK: hipblasLtMatrixLayoutAttribute_t blasLtMatrixLayoutAttribute;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_TYPE = HIPBLASLT_MATRIX_LAYOUT_TYPE;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_ORDER = HIPBLASLT_MATRIX_LAYOUT_ORDER;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_ROWS = HIPBLASLT_MATRIX_LAYOUT_ROWS;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_COLS = HIPBLASLT_MATRIX_LAYOUT_COLS;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_LD = HIPBLASLT_MATRIX_LAYOUT_LD;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_BATCH_COUNT = HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
  // CHECK-NEXT: hipblasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;
  cublasLtMatrixLayoutAttribute_t blasLtMatrixLayoutAttribute;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_TYPE = CUBLASLT_MATRIX_LAYOUT_TYPE;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_ORDER = CUBLASLT_MATRIX_LAYOUT_ORDER;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_ROWS = CUBLASLT_MATRIX_LAYOUT_ROWS;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_COLS = CUBLASLT_MATRIX_LAYOUT_COLS;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_LD = CUBLASLT_MATRIX_LAYOUT_LD;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_BATCH_COUNT = CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT;
  cublasLtMatrixLayoutAttribute_t BLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET;

  // CHECK: hipblasLtMatmulDescAttributes_t blasLtMatmulDescAttributes;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_POINTER_MODE = HIPBLASLT_MATMUL_DESC_POINTER_MODE;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_TRANSA = HIPBLASLT_MATMUL_DESC_TRANSA;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_TRANSB = HIPBLASLT_MATMUL_DESC_TRANSB;
  cublasLtMatmulDescAttributes_t blasLtMatmulDescAttributes;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_POINTER_MODE = CUBLASLT_MATMUL_DESC_POINTER_MODE;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_TRANSA = CUBLASLT_MATMUL_DESC_TRANSA;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_TRANSB = CUBLASLT_MATMUL_DESC_TRANSB;

  // CHECK: hipblasLtMatrixTransformDescAttributes_t blasLtMatrixTransformDescAttributes;
  // CHECK-NEXT: hipblasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE;
  // CHECK-NEXT: hipblasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE;
  // CHECK-NEXT: hipblasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_TRANSA = HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA;
  // CHECK-NEXT: hipblasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_TRANSB = HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB;
  cublasLtMatrixTransformDescAttributes_t blasLtMatrixTransformDescAttributes;
  cublasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE;
  cublasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE;
  cublasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_TRANSA = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA;
  cublasLtMatrixTransformDescAttributes_t BLASLT_MATRIX_TRANSFORM_DESC_TRANSB = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB;

  // CHECK: hipblasLtMatmulPreferenceAttributes_t blasLtMatmulPreferenceAttributes;
  // CHECK-NEXT: hipblasLtMatmulPreferenceAttributes_t BLASLT_MATMUL_PREF_SEARCH_MODE = HIPBLASLT_MATMUL_PREF_SEARCH_MODE;
  // CHECK-NEXT: hipblasLtMatmulPreferenceAttributes_t BLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;
  cublasLtMatmulPreferenceAttributes_t blasLtMatmulPreferenceAttributes;
  cublasLtMatmulPreferenceAttributes_t BLASLT_MATMUL_PREF_SEARCH_MODE = CUBLASLT_MATMUL_PREF_SEARCH_MODE;
  cublasLtMatmulPreferenceAttributes_t BLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES;

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t* lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
  // CHECK: status = hipblasLtCreate(&blasLtHandle);
  status = cublasLtCreate(&blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t lightHandle);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
  // CHECK: status = hipblasLtDestroy(blasLtHandle);
  status = cublasLtDestroy(blasLtHandle);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* B, cublasLtMatrixLayout_t Bdesc, const void* beta, const void* C, cublasLtMatrixLayout_t Cdesc, void* D, cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, cudaStream_t stream);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc, const void* B, hipblasLtMatrixLayout_t Bdesc, const void* beta, const void* C, hipblasLtMatrixLayout_t Cdesc, void* D, hipblasLtMatrixLayout_t Ddesc, const hipblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, hipStream_t stream);
  // CHECK: status = hipblasLtMatmul(blasLtHandle, blasLtMatmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, &blasLtMatmulAlgo, workspace, workspaceSizeInBytes, stream);
  status = cublasLtMatmul(blasLtHandle, blasLtMatmulDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc, &blasLtMatmulAlgo, workspace, workspaceSizeInBytes, stream);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(cublasLtHandle_t lightHandle, cublasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, cublasLtMatrixLayout_t Adesc, const void* beta, const void* B, cublasLtMatrixLayout_t Bdesc, void* C, cublasLtMatrixLayout_t Cdesc, cudaStream_t stream);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t lightHandle, hipblasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc, const void* beta, const void* B, hipblasLtMatrixLayout_t Bdesc, void* C, hipblasLtMatrixLayout_t Cdesc, hipStream_t stream);
  // CHECK: status = hipblasLtMatrixTransform(blasLtHandle, blasLtMatrixTransformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);
  status = cublasLtMatrixTransform(blasLtHandle, blasLtMatrixTransformDesc, alpha, A, Adesc, beta, B, Bdesc, C, Cdesc, stream);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* matLayout, cudaDataType type, uint64_t rows, uint64_t cols, int64_t ld);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* matLayout, hipDataType type, uint64_t rows, uint64_t cols, int64_t ld);
  // CHECK: status = hipblasLtMatrixLayoutCreate(&blasLtMatrixLayout, dataType, rows, cols, ld);
  status = cublasLtMatrixLayoutCreate(&blasLtMatrixLayout, dataType, rows, cols, ld);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t matLayout);
  // CHECK: status = hipblasLtMatrixLayoutDestroy(blasLtMatrixLayout);
  status = cublasLtMatrixLayoutDestroy(blasLtMatrixLayout);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hipblasLtMatrixLayoutSetAttribute(blasLtMatrixLayout, blasLtMatrixLayoutAttribute, buf, workspaceSizeInBytes);
  status = cublasLtMatrixLayoutSetAttribute(blasLtMatrixLayout, blasLtMatrixLayoutAttribute, buf, workspaceSizeInBytes);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout, cublasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t matLayout, hipblasLtMatrixLayoutAttribute_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // CHECK: status = hipblasLtMatrixLayoutGetAttribute(blasLtMatrixLayout, blasLtMatrixLayoutAttribute, buf, workspaceSizeInBytes, &sizeWritten);
  status = cublasLtMatrixLayoutGetAttribute(blasLtMatrixLayout, blasLtMatrixLayoutAttribute, buf, workspaceSizeInBytes, &sizeWritten);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t matmulDesc);
  // CHECK: status = hipblasLtMatmulDescDestroy(blasLtMatmulDesc);
  status = cublasLtMatmulDescDestroy(blasLtMatmulDesc);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hipblasLtMatmulDescSetAttribute(blasLtMatmulDesc, blasLtMatmulDescAttributes, buf, workspaceSizeInBytes);
  status = cublasLtMatmulDescSetAttribute(blasLtMatmulDesc, blasLtMatmulDescAttributes, buf, workspaceSizeInBytes);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc, cublasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatmulDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // CHECK: status = hipblasLtMatmulDescGetAttribute(blasLtMatmulDesc, blasLtMatmulDescAttributes, buf, workspaceSizeInBytes, &sizeWritten);
  status = cublasLtMatmulDescGetAttribute(blasLtMatmulDesc, blasLtMatmulDescAttributes, buf, workspaceSizeInBytes, &sizeWritten);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t* transformDesc, cudaDataType scaleType);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransformDescCreate(hipblasLtMatrixTransformDesc_t* transformDesc, hipDataType scaleType);
  // CHECK: status = hipblasLtMatrixTransformDescCreate(&blasLtMatrixTransformDesc, dataType);
  status = cublasLtMatrixTransformDescCreate(&blasLtMatrixTransformDesc, dataType);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescDestroy(cublasLtMatrixTransformDesc_t transformDesc);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc);
  // CHECK: status = hipblasLtMatrixTransformDescDestroy(blasLtMatrixTransformDesc);
  status = cublasLtMatrixTransformDescDestroy(blasLtMatrixTransformDesc);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescSetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, const void* buf, size_t sizeInBytes);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransformDescSetAttribute( hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hipblasLtMatrixTransformDescSetAttribute(blasLtMatrixTransformDesc, blasLtMatrixTransformDescAttributes, buf, workspaceSizeInBytes);
  status = cublasLtMatrixTransformDescSetAttribute(blasLtMatrixTransformDesc, blasLtMatrixTransformDescAttributes, buf, workspaceSizeInBytes);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatrixTransformDescGetAttribute(cublasLtMatrixTransformDesc_t transformDesc, cublasLtMatrixTransformDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t transformDesc, hipblasLtMatrixTransformDescAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // CHECK: status = hipblasLtMatrixTransformDescGetAttribute(blasLtMatrixTransformDesc, blasLtMatrixTransformDescAttributes, buf, workspaceSizeInBytes, &sizeWritten);
  status = cublasLtMatrixTransformDescGetAttribute(blasLtMatrixTransformDesc, blasLtMatrixTransformDescAttributes, buf, workspaceSizeInBytes, &sizeWritten);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* pref);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref);
  // CHECK: status = hipblasLtMatmulPreferenceCreate(&blasLtMatmulPreference);
  status = cublasLtMatmulPreferenceCreate(&blasLtMatmulPreference);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref);
  // CHECK: status = hipblasLtMatmulPreferenceDestroy(blasLtMatmulPreference);
  status = cublasLtMatmulPreferenceDestroy(blasLtMatmulPreference);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hipblasLtMatmulPreferenceSetAttribute(blasLtMatmulPreference, blasLtMatmulPreferenceAttributes, buf, workspaceSizeInBytes);
  status = cublasLtMatmulPreferenceSetAttribute(blasLtMatmulPreference, blasLtMatmulPreferenceAttributes, buf, workspaceSizeInBytes);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t pref, hipblasLtMatmulPreferenceAttributes_t attr, void* buf, size_t sizeInBytes, size_t* sizeWritten);
  // CHECK: status = hipblasLtMatmulPreferenceGetAttribute(blasLtMatmulPreference, blasLtMatmulPreferenceAttributes, buf, workspaceSizeInBytes, &sizeWritten);
  status = cublasLtMatmulPreferenceGetAttribute(blasLtMatmulPreference, blasLtMatmulPreferenceAttributes, buf, workspaceSizeInBytes, &sizeWritten);

  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t operationDesc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc, cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t preference, int requestedAlgoCount, cublasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc, hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc, hipblasLtMatmulPreference_t pref, int requestedAlgoCount, hipblasLtMatmulHeuristicResult_t heuristicResultsArray[], int* returnAlgoCount);
  // CHECK: status = hipblasLtMatmulAlgoGetHeuristic(blasLtHandle, blasLtMatmulDesc, Adesc, Bdesc, Cdesc, Ddesc, blasLtMatmulPreference, requestedAlgoCount, heuristicResultsArray, &returnAlgoCount);
  status = cublasLtMatmulAlgoGetHeuristic(blasLtHandle, blasLtMatmulDesc, Adesc, Bdesc, Cdesc, Ddesc, blasLtMatmulPreference, requestedAlgoCount, heuristicResultsArray, &returnAlgoCount);
#endif

#if CUBLAS_VERSION >= 10200
  // CHECK: hipblasLtPointerMode_t blasLtPointerMode;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_HOST = HIPBLASLT_POINTER_MODE_HOST;
  // CHECK-NEXT: hipblasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = HIPBLASLT_POINTER_MODE_DEVICE;
  cublasLtPointerMode_t blasLtPointerMode;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_HOST = CUBLASLT_POINTER_MODE_HOST;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_DEVICE = CUBLASLT_POINTER_MODE_DEVICE;

  // CHECK: hipblasLtEpilogue_t blasLtEpilogue;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_DEFAULT = HIPBLASLT_EPILOGUE_DEFAULT;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_RELU = HIPBLASLT_EPILOGUE_RELU;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_BIAS = HIPBLASLT_EPILOGUE_BIAS;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_RELU_BIAS = HIPBLASLT_EPILOGUE_RELU_BIAS;
  cublasLtEpilogue_t blasLtEpilogue;
  cublasLtEpilogue_t BLASLT_EPILOGUE_DEFAULT = CUBLASLT_EPILOGUE_DEFAULT;
  cublasLtEpilogue_t BLASLT_EPILOGUE_RELU = CUBLASLT_EPILOGUE_RELU;
  cublasLtEpilogue_t BLASLT_EPILOGUE_BIAS = CUBLASLT_EPILOGUE_BIAS;
  cublasLtEpilogue_t BLASLT_EPILOGUE_RELU_BIAS = CUBLASLT_EPILOGUE_RELU_BIAS;

  // CHECK: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE = HIPBLASLT_MATMUL_DESC_EPILOGUE;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_BIAS_POINTER = HIPBLASLT_MATMUL_DESC_BIAS_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE = CUBLASLT_MATMUL_DESC_EPILOGUE;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_BIAS_POINTER = CUBLASLT_MATMUL_DESC_BIAS_POINTER;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipblasComputeType_t blasComputeType;
  cublasComputeType_t blasComputeType;

  // [hipBLASLt] TODO: Use hipblasComputeType_t instead of incompatible hipblasLtComputeType_t
  // [HIPIFY] TODO: For CUDA < 11.0 throw an error cublasLtMatmulDescCreate is not supported by HIP, please use the newer version of cublasLtMatmulDescCreate (>=11.0)
  // [Reason] The signature change in 11.0.1 from cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, cudaDataType computeType);
  // CUDA: cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* matmulDesc, cublasComputeType_t computeType, cudaDataType_t scaleType);
  // HIP: HIPBLASLT_EXPORT hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* matmulDesc, hipblasLtComputeType_t computeType, hipblasDatatype_t scaleType);
  // CHECK: status = hipblasLtMatmulDescCreate(&blasLtMatmulDesc, blasComputeType, dataType);
  status = cublasLtMatmulDescCreate(&blasLtMatmulDesc, blasComputeType, dataType);
#endif

#if CUDA_VERSION >= 11000 && CUBLAS_VERSION >= 11000
  // CHECK: hipblasLtMatrixLayoutOpaque_t blasLtMatrixLayoutOpaque;
  cublasLtMatrixLayoutOpaque_t blasLtMatrixLayoutOpaque;

  // CHECK: hipblasLtMatmulDescOpaque_t blasLtMatmulDescOpaque;
  cublasLtMatmulDescOpaque_t blasLtMatmulDescOpaque;

  // CHECK: hipblasLtMatrixTransformDescOpaque_t blasLtMatrixTransformDescOpaque;
  cublasLtMatrixTransformDescOpaque_t blasLtMatrixTransformDescOpaque;

  // CHECK: hipblasLtMatmulPreferenceOpaque_t blasLtMatmulPreferenceOpaque;
  cublasLtMatmulPreferenceOpaque_t blasLtMatmulPreferenceOpaque;
#endif

#if CUDA_VERSION >= 11030 && CUBLAS_VERSION >= 11501
  // CHECK: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE;

  // CHECK: hipblasLtEpilogue_t BLASLT_EPILOGUE_GELU = HIPBLASLT_EPILOGUE_GELU;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_GELU_AUX = HIPBLASLT_EPILOGUE_GELU_AUX;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_GELU_BIAS = HIPBLASLT_EPILOGUE_GELU_BIAS;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_GELU_AUX_BIAS = HIPBLASLT_EPILOGUE_GELU_AUX_BIAS;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_DGELU_BGRAD = HIPBLASLT_EPILOGUE_DGELU_BGRAD;
  cublasLtEpilogue_t BLASLT_EPILOGUE_GELU = CUBLASLT_EPILOGUE_GELU;
  cublasLtEpilogue_t BLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_GELU_AUX;
  cublasLtEpilogue_t BLASLT_EPILOGUE_GELU_BIAS = CUBLASLT_EPILOGUE_GELU_BIAS;
  cublasLtEpilogue_t BLASLT_EPILOGUE_GELU_AUX_BIAS = CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
  cublasLtEpilogue_t BLASLT_EPILOGUE_DGELU_BGRAD = CUBLASLT_EPILOGUE_DGELU_BGRAD;
#endif

#if CUDA_VERSION >= 11040 && CUBLAS_VERSION >= 11601
  // CHECK: hipblasLtPointerMode_t BLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
  cublasLtPointerMode_t BLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;

  // CHECK: hipblasLtEpilogue_t BLASLT_EPILOGUE_BGRADA = HIPBLASLT_EPILOGUE_BGRADA;
  // CHECK-NEXT: hipblasLtEpilogue_t BLASLT_EPILOGUE_BGRADB = HIPBLASLT_EPILOGUE_BGRADB;
  cublasLtEpilogue_t BLASLT_EPILOGUE_BGRADA = CUBLASLT_EPILOGUE_BGRADA;
  cublasLtEpilogue_t BLASLT_EPILOGUE_BGRADB = CUBLASLT_EPILOGUE_BGRADB;
#endif

#if CUDA_VERSION >= 11060
  // CHECK: hipblasLtEpilogue_t BLASLT_EPILOGUE_DGELU = HIPBLASLT_EPILOGUE_DGELU;
  cublasLtEpilogue_t BLASLT_EPILOGUE_DGELU = CUBLASLT_EPILOGUE_DGELU;
#endif

#if CUDA_VERSION >= 11080
  // CHECK: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_A_SCALE_POINTER = HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_B_SCALE_POINTER = HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_C_SCALE_POINTER = HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_D_SCALE_POINTER = HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_BIAS_DATA_TYPE = HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
  // CHECK-NEXT: hipblasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_AMAX_D_POINTER = HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_A_SCALE_POINTER = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_B_SCALE_POINTER = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_C_SCALE_POINTER = CUBLASLT_MATMUL_DESC_C_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_D_SCALE_POINTER = CUBLASLT_MATMUL_DESC_D_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_BIAS_DATA_TYPE = CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE;
  cublasLtMatmulDescAttributes_t BLASLT_MATMUL_DESC_AMAX_D_POINTER = CUBLASLT_MATMUL_DESC_AMAX_D_POINTER;
#endif
  return 0;
}
