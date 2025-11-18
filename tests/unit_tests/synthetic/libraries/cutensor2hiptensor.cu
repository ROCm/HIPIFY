// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hiptensor.h"
#include "cutensor.h"
// CHECK-NOT: #include "hiptensor.h"

int main() {
  printf("25. cuTensor API to hipTensor API synthetic test\n");

  // CHECK: hiptensorHandle *handle_p = nullptr;
  // CHECK-NEXT: hiptensorHandle_t handle;
  // CHECK-NEXT: const hiptensorHandle_t *handle_c = nullptr;
  // CHECK-NEXT: hiptensorHandle_t *handle2 = nullptr;
  cutensorHandle *handle_p = nullptr;
  cutensorHandle_t handle;
  const cutensorHandle_t *handle_c = nullptr;
  cutensorHandle_t *handle2 = nullptr;

  //CHECK: hiptensorStatus_t status;
  cutensorStatus_t status;

  // CHECK: hiptensorTensorDescriptor_t *tensorDescriptor = nullptr;
  // CHECK-NEXT: hiptensorTensorDescriptor_t *descA = nullptr;
  // CHECK-NEXT: hiptensorTensorDescriptor_t *descB = nullptr;
  // CHECK-NEXT: hiptensorTensorDescriptor_t *descC = nullptr;
  // CHECK-NEXT: hiptensorTensorDescriptor_t *descD = nullptr;
  cutensorTensorDescriptor_t *tensorDescriptor = nullptr;
  cutensorTensorDescriptor_t *descA = nullptr;
  cutensorTensorDescriptor_t *descB = nullptr;
  cutensorTensorDescriptor_t *descC = nullptr;
  cutensorTensorDescriptor_t *descD = nullptr;

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType;
  cudaDataType dataType;
#endif

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  const uint32_t numModes = 0;
  uint32_t numCachelinesRead = 0;
  uint32_t alignmentRequirement = 0;
  const int64_t *extent = nullptr;
  const int64_t *stride = nullptr;
  const uint64_t workspaceSize = 0;
  uint64_t workspaceSize2 = 0;
  const void *alpha = nullptr;
  const void *A = nullptr;
  const int32_t *modeA = nullptr;
  void *B = nullptr;
  const void *B_1 = nullptr;
  const void *beta = nullptr;
  const int32_t *modeB = nullptr;
  const void *C = nullptr;
  const int32_t *modeC = nullptr;
  const void* gamma = nullptr;
  void *D = nullptr;
  const int32_t *modeD = nullptr;
  void *workspace = nullptr;
  const char *err = nullptr;
  const char *log = nullptr;
  const char *filename = nullptr;
  size_t ver = 0;
  size_t sizeInBytes = 0;
  FILE *file = nullptr;
  int32_t level = 0;
  int32_t mask = 0;
  void *buf = nullptr;
  uint64_t workspaceSizeEstimate = 0;

#if CUTENSOR_MAJOR >= 1
  // CHECK: hiptensorOperator_t tensorOperator_t, OpA, OpB, OpC, OpAB, OpABC, OpAC, OpReduce;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_IDENTITY = HIPTENSOR_OP_IDENTITY;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_SQRT = HIPTENSOR_OP_SQRT;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_RELU = HIPTENSOR_OP_RELU;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_CONJ = HIPTENSOR_OP_CONJ;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_RCP = HIPTENSOR_OP_RCP;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_SIGMOID = HIPTENSOR_OP_SIGM
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_TANH = HIPTENSOR_OP_TANH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_EXP = HIPTENSOR_OP_EXP;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_LOG = HIPTENSOR_OP_LOG;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ABS = HIPTENSOR_OP_ABS;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_NEG = HIPTENSOR_OP_NEG;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_SIN = HIPTENSOR_OP_SIN;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_COS = HIPTENSOR_OP_COS;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_TAN = HIPTENSOR_OP_TAN;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_SINH = HIPTENSOR_OP_SINH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_COSH = HIPTENSOR_OP_COSH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ASIN = HIPTENSOR_OP_ASIN;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ACOS = HIPTENSOR_OP_ACOS;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ATAN = HIPTENSOR_OP_ATAN;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ASINH = HIPTENSOR_OP_ASINH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ACOSH = HIPTENSOR_OP_ACOSH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ATANH = HIPTENSOR_OP_ATANH;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_CEIL = HIPTENSOR_OP_CEIL;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_FLOOR = HIPTENSOR_OP_FLOOR;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_ADD = HIPTENSOR_OP_ADD;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_MUL = HIPTENSOR_OP_MUL;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_MAX = HIPTENSOR_OP_MAX;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_MIN = HIPTENSOR_OP_MIN;
  // CHECK-NEXT: hiptensorOperator_t TENSOR_OP_UNKNOWN = HIPTENSOR_OP_UNKNOWN;
  cutensorOperator_t tensorOperator_t, OpA, OpB, OpC, OpAB, OpABC, OpAC, OpReduce;
  cutensorOperator_t TENSOR_OP_IDENTITY = CUTENSOR_OP_IDENTITY;
  cutensorOperator_t TENSOR_OP_SQRT = CUTENSOR_OP_SQRT;
  cutensorOperator_t TENSOR_OP_RELU = CUTENSOR_OP_RELU;
  cutensorOperator_t TENSOR_OP_CONJ = CUTENSOR_OP_CONJ;
  cutensorOperator_t TENSOR_OP_RCP = CUTENSOR_OP_RCP;
  cutensorOperator_t TENSOR_OP_SIGMOID = CUTENSOR_OP_SIGMOID;
  cutensorOperator_t TENSOR_OP_TANH = CUTENSOR_OP_TANH;
  cutensorOperator_t TENSOR_OP_EXP = CUTENSOR_OP_EXP;
  cutensorOperator_t TENSOR_OP_LOG = CUTENSOR_OP_LOG;
  cutensorOperator_t TENSOR_OP_ABS = CUTENSOR_OP_ABS;
  cutensorOperator_t TENSOR_OP_NEG = CUTENSOR_OP_NEG;
  cutensorOperator_t TENSOR_OP_SIN = CUTENSOR_OP_SIN;
  cutensorOperator_t TENSOR_OP_COS = CUTENSOR_OP_COS;
  cutensorOperator_t TENSOR_OP_TAN = CUTENSOR_OP_TAN;
  cutensorOperator_t TENSOR_OP_SINH = CUTENSOR_OP_SINH;
  cutensorOperator_t TENSOR_OP_COSH = CUTENSOR_OP_COSH;
  cutensorOperator_t TENSOR_OP_ASIN = CUTENSOR_OP_ASIN;
  cutensorOperator_t TENSOR_OP_ACOS = CUTENSOR_OP_ACOS;
  cutensorOperator_t TENSOR_OP_ATAN = CUTENSOR_OP_ATAN;
  cutensorOperator_t TENSOR_OP_ASINH = CUTENSOR_OP_ASINH;
  cutensorOperator_t TENSOR_OP_ACOSH = CUTENSOR_OP_ACOSH;
  cutensorOperator_t TENSOR_OP_ATANH = CUTENSOR_OP_ATANH;
  cutensorOperator_t TENSOR_OP_CEIL = CUTENSOR_OP_CEIL;
  cutensorOperator_t TENSOR_OP_FLOOR = CUTENSOR_OP_FLOOR;
  cutensorOperator_t TENSOR_OP_ADD = CUTENSOR_OP_ADD;
  cutensorOperator_t TENSOR_OP_MUL = CUTENSOR_OP_MUL;
  cutensorOperator_t TENSOR_OP_MAX = CUTENSOR_OP_MAX;
  cutensorOperator_t TENSOR_OP_MIN = CUTENSOR_OP_MIN;
  cutensorOperator_t TENSOR_OP_UNKNOWN = CUTENSOR_OP_UNKNOWN;

  // CHECK: hiptensorStatus_t tensorStatus_t;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_SUCCESS = HIPTENSOR_STATUS_SUCCESS;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_NOT_INITIALIZED = HIPTENSOR_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_ALLOC_FAILED = HIPTENSOR_STATUS_ALLOC_FAILED;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_INVALID_VALUE = HIPTENSOR_STATUS_INVALID_VALUE;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_ARCH_MISMATCH = HIPTENSOR_STATUS_ARCH_MISMATCH;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_EXECUTION_FAILED = HIPTENSOR_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_INTERNAL_ERROR = HIPTENSOR_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_NOT_SUPPORTED = HIPTENSOR_STATUS_NOT_SUPPORTED;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_INSUFFICIENT_WORKSPACE = HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
  // CHECK-NEXT: hiptensorStatus_t TENSOR_STATUS_INSUFFICIENT_DRIVER = HIPTENSOR_STATUS_INSUFFICIENT_DRIVER;
  cutensorStatus_t tensorStatus_t;
  cutensorStatus_t TENSOR_STATUS_SUCCESS = CUTENSOR_STATUS_SUCCESS;
  cutensorStatus_t TENSOR_STATUS_NOT_INITIALIZED = CUTENSOR_STATUS_NOT_INITIALIZED;
  cutensorStatus_t TENSOR_STATUS_ALLOC_FAILED = CUTENSOR_STATUS_ALLOC_FAILED;
  cutensorStatus_t TENSOR_STATUS_INVALID_VALUE = CUTENSOR_STATUS_INVALID_VALUE;
  cutensorStatus_t TENSOR_STATUS_ARCH_MISMATCH = CUTENSOR_STATUS_ARCH_MISMATCH;
  cutensorStatus_t TENSOR_STATUS_EXECUTION_FAILED = CUTENSOR_STATUS_EXECUTION_FAILED;
  cutensorStatus_t TENSOR_STATUS_INTERNAL_ERROR = CUTENSOR_STATUS_INTERNAL_ERROR;
  cutensorStatus_t TENSOR_STATUS_NOT_SUPPORTED = CUTENSOR_STATUS_NOT_SUPPORTED;
  cutensorStatus_t TENSOR_STATUS_INSUFFICIENT_WORKSPACE = CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
  cutensorStatus_t TENSOR_STATUS_INSUFFICIENT_DRIVER = CUTENSOR_STATUS_INSUFFICIENT_DRIVER;

  // CHECK: hiptensorAlgo_t tensorAlgo_t;
  // CHECK-NEXT: hiptensorAlgo_t TENSOR_ALGO_DEFAULT = HIPTENSOR_ALGO_DEFAULT;
  cutensorAlgo_t tensorAlgo_t;
  cutensorAlgo_t TENSOR_ALGO_DEFAULT = CUTENSOR_ALGO_DEFAULT;

  // CHECK: hiptensorWorksizePreference_t tensorWorksizePreference_t;
  // CHECK-NEXT: TENSOR_WORKSPACE_MIN = HIPTENSOR_WORKSPACE_MIN;
  // CHECK-NEXT: TENSOR_WORKSPACE_MAX = HIPTENSOR_WORKSPACE_MAX;
  cutensorWorksizePreference_t tensorWorksizePreference_t;
  cutensorWorksizePreference_t TENSOR_WORKSPACE_MIN = CUTENSOR_WORKSPACE_MIN;
  cutensorWorksizePreference_t TENSOR_WORKSPACE_MAX = CUTENSOR_WORKSPACE_MAX;

  // CUDA: const char* cutensorGetErrorString(const cutensorStatus_t error);
  // HIP: const char* hiptensorGetErrorString(const hiptensorStatus_t error);
  // CHECK: err = hiptensorGetErrorString(status);
  err = cutensorGetErrorString(status);

  // CUDA: size_t cutensorGetCudartVersion();
  // HIP: int hiptensorGetHiprtVersion();
  // CHECK: ver = hiptensorGetHiprtVersion();
  ver = cutensorGetCudartVersion();
#endif

#if (CUTENSOR_MAJOR == 1 && CUTENSOR_MINOR >= 4) || CUTENSOR_MAJOR >= 2
  // CHECK: hiptensorAlgo_t TENSOR_ALGO_DEFAULT_PATIENT = HIPTENSOR_ALGO_DEFAULT_PATIENT;
  cutensorAlgo_t TENSOR_ALGO_DEFAULT_PATIENT = CUTENSOR_ALGO_DEFAULT_PATIENT;
#endif

#if (CUTENSOR_MAJOR >= 1 && CUTENSOR_MAJOR < 2)
  // CHECK: hiptensorComputeDescriptor_t tensorComputeType_t;
  cutensorComputeType_t tensorComputeType_t;

  // CHECK: hiptensorContractionPlan_t tensorPlan2_t;
  cutensorContractionPlan_t tensorPlan2_t;

#if CUTENSOR_MINOR >= 2
  // CHECK: hiptensorAutotuneMode_t tensorAutotuneMode_t;
  cutensorAutotuneMode_t tensorAutotuneMode_t;

  // CHECK: hiptensorStatus_t TENSOR_STATUS_IO_ERROR = HIPTENSOR_STATUS_IO_ERROR;
  cutensorStatus_t TENSOR_STATUS_IO_ERROR = CUTENSOR_STATUS_IO_ERROR;

  // CHECK: hiptensorComputeDescriptor_t TENSOR_COMPUTE_16F = HIPTENSOR_COMPUTE_DESC_16F;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_16BF = HIPTENSOR_COMPUTE_DESC_16BF;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_32F = HIPTENSOR_COMPUTE_DESC_32F;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_64F = HIPTENSOR_COMPUTE_DESC_64F;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_8U = HIPTENSOR_COMPUTE_DESC_8U;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_8I = HIPTENSOR_COMPUTE_DESC_8I;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_32U = HIPTENSOR_COMPUTE_DESC_32U;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_32I = HIPTENSOR_COMPUTE_DESC_32I;
  cutensorComputeType_t TENSOR_COMPUTE_16F = CUTENSOR_COMPUTE_16F;
  cutensorComputeType_t TENSOR_COMPUTE_16BF = CUTENSOR_COMPUTE_16BF;
  cutensorComputeType_t TENSOR_COMPUTE_32F = CUTENSOR_COMPUTE_32F;
  cutensorComputeType_t TENSOR_COMPUTE_64F = CUTENSOR_COMPUTE_64F;
  cutensorComputeType_t TENSOR_COMPUTE_8U = CUTENSOR_COMPUTE_8U;
  cutensorComputeType_t TENSOR_COMPUTE_8I = CUTENSOR_COMPUTE_8I;
  cutensorComputeType_t TENSOR_COMPUTE_32U = CUTENSOR_COMPUTE_32U;
  cutensorComputeType_t TENSOR_COMPUTE_32I = CUTENSOR_COMPUTE_32I;

  // CHECK: hiptensorCacheMode_t tensorCacheMode_t;
  // CHECK-NEXT: hiptensorCacheMode_t TENSOR_CACHE_MODE_NONE = HIPTENSOR_CACHE_MODE_NONE;
  // CHECK-NEXT: hiptensorCacheMode_t TENSOR_CACHE_MODE_PEDANTIC = HIPTENSOR_CACHE_MODE_PEDANTIC;
  cutensorCacheMode_t tensorCacheMode_t;
  cutensorCacheMode_t TENSOR_CACHE_MODE_NONE = CUTENSOR_CACHE_MODE_NONE;
  cutensorCacheMode_t TENSOR_CACHE_MODE_PEDANTIC = CUTENSOR_CACHE_MODE_PEDANTIC;
#endif

  // CHECK: const hiptensorContractionPlan_t *plan_c = nullptr;
  const cutensorContractionPlan_t *plan_c = nullptr;

  // CHECK: hiptensorWorksizePreference_t TENSOR_WORKSPACE_DEFAULT = HIPTENSOR_WORKSPACE_DEFAULT;
  cutensorWorksizePreference_t TENSOR_WORKSPACE_DEFAULT = CUTENSOR_WORKSPACE_DEFAULT;

  // CUDA: cutensorStatus_t cutensorPermutation(const cutensorHandle_t* handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[], void* B, const cutensorTensorDescriptor_t* descB, const int32_t modeB[], const cudaDataType_t typeScalar, const cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorPermutation(const hiptensorHandle_t* handle, const void* alpha, const void* A, const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], void* B, const hiptensorTensorDescriptor_t* descB, const int32_t modeB[], const hipDataType typeScalar, const hipStream_t stream);
  // CHECK: status = hiptensorPermutation(handle_c, alpha, A, descA, modeA, B, descB, modeB, dataType, stream_t);
  status = cutensorPermutation(handle_c, alpha, A, descA, modeA, B, descB, modeB, dataType, stream_t);

  // CUDA: cutensorStatus_t cutensorReduction(const cutensorHandle_t* handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[], const void* beta, const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[], void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[], cutensorOperator_t opReduce, cutensorComputeType_t typeCompute, void *workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorReduction(const hiptensorHandle_t* handle, const void* alpha, const void* A, const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], const void* beta, const void* C, const hiptensorTensorDescriptor_t* descC, const int32_t modeC[], void* D, const hiptensorTensorDescriptor_t* descD,  const int32_t modeD[], hiptensorOperator_t opReduce, hiptensorComputeType_t typeCompute, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorReduction(handle_c, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, tensorOperator_t, tensorComputeType_t, workspace, workspaceSize2, stream_t);
  status = cutensorReduction(handle_c, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, tensorOperator_t, tensorComputeType_t, workspace, workspaceSize2, stream_t);
#endif

#if (CUTENSOR_MAJOR == 1 && CUTENSOR_MINOR >= 3 && CUTENSOR_PATCH >= 2) || CUTENSOR_MAJOR >= 2
  // CHECK: hiptensorLoggerCallback_t callback;
  cutensorLoggerCallback_t callback;

  // CUDA: cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback);
  // HIP: hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);
  // CHECK: status = hiptensorLoggerSetCallback(callback);
  status = cutensorLoggerSetCallback(callback);

  // CUDA: cutensorStatus_t cutensorLoggerSetFile(FILE* file);
  // HIP: hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);
  // CHECK: status = hiptensorLoggerSetFile(file);
  status = cutensorLoggerSetFile(file);

  // CUDA: cutensorStatus_t cutensorLoggerOpenFile(const char* logFile);
  // HIP: hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);
  // CHECK: status = hiptensorLoggerOpenFile(log);
  status = cutensorLoggerOpenFile(log);

  // CUDA: cutensorStatus_t cutensorLoggerSetLevel(int32_t level);
  // HIP: hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);
  // CHECK: status = hiptensorLoggerSetLevel(level);
  status = cutensorLoggerSetLevel(level);

  // CUDA: cutensorStatus_t cutensorLoggerSetMask(int32_t mask);
  // HIP: hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);
  // CHECK: status = hiptensorLoggerSetMask(mask);
  status = cutensorLoggerSetMask(mask);

  // CUDA: cutensorStatus_t cutensorLoggerForceDisable();
  // HIP: hiptensorStatus_t hiptensorLoggerForceDisable();
  // CHECK: status = hiptensorLoggerForceDisable();
  status = cutensorLoggerForceDisable();
#endif

#if CUTENSOR_MAJOR >= 2
  // CHECK: hiptensorDataType_t tensorDataType_t;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_16F = HIPTENSOR_R_16F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_16F = HIPTENSOR_C_16F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_16BF = HIPTENSOR_R_16BF;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_16BF = HIPTENSOR_C_16BF;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_32F = HIPTENSOR_R_32F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_32F = HIPTENSOR_C_32F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_64F = HIPTENSOR_R_64F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_64F = HIPTENSOR_C_64F;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_8I = HIPTENSOR_R_8I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_8U = HIPTENSOR_R_8U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_32I = HIPTENSOR_R_32I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_32U = HIPTENSOR_R_32U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_4I = HIPTENSOR_R_4I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_4I = HIPTENSOR_C_4I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_4U = HIPTENSOR_R_4U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_4U = HIPTENSOR_C_4U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_8I = HIPTENSOR_C_8I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_8U = HIPTENSOR_C_8U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_16I = HIPTENSOR_R_16I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_16I = HIPTENSOR_C_16I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_16U = HIPTENSOR_R_16U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_16U = HIPTENSOR_C_16U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_32I = HIPTENSOR_C_32I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_32U = HIPTENSOR_C_32U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_64I = HIPTENSOR_R_64I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_64I = HIPTENSOR_C_64I;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_R_64U = HIPTENSOR_R_64U;
  // CHECK-NEXT: hiptensorDataType_t TENSOR_C_64U = HIPTENSOR_C_64U;
  cutensorDataType_t tensorDataType_t;
  cutensorDataType_t TENSOR_R_16F = CUTENSOR_R_16F;
  cutensorDataType_t TENSOR_C_16F = CUTENSOR_C_16F;
  cutensorDataType_t TENSOR_R_16BF = CUTENSOR_R_16BF;
  cutensorDataType_t TENSOR_C_16BF = CUTENSOR_C_16BF;
  cutensorDataType_t TENSOR_R_32F = CUTENSOR_R_32F;
  cutensorDataType_t TENSOR_C_32F = CUTENSOR_C_32F;
  cutensorDataType_t TENSOR_R_64F = CUTENSOR_R_64F;
  cutensorDataType_t TENSOR_C_64F = CUTENSOR_C_64F;
  cutensorDataType_t TENSOR_R_8I = CUTENSOR_R_8I;
  cutensorDataType_t TENSOR_R_8U = CUTENSOR_R_8U;
  cutensorDataType_t TENSOR_R_32I = CUTENSOR_R_32I;
  cutensorDataType_t TENSOR_R_32U = CUTENSOR_R_32U;
  cutensorDataType_t TENSOR_R_4I = CUTENSOR_R_4I;
  cutensorDataType_t TENSOR_C_4I = CUTENSOR_C_4I;
  cutensorDataType_t TENSOR_R_4U = CUTENSOR_R_4U;
  cutensorDataType_t TENSOR_C_4U = CUTENSOR_C_4U;
  cutensorDataType_t TENSOR_C_8I = CUTENSOR_C_8I;
  cutensorDataType_t TENSOR_C_8U = CUTENSOR_C_8U;
  cutensorDataType_t TENSOR_R_16I = CUTENSOR_R_16I;
  cutensorDataType_t TENSOR_C_16I = CUTENSOR_C_16I;
  cutensorDataType_t TENSOR_R_16U = CUTENSOR_R_16U;
  cutensorDataType_t TENSOR_C_16U = CUTENSOR_C_16U;
  cutensorDataType_t TENSOR_C_32I = CUTENSOR_C_32I;
  cutensorDataType_t TENSOR_C_32U = CUTENSOR_C_32U;
  cutensorDataType_t TENSOR_R_64I = CUTENSOR_R_64I;
  cutensorDataType_t TENSOR_C_64I = CUTENSOR_C_64I;
  cutensorDataType_t TENSOR_R_64U = CUTENSOR_R_64U;
  cutensorDataType_t TENSOR_C_64U = CUTENSOR_C_64U;

  // CHECK: hiptensorComputeDescriptor_t TENSOR_COMPUTE_DESC_16F = HIPTENSOR_COMPUTE_DESC_16F;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_DESC_16BF = HIPTENSOR_COMPUTE_DESC_16BF;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_DESC_32F = HIPTENSOR_COMPUTE_DESC_32F;
  // CHECK-NEXT: hiptensorComputeDescriptor_t TENSOR_COMPUTE_DESC_64F = HIPTENSOR_COMPUTE_DESC_64F;
  cutensorComputeDescriptor_t TENSOR_COMPUTE_DESC_16F = CUTENSOR_COMPUTE_DESC_16F;
  cutensorComputeDescriptor_t TENSOR_COMPUTE_DESC_16BF = CUTENSOR_COMPUTE_DESC_16BF;
  cutensorComputeDescriptor_t TENSOR_COMPUTE_DESC_32F = CUTENSOR_COMPUTE_DESC_32F;
  cutensorComputeDescriptor_t TENSOR_COMPUTE_DESC_64F = CUTENSOR_COMPUTE_DESC_64F;

  // CHECK: hiptensorOperationDescriptorAttribute_t tensorOperationDescriptorAttribute_t;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_TAG = HIPTENSOR_OPERATION_DESCRIPTOR_TAG;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_FLOPS = HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT;
  // CHECK-NEXT: hiptensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE;
  cutensorOperationDescriptorAttribute_t tensorOperationDescriptorAttribute_t;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_TAG = CUTENSOR_OPERATION_DESCRIPTOR_TAG;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_FLOPS = CUTENSOR_OPERATION_DESCRIPTOR_FLOPS;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT;
  cutensorOperationDescriptorAttribute_t TENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE = CUTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE;

  // CHECK: hiptensorPlanPreferenceAttribute_t tensorPlanPreferenceAttribute_t;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_CACHE_MODE = HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_ALGO = HIPTENSOR_PLAN_PREFERENCE_ALGO;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_KERNEL_RANK = HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK;
  // CHECK-NEXT: hiptensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_JIT = HIPTENSOR_PLAN_PREFERENCE_JIT;
  cutensorPlanPreferenceAttribute_t tensorPlanPreferenceAttribute_t;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE = CUTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_CACHE_MODE = CUTENSOR_PLAN_PREFERENCE_CACHE_MODE;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT = CUTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_ALGO = CUTENSOR_PLAN_PREFERENCE_ALGO;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_KERNEL_RANK = CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK;
  cutensorPlanPreferenceAttribute_t TENSOR_PLAN_PREFERENCE_JIT = CUTENSOR_PLAN_PREFERENCE_JIT;

  // CHECK: hiptensorJitMode_t tensorJitMode_t;
  // CHECK-NEXT: hiptensorJitMode_t TENSOR_JIT_MODE_NONE = HIPTENSOR_JIT_MODE_NONE;
  // CHECK-NEXT: hiptensorJitMode_t TENSOR_JIT_MODE_DEFAULT = HIPTENSOR_JIT_MODE_DEFAULT;
  cutensorJitMode_t tensorJitMode_t;
  cutensorJitMode_t TENSOR_JIT_MODE_NONE = CUTENSOR_JIT_MODE_NONE;
  cutensorJitMode_t TENSOR_JIT_MODE_DEFAULT = CUTENSOR_JIT_MODE_DEFAULT;

  // CHECK: hiptensorPlanAttribute_t tensorPlanAttribute_t;
  // CHECK-NEXT: hiptensorPlanAttribute_t TENSOR_PLAN_REQUIRED_WORKSPACE = HIPTENSOR_PLAN_REQUIRED_WORKSPACE;
  cutensorPlanAttribute_t tensorPlanAttribute_t;
  cutensorPlanAttribute_t TENSOR_PLAN_REQUIRED_WORKSPACE = CUTENSOR_PLAN_REQUIRED_WORKSPACE;

  // CHECK: hiptensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_NONE = HIPTENSOR_AUTOTUNE_MODE_NONE;
  // CHECK-NEXT: hiptensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_INCREMENTAL = HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL;
  cutensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_NONE = CUTENSOR_AUTOTUNE_MODE_NONE;
  cutensorAutotuneMode_t TENSOR_AUTOTUNE_MODE_INCREMENTAL = CUTENSOR_AUTOTUNE_MODE_INCREMENTAL;

  // CHECK: hiptensorPlan *tensorPlan_p = nullptr;
  // CHECK-NEXT: hiptensorPlan_t tensorPlan_t;
  cutensorPlan *tensorPlan_p = nullptr;
  cutensorPlan_t tensorPlan_t;

  // CHECK: hiptensorPlanPreference *tensorPlanRef_p = nullptr;
  // CHECK-NEXT: hiptensorPlanPreference_t tensorPlanPreference_t;
  cutensorPlanPreference *tensorPlanRef_p = nullptr;
  cutensorPlanPreference_t tensorPlanPreference_t;

  // CHECK: hiptensorOperationDescriptor *tensorOperationDescriptor_p = nullptr;
  // CHECK-NEXT: hiptensorOperationDescriptor_t tensorOperationDescriptor_t;
  cutensorOperationDescriptor *tensorOperationDescriptor_p = nullptr;
  cutensorOperationDescriptor_t tensorOperationDescriptor_t;

  // CUDA: cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void *A, const void *B, const void* beta, const void *C, void *D, void* workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorContract(const hiptensorHandle_t handle, const hiptensorPlan_t plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorContract(handle, tensorPlan_t, alpha, A, B_1, beta, C, D, workspace,  workspaceSize2, stream_t);
  status = cutensorContract(handle, tensorPlan_t, alpha, A, B_1, beta, C, D, workspace, workspaceSize2, stream_t);

  // CUDA: cutensorStatus_t cutensorCreate(cutensorHandle_t* handle);
  // HIP: hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle);
  // CHECK: status = hiptensorCreate(&handle);
  status = cutensorCreate(&handle);

  // CUDA: cutensorStatus_t cutensorDestroy(cutensorHandle_t handle);
  // HIP: hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle);
  // CHECK: status = hiptensorDestroy(handle);
  status = cutensorDestroy(handle);

  // CUDA: cutensorStatus_t cutensorHandleResizePlanCache(cutensorHandle_t handle, const uint32_t numEntries);
  // HIP: hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle, const uint32_t numEntries);
  // CHECK: status = hiptensorHandleResizePlanCache(handle, numModes);
  status = cutensorHandleResizePlanCache(handle, numModes);

  // CUDA: cutensorStatus_t cutensorHandleWritePlanCacheToFile(const cutensorHandle_t handle, const char filename[]);
  // HIP: hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle, const char filename[]);
  // CHECK: status = hiptensorHandleWritePlanCacheToFile(handle, filename);
  status = cutensorHandleWritePlanCacheToFile(handle, filename);

  // CUDA: cutensorStatus_t cutensorHandleReadPlanCacheFromFile(cutensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead);
  // HIP: hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle, const char filename[], uint32_t* numCachelinesRead);
  // CHECK: status = hiptensorHandleReadPlanCacheFromFile(handle, filename, &numCachelinesRead);
  status = cutensorHandleReadPlanCacheFromFile(handle, filename, &numCachelinesRead);

  // CUDA: cutensorStatus_t cutensorWriteKernelCacheToFile(const cutensorHandle_t handle, const char filename[]);
  // HIP: hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle, const char filename[]);
  // CHECK: status = hiptensorWriteKernelCacheToFile(handle, filename);
  status = cutensorWriteKernelCacheToFile(handle, filename);

  // CUDA: cutensorStatus_t cutensorReadKernelCacheFromFile(cutensorHandle_t handle, const char filename[]);
  // HIP: hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle, const char filename[]);
  // CHECK: status = hiptensorReadKernelCacheFromFile(handle, filename);
  status = cutensorReadKernelCacheFromFile(handle, filename);

  // CUDA: cutensorStatus_t cutensorCreateTensorDescriptor(const cutensorHandle_t handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cutensorDataType_t dataType, uint32_t alignmentRequirement);
  // HIP: hiptensorStatus_t hiptensorCreateTensorDescriptor(const hiptensorHandle_t handle, hiptensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t lens[], const int64_t strides[], hiptensorDataType_t dataType, uint32_t alignmentRequirement);
  // CHECK: status = hiptensorCreateTensorDescriptor(handle, tensorDescriptor, numModes, extent, stride, tensorDataType_t, alignmentRequirement);
  status = cutensorCreateTensorDescriptor(handle, tensorDescriptor, numModes, extent, stride, tensorDataType_t, alignmentRequirement);

  // CUDA: cutensorStatus_t cutensorDestroyTensorDescriptor(cutensorTensorDescriptor_t desc);
  // HIP: hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc);
  // CHECK: status = hiptensorDestroyTensorDescriptor(*tensorDescriptor);
  status = cutensorDestroyTensorDescriptor(*tensorDescriptor);

  // CHECK: hiptensorComputeDescriptor_t tensorComputeDescriptor_t = nullptr;
  cutensorComputeDescriptor_t tensorComputeDescriptor_t = nullptr;

  // CUDA: cutensorStatus_t cutensorCreateContraction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], const cutensorComputeDescriptor_t descCompute);
  // HIP: hiptensorStatus_t hiptensorCreateContraction(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc, const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA, const hiptensorTensorDescriptor_t descB, const int32_t modeB[], hiptensorOperator_t opB, const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC, const hiptensorTensorDescriptor_t descD, const int32_t modeD[], const hiptensorComputeDescriptor_t descCompute);
  // CHECK: status = hiptensorCreateContraction(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, OpB, *descC, modeC, OpC, *descD, modeD, tensorComputeDescriptor_t);
  status = cutensorCreateContraction(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, OpB, *descC, modeC, OpC, *descD, modeD, tensorComputeDescriptor_t);

  // CUDA: cutensorStatus_t cutensorDestroyOperationDescriptor(cutensorOperationDescriptor_t desc);
  // HIP: hiptensorStatus_t hiptensorDestroyOperationDescriptor(hiptensorOperationDescriptor_t desc);
  // CHECK: status = hiptensorDestroyOperationDescriptor(tensorOperationDescriptor_t);
  status = cutensorDestroyOperationDescriptor(tensorOperationDescriptor_t);

  // CUDA: cutensorStatus_t cutensorOperationDescriptorSetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes);
  // HIP: hiptensorStatus_t hiptensorOperationDescriptorSetAttribute(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t desc, hiptensorOperationDescriptorAttribute_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hiptensorOperationDescriptorSetAttribute(handle, tensorOperationDescriptor_t, tensorOperationDescriptorAttribute_t, buf, sizeInBytes);
  status = cutensorOperationDescriptorSetAttribute(handle, tensorOperationDescriptor_t, tensorOperationDescriptorAttribute_t, buf, sizeInBytes);

  // CUDA: cutensorStatus_t cutensorOperationDescriptorGetAttribute(const cutensorHandle_t handle, cutensorOperationDescriptor_t desc, cutensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes);
  // HIP: hiptensorStatus_t hiptensorOperationDescriptorGetAttribute(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t desc, hiptensorOperationDescriptorAttribute_t attr, void* buf, size_t sizeInBytes);
  // CHECK: status = hiptensorOperationDescriptorGetAttribute(handle, tensorOperationDescriptor_t, tensorOperationDescriptorAttribute_t, buf, sizeInBytes);
  status = cutensorOperationDescriptorGetAttribute(handle, tensorOperationDescriptor_t, tensorOperationDescriptorAttribute_t, buf, sizeInBytes);

  // CUDA: cutensorStatus_t cutensorCreatePermutation(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], const cutensorComputeDescriptor_t descCompute);
  // HIP: hiptensorStatus_t hiptensorCreatePermutation(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc, const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA, const hiptensorTensorDescriptor_t descB, const int32_t modeB[], const hiptensorComputeDescriptor_t descCompute);
  // CHECK: status = hiptensorCreatePermutation(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, tensorComputeDescriptor_t);
  status = cutensorCreatePermutation(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, tensorComputeDescriptor_t);

  // CUDA: cutensorStatus_t cutensorCreatePlanPreference(const cutensorHandle_t handle, cutensorPlanPreference_t* pref, cutensorAlgo_t algo, cutensorJitMode_t jitMode);
  // HIP: hiptensorStatus_t hiptensorCreatePlanPreference(const hiptensorHandle_t handle, hiptensorPlanPreference_t* pref, hiptensorAlgo_t algo, hiptensorJitMode_t jitMode);
  // CHECK: status = hiptensorCreatePlanPreference(handle, &tensorPlanPreference_t, tensorAlgo_t, tensorJitMode_t);
  status = cutensorCreatePlanPreference(handle, &tensorPlanPreference_t, tensorAlgo_t, tensorJitMode_t);

  // CUDA: cutensorStatus_t cutensorDestroyPlanPreference(cutensorPlanPreference_t pref);
  // HIP: hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref);
  // CHECK: status = hiptensorDestroyPlanPreference(tensorPlanPreference_t);
  status = cutensorDestroyPlanPreference(tensorPlanPreference_t);

  // CUDA: cutensorStatus_t cutensorPlanPreferenceSetAttribute(const cutensorHandle_t handle, cutensorPlanPreference_t pref, cutensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes);
  // HIP: hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(const hiptensorHandle_t handle, hiptensorPlanPreference_t pref, hiptensorPlanPreferenceAttribute_t attr, const void* buf, size_t sizeInBytes);
  // CHECK: status = hiptensorPlanPreferenceSetAttribute(handle, tensorPlanPreference_t, tensorPlanPreferenceAttribute_t, buf, sizeInBytes);
  status = cutensorPlanPreferenceSetAttribute(handle, tensorPlanPreference_t, tensorPlanPreferenceAttribute_t, buf, sizeInBytes);

  // CUDA: cutensorStatus_t cutensorPlanGetAttribute(const cutensorHandle_t handle, const cutensorPlan_t plan, cutensorPlanAttribute_t attr, void* buf, size_t sizeInBytes);
  // HIP: hiptensorStatus_t hiptensorPlanGetAttribute(const hiptensorHandle_t  handle, const hiptensorPlan_t plan, hiptensorPlanAttribute_t attr, void* buf, size_t sizeInBytes);
  // CHECK: status = hiptensorPlanGetAttribute(handle, tensorPlan_t, tensorPlanAttribute_t, buf, sizeInBytes);
  status = cutensorPlanGetAttribute(handle, tensorPlan_t, tensorPlanAttribute_t, buf, sizeInBytes);

  // CUDA: cutensorStatus_t cutensorEstimateWorkspaceSize(const cutensorHandle_t handle, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t planPref, const cutensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate);
  // HIP: hiptensorStatus_t hiptensorEstimateWorkspaceSize(const hiptensorHandle_t handle, const hiptensorOperationDescriptor_t desc, const hiptensorPlanPreference_t planPref, const hiptensorWorksizePreference_t workspacePref, uint64_t* workspaceSizeEstimate);
  // CHECK: status = hiptensorEstimateWorkspaceSize(handle, tensorOperationDescriptor_t, tensorPlanPreference_t, tensorWorksizePreference_t, &workspaceSizeEstimate);
  status = cutensorEstimateWorkspaceSize(handle, tensorOperationDescriptor_t, tensorPlanPreference_t, tensorWorksizePreference_t, &workspaceSizeEstimate);

  // CUDA: cutensorStatus_t cutensorCreatePlan(const cutensorHandle_t handle, cutensorPlan_t* plan, const cutensorOperationDescriptor_t desc, const cutensorPlanPreference_t pref, uint64_t workspaceSizeLimit);
  // HIP: hiptensorStatus_t hiptensorCreatePlan(const hiptensorHandle_t handle, hiptensorPlan_t* plan, const hiptensorOperationDescriptor_t desc, const hiptensorPlanPreference_t pref, uint64_t workspaceSizeLimit);
  // CHECK: status = hiptensorCreatePlan(handle, &tensorPlan_t, tensorOperationDescriptor_t, tensorPlanPreference_t, workspaceSizeEstimate);
  status = cutensorCreatePlan(handle, &tensorPlan_t, tensorOperationDescriptor_t, tensorPlanPreference_t, workspaceSizeEstimate);

  // CUDA: cutensorStatus_t cutensorDestroyPlan(cutensorPlan_t plan);
  // HIP: hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan);
  // CHECK: status = hiptensorDestroyPlan(tensorPlan_t);
  status = cutensorDestroyPlan(tensorPlan_t);

  // CUDA: cutensorStatus_t cutensorPermute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, void* B, const cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorPermute(const hiptensorHandle_t handle, const hiptensorPlan_t plan, const void* alpha, const void* A, void* B, const hipStream_t stream);
  // CHECK: status = hiptensorPermute(handle, tensorPlan_t, alpha, A, B, stream_t);
  status = cutensorPermute(handle, tensorPlan_t, alpha, A, B, stream_t);

  // CUDA: cutensorStatus_t cutensorCreateElementwiseBinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAC, const cutensorComputeDescriptor_t descCompute);
  // HIP: hiptensorStatus_t hiptensorCreateElementwiseBinary(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc, const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA, const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC, const hiptensorTensorDescriptor_t descD, const int32_t modeD[], hiptensorOperator_t opAC, const hiptensorComputeDescriptor_t descCompute);
  // CHECK: status = hiptensorCreateElementwiseBinary(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descC, modeC, OpC, *descD, modeD, OpAC, tensorComputeDescriptor_t);
  status = cutensorCreateElementwiseBinary(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descC, modeC, OpC, *descD, modeD, OpAC, tensorComputeDescriptor_t);

  // CUDA: cutensorStatus_t cutensorElementwiseBinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorElementwiseBinaryExecute(const hiptensorHandle_t handle, const hiptensorPlan_t plan, const void* alpha, const void* A, const void* gamma, const void* C, void* D, hipStream_t stream);
  // CHECK: status = hiptensorElementwiseBinaryExecute(handle, tensorPlan_t, alpha, A, gamma, C, D, stream_t);
  status = cutensorElementwiseBinaryExecute(handle, tensorPlan_t, alpha, A, gamma, C, D, stream_t);

  // CUDA: cutensorStatus_t cutensorCreateElementwiseTrinary(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descB, const int32_t modeB[], cutensorOperator_t opB, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opAB, cutensorOperator_t opABC, const cutensorComputeDescriptor_t descCompute);
  // HIP: hiptensorStatus_t hiptensorCreateElementwiseTrinary(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc, const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA, const hiptensorTensorDescriptor_t descB, const int32_t modeB[], hiptensorOperator_t opB, const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC, const hiptensorTensorDescriptor_t descD, const int32_t modeD[], hiptensorOperator_t opAB, hiptensorOperator_t opABC, const hiptensorComputeDescriptor_t descCompute);
  // CHECK: status = hiptensorCreateElementwiseTrinary(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, OpB, *descC, modeC, OpC, *descD, modeD, OpAB, OpABC, tensorComputeDescriptor_t);
  status = cutensorCreateElementwiseTrinary(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descB, modeB, OpB, *descC, modeC, OpC, *descD, modeD, OpAB, OpABC, tensorComputeDescriptor_t);

  // CUDA: cutensorStatus_t cutensorElementwiseTrinaryExecute(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorElementwiseTrinaryExecute(const hiptensorHandle_t handle, const hiptensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* B, const void* gamma, const void* C, void* D, hipStream_t stream);
  // CHECK: status = hiptensorElementwiseTrinaryExecute(handle, tensorPlan_t, alpha, A, beta, B, gamma, C, D, stream_t);
  status = cutensorElementwiseTrinaryExecute(handle, tensorPlan_t, alpha, A, beta, B, gamma, C, D, stream_t);

  // CUDA: cutensorStatus_t cutensorCreateReduction(const cutensorHandle_t handle, cutensorOperationDescriptor_t* desc, const cutensorTensorDescriptor_t descA, const int32_t modeA[], cutensorOperator_t opA, const cutensorTensorDescriptor_t descC, const int32_t modeC[], cutensorOperator_t opC, const cutensorTensorDescriptor_t descD, const int32_t modeD[], cutensorOperator_t opReduce, const cutensorComputeDescriptor_t descCompute);
  // HIP: hiptensorStatus_t hiptensorCreateReduction(const hiptensorHandle_t handle, hiptensorOperationDescriptor_t* desc, const hiptensorTensorDescriptor_t descA, const int32_t modeA[], hiptensorOperator_t opA, const hiptensorTensorDescriptor_t descC, const int32_t modeC[], hiptensorOperator_t opC, const hiptensorTensorDescriptor_t descD, const int32_t modeD[], hiptensorOperator_t opReduce, const hiptensorComputeDescriptor_t descCompute);
  // CHECK: status = hiptensorCreateReduction(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descC, modeC, OpC, *descD, modeD, OpReduce, tensorComputeDescriptor_t);
  status = cutensorCreateReduction(handle, &tensorOperationDescriptor_t, *descA, modeA, OpA, *descC, modeC, OpC, *descD, modeD, OpReduce, tensorComputeDescriptor_t);

  // CUDA: cutensorStatus_t cutensorReduce(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorReduce(const hiptensorHandle_t handle, const hiptensorPlan_t plan, const void* alpha, const void* A, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorReduce(handle, tensorPlan_t, alpha, A, beta, C, D, workspace, workspaceSize, stream_t);
  status = cutensorReduce(handle, tensorPlan_t, alpha, A, beta, C, D, workspace, workspaceSize, stream_t);
#endif

  return 0;
}
