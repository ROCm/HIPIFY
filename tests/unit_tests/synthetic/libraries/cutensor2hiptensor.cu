// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hiptensor.h"
#include "cutensor.h"
// CHECK-NOT: #include "hiptensor.h"

int main() {
  printf("25. cuTensor API to hipTensor API synthetic test\n");

  // CHECK: hiptensorHandle_t handle;
  // CHECK-NEXT: const hiptensorHandle_t *handle_c = nullptr;
  // CHECK-NEXT: hiptensorHandle_t *handle2 = nullptr;
  cutensorHandle_t handle;
  const cutensorHandle_t *handle_c = nullptr;
  cutensorHandle_t *handle2 = nullptr;

  //CHECK: hiptensorStatus_t status;
  cutensorStatus_t status;

  //CHECK: hiptensorTensorDescriptor_t *tensorDescriptor = nullptr;
  //CHECK-NEXT: hiptensorTensorDescriptor_t *descA = nullptr;
  //CHECK-NEXT: hiptensorTensorDescriptor_t *descB = nullptr;
  //CHECK-NEXT: hiptensorTensorDescriptor_t *descC = nullptr;
  //CHECK-NEXT: hiptensorTensorDescriptor_t *descD = nullptr;
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
  void *D = nullptr;
  const int32_t *modeD = nullptr;
  void *workspace = nullptr;
  const char *err = nullptr;
  const char *log = nullptr;
  size_t ver = 0;
  FILE *file = nullptr;
  int32_t level = 0;
  int32_t mask = 0;

#if CUTENSOR_MAJOR >= 2
  // CHECK: hiptensorComputeType_t tensorDataType_t;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_16F = HIPTENSOR_COMPUTE_16F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_16BF = HIPTENSOR_COMPUTE_16BF;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_32F = HIPTENSOR_COMPUTE_32F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_C_32F = HIPTENSOR_COMPUTE_C32F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_64F = HIPTENSOR_COMPUTE_64F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_C_64F = HIPTENSOR_COMPUTE_C64F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_8I = HIPTENSOR_COMPUTE_8I;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_8U = HIPTENSOR_COMPUTE_8U;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_32I = HIPTENSOR_COMPUTE_32I;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_R_32U = HIPTENSOR_COMPUTE_32U;
  cutensorDataType_t tensorDataType_t;
  cutensorDataType_t TENSOR_R_16F = CUTENSOR_R_16F;
  cutensorDataType_t TENSOR_R_16BF = CUTENSOR_R_16BF;
  cutensorDataType_t TENSOR_R_32F = CUTENSOR_R_32F;
  cutensorDataType_t TENSOR_C_32F = CUTENSOR_C_32F;
  cutensorDataType_t TENSOR_R_64F = CUTENSOR_R_64F;
  cutensorDataType_t TENSOR_C_64F = CUTENSOR_C_64F;
  cutensorDataType_t TENSOR_R_8I = CUTENSOR_R_8I;
  cutensorDataType_t TENSOR_R_8U = CUTENSOR_R_8U;
  cutensorDataType_t TENSOR_R_32I = CUTENSOR_R_32I;
  cutensorDataType_t TENSOR_R_32U = CUTENSOR_R_32U;

  // CHECK: hiptensorContractionPlan_t tensorPlan2_t;
  cutensorPlan_t tensorPlan2_t;

  // CUDA: cutensorStatus_t cutensorContract(const cutensorHandle_t handle, const cutensorPlan_t plan, const void* alpha, const void *A, const void *B, const void* beta, const void *C, void *D, void* workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t* handle, const hiptensorContractionPlan_t* plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorContraction(handle, tensorPlan2_t, alpha, A, B_1, beta, C, D, workspace,  workspaceSize2, stream_t);
  status = cutensorContract(handle, tensorPlan2_t, alpha, A, B_1, beta, C, D, workspace, workspaceSize2, stream_t);

   // CUDA: cutensorStatus_t cutensorCreate(cutensorHandle_t* handle);
   // HIP: hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle);
   // CHECK: status = hiptensorCreate(&handle);
   status = cutensorCreate(&handle);

   // CUDA: cutensorStatus_t cutensorDestroy(cutensorHandle_t handle);
   // HIP: hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle);
   // CHECK: status = hiptensorDestroy(handle);
   status = cutensorDestroy(handle);
#endif

#if CUTENSOR_MAJOR >= 1
  // CHECK: hiptensorOperator_t tensorOperator_t;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_IDENTITY = HIPTENSOR_OP_IDENTITY;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_SQRT = HIPTENSOR_OP_SQRT;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_ADD = HIPTENSOR_OP_ADD;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_MUL = HIPTENSOR_OP_MUL;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_MAX = HIPTENSOR_OP_MAX;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_MIN = HIPTENSOR_OP_MIN;
  // CHECK-NEXT hiptensorOperator_t TENSOR_OP_UNKNOWN = HIPTENSOR_OP_UNKNOWN;
  cutensorOperator_t tensorOperator_t;
  cutensorOperator_t TENSOR_OP_IDENTITY = CUTENSOR_OP_IDENTITY;
  cutensorOperator_t TENSOR_OP_SQRT = CUTENSOR_OP_SQRT;
  cutensorOperator_t TENSOR_OP_ADD = CUTENSOR_OP_ADD;
  cutensorOperator_t TENSOR_OP_MUL = CUTENSOR_OP_MUL;
  cutensorOperator_t TENSOR_OP_MAX = CUTENSOR_OP_MAX;
  cutensorOperator_t TENSOR_OP_MIN = CUTENSOR_OP_MIN;
  cutensorOperator_t TENSOR_OP_UNKNOWN = CUTENSOR_OP_UNKNOWN;

  // CHECK: hiptensorStatus_t tensorStatus_t;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_SUCCESS = HIPTENSOR_STATUS_SUCCESS;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_NOT_INITIALIZED = HIPTENSOR_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_ALLOC_FAILED = HIPTENSOR_STATUS_ALLOC_FAILED;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_INVALID_VALUE = HIPTENSOR_STATUS_INVALID_VALUE;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_ARCH_MISMATCH = HIPTENSOR_STATUS_ARCH_MISMATCH;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_EXECUTION_FAILED = HIPTENSOR_STATUS_EXECUTION_FAILED;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_INTERNAL_ERROR = HIPTENSOR_STATUS_INTERNAL_ERROR;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_NOT_SUPPORTED = HIPTENSOR_STATUS_NOT_SUPPORTED;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_INSUFFICIENT_WORKSPACE = HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
  // CHECK-NEXT hiptensorStatus_t TENSOR_STATUS_INSUFFICIENT_DRIVER = HIPTENSOR_STATUS_INSUFFICIENT_DRIVER;
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
  // CHECK-NEXT hiptensorAlgo_t TENSOR_ALGO_DEFAULT = HIPTENSOR_ALGO_DEFAULT;
  cutensorAlgo_t tensorAlgo_t;
  cutensorAlgo_t TENSOR_ALGO_DEFAULT = CUTENSOR_ALGO_DEFAULT;

  // CHECK: hiptensorWorksizePreference_t tensorWorksizePreference_t;
  // Check-NEXT TENSOR_WORKSPACE_MIN = HIPTENSOR_WORKSPACE_MIN;
  // CHECK-NEXT TENSOR_WORKSPACE_MAX = HIPTENSOR_WORKSPACE_MAX;
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
  // CHECK: hiptensorComputeType_t tensorComputeType_t;
  cutensorComputeType_t tensorComputeType_t;

#if CUTENSOR_MINOR >= 2
  // CHECK: hiptensorStatus_t TENSOR_STATUS_IO_ERROR = HIPTENSOR_STATUS_IO_ERROR;
  cutensorStatus_t TENSOR_STATUS_IO_ERROR = CUTENSOR_STATUS_IO_ERROR;

  // CHECK hiptensorComputeType_t TENSOR_COMPUTE_16F = HIPTENSOR_COMPUTE_16F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_16BF = HIPTENSOR_COMPUTE_16BF;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_32F = HIPTENSOR_COMPUTE_32F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_64F = HIPTENSOR_COMPUTE_64F;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_8U = HIPTENSOR_COMPUTE_8U;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_8I = HIPTENSOR_COMPUTE_8I;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_32U = HIPTENSOR_COMPUTE_32U;
  // CHECK-NEXT hiptensorComputeType_t TENSOR_COMPUTE_32I = HIPTENSOR_COMPUTE_32I;
  cutensorComputeType_t TENSOR_COMPUTE_16F = CUTENSOR_COMPUTE_16F;
  cutensorComputeType_t TENSOR_COMPUTE_16BF = CUTENSOR_COMPUTE_16BF;
  cutensorComputeType_t TENSOR_COMPUTE_32F = CUTENSOR_COMPUTE_32F;
  cutensorComputeType_t TENSOR_COMPUTE_64F = CUTENSOR_COMPUTE_64F;
  cutensorComputeType_t TENSOR_COMPUTE_8U = CUTENSOR_COMPUTE_8U;
  cutensorComputeType_t TENSOR_COMPUTE_8I = CUTENSOR_COMPUTE_8I;
  cutensorComputeType_t TENSOR_COMPUTE_32U = CUTENSOR_COMPUTE_32U;
  cutensorComputeType_t TENSOR_COMPUTE_32I = CUTENSOR_COMPUTE_32I;
#endif

  // CHECK: const hiptensorContractionPlan_t *plan_c = nullptr;
  const cutensorContractionPlan_t *plan_c = nullptr;

  // CHECK: hiptensorWorksizePreference_t TENSOR_WORKSPACE_RECOMMENDED = HIPTENSOR_WORKSPACE_RECOMMENDED;
  cutensorWorksizePreference_t TENSOR_WORKSPACE_RECOMMENDED = CUTENSOR_WORKSPACE_RECOMMENDED;

#if CUDA_VERSION >= 8000
  // CUDA: cutensorStatus_t cutensorInitTensorDescriptor(const cutensorHandle_t* handle, cutensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t extent[], const int64_t stride[], cudaDataType_t dataType, cutensorOperator_t unaryOp);
  // HIP: hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t* handle, hiptensorTensorDescriptor_t* desc, const uint32_t numModes, const int64_t lens[], const int64_t strides[], hipDataType dataType, hiptensorOperator_t unaryOp);
  // CHECK: status = hiptensorInitTensorDescriptor(handle_c, tensorDescriptor, numModes, extent, stride, dataType, tensorOperator_t);
  status = cutensorInitTensorDescriptor(handle_c, tensorDescriptor, numModes, extent, stride, dataType, tensorOperator_t);
#endif

  // CUDA: cutensorStatus_t cutensorPermutation(const cutensorHandle_t* handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[], void* B, const cutensorTensorDescriptor_t* descB, const int32_t modeB[], const cudaDataType_t typeScalar, const cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorPermutation(const hiptensorHandle_t* handle, const void* alpha, const void* A, const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], void* B, const hiptensorTensorDescriptor_t* descB, const int32_t modeB[], const hipDataType typeScalar, const hipStream_t stream);
  // CHECK: status = hiptensorPermutation(handle_c, alpha, A, descA, modeA, B, descB, modeB, dataType, stream_t);
  status = cutensorPermutation(handle_c, alpha, A, descA, modeA, B, descB, modeB, dataType, stream_t);

  // CUDA: cutensorStatus_t cutensorContraction(const cutensorHandle_t* handle, const cutensorContractionPlan_t* plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void *workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorContraction(const hiptensorHandle_t* handle, const hiptensorContractionPlan_t* plan, const void* alpha, const void* A, const void* B, const void* beta, const void* C, void* D, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorContraction(handle_c, plan_c, alpha, A, B_1, beta, C, D, workspace, workspaceSize, stream_t);
  status = cutensorContraction(handle_c, plan_c, alpha, A, B_1, beta, C, D, workspace, workspaceSize, stream_t);

  // CUDA: cutensorStatus_t cutensorReduction(const cutensorHandle_t* handle, const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[], const void* beta, const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[], void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[], cutensorOperator_t opReduce, cutensorComputeType_t typeCompute, void *workspace, uint64_t workspaceSize, cudaStream_t stream);
  // HIP: hiptensorStatus_t hiptensorReduction(const hiptensorHandle_t* handle, const void* alpha, const void* A, const hiptensorTensorDescriptor_t* descA, const int32_t modeA[], const void* beta, const void* C, const hiptensorTensorDescriptor_t* descC, const int32_t modeC[], void* D, const hiptensorTensorDescriptor_t* descD,  const int32_t modeD[], hiptensorOperator_t opReduce, hiptensorComputeType_t typeCompute, void* workspace, uint64_t workspaceSize, hipStream_t stream);
  // CHECK: status = hiptensorReduction(handle_c, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, tensorOperator_t, tensorComputeType_t, workspace, workspaceSize2, stream_t);
  status = cutensorReduction(handle_c, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, tensorOperator_t, tensorComputeType_t, workspace, workspaceSize2, stream_t);
#endif

#if (CUTENSOR_MAJOR == 1 && CUTENSOR_MINOR >= 7)
  // CUDA: cutensorStatus_t cutensorCreate(cutensorHandle_t* handle);
  // HIP: hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle);
  // CHECK: status = hiptensorCreate(&handle2);
  status = cutensorCreate(&handle2);

  // CUDA: cutensorStatus_t cutensorDestroy(cutensorHandle_t handle);
  // HIP: hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle);
  // CHECK: status = hiptensorDestroy(handle2);
  status = cutensorDestroy(handle2);
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

  return 0;
}
