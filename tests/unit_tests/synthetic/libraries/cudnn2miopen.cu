// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --skip-excluded-preprocessor-conditional-blocks --experimental -roc %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "miopen/miopen.h"
#include "cudnn.h"

int main() {
  printf("15. cuDNN API to MIOpen API synthetic test\n");

  // CHECK: miopenStatus_t dnnStatus_t;
  // CHECK-NEXT: miopenStatus_t STATUS_SUCCESS = miopenStatusSuccess;
  // CHECK-NEXT: miopenStatus_t STATUS_NOT_INITIALIZED = miopenStatusNotInitialized;
  // CHECK-NEXT: miopenStatus_t STATUS_ALLOC_FAILED = miopenStatusAllocFailed;
  // CHECK-NEXT: miopenStatus_t STATUS_BAD_PARAM = miopenStatusBadParm;
  // CHECK-NEXT: miopenStatus_t STATUS_INTERNAL_ERROR = miopenStatusInternalError;
  // CHECK-NEXT: miopenStatus_t STATUS_INVALID_VALUE = miopenStatusInvalidValue;
  // CHECK-NEXT: miopenStatus_t STATUS_NOT_SUPPORTED = miopenStatusUnsupportedOp;
  cudnnStatus_t dnnStatus_t;
  cudnnStatus_t STATUS_SUCCESS = CUDNN_STATUS_SUCCESS;
  cudnnStatus_t STATUS_NOT_INITIALIZED = CUDNN_STATUS_NOT_INITIALIZED;
  cudnnStatus_t STATUS_ALLOC_FAILED = CUDNN_STATUS_ALLOC_FAILED;
  cudnnStatus_t STATUS_BAD_PARAM = CUDNN_STATUS_BAD_PARAM;
  cudnnStatus_t STATUS_INTERNAL_ERROR = CUDNN_STATUS_INTERNAL_ERROR;
  cudnnStatus_t STATUS_INVALID_VALUE = CUDNN_STATUS_INVALID_VALUE;
  cudnnStatus_t STATUS_NOT_SUPPORTED = CUDNN_STATUS_NOT_SUPPORTED;

  // CHECK: miopenStatus_t status;
  cudnnStatus_t status;

  // CHECK: miopenHandle_t handle;
  cudnnHandle_t handle;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreate(miopenHandle_t* handle);
  // CHECK: status = miopenCreate(&handle);
  status = cudnnCreate(&handle);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroy(miopenHandle_t handle);
  // CHECK: status = miopenDestroy(handle);
  status = cudnnDestroy(handle);

  const char* const_ch = nullptr;

  // CUDA: const char *CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status);
  // MIOPEN: MIOPEN_EXPORT const char* miopenGetErrorString(miopenStatus_t error);
  // CHECK: const_ch = miopenGetErrorString(status);
  const_ch = cudnnGetErrorString(status);

  // CHECK: miopenAcceleratorQueue_t streamId;
  cudaStream_t streamId;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetStream(miopenHandle_t handle, miopenAcceleratorQueue_t streamID);
  // CHECK: status = miopenSetStream(handle, streamId);
  status = cudnnSetStream(handle, streamId);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetStream(miopenHandle_t handle, miopenAcceleratorQueue_t* streamID);
  // CHECK: status = miopenGetStream(handle, &streamId);
  status = cudnnGetStream(handle, &streamId);

  // CHECK: miopenTensorDescriptor_t tensorDescriptor;
  cudnnTensorDescriptor_t tensorDescriptor;

  // CHECK: miopenConvolutionDescriptor_t convolutionDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;

  // CHECK: miopenPoolingDescriptor_t poolingDescriptor;
  cudnnPoolingDescriptor_t poolingDescriptor;

  // CHECK: miopenLRNDescriptor_t LRNDescriptor;
  cudnnLRNDescriptor_t LRNDescriptor;

  // CHECK: miopenActivationDescriptor_t activationDescriptor;
  cudnnActivationDescriptor_t activationDescriptor;

  // CHECK: miopenRNNDescriptor_t RNNDescriptor;
  cudnnRNNDescriptor_t RNNDescriptor;

  // CHECK: miopenCTCLossDescriptor_t CTCLossDescriptor;
  cudnnCTCLossDescriptor_t CTCLossDescriptor;

  // CHECK: miopenDropoutDescriptor_t DropoutDescriptor;
  cudnnDropoutDescriptor_t DropoutDescriptor;

  // CHECK: miopenReduceTensorDescriptor_t ReduceTensorDescriptor;
  cudnnReduceTensorDescriptor_t ReduceTensorDescriptor;

  // CHECK: miopenDataType_t dataType;
  // CHECK-NEXT: miopenDataType_t DATA_FLOAT = miopenFloat;
  // CHECK-NEXT: miopenDataType_t DATA_DOUBLE = miopenDouble;
  // CHECK-NEXT: miopenDataType_t DATA_HALF = miopenHalf;
  // CHECK-NEXT: miopenDataType_t DATA_INT8 = miopenInt8;
  // CHECK-NEXT: miopenDataType_t DATA_INT32 = miopenInt32;
  // CHECK-NEXT: miopenDataType_t DATA_INT8x4 = miopenInt8x4;
  // CHECK-NEXT: miopenDataType_t DATA_BFLOAT16 = miopenBFloat16;
  cudnnDataType_t dataType;
  cudnnDataType_t DATA_FLOAT = CUDNN_DATA_FLOAT;
  cudnnDataType_t DATA_DOUBLE = CUDNN_DATA_DOUBLE;
  cudnnDataType_t DATA_HALF = CUDNN_DATA_HALF;
  cudnnDataType_t DATA_INT8 = CUDNN_DATA_INT8;
  cudnnDataType_t DATA_INT32 = CUDNN_DATA_INT32;
  cudnnDataType_t DATA_INT8x4 = CUDNN_DATA_INT8x4;
  cudnnDataType_t DATA_BFLOAT16 = CUDNN_DATA_BFLOAT16;

  // CHECK: miopenTensorOp_t tensorOp;
  // CHECK-NEXT: miopenTensorOp_t OP_TENSOR_ADD = miopenTensorOpAdd;
  // CHECK-NEXT: miopenTensorOp_t OP_TENSOR_MUL = miopenTensorOpMul;
  // CHECK-NEXT: miopenTensorOp_t OP_TENSOR_MIN = miopenTensorOpMin;
  // CHECK-NEXT: miopenTensorOp_t OP_TENSOR_MAX = miopenTensorOpMax;
  cudnnOpTensorOp_t tensorOp;
  cudnnOpTensorOp_t OP_TENSOR_ADD = CUDNN_OP_TENSOR_ADD;
  cudnnOpTensorOp_t OP_TENSOR_MUL = CUDNN_OP_TENSOR_MUL;
  cudnnOpTensorOp_t OP_TENSOR_MIN = CUDNN_OP_TENSOR_MIN;
  cudnnOpTensorOp_t OP_TENSOR_MAX = CUDNN_OP_TENSOR_MAX;

  // CHECK: miopenConvolutionMode_t convolutionMode;
  cudnnConvolutionMode_t convolutionMode;

  // CHECK: miopenPoolingMode_t poolingMode;
  // CHECK-NEXT: miopenPoolingMode_t POOLING_MAX = miopenPoolingMax;
  cudnnPoolingMode_t poolingMode;
  cudnnPoolingMode_t POOLING_MAX = CUDNN_POOLING_MAX;

  // CHECK: miopenLRNMode_t LRNMode;
  // CHECK-NEXT: miopenLRNMode_t LRN_CROSS_CHANNEL_DIM1 = miopenLRNCrossChannel;
  cudnnLRNMode_t LRNMode;
  cudnnLRNMode_t LRN_CROSS_CHANNEL_DIM1 = CUDNN_LRN_CROSS_CHANNEL_DIM1;

  // CHECK: miopenBatchNormMode_t batchNormMode;
  // CHECK-NEXT: miopenBatchNormMode_t BATCHNORM_PER_ACTIVATION = miopenBNPerActivation;
  // CHECK-NEXT: miopenBatchNormMode_t BATCHNORM_SPATIAL = miopenBNSpatial;
  cudnnBatchNormMode_t batchNormMode;
  cudnnBatchNormMode_t BATCHNORM_PER_ACTIVATION = CUDNN_BATCHNORM_PER_ACTIVATION;
  cudnnBatchNormMode_t BATCHNORM_SPATIAL = CUDNN_BATCHNORM_SPATIAL;

  // CHECK: miopenActivationMode_t activationMode;
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_RELU = miopenActivationRELU;
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_TANH = miopenActivationTANH;
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_CLIPPED_RELU = miopenActivationCLIPPEDRELU;
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_ELU = miopenActivationELU;
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_IDENTITY = miopenActivationPASTHRU;
  cudnnActivationMode_t activationMode;
  cudnnActivationMode_t ACTIVATION_RELU = CUDNN_ACTIVATION_RELU;
  cudnnActivationMode_t ACTIVATION_TANH = CUDNN_ACTIVATION_TANH;
  cudnnActivationMode_t ACTIVATION_CLIPPED_RELU = CUDNN_ACTIVATION_CLIPPED_RELU;
  cudnnActivationMode_t ACTIVATION_ELU = CUDNN_ACTIVATION_ELU;
  cudnnActivationMode_t ACTIVATION_IDENTITY = CUDNN_ACTIVATION_IDENTITY;

  // CHECK: miopenSoftmaxAlgorithm_t softmaxAlgorithm;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_FAST = MIOPEN_SOFTMAX_FAST;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_ACCURATE = MIOPEN_SOFTMAX_ACCURATE;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_LOG = MIOPEN_SOFTMAX_LOG;
  cudnnSoftmaxAlgorithm_t softmaxAlgorithm;
  cudnnSoftmaxAlgorithm_t SOFTMAX_FAST = CUDNN_SOFTMAX_FAST;
  cudnnSoftmaxAlgorithm_t SOFTMAX_ACCURATE = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxAlgorithm_t SOFTMAX_LOG = CUDNN_SOFTMAX_LOG;

  // CHECK: miopenReduceTensorOp_t reduceTensorOp;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_ADD = MIOPEN_REDUCE_TENSOR_ADD;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_MUL = MIOPEN_REDUCE_TENSOR_MUL;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_MIN = MIOPEN_REDUCE_TENSOR_MIN;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_MAX = MIOPEN_REDUCE_TENSOR_MAX;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_AMAX = MIOPEN_REDUCE_TENSOR_AMAX;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_AVG = MIOPEN_REDUCE_TENSOR_AVG;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_NORM1 = MIOPEN_REDUCE_TENSOR_NORM1;
  // CHECK-NEXT: miopenReduceTensorOp_t REDUCE_TENSOR_NORM2 = MIOPEN_REDUCE_TENSOR_NORM2;
  cudnnReduceTensorOp_t reduceTensorOp;
  cudnnReduceTensorOp_t REDUCE_TENSOR_ADD = CUDNN_REDUCE_TENSOR_ADD;
  cudnnReduceTensorOp_t REDUCE_TENSOR_MUL = CUDNN_REDUCE_TENSOR_MUL;
  cudnnReduceTensorOp_t REDUCE_TENSOR_MIN = CUDNN_REDUCE_TENSOR_MIN;
  cudnnReduceTensorOp_t REDUCE_TENSOR_MAX = CUDNN_REDUCE_TENSOR_MAX;
  cudnnReduceTensorOp_t REDUCE_TENSOR_AMAX = CUDNN_REDUCE_TENSOR_AMAX;
  cudnnReduceTensorOp_t REDUCE_TENSOR_AVG = CUDNN_REDUCE_TENSOR_AVG;
  cudnnReduceTensorOp_t REDUCE_TENSOR_NORM1 = CUDNN_REDUCE_TENSOR_NORM1;
  cudnnReduceTensorOp_t REDUCE_TENSOR_NORM2 = CUDNN_REDUCE_TENSOR_NORM2;

  return 0;
}
