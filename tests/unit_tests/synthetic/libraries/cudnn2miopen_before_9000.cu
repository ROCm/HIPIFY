// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --miopen %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "miopen/miopen.h"
#include "cudnn.h"

int main() {
  printf("15.before.9000. cuDNN API to MIOpen API synthetic test\n");

  // CHECK: miopenStatus_t status;
  cudnnStatus_t status;

  // CHECK: miopenHandle_t handle;
  cudnnHandle_t handle;

  // CHECK: miopenRNNDescriptor_t RNNDescriptor;
  cudnnRNNDescriptor_t RNNDescriptor;

  // CHECK: miopenDropoutDescriptor_t DropoutDescriptor;
  cudnnDropoutDescriptor_t DropoutDescriptor;

  // CHECK: miopenRNNInputMode_t RNNInputMode;
  cudnnRNNInputMode_t RNNInputMode;

  // CHECK: miopenRNNDirectionMode_t DirectionMode;
  cudnnDirectionMode_t DirectionMode;

  // CHECK: miopenRNNMode_t RNNMode;
  cudnnRNNMode_t RNNMode;

  // CHECK: miopenRNNAlgo_t RNNAlgo;
  cudnnRNNAlgo_t RNNAlgo;

  // CHECK: miopenDataType_t dataType;
  cudnnDataType_t dataType;

  // CHECK: miopenTensorDescriptor_t xD;
  // CHECK-NEXT: miopenTensorDescriptor_t hxD;
  // CHECK-NEXT: miopenTensorDescriptor_t yD;
  cudnnTensorDescriptor_t xD;
  cudnnTensorDescriptor_t hxD;
  cudnnTensorDescriptor_t yD;

  // CHECK: miopenTensorDescriptor_t filterDescriptor;
  cudnnFilterDescriptor_t filterDescriptor;

  int hiddenSize = 0;
  int layer = 0;
  int seqLength = 0;

  void* x = nullptr;
  void* y = nullptr;
  void* hx = nullptr;
  void* dw = nullptr;
  void* workSpace = nullptr;
  void* reserveSpace = nullptr;

  size_t workSpaceSizeInBytes = 0;
  size_t reserveSpaceNumBytes = 0;

#if CUDNN_MAJOR < 9
  // TODO [#837]: Insert miopenRNNBiasMode_t biasMode in the hipified miopenSetRNNDescriptor_V2 after miopenRNNMode_t rnnMode: will need variable declaration
  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, const int hiddenSize, const int numLayers, cudnnDropoutDescriptor_t dropoutDesc, cudnnRNNInputMode_t inputMode, cudnnDirectionMode_t direction, cudnnRNNMode_t cellMode, cudnnRNNAlgo_t algo, cudnnDataType_t mathPrec);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc, const int hsize, const int nlayers, miopenDropoutDescriptor_t dropoutDesc, miopenRNNInputMode_t inMode, miopenRNNDirectionMode_t direction, miopenRNNMode_t rnnMode, miopenRNNBiasMode_t biasMode, miopenRNNAlgo_t algo, miopenDataType_t dataType);
  // CHECK: status = miopenSetRNNDescriptor_V2(RNNDescriptor, hiddenSize, layer, DropoutDescriptor, RNNInputMode, DirectionMode, RNNMode, RNNAlgo, dataType);
  status = cudnnSetRNNDescriptor_v6(handle, RNNDescriptor, hiddenSize, layer, DropoutDescriptor, RNNInputMode, DirectionMode, RNNMode, RNNAlgo, dataType);

  // TODO [#837]: Insert miopenRNNBiasMode_t* biasMode in the hipified miopenGetRNNDescriptor_V2 after miopenRNNMode_t* rnnMode: will need variable declaration
  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v6(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, int* hiddenSize, int* numLayers, cudnnDropoutDescriptor_t* dropoutDesc, cudnnRNNInputMode_t* inputMode, cudnnDirectionMode_t* direction, cudnnRNNMode_t* cellMode, cudnnRNNAlgo_t* algo, cudnnDataType_t* mathPrec);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetRNNDescriptor_V2(miopenRNNDescriptor_t rnnDesc, int* hiddenSize, int* layer, miopenDropoutDescriptor_t* dropoutDesc, miopenRNNInputMode_t* inputMode, miopenRNNDirectionMode_t* dirMode, miopenRNNMode_t* rnnMode, miopenRNNBiasMode_t* biasMode, miopenRNNAlgo_t* algoMode, miopenDataType_t* dataType);
  // CHECK: status = miopenGetRNNDescriptor_V2(RNNDescriptor, &hiddenSize, &layer, &DropoutDescriptor, &RNNInputMode, &DirectionMode, &RNNMode, &RNNAlgo, &dataType);
  status = cudnnGetRNNDescriptor_v6(handle, RNNDescriptor, &hiddenSize, &layer, &DropoutDescriptor, &RNNInputMode, &DirectionMode, &RNNMode, &RNNAlgo, &dataType);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t* yDesc, const void* y, const void* workSpace, size_t workSpaceSizeInBytes, const cudnnFilterDescriptor_t dwDesc, void* dw, const void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardWeights(miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* xDesc, const void* x, const miopenTensorDescriptor_t hxDesc, const void* hx, const miopenTensorDescriptor_t* yDesc, const void* y, const miopenTensorDescriptor_t dwDesc, void* dw, void* workSpace, size_t workSpaceNumBytes, const void* reserveSpace, size_t reserveSpaceNumBytes);
  // CHECK: status = miopenRNNBackwardWeights(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, &yD, y, filterDescriptor, dw, workSpace, workSpaceSizeInBytes, &reserveSpace, reserveSpaceNumBytes);
  status = cudnnRNNBackwardWeights(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, &yD, y, workSpace, workSpaceSizeInBytes, filterDescriptor, dw, &reserveSpace, reserveSpaceNumBytes);
#endif

  return 0;
}
