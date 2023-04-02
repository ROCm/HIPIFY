// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --miopen %clang_args -D__CUDA_API_VERSION_INTERNAL

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
  // CHECK-NEXT: miopenTensorDescriptor_t filterDescriptor;
  cudnnTensorDescriptor_t tensorDescriptor;
  cudnnFilterDescriptor_t filterDescriptor;

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

  // CHECK: miopenConvFwdAlgorithm_t convolutionFwdAlgo;
  // CHECK-NEXT: miopenConvFwdAlgorithm_t CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = miopenConvolutionFwdAlgoImplicitGEMM;
  // CHECK-NEXT: miopenConvFwdAlgorithm_t CONVOLUTION_FWD_ALGO_GEMM = miopenConvolutionFwdAlgoGEMM;
  // CHECK-NEXT: miopenConvFwdAlgorithm_t CONVOLUTION_FWD_ALGO_DIRECT = miopenConvolutionFwdAlgoDirect;
  // CHECK-NEXT: miopenConvFwdAlgorithm_t CONVOLUTION_FWD_ALGO_FFT = miopenConvolutionFwdAlgoFFT;
  // CHECK-NEXT: miopenConvFwdAlgorithm_t CONVOLUTION_FWD_ALGO_WINOGRAD = miopenConvolutionFwdAlgoWinograd;
  cudnnConvolutionFwdAlgo_t convolutionFwdAlgo;
  cudnnConvolutionFwdAlgo_t CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  cudnnConvolutionFwdAlgo_t CONVOLUTION_FWD_ALGO_GEMM = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  cudnnConvolutionFwdAlgo_t CONVOLUTION_FWD_ALGO_DIRECT = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
  cudnnConvolutionFwdAlgo_t CONVOLUTION_FWD_ALGO_FFT = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
  cudnnConvolutionFwdAlgo_t CONVOLUTION_FWD_ALGO_WINOGRAD = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;

  // CHECK: miopenNanPropagation_t nanPropagation_t;
  // CHECK-NEXT: miopenNanPropagation_t NOT_PROPAGATE_NAN = MIOPEN_NOT_PROPAGATE_NAN;
  // CHECK-NEXT: miopenNanPropagation_t PROPAGATE_NAN = MIOPEN_PROPAGATE_NAN;
  cudnnNanPropagation_t nanPropagation_t;
  cudnnNanPropagation_t NOT_PROPAGATE_NAN = CUDNN_NOT_PROPAGATE_NAN;
  cudnnNanPropagation_t PROPAGATE_NAN = CUDNN_PROPAGATE_NAN;

  // CHECK: miopenReduceTensorIndices_t reduceTensorIndices;
  // CHECK-NEXT: miopenReduceTensorIndices_t REDUCE_TENSOR_NO_INDICES = MIOPEN_REDUCE_TENSOR_NO_INDICES;
  // CHECK-NEXT: miopenReduceTensorIndices_t REDUCE_TENSOR_FLATTENED_INDICES = MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES;
  cudnnReduceTensorIndices_t reduceTensorIndices;
  cudnnReduceTensorIndices_t REDUCE_TENSOR_NO_INDICES = CUDNN_REDUCE_TENSOR_NO_INDICES;
  cudnnReduceTensorIndices_t REDUCE_TENSOR_FLATTENED_INDICES = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;

  // CHECK: miopenIndicesType_t indicesType;
  // CHECK-NEXT: miopenIndicesType_t _32BIT_INDICES = MIOPEN_32BIT_INDICES;
  // CHECK-NEXT: miopenIndicesType_t _64BIT_INDICES = MIOPEN_64BIT_INDICES;
  // CHECK-NEXT: miopenIndicesType_t _16BIT_INDICES = MIOPEN_16BIT_INDICES;
  // CHECK-NEXT: miopenIndicesType_t _8BIT_INDICES = MIOPEN_8BIT_INDICES;
  cudnnIndicesType_t indicesType;
  cudnnIndicesType_t _32BIT_INDICES = CUDNN_32BIT_INDICES;
  cudnnIndicesType_t _64BIT_INDICES = CUDNN_64BIT_INDICES;
  cudnnIndicesType_t _16BIT_INDICES = CUDNN_16BIT_INDICES;
  cudnnIndicesType_t _8BIT_INDICES = CUDNN_8BIT_INDICES;

  // CHECK: miopenConvBwdDataAlgorithm_t ConvolutionBwdDataAlgo_t;
  // CHECK-NEXT: miopenConvBwdDataAlgorithm_t CONVOLUTION_BWD_DATA_ALGO_0 = miopenConvolutionBwdDataAlgoGEMM;
  // CHECK-NEXT: miopenConvBwdDataAlgorithm_t CONVOLUTION_BWD_DATA_ALGO_1 = miopenConvolutionBwdDataAlgoDirect;
  // CHECK-NEXT: miopenConvBwdDataAlgorithm_t CONVOLUTION_BWD_DATA_ALGO_FFT = miopenConvolutionBwdDataAlgoFFT;
  // CHECK-NEXT: miopenConvBwdDataAlgorithm_t CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = miopenConvolutionBwdDataAlgoWinograd;
  cudnnConvolutionBwdDataAlgo_t ConvolutionBwdDataAlgo_t;
  cudnnConvolutionBwdDataAlgo_t CONVOLUTION_BWD_DATA_ALGO_0 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  cudnnConvolutionBwdDataAlgo_t CONVOLUTION_BWD_DATA_ALGO_1 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  cudnnConvolutionBwdDataAlgo_t CONVOLUTION_BWD_DATA_ALGO_FFT = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
  cudnnConvolutionBwdDataAlgo_t CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;

  // CHECK: miopenConvAlgoPerf_t ConvolutionFwdAlgoPerf_t;
  // CHECK-NEXT: miopenConvAlgoPerf_t ConvolutionFwdAlgoPerfStruct;
  cudnnConvolutionFwdAlgoPerf_t ConvolutionFwdAlgoPerf_t;
  cudnnConvolutionFwdAlgoPerfStruct ConvolutionFwdAlgoPerfStruct;

  // CHECK: miopenConvAlgoPerf_t ConvolutionBwdDataAlgoPerf_t;
  // CHECK-NEXT: miopenConvAlgoPerf_t ConvolutionBwdDataAlgoPerfStruct;
  cudnnConvolutionBwdDataAlgoPerf_t ConvolutionBwdDataAlgoPerf_t;
  cudnnConvolutionBwdDataAlgoPerfStruct ConvolutionBwdDataAlgoPerfStruct;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateTensorDescriptor(miopenTensorDescriptor_t* tensorDesc);
  // CHECK: status = miopenCreateTensorDescriptor(&tensorDescriptor);
  status = cudnnCreateTensorDescriptor(&tensorDescriptor);

  // TODO: cudnnSetTensor4dDescriptor -> miopenSet4dTensorDescriptor: different signatures
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w);

  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  int nStride = 0;
  int cStride = 0;
  int hStride = 0;
  int wStride = 0;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSet4dTensorDescriptorEx(miopenTensorDescriptor_t tensorDesc, miopenDataType_t dataType, int n, int c, int h, int w, int nStride, int cStride, int hStride, int wStride);
  // CHECK: status = miopenSet4dTensorDescriptorEx(tensorDescriptor, dataType, n, c, h, w, nStride, cStride, hStride, wStride);
  status = cudnnSetTensor4dDescriptorEx(tensorDescriptor, dataType, n, c, h, w, nStride, cStride, hStride, wStride);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(const cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t* dataType, int* n, int* c, int* h, int* w, int* nStride, int* cStride, int* hStride, int* wStride);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGet4dTensorDescriptor(miopenTensorDescriptor_t tensorDesc, miopenDataType_t* dataType, int* n, int* c, int* h, int* w, int* nStride, int* cStride, int* hStride, int* wStride);
  // CHECK: status = miopenGet4dTensorDescriptor(tensorDescriptor, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);
  status = cudnnGetTensor4dDescriptor(tensorDescriptor, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyTensorDescriptor(miopenTensorDescriptor_t tensorDesc);
  // CHECK: status = miopenDestroyTensorDescriptor(tensorDescriptor);
  status = cudnnDestroyTensorDescriptor(tensorDescriptor);


  // CHECK: miopenTensorDescriptor_t aD;
  // CHECK-NEXT: miopenTensorDescriptor_t bD;
  // CHECK-NEXT: miopenTensorDescriptor_t cD;
  // CHECK-NEXT: miopenTensorDescriptor_t xD;
  // CHECK-NEXT: miopenTensorDescriptor_t yD;
  // CHECK-NEXT: miopenTensorDescriptor_t wD;
  // CHECK-NEXT: miopenTensorDescriptor_t inputD;
  // CHECK-NEXT: miopenTensorDescriptor_t dbD;
  cudnnTensorDescriptor_t aD;
  cudnnTensorDescriptor_t bD;
  cudnnTensorDescriptor_t cD;
  cudnnTensorDescriptor_t xD;
  cudnnTensorDescriptor_t yD;
  cudnnTensorDescriptor_t wD;
  cudnnTensorDescriptor_t inputD;
  cudnnTensorDescriptor_t dbD;
  void* A = nullptr;
  void* B = nullptr;
  void* C = nullptr;
  void* alpha = nullptr;
  void* alpha1 = nullptr;
  void* alpha2 = nullptr;
  void* beta = nullptr;
  void* x = nullptr;
  void* dx = nullptr;
  void* y = nullptr;
  void* dy = nullptr;
  void* W = nullptr;
  void* db = nullptr;
  int groupCount = 0;
  int requestedAlgoCount = 0;
  int returnedAlgoCount = 0;
  void* workSpace = nullptr;
  size_t workSpaceSizeInBytes = 0;

  // TODO: cudnnOpTensor -> miopenOpTensor: different signatures: cudnnOpTensorDescriptor_t != miopenTensorOp_t
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnOpTensor(cudnnHandle_t handle, const cudnnOpTensorDescriptor_t opTensorDesc, const void* alpha1, const cudnnTensorDescriptor_t aDesc, const void* A, const void* alpha2, const cudnnTensorDescriptor_t bDesc, const void* B, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenOpTensor(miopenHandle_t handle, miopenTensorOp_t tensorOp, const void* alpha1, const miopenTensorDescriptor_t aDesc, const void* A, const void* alpha2, const miopenTensorDescriptor_t bDesc, const void* B, const void* beta, const miopenTensorDescriptor_t cDesc, void* C);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void* y, const void* valuePtr);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetTensor(miopenHandle_t handle, const miopenTensorDescriptor_t yDesc, void* y, const void* alpha);
  // CHECK: status = miopenSetTensor(handle, tensorDescriptor, y, alpha);
  status = cudnnSetTensor(handle, tensorDescriptor, y, alpha);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t handle, const cudnnTensorDescriptor_t yDesc, void* y, const void* alpha);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenScaleTensor(miopenHandle_t handle, const miopenTensorDescriptor_t yDesc, void* y, const void* alpha);
  // CHECK: status = miopenScaleTensor(handle, tensorDescriptor, y, alpha);
  status = cudnnScaleTensor(handle, tensorDescriptor, y, alpha);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnTransformTensor(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenTransformTensor(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x, const void* beta, const miopenTensorDescriptor_t yDesc, void* y);
  // CHECK: status = miopenTransformTensor(handle, alpha, xD, x, beta, yD, y);
  status = cudnnTransformTensor(handle, alpha, xD, x, beta, yD, y);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateConvolutionDescriptor(miopenConvolutionDescriptor_t* convDesc);
  // CHECK: status = miopenCreateConvolutionDescriptor(&convolutionDescriptor);
  status = cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetConvolutionGroupCount(miopenConvolutionDescriptor_t convDesc, int groupCount);
  // CHECK: status = miopenSetConvolutionGroupCount(convolutionDescriptor, groupCount);
  status = cudnnSetConvolutionGroupCount(convolutionDescriptor, groupCount);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t inputTensorDesc, const cudnnFilterDescriptor_t filterDesc, int* n, int* c, int* h, int* w);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetConvolutionForwardOutputDim(miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t inputTensorDesc, const miopenTensorDescriptor_t filterDesc, int* n, int* c, int* h, int* w);
  // CHECK: status = miopenGetConvolutionForwardOutputDim(convolutionDescriptor, inputD, filterDescriptor, &n, &c, &h, &w);
  status = cudnnGetConvolution2dForwardOutputDim(convolutionDescriptor, inputD, filterDescriptor, &n, &c, &h, &w);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyConvolutionDescriptor(miopenConvolutionDescriptor_t convDesc);
  // CHECK: status = miopenDestroyConvolutionDescriptor(convolutionDescriptor);
  status = cudnnDestroyConvolutionDescriptor(convolutionDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, void* y, const int requestedAlgoCount, int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults, void* workSpace, size_t workSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenFindConvolutionForwardAlgorithm(miopenHandle_t handle, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t yDesc, void* y, const int requestAlgoCount, int* returnedAlgoCount, miopenConvAlgoPerf_t* perfResults, void* workSpace, size_t workSpaceSize, bool exhaustiveSearch);
  // CHECK: status = miopenFindConvolutionForwardAlgorithm(handle, xD, x, filterDescriptor, W, convolutionDescriptor, yD, y, requestedAlgoCount, &returnedAlgoCount, &ConvolutionFwdAlgoPerf_t, workSpace, workSpaceSizeInBytes, true);
  status = cudnnFindConvolutionForwardAlgorithmEx(handle, xD, x, filterDescriptor, W, convolutionDescriptor, yD, y, requestedAlgoCount, &returnedAlgoCount, &ConvolutionFwdAlgoPerf_t, workSpace, workSpaceSizeInBytes);

  // TODO: swap 2 and 3 arguments (const miopenTensorDescriptor_t wDesc and const miopenTensorDescriptor_t xDesc)
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle, const miopenTensorDescriptor_t wDesc, const miopenTensorDescriptor_t xDesc, const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);
  // CHECK: status = miopenConvolutionForwardGetWorkSpaceSize(handle, xD, filterDescriptor, convolutionDescriptor, yD,  &workSpaceSizeInBytes);
  status = cudnnGetConvolutionForwardWorkspaceSize(handle, xD, filterDescriptor, convolutionDescriptor, yD, convolutionFwdAlgo, &workSpaceSizeInBytes);

  // TODO: swap correctly last 5 arguments
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionForward(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, miopenConvFwdAlgorithm_t algo, const void* beta, const miopenTensorDescriptor_t yDesc, void* y, void* workSpace, size_t workSpaceSize);
  // CHECK: status = miopenConvolutionForward(handle, alpha, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, workSpace, workSpaceSizeInBytes, beta, yD, y);
  status = cudnnConvolutionForward(handle, alpha, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, workSpace, workSpaceSizeInBytes, beta, yD, y);

  // TODO: swap 2 and 3 arguments
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc, const miopenTensorDescriptor_t wDesc, const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t dxDesc, size_t* workSpaceSize);
  // CHECK: status = miopenConvolutionBackwardDataGetWorkSpaceSize(handle, filterDescriptor, yD, convolutionDescriptor, xD,  &workSpaceSizeInBytes);
  status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filterDescriptor, yD, convolutionDescriptor, xD, ConvolutionBwdDataAlgo_t, &workSpaceSizeInBytes);

  // TODO: swap correctly all args, starting from 3rd
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
  // MIOPEN MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardData(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t dyDesc, const void* dy, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, miopenConvBwdDataAlgorithm_t algo, const void* beta, const miopenTensorDescriptor_t dxDesc, void* dx, void* workSpace, size_t workSpaceSize);
  // CHECK: status = miopenConvolutionBackwardData(handle, alpha, filterDescriptor, W, yD, dy, convolutionDescriptor, ConvolutionBwdDataAlgo_t, workSpace, workSpaceSizeInBytes, beta, xD, dx);
  status = cudnnConvolutionBackwardData(handle, alpha, filterDescriptor, W, yD, dy, convolutionDescriptor, ConvolutionBwdDataAlgo_t, workSpace, workSpaceSizeInBytes, beta, xD, dx);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dbDesc, void* db);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta, const miopenTensorDescriptor_t dbDesc, void* db);
  // CHECK: status = miopenConvolutionBackwardBias(handle, alpha, yD, dy, beta, dbD, db);
  status = cudnnConvolutionBackwardBias(handle, alpha, yD, dy, beta, dbD, db);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc);
  // CHECK: status = miopenCreatePoolingDescriptor(&poolingDescriptor);
  status = cudnnCreatePoolingDescriptor(&poolingDescriptor);

  return 0;
}
