// RUN: %run_test hipify "%s" "%t" %hipify_args 5 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --miopen --amap %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "miopen/miopen.h"
#include "cudnn.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

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

  // CHECK: miopenHandle *context = nullptr;
  // CHECK-NEXT: miopenHandle_t handle;
  cudnnContext *context = nullptr;
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

  // CHECK: hipStream_t streamId;
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
  cudnnDataType_t dataType;
  cudnnDataType_t DATA_FLOAT = CUDNN_DATA_FLOAT;
  cudnnDataType_t DATA_DOUBLE = CUDNN_DATA_DOUBLE;
  cudnnDataType_t DATA_HALF = CUDNN_DATA_HALF;
  cudnnDataType_t DATA_INT8 = CUDNN_DATA_INT8;
  cudnnDataType_t DATA_INT32 = CUDNN_DATA_INT32;
  cudnnDataType_t DATA_INT8x4 = CUDNN_DATA_INT8x4;

  // CHECK: miopenRNNMode_t RNNMode;
  // CHECK-NEXT: miopenRNNMode_t RNN_RELU = miopenRNNRELU;
  // CHECK-NEXT: miopenRNNMode_t RNN_TANH = miopenRNNTANH;
  // CHECK-NEXT: miopenRNNMode_t LSTM = miopenLSTM;
  // CHECK-NEXT: miopenRNNMode_t GRU = miopenGRU;
  cudnnRNNMode_t RNNMode;
  cudnnRNNMode_t RNN_RELU = CUDNN_RNN_RELU;
  cudnnRNNMode_t RNN_TANH = CUDNN_RNN_TANH;
  cudnnRNNMode_t LSTM = CUDNN_LSTM;
  cudnnRNNMode_t GRU = CUDNN_GRU;

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
  // CHECK-NEXT: miopenConvolutionMode_t CONVOLUTION = miopenConvolution;
  // CHECK-NEXT: miopenConvolutionMode_t CROSS_CORRELATION = miopenConvolution;
  cudnnConvolutionMode_t convolutionMode;
  cudnnConvolutionMode_t CONVOLUTION = CUDNN_CONVOLUTION;
  cudnnConvolutionMode_t CROSS_CORRELATION = CUDNN_CROSS_CORRELATION;

  // CHECK: miopenPoolingMode_t poolingMode;
  // CHECK-NEXT: miopenPoolingMode_t POOLING_MAX = miopenPoolingMax;
  cudnnPoolingMode_t poolingMode;
  cudnnPoolingMode_t POOLING_MAX = CUDNN_POOLING_MAX;

  // CHECK: miopenRNNInputMode_t RNNInputMode;
  // CHECK-NEXT: miopenRNNInputMode_t LINEAR_INPUT = miopenRNNlinear;
  // CHECK-NEXT: miopenRNNInputMode_t SKIP_INPUT = miopenRNNskip;
  cudnnRNNInputMode_t RNNInputMode;
  cudnnRNNInputMode_t LINEAR_INPUT = CUDNN_LINEAR_INPUT;
  cudnnRNNInputMode_t SKIP_INPUT = CUDNN_SKIP_INPUT;

  // CHECK: miopenRNNAlgo_t RNNAlgo;
  // CHECK-NEXT: miopenRNNAlgo_t RNN_ALGO_STANDARD = miopenRNNdefault;
  cudnnRNNAlgo_t RNNAlgo;
  cudnnRNNAlgo_t RNN_ALGO_STANDARD = CUDNN_RNN_ALGO_STANDARD;

  // CHECK: miopenRNNBiasMode_t RNNBiasMode;
  // CHECK-NEXT: miopenRNNBiasMode_t RNN_NO_BIAS = miopenRNNNoBias;
  // CHECK-NEXT: miopenRNNBiasMode_t RNN_SINGLE_INP_BIAS = miopenRNNwithBias;
  // CHECK-NEXT: miopenRNNBiasMode_t RNN_DOUBLE_BIAS = miopenRNNwithBias;
  // CHECK-NEXT: miopenRNNBiasMode_t RNN_SINGLE_REC_BIAS = miopenRNNwithBias;
  cudnnRNNBiasMode_t RNNBiasMode;
  cudnnRNNBiasMode_t RNN_NO_BIAS = CUDNN_RNN_NO_BIAS;
  cudnnRNNBiasMode_t RNN_SINGLE_INP_BIAS = CUDNN_RNN_SINGLE_INP_BIAS;
  cudnnRNNBiasMode_t RNN_DOUBLE_BIAS = CUDNN_RNN_DOUBLE_BIAS;
  cudnnRNNBiasMode_t RNN_SINGLE_REC_BIAS = CUDNN_RNN_SINGLE_REC_BIAS;

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
  // CHECK-NEXT: miopenActivationMode_t ACTIVATION_SIGMOID = miopenActivationLOGISTIC;
  cudnnActivationMode_t activationMode;
  cudnnActivationMode_t ACTIVATION_RELU = CUDNN_ACTIVATION_RELU;
  cudnnActivationMode_t ACTIVATION_TANH = CUDNN_ACTIVATION_TANH;
  cudnnActivationMode_t ACTIVATION_SIGMOID = CUDNN_ACTIVATION_SIGMOID;

  // CHECK: miopenSoftmaxAlgorithm_t softmaxAlgorithm;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_FAST = MIOPEN_SOFTMAX_FAST;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_ACCURATE = MIOPEN_SOFTMAX_ACCURATE;
  // CHECK-NEXT: miopenSoftmaxAlgorithm_t SOFTMAX_LOG = MIOPEN_SOFTMAX_LOG;
  cudnnSoftmaxAlgorithm_t softmaxAlgorithm;
  cudnnSoftmaxAlgorithm_t SOFTMAX_FAST = CUDNN_SOFTMAX_FAST;
  cudnnSoftmaxAlgorithm_t SOFTMAX_ACCURATE = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxAlgorithm_t SOFTMAX_LOG = CUDNN_SOFTMAX_LOG;

  // CHECK: miopenSoftmaxMode_t softmaxMode;
  // CHECK-NEXT: miopenSoftmaxMode_t SOFTMAX_MODE_INSTANCE = MIOPEN_SOFTMAX_MODE_INSTANCE;
  // CHECK-NEXT: miopenSoftmaxMode_t SOFTMAX_MODE_CHANNEL = MIOPEN_SOFTMAX_MODE_CHANNEL;
  cudnnSoftmaxMode_t softmaxMode;
  cudnnSoftmaxMode_t SOFTMAX_MODE_INSTANCE = CUDNN_SOFTMAX_MODE_INSTANCE;
  cudnnSoftmaxMode_t SOFTMAX_MODE_CHANNEL = CUDNN_SOFTMAX_MODE_CHANNEL;

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

  // CHECK: miopenRNNDirectionMode_t DirectionMode;
  // CHECK-NEXT: miopenRNNDirectionMode_t UNIDIRECTIONAL = miopenRNNunidirection;
  // CHECK-NEXT: miopenRNNDirectionMode_t BIDIRECTIONAL = miopenRNNbidirection;
  cudnnDirectionMode_t DirectionMode;
  cudnnDirectionMode_t UNIDIRECTIONAL = CUDNN_UNIDIRECTIONAL;
  cudnnDirectionMode_t BIDIRECTIONAL = CUDNN_BIDIRECTIONAL;

  // CHECK: miopenConvAlgoPerf_t ConvolutionFwdAlgoPerf_t;
  // CHECK-NEXT: miopenConvAlgoPerf_t ConvolutionFwdAlgoPerfStruct;
  cudnnConvolutionFwdAlgoPerf_t ConvolutionFwdAlgoPerf_t;
  cudnnConvolutionFwdAlgoPerfStruct ConvolutionFwdAlgoPerfStruct;

  // CHECK: miopenConvAlgoPerf_t ConvolutionBwdDataAlgoPerf_t;
  // CHECK-NEXT: miopenConvAlgoPerf_t ConvolutionBwdDataAlgoPerfStruct;
  cudnnConvolutionBwdDataAlgoPerf_t ConvolutionBwdDataAlgoPerf_t;
  cudnnConvolutionBwdDataAlgoPerfStruct ConvolutionBwdDataAlgoPerfStruct;

  // CHECK: miopenCTCLossAlgo_t CTCLossAlgo;
  // CHECK-NEXT: miopenCTCLossAlgo_t CTC_LOSS_ALGO_DETERMINISTIC = MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC;
  cudnnCTCLossAlgo_t CTCLossAlgo;
  cudnnCTCLossAlgo_t CTC_LOSS_ALGO_DETERMINISTIC = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;

  // CHECK: miopenTensorLayout_t tensorFormat;
  // CHECK-NEXT: miopenTensorLayout_t TENSOR_NCHW = miopenTensorNCHW;
  // CHECK-NEXT: miopenTensorLayout_t TENSOR_NHWC = miopenTensorNHWC;
  cudnnTensorFormat_t tensorFormat;
  cudnnTensorFormat_t TENSOR_NCHW = CUDNN_TENSOR_NCHW;
  cudnnTensorFormat_t TENSOR_NHWC = CUDNN_TENSOR_NHWC;

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
  int64_t elementCount = 0;
  int64_t requestedElementCount = 0;
  void *arrayOfElements = nullptr;

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
  // CHECK-NEXT: miopenTensorDescriptor_t hxD;
  // CHECK-NEXT: miopenTensorDescriptor_t dhxD;
  // CHECK-NEXT: miopenTensorDescriptor_t cxD;
  // CHECK-NEXT: miopenTensorDescriptor_t dcxD;
  // CHECK-NEXT: miopenTensorDescriptor_t yD;
  // CHECK-NEXT: miopenTensorDescriptor_t dyD;
  // CHECK-NEXT: miopenTensorDescriptor_t hyD;
  // CHECK-NEXT: miopenTensorDescriptor_t dhyD;
  // CHECK-NEXT: miopenTensorDescriptor_t cyD;
  // CHECK-NEXT: miopenTensorDescriptor_t dcyD;
  // CHECK-NEXT: miopenTensorDescriptor_t wD;
  // CHECK-NEXT: miopenTensorDescriptor_t zD;
  // CHECK-NEXT: miopenTensorDescriptor_t inputD;
  // CHECK-NEXT: miopenTensorDescriptor_t dbD;
  // CHECK-NEXT: miopenTensorDescriptor_t dxD;
  // CHECK-NEXT: miopenTensorDescriptor_t biasD;
  // CHECK-NEXT: miopenTensorDescriptor_t probsD;
  // CHECK-NEXT: miopenTensorDescriptor_t gradientsD;
  cudnnTensorDescriptor_t aD;
  cudnnTensorDescriptor_t bD;
  cudnnTensorDescriptor_t cD;
  cudnnTensorDescriptor_t xD;
  cudnnTensorDescriptor_t hxD;
  cudnnTensorDescriptor_t dhxD;
  cudnnTensorDescriptor_t cxD;
  cudnnTensorDescriptor_t dcxD;
  cudnnTensorDescriptor_t yD;
  cudnnTensorDescriptor_t dyD;
  cudnnTensorDescriptor_t hyD;
  cudnnTensorDescriptor_t dhyD;
  cudnnTensorDescriptor_t cyD;
  cudnnTensorDescriptor_t dcyD;
  cudnnTensorDescriptor_t wD;
  cudnnTensorDescriptor_t zD;
  cudnnTensorDescriptor_t inputD;
  cudnnTensorDescriptor_t dbD;
  cudnnTensorDescriptor_t dxD;
  cudnnTensorDescriptor_t biasD;
  cudnnTensorDescriptor_t probsD;
  cudnnTensorDescriptor_t gradientsD;
  void* A = nullptr;
  void* B = nullptr;
  void* C = nullptr;
  void* alpha = nullptr;
  void* alpha1 = nullptr;
  void* alpha2 = nullptr;
  void* beta = nullptr;
  void* x = nullptr;
  void* dx = nullptr;
  void* hx = nullptr;
  void* dhx = nullptr;
  void* cx = nullptr;
  void* dcx = nullptr;
  void* y = nullptr;
  void* dy = nullptr;
  void* hy = nullptr;
  void* cy = nullptr;
  void* dcy = nullptr;
  void* z = nullptr;
  void* dhy = nullptr;
  void* W = nullptr;
  void* dw = nullptr;
  void* db = nullptr;
  void* bias = nullptr;
  void* workSpace = nullptr;
  void* indices = nullptr;
  void* reserveSpace = nullptr;
  void* probs = nullptr;
  void* gradients = nullptr;
  void* losses = nullptr;
  int groupCount = 0;
  int requestedAlgoCount = 0;
  int returnedAlgoCount = 0;
  size_t workSpaceSizeInBytes = 0;
  size_t reserveSpaceNumBytes = 0;
  size_t indicesSizeInBytes = 0;

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

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t algo, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionForwardGetWorkSpaceSize(miopenHandle_t handle, const miopenTensorDescriptor_t wDesc, const miopenTensorDescriptor_t xDesc, const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t yDesc, size_t* workSpaceSize);
  // CHECK: status = miopenConvolutionForwardGetWorkSpaceSize(handle, filterDescriptor, xD, convolutionDescriptor, yD, &workSpaceSizeInBytes);
  status = cudnnGetConvolutionForwardWorkspaceSize(handle, xD, filterDescriptor, convolutionDescriptor, yD, convolutionFwdAlgo, &workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionForward(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, miopenConvFwdAlgorithm_t algo, const void* beta, const miopenTensorDescriptor_t yDesc, void* y, void* workSpace, size_t workSpaceSize);
  // CHECK: status = miopenConvolutionForward(handle, alpha, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, beta, yD, y, workSpace, workSpaceSizeInBytes);
  status = cudnnConvolutionForward(handle, alpha, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, workSpace, workSpaceSizeInBytes, beta, yD, y);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle, const cudnnFilterDescriptor_t wDesc, const cudnnTensorDescriptor_t dyDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t algo, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardDataGetWorkSpaceSize(miopenHandle_t handle, const miopenTensorDescriptor_t dyDesc, const miopenTensorDescriptor_t wDesc, const miopenConvolutionDescriptor_t convDesc, const miopenTensorDescriptor_t dxDesc, size_t* workSpaceSize);
  // CHECK: status = miopenConvolutionBackwardDataGetWorkSpaceSize(handle, yD, filterDescriptor, convolutionDescriptor, xD, &workSpaceSizeInBytes);
  status = cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filterDescriptor, yD, convolutionDescriptor, xD, ConvolutionBwdDataAlgo_t, &workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle, const void* alpha, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
  // MIOPEN MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardData(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t dyDesc, const void* dy, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, miopenConvBwdDataAlgorithm_t algo, const void* beta, const miopenTensorDescriptor_t dxDesc, void* dx, void* workSpace, size_t workSpaceSize);
  // CHECK: status = miopenConvolutionBackwardData(handle, alpha, yD, dy, filterDescriptor, W, convolutionDescriptor, ConvolutionBwdDataAlgo_t, beta, xD, dx, workSpace, workSpaceSizeInBytes);
  status = cudnnConvolutionBackwardData(handle, alpha, filterDescriptor, W, yD, dy, convolutionDescriptor, ConvolutionBwdDataAlgo_t, workSpace, workSpaceSizeInBytes, beta, xD, dx);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void* alpha, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dbDesc, void* db);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionBackwardBias(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta, const miopenTensorDescriptor_t dbDesc, void* db);
  // CHECK: status = miopenConvolutionBackwardBias(handle, alpha, yD, dy, beta, dbD, db);
  status = cudnnConvolutionBackwardBias(handle, alpha, yD, dy, beta, dbD, db);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreatePoolingDescriptor(miopenPoolingDescriptor_t* poolDesc);
  // CHECK: status = miopenCreatePoolingDescriptor(&poolingDescriptor);
  status = cudnnCreatePoolingDescriptor(&poolingDescriptor);

  // CHECK: miopenNanPropagation_t maxpoolingNanOpt;
  cudnnNanPropagation_t maxpoolingNanOpt;
  int wH = 0;
  int wW = 0;
  int pad_h = 0;
  int pad_w = 0;
  int stride_h = 0;
  int stride_w = 0;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t mode, cudnnNanPropagation_t maxpoolingNanOpt, int windowHeight, int windowWidth, int verticalPadding, int horizontalPadding, int verticalStride, int horizontalStride);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSet2dPoolingDescriptor(miopenPoolingDescriptor_t poolDesc, miopenPoolingMode_t mode, int windowHeight, int windowWidth, int pad_h, int pad_w, int stride_h, int stride_w);
  // CHECK: status = miopenSet2dPoolingDescriptor(poolingDescriptor, poolingMode, wH, wW, pad_h, pad_w, stride_h, stride_w);
  status = cudnnSetPooling2dDescriptor(poolingDescriptor, poolingMode, maxpoolingNanOpt, wH, wW, pad_h, pad_w, stride_h, stride_w);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescriptor(const cudnnPoolingDescriptor_t poolingDesc, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* windowHeight, int* windowWidth, int* verticalPadding, int* horizontalPadding, int* verticalStride, int* horizontalStride);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGet2dPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc, miopenPoolingMode_t* mode, int* windowHeight, int* windowWidth, int* pad_h, int* pad_w, int* stride_h, int* stride_w);
  // CHECK: status = miopenGet2dPoolingDescriptor(poolingDescriptor, &poolingMode, &wH, &wW, &pad_h, &pad_w, &stride_h, &stride_w);
  status = cudnnGetPooling2dDescriptor(poolingDescriptor, &poolingMode, &maxpoolingNanOpt, &wH, &wW, &pad_h, &pad_w, &stride_h, &stride_w);

  int nbDims = 0;
  int nbDimsRequested = 0;
  int* windowDimA = nullptr;
  int* padA = nullptr;
  int* stridesA = nullptr;
  int* tensorDimArr = nullptr;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int* n, int* c, int* h, int* w);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetPoolingForwardOutputDim(const miopenPoolingDescriptor_t poolDesc, const miopenTensorDescriptor_t tensorDesc, int* n, int* c, int* h, int* w);
  // CHECK: status = miopenGetPoolingForwardOutputDim(poolingDescriptor, tensorDescriptor, &n, &c, &h, &w);
  status = cudnnGetPooling2dForwardOutputDim(poolingDescriptor, tensorDescriptor, &n, &c, &h, &w);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc, const cudnnTensorDescriptor_t inputTensorDesc, int nbDims, int outputTensorDimA[]);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetPoolingNdForwardOutputDim(const miopenPoolingDescriptor_t poolDesc, const miopenTensorDescriptor_t tensorDesc, int dims, int* tensorDimArr);
  // CHECK: status = miopenGetPoolingNdForwardOutputDim(poolingDescriptor, tensorDescriptor, nbDims, tensorDimArr);
  status = cudnnGetPoolingNdForwardOutputDim(poolingDescriptor, tensorDescriptor, nbDims, tensorDimArr);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc, const cudnnPoolingMode_t mode, const cudnnNanPropagation_t maxpoolingNanOpt, int nbDims, const int windowDimA[], const int paddingA[], const int strideA[]);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetNdPoolingDescriptor(miopenPoolingDescriptor_t poolDesc, const miopenPoolingMode_t mode, int nbDims, int* windowDimA, int* padA, int* stridesA);
  // CHECK: status = miopenSetNdPoolingDescriptor(poolingDescriptor, poolingMode, nbDims, windowDimA, padA, stridesA);
  status = cudnnSetPoolingNdDescriptor(poolingDescriptor, poolingMode, maxpoolingNanOpt, nbDims, windowDimA, padA, stridesA);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc, int nbDimsRequested, cudnnPoolingMode_t* mode, cudnnNanPropagation_t* maxpoolingNanOpt, int* nbDims, int windowDimA[], int paddingA[], int strideA[]);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetNdPoolingDescriptor(const miopenPoolingDescriptor_t poolDesc, int nbDimsRequested, miopenPoolingMode_t* mode, int* nbDims, int* windowDimA, int* padA, int* stridesA);
  // CHECK: status = miopenGetNdPoolingDescriptor(poolingDescriptor, nbDimsRequested, &poolingMode, &nbDims, windowDimA, padA, stridesA);
  status = cudnnGetPoolingNdDescriptor(poolingDescriptor, nbDimsRequested, &poolingMode, &maxpoolingNanOpt, &nbDims, windowDimA, padA, stridesA);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyPoolingDescriptor(miopenPoolingDescriptor_t poolDesc);
  // CHECK: status = miopenDestroyPoolingDescriptor(poolingDescriptor);
  status = cudnnDestroyPoolingDescriptor(poolingDescriptor);

  unsigned lrnN = 0;
  double lrnAlpha = 0.0f;
  double lrnBeta = 0.0f;
  double lrnK = 0.0f;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t* normDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateLRNDescriptor(miopenLRNDescriptor_t* lrnDesc);
  // CHECK: status = miopenCreateLRNDescriptor(&LRNDescriptor);
  status = cudnnCreateLRNDescriptor(&LRNDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc, miopenLRNMode_t mode, unsigned int lrnN, double lrnAlpha, double lrnBeta, double lrnK);
  // CHECK: status = miopenSetLRNDescriptor(LRNDescriptor, miopenLRNCrossChannel, lrnN, lrnAlpha, lrnBeta, lrnK);
  status = cudnnSetLRNDescriptor(LRNDescriptor, lrnN, lrnAlpha, lrnBeta, lrnK);

  // TODO: add a referrence to miopenLRNMode_t as a 2nd arg
  // TODO: [feature] Add a new type of transformation by declaring a var before the function call to add that var reference as an arg to the below function call
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc, unsigned* lrnN, double* lrnAlpha, double* lrnBeta, double* lrnK);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetLRNDescriptor(const miopenLRNDescriptor_t lrnDesc, miopenLRNMode_t* mode, unsigned int* lrnN, double* lrnAlpha, double* lrnBeta, double* lrnK);
  // CHECK: status = miopenGetLRNDescriptor(LRNDescriptor, &lrnN, &lrnAlpha, &lrnBeta, &lrnK);
  status = cudnnGetLRNDescriptor(LRNDescriptor, &lrnN, &lrnAlpha, &lrnBeta, &lrnK);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyLRNDescriptor(miopenLRNDescriptor_t lrnDesc);
  // CHECK: status = miopenDestroyLRNDescriptor(LRNDescriptor);
  status = cudnnDestroyLRNDescriptor(LRNDescriptor);

  // CHECK: miopenTensorDescriptor_t bnScaleBiasMeanVarDesc;
  // CHECK: miopenTensorDescriptor_t bnScaleBiasDiffDesc;
  cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
  cudnnTensorDescriptor_t bnScaleBiasDiffDesc;
  void *bnScale = nullptr;
  void *bnBias = nullptr;
  double expAvgFactor = 0.0f;
  void *resultRunningMean = nullptr;
  void *resultRunningVariance = nullptr;
  double epsilon = 0.0f;
  void *resultSaveMean = nullptr;
  void *resultSaveInvVariance = nullptr;
  void *estimatedMean = nullptr;
  void *estimatedVariance = nullptr;
  void *alphaDataDiff = nullptr;
  void *betaDataDiff = nullptr;
  void *alphaParamDiff = nullptr;
  void *betaParamDiff = nullptr;
  void *resultBnScaleDiff = nullptr;
  void *resultBnBiasDiff = nullptr;
  void *savedMean = nullptr;
  void *savedInvVariance = nullptr;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc, const cudnnTensorDescriptor_t xDesc, cudnnBatchNormMode_t mode);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc, const miopenTensorDescriptor_t xDesc, miopenBatchNormMode_t bn_mode);
  // CHECK: status = miopenDeriveBNTensorDescriptor(tensorDescriptor, xD, batchNormMode);
  status = cudnnDeriveBNTensorDescriptor(tensorDescriptor, xD, batchNormMode);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias, double exponentialAverageFactor, void* resultRunningMean, double epsilon, void* resultSaveMean, void* resultSaveInvVariance);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationForwardTraining(miopenHandle_t handle, miopenBatchNormMode_t bn_mode, void* alpha, void* beta, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t yDesc, void* y, const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc, void* bnScale, void* bnBias, double expAvgFactor, void* resultRunningMean, void* resultRunningVariance, double epsilon, void* resultSaveMean, void* resultSaveInvVariance);
  // CHECK: status = miopenBatchNormalizationForwardTraining(handle, batchNormMode, alpha, beta, xD, x, yD, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, expAvgFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);
  status = cudnnBatchNormalizationForwardTraining(handle, batchNormMode, alpha, beta, xD, x, yD, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, expAvgFactor, resultRunningMean, resultRunningVariance, epsilon, resultSaveMean, resultSaveInvVariance);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alpha, const void* beta, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t yDesc, void* y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias, const void* estimatedMean, const void* estimatedVariance, double epsilon);
  // MIOPEN: miopenBatchNormalizationForwardInference(miopenHandle_t handle, miopenBatchNormMode_t bn_mode, void* alpha, void* beta, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t yDesc, void* y, const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc, void* bnScale, void* bnBias, void* estimatedMean, void* estimatedVariance, double epsilon);
  // CHECK: status = miopenBatchNormalizationForwardInference(handle, batchNormMode, alpha, beta, xD, x, yD, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);
  status = cudnnBatchNormalizationForwardInference(handle, batchNormMode, alpha, beta, xD, x, yD, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void* alphaDataDiff, const void* betaDataDiff, const void* alphaParamDiff, const void* betaParamDiff, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t dxDesc, void* dx, const cudnnTensorDescriptor_t dBnScaleBiasDesc, const void* bnScale, void* dBnScaleResult, void* dBnBiasResult, double epsilon, const void* savedMean, const void* savedInvVariance);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBatchNormalizationBackward(miopenHandle_t handle, miopenBatchNormMode_t bn_mode, const void* alphaDataDiff, const void* betaDataDiff, const void* alphaParamDiff, const void* betaParamDiff, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t dyDesc, const void* dy, const miopenTensorDescriptor_t dxDesc, void* dx, const miopenTensorDescriptor_t bnScaleBiasDiffDesc, const void* bnScale, void* resultBnScaleDiff, void* resultBnBiasDiff, double epsilon, const void* savedMean, const void* savedInvVariance);
  // CHECK: status = miopenBatchNormalizationBackward(handle, batchNormMode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xD, x, yD, y, dxD, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance);
  status = cudnnBatchNormalizationBackward(handle, batchNormMode, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xD, x, yD, y, dxD, dx, bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVariance);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateActivationDescriptor(miopenActivationDescriptor_t* activDesc);
  // CHECK: status = miopenCreateActivationDescriptor(&activationDescriptor);
  status = cudnnCreateActivationDescriptor(&activationDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyActivationDescriptor(miopenActivationDescriptor_t activDesc);
  // CHECK: status = miopenDestroyActivationDescriptor(activationDescriptor);
  status = cudnnDestroyActivationDescriptor(activationDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenActivationForward(miopenHandle_t handle, const miopenActivationDescriptor_t activDesc, const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x, const void* beta, const miopenTensorDescriptor_t yDesc, void* y);
  // CHECK: status = miopenActivationForward(handle, activationDescriptor, alpha, xD, x, beta, yD, y);
  status = cudnnActivationForward(handle, activationDescriptor, alpha, xD, x, beta, yD, y);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenActivationBackward(miopenHandle_t handle, const miopenActivationDescriptor_t activDesc, const void* alpha, const miopenTensorDescriptor_t yDesc, const void* y, const miopenTensorDescriptor_t dyDesc, const void* dy, const miopenTensorDescriptor_t xDesc, const void* x, const void* beta, const miopenTensorDescriptor_t dxDesc, void* dx);
  // CHECK: status = miopenActivationBackward(handle, activationDescriptor, alpha, yD, y, dyD, dy, xD, x, beta, dxD, dx);
  status = cudnnActivationBackward(handle, activationDescriptor, alpha, yD, y, dyD, dy, xD, x, beta, dxD, dx);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void* alpha, const cudnnTensorDescriptor_t xDesc, const void* x, const void* beta, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSoftmaxForward_V2(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t xDesc, const void* x, const void* beta, const miopenTensorDescriptor_t yDesc, void* y, miopenSoftmaxAlgorithm_t algorithm, miopenSoftmaxMode_t mode);
  // CHECK: status = miopenSoftmaxForward_V2(handle, alpha, xD, x, beta, yD, y, softmaxAlgorithm, softmaxMode);
  status = cudnnSoftmaxForward(handle, softmaxAlgorithm, softmaxMode, alpha, xD, x, beta, yD, y);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode, const void* alpha, const cudnnTensorDescriptor_t yDesc, const void* y, const cudnnTensorDescriptor_t dyDesc, const void* dy, const void* beta, const cudnnTensorDescriptor_t dxDesc, void* dx);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSoftmaxBackward_V2(miopenHandle_t handle, const void* alpha, const miopenTensorDescriptor_t yDesc, const void* y, const miopenTensorDescriptor_t dyDesc, const void* dy, const void* beta, const miopenTensorDescriptor_t dxDesc, void* dx, miopenSoftmaxAlgorithm_t algorithm, miopenSoftmaxMode_t mode);
  // CHECK: status = miopenSoftmaxBackward_V2(handle, alpha, yD, y, dyD, dy, beta, dxD, dx, softmaxAlgorithm, softmaxMode);
  status = cudnnSoftmaxBackward(handle, softmaxAlgorithm, softmaxMode, alpha, yD, y, dyD, dy, beta, dxD, dx);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(cudnnHandle_t handle, const void* alpha1, const cudnnTensorDescriptor_t xDesc, const void* x, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes, const void* alpha2, const cudnnTensorDescriptor_t zDesc, const void* z, const cudnnTensorDescriptor_t biasDesc, const void* bias, const cudnnActivationDescriptor_t activationDesc, const cudnnTensorDescriptor_t yDesc, void* y);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenConvolutionBiasActivationForward(miopenHandle_t handle, const void* alpha1, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t wDesc, const void* w, const miopenConvolutionDescriptor_t convDesc, miopenConvFwdAlgorithm_t algo, void* workspace, size_t workspaceSizeInBytes, const void* alpha2, const miopenTensorDescriptor_t zDesc, const void* z, const miopenTensorDescriptor_t biasDesc, const void* bias, const miopenActivationDescriptor_t activationDesc, const miopenTensorDescriptor_t yDesc, void* y);
  // CHECK: status = miopenConvolutionBiasActivationForward(handle, alpha1, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, workSpace, workSpaceSizeInBytes, alpha2, zD, z, biasD, bias, activationDescriptor, yD, y);
  status = cudnnConvolutionBiasActivationForward(handle, alpha1, xD, x, filterDescriptor, W, convolutionDescriptor, convolutionFwdAlgo, workSpace, workSpaceSizeInBytes, alpha2, zD, z, biasD, bias, activationDescriptor, yD, y);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateRNNDescriptor(miopenRNNDescriptor_t* rnnDesc);
  // CHECK: status = miopenCreateRNNDescriptor(&RNNDescriptor);
  status = cudnnCreateRNNDescriptor(&RNNDescriptor);

  // NOTE: cudnnGetRNNDescriptor - removed after cuDNN 7.6.5
  // TODO: add cudnnGetRNNDescriptor -> miopenGetRNNDescriptor_V2 mapping after implementing cuDNN versioning in tests

  int hiddenSize = 0;
  int layer = 0;

  // NOTE: cudnnSetRNNDescriptor - removed after cuDNN 7.6.5
  // NOTE: cudnnSetRNNDescriptor_v5 - removed after cuDNN 7.6.5
  // TODO: add cudnnSetRNNDescriptor -> miopenSetRNNDescriptor_V2 mapping after implementing cuDNN versioning in tests

  int seqLength = 0;

#if CUDNN_MAJOR < 9
  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetRNNWorkspaceSize(miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* xDesc, size_t* numBytes);
  // CHECK: status = miopenGetRNNWorkspaceSize(handle, RNNDescriptor, seqLength, &xD, &workSpaceSizeInBytes);
  status = cudnnGetRNNWorkspaceSize(handle, RNNDescriptor, seqLength, &xD, &workSpaceSizeInBytes);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetRNNTrainingReserveSize(miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* xDesc, size_t* numBytes);
  // CHECK: status = miopenGetRNNTrainingReserveSize(handle, RNNDescriptor, seqLength, &xD, &workSpaceSizeInBytes);
  status = cudnnGetRNNTrainingReserveSize(handle, RNNDescriptor, seqLength, &xD, &workSpaceSizeInBytes);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const cudnnTensorDescriptor_t xDesc, size_t* sizeInBytes, cudnnDataType_t dataType);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetRNNParamsSize(miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, miopenTensorDescriptor_t xDesc, size_t* numBytes, miopenDataType_t dtype);
  // CHECK: status = miopenGetRNNParamsSize(handle, RNNDescriptor, xD, &workSpaceSizeInBytes, dataType);
  status = cudnnGetRNNParamsSize(handle, RNNDescriptor, xD, &workSpaceSizeInBytes, dataType);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, void* workSpace, size_t workSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenRNNForwardInference(miopenHandle_t handle, miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* xDesc, const void* x, const miopenTensorDescriptor_t hxDesc, const void* hx, const miopenTensorDescriptor_t cxDesc, const void* cx, const miopenTensorDescriptor_t wDesc, const void* w, const miopenTensorDescriptor_t* yDesc, void* y, const miopenTensorDescriptor_t hyDesc, void* hy, const miopenTensorDescriptor_t cyDesc, void* cy, void* workSpace, size_t workSpaceNumBytes);
  // CHECK: status = miopenRNNForwardInference(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, cxD, cx, filterDescriptor, W, &yD, y, hyD, hy, cyD, cy, workSpace, workSpaceSizeInBytes);
  status = cudnnRNNForwardInference(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, cxD, cx, filterDescriptor, W, &yD, y, hyD, hy, cyD, cy, workSpace, workSpaceSizeInBytes);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* xDesc, const void* x, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t* yDesc, void* y, const cudnnTensorDescriptor_t hyDesc, void* hy, const cudnnTensorDescriptor_t cyDesc, void* cy, void* workSpace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenRNNForwardTraining(miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* xDesc, const void* x, const miopenTensorDescriptor_t hxDesc, const void* hx, const miopenTensorDescriptor_t cxDesc, const void* cx, const miopenTensorDescriptor_t wDesc, const void* w, const miopenTensorDescriptor_t* yDesc, void* y, const miopenTensorDescriptor_t hyDesc, void* hy, const miopenTensorDescriptor_t cyDesc, void* cy, void* workSpace, size_t workSpaceNumBytes, void* reserveSpace, size_t reserveSpaceNumBytes);
  // CHECK: status = miopenRNNForwardTraining(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, cxD, cx, filterDescriptor, W, &yD, y, hyD, hy, cyD, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceNumBytes);
  status = cudnnRNNForwardTraining(handle, RNNDescriptor, seqLength, &xD, x, hxD, hx, cxD, cx, filterDescriptor, W, &yD, y, hyD, hy, cyD, cy, workSpace, workSpaceSizeInBytes, reserveSpace, reserveSpaceNumBytes);

  // CUDA: CUDNN_DEPRECATED cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, const int seqLength, const cudnnTensorDescriptor_t* yDesc, const void* y, const cudnnTensorDescriptor_t* dyDesc, const void* dy, const cudnnTensorDescriptor_t dhyDesc, const void* dhy, const cudnnTensorDescriptor_t dcyDesc, const void* dcy, const cudnnFilterDescriptor_t wDesc, const void* w, const cudnnTensorDescriptor_t hxDesc, const void* hx, const cudnnTensorDescriptor_t cxDesc, const void* cx, const cudnnTensorDescriptor_t* dxDesc, void* dx, const cudnnTensorDescriptor_t dhxDesc, void* dhx, const cudnnTensorDescriptor_t dcxDesc, void* dcx, void* workSpace, size_t workSpaceSizeInBytes, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenRNNBackwardData(miopenHandle_t handle, const miopenRNNDescriptor_t rnnDesc, const int sequenceLen, const miopenTensorDescriptor_t* yDesc, const void* y, const miopenTensorDescriptor_t* dyDesc, const void* dy, const miopenTensorDescriptor_t dhyDesc, const void* dhy, const miopenTensorDescriptor_t dcyDesc, const void* dcy, const miopenTensorDescriptor_t wDesc, const void* w, const miopenTensorDescriptor_t hxDesc, const void* hx, const miopenTensorDescriptor_t cxDesc, const void* cx, const miopenTensorDescriptor_t* dxDesc, void* dx, const miopenTensorDescriptor_t dhxDesc, void* dhx, const miopenTensorDescriptor_t dcxDesc, void* dcx, void* workSpace, size_t workSpaceNumBytes, void* reserveSpace, size_t reserveSpaceNumBytes);
  // CHECK: status = miopenRNNBackwardData(handle, RNNDescriptor, seqLength, &yD, y, &dyD, dy, dhyD, dhy, dcyD, dcy, filterDescriptor, W, hxD, hx, cxD, cx, &dxD, dx, dhxD, dhx, dcxD, dcx, workSpace, workSpaceSizeInBytes, &reserveSpace, reserveSpaceNumBytes);
  status = cudnnRNNBackwardData(handle, RNNDescriptor, seqLength, &yD, y, &dyD, dy, dhyD, dhy, dcyD, dcy, filterDescriptor, W, hxD, hx, cxD, cx, &dxD, dx, dhxD, dhx, dcxD, dcx, workSpace, workSpaceSizeInBytes, &reserveSpace, reserveSpaceNumBytes);

  // TODO [#837]: Insert int* blank_label_id, bool* apply_softmax_layer in the hipified miopenGetCTCLossDescriptor: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t* compType);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc, miopenDataType_t* dataType, int* blank_label_id, bool* apply_softmax_layer);
  // CHECK: status = miopenGetCTCLossDescriptor(CTCLossDescriptor, &dataType);
  status = cudnnGetCTCLossDescriptor(CTCLossDescriptor, &dataType);

  // TODO [#837]: Insert int blank_label_id, bool apply_softmax_layer in the hipified miopenSetCTCLossDescriptor: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc, miopenDataType_t dataType, const int blank_label_id, bool apply_softmax_layer);
  // CHECK: status = miopenSetCTCLossDescriptor(CTCLossDescriptor, dataType);
  status = cudnnSetCTCLossDescriptor(CTCLossDescriptor, dataType);
#endif

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyRNNDescriptor(miopenRNNDescriptor_t rnnDesc);
  // CHECK: status = miopenDestroyRNNDescriptor(RNNDescriptor);
  status = cudnnDestroyRNNDescriptor(RNNDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t* ctcLossDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateCTCLossDescriptor(miopenCTCLossDescriptor_t* ctcLossDesc);
  // CHECK: status = miopenCreateCTCLossDescriptor(&CTCLossDescriptor);
  status = cudnnCreateCTCLossDescriptor(&CTCLossDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyCTCLossDescriptor(miopenCTCLossDescriptor_t ctcLossDesc);
  // CHECK: status = miopenDestroyCTCLossDescriptor(CTCLossDescriptor);
  status = cudnnDestroyCTCLossDescriptor(CTCLossDescriptor);

  int labels = 0;
  int labelLengths = 0;
  int inputLengths = 0;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const cudnnTensorDescriptor_t gradientsDesc, const int* labels, const int* labelLengths, const int* inputLengths, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetCTCLossWorkspaceSize(miopenHandle_t handle, const miopenTensorDescriptor_t probsDesc, const miopenTensorDescriptor_t gradientsDesc, const int* labels, const int* labelLengths, const int* inputLengths, miopenCTCLossAlgo_t algo, const miopenCTCLossDescriptor_t ctcLossDesc, size_t* workSpaceSize);
  // CHECK: status = miopenGetCTCLossWorkspaceSize(handle, probsD, gradientsD, &labels, &labelLengths, &inputLengths, CTCLossAlgo, CTCLossDescriptor, &workSpaceSizeInBytes);
  status = cudnnGetCTCLossWorkspaceSize(handle, probsD, gradientsD, &labels, &labelLengths, &inputLengths, CTCLossAlgo, CTCLossDescriptor, &workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(cudnnHandle_t handle, const cudnnTensorDescriptor_t probsDesc, const void* probs, const int hostLabels[], const int hostLabelLengths[], const int hostInputLengths[], void* costs, const cudnnTensorDescriptor_t gradientsDesc, void* gradients, cudnnCTCLossAlgo_t algo, cudnnCTCLossDescriptor_t ctcLossDesc, void* workspace, size_t workSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCTCLoss(miopenHandle_t handle, const miopenTensorDescriptor_t probsDesc, const void* probs, const int* labels, const int* labelLengths, const int* inputLengths, void* losses, const miopenTensorDescriptor_t gradientsDesc, void* gradients, miopenCTCLossAlgo_t algo, const miopenCTCLossDescriptor_t ctcLossDesc, void* workSpace, size_t workSpaceSize);
  // CHECK: status = miopenCTCLoss(handle, probsD, probs, &labels, &labelLengths, &inputLengths, losses, gradientsD, gradients, CTCLossAlgo, CTCLossDescriptor, workSpace , workSpaceSizeInBytes);
  status = cudnnCTCLoss(handle, probsD, probs, &labels, &labelLengths, &inputLengths, losses, gradientsD, gradients, CTCLossAlgo, CTCLossDescriptor, workSpace , workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateDropoutDescriptor(miopenDropoutDescriptor_t* dropoutDesc);
  // CHECK: status = miopenCreateDropoutDescriptor(&DropoutDescriptor);
  status = cudnnCreateDropoutDescriptor(&DropoutDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc);
  // CHECK: status = miopenDestroyDropoutDescriptor(DropoutDescriptor);
  status = cudnnDestroyDropoutDescriptor(DropoutDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDropoutGetReserveSpaceSize(const miopenTensorDescriptor_t xDesc, size_t* reserveSpaceSizeInBytes);
  // CHECK: status = miopenDropoutGetReserveSpaceSize(xD, &reserveSpaceNumBytes);
  status = cudnnDropoutGetReserveSpaceSize(xD, &reserveSpaceNumBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDropoutGetStatesSize(miopenHandle_t handle, size_t* stateSizeInBytes);
  // CHECK: status = miopenDropoutGetStatesSize(handle, &reserveSpaceNumBytes);
  status = cudnnDropoutGetStatesSize(handle, &reserveSpaceNumBytes);

  float dropout = 0.0f;
  void* states = nullptr;
  unsigned long long seed = 0;

  // TODO [#837]: Insert float* dropout, void** states, unsigned long long* seed in the hipified miopenGetDropoutDescriptor: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float* dropout, void** states, unsigned long long* seed);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float* dropout, void** states, unsigned long long* seed, bool* use_mask, bool* state_evo, miopenRNGType_t* rng_mode);
  // CHECK: status = miopenGetDropoutDescriptor(DropoutDescriptor, handle, &dropout, &states, &seed);
  status = cudnnGetDropoutDescriptor(DropoutDescriptor, handle, &dropout, &states, &seed);

  // TODO [#837]: Insert bool use_mask, bool state_evo, miopenRNGType_t rng_mode in the hipified miopenGetDropoutDescriptor: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed, bool use_mask, bool state_evo, miopenRNGType_t rng_mode);
  // CHECK: status = miopenSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);
  status = cudnnSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);

  // TODO [#837]: Insert bool use_mask, bool state_evo, miopenRNGType_t rng_mode in the hipified miopenRestoreDropoutDescriptor: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenRestoreDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed, bool use_mask, bool state_evo, miopenRNGType_t rng_mode);
  // CHECK: status = miopenRestoreDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);
  status = cudnnRestoreDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);

  // TODO [#837]: Insert const miopenTensorDescriptor_t noise_shape in the hipified miopenDropoutForward: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t xdesc, const void* x, const cudnnTensorDescriptor_t ydesc, void* y, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDropoutForward(miopenHandle_t handle, const miopenDropoutDescriptor_t dropoutDesc, const miopenTensorDescriptor_t noise_shape, const miopenTensorDescriptor_t xDesc, const void* x, const miopenTensorDescriptor_t yDesc, void* y, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // CHECK: status = miopenDropoutForward(handle, DropoutDescriptor, xD, x, yD, y, reserveSpace, reserveSpaceNumBytes);
  status = cudnnDropoutForward(handle, DropoutDescriptor, xD, x, yD, y, reserveSpace, reserveSpaceNumBytes);

  // TODO [#837]: Insert const miopenTensorDescriptor_t noise_shape in the hipified miopenDropoutBackward: will need variable declaration
  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle, const cudnnDropoutDescriptor_t dropoutDesc, const cudnnTensorDescriptor_t dydesc, const void* dy, const cudnnTensorDescriptor_t dxdesc, void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // MIOPNE: MIOPEN_EXPORT miopenStatus_t miopenDropoutBackward(miopenHandle_t handle, const miopenDropoutDescriptor_t dropoutDesc, const miopenTensorDescriptor_t noise_shape, const miopenTensorDescriptor_t dyDesc, const void* dy, const miopenTensorDescriptor_t dxDesc, void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes);
  // CHECK: status = miopenDropoutBackward(handle, DropoutDescriptor, yD, y, xD, x, reserveSpace, reserveSpaceNumBytes);
  status = cudnnDropoutBackward(handle, DropoutDescriptor, yD, y, xD, x, reserveSpace, reserveSpaceNumBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* reduceTensorDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenCreateReduceTensorDescriptor(miopenReduceTensorDescriptor_t* reduceTensorDesc);
  // CHECK: status = miopenCreateReduceTensorDescriptor(&ReduceTensorDescriptor);
  status = cudnnCreateReduceTensorDescriptor(&ReduceTensorDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenDestroyReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc);
  // CHECK: status = miopenDestroyReduceTensorDescriptor(ReduceTensorDescriptor);
  status = cudnnDestroyReduceTensorDescriptor(ReduceTensorDescriptor);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetReductionIndicesSize(miopenHandle_t handle, const miopenReduceTensorDescriptor_t reduceTensorDesc, const miopenTensorDescriptor_t aDesc, const miopenTensorDescriptor_t cDesc, size_t* sizeInBytes);
  // CHECK: status = miopenGetReductionIndicesSize(handle, ReduceTensorDescriptor, aD, cD, &workSpaceSizeInBytes);
  status = cudnnGetReductionIndicesSize(handle, ReduceTensorDescriptor, aD, cD, &workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, const cudnnTensorDescriptor_t aDesc, const cudnnTensorDescriptor_t cDesc, size_t* sizeInBytes);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetReductionWorkspaceSize(miopenHandle_t handle, const miopenReduceTensorDescriptor_t reduceTensorDesc, const miopenTensorDescriptor_t aDesc, const miopenTensorDescriptor_t cDesc, size_t* sizeInBytes);
  // CHECK: status = miopenGetReductionWorkspaceSize(handle, ReduceTensorDescriptor, aD, cD, &workSpaceSizeInBytes);
  status = cudnnGetReductionWorkspaceSize(handle, ReduceTensorDescriptor, aD, cD, &workSpaceSizeInBytes);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle, const cudnnReduceTensorDescriptor_t reduceTensorDesc, void* indices, size_t indicesSizeInBytes, void* workspace, size_t workspaceSizeInBytes, const void* alpha, const cudnnTensorDescriptor_t aDesc, const void* A, const void* beta, const cudnnTensorDescriptor_t cDesc, void* C);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenReduceTensor(miopenHandle_t handle, const miopenReduceTensorDescriptor_t reduceTensorDesc, void* indices, size_t indicesSizeInBytes, void* workspace, size_t workspaceSizeInBytes, const void* alpha, const miopenTensorDescriptor_t aDesc, const void* A, const void* beta, const miopenTensorDescriptor_t cDesc, void* C);
  // CHECK: status = miopenReduceTensor(handle, ReduceTensorDescriptor, indices, indicesSizeInBytes, workSpace, workSpaceSizeInBytes, alpha, aD, A, beta, cD, C);
  status = cudnnReduceTensor(handle, ReduceTensorDescriptor, indices, indicesSizeInBytes, workSpace, workSpaceSizeInBytes, alpha, aD, A, beta, cD, C);

#if CUDNN_VERSION >= 2000
  // CHECK: miopenPoolingMode_t POOLING_AVERAGE_COUNT_INCLUDE_PADDING = miopenPoolingAverageInclusive;
  // CHECK-NEXT: miopenPoolingMode_t POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = miopenPoolingAverage;
  cudnnPoolingMode_t POOLING_AVERAGE_COUNT_INCLUDE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  cudnnPoolingMode_t POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
#endif

#if CUDNN_VERSION >= 3008
  // CHECK: miopenDataType_t DATA_BFLOAT16 = miopenBFloat16;
  cudnnDataType_t DATA_BFLOAT16 = CUDNN_DATA_BFLOAT16;
#endif

#if CUDNN_VERSION >= 4008
  // CHECK: miopenNanPropagation_t nanPropagation_t;
  // CHECK-NEXT: miopenNanPropagation_t NOT_PROPAGATE_NAN = MIOPEN_NOT_PROPAGATE_NAN;
  // CHECK-NEXT: miopenNanPropagation_t PROPAGATE_NAN = MIOPEN_PROPAGATE_NAN;
  cudnnNanPropagation_t nanPropagation_t;
  cudnnNanPropagation_t NOT_PROPAGATE_NAN = CUDNN_NOT_PROPAGATE_NAN;
  cudnnNanPropagation_t PROPAGATE_NAN = CUDNN_PROPAGATE_NAN;

  // CHECK: miopenActivationMode_t ACTIVATION_CLIPPED_RELU = miopenActivationCLIPPEDRELU;
  cudnnActivationMode_t ACTIVATION_CLIPPED_RELU = CUDNN_ACTIVATION_CLIPPED_RELU;
#endif

#if CUDNN_VERSION >= 6021
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

  // CHECK: miopenActivationMode_t ACTIVATION_ELU = miopenActivationELU;
  cudnnActivationMode_t ACTIVATION_ELU = CUDNN_ACTIVATION_ELU;

  // CHECK: miopenConvBwdDataAlgorithm_t CONVOLUTION_BWD_DATA_ALGO_COUNT = miopenTransposeBwdDataAlgoGEMM;
  cudnnConvolutionBwdDataAlgo_t CONVOLUTION_BWD_DATA_ALGO_COUNT = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t reduceTensorOp, cudnnDataType_t reduceTensorCompType, cudnnNanPropagation_t reduceTensorNanOpt, cudnnReduceTensorIndices_t reduceTensorIndices, cudnnIndicesType_t reduceTensorIndicesType);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetReduceTensorDescriptor(miopenReduceTensorDescriptor_t reduceTensorDesc, miopenReduceTensorOp_t reduceTensorOp, miopenDataType_t reduceTensorCompType, miopenNanPropagation_t reduceTensorNanOpt, miopenReduceTensorIndices_t reduceTensorIndices, miopenIndicesType_t reduceTensorIndicesType);
  // CHECK: status = miopenSetReduceTensorDescriptor(ReduceTensorDescriptor, reduceTensorOp, dataType, nanPropagation_t, reduceTensorIndices, indicesType);
  status = cudnnSetReduceTensorDescriptor(ReduceTensorDescriptor, reduceTensorOp, dataType, nanPropagation_t, reduceTensorIndices, indicesType);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc, cudnnReduceTensorOp_t* reduceTensorOp, cudnnDataType_t* reduceTensorCompType, cudnnNanPropagation_t* reduceTensorNanOpt, cudnnReduceTensorIndices_t* reduceTensorIndices, cudnnIndicesType_t* reduceTensorIndicesType);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenGetReduceTensorDescriptor(const miopenReduceTensorDescriptor_t reduceTensorDesc, miopenReduceTensorOp_t* reduceTensorOp, miopenDataType_t* reduceTensorCompType, miopenNanPropagation_t* reduceTensorNanOpt, miopenReduceTensorIndices_t* reduceTensorIndices, miopenIndicesType_t* reduceTensorIndicesType);
  // CHECK: status = miopenGetReduceTensorDescriptor(ReduceTensorDescriptor, &reduceTensorOp, &dataType, &nanPropagation_t, &reduceTensorIndices, &indicesType);
  status = cudnnGetReduceTensorDescriptor(ReduceTensorDescriptor, &reduceTensorOp, &dataType, &nanPropagation_t, &reduceTensorIndices, &indicesType);
#endif

#if CUDNN_VERSION >= 7103
  // CHECK: miopenActivationMode_t ACTIVATION_IDENTITY = miopenActivationPASTHRU;
  cudnnActivationMode_t ACTIVATION_IDENTITY = CUDNN_ACTIVATION_IDENTITY;
#endif

#if CUDNN_VERSION >= 7201 && CUDNN_VERSION <= 8907
  // CHECK: miopenRNNPaddingMode_t RNNPaddingMode_t;
  // CHECK-NEXT: miopenRNNPaddingMode_t RNN_PADDED_IO_DISABLED = miopenRNNIONotPadded;
  // CHECK-NEXT: miopenRNNPaddingMode_t RNN_PADDED_IO_ENABLED = miopenRNNIOWithPadding;
  cudnnRNNPaddingMode_t RNNPaddingMode_t;
  cudnnRNNPaddingMode_t RNN_PADDED_IO_DISABLED = CUDNN_RNN_PADDED_IO_DISABLED;
  cudnnRNNPaddingMode_t RNN_PADDED_IO_ENABLED = CUDNN_RNN_PADDED_IO_ENABLED;
#endif

#if CUDNN_VERSION >= 8001
  // CHECK: miopenStatus_t STATUS_VERSION_MISMATCH = miopenStatusVersionMismatch;
  cudnnStatus_t STATUS_VERSION_MISMATCH = CUDNN_STATUS_VERSION_MISMATCH;

  // CHECK: miopenBackendDescriptorType_t backendDescriptorType_t;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_POINTWISE_DESCRIPTOR = MIOPEN_BACKEND_POINTWISE_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_CONVOLUTION_DESCRIPTOR = MIOPEN_BACKEND_CONVOLUTION_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_ENGINE_DESCRIPTOR = MIOPEN_BACKEND_ENGINE_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_ENGINECFG_DESCRIPTOR = MIOPEN_BACKEND_ENGINECFG_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_ENGINEHEUR_DESCRIPTOR = MIOPEN_BACKEND_ENGINEHEUR_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_EXECUTION_PLAN_DESCRIPTOR = MIOPEN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = MIOPEN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_KNOB_CHOICE_DESCRIPTOR = MIOPEN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_KNOB_INFO_DESCRIPTOR = MIOPEN_BACKEND_KNOB_INFO_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_LAYOUT_INFO_DESCRIPTOR = MIOPEN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_POINTWISE_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATIONGRAPH_DESCRIPTOR = MIOPEN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_VARIANT_PACK_DESCRIPTOR = MIOPEN_BACKEND_VARIANT_PACK_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_TENSOR_DESCRIPTOR = MIOPEN_BACKEND_TENSOR_DESCRIPTOR;
  cudnnBackendDescriptorType_t backendDescriptorType_t;
  cudnnBackendDescriptorType_t BACKEND_POINTWISE_DESCRIPTOR = CUDNN_BACKEND_POINTWISE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_CONVOLUTION_DESCRIPTOR = CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_ENGINE_DESCRIPTOR = CUDNN_BACKEND_ENGINE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_ENGINECFG_DESCRIPTOR = CUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_ENGINEHEUR_DESCRIPTOR = CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_EXECUTION_PLAN_DESCRIPTOR = CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_INTERMEDIATE_INFO_DESCRIPTOR = CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_KNOB_CHOICE_DESCRIPTOR = CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_KNOB_INFO_DESCRIPTOR = CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_LAYOUT_INFO_DESCRIPTOR = CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_POINTWISE_DESCRIPTOR = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATIONGRAPH_DESCRIPTOR = CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_VARIANT_PACK_DESCRIPTOR = CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_TENSOR_DESCRIPTOR = CUDNN_BACKEND_TENSOR_DESCRIPTOR;

  // CHECK: miopenBackendAttributeType_t backendAttributeType_t;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_HANDLE = MIOPEN_TYPE_HANDLE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_DATA_TYPE = MIOPEN_TYPE_DATA_TYPE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_BOOLEAN = MIOPEN_TYPE_BOOLEAN;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_INT64 = MIOPEN_TYPE_INT64;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_FLOAT = MIOPEN_TYPE_FLOAT;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_DOUBLE = MIOPEN_TYPE_DOUBLE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_VOID_PTR = MIOPEN_TYPE_VOID_PTR;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_CONVOLUTION_MODE = MIOPEN_TYPE_CONVOLUTION_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_HEUR_MODE = MIOPEN_TYPE_HEUR_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_KNOB_TYPE = MIOPEN_TYPE_KNOB_TYPE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_NAN_PROPOGATION = MIOPEN_TYPE_NAN_PROPOGATION;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_NUMERICAL_NOTE = MIOPEN_TYPE_NUMERICAL_NOTE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_ATTRIB_NAME = MIOPEN_TYPE_ATTRIB_NAME;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_POINTWISE_MODE = MIOPEN_TYPE_POINTWISE_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_BACKEND_DESCRIPTOR = MIOPEN_TYPE_BACKEND_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_GENSTATS_MODE = MIOPEN_TYPE_GENSTATS_MODE;
  cudnnBackendAttributeType_t backendAttributeType_t;
  cudnnBackendAttributeType_t TYPE_HANDLE = CUDNN_TYPE_HANDLE;
  cudnnBackendAttributeType_t TYPE_DATA_TYPE = CUDNN_TYPE_DATA_TYPE;
  cudnnBackendAttributeType_t TYPE_BOOLEAN = CUDNN_TYPE_BOOLEAN;
  cudnnBackendAttributeType_t TYPE_INT64 = CUDNN_TYPE_INT64;
  cudnnBackendAttributeType_t TYPE_FLOAT = CUDNN_TYPE_FLOAT;
  cudnnBackendAttributeType_t TYPE_DOUBLE = CUDNN_TYPE_DOUBLE;
  cudnnBackendAttributeType_t TYPE_VOID_PTR = CUDNN_TYPE_VOID_PTR;
  cudnnBackendAttributeType_t TYPE_CONVOLUTION_MODE = CUDNN_TYPE_CONVOLUTION_MODE;
  cudnnBackendAttributeType_t TYPE_HEUR_MODE = CUDNN_TYPE_HEUR_MODE;
  cudnnBackendAttributeType_t TYPE_KNOB_TYPE = CUDNN_TYPE_KNOB_TYPE;
  cudnnBackendAttributeType_t TYPE_NAN_PROPOGATION = CUDNN_TYPE_NAN_PROPOGATION;
  cudnnBackendAttributeType_t TYPE_NUMERICAL_NOTE = CUDNN_TYPE_NUMERICAL_NOTE;
  cudnnBackendAttributeType_t TYPE_ATTRIB_NAME = CUDNN_TYPE_ATTRIB_NAME;
  cudnnBackendAttributeType_t TYPE_POINTWISE_MODE = CUDNN_TYPE_POINTWISE_MODE;
  cudnnBackendAttributeType_t TYPE_BACKEND_DESCRIPTOR = CUDNN_TYPE_BACKEND_DESCRIPTOR;
  cudnnBackendAttributeType_t TYPE_GENSTATS_MODE = CUDNN_TYPE_GENSTATS_MODE;

  // CHECK: miopenBackendAttributeName_t backendAttributeName_t;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_MODE = MIOPEN_ATTR_POINTWISE_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_MATH_PREC = MIOPEN_ATTR_POINTWISE_MATH_PREC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_NAN_PROPAGATION = MIOPEN_ATTR_POINTWISE_NAN_PROPAGATION;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_RELU_LOWER_CLIP = MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_RELU_UPPER_CLIP = MIOPEN_ATTR_POINTWISE_RELU_UPPER_CLIP;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_COMP_TYPE = MIOPEN_ATTR_CONVOLUTION_COMP_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_DILATIONS = MIOPEN_ATTR_CONVOLUTION_DILATIONS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_FILTER_STRIDES = MIOPEN_ATTR_CONVOLUTION_FILTER_STRIDES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_POST_PADDINGS = MIOPEN_ATTR_CONVOLUTION_POST_PADDINGS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_PRE_PADDINGS = MIOPEN_ATTR_CONVOLUTION_PRE_PADDINGS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_CONVOLUTION_SPATIAL_DIMS = MIOPEN_ATTR_CONVOLUTION_SPATIAL_DIMS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINEHEUR_MODE = MIOPEN_ATTR_ENGINEHEUR_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINEHEUR_OPERATION_GRAPH = MIOPEN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINEHEUR_RESULTS = MIOPEN_ATTR_ENGINEHEUR_RESULTS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINECFG_ENGINE = MIOPEN_ATTR_ENGINECFG_ENGINE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINECFG_INTERMEDIATE_INFO = MIOPEN_ATTR_ENGINECFG_INTERMEDIATE_INFO;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINECFG_KNOB_CHOICES = MIOPEN_ATTR_ENGINECFG_KNOB_CHOICES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_HANDLE = MIOPEN_ATTR_EXECUTION_PLAN_HANDLE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_ENGINE_CONFIG = MIOPEN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = MIOPEN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_INTERMEDIATE_INFO_SIZE = MIOPEN_ATTR_INTERMEDIATE_INFO_SIZE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_CHOICE_KNOB_TYPE = MIOPEN_ATTR_KNOB_CHOICE_KNOB_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_CHOICE_KNOB_VALUE = MIOPEN_ATTR_KNOB_CHOICE_KNOB_VALUE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_W = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_W;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_X = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_X;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_Y = MIOPEN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = MIOPEN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = MIOPEN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_XDESC = MIOPEN_ATTR_OPERATION_POINTWISE_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_BDESC = MIOPEN_ATTR_OPERATION_POINTWISE_BDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_YDESC = MIOPEN_ATTR_OPERATION_POINTWISE_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_ALPHA1 = MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA1;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_ALPHA2 = MIOPEN_ATTR_OPERATION_POINTWISE_ALPHA2;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_GENSTATS_MODE = MIOPEN_ATTR_OPERATION_GENSTATS_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_GENSTATS_MATH_PREC = MIOPEN_ATTR_OPERATION_GENSTATS_MATH_PREC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_GENSTATS_XDESC = MIOPEN_ATTR_OPERATION_GENSTATS_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_GENSTATS_SUMDESC = MIOPEN_ATTR_OPERATION_GENSTATS_SUMDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_GENSTATS_SQSUMDESC = MIOPEN_ATTR_OPERATION_GENSTATS_SQSUMDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATIONGRAPH_HANDLE = MIOPEN_ATTR_OPERATIONGRAPH_HANDLE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATIONGRAPH_OPS = MIOPEN_ATTR_OPERATIONGRAPH_OPS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = MIOPEN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_BYTE_ALIGNMENT = MIOPEN_ATTR_TENSOR_BYTE_ALIGNMENT;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_DATA_TYPE = MIOPEN_ATTR_TENSOR_DATA_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_DIMENSIONS = MIOPEN_ATTR_TENSOR_DIMENSIONS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_STRIDES = MIOPEN_ATTR_TENSOR_STRIDES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_VECTOR_COUNT = MIOPEN_ATTR_TENSOR_VECTOR_COUNT;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_VECTORIZED_DIMENSION = MIOPEN_ATTR_TENSOR_VECTORIZED_DIMENSION;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_UNIQUE_ID = MIOPEN_ATTR_TENSOR_UNIQUE_ID;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_IS_VIRTUAL = MIOPEN_ATTR_TENSOR_IS_VIRTUAL;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_VARIANT_PACK_UNIQUE_IDS = MIOPEN_ATTR_VARIANT_PACK_UNIQUE_IDS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_VARIANT_PACK_DATA_POINTERS = MIOPEN_ATTR_VARIANT_PACK_DATA_POINTERS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_VARIANT_PACK_INTERMEDIATES = MIOPEN_ATTR_VARIANT_PACK_INTERMEDIATES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_VARIANT_PACK_WORKSPACE = MIOPEN_ATTR_VARIANT_PACK_WORKSPACE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_INFO_TYPE = MIOPEN_ATTR_KNOB_INFO_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_INFO_MAXIMUM_VALUE = MIOPEN_ATTR_KNOB_INFO_MAXIMUM_VALUE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_INFO_MINIMUM_VALUE = MIOPEN_ATTR_KNOB_INFO_MINIMUM_VALUE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_KNOB_INFO_STRIDE = MIOPEN_ATTR_KNOB_INFO_STRIDE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_OPERATION_GRAPH = MIOPEN_ATTR_ENGINE_OPERATION_GRAPH;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_GLOBAL_INDEX = MIOPEN_ATTR_ENGINE_GLOBAL_INDEX;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_NUMERICAL_NOTE = MIOPEN_ATTR_ENGINE_NUMERICAL_NOTE;
  cudnnBackendAttributeName_t backendAttributeName_t;
  cudnnBackendAttributeName_t ATTR_POINTWISE_MODE = CUDNN_ATTR_POINTWISE_MODE;
  cudnnBackendAttributeName_t ATTR_POINTWISE_MATH_PREC = CUDNN_ATTR_POINTWISE_MATH_PREC;
  cudnnBackendAttributeName_t ATTR_POINTWISE_NAN_PROPAGATION = CUDNN_ATTR_POINTWISE_NAN_PROPAGATION;
  cudnnBackendAttributeName_t ATTR_POINTWISE_RELU_LOWER_CLIP = CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP;
  cudnnBackendAttributeName_t ATTR_POINTWISE_RELU_UPPER_CLIP = CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_COMP_TYPE = CUDNN_ATTR_CONVOLUTION_COMP_TYPE;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_DILATIONS = CUDNN_ATTR_CONVOLUTION_DILATIONS;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_FILTER_STRIDES = CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_POST_PADDINGS = CUDNN_ATTR_CONVOLUTION_POST_PADDINGS;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_PRE_PADDINGS = CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_SPATIAL_DIMS = CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS;
  cudnnBackendAttributeName_t ATTR_ENGINEHEUR_MODE = CUDNN_ATTR_ENGINEHEUR_MODE;
  cudnnBackendAttributeName_t ATTR_ENGINEHEUR_OPERATION_GRAPH = CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
  cudnnBackendAttributeName_t ATTR_ENGINEHEUR_RESULTS = CUDNN_ATTR_ENGINEHEUR_RESULTS;
  cudnnBackendAttributeName_t ATTR_ENGINECFG_ENGINE = CUDNN_ATTR_ENGINECFG_ENGINE;
  cudnnBackendAttributeName_t ATTR_ENGINECFG_INTERMEDIATE_INFO = CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO;
  cudnnBackendAttributeName_t ATTR_ENGINECFG_KNOB_CHOICES = CUDNN_ATTR_ENGINECFG_KNOB_CHOICES;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_HANDLE = CUDNN_ATTR_EXECUTION_PLAN_HANDLE;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_ENGINE_CONFIG = CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE;
  cudnnBackendAttributeName_t ATTR_INTERMEDIATE_INFO_SIZE = CUDNN_ATTR_INTERMEDIATE_INFO_SIZE;
  cudnnBackendAttributeName_t ATTR_KNOB_CHOICE_KNOB_TYPE = CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE;
  cudnnBackendAttributeName_t ATTR_KNOB_CHOICE_KNOB_VALUE = CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_W = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_X = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_FORWARD_Y = CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_XDESC = CUDNN_ATTR_OPERATION_POINTWISE_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_BDESC = CUDNN_ATTR_OPERATION_POINTWISE_BDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_YDESC = CUDNN_ATTR_OPERATION_POINTWISE_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_ALPHA1 = CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_ALPHA2 = CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2;
  cudnnBackendAttributeName_t ATTR_OPERATION_GENSTATS_MODE = CUDNN_ATTR_OPERATION_GENSTATS_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_GENSTATS_MATH_PREC = CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC;
  cudnnBackendAttributeName_t ATTR_OPERATION_GENSTATS_XDESC = CUDNN_ATTR_OPERATION_GENSTATS_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_GENSTATS_SUMDESC = CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_GENSTATS_SQSUMDESC = CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC;
  cudnnBackendAttributeName_t ATTR_OPERATIONGRAPH_HANDLE = CUDNN_ATTR_OPERATIONGRAPH_HANDLE;
  cudnnBackendAttributeName_t ATTR_OPERATIONGRAPH_OPS = CUDNN_ATTR_OPERATIONGRAPH_OPS;
  cudnnBackendAttributeName_t ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT;
  cudnnBackendAttributeName_t ATTR_TENSOR_BYTE_ALIGNMENT = CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT;
  cudnnBackendAttributeName_t ATTR_TENSOR_DATA_TYPE = CUDNN_ATTR_TENSOR_DATA_TYPE;
  cudnnBackendAttributeName_t ATTR_TENSOR_DIMENSIONS = CUDNN_ATTR_TENSOR_DIMENSIONS;
  cudnnBackendAttributeName_t ATTR_TENSOR_STRIDES = CUDNN_ATTR_TENSOR_STRIDES;
  cudnnBackendAttributeName_t ATTR_TENSOR_VECTOR_COUNT = CUDNN_ATTR_TENSOR_VECTOR_COUNT;
  cudnnBackendAttributeName_t ATTR_TENSOR_VECTORIZED_DIMENSION = CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION;
  cudnnBackendAttributeName_t ATTR_TENSOR_UNIQUE_ID = CUDNN_ATTR_TENSOR_UNIQUE_ID;
  cudnnBackendAttributeName_t ATTR_TENSOR_IS_VIRTUAL = CUDNN_ATTR_TENSOR_IS_VIRTUAL;
  cudnnBackendAttributeName_t ATTR_VARIANT_PACK_UNIQUE_IDS = CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS;
  cudnnBackendAttributeName_t ATTR_VARIANT_PACK_DATA_POINTERS = CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS;
  cudnnBackendAttributeName_t ATTR_VARIANT_PACK_INTERMEDIATES = CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES;
  cudnnBackendAttributeName_t ATTR_VARIANT_PACK_WORKSPACE = CUDNN_ATTR_VARIANT_PACK_WORKSPACE;
  cudnnBackendAttributeName_t ATTR_KNOB_INFO_TYPE = CUDNN_ATTR_KNOB_INFO_TYPE;
  cudnnBackendAttributeName_t ATTR_KNOB_INFO_MAXIMUM_VALUE = CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE;
  cudnnBackendAttributeName_t ATTR_KNOB_INFO_MINIMUM_VALUE = CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE;
  cudnnBackendAttributeName_t ATTR_KNOB_INFO_STRIDE = CUDNN_ATTR_KNOB_INFO_STRIDE;
  cudnnBackendAttributeName_t ATTR_ENGINE_OPERATION_GRAPH = CUDNN_ATTR_ENGINE_OPERATION_GRAPH;
  cudnnBackendAttributeName_t ATTR_ENGINE_GLOBAL_INDEX = CUDNN_ATTR_ENGINE_GLOBAL_INDEX;
  cudnnBackendAttributeName_t ATTR_ENGINE_NUMERICAL_NOTE = CUDNN_ATTR_ENGINE_NUMERICAL_NOTE;

  // CHECK: miopenPointwiseMode_t pointwiseMode_t;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_ADD = MIOPEN_POINTWISE_ADD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_MUL = MIOPEN_POINTWISE_MUL;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_MIN = MIOPEN_POINTWISE_MIN;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_MAX = MIOPEN_POINTWISE_MAX;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SQRT = MIOPEN_POINTWISE_SQRT;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_RELU_FWD = MIOPEN_POINTWISE_RELU_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_TANH_FWD = MIOPEN_POINTWISE_TANH_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SIGMOID_FWD = MIOPEN_POINTWISE_SIGMOID_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_ELU_FWD = MIOPEN_POINTWISE_ELU_FWD;
  cudnnPointwiseMode_t pointwiseMode_t;
  cudnnPointwiseMode_t POINTWISE_ADD = CUDNN_POINTWISE_ADD;
  cudnnPointwiseMode_t POINTWISE_MUL = CUDNN_POINTWISE_MUL;
  cudnnPointwiseMode_t POINTWISE_MIN = CUDNN_POINTWISE_MIN;
  cudnnPointwiseMode_t POINTWISE_MAX = CUDNN_POINTWISE_MAX;
  cudnnPointwiseMode_t POINTWISE_SQRT = CUDNN_POINTWISE_SQRT;
  cudnnPointwiseMode_t POINTWISE_RELU_FWD = CUDNN_POINTWISE_RELU_FWD;
  cudnnPointwiseMode_t POINTWISE_TANH_FWD = CUDNN_POINTWISE_TANH_FWD;
  cudnnPointwiseMode_t POINTWISE_SIGMOID_FWD = CUDNN_POINTWISE_SIGMOID_FWD;
  cudnnPointwiseMode_t POINTWISE_ELU_FWD = CUDNN_POINTWISE_ELU_FWD;

  // CHECK: miopenBackendDescriptor_t backendDescriptor_t, backendDescriptor_2;
  cudnnBackendDescriptor_t backendDescriptor_t, backendDescriptor_2;

  // CHECK: miopenBackendHeurMode_t backendHeurMode;
  // CHECK-NEXT: miopenBackendHeurMode_t HEUR_MODE_INSTANT = MIOPEN_HEUR_MODE_INSTANT;
  // CHECK-NEXT: miopenBackendHeurMode_t HEUR_MODE_B = MIOPEN_HEUR_MODE_B;
  // CHECK-NEXT: miopenBackendHeurMode_t HEUR_MODES_COUNT = MIOPEN_HEUR_MODES_COUNT;
  cudnnBackendHeurMode_t backendHeurMode;
  cudnnBackendHeurMode_t HEUR_MODE_INSTANT = CUDNN_HEUR_MODE_INSTANT;
  cudnnBackendHeurMode_t HEUR_MODE_B = CUDNN_HEUR_MODE_B;
  cudnnBackendHeurMode_t HEUR_MODES_COUNT = CUDNN_HEUR_MODES_COUNT;

  // CHECK: miopenRNNFWDMode_t RNNFWDMode_t;
  // CHECK-NEXT: miopenRNNFWDMode_t FWD_MODE_INFERENCE = miopenRNNInference;
  // CHECK-NEXT: miopenRNNFWDMode_t FWD_MODE_TRAINING = miopenRNNTraining;
  cudnnForwardMode_t RNNFWDMode_t;
  cudnnForwardMode_t FWD_MODE_INFERENCE = CUDNN_FWD_MODE_INFERENCE;
  cudnnForwardMode_t FWD_MODE_TRAINING = CUDNN_FWD_MODE_TRAINING;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendCreateDescriptor(miopenBackendDescriptorType_t descriptorType, miopenBackendDescriptor_t* descriptor);
  // CHECK: status = miopenBackendCreateDescriptor(backendDescriptorType_t, &backendDescriptor_t);
  status = cudnnBackendCreateDescriptor(backendDescriptorType_t, &backendDescriptor_t);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendDestroyDescriptor(miopenBackendDescriptor_t descriptor);
  // CHECK: status = miopenBackendDestroyDescriptor(backendDescriptor_t);
  status = cudnnBackendDestroyDescriptor(backendDescriptor_t);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendFinalize(miopenBackendDescriptor_t descriptor);
  // CHECK: status = miopenBackendFinalize(backendDescriptor_t);
  status = cudnnBackendFinalize(backendDescriptor_t);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t attributeName, cudnnBackendAttributeType_t attributeType, int64_t elementCount, const void *arrayOfElements);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendSetAttribute(miopenBackendDescriptor_t descriptor, miopenBackendAttributeName_t attributeName, miopenBackendAttributeType_t attributeType, int64_t elementCount, void* arrayOfElements);
  // CHECK: status = miopenBackendSetAttribute(backendDescriptor_t, backendAttributeName_t, backendAttributeType_t, elementCount, arrayOfElements);
  status = cudnnBackendSetAttribute(backendDescriptor_t, backendAttributeName_t, backendAttributeType_t, elementCount, arrayOfElements);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor, cudnnBackendAttributeName_t attributeName, cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount, int64_t *elementCount, void *arrayOfElements);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendGetAttribute(miopenBackendDescriptor_t descriptor, miopenBackendAttributeName_t attributeName, miopenBackendAttributeType_t attributeType, int64_t requestedElementCount, int64_t* elementCount, void* arrayOfElements);
  // CHECK: status = miopenBackendGetAttribute(backendDescriptor_t, backendAttributeName_t, backendAttributeType_t, requestedElementCount, &elementCount, arrayOfElements);
  status = cudnnBackendGetAttribute(backendDescriptor_t, backendAttributeName_t, backendAttributeType_t, requestedElementCount, &elementCount, arrayOfElements);

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenBackendExecute(miopenHandle_t handle, miopenBackendDescriptor_t executionPlan, miopenBackendDescriptor_t variantPack);
  // CHECK: status = miopenBackendExecute(handle, backendDescriptor_t, backendDescriptor_2);
  status = cudnnBackendExecute(handle, backendDescriptor_t, backendDescriptor_2);
#endif

#if CUDNN_VERSION >= 8002
  // CHECK: miopenBackendAttributeType_t TYPE_LAYOUT_TYPE = MIOPEN_TYPE_LAYOUT_TYPE;
  cudnnBackendAttributeType_t TYPE_LAYOUT_TYPE = CUDNN_TYPE_LAYOUT_TYPE;

  // CHECK: miopenBackendAttributeName_t ATTR_CONVOLUTION_CONV_MODE = MIOPEN_ATTR_CONVOLUTION_CONV_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = MIOPEN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = MIOPEN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_INTERMEDIATE_INFO_UNIQUE_ID = MIOPEN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = MIOPEN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_LAYOUT_INFO_TENSOR_UID = MIOPEN_ATTR_LAYOUT_INFO_TENSOR_UID;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_LAYOUT_INFO_TYPES = MIOPEN_ATTR_LAYOUT_INFO_TYPES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_KNOB_INFO = MIOPEN_ATTR_ENGINE_KNOB_INFO;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_LAYOUT_INFO = MIOPEN_ATTR_ENGINE_LAYOUT_INFO;
  cudnnBackendAttributeName_t ATTR_CONVOLUTION_CONV_MODE = CUDNN_ATTR_CONVOLUTION_CONV_MODE;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS;
  cudnnBackendAttributeName_t ATTR_INTERMEDIATE_INFO_UNIQUE_ID = CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID;
  cudnnBackendAttributeName_t ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS = CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS;
  cudnnBackendAttributeName_t ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES;
  cudnnBackendAttributeName_t ATTR_LAYOUT_INFO_TENSOR_UID = CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID;
  cudnnBackendAttributeName_t ATTR_LAYOUT_INFO_TYPES = CUDNN_ATTR_LAYOUT_INFO_TYPES;
  cudnnBackendAttributeName_t ATTR_ENGINE_KNOB_INFO = CUDNN_ATTR_ENGINE_KNOB_INFO;
  cudnnBackendAttributeName_t ATTR_ENGINE_LAYOUT_INFO = CUDNN_ATTR_ENGINE_LAYOUT_INFO;
#endif

#if CUDNN_VERSION >= 8100
  // CHECK: miopenDataType_t DATA_INT64 = miopenInt64;
  cudnnDataType_t DATA_INT64 = CUDNN_DATA_INT64;

  // CHECK: miopenBackendDescriptorType_t BACKEND_MATMUL_DESCRIPTOR = MIOPEN_BACKEND_MATMUL_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_MATMUL_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_REDUCTION_DESCRIPTOR = MIOPEN_BACKEND_REDUCTION_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_REDUCTION_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_MATMUL_DESCRIPTOR = CUDNN_BACKEND_MATMUL_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_MATMUL_DESCRIPTOR = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_REDUCTION_DESCRIPTOR = CUDNN_BACKEND_REDUCTION_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_REDUCTION_DESCRIPTOR = CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;

  // CHECK: miopenBackendAttributeType_t TYPE_BN_FINALIZE_STATS_MODE = MIOPEN_TYPE_BN_FINALIZE_STATS_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_REDUCTION_OPERATOR_TYPE = MIOPEN_TYPE_REDUCTION_OPERATOR_TYPE;
  cudnnBackendAttributeType_t TYPE_BN_FINALIZE_STATS_MODE = CUDNN_TYPE_BN_FINALIZE_STATS_MODE;
  cudnnBackendAttributeType_t TYPE_REDUCTION_OPERATOR_TYPE = CUDNN_TYPE_REDUCTION_OPERATOR_TYPE;

  // CHECK: miopenBackendAttributeName_t ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = MIOPEN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_ELU_ALPHA = MIOPEN_ATTR_POINTWISE_ELU_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_SOFTPLUS_BETA = MIOPEN_ATTR_POINTWISE_SOFTPLUS_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_POINTWISE_SWISH_BETA = MIOPEN_ATTR_POINTWISE_SWISH_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_DXDESC = MIOPEN_ATTR_OPERATION_POINTWISE_DXDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_DYDESC = MIOPEN_ATTR_OPERATION_POINTWISE_DYDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_STATS_MODE = MIOPEN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_MATH_PREC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = MIOPEN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_IS_BY_VALUE = MIOPEN_ATTR_TENSOR_IS_BY_VALUE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_MATMUL_COMP_TYPE = MIOPEN_ATTR_MATMUL_COMP_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_ADESC = MIOPEN_ATTR_OPERATION_MATMUL_ADESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_BDESC = MIOPEN_ATTR_OPERATION_MATMUL_BDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_CDESC = MIOPEN_ATTR_OPERATION_MATMUL_CDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_DESC = MIOPEN_ATTR_OPERATION_MATMUL_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = MIOPEN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_REDUCTION_OPERATOR = MIOPEN_ATTR_REDUCTION_OPERATOR;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_REDUCTION_COMP_TYPE = MIOPEN_ATTR_REDUCTION_COMP_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_REDUCTION_XDESC = MIOPEN_ATTR_OPERATION_REDUCTION_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_REDUCTION_YDESC = MIOPEN_ATTR_OPERATION_REDUCTION_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_REDUCTION_DESC = MIOPEN_ATTR_OPERATION_REDUCTION_DESC;
  cudnnBackendAttributeName_t ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE;
  cudnnBackendAttributeName_t ATTR_POINTWISE_ELU_ALPHA = CUDNN_ATTR_POINTWISE_ELU_ALPHA;
  cudnnBackendAttributeName_t ATTR_POINTWISE_SOFTPLUS_BETA = CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA;
  cudnnBackendAttributeName_t ATTR_POINTWISE_SWISH_BETA = CUDNN_ATTR_POINTWISE_SWISH_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_DXDESC = CUDNN_ATTR_OPERATION_POINTWISE_DXDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_DYDESC = CUDNN_ATTR_OPERATION_POINTWISE_DYDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_STATS_MODE = CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_MATH_PREC = CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC;
  cudnnBackendAttributeName_t ATTR_TENSOR_IS_BY_VALUE = CUDNN_ATTR_TENSOR_IS_BY_VALUE;
  cudnnBackendAttributeName_t ATTR_MATMUL_COMP_TYPE = CUDNN_ATTR_MATMUL_COMP_TYPE;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_ADESC = CUDNN_ATTR_OPERATION_MATMUL_ADESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_BDESC = CUDNN_ATTR_OPERATION_MATMUL_BDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_CDESC = CUDNN_ATTR_OPERATION_MATMUL_CDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_DESC = CUDNN_ATTR_OPERATION_MATMUL_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT;
  cudnnBackendAttributeName_t ATTR_REDUCTION_OPERATOR = CUDNN_ATTR_REDUCTION_OPERATOR;
  cudnnBackendAttributeName_t ATTR_REDUCTION_COMP_TYPE = CUDNN_ATTR_REDUCTION_COMP_TYPE;
  cudnnBackendAttributeName_t ATTR_OPERATION_REDUCTION_XDESC = CUDNN_ATTR_OPERATION_REDUCTION_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_REDUCTION_YDESC = CUDNN_ATTR_OPERATION_REDUCTION_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_REDUCTION_DESC = CUDNN_ATTR_OPERATION_REDUCTION_DESC;

  // CHECK: miopenPointwiseMode_t POINTWISE_GELU_FWD = MIOPEN_POINTWISE_GELU_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SOFTPLUS_FWD = MIOPEN_POINTWISE_SOFTPLUS_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SWISH_FWD = MIOPEN_POINTWISE_SWISH_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_RELU_BWD = MIOPEN_POINTWISE_RELU_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_TANH_BWD = MIOPEN_POINTWISE_TANH_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SIGMOID_BWD = MIOPEN_POINTWISE_SIGMOID_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_ELU_BWD = MIOPEN_POINTWISE_ELU_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_GELU_BWD = MIOPEN_POINTWISE_GELU_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SOFTPLUS_BWD = MIOPEN_POINTWISE_SOFTPLUS_BWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SWISH_BWD = MIOPEN_POINTWISE_SWISH_BWD;
  cudnnPointwiseMode_t POINTWISE_GELU_FWD = CUDNN_POINTWISE_GELU_FWD;
  cudnnPointwiseMode_t POINTWISE_SOFTPLUS_FWD = CUDNN_POINTWISE_SOFTPLUS_FWD;
  cudnnPointwiseMode_t POINTWISE_SWISH_FWD = CUDNN_POINTWISE_SWISH_FWD;
  cudnnPointwiseMode_t POINTWISE_RELU_BWD = CUDNN_POINTWISE_RELU_BWD;
  cudnnPointwiseMode_t POINTWISE_TANH_BWD = CUDNN_POINTWISE_TANH_BWD;
  cudnnPointwiseMode_t POINTWISE_SIGMOID_BWD = CUDNN_POINTWISE_SIGMOID_BWD;
  cudnnPointwiseMode_t POINTWISE_ELU_BWD = CUDNN_POINTWISE_ELU_BWD;
  cudnnPointwiseMode_t POINTWISE_GELU_BWD = CUDNN_POINTWISE_GELU_BWD;
  cudnnPointwiseMode_t POINTWISE_SOFTPLUS_BWD = CUDNN_POINTWISE_SOFTPLUS_BWD;
  cudnnPointwiseMode_t POINTWISE_SWISH_BWD = CUDNN_POINTWISE_SWISH_BWD;
#endif

#if CUDNN_VERSION >= 8200
  // CHECK: miopenBackendAttributeType_t TYPE_BEHAVIOR_NOTE = MIOPEN_TYPE_BEHAVIOR_NOTE;
  cudnnBackendAttributeType_t TYPE_BEHAVIOR_NOTE = CUDNN_TYPE_BEHAVIOR_NOTE;

  // CHECK: miopenBackendAttributeName_t ATTR_ENGINE_BEHAVIOR_NOTE = MIOPEN_ATTR_ENGINE_BEHAVIOR_NOTE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = MIOPEN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS;
  cudnnBackendAttributeName_t ATTR_ENGINE_BEHAVIOR_NOTE = CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS;
#endif

#if CUDNN_VERSION >= 8300
  // CHECK: miopenBackendDescriptorType_t BACKEND_RESAMPLE_DESCRIPTOR = MIOPEN_BACKEND_RESAMPLE_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_RESAMPLE_DESCRIPTOR = CUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;

  // CHECK: miopenBackendAttributeType_t TYPE_TENSOR_REORDERING_MODE = MIOPEN_TYPE_TENSOR_REORDERING_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_RESAMPLE_MODE = MIOPEN_TYPE_RESAMPLE_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_PADDING_MODE = MIOPEN_TYPE_PADDING_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_INT32 = MIOPEN_TYPE_INT32;
  cudnnBackendAttributeType_t TYPE_TENSOR_REORDERING_MODE = CUDNN_TYPE_TENSOR_REORDERING_MODE;
  cudnnBackendAttributeType_t TYPE_RESAMPLE_MODE = CUDNN_TYPE_RESAMPLE_MODE;
  cudnnBackendAttributeType_t TYPE_PADDING_MODE = CUDNN_TYPE_PADDING_MODE;
  cudnnBackendAttributeType_t TYPE_INT32 = CUDNN_TYPE_INT32;

  // CHECK: miopenBackendAttributeName_t ATTR_OPERATION_POINTWISE_TDESC = MIOPEN_ATTR_OPERATION_POINTWISE_TDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_TENSOR_REORDERING_MODE = MIOPEN_ATTR_TENSOR_REORDERING_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_MODE = MIOPEN_ATTR_RESAMPLE_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_COMP_TYPE = MIOPEN_ATTR_RESAMPLE_COMP_TYPE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_SPATIAL_DIMS = MIOPEN_ATTR_RESAMPLE_SPATIAL_DIMS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_POST_PADDINGS = MIOPEN_ATTR_RESAMPLE_POST_PADDINGS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_PRE_PADDINGS = MIOPEN_ATTR_RESAMPLE_PRE_PADDINGS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_STRIDES = MIOPEN_ATTR_RESAMPLE_STRIDES;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_WINDOW_DIMS = MIOPEN_ATTR_RESAMPLE_WINDOW_DIMS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_NAN_PROPAGATION = MIOPEN_ATTR_RESAMPLE_NAN_PROPAGATION;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RESAMPLE_PADDING_MODE = MIOPEN_ATTR_RESAMPLE_PADDING_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_XDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_YDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_ALPHA = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_BETA = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_DESC = MIOPEN_ATTR_OPERATION_RESAMPLE_FWD_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DXDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DYDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_ALPHA = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_BETA = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_BETA;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_POINTWISE_TDESC = CUDNN_ATTR_OPERATION_POINTWISE_TDESC;
  cudnnBackendAttributeName_t ATTR_TENSOR_REORDERING_MODE = CUDNN_ATTR_TENSOR_REORDERING_MODE;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_MODE = CUDNN_ATTR_RESAMPLE_MODE;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_COMP_TYPE = CUDNN_ATTR_RESAMPLE_COMP_TYPE;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_SPATIAL_DIMS = CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_POST_PADDINGS = CUDNN_ATTR_RESAMPLE_POST_PADDINGS;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_PRE_PADDINGS = CUDNN_ATTR_RESAMPLE_PRE_PADDINGS;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_STRIDES = CUDNN_ATTR_RESAMPLE_STRIDES;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_WINDOW_DIMS = CUDNN_ATTR_RESAMPLE_WINDOW_DIMS;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_NAN_PROPAGATION = CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION;
  cudnnBackendAttributeName_t ATTR_RESAMPLE_PADDING_MODE = CUDNN_ATTR_RESAMPLE_PADDING_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_XDESC = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_YDESC = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_ALPHA = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_BETA = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_FWD_DESC = CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DXDESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DYDESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_ALPHA = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_BETA = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_DESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC;

  // CHECK: miopenPointwiseMode_t POINTWISE_ADD_SQUARE = MIOPEN_POINTWISE_ADD_SQUARE;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_DIV = MIOPEN_POINTWISE_DIV;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_MOD = MIOPEN_POINTWISE_MOD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_POW = MIOPEN_POINTWISE_POW;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SUB = MIOPEN_POINTWISE_SUB;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_ABS = MIOPEN_POINTWISE_ABS;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CEIL = MIOPEN_POINTWISE_CEIL;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_COS = MIOPEN_POINTWISE_COS;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_EXP = MIOPEN_POINTWISE_EXP;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_FLOOR = MIOPEN_POINTWISE_FLOOR;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_LOG = MIOPEN_POINTWISE_LOG;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_NEG = MIOPEN_POINTWISE_NEG;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_RSQRT = MIOPEN_POINTWISE_RSQRT;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_SIN = MIOPEN_POINTWISE_SIN;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_TAN = MIOPEN_POINTWISE_TAN;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_EQ = MIOPEN_POINTWISE_CMP_EQ;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_NEQ = MIOPEN_POINTWISE_CMP_NEQ;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_GT = MIOPEN_POINTWISE_CMP_GT;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_GE = MIOPEN_POINTWISE_CMP_GE;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_LT = MIOPEN_POINTWISE_CMP_LT;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_CMP_LE = MIOPEN_POINTWISE_CMP_LE;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_LOGICAL_AND = MIOPEN_POINTWISE_LOGICAL_AND;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_LOGICAL_OR = MIOPEN_POINTWISE_LOGICAL_OR;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_LOGICAL_NOT = MIOPEN_POINTWISE_LOGICAL_NOT;
  cudnnPointwiseMode_t POINTWISE_ADD_SQUARE = CUDNN_POINTWISE_ADD_SQUARE;
  cudnnPointwiseMode_t POINTWISE_DIV = CUDNN_POINTWISE_DIV;
  cudnnPointwiseMode_t POINTWISE_MOD = CUDNN_POINTWISE_MOD;
  cudnnPointwiseMode_t POINTWISE_POW = CUDNN_POINTWISE_POW;
  cudnnPointwiseMode_t POINTWISE_SUB = CUDNN_POINTWISE_SUB;
  cudnnPointwiseMode_t POINTWISE_ABS = CUDNN_POINTWISE_ABS;
  cudnnPointwiseMode_t POINTWISE_CEIL = CUDNN_POINTWISE_CEIL;
  cudnnPointwiseMode_t POINTWISE_COS = CUDNN_POINTWISE_COS;
  cudnnPointwiseMode_t POINTWISE_EXP = CUDNN_POINTWISE_EXP;
  cudnnPointwiseMode_t POINTWISE_FLOOR = CUDNN_POINTWISE_FLOOR;
  cudnnPointwiseMode_t POINTWISE_LOG = CUDNN_POINTWISE_LOG;
  cudnnPointwiseMode_t POINTWISE_NEG = CUDNN_POINTWISE_NEG;
  cudnnPointwiseMode_t POINTWISE_RSQRT = CUDNN_POINTWISE_RSQRT;
  cudnnPointwiseMode_t POINTWISE_SIN = CUDNN_POINTWISE_SIN;
  cudnnPointwiseMode_t POINTWISE_TAN = CUDNN_POINTWISE_TAN;
  cudnnPointwiseMode_t POINTWISE_CMP_EQ = CUDNN_POINTWISE_CMP_EQ;
  cudnnPointwiseMode_t POINTWISE_CMP_NEQ = CUDNN_POINTWISE_CMP_NEQ;
  cudnnPointwiseMode_t POINTWISE_CMP_GT = CUDNN_POINTWISE_CMP_GT;
  cudnnPointwiseMode_t POINTWISE_CMP_GE = CUDNN_POINTWISE_CMP_GE;
  cudnnPointwiseMode_t POINTWISE_CMP_LT = CUDNN_POINTWISE_CMP_LT;
  cudnnPointwiseMode_t POINTWISE_CMP_LE = CUDNN_POINTWISE_CMP_LE;
  cudnnPointwiseMode_t POINTWISE_LOGICAL_AND = CUDNN_POINTWISE_LOGICAL_AND;
  cudnnPointwiseMode_t POINTWISE_LOGICAL_OR = CUDNN_POINTWISE_LOGICAL_OR;
  cudnnPointwiseMode_t POINTWISE_LOGICAL_NOT = CUDNN_POINTWISE_LOGICAL_NOT;

  // CHECK: miopenBackendHeurMode_t HEUR_MODE_FALLBACK = MIOPEN_HEUR_MODE_FALLBACK;
  // CHECK: miopenBackendHeurMode_t HEUR_MODE_A = MIOPEN_HEUR_MODE_A;
  cudnnBackendHeurMode_t HEUR_MODE_FALLBACK = CUDNN_HEUR_MODE_FALLBACK;
  cudnnBackendHeurMode_t HEUR_MODE_A = CUDNN_HEUR_MODE_A;

  // CHECK: miopenPaddingMode_t PaddingMode_t;
  // CHECK-NEXT: miopenPaddingMode_t ZERO_PAD = miopenPaddingDefault;
  // CHECK-NEXT: miopenPaddingMode_t NEG_INF_PAD = miopenPaddingSame;
  // CHECK-NEXT: miopenPaddingMode_t EDGE_VAL_PAD = miopenPaddingValid;
  cudnnPaddingMode_t PaddingMode_t;
  cudnnPaddingMode_t ZERO_PAD = CUDNN_ZERO_PAD;
  cudnnPaddingMode_t NEG_INF_PAD = CUDNN_NEG_INF_PAD;
  cudnnPaddingMode_t EDGE_VAL_PAD = CUDNN_EDGE_VAL_PAD;
#endif

#if CUDNN_VERSION >= 8400
  // CHECK: miopenBackendAttributeType_t TYPE_CHAR = MIOPEN_TYPE_CHAR;
  cudnnBackendAttributeType_t TYPE_CHAR = CUDNN_TYPE_CHAR;

  // CHECK: miopenBackendAttributeName_t ATTR_POINTWISE_AXIS = MIOPEN_ATTR_POINTWISE_AXIS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = MIOPEN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION;
  cudnnBackendAttributeName_t ATTR_POINTWISE_AXIS = CUDNN_ATTR_POINTWISE_AXIS;
  cudnnBackendAttributeName_t ATTR_EXECUTION_PLAN_JSON_REPRESENTATION = CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION;

  // CHECK: miopenPointwiseMode_t POINTWISE_GEN_INDEX = MIOPEN_POINTWISE_GEN_INDEX;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_BINARY_SELECT = MIOPEN_POINTWISE_BINARY_SELECT;
  cudnnPointwiseMode_t POINTWISE_GEN_INDEX = CUDNN_POINTWISE_GEN_INDEX;
  cudnnPointwiseMode_t POINTWISE_BINARY_SELECT = CUDNN_POINTWISE_BINARY_SELECT;
#endif

#if CUDNN_VERSION >= 8500
  // CHECK: miopenBackendDescriptorType_t BACKEND_OPERATION_CONCAT_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_SIGNAL_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_CONCAT_DESCRIPTOR = CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_SIGNAL_DESCRIPTOR = CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;

  // CHECK: miopenBackendAttributeType_t TYPE_SIGNAL_MODE = MIOPEN_TYPE_SIGNAL_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_FRACTION = MIOPEN_TYPE_FRACTION;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_NORM_MODE = MIOPEN_TYPE_NORM_MODE;
  // CHECK-NEXT: miopenBackendAttributeType_t TYPE_NORM_FWD_PHASE = MIOPEN_TYPE_NORM_FWD_PHASE;
  cudnnBackendAttributeType_t TYPE_SIGNAL_MODE = CUDNN_TYPE_SIGNAL_MODE;
  cudnnBackendAttributeType_t TYPE_FRACTION = CUDNN_TYPE_FRACTION;
  cudnnBackendAttributeType_t TYPE_NORM_MODE = CUDNN_TYPE_NORM_MODE;
  cudnnBackendAttributeType_t TYPE_NORM_FWD_PHASE = CUDNN_TYPE_NORM_FWD_PHASE;

  // CHECK: miopenBackendAttributeName_t ATTR_OPERATION_CONCAT_AXIS = MIOPEN_ATTR_OPERATION_CONCAT_AXIS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONCAT_INPUT_DESCS = MIOPEN_ATTR_OPERATION_CONCAT_INPUT_DESCS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONCAT_INPLACE_INDEX = MIOPEN_ATTR_OPERATION_CONCAT_INPLACE_INDEX;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_CONCAT_OUTPUT_DESC = MIOPEN_ATTR_OPERATION_CONCAT_OUTPUT_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_SIGNAL_MODE = MIOPEN_ATTR_OPERATION_SIGNAL_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_SIGNAL_FLAGDESC = MIOPEN_ATTR_OPERATION_SIGNAL_FLAGDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_SIGNAL_VALUE = MIOPEN_ATTR_OPERATION_SIGNAL_VALUE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_SIGNAL_XDESC = MIOPEN_ATTR_OPERATION_SIGNAL_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_SIGNAL_YDESC = MIOPEN_ATTR_OPERATION_SIGNAL_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_MODE = MIOPEN_ATTR_OPERATION_NORM_FWD_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_PHASE = MIOPEN_ATTR_OPERATION_NORM_FWD_PHASE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_XDESC = MIOPEN_ATTR_OPERATION_NORM_FWD_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_MEAN_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_SCALE_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_BIAS_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_BIAS_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_EPSILON_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = MIOPEN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_YDESC = MIOPEN_ATTR_OPERATION_NORM_FWD_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = MIOPEN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_MODE = MIOPEN_ATTR_OPERATION_NORM_BWD_MODE;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_XDESC = MIOPEN_ATTR_OPERATION_NORM_BWD_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_MEAN_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_MEAN_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DYDESC = MIOPEN_ATTR_OPERATION_NORM_BWD_DYDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_SCALE_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_SCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_EPSILON_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DSCALE_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DBIAS_DESC = MIOPEN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DXDESC = MIOPEN_ATTR_OPERATION_NORM_BWD_DXDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = MIOPEN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONCAT_AXIS = CUDNN_ATTR_OPERATION_CONCAT_AXIS;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONCAT_INPUT_DESCS = CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONCAT_INPLACE_INDEX = CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX;
  cudnnBackendAttributeName_t ATTR_OPERATION_CONCAT_OUTPUT_DESC = CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_SIGNAL_MODE = CUDNN_ATTR_OPERATION_SIGNAL_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_SIGNAL_FLAGDESC = CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_SIGNAL_VALUE = CUDNN_ATTR_OPERATION_SIGNAL_VALUE;
  cudnnBackendAttributeName_t ATTR_OPERATION_SIGNAL_XDESC = CUDNN_ATTR_OPERATION_SIGNAL_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_SIGNAL_YDESC = CUDNN_ATTR_OPERATION_SIGNAL_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_MODE = CUDNN_ATTR_OPERATION_NORM_FWD_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_PHASE = CUDNN_ATTR_OPERATION_NORM_FWD_PHASE;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_XDESC = CUDNN_ATTR_OPERATION_NORM_FWD_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_MEAN_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_SCALE_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_BIAS_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_EPSILON_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_YDESC = CUDNN_ATTR_OPERATION_NORM_FWD_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS = CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_MODE = CUDNN_ATTR_OPERATION_NORM_BWD_MODE;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_XDESC = CUDNN_ATTR_OPERATION_NORM_BWD_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_MEAN_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DYDESC = CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_SCALE_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_EPSILON_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DSCALE_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DBIAS_DESC = CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_DXDESC = CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS = CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS;

  // CHECK: miopenPointwiseMode_t POINTWISE_ERF = MIOPEN_POINTWISE_ERF;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_IDENTITY = MIOPEN_POINTWISE_IDENTITY;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_GELU_APPROX_TANH_FWD = MIOPEN_POINTWISE_GELU_APPROX_TANH_FWD;
  // CHECK-NEXT: miopenPointwiseMode_t POINTWISE_GELU_APPROX_TANH_BWD = MIOPEN_POINTWISE_GELU_APPROX_TANH_BWD;
  cudnnPointwiseMode_t POINTWISE_ERF = CUDNN_POINTWISE_ERF;
  cudnnPointwiseMode_t POINTWISE_IDENTITY = CUDNN_POINTWISE_IDENTITY;
  cudnnPointwiseMode_t POINTWISE_GELU_APPROX_TANH_FWD = CUDNN_POINTWISE_GELU_APPROX_TANH_FWD;
  cudnnPointwiseMode_t POINTWISE_GELU_APPROX_TANH_BWD = CUDNN_POINTWISE_GELU_APPROX_TANH_BWD;
#endif

#if CUDNN_VERSION >= 8600
  // CHECK: miopenDataType_t DATA_FP8_E4M3 = miopenFloat8;
  // CHECK-NEXT: miopenDataType_t DATA_FP8_E5M2 = miopenBFloat8;
  cudnnDataType_t DATA_FP8_E4M3 = CUDNN_DATA_FP8_E4M3;
  cudnnDataType_t DATA_FP8_E5M2 = CUDNN_DATA_FP8_E5M2;
#endif

#if CUDNN_VERSION >= 8700
  // CHECK: miopenBackendDescriptorType_t BACKEND_RNG_DESCRIPTOR = MIOPEN_BACKEND_RNG_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_RNG_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_RNG_DESCRIPTOR;
  // CHECK-NEXT: miopenBackendDescriptorType_t BACKEND_OPERATION_RESHAPE_DESCRIPTOR = MIOPEN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_RNG_DESCRIPTOR = CUDNN_BACKEND_RNG_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_RNG_DESCRIPTOR = CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
  cudnnBackendDescriptorType_t BACKEND_OPERATION_RESHAPE_DESCRIPTOR = CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;

  // CHECK: miopenBackendAttributeType_t TYPE_RNG_DISTRIBUTION = MIOPEN_TYPE_RNG_DISTRIBUTION;
  cudnnBackendAttributeType_t TYPE_RNG_DISTRIBUTION = CUDNN_TYPE_RNG_DISTRIBUTION;

  // CHECK: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = MIOPEN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = MIOPEN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = MIOPEN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_XDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_YDESC = MIOPEN_ATTR_OPERATION_RESAMPLE_BWD_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESHAPE_XDESC = MIOPEN_ATTR_OPERATION_RESHAPE_XDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RESHAPE_YDESC = MIOPEN_ATTR_OPERATION_RESHAPE_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_DISTRIBUTION = MIOPEN_ATTR_RNG_DISTRIBUTION;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_NORMAL_DIST_MEAN = MIOPEN_ATTR_RNG_NORMAL_DIST_MEAN;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = MIOPEN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_UNIFORM_DIST_MAXIMUM = MIOPEN_ATTR_RNG_UNIFORM_DIST_MAXIMUM;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_UNIFORM_DIST_MINIMUM = MIOPEN_ATTR_RNG_UNIFORM_DIST_MINIMUM;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_RNG_BERNOULLI_DIST_PROBABILITY = MIOPEN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RNG_YDESC = MIOPEN_ATTR_OPERATION_RNG_YDESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RNG_SEED = MIOPEN_ATTR_OPERATION_RNG_SEED;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_OPERATION_RNG_DESC = MIOPEN_ATTR_OPERATION_RNG_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC = CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC = CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC = CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_XDESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESAMPLE_BWD_YDESC = CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESHAPE_XDESC = CUDNN_ATTR_OPERATION_RESHAPE_XDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RESHAPE_YDESC = CUDNN_ATTR_OPERATION_RESHAPE_YDESC;
  cudnnBackendAttributeName_t ATTR_RNG_DISTRIBUTION = CUDNN_ATTR_RNG_DISTRIBUTION;
  cudnnBackendAttributeName_t ATTR_RNG_NORMAL_DIST_MEAN = CUDNN_ATTR_RNG_NORMAL_DIST_MEAN;
  cudnnBackendAttributeName_t ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION;
  cudnnBackendAttributeName_t ATTR_RNG_UNIFORM_DIST_MAXIMUM = CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM;
  cudnnBackendAttributeName_t ATTR_RNG_UNIFORM_DIST_MINIMUM = CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM;
  cudnnBackendAttributeName_t ATTR_RNG_BERNOULLI_DIST_PROBABILITY = CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY;
  cudnnBackendAttributeName_t ATTR_OPERATION_RNG_YDESC = CUDNN_ATTR_OPERATION_RNG_YDESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RNG_SEED = CUDNN_ATTR_OPERATION_RNG_SEED;
  cudnnBackendAttributeName_t ATTR_OPERATION_RNG_DESC = CUDNN_ATTR_OPERATION_RNG_DESC;

  // CHECK: miopenRngDistribution_t rngDistribution_t;
  // CHECK-NEXT: miopenRngDistribution_t RNG_DISTRIBUTION_BERNOULLI = MIOPEN_RNG_DISTRIBUTION_BERNOULLI;
  // CHECK-NEXT: miopenRngDistribution_t RNG_DISTRIBUTION_UNIFORM = MIOPEN_RNG_DISTRIBUTION_UNIFORM;
  // CHECK-NEXT: miopenRngDistribution_t RNG_DISTRIBUTION_NORMAL = MIOPEN_RNG_DISTRIBUTION_NORMAL;
  cudnnRngDistribution_t rngDistribution_t;
  cudnnRngDistribution_t RNG_DISTRIBUTION_BERNOULLI = CUDNN_RNG_DISTRIBUTION_BERNOULLI;
  cudnnRngDistribution_t RNG_DISTRIBUTION_UNIFORM = CUDNN_RNG_DISTRIBUTION_UNIFORM;
  cudnnRngDistribution_t RNG_DISTRIBUTION_NORMAL = CUDNN_RNG_DISTRIBUTION_NORMAL;
#endif

#if CUDNN_VERSION >= 8800
  // CHECK: miopenBackendAttributeName_t ATTR_OPERATION_RNG_OFFSET_DESC = MIOPEN_ATTR_OPERATION_RNG_OFFSET_DESC;
  cudnnBackendAttributeName_t ATTR_OPERATION_RNG_OFFSET_DESC = CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC;
#endif

#if CUDNN_VERSION >= 8900
  // CHECK: miopenBackendAttributeName_t ATTR_TENSOR_RAGGED_OFFSET_DESC = MIOPEN_ATTR_TENSOR_RAGGED_OFFSET_DESC;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_MATMUL_PADDING_VALUE = MIOPEN_ATTR_MATMUL_PADDING_VALUE;
  cudnnBackendAttributeName_t ATTR_TENSOR_RAGGED_OFFSET_DESC = CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC;
  cudnnBackendAttributeName_t ATTR_MATMUL_PADDING_VALUE = CUDNN_ATTR_MATMUL_PADDING_VALUE;

  // CHECK: miopenPointwiseMode_t POINTWISE_RECIPROCAL = MIOPEN_POINTWISE_RECIPROCAL;
  cudnnPointwiseMode_t POINTWISE_RECIPROCAL = CUDNN_POINTWISE_RECIPROCAL;
#endif

#if CUDNN_VERSION >= 8905
  // CHECK: miopenBackendAttributeName_t ATTR_ENGINEHEUR_SM_COUNT_TARGET = MIOPEN_ATTR_ENGINEHEUR_SM_COUNT_TARGET;
  // CHECK-NEXT: miopenBackendAttributeName_t ATTR_ENGINE_SM_COUNT_TARGET = MIOPEN_ATTR_ENGINE_SM_COUNT_TARGET;
  cudnnBackendAttributeName_t ATTR_ENGINEHEUR_SM_COUNT_TARGET = CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET;
  cudnnBackendAttributeName_t ATTR_ENGINE_SM_COUNT_TARGET = CUDNN_ATTR_ENGINE_SM_COUNT_TARGET;
#endif

  return 0;
}
