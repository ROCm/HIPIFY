/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_DNN_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  // NOTE: MIOPEN_EXPORT miopenStatus_t miopenGetVersion(size_t* major, size_t* minor, size_t* patch) and size_t CUDNNWINAPI cudnnGetVersion(void) have different signatures
  m["cudnnGetVersion"]                                          = {"hipdnnGetVersion",                                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};
  m["cudnnGetCudartVersion"]                                    = {"hipdnnGetCudartVersion",                                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetMaxDeviceVersion"]                                 = {"hipdnnGetMaxDeviceVersion",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnQueryRuntimeError"]                                   = {"hipdnnQueryRuntimeError",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetProperty"]                                         = {"hipdnnGetProperty",                                         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetErrorString"]                                      = {"hipdnnGetErrorString",                                      "miopenGetErrorString",                                               CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetLastErrorString"]                                  = {"hipdnnGetLastErrorString",                                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnIm2Col"]                                              = {"hipdnnIm2Col",                                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnCreate"]                                              = {"hipdnnCreate",                                              "miopenCreate",                                                       CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDestroy"]                                             = {"hipdnnDestroy",                                             "miopenDestroy",                                                      CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSetStream"]                                           = {"hipdnnSetStream",                                           "miopenSetStream",                                                    CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetStream"]                                           = {"hipdnnGetStream",                                           "miopenGetStream",                                                    CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSetCallback"]                                         = {"hipdnnSetCallback",                                         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetCallback"]                                         = {"hipdnnGetCallback",                                         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnAdvInferVersionCheck"]                                = {"hipdnnAdvInferVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_REMOVED};
  m["cudnnAdvVersionCheck"]                                     = {"hipdnnAdvVersionCheck",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnAdvTrainVersionCheck"]                                = {"hipdnnAdvTrainVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_REMOVED};
  m["cudnnCnnInferVersionCheck"]                                = {"hipdnnCnnInferVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnCnnTrainVersionCheck"]                                = {"hipdnnCnnTrainVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnOpsInferVersionCheck"]                                = {"hipdnnOpsInferVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnOpsTrainVersionCheck"]                                = {"hipdnnOpsTrainVersionCheck",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_REMOVED};
  m["cudnnGraphVersionCheck"]                                   = {"hipdnnGraphVersionCheck",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnOpsVersionCheck"]                                     = {"hipdnnOpsVersionCheck",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Tensor functions
  m["cudnnCreateTensorDescriptor"]                              = {"hipdnnCreateTensorDescriptor",                              "miopenCreateTensorDescriptor",                                       CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSetTensor4dDescriptor"]                               = {"hipdnnSetTensor4dDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};
  m["cudnnSetTensor4dDescriptorEx"]                             = {"hipdnnSetTensor4dDescriptorEx",                             "miopenSet4dTensorDescriptorEx",                                      CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetTensor4dDescriptor"]                               = {"hipdnnGetTensor4dDescriptor",                               "miopenGet4dTensorDescriptor",                                        CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSetTensorNdDescriptor"]                               = {"hipdnnSetTensorNdDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};
  m["cudnnSetTensorNdDescriptorEx"]                             = {"hipdnnSetTensorNdDescriptorEx",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetTensorNdDescriptor"]                               = {"hipdnnGetTensorNdDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};
  m["cudnnGetTensorSizeInBytes"]                                = {"hipdnnGetTensorSizeInBytes",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnDestroyTensorDescriptor"]                             = {"hipdnnDestroyTensorDescriptor",                             "miopenDestroyTensorDescriptor",                                      CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnTransformTensor"]                                     = {"hipdnnTransformTensor",                                     "miopenTransformTensor",                                              CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnTransformTensorEx"]                                   = {"hipdnnTransformTensorEx",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnInitTransformDest"]                                   = {"hipdnnInitTransformDest",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnCreateTensorTransformDescriptor"]                     = {"hipdnnCreateTensorTransformDescriptor",                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetTensorTransformDescriptor"]                        = {"hipdnnSetTensorTransformDescriptor",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetTensorTransformDescriptor"]                        = {"hipdnnGetTensorTransformDescriptor",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyTensorTransformDescriptor"]                    = {"hipdnnDestroyTensorTransformDescriptor",                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnAddTensor"]                                           = {"hipdnnAddTensor",                                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnCreateOpTensorDescriptor"]                            = {"hipdnnCreateOpTensorDescriptor",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetOpTensorDescriptor"]                               = {"hipdnnSetOpTensorDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetOpTensorDescriptor"]                               = {"hipdnnGetOpTensorDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyOpTensorDescriptor"]                           = {"hipdnnDestroyOpTensorDescriptor",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnOpTensor"]                                            = {"hipdnnOpTensor",                                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFoldedConvBackwardDataDescriptors"]                = {"hipdnnGetFoldedConvBackwardDataDescriptors",                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Reduce Tensor functions
  m["cudnnCreateReduceTensorDescriptor"]                        = {"hipdnnCreateReduceTensorDescriptor",                        "miopenCreateReduceTensorDescriptor",                                 CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetReduceTensorDescriptor"]                           = {"hipdnnSetReduceTensorDescriptor",                           "miopenSetReduceTensorDescriptor",                                    CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetReduceTensorDescriptor"]                           = {"hipdnnGetReduceTensorDescriptor",                           "miopenGetReduceTensorDescriptor",                                    CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnDestroyReduceTensorDescriptor"]                       = {"hipdnnDestroyReduceTensorDescriptor",                       "miopenDestroyReduceTensorDescriptor",                                CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetReductionIndicesSize"]                             = {"hipdnnGetReductionIndicesSize",                             "miopenGetReductionIndicesSize",                                      CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetReductionWorkspaceSize"]                           = {"hipdnnGetReductionWorkspaceSize",                           "miopenGetReductionWorkspaceSize",                                    CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnReduceTensor"]                                        = {"hipdnnReduceTensor",                                        "miopenReduceTensor",                                                 CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetTensor"]                                           = {"hipdnnSetTensor",                                           "miopenSetTensor",                                                    CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnScaleTensor"]                                         = {"hipdnnScaleTensor",                                         "miopenScaleTensor",                                                  CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnDeriveNormTensorDescriptor"]                          = {"hipdnnDeriveNormTensorDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Filter functions
  m["cudnnCreateFilterDescriptor"]                              = {"hipdnnCreateFilterDescriptor",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetFilter4dDescriptor"]                               = {"hipdnnSetFilter4dDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFilter4dDescriptor"]                               = {"hipdnnGetFilter4dDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetFilterNdDescriptor"]                               = {"hipdnnSetFilterNdDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFilterNdDescriptor"]                               = {"hipdnnGetFilterNdDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFilterSizeInBytes"]                                = {"hipdnnGetFilterSizeInBytes",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnTransformFilter"]                                     = {"hipdnnTransformFilter",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyFilterDescriptor"]                             = {"hipdnnDestroyFilterDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnReorderFilterAndBias"]                                = {"hipdnnReorderFilterAndBias",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Convolution functions
  m["cudnnCreateConvolutionDescriptor"]                         = {"hipdnnCreateConvolutionDescriptor",                         "miopenCreateConvolutionDescriptor",                                  CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetConvolutionMathType"]                              = {"hipdnnSetConvolutionMathType",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionMathType"]                              = {"hipdnnGetConvolutionMathType",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetConvolutionGroupCount"]                            = {"hipdnnSetConvolutionGroupCount",                            "miopenSetConvolutionGroupCount",                                     CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetConvolutionGroupCount"]                            = {"hipdnnGetConvolutionGroupCount",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetConvolutionReorderType"]                           = {"hipdnnSetConvolutionReorderType",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionReorderType"]                           = {"hipdnnGetConvolutionReorderType",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetConvolution2dDescriptor"]                          = {"hipdnnSetConvolution2dDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolution2dDescriptor"]                          = {"hipdnnGetConvolution2dDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolution2dForwardOutputDim"]                    = {"hipdnnGetConvolution2dForwardOutputDim",                    "miopenGetConvolutionForwardOutputDim",                               CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetConvolutionNdDescriptor"]                          = {"hipdnnSetConvolutionNdDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionNdDescriptor"]                          = {"hipdnnGetConvolutionNdDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionNdForwardOutputDim"]                    = {"hipdnnGetConvolutionNdForwardOutputDim",                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyConvolutionDescriptor"]                        = {"hipdnnDestroyConvolutionDescriptor",                        "miopenDestroyConvolutionDescriptor",                                 CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetConvolutionForwardAlgorithmMaxCount"]              = {"hipdnnGetConvolutionForwardAlgorithmMaxCount",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFindConvolutionForwardAlgorithm"]                     = {"hipdnnFindConvolutionForwardAlgorithm",                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFindConvolutionForwardAlgorithmEx"]                   = {"hipdnnFindConvolutionForwardAlgorithmEx",                   "miopenFindConvolutionForwardAlgorithm",                              CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetConvolutionForwardAlgorithm"]                      = {"hipdnnGetConvolutionForwardAlgorithm",                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetConvolutionForwardAlgorithm_v7"]                   = {"hipdnnGetConvolutionForwardAlgorithm_v7",                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionForwardWorkspaceSize"]                  = {"hipdnnGetConvolutionForwardWorkspaceSize",                  "miopenConvolutionForwardGetWorkSpaceSize",                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnConvolutionForward"]                                  = {"hipdnnConvolutionForward",                                  "miopenConvolutionForward",                                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnConvolutionBiasActivationForward"]                    = {"hipdnnConvolutionBiasActivationForward",                    "miopenConvolutionBiasActivationForward",                             CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnConvolutionBackwardBias"]                             = {"hipdnnConvolutionBackwardBias",                             "miopenConvolutionBackwardBias",                                      CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"]       = {"hipdnnGetConvolutionBackwardFilterAlgorithmMaxCount",       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnFindConvolutionBackwardFilterAlgorithm"]              = {"hipdnnFindConvolutionBackwardFilterAlgorithm",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFindConvolutionBackwardFilterAlgorithmEx"]            = {"hipdnnFindConvolutionBackwardFilterAlgorithmEx",            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardFilterAlgorithm"]               = {"hipdnnGetConvolutionBackwardFilterAlgorithm",               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetConvolutionBackwardFilterAlgorithm_v7"]            = {"hipdnnGetConvolutionBackwardFilterAlgorithm_v7",            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardFilterWorkspaceSize"]           = {"hipdnnGetConvolutionBackwardFilterWorkspaceSize",           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnConvolutionBackwardFilter"]                           = {"hipdnnConvolutionBackwardFilter",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardDataAlgorithmMaxCount"]         = {"hipdnnGetConvolutionBackwardDataAlgorithmMaxCount",         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFindConvolutionBackwardDataAlgorithm"]                = {"hipdnnFindConvolutionBackwardDataAlgorithm",                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFindConvolutionBackwardDataAlgorithmEx"]              = {"hipdnnFindConvolutionBackwardDataAlgorithmEx",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardDataAlgorithm"]                 = {"hipdnnGetConvolutionBackwardDataAlgorithm",                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetConvolutionBackwardDataAlgorithm_v7"]              = {"hipdnnGetConvolutionBackwardDataAlgorithm_v7",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetConvolutionBackwardDataWorkspaceSize"]             = {"hipdnnGetConvolutionBackwardDataWorkspaceSize",             "miopenConvolutionBackwardDataGetWorkSpaceSize",                      CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnConvolutionBackwardData"]                             = {"hipdnnConvolutionBackwardData",                             "miopenConvolutionBackwardData",                                      CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};

  // cuDNN Softmax functions
  m["cudnnSoftmaxForward"]                                      = {"hipdnnSoftmaxForward",                                      "miopenSoftmaxForward_V2",                                            CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSoftmaxBackward"]                                     = {"hipdnnSoftmaxBackward",                                     "miopenSoftmaxBackward_V2",                                           CONV_LIB_FUNC, API_DNN, 2};

  // cuDNN Pooling functions
  m["cudnnCreatePoolingDescriptor"]                             = {"hipdnnCreatePoolingDescriptor",                             "miopenCreatePoolingDescriptor",                                      CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetPooling2dDescriptor"]                              = {"hipdnnSetPooling2dDescriptor",                              "miopenSet2dPoolingDescriptor",                                       CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetPooling2dDescriptor"]                              = {"hipdnnGetPooling2dDescriptor",                              "miopenGet2dPoolingDescriptor",                                       CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetPoolingNdDescriptor"]                              = {"hipdnnSetPoolingNdDescriptor",                              "miopenSetNdPoolingDescriptor",                                       CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnGetPoolingNdDescriptor"]                              = {"hipdnnGetPoolingNdDescriptor",                              "miopenGetNdPoolingDescriptor",                                       CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetPoolingNdForwardOutputDim"]                        = {"hipdnnGetPoolingNdForwardOutputDim",                        "miopenGetPoolingNdForwardOutputDim",                                 CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetPooling2dForwardOutputDim"]                        = {"hipdnnGetPooling2dForwardOutputDim",                        "miopenGetPoolingForwardOutputDim",                                   CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnDestroyPoolingDescriptor"]                            = {"hipdnnDestroyPoolingDescriptor",                            "miopenDestroyPoolingDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnPoolingForward"]                                      = {"hipdnnPoolingForward",                                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnPoolingBackward"]                                     = {"hipdnnPoolingBackward",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Activation functions
  m["cudnnCreateActivationDescriptor"]                          = {"hipdnnCreateActivationDescriptor",                          "miopenCreateActivationDescriptor",                                   CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetActivationDescriptor"]                             = {"hipdnnSetActivationDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetActivationDescriptor"]                             = {"hipdnnGetActivationDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyActivationDescriptor"]                         = {"hipdnnDestroyActivationDescriptor",                         "miopenDestroyActivationDescriptor",                                  CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnActivationForward"]                                   = {"hipdnnActivationForward",                                   "miopenActivationForward",                                            CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnActivationBackward"]                                  = {"hipdnnActivationBackward",                                  "miopenActivationBackward",                                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnSetActivationDescriptorSwishBeta"]                    = {"hipdnnSetActivationDescriptorSwishBeta",                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetActivationDescriptorSwishBeta"]                    = {"hipdnnGetActivationDescriptorSwishBeta",                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN LRN functions
  m["cudnnCreateLRNDescriptor"]                                 = {"hipdnnCreateLRNDescriptor",                                 "miopenCreateLRNDescriptor",                                          CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnSetLRNDescriptor"]                                    = {"hipdnnSetLRNDescriptor",                                    "miopenSetLRNDescriptor",                                             CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetLRNDescriptor"]                                    = {"hipdnnGetLRNDescriptor",                                    "miopenGetLRNDescriptor",                                             CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDestroyLRNDescriptor"]                                = {"hipdnnDestroyLRNDescriptor",                                "miopenDestroyLRNDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnLRNCrossChannelForward"]                              = {"hipdnnLRNCrossChannelForward",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};
  m["cudnnLRNCrossChannelBackward"]                             = {"hipdnnLRNCrossChannelBackward",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED};

  // cuDNN Divisive Normalization functions
  m["cudnnDivisiveNormalizationForward"]                        = {"hipdnnDivisiveNormalizationForward",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnDivisiveNormalizationBackward"]                       = {"hipdnnDivisiveNormalizationBackward",                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Batch Normalization functions
  m["cudnnDeriveBNTensorDescriptor"]                            = {"hipdnnDeriveBNTensorDescriptor",                            "miopenDeriveBNTensorDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnBatchNormalizationForwardTraining"]                   = {"hipdnnBatchNormalizationForwardTraining",                   "miopenBatchNormalizationForwardTraining",                            CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnBatchNormalizationForwardTrainingEx"]                 = {"hipdnnBatchNormalizationForwardTrainingEx",                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnBatchNormalizationForwardInference"]                  = {"hipdnnBatchNormalizationForwardInference",                  "miopenBatchNormalizationForwardInference",                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnBatchNormalizationBackward"]                          = {"hipdnnBatchNormalizationBackward",                          "miopenBatchNormalizationBackward",                                   CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED};
  m["cudnnBatchNormalizationBackwardEx"]                        = {"hipdnnBatchNormalizationBackwardEx",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize"] = {"hipdnnGetBatchNormalizationForwardTrainingExWorkspaceSize", "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetBatchNormalizationBackwardExWorkspaceSize"]        = {"hipdnnGetBatchNormalizationBackwardExWorkspaceSize",        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetBatchNormalizationTrainingExReserveSpaceSize"]     = {"hipdnnGetBatchNormalizationTrainingExReserveSpaceSize",     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnNormalizationForwardInference"]                       = {"hipdnnNormalizationForwardInference",                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetNormalizationForwardTrainingWorkspaceSize"]        = {"hipdnnGetNormalizationForwardTrainingWorkspaceSize",        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetNormalizationBackwardWorkspaceSize"]               = {"hipdnnGetNormalizationBackwardWorkspaceSize",               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetNormalizationTrainingReserveSpaceSize"]            = {"hipdnnGetNormalizationTrainingReserveSpaceSize",            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnNormalizationForwardTraining"]                        = {"hipdnnNormalizationForwardTraining",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnNormalizationBackward"]                               = {"hipdnnNormalizationBackward",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Spatial Transformer functions
  m["cudnnCreateSpatialTransformerDescriptor"]                  = {"hipdnnCreateSpatialTransformerDescriptor",                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetSpatialTransformerNdDescriptor"]                   = {"hipdnnSetSpatialTransformerNdDescriptor",                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnDestroySpatialTransformerDescriptor"]                 = {"hipdnnDestroySpatialTransformerDescriptor",                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSpatialTfGridGeneratorForward"]                       = {"hipdnnSpatialTfGridGeneratorForward",                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSpatialTfGridGeneratorBackward"]                      = {"hipdnnSpatialTfGridGeneratorBackward",                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSpatialTfSamplerForward"]                             = {"hipdnnSpatialTfSamplerForward",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSpatialTfSamplerBackward"]                            = {"hipdnnSpatialTfSamplerBackward",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Dropout functions
  m["cudnnCreateDropoutDescriptor"]                             = {"hipdnnCreateDropoutDescriptor",                             "miopenCreateDropoutDescriptor",                                      CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDestroyDropoutDescriptor"]                            = {"hipdnnDestroyDropoutDescriptor",                            "miopenDestroyDropoutDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDropoutGetStatesSize"]                                = {"hipdnnDropoutGetStatesSize",                                "miopenDropoutGetStatesSize",                                         CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDropoutGetReserveSpaceSize"]                          = {"hipdnnDropoutGetReserveSpaceSize",                          "miopenDropoutGetReserveSpaceSize",                                   CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnSetDropoutDescriptor"]                                = {"hipdnnSetDropoutDescriptor",                                "miopenSetDropoutDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetDropoutDescriptor"]                                = {"hipdnnGetDropoutDescriptor",                                "miopenGetDropoutDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnRestoreDropoutDescriptor"]                            = {"hipdnnRestoreDropoutDescriptor",                            "miopenRestoreDropoutDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnDropoutForward"]                                      = {"hipdnnDropoutForward",                                      "miopenDropoutForward",                                               CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnDropoutBackward"]                                     = {"hipdnnDropoutBackward",                                     "miopenDropoutBackward",                                              CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};

  // cuDNN RNN functions
  m["cudnnCreateRNNDescriptor"]                                 = {"hipdnnCreateRNNDescriptor",                                 "miopenCreateRNNDescriptor",                                          CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnDestroyRNNDescriptor"]                                = {"hipdnnDestroyRNNDescriptor",                                "miopenDestroyRNNDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2};
  m["cudnnGetRNNForwardInferenceAlgorithmMaxCount"]             = {"hipdnnGetRNNForwardInferenceAlgorithmMaxCount",             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnFindRNNForwardInferenceAlgorithmEx"]                  = {"hipdnnFindRNNForwardInferenceAlgorithmEx",                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNForwardTrainingAlgorithmMaxCount"]              = {"hipdnnGetRNNForwardTrainingAlgorithmMaxCount",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnFindRNNForwardTrainingAlgorithmEx"]                   = {"hipdnnFindRNNForwardTrainingAlgorithmEx",                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNBackwardDataAlgorithmMaxCount"]                 = {"hipdnnGetRNNBackwardDataAlgorithmMaxCount",                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnFindRNNBackwardDataAlgorithmEx"]                      = {"hipdnnFindRNNBackwardDataAlgorithmEx",                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNBackwardWeightsAlgorithmMaxCount"]              = {"hipdnnGetRNNBackwardWeightsAlgorithmMaxCount",              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnFindRNNBackwardWeightsAlgorithmEx"]                   = {"hipdnnFindRNNBackwardWeightsAlgorithmEx",                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnCreatePersistentRNNPlan"]                             = {"hipdnnCreatePersistentRNNPlan",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetPersistentRNNPlan"]                                = {"hipdnnSetPersistentRNNPlan",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnDestroyPersistentRNNPlan"]                            = {"hipdnnDestroyPersistentRNNPlan",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // NOTE" hipdnnSetRNNDescriptor has additional argument hipdnnRNNBiasMode_t *biasMode without default value
  m["cudnnSetRNNDescriptor"]                                    = {"hipdnnSetRNNDescriptor",                                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  // NOTE" hipdnnGetRNNDescriptor has additional argument hipdnnRNNBiasMode_t *biasMode without default value
  m["cudnnGetRNNDescriptor"]                                    = {"hipdnnGetRNNDescriptor",                                    "miopenGetRNNDescriptor_V2",                                          CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNDescriptor_v6"]                                 = {"hipdnnGetRNNDescriptor_v6",                                 "miopenGetRNNDescriptor_V2",                                          CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNDescriptor_v8"]                                 = {"hipdnnGetRNNDescriptor_v8",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetRNNProjectionLayers"]                              = {"hipdnnSetRNNProjectionLayers",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNProjectionLayers"]                              = {"hipdnnGetRNNProjectionLayers",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetRNNAlgorithmDescriptor"]                           = {"hipdnnSetRNNAlgorithmDescriptor",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetRNNMatrixMathType"]                                = {"hipdnnSetRNNMatrixMathType",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNMatrixMathType"]                                = {"hipdnnGetRNNMatrixMathType",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNWorkspaceSize"]                                 = {"hipdnnGetRNNWorkspaceSize",                                 "miopenGetRNNWorkspaceSize",                                          CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNTrainingReserveSize"]                           = {"hipdnnGetRNNTrainingReserveSize",                           "miopenGetRNNTrainingReserveSize",                                    CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNParamsSize"]                                    = {"hipdnnGetRNNParamsSize",                                    "miopenGetRNNParamsSize",                                             CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNLinLayerMatrixParams"]                          = {"hipdnnGetRNNLinLayerMatrixParams",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNLinLayerBiasParams"]                            = {"hipdnnGetRNNLinLayerBiasParams",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNForward"]                                          = {"hipdnnRNNForward",                                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnRNNForwardInference"]                                 = {"hipdnnRNNForwardInference",                                 "miopenRNNForwardInference",                                          CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNForwardInferenceEx"]                               = {"hipdnnRNNForwardInferenceEx",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNForwardTraining"]                                  = {"hipdnnRNNForwardTraining",                                  "miopenRNNForwardTraining",                                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNForwardTrainingEx"]                                = {"hipdnnRNNForwardTrainingEx",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNBackwardData"]                                     = {"hipdnnRNNBackwardData",                                     "miopenRNNBackwardData",                                              CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNBackwardData_v8"]                                  = {"hipdnnRNNBackwardData_v8",                                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnRNNBackwardDataEx"]                                   = {"hipdnnRNNBackwardDataEx",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNBackwardWeights"]                                  = {"hipdnnRNNBackwardWeights",                                  "miopenRNNBackwardWeights",                                           CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNBackwardWeights_v8"]                               = {"hipdnnRNNBackwardWeights_v8",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnRNNBackwardWeightsEx"]                                = {"hipdnnRNNBackwardWeightsEx",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetRNNDescriptor_v5"]                                 = {"hipdnnSetRNNDescriptor_v5",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetRNNDescriptor_v6"]                                 = {"hipdnnSetRNNDescriptor_v6",                                 "miopenSetRNNDescriptor_V2",                                          CONV_LIB_FUNC, API_DNN, 2, CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetRNNDescriptor_v8"]                                 = {"hipdnnSetRNNDescriptor_v8",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetRNNPaddingMode"]                                   = {"hipdnnSetRNNPaddingMode",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNPaddingMode"]                                   = {"hipdnnGetRNNPaddingMode",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnCreateRNNDataDescriptor"]                             = {"hipdnnCreateRNNDataDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnDestroyRNNDataDescriptor"]                            = {"hipdnnDestroyRNNDataDescriptor",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetRNNDataDescriptor"]                                = {"hipdnnSetRNNDataDescriptor",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetRNNDataDescriptor"]                                = {"hipdnnGetRNNDataDescriptor",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetRNNBiasMode"]                                      = {"hipdnnSetRNNBiasMode",                                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetRNNBiasMode"]                                      = {"hipdnnGetRNNBiasMode",                                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnBuildRNNDynamic"]                                     = {"hipdnnBuildRNNDynamic",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetRNNTempSpaceSizes"]                                = {"hipdnnGetRNNTempSpaceSizes",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetRNNWeightSpaceSize"]                               = {"hipdnnGetRNNWeightSpaceSize",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetRNNWeightParams"]                                  = {"hipdnnGetRNNWeightParams",                                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Connectionist Temporal Classification loss functions
  m["cudnnCreateCTCLossDescriptor"]                             = {"hipdnnCreateCTCLossDescriptor",                             "miopenCreateCTCLossDescriptor",                                      CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnSetCTCLossDescriptor"]                                = {"hipdnnSetCTCLossDescriptor",                                "miopenSetCTCLossDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetCTCLossDescriptor_v8"]                             = {"hipdnnSetCTCLossDescriptor_v8",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetCTCLossDescriptor_v9"]                             = {"hipdnnSetCTCLossDescriptor_v9",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnSetCTCLossDescriptorEx"]                              = {"hipdnnSetCTCLossDescriptorEx",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetCTCLossDescriptor"]                                = {"hipdnnGetCTCLossDescriptor",                                "miopenGetCTCLossDescriptor",                                         CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetCTCLossDescriptor_v8"]                             = {"hipdnnGetCTCLossDescriptor_v8",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetCTCLossDescriptor_v9"]                             = {"hipdnnGetCTCLossDescriptor_v9",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetCTCLossDescriptorEx"]                              = {"hipdnnGetCTCLossDescriptorEx",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyCTCLossDescriptor"]                            = {"hipdnnDestroyCTCLossDescriptor",                            "miopenDestroyCTCLossDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnCTCLoss"]                                             = {"hipdnnCTCLoss",                                             "miopenCTCLoss",                                                      CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnCTCLoss_v8"]                                          = {"hipdnnCTCLoss_v8",                                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetCTCLossWorkspaceSize"]                             = {"hipdnnGetCTCLossWorkspaceSize",                             "miopenGetCTCLossWorkspaceSize",                                      CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnGetCTCLossWorkspaceSize_v8"]                          = {"hipdnnGetCTCLossWorkspaceSize_v8",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Algorithm functions
  m["cudnnCreateAlgorithmDescriptor"]                           = {"hipdnnCreateAlgorithmDescriptor",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetAlgorithmDescriptor"]                              = {"hipdnnSetAlgorithmDescriptor",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetAlgorithmDescriptor"]                              = {"hipdnnGetAlgorithmDescriptor",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnCopyAlgorithmDescriptor"]                             = {"hipdnnCopyAlgorithmDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnDestroyAlgorithmDescriptor"]                          = {"hipdnnDestroyAlgorithmDescriptor",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnCreateAlgorithmPerformance"]                          = {"hipdnnCreateAlgorithmPerformance",                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSetAlgorithmPerformance"]                             = {"hipdnnSetAlgorithmPerformance",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetAlgorithmPerformance"]                             = {"hipdnnGetAlgorithmPerformance",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnDestroyAlgorithmPerformance"]                         = {"hipdnnDestroyAlgorithmPerformance",                         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnGetAlgorithmSpaceSize"]                               = {"hipdnnGetAlgorithmSpaceSize",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnSaveAlgorithm"]                                       = {"hipdnnSaveAlgorithm",                                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRestoreAlgorithm"]                                    = {"hipdnnRestoreAlgorithm",                                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};

  // cuDNN Clipping functions
  m["cudnnRNNSetClip"]                                          = {"hipdnnRNNSetClip",                                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNSetClip_v8"]                                       = {"hipdnnRNNSetClip_v8",                                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnRNNSetClip_v9"]                                       = {"hipdnnRNNSetClip_v9",                                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnRNNGetClip"]                                          = {"hipdnnRNNGetClip",                                          "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cudnnRNNGetClip_v8"]                                       = {"hipdnnRNNGetClip_v8",                                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnRNNGetClip_v9"]                                       = {"hipdnnRNNGetClip_v9",                                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Sequence functions
  m["cudnnCreateSeqDataDescriptor"]                             = {"hipdnnCreateSeqDataDescriptor",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroySeqDataDescriptor"]                            = {"hipdnnDestroySeqDataDescriptor",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetSeqDataDescriptor"]                                = {"hipdnnSetSeqDataDescriptor",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetSeqDataDescriptor"]                                = {"hipdnnGetSeqDataDescriptor",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Multihead Attention functions
  m["cudnnCreateAttnDescriptor"]                                = {"hipdnnCreateAttnDescriptor",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyAttnDescriptor"]                               = {"hipdnnDestroyAttnDescriptor",                               "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetAttnDescriptor"]                                   = {"hipdnnSetAttnDescriptor",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetAttnDescriptor"]                                   = {"hipdnnGetAttnDescriptor",                                   "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetMultiHeadAttnBuffers"]                             = {"hipdnnGetMultiHeadAttnBuffers",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetMultiHeadAttnWeights"]                             = {"hipdnnGetMultiHeadAttnWeights",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnMultiHeadAttnForward"]                                = {"hipdnnMultiHeadAttnForward",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnMultiHeadAttnBackwardData"]                           = {"hipdnnMultiHeadAttnBackwardData",                           "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnMultiHeadAttnBackwardWeights"]                        = {"hipdnnMultiHeadAttnBackwardWeights",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Fuse functions
  m["cudnnCreateFusedOpsConstParamPack"]                        = {"hipdnnCreateFusedOpsConstParamPack",                        "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyFusedOpsConstParamPack"]                       = {"hipdnnDestroyFusedOpsConstParamPack",                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetFusedOpsConstParamPackAttribute"]                  = {"hipdnnSetFusedOpsConstParamPackAttribute",                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFusedOpsConstParamPackAttribute"]                  = {"hipdnnGetFusedOpsConstParamPackAttribute",                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnCreateFusedOpsVariantParamPack"]                      = {"hipdnnCreateFusedOpsVariantParamPack",                      "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyFusedOpsVariantParamPack"]                     = {"hipdnnDestroyFusedOpsVariantParamPack",                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnSetFusedOpsVariantParamPackAttribute"]                = {"hipdnnSetFusedOpsVariantParamPackAttribute",                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnGetFusedOpsVariantParamPackAttribute"]                = {"hipdnnGetFusedOpsVariantParamPackAttribute",                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnCreateFusedOpsPlan"]                                  = {"hipdnnCreateFusedOpsPlan",                                  "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnDestroyFusedOpsPlan"]                                 = {"hipdnnDestroyFusedOpsPlan",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnMakeFusedOpsPlan"]                                    = {"hipdnnMakeFusedOpsPlan",                                    "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};
  m["cudnnFusedOpsExecute"]                                     = {"hipdnnFusedOpsExecute",                                     "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED | CUDA_DEPRECATED};

  // cuDNN Backend
  m["cudnnBackendCreateDescriptor"]                             = {"hipdnnBackendCreateDescriptor",                             "miopenBackendCreateDescriptor",                                      CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnBackendDestroyDescriptor"]                            = {"hipdnnBackendDestroyDescriptor",                            "miopenBackendDestroyDescriptor",                                     CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  // NOTE: cudnnBackendInitialize and miopenBackendInitialize have different signatures
  m["cudnnBackendInitialize"]                                   = {"hipdnnBackendInitialize",                                   "miopenBackendInitialize",                                            CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnBackendFinalize"]                                     = {"hipdnnBackendFinalize",                                     "miopenBackendFinalize",                                              CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnBackendSetAttribute"]                                 = {"hipdnnBackendSetAttribute",                                 "miopenBackendSetAttribute",                                          CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnBackendGetAttribute"]                                 = {"hipdnnBackendGetAttribute",                                 "miopenBackendGetAttribute",                                          CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnBackendExecute"]                                      = {"hipdnnBackendExecute",                                      "miopenBackendExecute",                                               CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED};
  m["cudnnBackendPopulateCudaGraph"]                            = {"hipdnnBackendPopulateCudaGraph",                            "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnBackendUpdateCudaGraph"]                              = {"hipdnnBackendUpdateCudaGraph",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  // cuDNN Subquadratic Ops functions
  m["cudnnSubquadraticOpsVersionCheck"]                         = {"hipdnnSubquadraticOpsVersionCheck",                         "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnCausalConv1dForward"]                                 = {"hipdnnCausalConv1dForward",                                 "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnCausalConv1dBackward"]                                = {"hipdnnCausalConv1dBackward",                                "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnGetExecutionPlanWorkspaceSize"]                       = {"hipdnnGetExecutionPlanWorkspaceSize",                       "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnCausalConv1dNwhForward"]                              = {"hipdnnCausalConv1dNwhForward",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnCausalConv1dNwhBackward"]                             = {"hipdnnCausalConv1dNwhBackward",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnB2BCausalConv1dForward"]                              = {"hipdnnB2BCausalConv1dForward",                              "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};
  m["cudnnB2BCausalConv1dBackward"]                             = {"hipdnnB2BCausalConv1dBackward",                             "",                                                                   CONV_LIB_FUNC, API_DNN, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cudnnCreateRNNDescriptor"]                                 = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDestroyRNNDescriptor"]                                = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSetRNNDescriptor_v8"]                                 = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnGetRNNDescriptor_v8"]                                 = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnSetRNNDescriptor_v6"]                                 = {CUDNN_60,  CUDNN_801, CUDNN_900};
  m["cudnnGetRNNDescriptor_v6"]                                 = {CUDNN_801, CUDNN_801, CUDNN_900};
  m["cudnnSetRNNMatrixMathType"]                                = {CUDNN_705, CUDNN_801, CUDNN_900};
  m["cudnnGetRNNMatrixMathType"]                                = {CUDNN_713, CUDNN_801, CUDNN_900};
  m["cudnnSetRNNBiasMode"]                                      = {CUDNN_750, CUDNN_801, CUDNN_900};
  m["cudnnGetRNNBiasMode"]                                      = {CUDNN_750, CUDNN_801, CUDNN_900};
  m["cudnnRNNSetClip_v8"]                                       = {CUDNN_801, CUDNN_9220,CUDA_0   };  
  m["cudnnRNNGetClip_v8"]                                       = {CUDNN_801, CUDNN_9220,CUDA_0   };
  m["cudnnRNNSetClip"]                                          = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnRNNGetClip"]                                          = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnSetRNNProjectionLayers"]                              = {CUDNN_713, CUDNN_801, CUDNN_900};
  m["cudnnGetRNNProjectionLayers"]                              = {CUDNN_713, CUDNN_801, CUDNN_900};
  m["cudnnCreatePersistentRNNPlan"]                             = {CUDNN_60,  CUDNN_801, CUDNN_900};
  m["cudnnDestroyPersistentRNNPlan"]                            = {CUDNN_60,  CUDNN_801, CUDNN_900};
  m["cudnnSetPersistentRNNPlan"]                                = {CUDNN_60,  CUDNN_801, CUDNN_900};
  m["cudnnBuildRNNDynamic"]                                     = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnGetRNNWorkspaceSize"]                                 = {CUDNN_50,  CUDNN_801, CUDNN_900};  
  m["cudnnGetRNNTrainingReserveSize"]                           = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnGetRNNTempSpaceSizes"]                                = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnGetRNNParamsSize"]                                    = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnGetRNNWeightSpaceSize"]                               = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnGetRNNLinLayerMatrixParams"]                          = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnGetRNNLinLayerBiasParams"]                            = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnGetRNNWeightParams"]                                  = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnRNNForwardInference"]                                 = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnSetRNNPaddingMode"]                                   = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnGetRNNPaddingMode"]                                   = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnCreateRNNDataDescriptor"]                             = {CUDNN_721, CUDA_0,    CUDA_0   };
  m["cudnnDestroyRNNDataDescriptor"]                            = {CUDNN_721, CUDA_0,    CUDA_0   };
  m["cudnnSetRNNDataDescriptor"]                                = {CUDNN_721, CUDA_0,    CUDA_0   };
  m["cudnnGetRNNDataDescriptor"]                                = {CUDNN_721, CUDA_0,    CUDA_0   };
  m["cudnnRNNForwardInferenceEx"]                               = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnRNNForward"]                                          = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnSetRNNAlgorithmDescriptor"]                           = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetRNNForwardInferenceAlgorithmMaxCount"]             = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnFindRNNForwardInferenceAlgorithmEx"]                  = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnCreateSeqDataDescriptor"]                             = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnDestroySeqDataDescriptor"]                            = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnSetSeqDataDescriptor"]                                = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetSeqDataDescriptor"]                                = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnCreateAttnDescriptor"]                                = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnDestroyAttnDescriptor"]                               = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnSetAttnDescriptor"]                                   = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetAttnDescriptor"]                                   = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetMultiHeadAttnBuffers"]                             = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetMultiHeadAttnWeights"]                             = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnMultiHeadAttnForward"]                                = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnAdvInferVersionCheck"]                                = {CUDNN_801, CUDA_0,    CUDNN_900};
  m["cudnnRNNForwardTraining"]                                  = {CUDNN_50,  CUDNN_801, CUDNN_900};
  m["cudnnRNNBackwardData"]                                     = {CUDNN_50,  CUDNN_802, CUDNN_900};
  m["cudnnRNNBackwardData_v8"]                                  = {CUDNN_802, CUDA_0,    CUDA_0   };
  m["cudnnRNNBackwardWeights"]                                  = {CUDNN_50,  CUDNN_802, CUDNN_900};
  m["cudnnRNNBackwardWeights_v8"]                               = {CUDNN_802, CUDA_0,    CUDA_0   };
  m["cudnnRNNForwardTrainingEx"]                                = {CUDNN_721, CUDNN_801, CUDNN_900};
  m["cudnnRNNBackwardDataEx"]                                   = {CUDNN_721, CUDNN_802, CUDNN_900};
  m["cudnnRNNBackwardWeightsEx"]                                = {CUDNN_721, CUDNN_802, CUDNN_900};
  m["cudnnGetRNNForwardTrainingAlgorithmMaxCount"]              = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnFindRNNForwardTrainingAlgorithmEx"]                   = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetRNNBackwardDataAlgorithmMaxCount"]                 = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnFindRNNBackwardDataAlgorithmEx"]                      = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetRNNBackwardWeightsAlgorithmMaxCount"]              = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnFindRNNBackwardWeightsAlgorithmEx"]                   = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnMultiHeadAttnBackwardData"]                           = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnMultiHeadAttnBackwardWeights"]                        = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnCreateCTCLossDescriptor"]                             = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnSetCTCLossDescriptor"]                                = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnSetCTCLossDescriptorEx"]                              = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnSetCTCLossDescriptor_v8"]                             = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnGetCTCLossDescriptor"]                                = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetCTCLossDescriptorEx"]                              = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetCTCLossDescriptor_v8"]                             = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnDestroyCTCLossDescriptor"]                            = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnCTCLoss"]                                             = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnCTCLoss_v8"]                                          = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnGetCTCLossWorkspaceSize"]                             = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnGetCTCLossWorkspaceSize_v8"]                          = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnAdvTrainVersionCheck"]                                = {CUDNN_801, CUDA_0,    CUDNN_900};
  m["cudnnBackendCreateDescriptor"]                             = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnBackendDestroyDescriptor"]                            = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnBackendInitialize"]                                   = {CUDNN_801, CUDNN_930, CUDA_0   };
  m["cudnnBackendFinalize"]                                     = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnBackendSetAttribute"]                                 = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnBackendGetAttribute"]                                 = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnBackendExecute"]                                      = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnCreateConvolutionDescriptor"]                         = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnDestroyConvolutionDescriptor"]                        = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnSetConvolutionMathType"]                              = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionMathType"]                              = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnSetConvolutionGroupCount"]                            = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionGroupCount"]                            = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnSetConvolutionReorderType"]                           = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionReorderType"]                           = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnSetConvolution2dDescriptor"]                          = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolution2dDescriptor"]                          = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnSetConvolutionNdDescriptor"]                          = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionNdDescriptor"]                          = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolution2dForwardOutputDim"]                    = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionNdForwardOutputDim"]                    = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionForwardAlgorithmMaxCount"]              = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionForwardAlgorithm_v7"]                   = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnFindConvolutionForwardAlgorithm"]                     = {CUDNN_30,  CUDNN_900, CUDA_0   };
  m["cudnnFindConvolutionForwardAlgorithmEx"]                   = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnIm2Col"]                                              = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnReorderFilterAndBias"]                                = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionForwardWorkspaceSize"]                  = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnConvolutionForward"]                                  = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnConvolutionBiasActivationForward"]                    = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionBackwardDataAlgorithmMaxCount"]         = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnFindConvolutionBackwardDataAlgorithm"]                = {CUDNN_30,  CUDNN_900, CUDA_0   };
  m["cudnnFindConvolutionBackwardDataAlgorithmEx"]              = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionBackwardDataAlgorithm_v7"]              = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionBackwardDataWorkspaceSize"]             = {CUDNN_30,  CUDNN_900, CUDA_0   };
  m["cudnnConvolutionBackwardData"]                             = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnGetFoldedConvBackwardDataDescriptors"]                = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnCnnInferVersionCheck"]                                = {CUDNN_802, CUDA_0,    CUDA_0   };
  m["cudnnGetConvolutionBackwardFilterAlgorithmMaxCount"]       = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnFindConvolutionBackwardFilterAlgorithm"]              = {CUDNN_30,  CUDNN_900, CUDA_0   };
  m["cudnnFindConvolutionBackwardFilterAlgorithmEx"]            = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionBackwardFilterAlgorithm_v7"]            = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetConvolutionBackwardFilterWorkspaceSize"]           = {CUDNN_30,  CUDNN_900, CUDA_0   };
  m["cudnnConvolutionBackwardFilter"]                           = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnConvolutionBackwardBias"]                             = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnCreateFusedOpsConstParamPack"]                        = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnDestroyFusedOpsConstParamPack"]                       = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnSetFusedOpsConstParamPackAttribute"]                  = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnGetFusedOpsConstParamPackAttribute"]                  = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnCreateFusedOpsVariantParamPack"]                      = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnDestroyFusedOpsVariantParamPack"]                     = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnSetFusedOpsVariantParamPackAttribute"]                = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnGetFusedOpsVariantParamPackAttribute"]                = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnCreateFusedOpsPlan"]                                  = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnDestroyFusedOpsPlan"]                                 = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnMakeFusedOpsPlan"]                                    = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnFusedOpsExecute"]                                     = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnCnnTrainVersionCheck"]                                = {CUDNN_802, CUDA_0,    CUDA_0   };
  m["cudnnGetVersion"]                                          = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnGetCudartVersion"]                                    = {CUDNN_60,  CUDA_0,    CUDA_0   };
  m["cudnnGetErrorString"]                                      = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnQueryRuntimeError"]                                   = {CUDNN_705, CUDNN_900, CUDA_0   };
  m["cudnnGetProperty"]                                         = {CUDNN_60,  CUDA_0,    CUDA_0   };
  m["cudnnCreate"]                                              = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnDestroy"]                                             = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnSetStream"]                                           = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnGetStream"]                                           = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnCreateTensorDescriptor"]                              = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnSetTensor4dDescriptor"]                               = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnSetTensor4dDescriptorEx"]                             = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnGetTensor4dDescriptor"]                               = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnSetTensorNdDescriptor"]                               = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnSetTensorNdDescriptorEx"]                             = {CUDNN_60,  CUDA_0,    CUDA_0   };
  m["cudnnGetTensorNdDescriptor"]                               = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnGetTensorSizeInBytes"]                                = {CUDNN_60,  CUDA_0,    CUDA_0   };
  m["cudnnDestroyTensorDescriptor"]                             = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnInitTransformDest"]                                   = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnCreateTensorTransformDescriptor"]                     = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnSetTensorTransformDescriptor"]                        = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnGetTensorTransformDescriptor"]                        = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnDestroyTensorTransformDescriptor"]                    = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnTransformTensor"]                                     = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnTransformTensorEx"]                                   = {CUDNN_750, CUDNN_900, CUDA_0   };
  m["cudnnAddTensor"]                                           = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnCreateOpTensorDescriptor"]                            = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnSetOpTensorDescriptor"]                               = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnGetOpTensorDescriptor"]                               = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnDestroyOpTensorDescriptor"]                           = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnOpTensor"]                                            = {CUDNN_50,  CUDNN_900, CUDA_0   };
  m["cudnnCreateReduceTensorDescriptor"]                        = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnSetReduceTensorDescriptor"]                           = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnGetReduceTensorDescriptor"]                           = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnDestroyReduceTensorDescriptor"]                       = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnGetReductionIndicesSize"]                             = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnGetReductionWorkspaceSize"]                           = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnReduceTensor"]                                        = {CUDNN_60,  CUDNN_900, CUDA_0   };
  m["cudnnSetTensor"]                                           = {CUDNN_20,  CUDA_0,    CUDA_0   };
  m["cudnnScaleTensor"]                                         = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnCreateFilterDescriptor"]                              = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnSetFilter4dDescriptor"]                               = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetFilter4dDescriptor"]                               = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnSetFilterNdDescriptor"]                               = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetFilterNdDescriptor"]                               = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetFilterSizeInBytes"]                                = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnTransformFilter"]                                     = {CUDNN_760, CUDNN_900, CUDA_0   };
  m["cudnnDestroyFilterDescriptor"]                             = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnSoftmaxForward"]                                      = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnCreatePoolingDescriptor"]                             = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnSetPooling2dDescriptor"]                              = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetPooling2dDescriptor"]                              = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnSetPoolingNdDescriptor"]                              = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetPoolingNdDescriptor"]                              = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetPoolingNdForwardOutputDim"]                        = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnGetPooling2dForwardOutputDim"]                        = {CUDNN_20,  CUDNN_900, CUDA_0   };
  m["cudnnDestroyPoolingDescriptor"]                            = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnPoolingForward"]                                      = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnCreateActivationDescriptor"]                          = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnSetActivationDescriptor"]                             = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnGetActivationDescriptor"]                             = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnDestroyActivationDescriptor"]                         = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnActivationForward"]                                   = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnCreateLRNDescriptor"]                                 = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnSetLRNDescriptor"]                                    = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnGetLRNDescriptor"]                                    = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnDestroyLRNDescriptor"]                                = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnLRNCrossChannelForward"]                              = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnDivisiveNormalizationForward"]                        = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnDeriveBNTensorDescriptor"]                            = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnBatchNormalizationForwardInference"]                  = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnDeriveNormTensorDescriptor"]                          = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnNormalizationForwardInference"]                       = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnCreateSpatialTransformerDescriptor"]                  = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSetSpatialTransformerNdDescriptor"]                   = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDestroySpatialTransformerDescriptor"]                 = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSpatialTfGridGeneratorForward"]                       = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSpatialTfSamplerForward"]                             = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnCreateDropoutDescriptor"]                             = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDestroyDropoutDescriptor"]                            = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDropoutGetStatesSize"]                                = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDropoutGetReserveSpaceSize"]                          = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSetDropoutDescriptor"]                                = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnRestoreDropoutDescriptor"]                            = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnGetDropoutDescriptor"]                                = {CUDNN_705, CUDA_0,    CUDA_0   };
  m["cudnnDropoutForward"]                                      = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnCreateAlgorithmDescriptor"]                           = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnSetAlgorithmDescriptor"]                              = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetAlgorithmDescriptor"]                              = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnCopyAlgorithmDescriptor"]                             = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnDestroyAlgorithmDescriptor"]                          = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnCreateAlgorithmPerformance"]                          = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnSetAlgorithmPerformance"]                             = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetAlgorithmPerformance"]                             = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnDestroyAlgorithmPerformance"]                         = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnGetAlgorithmSpaceSize"]                               = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnSaveAlgorithm"]                                       = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnRestoreAlgorithm"]                                    = {CUDNN_713, CUDNN_802, CUDNN_900};
  m["cudnnSetCallback"]                                         = {CUDNN_713, CUDA_0,    CUDA_0   };
  m["cudnnGetCallback"]                                         = {CUDNN_713, CUDA_0,    CUDA_0   };
  m["cudnnOpsInferVersionCheck"]                                = {CUDNN_801, CUDA_0,    CUDA_0   };
  m["cudnnSoftmaxBackward"]                                     = {CUDNN_10,  CUDA_0,    CUDA_0   };
  m["cudnnPoolingBackward"]                                     = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnActivationBackward"]                                  = {CUDNN_10,  CUDNN_900, CUDA_0   };
  m["cudnnLRNCrossChannelBackward"]                             = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnDivisiveNormalizationBackward"]                       = {CUDNN_30,  CUDA_0,    CUDA_0   };
  m["cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize"] = {CUDNN_741, CUDNN_900, CUDA_0   };
  m["cudnnGetBatchNormalizationBackwardExWorkspaceSize"]        = {CUDNN_741, CUDNN_900, CUDA_0   };
  m["cudnnGetBatchNormalizationTrainingExReserveSpaceSize"]     = {CUDNN_741, CUDNN_900, CUDA_0   };
  m["cudnnBatchNormalizationForwardTraining"]                   = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnBatchNormalizationForwardTrainingEx"]                 = {CUDNN_741, CUDNN_900, CUDA_0   };
  m["cudnnBatchNormalizationBackward"]                          = {CUDNN_40,  CUDNN_900, CUDA_0   };
  m["cudnnBatchNormalizationBackwardEx"]                        = {CUDNN_741, CUDNN_900, CUDA_0   };
  m["cudnnGetNormalizationForwardTrainingWorkspaceSize"]        = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnGetNormalizationBackwardWorkspaceSize"]               = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnGetNormalizationTrainingReserveSpaceSize"]            = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnNormalizationForwardTraining"]                        = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnNormalizationBackward"]                               = {CUDNN_801, CUDNN_900, CUDA_0   };
  m["cudnnSpatialTfGridGeneratorBackward"]                      = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnSpatialTfSamplerBackward"]                            = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnDropoutBackward"]                                     = {CUDNN_50,  CUDA_0,    CUDA_0   };
  m["cudnnOpsTrainVersionCheck"]                                = {CUDNN_801, CUDA_0,    CUDNN_900};
  m["cudnnGetConvolutionBackwardDataAlgorithm"]                 = {CUDNN_30,  CUDNN_765, CUDNN_801};
  m["cudnnGetConvolutionBackwardFilterAlgorithm"]               = {CUDNN_30,  CUDNN_765, CUDNN_801};
  m["cudnnGetConvolutionForwardAlgorithm"]                      = {CUDNN_20,  CUDNN_765, CUDNN_801};
  m["cudnnGetRNNDescriptor"]                                    = {CUDNN_705, CUDNN_765, CUDNN_801};
  m["cudnnSetRNNDescriptor"]                                    = {CUDNN_50,  CUDNN_765, CUDNN_801};
  m["cudnnSetRNNDescriptor_v5"]                                 = {CUDNN_705, CUDNN_765, CUDNN_801};
  m["cudnnSetActivationDescriptorSwishBeta"]                    = {CUDNN_820, CUDNN_900, CUDA_0   };
  m["cudnnGetActivationDescriptorSwishBeta"]                    = {CUDNN_820, CUDNN_900, CUDA_0   };
  m["cudnnGetMaxDeviceVersion"]                                 = {CUDNN_860, CUDA_0,    CUDA_0   };
  m["cudnnRNNSetClip_v9"]                                       = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnRNNGetClip_v9"]                                       = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnAdvVersionCheck"]                                     = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnSetCTCLossDescriptor_v9"]                             = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnGetCTCLossDescriptor_v9"]                             = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnGetLastErrorString"]                                  = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnGraphVersionCheck"]                                   = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnOpsVersionCheck"]                                     = {CUDNN_900, CUDA_0,    CUDA_0   };
  m["cudnnBackendPopulateCudaGraph"]                            = {CUDNN_950, CUDA_0,    CUDA_0   };
  m["cudnnBackendUpdateCudaGraph"]                              = {CUDNN_950, CUDA_0,    CUDA_0   };
  m["cudnnSubquadraticOpsVersionCheck"]                         = {CUDNN_9220,CUDA_0,    CUDA_0   };
  m["cudnnCausalConv1dForward"]                                 = {CUDNN_9220,CUDA_0,    CUDA_0   };
  m["cudnnCausalConv1dBackward"]                                = {CUDNN_9220,CUDA_0,    CUDA_0   };
  m["cudnnGetExecutionPlanWorkspaceSize"]                       = {CUDNN_9230,CUDA_0,    CUDA_0   };
  m["cudnnCausalConv1dNwhForward"]                              = {CUDNN_9240,CUDA_0,    CUDA_0   };
  m["cudnnCausalConv1dNwhBackward"]                             = {CUDNN_9240,CUDA_0,    CUDA_0   };
  m["cudnnB2BCausalConv1dForward"]                              = {CUDNN_9240,CUDA_0,    CUDA_0   };
  m["cudnnB2BCausalConv1dBackward"]                             = {CUDNN_9240,CUDA_0,    CUDA_0   };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_DNN_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["miopenGetErrorString"]                                     = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreate"]                                             = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroy"]                                            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSetStream"]                                          = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetStream"]                                          = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateTensorDescriptor"]                             = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSet4dTensorDescriptorEx"]                            = {HIP_5030,  HIP_0,     HIP_0    };
  m["miopenGet4dTensorDescriptor"]                              = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyTensorDescriptor"]                            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenTransformTensor"]                                    = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateReduceTensorDescriptor"]                       = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenSetReduceTensorDescriptor"]                          = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenGetReduceTensorDescriptor"]                          = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenDestroyReduceTensorDescriptor"]                      = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenGetReductionIndicesSize"]                            = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenGetReductionWorkspaceSize"]                          = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenReduceTensor"]                                       = {HIP_3090,  HIP_0,     HIP_0    };
  m["miopenSetTensor"]                                          = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenScaleTensor"]                                        = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateConvolutionDescriptor"]                        = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSetConvolutionGroupCount"]                           = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetConvolutionForwardOutputDim"]                     = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyConvolutionDescriptor"]                       = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenFindConvolutionForwardAlgorithm"]                    = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenConvolutionForwardGetWorkSpaceSize"]                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenConvolutionForward"]                                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenConvolutionBiasActivationForward"]                   = {HIP_5040,  HIP_0,     HIP_0    };
  m["miopenConvolutionBackwardBias"]                            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenConvolutionBackwardDataGetWorkSpaceSize"]            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenConvolutionBackwardData"]                            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSoftmaxForward_V2"]                                  = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenSoftmaxBackward_V2"]                                 = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenCreatePoolingDescriptor"]                            = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSet2dPoolingDescriptor"]                             = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGet2dPoolingDescriptor"]                             = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSetNdPoolingDescriptor"]                             = {HIP_3030,  HIP_0,     HIP_0    };
  m["miopenGetNdPoolingDescriptor"]                             = {HIP_3030,  HIP_0,     HIP_0    };
  m["miopenGetPoolingNdForwardOutputDim"]                       = {HIP_3030,  HIP_0,     HIP_0    };
  m["miopenGetPoolingForwardOutputDim"]                         = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyPoolingDescriptor"]                           = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateActivationDescriptor"]                         = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyActivationDescriptor"]                        = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenActivationForward"]                                  = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenActivationBackward"]                                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateLRNDescriptor"]                                = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSetLRNDescriptor"]                                   = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetLRNDescriptor"]                                   = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyLRNDescriptor"]                               = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDeriveBNTensorDescriptor"]                           = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenBatchNormalizationForwardTraining"]                  = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenBatchNormalizationForwardInference"]                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenBatchNormalizationBackward"]                         = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenCreateDropoutDescriptor"]                            = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenDestroyDropoutDescriptor"]                           = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenDropoutGetStatesSize"]                               = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenDropoutGetReserveSpaceSize"]                         = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenSetDropoutDescriptor"]                               = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenGetDropoutDescriptor"]                               = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenRestoreDropoutDescriptor"]                           = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenDropoutForward"]                                     = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenDropoutBackward"]                                    = {HIP_2080,  HIP_0,     HIP_0    };
  m["miopenCreateRNNDescriptor"]                                = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenDestroyRNNDescriptor"]                               = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetRNNDescriptor_V2"]                                = {HIP_3050,  HIP_0,     HIP_0    };
  m["miopenGetRNNWorkspaceSize"]                                = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetRNNTrainingReserveSize"]                          = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenGetRNNParamsSize"]                                   = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenRNNForwardInference"]                                = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenRNNForwardTraining"]                                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenRNNBackwardData"]                                    = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenRNNBackwardWeights"]                                 = {HIP_2010,  HIP_0,     HIP_0    };
  m["miopenSetRNNDescriptor_V2"]                                = {HIP_3050,  HIP_0,     HIP_0    };
  m["miopenCreateCTCLossDescriptor"]                            = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenSetCTCLossDescriptor"]                               = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenGetCTCLossDescriptor"]                               = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenDestroyCTCLossDescriptor"]                           = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenCTCLoss"]                                            = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenGetCTCLossWorkspaceSize"]                            = {HIP_2060,  HIP_0,     HIP_0    };
  m["miopenBackendCreateDescriptor"]                            = {HIP_6020,  HIP_0,     HIP_0    };
  m["miopenBackendDestroyDescriptor"]                           = {HIP_6020,  HIP_0,     HIP_0    };
  m["miopenBackendFinalize"]                                    = {HIP_6020,  HIP_0,     HIP_0    };
  m["miopenBackendSetAttribute"]                                = {HIP_6020,  HIP_0,     HIP_0    };
  m["miopenBackendGetAttribute"]                                = {HIP_6020,  HIP_0,     HIP_0    };
  m["miopenBackendExecute"]                                     = {HIP_6020,  HIP_0,     HIP_0    };

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_DNN_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1] = "CUDNN Data types";
  m[2] = "CUDNN Functions";

  return m;
}();
