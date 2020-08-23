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
const std::map<llvm::StringRef, hipCounter> CUDA_DNN_FUNCTION_MAP {

  {"cudnnGetVersion",                                     {"hipdnnGetVersion",                                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetCudartVersion",                               {"hipdnnGetCudartVersion",                               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnQueryRuntimeError",                              {"hipdnnQueryRuntimeError",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetProperty",                                    {"hipdnnGetProperty",                                    "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetErrorString",                                 {"hipdnnGetErrorString",                                 "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnIm2Col",                                         {"hipdnnIm2Col",                                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreate",                                         {"hipdnnCreate",                                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroy",                                        {"hipdnnDestroy",                                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetStream",                                      {"hipdnnSetStream",                                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetStream",                                      {"hipdnnGetStream",                                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetCallback",                                    {"hipdnnSetCallback",                                    "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetCallback",                                    {"hipdnnGetCallback",                                    "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Tensor functions
  {"cudnnCreateTensorDescriptor",                         {"hipdnnCreateTensorDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetTensor4dDescriptor",                          {"hipdnnSetTensor4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetTensor4dDescriptorEx",                        {"hipdnnSetTensor4dDescriptorEx",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetTensor4dDescriptor",                          {"hipdnnGetTensor4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetTensorNdDescriptor",                          {"hipdnnSetTensorNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetTensorNdDescriptorEx",                        {"hipdnnSetTensorNdDescriptorEx",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetTensorNdDescriptor",                          {"hipdnnGetTensorNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetTensorSizeInBytes",                           {"hipdnnGetTensorSizeInBytes",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyTensorDescriptor",                        {"hipdnnDestroyTensorDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnTransformTensor",                                {"hipdnnTransformTensor",                                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnTransformTensorEx",                              {"hipdnnTransformTensorEx",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnInitTransformDest",                              {"hipdnnInitTransformDest",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreateTensorTransformDescriptor",                {"hipdnnCreateTensorTransformDescriptor",                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetTensorTransformDescriptor",                   {"hipdnnSetTensorTransformDescriptor",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetTensorTransformDescriptor",                   {"hipdnnGetTensorTransformDescriptor",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyTensorTransformDescriptor",               {"hipdnnDestroyTensorTransformDescriptor",               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnAddTensor",                                      {"hipdnnAddTensor",                                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnCreateOpTensorDescriptor",                       {"hipdnnCreateOpTensorDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetOpTensorDescriptor",                          {"hipdnnSetOpTensorDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetOpTensorDescriptor",                          {"hipdnnGetOpTensorDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyOpTensorDescriptor",                      {"hipdnnDestroyOpTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnOpTensor",                                       {"hipdnnOpTensor",                                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetFoldedConvBackwardDataDescriptors",           {"hipdnnGetFoldedConvBackwardDataDescriptors",           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Reduce Tensor functions
  {"cudnnCreateReduceTensorDescriptor",                   {"hipdnnCreateReduceTensorDescriptor",                   "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetReduceTensorDescriptor",                      {"hipdnnSetReduceTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetReduceTensorDescriptor",                      {"hipdnnGetReduceTensorDescriptor",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyReduceTensorDescriptor",                  {"hipdnnDestroyReduceTensorDescriptor",                  "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetReductionIndicesSize",                        {"hipdnnGetReductionIndicesSize",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetReductionWorkspaceSize",                      {"hipdnnGetReductionWorkspaceSize",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnReduceTensor",                                   {"hipdnnReduceTensor",                                   "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetTensor",                                      {"hipdnnSetTensor",                                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnScaleTensor",                                    {"hipdnnScaleTensor",                                    "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN Filter functions
  {"cudnnCreateFilterDescriptor",                         {"hipdnnCreateFilterDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetFilter4dDescriptor",                          {"hipdnnSetFilter4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetFilter4dDescriptor",                          {"hipdnnGetFilter4dDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetFilterNdDescriptor",                          {"hipdnnSetFilterNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetFilterNdDescriptor",                          {"hipdnnGetFilterNdDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetFilterSizeInBytes",                           {"hipdnnGetFilterSizeInBytes",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnTransformFilter",                                {"hipdnnTransformFilter",                                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyFilterDescriptor",                        {"hipdnnDestroyFilterDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnReorderFilterAndBias",                           {"hipdnnReorderFilterAndBias",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Convolution functions
  {"cudnnCreateConvolutionDescriptor",                    {"hipdnnCreateConvolutionDescriptor",                    "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetConvolutionMathType",                         {"hipdnnSetConvolutionMathType",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionMathType",                         {"hipdnnGetConvolutionMathType",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetConvolutionGroupCount",                       {"hipdnnSetConvolutionGroupCount",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionGroupCount",                       {"hipdnnGetConvolutionGroupCount",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetConvolutionReorderType",                      {"hipdnnSetConvolutionReorderType",                      "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionReorderType",                      {"hipdnnGetConvolutionReorderType",                      "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetConvolution2dDescriptor",                     {"hipdnnSetConvolution2dDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolution2dDescriptor",                     {"hipdnnGetConvolution2dDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolution2dForwardOutputDim",               {"hipdnnGetConvolution2dForwardOutputDim",               "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetConvolutionNdDescriptor",                     {"hipdnnSetConvolutionNdDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionNdDescriptor",                     {"hipdnnGetConvolutionNdDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionNdForwardOutputDim",               {"hipdnnGetConvolutionNdForwardOutputDim",               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyConvolutionDescriptor",                   {"hipdnnDestroyConvolutionDescriptor",                   "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionForwardAlgorithmMaxCount",         {"hipdnnGetConvolutionForwardAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionForwardAlgorithm",                {"hipdnnFindConvolutionForwardAlgorithm",                "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnFindConvolutionForwardAlgorithmEx",              {"hipdnnFindConvolutionForwardAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionForwardAlgorithm",                 {"hipdnnGetConvolutionForwardAlgorithm",                 "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionForwardAlgorithm_v7",              {"hipdnnGetConvolutionForwardAlgorithm_v7",              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionForwardWorkspaceSize",             {"hipdnnGetConvolutionForwardWorkspaceSize",             "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnConvolutionForward",                             {"hipdnnConvolutionForward",                             "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnConvolutionBiasActivationForward",               {"hipdnnConvolutionBiasActivationForward",               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnConvolutionBackwardBias",                        {"hipdnnConvolutionBackwardBias",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount",  {"hipdnnGetConvolutionBackwardFilterAlgorithmMaxCount",  "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionBackwardFilterAlgorithm",         {"hipdnnFindConvolutionBackwardFilterAlgorithm",         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnFindConvolutionBackwardFilterAlgorithmEx",       {"hipdnnFindConvolutionBackwardFilterAlgorithmEx",       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardFilterAlgorithm",          {"hipdnnGetConvolutionBackwardFilterAlgorithm",          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardFilterAlgorithm_v7",       {"hipdnnGetConvolutionBackwardFilterAlgorithm_v7",       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionBackwardFilterWorkspaceSize",      {"hipdnnGetConvolutionBackwardFilterWorkspaceSize",      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnConvolutionBackwardFilter",                      {"hipdnnConvolutionBackwardFilter",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardDataAlgorithmMaxCount",    {"hipdnnGetConvolutionBackwardDataAlgorithmMaxCount",    "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindConvolutionBackwardDataAlgorithm",           {"hipdnnFindConvolutionBackwardDataAlgorithm",           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnFindConvolutionBackwardDataAlgorithmEx",         {"hipdnnFindConvolutionBackwardDataAlgorithmEx",         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardDataAlgorithm",            {"hipdnnGetConvolutionBackwardDataAlgorithm",            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetConvolutionBackwardDataAlgorithm_v7",         {"hipdnnGetConvolutionBackwardDataAlgorithm_v7",         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetConvolutionBackwardDataWorkspaceSize",        {"hipdnnGetConvolutionBackwardDataWorkspaceSize",        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnConvolutionBackwardData",                        {"hipdnnConvolutionBackwardData",                        "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN Sortmax functions
  {"cudnnSoftmaxForward",                                 {"hipdnnSoftmaxForward",                                 "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSoftmaxBackward",                                {"hipdnnSoftmaxBackward",                                "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN Pooling functions
  {"cudnnCreatePoolingDescriptor",                        {"hipdnnCreatePoolingDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetPooling2dDescriptor",                         {"hipdnnSetPooling2dDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetPooling2dDescriptor",                         {"hipdnnGetPooling2dDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetPoolingNdDescriptor",                         {"hipdnnSetPoolingNdDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetPoolingNdDescriptor",                         {"hipdnnGetPoolingNdDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetPoolingNdForwardOutputDim",                   {"hipdnnGetPoolingNdForwardOutputDim",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetPooling2dForwardOutputDim",                   {"hipdnnGetPooling2dForwardOutputDim",                   "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyPoolingDescriptor",                       {"hipdnnDestroyPoolingDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnPoolingForward",                                 {"hipdnnPoolingForward",                                 "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnPoolingBackward",                                {"hipdnnPoolingBackward",                                "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN Activation functions
  {"cudnnCreateActivationDescriptor",                     {"hipdnnCreateActivationDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetActivationDescriptor",                        {"hipdnnSetActivationDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetActivationDescriptor",                        {"hipdnnGetActivationDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyActivationDescriptor",                    {"hipdnnDestroyActivationDescriptor",                    "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnActivationForward",                              {"hipdnnActivationForward",                              "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnActivationBackward",                             {"hipdnnActivationBackward",                             "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN LRN functions
  {"cudnnCreateLRNDescriptor",                            {"hipdnnCreateLRNDescriptor",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetLRNDescriptor",                               {"hipdnnSetLRNDescriptor",                               "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetLRNDescriptor",                               {"hipdnnGetLRNDescriptor",                               "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyLRNDescriptor",                           {"hipdnnDestroyLRNDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnLRNCrossChannelForward",                         {"hipdnnLRNCrossChannelForward",                         "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnLRNCrossChannelBackward",                        {"hipdnnLRNCrossChannelBackward",                        "", CONV_LIB_FUNC, API_DNN, 2}},

  // cuDNN Divisive Normalization functions
  {"cudnnDivisiveNormalizationForward",                   {"hipdnnDivisiveNormalizationForward",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDivisiveNormalizationBackward",                  {"hipdnnDivisiveNormalizationBackward",                  "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Batch Normalization functions
  {"cudnnDeriveBNTensorDescriptor",                            {"hipdnnDeriveBNTensorDescriptor",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnBatchNormalizationForwardTraining",                   {"hipdnnBatchNormalizationForwardTraining",                   "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnBatchNormalizationForwardTrainingEx",                 {"hipdnnBatchNormalizationForwardTrainingEx",                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnBatchNormalizationForwardInference",                  {"hipdnnBatchNormalizationForwardInference",                  "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnBatchNormalizationBackward",                          {"hipdnnBatchNormalizationBackward",                          "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnBatchNormalizationBackwardEx",                        {"hipdnnBatchNormalizationBackwardEx",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize", {"hipdnnGetBatchNormalizationForwardTrainingExWorkspaceSize", "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetBatchNormalizationBackwardExWorkspaceSize",        {"hipdnnGetBatchNormalizationBackwardExWorkspaceSize",        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetBatchNormalizationTrainingExReserveSpaceSize",     {"hipdnnGetBatchNormalizationTrainingExReserveSpaceSize",     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Spatial Transformer functions
  {"cudnnCreateSpatialTransformerDescriptor",             {"hipdnnCreateSpatialTransformerDescriptor",             "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetSpatialTransformerNdDescriptor",              {"hipdnnSetSpatialTransformerNdDescriptor",              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroySpatialTransformerDescriptor",            {"hipdnnDestroySpatialTransformerDescriptor",            "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfGridGeneratorForward",                  {"hipdnnSpatialTfGridGeneratorForward",                  "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfGridGeneratorBackward",                 {"hipdnnSpatialTfGridGeneratorBackward",                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfSamplerForward",                        {"hipdnnSpatialTfSamplerForward",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSpatialTfSamplerBackward",                       {"hipdnnSpatialTfSamplerBackward",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Dropout functions
  {"cudnnCreateDropoutDescriptor",                        {"hipdnnCreateDropoutDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyDropoutDescriptor",                       {"hipdnnDestroyDropoutDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDropoutGetStatesSize",                           {"hipdnnDropoutGetStatesSize",                           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDropoutGetReserveSpaceSize",                     {"hipdnnDropoutGetReserveSpaceSize",                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetDropoutDescriptor",                           {"hipdnnSetDropoutDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetDropoutDescriptor",                           {"hipdnnGetDropoutDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRestoreDropoutDescriptor",                       {"hipdnnRestoreDropoutDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDropoutForward",                                 {"hipdnnDropoutForward",                                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDropoutBackward",                                {"hipdnnDropoutBackward",                                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN RNN functions
  {"cudnnCreateRNNDescriptor",                            {"hipdnnCreateRNNDescriptor",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyRNNDescriptor",                           {"hipdnnDestroyRNNDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetRNNForwardInferenceAlgorithmMaxCount",        {"hipdnnGetRNNForwardInferenceAlgorithmMaxCount",        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindRNNForwardInferenceAlgorithmEx",             {"hipdnnFindRNNForwardInferenceAlgorithmEx",             "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNForwardTrainingAlgorithmMaxCount",         {"hipdnnGetRNNForwardTrainingAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindRNNForwardTrainingAlgorithmEx",              {"hipdnnFindRNNForwardTrainingAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNBackwardDataAlgorithmMaxCount",            {"hipdnnGetRNNBackwardDataAlgorithmMaxCount",            "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindRNNBackwardDataAlgorithmEx",                 {"hipdnnFindRNNBackwardDataAlgorithmEx",                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNBackwardWeightsAlgorithmMaxCount",         {"hipdnnGetRNNBackwardWeightsAlgorithmMaxCount",         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFindRNNBackwardWeightsAlgorithmEx",              {"hipdnnFindRNNBackwardWeightsAlgorithmEx",              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreatePersistentRNNPlan",                        {"hipdnnCreatePersistentRNNPlan",                        "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetPersistentRNNPlan",                           {"hipdnnSetPersistentRNNPlan",                           "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnDestroyPersistentRNNPlan",                       {"hipdnnDestroyPersistentRNNPlan",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  // NOTE" hipdnnSetRNNDescriptor has additional argument hipdnnRNNBiasMode_t *biasMode without default value
  {"cudnnSetRNNDescriptor",                               {"hipdnnSetRNNDescriptor",                               "", CONV_LIB_FUNC, API_DNN, 2}},
  // NOTE" hipdnnGetRNNDescriptor has additional argument hipdnnRNNBiasMode_t *biasMode without default value
  {"cudnnGetRNNDescriptor",                               {"hipdnnGetRNNDescriptor",                               "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetRNNProjectionLayers",                         {"hipdnnSetRNNProjectionLayers",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNProjectionLayers",                         {"hipdnnGetRNNProjectionLayers",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetRNNAlgorithmDescriptor",                      {"hipdnnSetRNNAlgorithmDescriptor",                      "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetRNNMatrixMathType",                           {"hipdnnSetRNNMatrixMathType",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNMatrixMathType",                           {"hipdnnGetRNNMatrixMathType",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNWorkspaceSize",                            {"hipdnnGetRNNWorkspaceSize",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetRNNTrainingReserveSize",                      {"hipdnnGetRNNTrainingReserveSize",                      "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetRNNParamsSize",                               {"hipdnnGetRNNParamsSize",                               "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetRNNLinLayerMatrixParams",                     {"hipdnnGetRNNLinLayerMatrixParams",                     "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnGetRNNLinLayerBiasParams",                       {"hipdnnGetRNNLinLayerBiasParams",                       "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnRNNForwardInference",                            {"hipdnnRNNForwardInference",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnRNNForwardInferenceEx",                          {"hipdnnRNNForwardInferenceEx",                          "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRNNForwardTraining",                             {"hipdnnRNNForwardTraining",                             "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnRNNForwardTrainingEx",                           {"hipdnnRNNForwardTrainingEx",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRNNBackwardData",                                {"hipdnnRNNBackwardData",                                "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnRNNBackwardDataEx",                              {"hipdnnRNNBackwardDataEx",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRNNBackwardWeights",                             {"hipdnnRNNBackwardWeights",                             "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnRNNBackwardWeightsEx",                           {"hipdnnRNNBackwardWeightsEx",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetRNNDescriptor_v5",                            {"hipdnnSetRNNDescriptor_v5",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetRNNDescriptor_v6",                            {"hipdnnSetRNNDescriptor_v6",                            "", CONV_LIB_FUNC, API_DNN, 2}},
  {"cudnnSetRNNPaddingMode",                              {"hipdnnSetRNNPaddingMode",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNPaddingMode",                              {"hipdnnGetRNNPaddingMode",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreateRNNDataDescriptor",                        {"hipdnnCreateRNNDataDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyRNNDataDescriptor",                       {"hipdnnDestroyRNNDataDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetRNNDataDescriptor",                           {"hipdnnSetRNNDataDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNDataDescriptor",                           {"hipdnnGetRNNDataDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetRNNBiasMode",                                 {"hipdnnSetRNNBiasMode",                                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetRNNBiasMode",                                 {"hipdnnGetRNNBiasMode",                                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Connectionist Temporal Classification loss functions
  {"cudnnCreateCTCLossDescriptor",                        {"hipdnnCreateCTCLossDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetCTCLossDescriptor",                           {"hipdnnSetCTCLossDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetCTCLossDescriptorEx",                         {"hipdnnSetCTCLossDescriptorEx",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetCTCLossDescriptor",                           {"hipdnnGetCTCLossDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetCTCLossDescriptorEx",                         {"hipdnnGetCTCLossDescriptorEx",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyCTCLossDescriptor",                       {"hipdnnDestroyCTCLossDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCTCLoss",                                        {"hipdnnCTCLoss",                                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetCTCLossWorkspaceSize",                        {"hipdnnGetCTCLossWorkspaceSize",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Algorithm functions
  {"cudnnCreateAlgorithmDescriptor",                      {"hipdnnCreateAlgorithmDescriptor",                      "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetAlgorithmDescriptor",                         {"hipdnnSetAlgorithmDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmDescriptor",                         {"hipdnnGetAlgorithmDescriptor",                         "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCopyAlgorithmDescriptor",                        {"hipdnnCopyAlgorithmDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyAlgorithmDescriptor",                     {"hipdnnDestroyAlgorithmDescriptor",                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreateAlgorithmPerformance",                     {"hipdnnCreateAlgorithmPerformance",                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetAlgorithmPerformance",                        {"hipdnnSetAlgorithmPerformance",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmPerformance",                        {"hipdnnGetAlgorithmPerformance",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyAlgorithmPerformance",                    {"hipdnnDestroyAlgorithmPerformance",                    "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetAlgorithmSpaceSize",                          {"hipdnnGetAlgorithmSpaceSize",                          "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSaveAlgorithm",                                  {"hipdnnSaveAlgorithm",                                  "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRestoreAlgorithm",                               {"hipdnnRestoreAlgorithm",                               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Clipping functions
  {"cudnnRNNSetClip",                                     {"hipdnnRNNSetClip",                                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnRNNGetClip",                                     {"hipdnnRNNGetClip",                                     "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Sequence functions
  {"cudnnCreateSeqDataDescriptor",                        {"hipdnnCreateSeqDataDescriptor",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroySeqDataDescriptor",                       {"hipdnnDestroySeqDataDescriptor",                       "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetSeqDataDescriptor",                           {"hipdnnSetSeqDataDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetSeqDataDescriptor",                           {"hipdnnGetSeqDataDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Multihead Attention functions
  {"cudnnCreateAttnDescriptor",                           {"hipdnnCreateAttnDescriptor",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyAttnDescriptor",                          {"hipdnnDestroyAttnDescriptor",                          "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetAttnDescriptor",                              {"hipdnnSetAttnDescriptor",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetAttnDescriptor",                              {"hipdnnGetAttnDescriptor",                              "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetMultiHeadAttnBuffers",                        {"hipdnnGetMultiHeadAttnBuffers",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetMultiHeadAttnWeights",                        {"hipdnnGetMultiHeadAttnWeights",                        "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnMultiHeadAttnForward",                           {"hipdnnMultiHeadAttnForward",                           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnMultiHeadAttnBackwardData",                      {"hipdnnMultiHeadAttnBackwardData",                      "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnMultiHeadAttnBackwardWeights",                   {"hipdnnMultiHeadAttnBackwardWeights",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},

  // cuDNN Fuse functions
  {"cudnnCreateFusedOpsConstParamPack",                   {"hipdnnCreateFusedOpsConstParamPack",                   "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyFusedOpsConstParamPack",                  {"hipdnnDestroyFusedOpsConstParamPack",                  "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetFusedOpsConstParamPackAttribute",             {"hipdnnSetFusedOpsConstParamPackAttribute",             "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetFusedOpsConstParamPackAttribute",             {"hipdnnGetFusedOpsConstParamPackAttribute",             "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreateFusedOpsVariantParamPack",                 {"hipdnnCreateFusedOpsVariantParamPack",                 "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyFusedOpsVariantParamPack",                {"hipdnnDestroyFusedOpsVariantParamPack",                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnSetFusedOpsVariantParamPackAttribute",           {"hipdnnSetFusedOpsVariantParamPackAttribute",           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnGetFusedOpsVariantParamPackAttribute",           {"hipdnnGetFusedOpsVariantParamPackAttribute",           "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnCreateFusedOpsPlan",                             {"hipdnnCreateFusedOpsPlan",                             "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnDestroyFusedOpsPlan",                            {"hipdnnDestroyFusedOpsPlan",                            "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnMakeFusedOpsPlan",                               {"hipdnnMakeFusedOpsPlan",                               "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsExecute",                                {"hipdnnFusedOpsExecute",                                "", CONV_LIB_FUNC, API_DNN, 2, HIP_UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_FUNCTION_VER_MAP {
};

const std::map<unsigned int, llvm::StringRef> CUDA_DNN_API_SECTION_MAP {
  {1, "CUNN Data types"},
  {2, "CUNN Functions"},
};
