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
const std::map<llvm::StringRef, hipCounter> CUDA_DNN_TYPE_NAME_MAP {
  // cuDNN defines
  {"CUDNN_VERSION",                                                  {"HIPDNN_VERSION",                                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1}},
  {"CUDNN_DIM_MAX",                                                  {"HIPDNN_DIM_MAX",                                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    //  8
  {"CUDNN_LRN_MIN_N",                                                {"HIPDNN_LRN_MIN_N",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    //  1
  {"CUDNN_LRN_MAX_N",                                                {"HIPDNN_LRN_MAX_N",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_LRN_MIN_K",                                                {"HIPDNN_LRN_MIN_K",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1e-5
  {"CUDNN_LRN_MIN_BETA",                                             {"HIPDNN_LRN_MIN_BETA",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0.01
  {"CUDNN_BN_MIN_EPSILON",                                           {"HIPDNN_BN_MIN_EPSILON",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1e-5
  {"CUDNN_SEV_ERROR_EN",                                             {"HIPDNN_SEV_ERROR_EN",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_WARNING_EN",                                           {"HIPDNN_SEV_WARNING_EN",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_INFO_EN",                                              {"HIPDNN_SEV_INFO_EN",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SEQDATA_DIM_COUNT",                                        {"HIPDNN_SEQDATA_DIM_COUNT",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_MAJOR",                                                    {"HIPDNN_MAJOR",                                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_MINOR",                                                    {"HIPDNN_MINOR",                                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_PATCHLEVEL",                                               {"HIPDNN_PATCHLEVEL",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_INFER_MAJOR",                                          {"HIPDNN_ADV_INFER_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_INFER_MINOR",                                          {"HIPDNN_ADV_INFER_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_INFER_PATCH",                                          {"HIPDNN_ADV_INFER_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_TRAIN_MAJOR",                                          {"HIPDNN_ADV_TRAIN_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_TRAIN_MINOR",                                          {"HIPDNN_ADV_TRAIN_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ADV_TRAIN_PATCH",                                          {"HIPDNN_ADV_TRAIN_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_INFER_MAJOR",                                          {"HIPDNN_CNN_INFER_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_INFER_MINOR",                                          {"HIPDNN_CNN_INFER_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_INFER_PATCH",                                          {"HIPDNN_CNN_INFER_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_TRAIN_MAJOR",                                          {"HIPDNN_CNN_TRAIN_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_TRAIN_MINOR",                                          {"HIPDNN_CNN_TRAIN_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CNN_TRAIN_PATCH",                                          {"HIPDNN_CNN_TRAIN_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_INFER_MAJOR",                                          {"HIPDNN_OPS_INFER_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_INFER_MINOR",                                          {"HIPDNN_OPS_INFER_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_INFER_PATCH",                                          {"HIPDNN_OPS_INFER_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_TRAIN_MAJOR",                                          {"HIPDNN_OPS_TRAIN_MAJOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_TRAIN_MINOR",                                          {"HIPDNN_OPS_TRAIN_MINOR",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_OPS_TRAIN_PATCH",                                          {"HIPDNN_OPS_TRAIN_PATCH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},

  // cuDNN enums
  {"cudnnStatus_t",                                                  {"hipdnnStatus_t",                                                  "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_STATUS_SUCCESS",                                           {"HIPDNN_STATUS_SUCCESS",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  0
  {"CUDNN_STATUS_NOT_INITIALIZED",                                   {"HIPDNN_STATUS_NOT_INITIALIZED",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  1
  {"CUDNN_STATUS_ALLOC_FAILED",                                      {"HIPDNN_STATUS_ALLOC_FAILED",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  2
  {"CUDNN_STATUS_BAD_PARAM",                                         {"HIPDNN_STATUS_BAD_PARAM",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  3
  {"CUDNN_STATUS_INTERNAL_ERROR",                                    {"HIPDNN_STATUS_INTERNAL_ERROR",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  4
  {"CUDNN_STATUS_INVALID_VALUE",                                     {"HIPDNN_STATUS_INVALID_VALUE",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  5
  {"CUDNN_STATUS_ARCH_MISMATCH",                                     {"HIPDNN_STATUS_ARCH_MISMATCH",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  6
  {"CUDNN_STATUS_MAPPING_ERROR",                                     {"HIPDNN_STATUS_MAPPING_ERROR",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  7
  {"CUDNN_STATUS_EXECUTION_FAILED",                                  {"HIPDNN_STATUS_EXECUTION_FAILED",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  8
  {"CUDNN_STATUS_NOT_SUPPORTED",                                     {"HIPDNN_STATUS_NOT_SUPPORTED",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  9
  {"CUDNN_STATUS_LICENSE_ERROR",                                     {"HIPDNN_STATUS_LICENSE_ERROR",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 10
  {"CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",                      {"HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",                      "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 11
  {"CUDNN_STATUS_RUNTIME_IN_PROGRESS",                               {"HIPDNN_STATUS_RUNTIME_IN_PROGRESS",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_STATUS_RUNTIME_FP_OVERFLOW",                               {"HIPDNN_STATUS_RUNTIME_FP_OVERFLOW",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 13
  {"CUDNN_STATUS_VERSION_MISMATCH",                                  {"HIPDNN_STATUS_VERSION_MISMATCH",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 14
  {"cudnnRuntimeTag_t",                                              {"hipdnnRuntimeTag_t",                                              "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnConvolutionMode_t",                                         {"hipdnnConvolutionMode_t",                                         "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION",                                              {"HIPDNN_CONVOLUTION",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CROSS_CORRELATION",                                        {"HIPDNN_CROSS_CORRELATION",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnTensorFormat_t",                                            {"hipdnnTensorFormat_t",                                            "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_TENSOR_NCHW",                                              {"HIPDNN_TENSOR_NCHW",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_TENSOR_NHWC",                                              {"HIPDNN_TENSOR_NHWC",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_TENSOR_NCHW_VECT_C",                                       {"HIPDNN_TENSOR_NCHW_VECT_C",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"cudnnFoldingDirection_t",                                        {"hipdnnFoldingDirection_t",                                        "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TRANSFORM_FOLD",                                           {"HIPDNN_TRANSFORM_FOLD",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0U
  {"CUDNN_TRANSFORM_UNFOLD",                                         {"HIPDNN_TRANSFORM_UNFOLD",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1U
  {"cudnnDataType_t",                                                {"hipdnnDataType_t",                                                "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_DATA_FLOAT",                                               {"HIPDNN_DATA_FLOAT",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_DATA_DOUBLE",                                              {"HIPDNN_DATA_DOUBLE",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_DATA_HALF",                                                {"HIPDNN_DATA_HALF",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_DATA_INT8",                                                {"HIPDNN_DATA_INT8",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_DATA_INT32",                                               {"HIPDNN_DATA_INT32",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_DATA_INT8x4",                                              {"HIPDNN_DATA_INT8x4",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_DATA_UINT8",                                               {"HIPDNN_DATA_UINT8",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 6
  {"CUDNN_DATA_UINT8x4",                                             {"HIPDNN_DATA_UINT8x4",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 7
  {"CUDNN_DATA_INT8x32",                                             {"HIPDNN_DATA_INT8x32",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 8
  {"CUDNN_DATA_BFLOAT16",                                            {"HIPDNN_DATA_BFLOAT16",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 9
  {"CUDNN_DATA_INT64",                                               {"HIPDNN_DATA_INT64",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 10
  {"cudnnErrQueryMode_t",                                            {"hipdnnErrQueryMode_t",                                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ERRQUERY_RAWCODE",                                         {"HIPDNN_ERRQUERY_RAWCODE",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ERRQUERY_NONBLOCKING",                                     {"HIPDNN_ERRQUERY_NONBLOCKING",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_ERRQUERY_BLOCKING",                                        {"HIPDNN_ERRQUERY_BLOCKING",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"cudnnSeverity_t",                                                {"hipdnnSeverity_t",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SEV_FATAL",                                                {"HIPDNN_SEV_FATAL",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_SEV_ERROR",                                                {"HIPDNN_SEV_ERROR",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_SEV_WARNING",                                              {"HIPDNN_SEV_WARNING",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_SEV_INFO",                                                 {"HIPDNN_SEV_INFO",                                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"cudnnConvolutionFwdAlgo_t",                                      {"hipdnnConvolutionFwdAlgo_t",                                      "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",                       {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",               {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",                                {"HIPDNN_CONVOLUTION_FWD_ALGO_GEMM",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",                              {"HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT",                                 {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",                          {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                            {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",                   {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 7
  {"CUDNN_CONVOLUTION_FWD_ALGO_COUNT",                               {"HIPDNN_CONVOLUTION_FWD_ALGO_COUNT",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 8
  {"cudnnConvolutionFwdPreference_t",                                {"hipdnnConvolutionFwdPreference_t",                                "", CONV_TYPE, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_FWD_NO_WORKSPACE",                             {"HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_FWD_PREFER_FASTEST",                           {"HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",                  {"HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnDeterminism_t",                                             {"hipdnnDeterminism_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NON_DETERMINISTIC",                                        {"HIPDNN_NON_DETERMINISTIC",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_DETERMINISTIC",                                            {"HIPDNN_DETERMINISTIC",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnDivNormMode_t",                                             {"hipdnnDivNormMode_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_DIVNORM_PRECOMPUTED_MEANS",                                {"HIPDNN_DIVNORM_PRECOMPUTED_MEANS",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"cudnnCTCLossAlgo_t",                                             {"hipdnnCTCLossAlgo_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",                              {"HIPDNN_CTC_LOSS_ALGO_DETERMINISTIC",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",                          {"HIPDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnLRNMode_t",                                                 {"hipdnnLRNMode_t",                                                 "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_LRN_CROSS_CHANNEL_DIM1",                                   {"HIPDNN_LRN_CROSS_CHANNEL",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0 vs 1
  {"cudnnRNNInputMode_t",                                            {"hipdnnRNNInputMode_t",                                            "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_LINEAR_INPUT",                                             {"HIPDNN_LINEAR_INPUT",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SKIP_INPUT",                                               {"HIPDNN_SKIP_INPUT",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnDirectionMode_t",                                           {"hipdnnDirectionMode_t",                                           "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_UNIDIRECTIONAL",                                           {"HIPDNN_UNIDIRECTIONAL",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_BIDIRECTIONAL",                                            {"HIPDNN_BIDIRECTIONAL",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnMathType_t",                                                {"hipdnnMathType_t",                                                "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_DEFAULT_MATH",                                             {"HIPDNN_DEFAULT_MATH",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_TENSOR_OP_MATH",                                           {"HIPDNN_TENSOR_OP_MATH",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",                          {"HIPDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_FMA_MATH",                                                 {"HIPDNN_FMA_MATH",                                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"cudnnNanPropagation_t",                                          {"hipdnnNanPropagation_t",                                          "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_NOT_PROPAGATE_NAN",                                        {"HIPDNN_NOT_PROPAGATE_NAN",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_PROPAGATE_NAN",                                            {"HIPDNN_PROPAGATE_NAN",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnConvolutionBwdDataAlgo_t",                                  {"hipdnnConvolutionBwdDataAlgo_t",                                  "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",                              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",                              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                            {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",                     {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",                       {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",                          {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"cudnnConvolutionBwdFilterAlgo_t",                                {"hipdnnConvolutionBwdFilterAlgo_t",                                "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",                          {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",                     {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",                   {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",                        {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 7
  {"cudnnConvolutionBwdFilterPreference_t",                          {"hipdnnConvolutionBwdFilterPreference_t",                          "", CONV_TYPE, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",                      {"HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",                    {"HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",           {"HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",           "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnRNNAlgo_t",                                                 {"hipdnnRNNAlgo_t",                                                 "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_ALGO_STANDARD",                                        {"HIPDNN_RNN_ALGO_STANDARD",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_ALGO_PERSIST_STATIC",                                  {"HIPDNN_RNN_ALGO_PERSIST_STATIC",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_RNN_ALGO_PERSIST_DYNAMIC",                                 {"HIPDNN_RNN_ALGO_PERSIST_DYNAMIC",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",                          {"HIPDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_RNN_ALGO_COUNT",                                           {"HIPDNN_RNN_ALGO_COUNT",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"cudnnRNNMode_t",                                                 {"hipdnnRNNMode_t",                                                 "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_RELU",                                                 {"HIPDNN_RNN_RELU",                                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_TANH",                                                 {"HIPDNN_RNN_TANH",                                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_LSTM",                                                     {"HIPDNN_LSTM",                                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_GRU",                                                      {"HIPDNN_GRU",                                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"cudnnRNNBiasMode_t",                                             {"hipdnnRNNBiasMode_t",                                             "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_NO_BIAS",                                              {"HIPDNN_RNN_NO_BIAS",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_SINGLE_INP_BIAS",                                      {"HIPDNN_RNN_WITH_BIAS",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_RNN_DOUBLE_BIAS",                                          {"HIPDNN_RNN_WITH_BIAS",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_RNN_SINGLE_REC_BIAS",                                      {"HIPDNN_RNN_WITH_BIAS",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnOpTensorOp_t",                                              {"hipdnnOpTensorOp_t",                                              "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_OP_TENSOR_ADD",                                            {"HIPDNN_OP_TENSOR_ADD",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_OP_TENSOR_MUL",                                            {"HIPDNN_OP_TENSOR_MUL",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_OP_TENSOR_MIN",                                            {"HIPDNN_OP_TENSOR_MIN",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_OP_TENSOR_MAX",                                            {"HIPDNN_OP_TENSOR_MAX",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_OP_TENSOR_SQRT",                                           {"HIPDNN_OP_TENSOR_SQRT",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_OP_TENSOR_NOT",                                            {"HIPDNN_OP_TENSOR_NOT",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"cudnnReduceTensorOp_t",                                          {"hipdnnReduceTensorOp_t",                                          "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_REDUCE_TENSOR_ADD",                                        {"HIPDNN_REDUCE_TENSOR_ADD",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_REDUCE_TENSOR_MUL",                                        {"HIPDNN_REDUCE_TENSOR_MUL",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_REDUCE_TENSOR_MIN",                                        {"HIPDNN_REDUCE_TENSOR_MIN",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_REDUCE_TENSOR_MAX",                                        {"HIPDNN_REDUCE_TENSOR_MAX",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_REDUCE_TENSOR_AMAX",                                       {"HIPDNN_REDUCE_TENSOR_AMAX",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_REDUCE_TENSOR_AVG",                                        {"HIPDNN_REDUCE_TENSOR_AVG",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_REDUCE_TENSOR_NORM1",                                      {"HIPDNN_REDUCE_TENSOR_NORM1",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"CUDNN_REDUCE_TENSOR_NORM2",                                      {"HIPDNN_REDUCE_TENSOR_NORM2",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 7
  {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                               {"HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 8
  {"cudnnReduceTensorIndices_t",                                     {"hipdnnReduceTensorIndices_t",                                     "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_REDUCE_TENSOR_NO_INDICES",                                 {"HIPDNN_REDUCE_TENSOR_NO_INDICES",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_REDUCE_TENSOR_FLATTENED_INDICES",                          {"HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnConvolutionBwdDataPreference_t",                            {"hipdnnConvolutionBwdDataPreference_t",                            "", CONV_TYPE, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",                        {"HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",                      {"HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",             {"HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",             "", CONV_NUMERIC_LITERAL, API_DNN, 1, CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnIndicesType_t",                                             {"hipdnnIndicesType_t",                                             "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_32BIT_INDICES",                                            {"HIPDNN_32BIT_INDICES",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_64BIT_INDICES",                                            {"HIPDNN_64BIT_INDICES",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_16BIT_INDICES",                                            {"HIPDNN_16BIT_INDICES",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_8BIT_INDICES",                                             {"HIPDNN_8BIT_INDICES",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"cudnnSoftmaxAlgorithm_t",                                        {"hipdnnSoftmaxAlgorithm_t",                                        "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_SOFTMAX_FAST",                                             {"HIPDNN_SOFTMAX_FAST",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SOFTMAX_ACCURATE",                                         {"HIPDNN_SOFTMAX_ACCURATE",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_SOFTMAX_LOG",                                              {"HIPDNN_SOFTMAX_LOG",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"cudnnSoftmaxMode_t",                                             {"hipdnnSoftmaxMode_t",                                             "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_SOFTMAX_MODE_INSTANCE",                                    {"HIPDNN_SOFTMAX_MODE_INSTANCE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SOFTMAX_MODE_CHANNEL",                                     {"HIPDNN_SOFTMAX_MODE_CHANNEL",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnPoolingMode_t",                                             {"hipdnnPoolingMode_t",                                             "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_POOLING_MAX",                                              {"HIPDNN_POOLING_MAX",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",                    {"HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",                    {"HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_POOLING_MAX_DETERMINISTIC",                                {"HIPDNN_POOLING_MAX_DETERMINISTIC",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"cudnnActivationMode_t",                                          {"hipdnnActivationMode_t",                                          "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_ACTIVATION_SIGMOID",                                       {"HIPDNN_ACTIVATION_SIGMOID",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_ACTIVATION_RELU",                                          {"HIPDNN_ACTIVATION_RELU",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_ACTIVATION_TANH",                                          {"HIPDNN_ACTIVATION_TANH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_ACTIVATION_CLIPPED_RELU",                                  {"HIPDNN_ACTIVATION_CLIPPED_RELU",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_ACTIVATION_ELU",                                           {"HIPDNN_ACTIVATION_ELU",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_ACTIVATION_IDENTITY",                                      {"HIPDNN_ACTIVATION_PATHTRU",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"cudnnBatchNormMode_t",                                           {"hipdnnBatchNormMode_t",                                           "", CONV_TYPE, API_DNN, 1}},
  {"CUDNN_BATCHNORM_PER_ACTIVATION",                                 {"HIPDNN_BATCHNORM_PER_ACTIVATION",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_BATCHNORM_SPATIAL",                                        {"HIPDNN_BATCHNORM_SPATIAL",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",                             {"HIPDNN_BATCHNORM_SPATIAL_PERSISTENT",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"cudnnSamplerType_t",                                             {"hipdnnSamplerType_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SAMPLER_BILINEAR",                                         {"HIPDNN_SAMPLER_BILINEAR",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"cudnnBatchNormOps_t",                                            {"hipdnnBatchNormOps_t",                                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BATCHNORM_OPS_BN",                                         {"HIPDNN_BATCHNORM_OPS_BN",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",                              {"HIPDNN_BATCHNORM_OPS_BN_ACTIVATION",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",                          {"HIPDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"cudnnRNNClipMode_t",                                             {"hipdnnRNNClipMode_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_CLIP_NONE",                                            {"HIPDNN_RNN_CLIP_NONE",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_CLIP_MINMAX",                                          {"HIPDNN_RNN_CLIP_MINMAX",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnRNNDataLayout_t",                                           {"hipdnnRNNDataLayout_t",                                           "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",                       {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",                         {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",                     {"HIPDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"cudnnRNNPaddingMode_t",                                          {"hipdnnRNNPaddingMode_t",                                          "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_RNN_PADDED_IO_DISABLED",                                   {"HIPDNN_RNN_PADDED_IO_DISABLED",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_RNN_PADDED_IO_ENABLED",                                    {"HIPDNN_RNN_PADDED_IO_ENABLED",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnSeqDataAxis_t",                                             {"hipdnnSeqDataAxis_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_SEQDATA_TIME_DIM",                                         {"HIPDNN_SEQDATA_TIME_DIM",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_SEQDATA_BATCH_DIM",                                        {"HIPDNN_SEQDATA_BATCH_DIM",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_SEQDATA_BEAM_DIM",                                         {"HIPDNN_SEQDATA_BEAM_DIM",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_SEQDATA_VECT_DIM",                                         {"HIPDNN_SEQDATA_VECT_DIM",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"cudnnAttnQueryMap_t",                                            {"hipdnnAttnQueryMap_t",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ATTN_QUERYMAP_ALL_TO_ONE",                                 {"HIPDNN_ATTN_QUERYMAP_ALL_TO_ONE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_QUERYMAP_ONE_TO_ONE",                                 {"HIPDNN_ATTN_QUERYMAP_ONE_TO_ONE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1U << 0
  {"CUDNN_ATTN_DISABLE_PROJ_BIASES",                                 {"HIPDNN_ATTN_DISABLE_PROJ_BIASES",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_ENABLE_PROJ_BIASES",                                  {"HIPDNN_ATTN_ENABLE_PROJ_BIASES",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1U << 1
  {"cudnnMultiHeadAttnWeightKind_t",                                 {"hipdnnMultiHeadAttnWeightKind_t",                                 "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_MH_ATTN_Q_WEIGHTS",                                        {"HIPDNN_MH_ATTN_Q_WEIGHTS",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_MH_ATTN_K_WEIGHTS",                                        {"HIPDNN_MH_ATTN_K_WEIGHTS",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_MH_ATTN_V_WEIGHTS",                                        {"HIPDNN_MH_ATTN_V_WEIGHTS",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_MH_ATTN_O_WEIGHTS",                                        {"HIPDNN_MH_ATTN_O_WEIGHTS",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_MH_ATTN_Q_BIASES",                                         {"HIPDNN_MH_ATTN_Q_BIASES",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_MH_ATTN_K_BIASES",                                         {"HIPDNN_MH_ATTN_K_BIASES",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_MH_ATTN_V_BIASES",                                         {"HIPDNN_MH_ATTN_V_BIASES",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_MH_ATTN_O_BIASES",                                         {"HIPDNN_MH_ATTN_O_BIASES",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_ATTN_WKIND_COUNT",                                         {"HIPDNN_ATTN_WKIND_COUNT",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 8
  {"cudnnWgradMode_t",                                               {"hipdnnWgradMode_t",                                               "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_WGRAD_MODE_ADD",                                           {"HIPDNN_WGRAD_MODE_ADD",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_WGRAD_MODE_SET",                                           {"HIPDNN_WGRAD_MODE_SET",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnReorderType_t",                                             {"hipdnnReorderType_t",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_DEFAULT_REORDER",                                          {"HIPDNN_DEFAULT_REORDER",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_NO_REORDER",                                               {"HIPDNN_NO_REORDER",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnLossNormalizationMode_t",                                   {"hipdnnLossNormalizationMode_t",                                   "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LOSS_NORMALIZATION_NONE",                                  {"HIPDNN_LOSS_NORMALIZATION_NONE",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_LOSS_NORMALIZATION_SOFTMAX",                               {"HIPDNN_LOSS_NORMALIZATION_SOFTMAX",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnFusedOps_t",                                                {"hipdnnFusedOps_t",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",                 {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",                        {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",                    {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",                   {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",                     {"HIPDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",              {"HIPDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",                        {"HIPDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 6
  {"cudnnFusedOpsConstParamLabel_t",                                 {"hipdnnFusedOpsConstParamLabel_t",                                 "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_PARAM_XDESC",                                              {"HIPDNN_PARAM_XDESC",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PARAM_XDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_XDATA_PLACEHOLDER",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PARAM_BN_MODE",                                            {"HIPDNN_PARAM_BN_MODE",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_PARAM_BN_EQSCALEBIAS_DESC",                                {"HIPDNN_PARAM_BN_EQSCALEBIAS_DESC",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                             {"HIPDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                              {"HIPDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_PARAM_ACTIVATION_DESC",                                    {"HIPDNN_PARAM_ACTIVATION_DESC",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_PARAM_CONV_DESC",                                          {"HIPDNN_PARAM_CONV_DESC",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_PARAM_WDESC",                                              {"HIPDNN_PARAM_WDESC",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 8
  {"CUDNN_PARAM_WDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_WDATA_PLACEHOLDER",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 9
  {"CUDNN_PARAM_DWDESC",                                             {"HIPDNN_PARAM_DWDESC",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 10
  {"CUDNN_PARAM_DWDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DWDATA_PLACEHOLDER",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 11
  {"CUDNN_PARAM_YDESC",                                              {"HIPDNN_PARAM_YDESC",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_PARAM_YDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_YDATA_PLACEHOLDER",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 13
  {"CUDNN_PARAM_DYDESC",                                             {"HIPDNN_PARAM_DYDESC",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 14
  {"CUDNN_PARAM_DYDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DYDATA_PLACEHOLDER",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 15
  {"CUDNN_PARAM_YSTATS_DESC",                                        {"HIPDNN_PARAM_YSTATS_DESC",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_PARAM_YSUM_PLACEHOLDER",                                   {"HIPDNN_PARAM_YSUM_PLACEHOLDER",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 17
  {"CUDNN_PARAM_YSQSUM_PLACEHOLDER",                                 {"HIPDNN_PARAM_YSQSUM_PLACEHOLDER",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 18
  {"CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",                          {"HIPDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 19
  {"CUDNN_PARAM_BN_SCALE_PLACEHOLDER",                               {"HIPDNN_PARAM_BN_SCALE_PLACEHOLDER",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 20
  {"CUDNN_PARAM_BN_BIAS_PLACEHOLDER",                                {"HIPDNN_PARAM_BN_BIAS_PLACEHOLDER",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 21
  {"CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",                          {"HIPDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 22
  {"CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",                        {"HIPDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 23
  {"CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",                        {"HIPDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 24
  {"CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",                         {"HIPDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 25
  {"CUDNN_PARAM_ZDESC",                                              {"HIPDNN_PARAM_ZDESC",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 26
  {"CUDNN_PARAM_ZDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_ZDATA_PLACEHOLDER",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 27
  {"CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                              {"HIPDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 28
  {"CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                           {"HIPDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 29
  {"CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                            {"HIPDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 30
  {"CUDNN_PARAM_ACTIVATION_BITMASK_DESC",                            {"HIPDNN_PARAM_ACTIVATION_BITMASK_DESC",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 31
  {"CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",                     {"HIPDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 32
  {"CUDNN_PARAM_DXDESC",                                             {"HIPDNN_PARAM_DXDESC",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 33
  {"CUDNN_PARAM_DXDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DXDATA_PLACEHOLDER",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 34
  {"CUDNN_PARAM_DZDESC",                                             {"HIPDNN_PARAM_DZDESC",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 35
  {"CUDNN_PARAM_DZDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DZDATA_PLACEHOLDER",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 36
  {"CUDNN_PARAM_BN_DSCALE_PLACEHOLDER",                              {"HIPDNN_PARAM_BN_DSCALE_PLACEHOLDER",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 37
  {"CUDNN_PARAM_BN_DBIAS_PLACEHOLDER",                               {"HIPDNN_PARAM_BN_DBIAS_PLACEHOLDER",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 38
  {"cudnnFusedOpsPointerPlaceHolder_t",                              {"hipdnnActivationMode_t",                                          "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_PTR_NULL",                                                 {"HIPDNN_ACTIVATION_SIGMOID",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PTR_ELEM_ALIGNED",                                         {"HIPDNN_ACTIVATION_RELU",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PTR_16B_ALIGNED",                                          {"HIPDNN_ACTIVATION_TANH",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"cudnnFusedOpsVariantParamLabel_t",                               {"hipdnnFusedOpsVariantParamLabel_t",                               "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_PTR_XDATA",                                                {"HIPDNN_PTR_XDATA",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_PTR_BN_EQSCALE",                                           {"HIPDNN_PTR_BN_EQSCALE",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_PTR_BN_EQBIAS",                                            {"HIPDNN_PTR_BN_EQBIAS",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_PTR_WDATA",                                                {"HIPDNN_PTR_WDATA",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_PTR_DWDATA",                                               {"HIPDNN_PTR_DWDATA",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_PTR_YDATA",                                                {"HIPDNN_PTR_YDATA",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_PTR_DYDATA",                                               {"HIPDNN_PTR_DYDATA",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_PTR_YSUM",                                                 {"HIPDNN_PTR_YSUM",                                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_PTR_YSQSUM",                                               {"HIPDNN_PTR_YSQSUM",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 8
  {"CUDNN_PTR_WORKSPACE",                                            {"HIPDNN_PTR_WORKSPACE",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 9
  {"CUDNN_PTR_BN_SCALE",                                             {"HIPDNN_PTR_BN_SCALE",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 10
  {"CUDNN_PTR_BN_BIAS",                                              {"HIPDNN_PTR_BN_BIAS",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 11
  {"CUDNN_PTR_BN_SAVED_MEAN",                                        {"HIPDNN_PTR_BN_SAVED_MEAN",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 12
  {"CUDNN_PTR_BN_SAVED_INVSTD",                                      {"HIPDNN_PTR_BN_SAVED_INVSTD",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 13
  {"CUDNN_PTR_BN_RUNNING_MEAN",                                      {"HIPDNN_PTR_BN_RUNNING_MEAN",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 14
  {"CUDNN_PTR_BN_RUNNING_VAR",                                       {"HIPDNN_PTR_BN_RUNNING_VAR",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 15
  {"CUDNN_PTR_ZDATA",                                                {"HIPDNN_PTR_ZDATA",                                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 16
  {"CUDNN_PTR_BN_Z_EQSCALE",                                         {"HIPDNN_PTR_BN_Z_EQSCALE",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 17
  {"CUDNN_PTR_BN_Z_EQBIAS",                                          {"HIPDNN_PTR_BN_Z_EQBIAS",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 18
  {"CUDNN_PTR_ACTIVATION_BITMASK",                                   {"HIPDNN_PTR_ACTIVATION_BITMASK",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 19
  {"CUDNN_PTR_DXDATA",                                               {"HIPDNN_PTR_DXDATA",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 20
  {"CUDNN_PTR_DZDATA",                                               {"HIPDNN_PTR_DZDATA",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 21
  {"CUDNN_PTR_BN_DSCALE",                                            {"HIPDNN_PTR_BN_DSCALE",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 22
  {"CUDNN_PTR_BN_DBIAS",                                             {"HIPDNN_PTR_BN_DBIAS",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 23
  {"CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",                    {"HIPDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 100
  {"CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",                     {"HIPDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 101
  {"CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",                          {"HIPDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 102
  {"CUDNN_SCALAR_DOUBLE_BN_EPSILON",                                 {"HIPDNN_SCALAR_DOUBLE_BN_EPSILON",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 103
  {"cudnnForwardMode_t",                                             {"hipdnnForwardMode_t",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_FWD_MODE_INFERENCE",                                       {"HIPDNN_FWD_MODE_INFERENCE",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_FWD_MODE_TRAINING",                                        {"HIPDNN_FWD_MODE_TRAINING",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"cudnnPointwiseMode_t",                                           {"hipdnnPointwiseMode_t",                                           "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_POINTWISE_ADD",                                            {"HIPDNN_POINTWISE_ADD",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_POINTWISE_MUL",                                            {"HIPDNN_POINTWISE_MUL",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_POINTWISE_MIN",                                            {"HIPDNN_POINTWISE_MIN",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_POINTWISE_MAX",                                            {"HIPDNN_POINTWISE_MAX",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_POINTWISE_SQRT",                                           {"HIPDNN_POINTWISE_SQRT",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_POINTWISE_RELU_FWD",                                       {"HIPDNN_POINTWISE_RELU_FWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 100
  {"CUDNN_POINTWISE_TANH_FWD",                                       {"HIPDNN_POINTWISE_TANH_FWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 101
  {"CUDNN_POINTWISE_SIGMOID_FWD",                                    {"HIPDNN_POINTWISE_SIGMOID_FWD",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 102
  {"CUDNN_POINTWISE_ELU_FWD",                                        {"HIPDNN_POINTWISE_ELU_FWD",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 103
  {"CUDNN_POINTWISE_GELU_FWD",                                       {"HIPDNN_POINTWISE_GELU_FWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 104
  {"CUDNN_POINTWISE_SOFTPLUS_FWD",                                   {"HIPDNN_POINTWISE_SOFTPLUS_FWD",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 105
  {"CUDNN_POINTWISE_SWISH_FWD",                                      {"HIPDNN_POINTWISE_SWISH_FWD",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 106
  {"CUDNN_POINTWISE_RELU_BWD",                                       {"HIPDNN_POINTWISE_RELU_BWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 200
  {"CUDNN_POINTWISE_TANH_BWD",                                       {"HIPDNN_POINTWISE_TANH_BWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 201
  {"CUDNN_POINTWISE_SIGMOID_BWD",                                    {"HIPDNN_POINTWISE_SIGMOID_BWD",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 202
  {"CUDNN_POINTWISE_ELU_BWD",                                        {"HIPDNN_POINTWISE_ELU_BWD",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 203
  {"CUDNN_POINTWISE_GELU_BWD",                                       {"HIPDNN_POINTWISE_GELU_BWD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 204
  {"CUDNN_POINTWISE_SOFTPLUS_BWD",                                   {"HIPDNN_POINTWISE_SOFTPLUS_BWD",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 205
  {"CUDNN_POINTWISE_SWISH_BWD",                                      {"HIPDNN_POINTWISE_SWISH_BWD",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 206
  {"cudnnGenStatsMode_t",                                            {"hipdnnGenStatsMode_t",                                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_GENSTATS_SUM_SQSUM",                                       {"HIPDNN_GENSTATS_SUM_SQSUM",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"cudnnBackendAttributeName_t",                                    {"hipdnnBackendAttributeName_t",                                    "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_ATTR_POINTWISE_MODE",                                      {"HIPDNN_ATTR_POINTWISE_MODE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_ATTR_POINTWISE_MATH_PREC",                                 {"HIPDNN_ATTR_POINTWISE_MATH_PREC",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1
  {"CUDNN_ATTR_POINTWISE_NAN_PROPAGATION",                           {"HIPDNN_ATTR_POINTWISE_NAN_PROPAGATION",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 2
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",                           {"HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 3
  {"CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",                           {"HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 4
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",                     {"HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 5
  {"CUDNN_ATTR_POINTWISE_ELU_ALPHA",                                 {"HIPDNN_ATTR_POINTWISE_ELU_ALPHA",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 6
  {"CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA",                             {"HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 7
  {"CUDNN_ATTR_POINTWISE_SWISH_BETA",                                {"HIPDNN_ATTR_POINTWISE_SWISH_BETA",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 8
  {"CUDNN_ATTR_CONVOLUTION_COMP_TYPE",                               {"HIPDNN_ATTR_CONVOLUTION_COMP_TYPE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 100
  {"CUDNN_ATTR_CONVOLUTION_CONV_MODE",                               {"HIPDNN_ATTR_CONVOLUTION_CONV_MODE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 101
  {"CUDNN_ATTR_CONVOLUTION_DILATIONS",                               {"HIPDNN_ATTR_CONVOLUTION_DILATIONS",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 102
  {"CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES",                          {"HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 103
  {"CUDNN_ATTR_CONVOLUTION_POST_PADDINGS",                           {"HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 104
  {"CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS",                            {"HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 105
  {"CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",                            {"HIPDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 106
  {"CUDNN_ATTR_ENGINEHEUR_MODE",                                     {"HIPDNN_ATTR_ENGINEHEUR_MODE",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 200
  {"CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",                          {"HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 201
  {"CUDNN_ATTR_ENGINEHEUR_RESULTS",                                  {"HIPDNN_ATTR_ENGINEHEUR_RESULTS",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 202
  {"CUDNN_ATTR_ENGINECFG_ENGINE",                                    {"HIPDNN_ATTR_ENGINECFG_ENGINE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 300
  {"CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",                         {"HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 301
  {"CUDNN_ATTR_ENGINECFG_KNOB_CHOICES",                              {"HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 302
  {"CUDNN_ATTR_EXECUTION_PLAN_HANDLE",                               {"HIPDNN_ATTR_EXECUTION_PLAN_HANDLE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 400
  {"CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",                        {"HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 401
  {"CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",                       {"HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 402
  {"CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",           {"HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 403
  {"CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",           {"HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 404
  {"CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",                         {"HIPDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 500
  {"CUDNN_ATTR_INTERMEDIATE_INFO_SIZE",                              {"HIPDNN_ATTR_INTERMEDIATE_INFO_SIZE",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 501
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",               {"HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 502
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",              {"HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 503
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",                               {"HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 600
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",                              {"HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 601
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 700
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",                  {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 701
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",             {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 702
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 703
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 704
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 705
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",                {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 706
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 707
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",            {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 708
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",                    {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 709
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",                   {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 710
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",                   {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 711
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",              {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 712
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",               {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 713
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",          {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 714
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 715
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",                  {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 716
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 717
  {"CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",                   {"HIPDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 750
  {"CUDNN_ATTR_OPERATION_POINTWISE_XDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_XDESC",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 751
  {"CUDNN_ATTR_OPERATION_POINTWISE_BDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_BDESC",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 752
  {"CUDNN_ATTR_OPERATION_POINTWISE_YDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_YDESC",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 753
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_ALPHA1",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 754
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_ALPHA2",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 755
  {"CUDNN_ATTR_OPERATION_POINTWISE_DXDESC",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_DXDESC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 756
  {"CUDNN_ATTR_OPERATION_POINTWISE_DYDESC",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_DYDESC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 757
  {"CUDNN_ATTR_OPERATION_GENSTATS_MODE",                             {"HIPDNN_ATTR_OPERATION_GENSTATS_MODE",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 770
  {"CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",                        {"HIPDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 771
  {"CUDNN_ATTR_OPERATION_GENSTATS_XDESC",                            {"HIPDNN_ATTR_OPERATION_GENSTATS_XDESC",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 772
  {"CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC",                          {"HIPDNN_ATTR_OPERATION_GENSTATS_SUMDESC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 773
  {"CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",                        {"HIPDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 774
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 780
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",                     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 781
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 782
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",                 {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 783
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 784
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",                     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 785
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",        {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 786
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",         {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 787
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 788
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",      {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 789
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",               {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 790
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",            {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 791
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",                 {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 792
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",                  {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 793
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",              {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 794
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",                  {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 795
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",       {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 796
  {"CUDNN_ATTR_OPERATIONGRAPH_HANDLE",                               {"HIPDNN_ATTR_OPERATIONGRAPH_HANDLE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 800
  {"CUDNN_ATTR_OPERATIONGRAPH_OPS",                                  {"HIPDNN_ATTR_OPERATIONGRAPH_OPS",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 801
  {"CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",                  {"HIPDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 802
  {"CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT",                               {"HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 900
  {"CUDNN_ATTR_TENSOR_DATA_TYPE",                                    {"HIPDNN_ATTR_TENSOR_DATA_TYPE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 901
  {"CUDNN_ATTR_TENSOR_DIMENSIONS",                                   {"HIPDNN_ATTR_TENSOR_DIMENSIONS",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 902
  {"CUDNN_ATTR_TENSOR_STRIDES",                                      {"HIPDNN_ATTR_TENSOR_STRIDES",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 903
  {"CUDNN_ATTR_TENSOR_VECTOR_COUNT",                                 {"HIPDNN_ATTR_TENSOR_VECTOR_COUNT",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 904
  {"CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",                         {"HIPDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 905
  {"CUDNN_ATTR_TENSOR_UNIQUE_ID",                                    {"HIPDNN_ATTR_TENSOR_UNIQUE_ID",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 906
  {"CUDNN_ATTR_TENSOR_IS_VIRTUAL",                                   {"HIPDNN_ATTR_TENSOR_IS_VIRTUAL",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 907
  {"CUDNN_ATTR_TENSOR_IS_BY_VALUE",                                  {"HIPDNN_ATTR_TENSOR_IS_BY_VALUE",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 908
  {"CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",                             {"HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1000
  {"CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",                          {"HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1001
  {"CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES",                          {"HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1002
  {"CUDNN_ATTR_VARIANT_PACK_WORKSPACE",                              {"HIPDNN_ATTR_VARIANT_PACK_WORKSPACE",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1003
  {"CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID",                              {"HIPDNN_ATTR_LAYOUT_INFO_TENSOR_UID",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1100
  {"CUDNN_ATTR_LAYOUT_INFO_TYPES",                                   {"HIPDNN_ATTR_LAYOUT_INFO_TYPES",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1101
  {"CUDNN_ATTR_KNOB_INFO_TYPE",                                      {"HIPDNN_ATTR_KNOB_INFO_TYPE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1200
  {"CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",                             {"HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1201
  {"CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",                             {"HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1202
  {"CUDNN_ATTR_KNOB_INFO_STRIDE",                                    {"HIPDNN_ATTR_KNOB_INFO_STRIDE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1203
  {"CUDNN_ATTR_ENGINE_OPERATION_GRAPH",                              {"HIPDNN_ATTR_ENGINE_OPERATION_GRAPH",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1300
  {"CUDNN_ATTR_ENGINE_GLOBAL_INDEX",                                 {"HIPDNN_ATTR_ENGINE_GLOBAL_INDEX",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1301
  {"CUDNN_ATTR_ENGINE_KNOB_INFO",                                    {"HIPDNN_ATTR_ENGINE_KNOB_INFO",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1302
  {"CUDNN_ATTR_ENGINE_NUMERICAL_NOTE",                               {"HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1303
  {"CUDNN_ATTR_ENGINE_LAYOUT_INFO",                                  {"HIPDNN_ATTR_ENGINE_LAYOUT_INFO",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1304
  {"CUDNN_ATTR_MATMUL_COMP_TYPE",                                    {"HIPDNN_ATTR_MATMUL_COMP_TYPE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1500
  {"CUDNN_ATTR_OPERATION_MATMUL_ADESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_ADESC",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1520
  {"CUDNN_ATTR_OPERATION_MATMUL_BDESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_BDESC",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1521
  {"CUDNN_ATTR_OPERATION_MATMUL_CDESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_CDESC",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1522
  {"CUDNN_ATTR_OPERATION_MATMUL_DESC",                               {"HIPDNN_ATTR_OPERATION_MATMUL_DESC",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1523
  {"CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",    {"HIPDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1524
  {"CUDNN_ATTR_REDUCTION_OPERATOR",                                  {"HIPDNN_ATTR_REDUCTION_OPERATOR",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1600
  {"CUDNN_ATTR_REDUCTION_COMP_TYPE",                                 {"HIPDNN_ATTR_REDUCTION_COMP_TYPE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1601
  {"CUDNN_ATTR_OPERATION_REDUCTION_XDESC",                           {"HIPDNN_ATTR_OPERATION_REDUCTION_XDESC",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1610
  {"CUDNN_ATTR_OPERATION_REDUCTION_YDESC",                           {"HIPDNN_ATTR_OPERATION_REDUCTION_YDESC",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1611
  {"CUDNN_ATTR_OPERATION_REDUCTION_DESC",                            {"HIPDNN_ATTR_OPERATION_REDUCTION_DESC",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 1612
  {"cudnnBackendAttributeType_t",                                    {"hipdnnBackendAttributeType_t",                                    "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_HANDLE",                                              {"HIPDNN_TYPE_HANDLE",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_DATA_TYPE",                                           {"HIPDNN_TYPE_DATA_TYPE",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_BOOLEAN",                                             {"HIPDNN_TYPE_BOOLEAN",                                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_INT64",                                               {"HIPDNN_TYPE_INT64",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_FLOAT",                                               {"HIPDNN_TYPE_FLOAT",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_DOUBLE",                                              {"HIPDNN_TYPE_FLOAT",                                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_VOID_PTR",                                            {"HIPDNN_TYPE_VOID_PTR",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_CONVOLUTION_MODE",                                    {"HIPDNN_TYPE_CONVOLUTION_MODE",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_HEUR_MODE",                                           {"HIPDNN_TYPE_HEUR_MODE",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_KNOB_TYPE",                                           {"HIPDNN_TYPE_KNOB_TYPE",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_NAN_PROPOGATION",                                     {"HIPDNN_TYPE_NAN_PROPOGATION",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_NUMERICAL_NOTE",                                      {"HIPDNN_TYPE_NUMERICAL_NOTE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_LAYOUT_TYPE",                                         {"HIPDNN_TYPE_LAYOUT_TYPE",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_ATTRIB_NAME",                                         {"HIPDNN_TYPE_ATTRIB_NAME",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_POINTWISE_MODE",                                      {"HIPDNN_TYPE_POINTWISE_MODE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_BACKEND_DESCRIPTOR",                                  {"HIPDNN_TYPE_BACKEND_DESCRIPTOR",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_GENSTATS_MODE",                                       {"HIPDNN_TYPE_GENSTATS_MODE",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_BN_FINALIZE_STATS_MODE",                              {"HIPDNN_TYPE_BN_FINALIZE_STATS_MODE",                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_TYPE_REDUCTION_OPERATOR_TYPE",                             {"HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendDescriptorType_t",                                   {"hipdnnBackendDescriptorType_t",                                   "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_POINTWISE_DESCRIPTOR",                             {"HIPDNN_BACKEND_POINTWISE_DESCRIPTOR",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR",                           {"HIPDNN_BACKEND_CONVOLUTION_DESCRIPTOR",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINE_DESCRIPTOR",                                {"HIPDNN_BACKEND_ENGINE_DESCRIPTOR",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINECFG_DESCRIPTOR",                             {"HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",                            {"HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",                        {"HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",                     {"HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",                           {"HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR",                             {"HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",                           {"HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",         {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR", {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR", "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",   {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",                        {"HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",                          {"HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_TENSOR_DESCRIPTOR",                                {"HIPDNN_BACKEND_TENSOR_DESCRIPTOR",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_MATMUL_DESCRIPTOR",                                {"HIPDNN_BACKEND_MATMUL_DESCRIPTOR",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",                      {"HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",      {"HIPDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_REDUCTION_DESCRIPTOR",                             {"HIPDNN_BACKEND_REDUCTION_DESCRIPTOR",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendNumericalNote_t",                                    {"hipdnnBackendNumericalNote_t",                                    "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_TENSOR_CORE",                               {"HIPDNN_NUMERICAL_NOTE_TENSOR_CORE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",                       {"HIPDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",               {"HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_FFT",                                       {"HIPDNN_NUMERICAL_NOTE_FFT",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC",                          {"HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD",                                  {"HIPDNN_NUMERICAL_NOTE_WINOGRAD",                                  "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_TYPE_COUNT",                                {"HIPDNN_NUMERICAL_NOTE_TYPE_COUNT",                                "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendLayoutType_t",                                       {"hipdnnBackendLayoutType_t",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NCHW",                               {"HIPDNN_LAYOUT_TYPE_PREFERRED_NCHW",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NHWC",                               {"HIPDNN_LAYOUT_TYPE_PREFERRED_NHWC",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",                             {"HIPDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",                             {"HIPDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_COUNT",                                        {"HIPDNN_LAYOUT_TYPE_COUNT",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendKnobType_t",                                         {"hipdnnBackendKnobType_t",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K",                                        {"HIPDNN_KNOB_TYPE_SPLIT_K",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SWIZZLE",                                        {"HIPDNN_KNOB_TYPE_SWIZZLE",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_SIZE",                                      {"HIPDNN_KNOB_TYPE_TILE_SIZE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_USE_TEX",                                        {"HIPDNN_KNOB_TYPE_USE_TEX",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_EDGE",                                           {"HIPDNN_KNOB_TYPE_EDGE",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_KBLOCK",                                         {"HIPDNN_KNOB_TYPE_KBLOCK",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGA",                                           {"HIPDNN_KNOB_TYPE_LDGA",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGB",                                           {"HIPDNN_KNOB_TYPE_LDGB",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_CHUNK_K",                                        {"HIPDNN_KNOB_TYPE_CHUNK_K",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_H",                                        {"HIPDNN_KNOB_TYPE_SPLIT_H",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_WINO_TILE",                                      {"HIPDNN_KNOB_TYPE_WINO_TILE",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_MULTIPLY",                                       {"HIPDNN_KNOB_TYPE_MULTIPLY",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K_BUF",                                    {"HIPDNN_KNOB_TYPE_SPLIT_K_BUF",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILEK",                                          {"HIPDNN_KNOB_TYPE_TILEK",                                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_STAGES",                                         {"HIPDNN_KNOB_TYPE_STAGES",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_REDUCTION_MODE",                                 {"HIPDNN_KNOB_TYPE_REDUCTION_MODE",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",                               {"HIPDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",                               "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K_SLC",                                    {"HIPDNN_KNOB_TYPE_SPLIT_K_SLC",                                    "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_IDX_MODE",                                       {"HIPDNN_KNOB_TYPE_IDX_MODE",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SLICED",                                         {"HIPDNN_KNOB_TYPE_SLICED",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_RS",                                       {"HIPDNN_KNOB_TYPE_SPLIT_RS",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SINGLEBUFFER",                                   {"HIPDNN_KNOB_TYPE_SINGLEBUFFER",                                   "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGC",                                           {"HIPDNN_KNOB_TYPE_LDGC",                                           "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPECFILT",                                       {"HIPDNN_KNOB_TYPE_SPECFILT",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_KERNEL_CFG",                                     {"HIPDNN_KNOB_TYPE_KERNEL_CFG",                                     "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_COUNTS",                                         {"HIPDNN_KNOB_TYPE_COUNTS",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendHeurMode_t",                                         {"hipdnnBackendHeurMode_t",                                         "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_INSTANT",                                        {"HIPDNN_HEUR_MODE_INSTANT",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_B",                                              {"HIPDNN_HEUR_MODE_B",                                              "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_HEUR_MODES_COUNT",                                         {"HIPDNN_HEUR_MODES_COUNT",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnNormMode_t",                                                {"hipdnnNormMode_t",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_PER_ACTIVATION",                                      {"HIPDNN_NORM_PER_ACTIVATION",                                      "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_PER_CHANNEL",                                         {"HIPDNN_NORM_PER_CHANNEL",                                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnNormAlgo_t",                                                {"hipdnnNormAlgo_t",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_ALGO_STANDARD",                                       {"HIPDNN_NORM_ALGO_STANDARD",                                       "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_ALGO_PERSIST",                                        {"HIPDNN_NORM_ALGO_PERSIST",                                        "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnNormOps_t",                                                 {"hipdnnNormOps_t",                                                 "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM",                                            {"HIPDNN_NORM_OPS_NORM",                                            "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM_ACTIVATION",                                 {"HIPDNN_NORM_OPS_NORM_ACTIVATION",                                 "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION",                             {"HIPDNN_NORM_OPS_NORM_ADD_ACTIVATION",                             "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBnFinalizeStatsMode_t",                                     {"hipdnnBnFinalizeStatsMode_t",                                     "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BN_FINALIZE_STATISTICS_TRAINING",                          {"HIPDNN_BN_FINALIZE_STATISTICS_TRAINING",                          "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         {"HIPDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         "", CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},

  // cuDNN types
  {"cudnnContext",                                                   {"hipdnnContext",                                                   "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnHandle_t",                                                  {"hipdnnHandle_t",                                                  "", CONV_TYPE, API_DNN, 1}},
  {"cudnnTensorStruct",                                              {"hipdnnTensorStruct",                                              "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnTensorDescriptor_t",                                        {"hipdnnTensorDescriptor_t",                                        "", CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionStruct",                                         {"hipdnnConvolutionStruct",                                         "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnConvolutionDescriptor_t",                                   {"hipdnnConvolutionDescriptor_t",                                   "", CONV_TYPE, API_DNN, 1}},
  {"cudnnPoolingStruct",                                             {"hipdnnPoolingStruct",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnPoolingDescriptor_t",                                       {"hipdnnPoolingDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1}},
  {"cudnnFilterStruct",                                              {"hipdnnFilterStruct",                                              "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFilterDescriptor_t",                                        {"hipdnnFilterDescriptor_t",                                        "", CONV_TYPE, API_DNN, 1}},
  {"cudnnLRNStruct",                                                 {"hipdnnLRNStruct",                                                 "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnLRNDescriptor_t",                                           {"hipdnnLRNDescriptor_t",                                           "", CONV_TYPE, API_DNN, 1}},
  {"cudnnActivationStruct",                                          {"hipdnnActivationStruct",                                          "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnActivationDescriptor_t",                                    {"hipdnnActivationDescriptor_t",                                    "", CONV_TYPE, API_DNN, 1}},
  {"cudnnSpatialTransformerStruct",                                  {"hipdnnSpatialTransformerStruct",                                  "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnSpatialTransformerDescriptor_t",                            {"hipdnnSpatialTransformerDescriptor_t",                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnOpTensorStruct",                                            {"hipdnnOpTensorStruct",                                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnOpTensorDescriptor_t",                                      {"hipdnnOpTensorDescriptor_t",                                      "", CONV_TYPE, API_DNN, 1}},
  {"cudnnReduceTensorStruct",                                        {"hipdnnReduceTensorStruct",                                        "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnReduceTensorDescriptor_t",                                  {"hipdnnReduceTensorDescriptor_t",                                  "", CONV_TYPE, API_DNN, 1}},
  {"cudnnCTCLossStruct",                                             {"hipdnnCTCLossStruct",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnCTCLossDescriptor_t",                                       {"hipdnnCTCLossDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnTensorTransformStruct",                                     {"hipdnnTensorTransformStruct",                                     "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnTensorTransformDescriptor_t",                               {"hipdnnTensorTransformDescriptor_t",                               "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnConvolutionFwdAlgoPerf_t",                                  {"hipdnnConvolutionFwdAlgoPerf_t",                                  "", CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionBwdFilterAlgoPerf_t",                            {"hipdnnConvolutionBwdFilterAlgoPerf_t",                            "", CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionBwdDataAlgoPerf_t",                              {"hipdnnConvolutionBwdDataAlgoPerf_t",                              "", CONV_TYPE, API_DNN, 1}},
  {"cudnnDropoutStruct",                                             {"hipdnnDropoutStruct",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnDropoutDescriptor_t",                                       {"hipdnnDropoutDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1}},
  {"cudnnAlgorithmStruct",                                           {"hipdnnAlgorithmStruct",                                           "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmDescriptor_t",                                     {"hipdnnAlgorithmDescriptor_t",                                     "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmPerformanceStruct",                                {"hipdnnAlgorithmPerformanceStruct",                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnAlgorithmPerformance_t",                                    {"hipdnnAlgorithmPerformance_t",                                    "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnRNNStruct",                                                 {"hipdnnRNNStruct",                                                 "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnRNNDescriptor_t",                                           {"hipdnnRNNDescriptor_t",                                           "", CONV_TYPE, API_DNN, 1}},
  {"cudnnPersistentRNNPlan",                                         {"hipdnnPersistentRNNPlan",                                         "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnPersistentRNNPlan_t",                                       {"hipdnnPersistentRNNPlan_t",                                       "", CONV_TYPE, API_DNN, 1}},
  {"cudnnAlgorithm_t",                                               {"hipdnnAlgorithm_t",                                               "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnDebug_t",                                                   {"hipdnnDebug_t",                                                   "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnCallback_t",                                                {"hipdnnCallback_t",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnRNNDataStruct",                                             {"hipdnnRNNDataStruct",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnRNNDataDescriptor_t",                                       {"hipdnnRNNDataDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnSeqDataStruct",                                             {"hipdnnSeqDataStruct",                                             "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnSeqDataDescriptor_t",                                       {"hipdnnSeqDataDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnAttnStruct",                                                {"hipdnnAttnStruct",                                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnAttnDescriptor_t",                                          {"hipdnnAttnDescriptor_t",                                          "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsConstParamStruct",                                  {"hipdnnFusedOpsConstParamStruct",                                  "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsConstParamPack_t",                                  {"hipdnnFusedOpsConstParamPack_t",                                  "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamStruct",                                {"hipdnnFusedOpsVariantParamStruct",                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamPack_t",                                {"hipdnnFusedOpsVariantParamPack_t",                                "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsPlanStruct",                                        {"hipdnnFusedOpsPlanStruct",                                        "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnFusedOpsPlan_t",                                            {"hipdnnFusedOpsPlan_t",                                            "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnBackendDescriptor_t",                                       {"hipdnnBackendDescriptor_t",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"libraryPropertyType",                                            {"hipdnnLibraryPropertyType",                                       "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"libraryPropertyType_t",                                          {"hipdnnLibraryPropertyType_t",                                     "", CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_TYPE_NAME_VER_MAP {
  {"CUDNN_MAJOR",                                                    {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_MINOR",                                                    {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_PATCHLEVEL",                                               {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnForwardMode_t",                                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_FWD_MODE_INFERENCE",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_FWD_MODE_TRAINING",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnRNNMode_t",                                                 {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_RELU",                                                 {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_TANH",                                                 {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LSTM",                                                     {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_GRU",                                                      {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNBiasMode_t",                                             {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_NO_BIAS",                                              {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_SINGLE_INP_BIAS",                                      {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_DOUBLE_BIAS",                                          {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_SINGLE_REC_BIAS",                                      {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnDirectionMode_t",                                           {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_UNIDIRECTIONAL",                                           {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_BIDIRECTIONAL",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNInputMode_t",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LINEAR_INPUT",                                             {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SKIP_INPUT",                                               {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNClipMode_t",                                             {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_CLIP_NONE",                                            {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_CLIP_MINMAX",                                          {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"cudnnRNNDataLayout_t",                                           {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",                       {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",                         {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",                     {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"cudnnRNNPaddingMode_t",                                          {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_PADDED_IO_DISABLED",                                   {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_PADDED_IO_ENABLED",                                    {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"cudnnRNNStruct",                                                 {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNDescriptor_t",                                           {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnPersistentRNNPlan",                                         {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnPersistentRNNPlan_t",                                       {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNDataStruct",                                             {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"cudnnRNNDataDescriptor_t",                                       {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_TRAIN_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_TRAIN_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_TRAIN_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_INFER_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_INFER_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ADV_INFER_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnWgradMode_t",                                               {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_WGRAD_MODE_ADD",                                           {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_WGRAD_MODE_SET",                                           {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnBackendDescriptor_t",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnPointwiseMode_t",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ADD",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_MUL",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_MIN",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_MAX",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SQRT",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_RELU_FWD",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_TANH_FWD",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SIGMOID_FWD",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ELU_FWD",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_GELU_FWD",                                       {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SOFTPLUS_FWD",                                   {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SWISH_FWD",                                      {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_RELU_BWD",                                       {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_TANH_BWD",                                       {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SIGMOID_BWD",                                    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ELU_BWD",                                        {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_GELU_BWD",                                       {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SOFTPLUS_BWD",                                   {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SWISH_BWD",                                      {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnGenStatsMode_t",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_GENSTATS_SUM_SQSUM",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnBackendAttributeName_t",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_MODE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_MATH_PREC",                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_NAN_PROPAGATION",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_ELU_ALPHA",                                 {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA",                             {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_SWISH_BETA",                                {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_COMP_TYPE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_CONV_MODE",                               {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_DILATIONS",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_POST_PADDINGS",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS",                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINEHEUR_MODE",                                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINEHEUR_RESULTS",                                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINECFG_ENGINE",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINECFG_KNOB_CHOICES",                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_HANDLE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",           {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",           {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",                         {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_INTERMEDIATE_INFO_SIZE",                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",               {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",              {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_XDESC",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_BDESC",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_YDESC",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_DXDESC",                          {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_DYDESC",                          {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_GENSTATS_MODE",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_GENSTATS_XDESC",                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",                    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",                    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",                 {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",                    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",        {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",         {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",      {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",               {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",            {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",                 {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",                  {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",              {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",                  {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",       {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATIONGRAPH_HANDLE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATIONGRAPH_OPS",                                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_DATA_TYPE",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_DIMENSIONS",                                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_STRIDES",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_VECTOR_COUNT",                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_UNIQUE_ID",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_IS_VIRTUAL",                                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_IS_BY_VALUE",                                  {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_VARIANT_PACK_WORKSPACE",                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID",                              {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_LAYOUT_INFO_TYPES",                                   {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_INFO_TYPE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_KNOB_INFO_STRIDE",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_OPERATION_GRAPH",                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_GLOBAL_INDEX",                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_KNOB_INFO",                                    {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_NUMERICAL_NOTE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_LAYOUT_INFO",                                  {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_MATMUL_COMP_TYPE",                                    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_ADESC",                              {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_BDESC",                              {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_CDESC",                              {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_DESC",                               {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",    {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_REDUCTION_OPERATOR",                                  {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_REDUCTION_COMP_TYPE",                                 {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_REDUCTION_XDESC",                           {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_REDUCTION_YDESC",                           {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_REDUCTION_DESC",                            {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnBackendAttributeType_t",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_HANDLE",                                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_DATA_TYPE",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_BOOLEAN",                                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_INT64",                                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_FLOAT",                                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_DOUBLE",                                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_VOID_PTR",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_CONVOLUTION_MODE",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_HEUR_MODE",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_KNOB_TYPE",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_NAN_PROPOGATION",                                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_NUMERICAL_NOTE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_LAYOUT_TYPE",                                         {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_ATTRIB_NAME",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_POINTWISE_MODE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_BACKEND_DESCRIPTOR",                                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_GENSTATS_MODE",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_BN_FINALIZE_STATS_MODE",                              {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_REDUCTION_OPERATOR_TYPE",                             {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnBackendDescriptorType_t",                                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_POINTWISE_DESCRIPTOR",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_ENGINE_DESCRIPTOR",                                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_ENGINECFG_DESCRIPTOR",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",                     {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR", {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_TENSOR_DESCRIPTOR",                                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_MATMUL_DESCRIPTOR",                                {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",                      {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",      {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_REDUCTION_DESCRIPTOR",                             {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",                   {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnBackendNumericalNote_t",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_TENSOR_CORE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_FFT",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC",                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD",                                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_TYPE_COUNT",                                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnBackendLayoutType_t",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NCHW",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NHWC",                               {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",                             {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",                             {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYOUT_TYPE_COUNT",                                        {CUDNN_802, CUDA_0,   CUDA_0  }},
  {"cudnnBackendKnobType_t",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_K",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SWIZZLE",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_SIZE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_USE_TEX",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_EDGE",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_KBLOCK",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_LDGA",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_LDGB",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_CHUNK_K",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_H",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_WINO_TILE",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_MULTIPLY",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_K_BUF",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILEK",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_STAGES",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_REDUCTION_MODE",                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",                               {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_K_SLC",                                    {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_IDX_MODE",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SLICED",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_RS",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SINGLEBUFFER",                                   {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_LDGC",                                           {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPECFILT",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_KERNEL_CFG",                                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_COUNTS",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnBackendHeurMode_t",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_HEUR_MODE_INSTANT",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_HEUR_MODE_B",                                              {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_HEUR_MODES_COUNT",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionFwdAlgoPerf_t",                                  {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnReorderType_t",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_DEFAULT_REORDER",                                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_NO_REORDER",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionMode_t",                                         {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CROSS_CORRELATION",                                        {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionStruct",                                         {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionDescriptor_t",                                   {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_INFER_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_INFER_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_INFER_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdFilterAlgoPerf_t",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_TRAIN_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_TRAIN_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_CNN_TRAIN_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnContext",                                                   {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnHandle_t",                                                  {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_INFER_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_INFER_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_INFER_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnStatus_t",                                                  {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_SUCCESS",                                           {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_NOT_INITIALIZED",                                   {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_ALLOC_FAILED",                                      {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_BAD_PARAM",                                         {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_INTERNAL_ERROR",                                    {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_INVALID_VALUE",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_ARCH_MISMATCH",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_MAPPING_ERROR",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_EXECUTION_FAILED",                                  {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_NOT_SUPPORTED",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_LICENSE_ERROR",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",                      {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_RUNTIME_IN_PROGRESS",                               {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_RUNTIME_FP_OVERFLOW",                               {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_STATUS_VERSION_MISMATCH",                                  {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnRuntimeTag_t",                                              {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnErrQueryMode_t",                                            {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_ERRQUERY_RAWCODE",                                         {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_ERRQUERY_NONBLOCKING",                                     {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_ERRQUERY_BLOCKING",                                        {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_TRAIN_MAJOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_TRAIN_MINOR",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_OPS_TRAIN_PATCH",                                          {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"libraryPropertyType",                                            {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"libraryPropertyType_t",                                          {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnTensorDescriptor_t",                                        {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"cudnnTensorStruct",                                              {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"cudnnPoolingDescriptor_t",                                       {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnPoolingStruct",                                             {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnFilterDescriptor_t",                                        {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnFilterStruct",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnLRNDescriptor_t",                                           {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnLRNStruct",                                                 {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnActivationDescriptor_t",                                    {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnActivationStruct",                                          {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnFilterStruct",                                              {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnSpatialTransformerDescriptor_t",                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnSpatialTransformerStruct",                                  {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnOpTensorDescriptor_t",                                      {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnOpTensorStruct",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnReduceTensorDescriptor_t",                                  {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnReduceTensorStruct",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnCTCLossDescriptor_t",                                       {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnReduceTensorStruct",                                        {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnTensorTransformDescriptor_t",                               {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnTensorTransformStruct",                                     {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnDataType_t",                                                {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_FLOAT",                                               {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_DOUBLE",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_HALF",                                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_INT8",                                                {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_INT32",                                               {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_INT8x4",                                              {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_UINT8",                                               {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_UINT8x4",                                             {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_INT8x32",                                             {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_BFLOAT16",                                            {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_INT64",                                               {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnMathType_t",                                                {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_DEFAULT_MATH",                                             {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_OP_MATH",                                           {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",                          {CUDNN_721, CUDA_0,   CUDA_0  }},
  {"CUDNN_FMA_MATH",                                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnNanPropagation_t",                                          {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_NOT_PROPAGATE_NAN",                                        {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_PROPAGATE_NAN",                                            {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnDeterminism_t",                                             {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_NON_DETERMINISTIC",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DETERMINISTIC",                                            {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DIM_MAX",                                                  {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnTensorFormat_t",                                            {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_NCHW",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_NHWC",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_NCHW_VECT_C",                                       {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnFoldingDirection_t",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_TRANSFORM_FOLD",                                           {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_TRANSFORM_UNFOLD",                                         {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnOpTensorOp_t",                                              {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_ADD",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_MUL",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_MIN",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_MAX",                                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_SQRT",                                           {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_OP_TENSOR_NOT",                                            {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnReduceTensorOp_t",                                          {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_ADD",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_MUL",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_MIN",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_MAX",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_AMAX",                                       {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_AVG",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_NORM1",                                      {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_NORM2",                                      {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                               {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnReduceTensorIndices_t",                                     {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_NO_INDICES",                                 {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_REDUCE_TENSOR_FLATTENED_INDICES",                          {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnIndicesType_t",                                             {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_32BIT_INDICES",                                            {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_64BIT_INDICES",                                            {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_16BIT_INDICES",                                            {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_8BIT_INDICES",                                             {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnSoftmaxAlgorithm_t",                                        {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SOFTMAX_FAST",                                             {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SOFTMAX_ACCURATE",                                         {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SOFTMAX_LOG",                                              {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnSoftmaxMode_t",                                             {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SOFTMAX_MODE_INSTANCE",                                    {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SOFTMAX_MODE_CHANNEL",                                     {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnPoolingMode_t",                                             {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_POOLING_MAX",                                              {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",                    {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",                    {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_POOLING_MAX_DETERMINISTIC",                                {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnActivationMode_t",                                          {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_SIGMOID",                                       {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_RELU",                                          {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_TANH",                                          {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_CLIPPED_RELU",                                  {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_ELU",                                           {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_ACTIVATION_IDENTITY",                                      {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_LRN_MIN_N",                                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LRN_MAX_N",                                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LRN_MIN_K",                                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LRN_MIN_BETA",                                             {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnLRNMode_t",                                                 {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_LRN_CROSS_CHANNEL_DIM1",                                   {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnDivNormMode_t",                                             {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_DIVNORM_PRECOMPUTED_MEANS",                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnBatchNormMode_t",                                           {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_PER_ACTIVATION",                                 {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_SPATIAL",                                        {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",                             {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_BN_MIN_EPSILON",                                           {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"cudnnBatchNormOps_t",                                            {CUDNN_741, CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_OPS_BN",                                         {CUDNN_741, CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",                              {CUDNN_741, CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",                          {CUDNN_741, CUDA_0,   CUDA_0  }},
  {"cudnnNormMode_t",                                                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_PER_ACTIVATION",                                      {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_PER_CHANNEL",                                         {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnNormAlgo_t",                                                {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_ALGO_STANDARD",                                       {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_ALGO_PERSIST",                                        {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnNormOps_t",                                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_OPS_NORM",                                            {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_OPS_NORM_ACTIVATION",                                 {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION",                             {CUDNN_801, CUDA_0,   CUDA_0  }},
  {"cudnnSamplerType_t",                                             {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_SAMPLER_BILINEAR",                                         {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnDropoutDescriptor_t",                                       {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnDropoutStruct",                                             {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithmDescriptor_t",                                     {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithmStruct",                                           {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithmPerformance_t",                                    {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithmPerformanceStruct",                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionFwdAlgo_t",                                      {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",                       {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",               {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",                                {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",                              {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT",                                 {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",                          {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                            {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",                   {CUDNN_51,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_FWD_ALGO_COUNT",                               {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdFilterAlgo_t",                                {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",                          {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",                     {CUDNN_51,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",            {CUDNN_51,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",                   {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdDataAlgo_t",                                  {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",                              {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",                              {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",                     {CUDNN_40,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",                       {CUDNN_50,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",              {CUDNN_51,  CUDA_0,   CUDA_0  }},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",                          {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"cudnnRNNAlgo_t",                                                 {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_ALGO_STANDARD",                                        {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_ALGO_PERSIST_STATIC",                                  {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_ALGO_PERSIST_DYNAMIC",                                 {CUDNN_60,  CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",                          {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNN_ALGO_COUNT",                                           {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnCTCLossAlgo_t",                                             {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",                              {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",                          {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithm_t",                                               {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnSeverity_t",                                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_FATAL",                                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_ERROR",                                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_WARNING",                                              {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_INFO",                                                 {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_ERROR_EN",                                             {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_WARNING_EN",                                           {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEV_INFO_EN",                                              {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnDebug_t",                                                   {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionFwdPreference_t",                                {CUDNN_20,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_FWD_NO_WORKSPACE",                             {CUDNN_20,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_FWD_PREFER_FASTEST",                           {CUDNN_20,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",                  {CUDNN_20,  CUDNN_765,CUDNN_801}},
  {"cudnnConvolutionBwdDataAlgoPerf_t",                              {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsConstParamStruct",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsConstParamPack_t",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsVariantParamStruct",                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsVariantParamPack_t",                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsPlanStruct",                                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsPlan_t",                                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOps_t",                                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",                    {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",                   {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",                     {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsConstParamLabel_t",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_XDESC",                                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_XDATA_PLACEHOLDER",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_MODE",                                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_EQSCALEBIAS_DESC",                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_ACTIVATION_DESC",                                    {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_CONV_DESC",                                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_WDESC",                                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_WDATA_PLACEHOLDER",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DWDESC",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DWDATA_PLACEHOLDER",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_YDESC",                                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_YDATA_PLACEHOLDER",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DYDESC",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DYDATA_PLACEHOLDER",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_YSTATS_DESC",                                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_YSUM_PLACEHOLDER",                                   {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_YSQSUM_PLACEHOLDER",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_SCALE_PLACEHOLDER",                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_BIAS_PLACEHOLDER",                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",                         {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_ZDESC",                                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_ZDATA_PLACEHOLDER",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                           {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_ACTIVATION_BITMASK_DESC",                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",                     {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DXDESC",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DXDATA_PLACEHOLDER",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DZDESC",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_DZDATA_PLACEHOLDER",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_DSCALE_PLACEHOLDER",                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PARAM_BN_DBIAS_PLACEHOLDER",                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsPointerPlaceHolder_t",                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_NULL",                                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_ELEM_ALIGNED",                                         {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_16B_ALIGNED",                                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnFusedOpsVariantParamLabel_t",                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_XDATA",                                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_EQSCALE",                                           {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_EQBIAS",                                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_WDATA",                                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_DWDATA",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_YDATA",                                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_DYDATA",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_YSUM",                                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_YSQSUM",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_WORKSPACE",                                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_SCALE",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_BIAS",                                              {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_SAVED_MEAN",                                        {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_SAVED_INVSTD",                                      {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_RUNNING_MEAN",                                      {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_RUNNING_VAR",                                       {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_ZDATA",                                                {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_Z_EQSCALE",                                         {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_Z_EQBIAS",                                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_ACTIVATION_BITMASK",                                   {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_DXDATA",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_DZDATA",                                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_DSCALE",                                            {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_PTR_BN_DBIAS",                                             {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",                    {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",                     {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",                          {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_SCALAR_DOUBLE_BN_EPSILON",                                 {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"cudnnSeqDataAxis_t",                                             {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEQDATA_TIME_DIM",                                         {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEQDATA_BATCH_DIM",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEQDATA_BEAM_DIM",                                         {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEQDATA_VECT_DIM",                                         {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_SEQDATA_DIM_COUNT",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdFilterPreference_t",                          {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",                      {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",                    {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",           {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"cudnnSeqDataStruct",                                             {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnSeqDataDescriptor_t",                                       {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnAttnQueryMap_t",                                            {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTN_QUERYMAP_ALL_TO_ONE",                                 {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTN_QUERYMAP_ONE_TO_ONE",                                 {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTN_DISABLE_PROJ_BIASES",                                 {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTN_ENABLE_PROJ_BIASES",                                  {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"cudnnAttnStruct",                                                {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnAttnDescriptor_t",                                          {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"cudnnMultiHeadAttnWeightKind_t",                                 {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_Q_WEIGHTS",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_K_WEIGHTS",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_V_WEIGHTS",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_O_WEIGHTS",                                        {CUDNN_750, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_Q_BIASES",                                         {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_K_BIASES",                                         {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_V_BIASES",                                         {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"CUDNN_MH_ATTN_O_BIASES",                                         {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTN_WKIND_COUNT",                                         {CUDNN_763, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdDataPreference_t",                            {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",                        {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",                      {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",             {CUDNN_30,  CUDNN_765,CUDNN_801}},
  {"cudnnLossNormalizationMode_t",                                   {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_LOSS_NORMALIZATION_NONE",                                  {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_LOSS_NORMALIZATION_SOFTMAX",                               {CUDNN_760, CUDA_0,   CUDA_0  }},
  {"CUDNN_VERSION",                                                  {CUDNN_20,  CUDA_0,   CUDA_0  }},
  {"cudnnCTCLossStruct",                                             {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnCallback_t",                                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnBnFinalizeStatsMode_t",                                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BN_FINALIZE_STATISTICS_TRAINING",                          {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         {CUDNN_810, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_DNN_TYPE_NAME_VER_MAP {
};
