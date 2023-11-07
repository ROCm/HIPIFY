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
  {"CUDNN_DIM_MAX",                                                  {"HIPDNN_DIM_MAX",                                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    //  8
  {"CUDNN_LRN_MIN_N",                                                {"HIPDNN_LRN_MIN_N",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    //  1
  {"CUDNN_LRN_MAX_N",                                                {"HIPDNN_LRN_MAX_N",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 16
  {"CUDNN_LRN_MIN_K",                                                {"HIPDNN_LRN_MIN_K",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1e-5
  {"CUDNN_LRN_MIN_BETA",                                             {"HIPDNN_LRN_MIN_BETA",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0.01
  {"CUDNN_BN_MIN_EPSILON",                                           {"HIPDNN_BN_MIN_EPSILON",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1e-5
  {"CUDNN_SEV_ERROR_EN",                                             {"HIPDNN_SEV_ERROR_EN",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SEV_WARNING_EN",                                           {"HIPDNN_SEV_WARNING_EN",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SEV_INFO_EN",                                              {"HIPDNN_SEV_INFO_EN",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SEQDATA_DIM_COUNT",                                        {"HIPDNN_SEQDATA_DIM_COUNT",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4

  // cuDNN enums
  {"cudnnStatus_t",                                                  {"hipdnnStatus_t",                                                  "miopenStatus_t",                                                  CONV_TYPE, API_DNN, 1}},
  {"CUDNN_STATUS_SUCCESS",                                           {"HIPDNN_STATUS_SUCCESS",                                           "miopenStatusSuccess",                                             CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  0
  {"CUDNN_STATUS_NOT_INITIALIZED",                                   {"HIPDNN_STATUS_NOT_INITIALIZED",                                   "miopenStatusNotInitialized",                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  1
  {"CUDNN_STATUS_ALLOC_FAILED",                                      {"HIPDNN_STATUS_ALLOC_FAILED",                                      "miopenStatusAllocFailed",                                         CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  2    //  4
  {"CUDNN_STATUS_BAD_PARAM",                                         {"HIPDNN_STATUS_BAD_PARAM",                                         "miopenStatusBadParm",                                             CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  3
  {"CUDNN_STATUS_INTERNAL_ERROR",                                    {"HIPDNN_STATUS_INTERNAL_ERROR",                                    "miopenStatusInternalError",                                       CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  4    //  5
  {"CUDNN_STATUS_INVALID_VALUE",                                     {"HIPDNN_STATUS_INVALID_VALUE",                                     "miopenStatusInvalidValue",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  5    //  2
  {"CUDNN_STATUS_ARCH_MISMATCH",                                     {"HIPDNN_STATUS_ARCH_MISMATCH",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    //  6
  {"CUDNN_STATUS_MAPPING_ERROR",                                     {"HIPDNN_STATUS_MAPPING_ERROR",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    //  7
  {"CUDNN_STATUS_EXECUTION_FAILED",                                  {"HIPDNN_STATUS_EXECUTION_FAILED",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    //  8
  {"CUDNN_STATUS_NOT_SUPPORTED",                                     {"HIPDNN_STATUS_NOT_SUPPORTED",                                     "miopenStatusUnsupportedOp",                                       CONV_NUMERIC_LITERAL, API_DNN, 1}},    //  9    //  8
  {"CUDNN_STATUS_LICENSE_ERROR",                                     {"HIPDNN_STATUS_LICENSE_ERROR",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 10
  {"CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",                      {"HIPDNN_STATUS_RUNTIME_PREREQUISITE_MISSING",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 11
  {"CUDNN_STATUS_RUNTIME_IN_PROGRESS",                               {"HIPDNN_STATUS_RUNTIME_IN_PROGRESS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 12
  {"CUDNN_STATUS_RUNTIME_FP_OVERFLOW",                               {"HIPDNN_STATUS_RUNTIME_FP_OVERFLOW",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 13
  {"CUDNN_STATUS_VERSION_MISMATCH",                                  {"HIPDNN_STATUS_VERSION_MISMATCH",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 14
  {"cudnnRuntimeTag_t",                                              {"hipdnnRuntimeTag_t",                                              "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnConvolutionMode_t",                                         {"hipdnnConvolutionMode_t",                                         "miopenConvolutionMode_t",                                         CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION",                                              {"HIPDNN_CONVOLUTION",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 0
  {"CUDNN_CROSS_CORRELATION",                                        {"HIPDNN_CROSS_CORRELATION",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"cudnnTensorFormat_t",                                            {"hipdnnTensorFormat_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"CUDNN_TENSOR_NCHW",                                              {"HIPDNN_TENSOR_NCHW",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 0
  {"CUDNN_TENSOR_NHWC",                                              {"HIPDNN_TENSOR_NHWC",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_TENSOR_NCHW_VECT_C",                                       {"HIPDNN_TENSOR_NCHW_VECT_C",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 2
  {"cudnnFoldingDirection_t",                                        {"hipdnnFoldingDirection_t",                                        "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TRANSFORM_FOLD",                                           {"HIPDNN_TRANSFORM_FOLD",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0U
  {"CUDNN_TRANSFORM_UNFOLD",                                         {"HIPDNN_TRANSFORM_UNFOLD",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1U
  {"cudnnDataType_t",                                                {"hipdnnDataType_t",                                                "miopenDataType_t",                                                CONV_TYPE, API_DNN, 1}},
  {"CUDNN_DATA_FLOAT",                                               {"HIPDNN_DATA_FLOAT",                                               "miopenFloat",                                                     CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_DATA_DOUBLE",                                              {"HIPDNN_DATA_DOUBLE",                                              "miopenDouble",                                                    CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_DATA_HALF",                                                {"HIPDNN_DATA_HALF",                                                "miopenHalf",                                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_DATA_INT8",                                                {"HIPDNN_DATA_INT8",                                                "miopenInt8",                                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_DATA_INT32",                                               {"HIPDNN_DATA_INT32",                                               "miopenInt32",                                                     CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_DATA_INT8x4",                                              {"HIPDNN_DATA_INT8x4",                                              "miopenInt8x4",                                                    CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_DATA_UINT8",                                               {"HIPDNN_DATA_UINT8",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 6
  {"CUDNN_DATA_UINT8x4",                                             {"HIPDNN_DATA_UINT8x4",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 7
  {"CUDNN_DATA_INT8x32",                                             {"HIPDNN_DATA_INT8x32",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 8
  {"CUDNN_DATA_BFLOAT16",                                            {"HIPDNN_DATA_BFLOAT16",                                            "miopenBFloat16",                                                  CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},  // 9
  {"CUDNN_DATA_INT64",                                               {"HIPDNN_DATA_INT64",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 10
  {"CUDNN_DATA_BOOLEAN",                                             {"HIPDNN_DATA_BOOLEAN",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 11
  {"CUDNN_DATA_FP8_E4M3",                                            {"HIPDNN_DATA_FP8_E4M3",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 12
  {"CUDNN_DATA_FP8_E5M2",                                            {"HIPDNN_DATA_FP8_E5M2",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 13
  {"CUDNN_DATA_FAST_FLOAT_FOR_FP8",                                  {"HIPDNN_DATA_FAST_FLOAT_FOR_FP8",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 14
  {"cudnnErrQueryMode_t",                                            {"hipdnnErrQueryMode_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_ERRQUERY_RAWCODE",                                         {"HIPDNN_ERRQUERY_RAWCODE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_ERRQUERY_NONBLOCKING",                                     {"HIPDNN_ERRQUERY_NONBLOCKING",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_ERRQUERY_BLOCKING",                                        {"HIPDNN_ERRQUERY_BLOCKING",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"cudnnSeverity_t",                                                {"hipdnnSeverity_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SEV_FATAL",                                                {"HIPDNN_SEV_FATAL",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_SEV_ERROR",                                                {"HIPDNN_SEV_ERROR",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_SEV_WARNING",                                              {"HIPDNN_SEV_WARNING",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_SEV_INFO",                                                 {"HIPDNN_SEV_INFO",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"cudnnConvolutionFwdAlgo_t",                                      {"hipdnnConvolutionFwdAlgo_t",                                      "miopenConvFwdAlgorithm_t",                                        CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",                       {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",                       "miopenConvolutionFwdAlgoImplicitGEMM",                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",               {"HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM",                                {"HIPDNN_CONVOLUTION_FWD_ALGO_GEMM",                                "miopenConvolutionFwdAlgoGEMM",                                    CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",                              {"HIPDNN_CONVOLUTION_FWD_ALGO_DIRECT",                              "miopenConvolutionFwdAlgoDirect",                                  CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT",                                 {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT",                                 "miopenConvolutionFwdAlgoFFT",                                     CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",                          {"HIPDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 5
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                            {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",                            "miopenConvolutionFwdAlgoWinograd",                                CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",                   {"HIPDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 7
  {"CUDNN_CONVOLUTION_FWD_ALGO_COUNT",                               {"HIPDNN_CONVOLUTION_FWD_ALGO_COUNT",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 8
  {"cudnnConvolutionFwdPreference_t",                                {"hipdnnConvolutionFwdPreference_t",                                "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_FWD_NO_WORKSPACE",                             {"HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_FWD_PREFER_FASTEST",                           {"HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",                  {"HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnDeterminism_t",                                             {"hipdnnDeterminism_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NON_DETERMINISTIC",                                        {"HIPDNN_NON_DETERMINISTIC",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_DETERMINISTIC",                                            {"HIPDNN_DETERMINISTIC",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnDivNormMode_t",                                             {"hipdnnDivNormMode_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_DIVNORM_PRECOMPUTED_MEANS",                                {"HIPDNN_DIVNORM_PRECOMPUTED_MEANS",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"cudnnCTCLossAlgo_t",                                             {"hipdnnCTCLossAlgo_t",                                             "miopenCTCLossAlgo_t",                                             CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",                              {"HIPDNN_CTC_LOSS_ALGO_DETERMINISTIC",                              "MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC",                              CONV_NUMERIC_LITERAL, API_DNN, 1, HIP_UNSUPPORTED}},    // 0
  {"CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",                          {"HIPDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnLRNMode_t",                                                 {"hipdnnLRNMode_t",                                                 "miopenLRNMode_t",                                                 CONV_TYPE, API_DNN, 1}},
  {"CUDNN_LRN_CROSS_CHANNEL_DIM1",                                   {"HIPDNN_LRN_CROSS_CHANNEL",                                        "miopenLRNCrossChannel",                                           CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0 vs 1
  {"cudnnRNNInputMode_t",                                            {"hipdnnRNNInputMode_t",                                            "miopenRNNInputMode_t",                                            CONV_TYPE, API_DNN, 1}},
  {"CUDNN_LINEAR_INPUT",                                             {"HIPDNN_LINEAR_INPUT",                                             "miopenRNNlinear",                                                 CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SKIP_INPUT",                                               {"HIPDNN_SKIP_INPUT",                                               "miopenRNNskip",                                                   CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnDirectionMode_t",                                           {"hipdnnDirectionMode_t",                                           "miopenRNNDirectionMode_t",                                        CONV_TYPE, API_DNN, 1}},
  {"CUDNN_UNIDIRECTIONAL",                                           {"HIPDNN_UNIDIRECTIONAL",                                           "miopenRNNunidirection",                                           CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_BIDIRECTIONAL",                                            {"HIPDNN_BIDIRECTIONAL",                                            "miopenRNNbidirection",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnMathType_t",                                                {"hipdnnMathType_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"CUDNN_DEFAULT_MATH",                                             {"HIPDNN_DEFAULT_MATH",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 0
  {"CUDNN_TENSOR_OP_MATH",                                           {"HIPDNN_TENSOR_OP_MATH",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",                          {"HIPDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_FMA_MATH",                                                 {"HIPDNN_FMA_MATH",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"cudnnNanPropagation_t",                                          {"hipdnnNanPropagation_t",                                          "miopenNanPropagation_t",                                          CONV_TYPE, API_DNN, 1}},
  {"CUDNN_NOT_PROPAGATE_NAN",                                        {"HIPDNN_NOT_PROPAGATE_NAN",                                        "MIOPEN_NOT_PROPAGATE_NAN",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_PROPAGATE_NAN",                                            {"HIPDNN_PROPAGATE_NAN",                                            "MIOPEN_PROPAGATE_NAN",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnConvolutionBwdDataAlgo_t",                                  {"hipdnnConvolutionBwdDataAlgo_t",                                  "miopenConvBwdDataAlgorithm_t",                                    CONV_TYPE, API_DNN, 1}},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0",                              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0",                              "miopenConvolutionBwdDataAlgoGEMM",                                CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",                              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1",                              "miopenConvolutionBwdDataAlgoDirect",                              CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                            {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",                            "miopenConvolutionBwdDataAlgoFFT",                                 CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",                     {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 3
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",                       {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",                       "miopenConvolutionBwdDataAlgoWinograd",                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",              {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 5
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT",                          {"HIPDNN_CONVOLUTION_BWD_DATA_ALGO_TRANSPOSE_GEMM",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 6
  {"cudnnConvolutionBwdFilterAlgo_t",                                {"hipdnnConvolutionBwdFilterAlgo_t",                                "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",                          {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 2
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_3",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 3
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",                     {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 4
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",            {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED",            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 5
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",                   {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 6
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",                        {"HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 7
  {"cudnnConvolutionBwdFilterPreference_t",                          {"hipdnnConvolutionBwdFilterPreference_t",                          "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",                      {"HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",                    {"HIPDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",           {"HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT",           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnRNNAlgo_t",                                                 {"hipdnnRNNAlgo_t",                                                 "miopenRNNAlgo_t",                                                 CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_ALGO_STANDARD",                                        {"HIPDNN_RNN_ALGO_STANDARD",                                        "miopenRNNdefault",                                                CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_ALGO_PERSIST_STATIC",                                  {"HIPDNN_RNN_ALGO_PERSIST_STATIC",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_RNN_ALGO_PERSIST_DYNAMIC",                                 {"HIPDNN_RNN_ALGO_PERSIST_DYNAMIC",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 2
  {"CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",                          {"HIPDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 3
  {"CUDNN_RNN_ALGO_COUNT",                                           {"HIPDNN_RNN_ALGO_COUNT",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"cudnnRNNMode_t",                                                 {"hipdnnRNNMode_t",                                                 "miopenRNNMode_t",                                                 CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_RELU",                                                 {"HIPDNN_RNN_RELU",                                                 "miopenRNNRELU",                                                   CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_TANH",                                                 {"HIPDNN_RNN_TANH",                                                 "miopenRNNTANH",                                                   CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_LSTM",                                                     {"HIPDNN_LSTM",                                                     "miopenLSTM",                                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_GRU",                                                      {"HIPDNN_GRU",                                                      "miopenGRU",                                                       CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"cudnnRNNBiasMode_t",                                             {"hipdnnRNNBiasMode_t",                                             "miopenRNNBiasMode_t",                                             CONV_TYPE, API_DNN, 1}},
  {"CUDNN_RNN_NO_BIAS",                                              {"HIPDNN_RNN_NO_BIAS",                                              "miopenRNNNoBias",                                                 CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_RNN_SINGLE_INP_BIAS",                                      {"HIPDNN_RNN_WITH_BIAS",                                            "miopenRNNwithBias",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_RNN_DOUBLE_BIAS",                                          {"HIPDNN_RNN_WITH_BIAS",                                            "miopenRNNwithBias",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_RNN_SINGLE_REC_BIAS",                                      {"HIPDNN_RNN_WITH_BIAS",                                            "miopenRNNwithBias",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnOpTensorOp_t",                                              {"hipdnnOpTensorOp_t",                                              "miopenTensorOp_t",                                                CONV_TYPE, API_DNN, 1}},
  {"CUDNN_OP_TENSOR_ADD",                                            {"HIPDNN_OP_TENSOR_ADD",                                            "miopenTensorOpAdd",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_OP_TENSOR_MUL",                                            {"HIPDNN_OP_TENSOR_MUL",                                            "miopenTensorOpMul",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_OP_TENSOR_MIN",                                            {"HIPDNN_OP_TENSOR_MIN",                                            "miopenTensorOpMin",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_OP_TENSOR_MAX",                                            {"HIPDNN_OP_TENSOR_MAX",                                            "miopenTensorOpMax",                                               CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_OP_TENSOR_SQRT",                                           {"HIPDNN_OP_TENSOR_SQRT",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 4
  {"CUDNN_OP_TENSOR_NOT",                                            {"HIPDNN_OP_TENSOR_NOT",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"cudnnReduceTensorOp_t",                                          {"hipdnnReduceTensorOp_t",                                          "miopenReduceTensorOp_t",                                          CONV_TYPE, API_DNN, 1}},
  {"CUDNN_REDUCE_TENSOR_ADD",                                        {"HIPDNN_REDUCE_TENSOR_ADD",                                        "MIOPEN_REDUCE_TENSOR_ADD",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_REDUCE_TENSOR_MUL",                                        {"HIPDNN_REDUCE_TENSOR_MUL",                                        "MIOPEN_REDUCE_TENSOR_MUL",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_REDUCE_TENSOR_MIN",                                        {"HIPDNN_REDUCE_TENSOR_MIN",                                        "MIOPEN_REDUCE_TENSOR_MIN",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_REDUCE_TENSOR_MAX",                                        {"HIPDNN_REDUCE_TENSOR_MAX",                                        "MIOPEN_REDUCE_TENSOR_MAX",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_REDUCE_TENSOR_AMAX",                                       {"HIPDNN_REDUCE_TENSOR_AMAX",                                       "MIOPEN_REDUCE_TENSOR_AMAX",                                       CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_REDUCE_TENSOR_AVG",                                        {"HIPDNN_REDUCE_TENSOR_AVG",                                        "MIOPEN_REDUCE_TENSOR_AVG",                                        CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_REDUCE_TENSOR_NORM1",                                      {"HIPDNN_REDUCE_TENSOR_NORM1",                                      "MIOPEN_REDUCE_TENSOR_NORM1",                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 6
  {"CUDNN_REDUCE_TENSOR_NORM2",                                      {"HIPDNN_REDUCE_TENSOR_NORM2",                                      "MIOPEN_REDUCE_TENSOR_NORM2",                                      CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 7
  {"CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                               {"HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 8
  {"cudnnReduceTensorIndices_t",                                     {"hipdnnReduceTensorIndices_t",                                     "miopenReduceTensorIndices_t",                                     CONV_TYPE, API_DNN, 1}},
  {"CUDNN_REDUCE_TENSOR_NO_INDICES",                                 {"HIPDNN_REDUCE_TENSOR_NO_INDICES",                                 "MIOPEN_REDUCE_TENSOR_NO_INDICES",                                 CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_REDUCE_TENSOR_FLATTENED_INDICES",                          {"HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES",                          "MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES",                          CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnConvolutionBwdDataPreference_t",                            {"hipdnnConvolutionBwdDataPreference_t",                            "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},
  {"CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",                        {"HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 0
  {"CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",                      {"HIPDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 1
  {"CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",             {"HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT",             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED}},    // 2
  {"cudnnIndicesType_t",                                             {"hipdnnIndicesType_t",                                             "miopenIndicesType_t",                                             CONV_TYPE, API_DNN, 1}},
  {"CUDNN_32BIT_INDICES",                                            {"HIPDNN_32BIT_INDICES",                                            "MIOPEN_32BIT_INDICES",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_64BIT_INDICES",                                            {"HIPDNN_64BIT_INDICES",                                            "MIOPEN_64BIT_INDICES",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_16BIT_INDICES",                                            {"HIPDNN_16BIT_INDICES",                                            "MIOPEN_16BIT_INDICES",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_8BIT_INDICES",                                             {"HIPDNN_8BIT_INDICES",                                             "MIOPEN_8BIT_INDICES",                                             CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"cudnnSoftmaxAlgorithm_t",                                        {"hipdnnSoftmaxAlgorithm_t",                                        "miopenSoftmaxAlgorithm_t",                                        CONV_TYPE, API_DNN, 1}},
  {"CUDNN_SOFTMAX_FAST",                                             {"HIPDNN_SOFTMAX_FAST",                                             "MIOPEN_SOFTMAX_FAST",                                             CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SOFTMAX_ACCURATE",                                         {"HIPDNN_SOFTMAX_ACCURATE",                                         "MIOPEN_SOFTMAX_ACCURATE",                                         CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_SOFTMAX_LOG",                                              {"HIPDNN_SOFTMAX_LOG",                                              "MIOPEN_SOFTMAX_LOG",                                              CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"cudnnSoftmaxMode_t",                                             {"hipdnnSoftmaxMode_t",                                             "miopenSoftmaxMode_t",                                             CONV_TYPE, API_DNN, 1}},
  {"CUDNN_SOFTMAX_MODE_INSTANCE",                                    {"HIPDNN_SOFTMAX_MODE_INSTANCE",                                    "MIOPEN_SOFTMAX_MODE_INSTANCE",                                    CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_SOFTMAX_MODE_CHANNEL",                                     {"HIPDNN_SOFTMAX_MODE_CHANNEL",                                     "MIOPEN_SOFTMAX_MODE_CHANNEL",                                     CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"cudnnPoolingMode_t",                                             {"hipdnnPoolingMode_t",                                             "miopenPoolingMode_t",                                             CONV_TYPE, API_DNN, 1}},
  {"CUDNN_POOLING_MAX",                                              {"HIPDNN_POOLING_MAX",                                              "miopenPoolingMax",                                                CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",                    {"HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 1
  {"CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",                    {"HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 2
  {"CUDNN_POOLING_MAX_DETERMINISTIC",                                {"HIPDNN_POOLING_MAX_DETERMINISTIC",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 3
  {"cudnnActivationMode_t",                                          {"hipdnnActivationMode_t",                                          "miopenActivationMode_t",                                          CONV_TYPE, API_DNN, 1}},
  {"CUDNN_ACTIVATION_SIGMOID",                                       {"HIPDNN_ACTIVATION_SIGMOID",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 0
  {"CUDNN_ACTIVATION_RELU",                                          {"HIPDNN_ACTIVATION_RELU",                                          "miopenActivationRELU",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_ACTIVATION_TANH",                                          {"HIPDNN_ACTIVATION_TANH",                                          "miopenActivationTANH",                                            CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 2
  {"CUDNN_ACTIVATION_CLIPPED_RELU",                                  {"HIPDNN_ACTIVATION_CLIPPED_RELU",                                  "miopenActivationCLIPPEDRELU",                                     CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 3
  {"CUDNN_ACTIVATION_ELU",                                           {"HIPDNN_ACTIVATION_ELU",                                           "miopenActivationELU",                                             CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 4
  {"CUDNN_ACTIVATION_IDENTITY",                                      {"HIPDNN_ACTIVATION_PATHTRU",                                       "miopenActivationPASTHRU",                                         CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 5
  {"CUDNN_ACTIVATION_SWISH",                                         {"HIPDNN_ACTIVATION_SWISH",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 6
  {"cudnnBatchNormMode_t",                                           {"hipdnnBatchNormMode_t",                                           "miopenBatchNormMode_t",                                           CONV_TYPE, API_DNN, 1}},
  {"CUDNN_BATCHNORM_PER_ACTIVATION",                                 {"HIPDNN_BATCHNORM_PER_ACTIVATION",                                 "miopenBNPerActivation",                                           CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 0
  {"CUDNN_BATCHNORM_SPATIAL",                                        {"HIPDNN_BATCHNORM_SPATIAL",                                        "miopenBNSpatial",                                                 CONV_NUMERIC_LITERAL, API_DNN, 1}},    // 1
  {"CUDNN_BATCHNORM_SPATIAL_PERSISTENT",                             {"HIPDNN_BATCHNORM_SPATIAL_PERSISTENT",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, ROC_UNSUPPORTED}},    // 2
  {"cudnnSamplerType_t",                                             {"hipdnnSamplerType_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SAMPLER_BILINEAR",                                         {"HIPDNN_SAMPLER_BILINEAR",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"cudnnBatchNormOps_t",                                            {"hipdnnBatchNormOps_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BATCHNORM_OPS_BN",                                         {"HIPDNN_BATCHNORM_OPS_BN",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_BATCHNORM_OPS_BN_ACTIVATION",                              {"HIPDNN_BATCHNORM_OPS_BN_ACTIVATION",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",                          {"HIPDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"cudnnRNNClipMode_t",                                             {"hipdnnRNNClipMode_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNN_CLIP_NONE",                                            {"HIPDNN_RNN_CLIP_NONE",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_RNN_CLIP_MINMAX",                                          {"HIPDNN_RNN_CLIP_MINMAX",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnRNNDataLayout_t",                                           {"hipdnnRNNDataLayout_t",                                           "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",                       {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",                         {"HIPDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",                     {"HIPDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"cudnnRNNPaddingMode_t",                                          {"hipdnnRNNPaddingMode_t",                                          "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNN_PADDED_IO_DISABLED",                                   {"HIPDNN_RNN_PADDED_IO_DISABLED",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_RNN_PADDED_IO_ENABLED",                                    {"HIPDNN_RNN_PADDED_IO_ENABLED",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnSeqDataAxis_t",                                             {"hipdnnSeqDataAxis_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SEQDATA_TIME_DIM",                                         {"HIPDNN_SEQDATA_TIME_DIM",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_SEQDATA_BATCH_DIM",                                        {"HIPDNN_SEQDATA_BATCH_DIM",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_SEQDATA_BEAM_DIM",                                         {"HIPDNN_SEQDATA_BEAM_DIM",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_SEQDATA_VECT_DIM",                                         {"HIPDNN_SEQDATA_VECT_DIM",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"cudnnAttnQueryMap_t",                                            {"hipdnnAttnQueryMap_t",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_ATTN_QUERYMAP_ALL_TO_ONE",                                 {"HIPDNN_ATTN_QUERYMAP_ALL_TO_ONE",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_QUERYMAP_ONE_TO_ONE",                                 {"HIPDNN_ATTN_QUERYMAP_ONE_TO_ONE",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1U << 0
  {"CUDNN_ATTN_DISABLE_PROJ_BIASES",                                 {"HIPDNN_ATTN_DISABLE_PROJ_BIASES",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_ATTN_ENABLE_PROJ_BIASES",                                  {"HIPDNN_ATTN_ENABLE_PROJ_BIASES",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1U << 1
  {"cudnnMultiHeadAttnWeightKind_t",                                 {"hipdnnMultiHeadAttnWeightKind_t",                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_MH_ATTN_Q_WEIGHTS",                                        {"HIPDNN_MH_ATTN_Q_WEIGHTS",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_MH_ATTN_K_WEIGHTS",                                        {"HIPDNN_MH_ATTN_K_WEIGHTS",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_MH_ATTN_V_WEIGHTS",                                        {"HIPDNN_MH_ATTN_V_WEIGHTS",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_MH_ATTN_O_WEIGHTS",                                        {"HIPDNN_MH_ATTN_O_WEIGHTS",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_MH_ATTN_Q_BIASES",                                         {"HIPDNN_MH_ATTN_Q_BIASES",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_MH_ATTN_K_BIASES",                                         {"HIPDNN_MH_ATTN_K_BIASES",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_MH_ATTN_V_BIASES",                                         {"HIPDNN_MH_ATTN_V_BIASES",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"CUDNN_MH_ATTN_O_BIASES",                                         {"HIPDNN_MH_ATTN_O_BIASES",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 7
  {"CUDNN_ATTN_WKIND_COUNT",                                         {"HIPDNN_ATTN_WKIND_COUNT",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 8
  {"cudnnWgradMode_t",                                               {"hipdnnWgradMode_t",                                               "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_WGRAD_MODE_ADD",                                           {"HIPDNN_WGRAD_MODE_ADD",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_WGRAD_MODE_SET",                                           {"HIPDNN_WGRAD_MODE_SET",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnReorderType_t",                                             {"hipdnnReorderType_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_DEFAULT_REORDER",                                          {"HIPDNN_DEFAULT_REORDER",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_NO_REORDER",                                               {"HIPDNN_NO_REORDER",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnLossNormalizationMode_t",                                   {"hipdnnLossNormalizationMode_t",                                   "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LOSS_NORMALIZATION_NONE",                                  {"HIPDNN_LOSS_NORMALIZATION_NONE",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_LOSS_NORMALIZATION_SOFTMAX",                               {"HIPDNN_LOSS_NORMALIZATION_SOFTMAX",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnFusedOps_t",                                                {"hipdnnFusedOps_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",                 {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",                        {"HIPDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",                    {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",                   {"HIPDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",                     {"HIPDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",              {"HIPDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",                        {"HIPDNN_FUSED_DACTIVATION_FORK_DBATCHNORM",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"cudnnFusedOpsConstParamLabel_t",                                 {"hipdnnFusedOpsConstParamLabel_t",                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_PARAM_XDESC",                                              {"HIPDNN_PARAM_XDESC",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_PARAM_XDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_XDATA_PLACEHOLDER",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_PARAM_BN_MODE",                                            {"HIPDNN_PARAM_BN_MODE",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_PARAM_BN_EQSCALEBIAS_DESC",                                {"HIPDNN_PARAM_BN_EQSCALEBIAS_DESC",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                             {"HIPDNN_PARAM_BN_EQSCALE_PLACEHOLDER",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                              {"HIPDNN_PARAM_BN_EQBIAS_PLACEHOLDER",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_PARAM_ACTIVATION_DESC",                                    {"HIPDNN_PARAM_ACTIVATION_DESC",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"CUDNN_PARAM_CONV_DESC",                                          {"HIPDNN_PARAM_CONV_DESC",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 7
  {"CUDNN_PARAM_WDESC",                                              {"HIPDNN_PARAM_WDESC",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 8
  {"CUDNN_PARAM_WDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_WDATA_PLACEHOLDER",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 9
  {"CUDNN_PARAM_DWDESC",                                             {"HIPDNN_PARAM_DWDESC",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 10
  {"CUDNN_PARAM_DWDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DWDATA_PLACEHOLDER",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 11
  {"CUDNN_PARAM_YDESC",                                              {"HIPDNN_PARAM_YDESC",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 12
  {"CUDNN_PARAM_YDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_YDATA_PLACEHOLDER",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 13
  {"CUDNN_PARAM_DYDESC",                                             {"HIPDNN_PARAM_DYDESC",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 14
  {"CUDNN_PARAM_DYDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DYDATA_PLACEHOLDER",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 15
  {"CUDNN_PARAM_YSTATS_DESC",                                        {"HIPDNN_PARAM_YSTATS_DESC",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 16
  {"CUDNN_PARAM_YSUM_PLACEHOLDER",                                   {"HIPDNN_PARAM_YSUM_PLACEHOLDER",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 17
  {"CUDNN_PARAM_YSQSUM_PLACEHOLDER",                                 {"HIPDNN_PARAM_YSQSUM_PLACEHOLDER",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 18
  {"CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",                          {"HIPDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 19
  {"CUDNN_PARAM_BN_SCALE_PLACEHOLDER",                               {"HIPDNN_PARAM_BN_SCALE_PLACEHOLDER",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 20
  {"CUDNN_PARAM_BN_BIAS_PLACEHOLDER",                                {"HIPDNN_PARAM_BN_BIAS_PLACEHOLDER",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 21
  {"CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",                          {"HIPDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 22
  {"CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",                        {"HIPDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 23
  {"CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",                        {"HIPDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 24
  {"CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",                         {"HIPDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 25
  {"CUDNN_PARAM_ZDESC",                                              {"HIPDNN_PARAM_ZDESC",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 26
  {"CUDNN_PARAM_ZDATA_PLACEHOLDER",                                  {"HIPDNN_PARAM_ZDATA_PLACEHOLDER",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 27
  {"CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                              {"HIPDNN_PARAM_BN_Z_EQSCALEBIAS_DESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 28
  {"CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                           {"HIPDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 29
  {"CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                            {"HIPDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 30
  {"CUDNN_PARAM_ACTIVATION_BITMASK_DESC",                            {"HIPDNN_PARAM_ACTIVATION_BITMASK_DESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 31
  {"CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",                     {"HIPDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 32
  {"CUDNN_PARAM_DXDESC",                                             {"HIPDNN_PARAM_DXDESC",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 33
  {"CUDNN_PARAM_DXDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DXDATA_PLACEHOLDER",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 34
  {"CUDNN_PARAM_DZDESC",                                             {"HIPDNN_PARAM_DZDESC",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 35
  {"CUDNN_PARAM_DZDATA_PLACEHOLDER",                                 {"HIPDNN_PARAM_DZDATA_PLACEHOLDER",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 36
  {"CUDNN_PARAM_BN_DSCALE_PLACEHOLDER",                              {"HIPDNN_PARAM_BN_DSCALE_PLACEHOLDER",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 37
  {"CUDNN_PARAM_BN_DBIAS_PLACEHOLDER",                               {"HIPDNN_PARAM_BN_DBIAS_PLACEHOLDER",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 38
  {"cudnnFusedOpsPointerPlaceHolder_t",                              {"hipdnnFusedOpsPointerPlaceHolder_t",                              "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_PTR_NULL",                                                 {"HIPDNN_PTR_NULL",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_PTR_ELEM_ALIGNED",                                         {"HIPDNN_PTR_ELEM_ALIGNED",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_PTR_16B_ALIGNED",                                          {"HIPDNN_PTR_16B_ALIGNED",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"cudnnFusedOpsVariantParamLabel_t",                               {"hipdnnFusedOpsVariantParamLabel_t",                               "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_PTR_XDATA",                                                {"HIPDNN_PTR_XDATA",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_PTR_BN_EQSCALE",                                           {"HIPDNN_PTR_BN_EQSCALE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_PTR_BN_EQBIAS",                                            {"HIPDNN_PTR_BN_EQBIAS",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_PTR_WDATA",                                                {"HIPDNN_PTR_WDATA",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_PTR_DWDATA",                                               {"HIPDNN_PTR_DWDATA",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_PTR_YDATA",                                                {"HIPDNN_PTR_YDATA",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_PTR_DYDATA",                                               {"HIPDNN_PTR_DYDATA",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"CUDNN_PTR_YSUM",                                                 {"HIPDNN_PTR_YSUM",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 7
  {"CUDNN_PTR_YSQSUM",                                               {"HIPDNN_PTR_YSQSUM",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 8
  {"CUDNN_PTR_WORKSPACE",                                            {"HIPDNN_PTR_WORKSPACE",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 9
  {"CUDNN_PTR_BN_SCALE",                                             {"HIPDNN_PTR_BN_SCALE",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 10
  {"CUDNN_PTR_BN_BIAS",                                              {"HIPDNN_PTR_BN_BIAS",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 11
  {"CUDNN_PTR_BN_SAVED_MEAN",                                        {"HIPDNN_PTR_BN_SAVED_MEAN",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 12
  {"CUDNN_PTR_BN_SAVED_INVSTD",                                      {"HIPDNN_PTR_BN_SAVED_INVSTD",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 13
  {"CUDNN_PTR_BN_RUNNING_MEAN",                                      {"HIPDNN_PTR_BN_RUNNING_MEAN",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 14
  {"CUDNN_PTR_BN_RUNNING_VAR",                                       {"HIPDNN_PTR_BN_RUNNING_VAR",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 15
  {"CUDNN_PTR_ZDATA",                                                {"HIPDNN_PTR_ZDATA",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 16
  {"CUDNN_PTR_BN_Z_EQSCALE",                                         {"HIPDNN_PTR_BN_Z_EQSCALE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 17
  {"CUDNN_PTR_BN_Z_EQBIAS",                                          {"HIPDNN_PTR_BN_Z_EQBIAS",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 18
  {"CUDNN_PTR_ACTIVATION_BITMASK",                                   {"HIPDNN_PTR_ACTIVATION_BITMASK",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 19
  {"CUDNN_PTR_DXDATA",                                               {"HIPDNN_PTR_DXDATA",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 20
  {"CUDNN_PTR_DZDATA",                                               {"HIPDNN_PTR_DZDATA",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 21
  {"CUDNN_PTR_BN_DSCALE",                                            {"HIPDNN_PTR_BN_DSCALE",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 22
  {"CUDNN_PTR_BN_DBIAS",                                             {"HIPDNN_PTR_BN_DBIAS",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 23
  {"CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",                    {"HIPDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 100
  {"CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",                     {"HIPDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 101
  {"CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",                          {"HIPDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 102
  {"CUDNN_SCALAR_DOUBLE_BN_EPSILON",                                 {"HIPDNN_SCALAR_DOUBLE_BN_EPSILON",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 103
  {"cudnnForwardMode_t",                                             {"hipdnnForwardMode_t",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_FWD_MODE_INFERENCE",                                       {"HIPDNN_FWD_MODE_INFERENCE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_FWD_MODE_TRAINING",                                        {"HIPDNN_FWD_MODE_TRAINING",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"cudnnPointwiseMode_t",                                           {"hipdnnPointwiseMode_t",                                           "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_POINTWISE_ADD",                                            {"HIPDNN_POINTWISE_ADD",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_POINTWISE_MUL",                                            {"HIPDNN_POINTWISE_MUL",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_POINTWISE_MIN",                                            {"HIPDNN_POINTWISE_MIN",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_POINTWISE_MAX",                                            {"HIPDNN_POINTWISE_MAX",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_POINTWISE_SQRT",                                           {"HIPDNN_POINTWISE_SQRT",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_POINTWISE_ADD_SQUARE",                                     {"HIPDNN_POINTWISE_ADD_SQUARE",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_POINTWISE_DIV",                                            {"HIPDNN_POINTWISE_DIV",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"CUDNN_POINTWISE_MOD",                                            {"HIPDNN_POINTWISE_MOD",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 7
  {"CUDNN_POINTWISE_POW",                                            {"HIPDNN_POINTWISE_POW",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 8
  {"CUDNN_POINTWISE_SUB",                                            {"HIPDNN_POINTWISE_SUB",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 9
  {"CUDNN_POINTWISE_ABS",                                            {"HIPDNN_POINTWISE_ABS",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 10
  {"CUDNN_POINTWISE_CEIL",                                           {"HIPDNN_POINTWISE_CEIL",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 11
  {"CUDNN_POINTWISE_COS",                                            {"HIPDNN_POINTWISE_COS",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 12
  {"CUDNN_POINTWISE_EXP",                                            {"HIPDNN_POINTWISE_EXP",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 13
  {"CUDNN_POINTWISE_FLOOR",                                          {"HIPDNN_POINTWISE_FLOOR",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 14
  {"CUDNN_POINTWISE_LOG",                                            {"HIPDNN_POINTWISE_LOG",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 15
  {"CUDNN_POINTWISE_NEG",                                            {"HIPDNN_POINTWISE_NEG",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 16
  {"CUDNN_POINTWISE_RSQRT",                                          {"HIPDNN_POINTWISE_RSQRT",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 17
  {"CUDNN_POINTWISE_SIN",                                            {"HIPDNN_POINTWISE_SIN",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 18
  {"CUDNN_POINTWISE_TAN",                                            {"HIPDNN_POINTWISE_TAN",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 19
  {"CUDNN_POINTWISE_ERF",                                            {"HIPDNN_POINTWISE_ERF",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 20
  {"CUDNN_POINTWISE_IDENTITY",                                       {"HIPDNN_POINTWISE_IDENTITY",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 21
  {"CUDNN_POINTWISE_RECIPROCAL",                                     {"HIPDNN_POINTWISE_RECIPROCAL",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 22
  {"CUDNN_POINTWISE_RELU_FWD",                                       {"HIPDNN_POINTWISE_RELU_FWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 100
  {"CUDNN_POINTWISE_TANH_FWD",                                       {"HIPDNN_POINTWISE_TANH_FWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 101
  {"CUDNN_POINTWISE_SIGMOID_FWD",                                    {"HIPDNN_POINTWISE_SIGMOID_FWD",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 102
  {"CUDNN_POINTWISE_ELU_FWD",                                        {"HIPDNN_POINTWISE_ELU_FWD",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 103
  {"CUDNN_POINTWISE_GELU_FWD",                                       {"HIPDNN_POINTWISE_GELU_FWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 104
  {"CUDNN_POINTWISE_SOFTPLUS_FWD",                                   {"HIPDNN_POINTWISE_SOFTPLUS_FWD",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 105
  {"CUDNN_POINTWISE_SWISH_FWD",                                      {"HIPDNN_POINTWISE_SWISH_FWD",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 106
  {"CUDNN_POINTWISE_GELU_APPROX_TANH_FWD",                           {"HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 107
  {"CUDNN_POINTWISE_RELU_BWD",                                       {"HIPDNN_POINTWISE_RELU_BWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 200
  {"CUDNN_POINTWISE_TANH_BWD",                                       {"HIPDNN_POINTWISE_TANH_BWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 201
  {"CUDNN_POINTWISE_SIGMOID_BWD",                                    {"HIPDNN_POINTWISE_SIGMOID_BWD",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 202
  {"CUDNN_POINTWISE_ELU_BWD",                                        {"HIPDNN_POINTWISE_ELU_BWD",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 203
  {"CUDNN_POINTWISE_GELU_BWD",                                       {"HIPDNN_POINTWISE_GELU_BWD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 204
  {"CUDNN_POINTWISE_SOFTPLUS_BWD",                                   {"HIPDNN_POINTWISE_SOFTPLUS_BWD",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 205
  {"CUDNN_POINTWISE_SWISH_BWD",                                      {"HIPDNN_POINTWISE_SWISH_BWD",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 206
  {"CUDNN_POINTWISE_GELU_APPROX_TANH_BWD",                           {"HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 207
  {"CUDNN_POINTWISE_CMP_EQ",                                         {"HIPDNN_POINTWISE_CMP_EQ",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 300
  {"CUDNN_POINTWISE_CMP_NEQ",                                        {"HIPDNN_POINTWISE_CMP_NEQ",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 301
  {"CUDNN_POINTWISE_CMP_GT",                                         {"HIPDNN_POINTWISE_CMP_GT",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 302
  {"CUDNN_POINTWISE_CMP_GE",                                         {"HIPDNN_POINTWISE_CMP_GE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 303
  {"CUDNN_POINTWISE_CMP_LT",                                         {"HIPDNN_POINTWISE_CMP_LT",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 304
  {"CUDNN_POINTWISE_CMP_LE",                                         {"HIPDNN_POINTWISE_CMP_LE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 305
  {"CUDNN_POINTWISE_LOGICAL_AND",                                    {"HIPDNN_POINTWISE_LOGICAL_AND",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 400
  {"CUDNN_POINTWISE_LOGICAL_OR",                                     {"HIPDNN_POINTWISE_LOGICAL_OR",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 401
  {"CUDNN_POINTWISE_LOGICAL_NOT",                                    {"HIPDNN_POINTWISE_LOGICAL_NOT",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 402
  {"CUDNN_POINTWISE_GEN_INDEX",                                      {"HIPDNN_POINTWISE_GEN_INDEX",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 501
  {"CUDNN_POINTWISE_BINARY_SELECT",                                  {"HIPDNN_POINTWISE_BINARY_SELECT",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 601
  {"cudnnGenStatsMode_t",                                            {"hipdnnGenStatsMode_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_GENSTATS_SUM_SQSUM",                                       {"HIPDNN_GENSTATS_SUM_SQSUM",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"cudnnBackendAttributeName_t",                                    {"hipdnnBackendAttributeName_t",                                    "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_ATTR_POINTWISE_MODE",                                      {"HIPDNN_ATTR_POINTWISE_MODE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 0
  {"CUDNN_ATTR_POINTWISE_MATH_PREC",                                 {"HIPDNN_ATTR_POINTWISE_MATH_PREC",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1
  {"CUDNN_ATTR_POINTWISE_NAN_PROPAGATION",                           {"HIPDNN_ATTR_POINTWISE_NAN_PROPAGATION",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",                           {"HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 3
  {"CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",                           {"HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 4
  {"CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",                     {"HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 5
  {"CUDNN_ATTR_POINTWISE_ELU_ALPHA",                                 {"HIPDNN_ATTR_POINTWISE_ELU_ALPHA",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 6
  {"CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA",                             {"HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 7
  {"CUDNN_ATTR_POINTWISE_SWISH_BETA",                                {"HIPDNN_ATTR_POINTWISE_SWISH_BETA",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 8
  {"CUDNN_ATTR_POINTWISE_AXIS",                                      {"HIPDNN_ATTR_POINTWISE_AXIS",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 9
  {"CUDNN_ATTR_CONVOLUTION_COMP_TYPE",                               {"HIPDNN_ATTR_CONVOLUTION_COMP_TYPE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 100
  {"CUDNN_ATTR_CONVOLUTION_CONV_MODE",                               {"HIPDNN_ATTR_CONVOLUTION_CONV_MODE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 101
  {"CUDNN_ATTR_CONVOLUTION_DILATIONS",                               {"HIPDNN_ATTR_CONVOLUTION_DILATIONS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 102
  {"CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES",                          {"HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 103
  {"CUDNN_ATTR_CONVOLUTION_POST_PADDINGS",                           {"HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 104
  {"CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS",                            {"HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 105
  {"CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",                            {"HIPDNN_ATTR_CONVOLUTION_SPATIAL_DIMS",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 106
  {"CUDNN_ATTR_ENGINEHEUR_MODE",                                     {"HIPDNN_ATTR_ENGINEHEUR_MODE",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 200
  {"CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",                          {"HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 201
  {"CUDNN_ATTR_ENGINEHEUR_RESULTS",                                  {"HIPDNN_ATTR_ENGINEHEUR_RESULTS",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 202
  {"CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",                          {"HIPDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 203
  {"CUDNN_ATTR_ENGINECFG_ENGINE",                                    {"HIPDNN_ATTR_ENGINECFG_ENGINE",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 300
  {"CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",                         {"HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 301
  {"CUDNN_ATTR_ENGINECFG_KNOB_CHOICES",                              {"HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 302
  {"CUDNN_ATTR_EXECUTION_PLAN_HANDLE",                               {"HIPDNN_ATTR_EXECUTION_PLAN_HANDLE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 400
  {"CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",                        {"HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 401
  {"CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",                       {"HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 402
  {"CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",           {"HIPDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS",           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 403
  {"CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",           {"HIPDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS",           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 404
  {"CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION",                  {"HIPDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 405
  {"CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",                         {"HIPDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 500
  {"CUDNN_ATTR_INTERMEDIATE_INFO_SIZE",                              {"HIPDNN_ATTR_INTERMEDIATE_INFO_SIZE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 501
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",               {"HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 502
  {"CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",              {"HIPDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 503
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",                               {"HIPDNN_ATTR_KNOB_CHOICE_KNOB_TYPE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 600
  {"CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",                              {"HIPDNN_ATTR_KNOB_CHOICE_KNOB_VALUE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 601
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 700
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",                  {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 701
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",             {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC",             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 702
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 703
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 704
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",                     {"HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 705
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",                {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 706
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 707
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",            {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC",            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 708
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",                    {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 709
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",                   {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 710
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",                   {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 711
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",              {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 712
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",               {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 713
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",          {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC",          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 714
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 715
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",                  {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 716
  {"CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",                 {"HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 717
  {"CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",                   {"HIPDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 750
  {"CUDNN_ATTR_OPERATION_POINTWISE_XDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_XDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 751
  {"CUDNN_ATTR_OPERATION_POINTWISE_BDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_BDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 752
  {"CUDNN_ATTR_OPERATION_POINTWISE_YDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_YDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 753
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_ALPHA1",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 754
  {"CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_ALPHA2",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 755
  {"CUDNN_ATTR_OPERATION_POINTWISE_DXDESC",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_DXDESC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 756
  {"CUDNN_ATTR_OPERATION_POINTWISE_DYDESC",                          {"HIPDNN_ATTR_OPERATION_POINTWISE_DYDESC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 757
  {"CUDNN_ATTR_OPERATION_POINTWISE_TDESC",                           {"HIPDNN_ATTR_OPERATION_POINTWISE_TDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 758
  {"CUDNN_ATTR_OPERATION_GENSTATS_MODE",                             {"HIPDNN_ATTR_OPERATION_GENSTATS_MODE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 770
  {"CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",                        {"HIPDNN_ATTR_OPERATION_GENSTATS_MATH_PREC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 771
  {"CUDNN_ATTR_OPERATION_GENSTATS_XDESC",                            {"HIPDNN_ATTR_OPERATION_GENSTATS_XDESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 772
  {"CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC",                          {"HIPDNN_ATTR_OPERATION_GENSTATS_SUMDESC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 773
  {"CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",                        {"HIPDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 774
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 780
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",                     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 781
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 782
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",                 {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 783
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",                    {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 784
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",                     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 785
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",        {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC",        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 786
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",         {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC",         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 787
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",     {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC",     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 788
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",      {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC",      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 789
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",               {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 790
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",            {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC",            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 791
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",                 {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC",                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 792
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",                  {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 793
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",              {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 794
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",                  {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 795
  {"CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",       {"HIPDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC",       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 796
  {"CUDNN_ATTR_OPERATIONGRAPH_HANDLE",                               {"HIPDNN_ATTR_OPERATIONGRAPH_HANDLE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 800
  {"CUDNN_ATTR_OPERATIONGRAPH_OPS",                                  {"HIPDNN_ATTR_OPERATIONGRAPH_OPS",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 801
  {"CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",                  {"HIPDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 802
  {"CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT",                               {"HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 900
  {"CUDNN_ATTR_TENSOR_DATA_TYPE",                                    {"HIPDNN_ATTR_TENSOR_DATA_TYPE",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 901
  {"CUDNN_ATTR_TENSOR_DIMENSIONS",                                   {"HIPDNN_ATTR_TENSOR_DIMENSIONS",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 902
  {"CUDNN_ATTR_TENSOR_STRIDES",                                      {"HIPDNN_ATTR_TENSOR_STRIDES",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 903
  {"CUDNN_ATTR_TENSOR_VECTOR_COUNT",                                 {"HIPDNN_ATTR_TENSOR_VECTOR_COUNT",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 904
  {"CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",                         {"HIPDNN_ATTR_TENSOR_VECTORIZED_DIMENSION",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 905
  {"CUDNN_ATTR_TENSOR_UNIQUE_ID",                                    {"HIPDNN_ATTR_TENSOR_UNIQUE_ID",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 906
  {"CUDNN_ATTR_TENSOR_IS_VIRTUAL",                                   {"HIPDNN_ATTR_TENSOR_IS_VIRTUAL",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 907
  {"CUDNN_ATTR_TENSOR_IS_BY_VALUE",                                  {"HIPDNN_ATTR_TENSOR_IS_BY_VALUE",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 908
  {"CUDNN_ATTR_TENSOR_REORDERING_MODE",                              {"HIPDNN_ATTR_TENSOR_REORDERING_MODE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 909
  {"CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC",                           {"HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 913
  {"CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",                             {"HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1000
  {"CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS",                          {"HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1001
  {"CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES",                          {"HIPDNN_ATTR_VARIANT_PACK_INTERMEDIATES",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1002
  {"CUDNN_ATTR_VARIANT_PACK_WORKSPACE",                              {"HIPDNN_ATTR_VARIANT_PACK_WORKSPACE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1003
  {"CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID",                              {"HIPDNN_ATTR_LAYOUT_INFO_TENSOR_UID",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1100
  {"CUDNN_ATTR_LAYOUT_INFO_TYPES",                                   {"HIPDNN_ATTR_LAYOUT_INFO_TYPES",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1101
  {"CUDNN_ATTR_KNOB_INFO_TYPE",                                      {"HIPDNN_ATTR_KNOB_INFO_TYPE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1200
  {"CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",                             {"HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1201
  {"CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",                             {"HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1202
  {"CUDNN_ATTR_KNOB_INFO_STRIDE",                                    {"HIPDNN_ATTR_KNOB_INFO_STRIDE",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1203
  {"CUDNN_ATTR_ENGINE_OPERATION_GRAPH",                              {"HIPDNN_ATTR_ENGINE_OPERATION_GRAPH",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1300
  {"CUDNN_ATTR_ENGINE_GLOBAL_INDEX",                                 {"HIPDNN_ATTR_ENGINE_GLOBAL_INDEX",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1301
  {"CUDNN_ATTR_ENGINE_KNOB_INFO",                                    {"HIPDNN_ATTR_ENGINE_KNOB_INFO",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1302
  {"CUDNN_ATTR_ENGINE_NUMERICAL_NOTE",                               {"HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1303
  {"CUDNN_ATTR_ENGINE_LAYOUT_INFO",                                  {"HIPDNN_ATTR_ENGINE_LAYOUT_INFO",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1304
  {"CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE",                                {"HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1305
  {"CUDNN_ATTR_ENGINE_SM_COUNT_TARGET",                              {"HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1306
  {"CUDNN_ATTR_MATMUL_COMP_TYPE",                                    {"HIPDNN_ATTR_MATMUL_COMP_TYPE",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1500
  {"CUDNN_ATTR_MATMUL_PADDING_VALUE",                                {"HIPDNN_ATTR_MATMUL_PADDING_VALUE",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1503
  {"CUDNN_ATTR_OPERATION_MATMUL_ADESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_ADESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1520
  {"CUDNN_ATTR_OPERATION_MATMUL_BDESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_BDESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1521
  {"CUDNN_ATTR_OPERATION_MATMUL_CDESC",                              {"HIPDNN_ATTR_OPERATION_MATMUL_CDESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1522
  {"CUDNN_ATTR_OPERATION_MATMUL_DESC",                               {"HIPDNN_ATTR_OPERATION_MATMUL_DESC",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1523
  {"CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",    {"HIPDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT",    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1524
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",               {"HIPDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1525
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",               {"HIPDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1526
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",               {"HIPDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1527
  {"CUDNN_ATTR_REDUCTION_OPERATOR",                                  {"HIPDNN_ATTR_REDUCTION_OPERATOR",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1600
  {"CUDNN_ATTR_REDUCTION_COMP_TYPE",                                 {"HIPDNN_ATTR_REDUCTION_COMP_TYPE",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1601
  {"CUDNN_ATTR_OPERATION_REDUCTION_XDESC",                           {"HIPDNN_ATTR_OPERATION_REDUCTION_XDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1610
  {"CUDNN_ATTR_OPERATION_REDUCTION_YDESC",                           {"HIPDNN_ATTR_OPERATION_REDUCTION_YDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1611
  {"CUDNN_ATTR_OPERATION_REDUCTION_DESC",                            {"HIPDNN_ATTR_OPERATION_REDUCTION_DESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1612
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC",                  {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1620
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC",                  {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1621
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC",                {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1622
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC",              {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1623
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC",                     {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1624
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC",                    {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1625
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",             {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1626
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC",              {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1627
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC",           {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC",           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1628
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC",            {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC",            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1629
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS",                    {"HIPDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS",                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1630
  {"CUDNN_ATTR_RESAMPLE_MODE",                                       {"HIPDNN_ATTR_RESAMPLE_MODE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1700
  {"CUDNN_ATTR_RESAMPLE_COMP_TYPE",                                  {"HIPDNN_ATTR_RESAMPLE_COMP_TYPE",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1701
  {"CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS",                               {"HIPDNN_ATTR_RESAMPLE_SPATIAL_DIMS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1702
  {"CUDNN_ATTR_RESAMPLE_POST_PADDINGS",                              {"HIPDNN_ATTR_RESAMPLE_POST_PADDINGS",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1703
  {"CUDNN_ATTR_RESAMPLE_PRE_PADDINGS",                               {"HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1704
  {"CUDNN_ATTR_RESAMPLE_STRIDES",                                    {"HIPDNN_ATTR_RESAMPLE_STRIDES",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1705
  {"CUDNN_ATTR_RESAMPLE_WINDOW_DIMS",                                {"HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1706
  {"CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION",                            {"HIPDNN_ATTR_RESAMPLE_NAN_PROPAGATION",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1707
  {"CUDNN_ATTR_RESAMPLE_PADDING_MODE",                               {"HIPDNN_ATTR_RESAMPLE_PADDING_MODE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1708
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1710
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1711
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC",                      {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1712
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1713
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA",                         {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1714
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC",                         {"HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1716
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC",                       {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1720
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC",                       {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1721
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC",                      {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1722
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1723
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA",                         {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1724
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC",                         {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1725
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1726
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC",                        {"HIPDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1727
  {"CUDNN_ATTR_OPERATION_CONCAT_AXIS",                               {"HIPDNN_ATTR_OPERATION_CONCAT_AXIS",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1800
  {"CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS",                        {"HIPDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1801
  {"CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX",                      {"HIPDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1802
  {"CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC",                        {"HIPDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1803
  {"CUDNN_ATTR_OPERATION_SIGNAL_MODE",                               {"HIPDNN_ATTR_OPERATION_SIGNAL_MODE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1900
  {"CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC",                           {"HIPDNN_ATTR_OPERATION_SIGNAL_FLAGDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1901
  {"CUDNN_ATTR_OPERATION_SIGNAL_VALUE",                              {"HIPDNN_ATTR_OPERATION_SIGNAL_VALUE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1902
  {"CUDNN_ATTR_OPERATION_SIGNAL_XDESC",                              {"HIPDNN_ATTR_OPERATION_SIGNAL_XDESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1903
  {"CUDNN_ATTR_OPERATION_SIGNAL_YDESC",                              {"HIPDNN_ATTR_OPERATION_SIGNAL_YDESC",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 1904
  {"CUDNN_ATTR_OPERATION_NORM_FWD_MODE",                             {"HIPDNN_ATTR_OPERATION_NORM_FWD_MODE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2000
  {"CUDNN_ATTR_OPERATION_NORM_FWD_PHASE",                            {"HIPDNN_ATTR_OPERATION_NORM_FWD_PHASE",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2001
  {"CUDNN_ATTR_OPERATION_NORM_FWD_XDESC",                            {"HIPDNN_ATTR_OPERATION_NORM_FWD_XDESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2002
  {"CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC",                        {"HIPDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2003
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC",                {"HIPDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2004
  {"CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC",                       {"HIPDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2005
  {"CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC",                        {"HIPDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2006
  {"CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC",                     {"HIPDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2007
  {"CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC",              {"HIPDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2008
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC",          {"HIPDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC",          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2009
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC",           {"HIPDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC",           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2010
  {"CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC",         {"HIPDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC",         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2011
  {"CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC",          {"HIPDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC",          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2012
  {"CUDNN_ATTR_OPERATION_NORM_FWD_YDESC",                            {"HIPDNN_ATTR_OPERATION_NORM_FWD_YDESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2013
  {"CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS",                  {"HIPDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2014
  {"CUDNN_ATTR_OPERATION_NORM_BWD_MODE",                             {"HIPDNN_ATTR_OPERATION_NORM_BWD_MODE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2100
  {"CUDNN_ATTR_OPERATION_NORM_BWD_XDESC",                            {"HIPDNN_ATTR_OPERATION_NORM_BWD_XDESC",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2101
  {"CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC",                        {"HIPDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2102
  {"CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC",                {"HIPDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2103
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC",                           {"HIPDNN_ATTR_OPERATION_NORM_BWD_DYDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2104
  {"CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC",                       {"HIPDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2105
  {"CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC",                     {"HIPDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2106
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC",                      {"HIPDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2107
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC",                       {"HIPDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2108
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC",                           {"HIPDNN_ATTR_OPERATION_NORM_BWD_DXDESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2109
  {"CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS",                  {"HIPDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2110
  {"CUDNN_ATTR_OPERATION_RESHAPE_XDESC",                             {"HIPDNN_ATTR_OPERATION_RESHAPE_XDESC",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2200
  {"CUDNN_ATTR_OPERATION_RESHAPE_YDESC",                             {"HIPDNN_ATTR_OPERATION_RESHAPE_YDESC",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2201
  {"CUDNN_ATTR_RNG_DISTRIBUTION",                                    {"HIPDNN_ATTR_RNG_DISTRIBUTION",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2300
  {"CUDNN_ATTR_RNG_NORMAL_DIST_MEAN",                                {"HIPDNN_ATTR_RNG_NORMAL_DIST_MEAN",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2301
  {"CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",                  {"HIPDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2302
  {"CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM",                            {"HIPDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2303
  {"CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM",                            {"HIPDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2304
  {"CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",                      {"HIPDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2305
  {"CUDNN_ATTR_OPERATION_RNG_YDESC",                                 {"HIPDNN_ATTR_OPERATION_RNG_YDESC",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2310
  {"CUDNN_ATTR_OPERATION_RNG_SEED",                                  {"HIPDNN_ATTR_OPERATION_RNG_SEED",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2311
  {"CUDNN_ATTR_OPERATION_RNG_DESC",                                  {"HIPDNN_ATTR_OPERATION_RNG_DESC",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2312
  {"CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC",                           {"HIPDNN_ATTR_OPERATION_RNG_OFFSET_DESC",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},    // 2313
  {"cudnnBackendAttributeType_t",                                    {"hipdnnBackendAttributeType_t",                                    "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_HANDLE",                                              {"HIPDNN_TYPE_HANDLE",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_DATA_TYPE",                                           {"HIPDNN_TYPE_DATA_TYPE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_BOOLEAN",                                             {"HIPDNN_TYPE_BOOLEAN",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_INT64",                                               {"HIPDNN_TYPE_INT64",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_FLOAT",                                               {"HIPDNN_TYPE_FLOAT",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_DOUBLE",                                              {"HIPDNN_TYPE_FLOAT",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_VOID_PTR",                                            {"HIPDNN_TYPE_VOID_PTR",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_CONVOLUTION_MODE",                                    {"HIPDNN_TYPE_CONVOLUTION_MODE",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_HEUR_MODE",                                           {"HIPDNN_TYPE_HEUR_MODE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_KNOB_TYPE",                                           {"HIPDNN_TYPE_KNOB_TYPE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_NAN_PROPOGATION",                                     {"HIPDNN_TYPE_NAN_PROPOGATION",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_NUMERICAL_NOTE",                                      {"HIPDNN_TYPE_NUMERICAL_NOTE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_LAYOUT_TYPE",                                         {"HIPDNN_TYPE_LAYOUT_TYPE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_ATTRIB_NAME",                                         {"HIPDNN_TYPE_ATTRIB_NAME",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_POINTWISE_MODE",                                      {"HIPDNN_TYPE_POINTWISE_MODE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_BACKEND_DESCRIPTOR",                                  {"HIPDNN_TYPE_BACKEND_DESCRIPTOR",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_GENSTATS_MODE",                                       {"HIPDNN_TYPE_GENSTATS_MODE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_BN_FINALIZE_STATS_MODE",                              {"HIPDNN_TYPE_BN_FINALIZE_STATS_MODE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_REDUCTION_OPERATOR_TYPE",                             {"HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_BEHAVIOR_NOTE",                                       {"HIPDNN_TYPE_BEHAVIOR_NOTE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_TENSOR_REORDERING_MODE",                              {"HIPDNN_TYPE_TENSOR_REORDERING_MODE",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_RESAMPLE_MODE",                                       {"HIPDNN_TYPE_RESAMPLE_MODE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_PADDING_MODE",                                        {"HIPDNN_TYPE_PADDING_MODE",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_INT32",                                               {"HIPDNN_TYPE_INT32",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_CHAR",                                                {"HIPDNN_TYPE_CHAR",                                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_SIGNAL_MODE",                                         {"HIPDNN_TYPE_SIGNAL_MODE",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_FRACTION",                                            {"HIPDNN_TYPE_FRACTION",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_NORM_MODE",                                           {"HIPDNN_TYPE_NORM_MODE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_NORM_FWD_PHASE",                                      {"HIPDNN_TYPE_NORM_FWD_PHASE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TYPE_RNG_DISTRIBUTION",                                    {"HIPDNN_TYPE_RNG_DISTRIBUTION",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendDescriptorType_t",                                   {"hipdnnBackendDescriptorType_t",                                   "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_POINTWISE_DESCRIPTOR",                             {"HIPDNN_BACKEND_POINTWISE_DESCRIPTOR",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR",                           {"HIPDNN_BACKEND_CONVOLUTION_DESCRIPTOR",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINE_DESCRIPTOR",                                {"HIPDNN_BACKEND_ENGINE_DESCRIPTOR",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINECFG_DESCRIPTOR",                             {"HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",                            {"HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR",                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",                        {"HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",                     {"HIPDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",                           {"HIPDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR",                             {"HIPDNN_BACKEND_KNOB_INFO_DESCRIPTOR",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",                           {"HIPDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR",                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",         {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR",         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR", {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR", "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",   {"HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR",   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",                        {"HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",                          {"HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_TENSOR_DESCRIPTOR",                                {"HIPDNN_BACKEND_TENSOR_DESCRIPTOR",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_MATMUL_DESCRIPTOR",                                {"HIPDNN_BACKEND_MATMUL_DESCRIPTOR",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",                      {"HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",      {"HIPDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR",      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_REDUCTION_DESCRIPTOR",                             {"HIPDNN_BACKEND_REDUCTION_DESCRIPTOR",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",                   {"HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR",                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR",              {"HIPDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_RESAMPLE_DESCRIPTOR",                              {"HIPDNN_BACKEND_RESAMPLE_DESCRIPTOR",                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR",                {"HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR",                {"HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR",                      {"HIPDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR",                      {"HIPDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR",                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR",                {"HIPDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR",                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR",               {"HIPDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR",                     {"HIPDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR",                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_RNG_DESCRIPTOR",                                   {"HIPDNN_BACKEND_RNG_DESCRIPTOR",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR",                         {"HIPDNN_BACKEND_OPERATION_RNG_DESCRIPTOR",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendNumericalNote_t",                                    {"hipdnnBackendNumericalNote_t",                                    "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_TENSOR_CORE",                               {"HIPDNN_NUMERICAL_NOTE_TENSOR_CORE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",                       {"HIPDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",               {"HIPDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION",               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_FFT",                                       {"HIPDNN_NUMERICAL_NOTE_FFT",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC",                          {"HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD",                                  {"HIPDNN_NUMERICAL_NOTE_WINOGRAD",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4",                         {"HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6",                         {"HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13",                       {"HIPDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13",                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NUMERICAL_NOTE_TYPE_COUNT",                                {"HIPDNN_NUMERICAL_NOTE_TYPE_COUNT",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendLayoutType_t",                                       {"hipdnnBackendLayoutType_t",                                       "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NCHW",                               {"HIPDNN_LAYOUT_TYPE_PREFERRED_NCHW",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_NHWC",                               {"HIPDNN_LAYOUT_TYPE_PREFERRED_NHWC",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",                             {"HIPDNN_LAYOUT_TYPE_PREFERRED_PAD4CK",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",                             {"HIPDNN_LAYOUT_TYPE_PREFERRED_PAD8CK",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYOUT_TYPE_COUNT",                                        {"HIPDNN_LAYOUT_TYPE_COUNT",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendKnobType_t",                                         {"hipdnnBackendKnobType_t",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K",                                        {"HIPDNN_KNOB_TYPE_SPLIT_K",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SWIZZLE",                                        {"HIPDNN_KNOB_TYPE_SWIZZLE",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_SIZE",                                      {"HIPDNN_KNOB_TYPE_TILE_SIZE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_USE_TEX",                                        {"HIPDNN_KNOB_TYPE_USE_TEX",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_EDGE",                                           {"HIPDNN_KNOB_TYPE_EDGE",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_KBLOCK",                                         {"HIPDNN_KNOB_TYPE_KBLOCK",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGA",                                           {"HIPDNN_KNOB_TYPE_LDGA",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGB",                                           {"HIPDNN_KNOB_TYPE_LDGB",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_CHUNK_K",                                        {"HIPDNN_KNOB_TYPE_CHUNK_K",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_H",                                        {"HIPDNN_KNOB_TYPE_SPLIT_H",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_WINO_TILE",                                      {"HIPDNN_KNOB_TYPE_WINO_TILE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_MULTIPLY",                                       {"HIPDNN_KNOB_TYPE_MULTIPLY",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K_BUF",                                    {"HIPDNN_KNOB_TYPE_SPLIT_K_BUF",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILEK",                                          {"HIPDNN_KNOB_TYPE_TILEK",                                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_STAGES",                                         {"HIPDNN_KNOB_TYPE_STAGES",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_REDUCTION_MODE",                                 {"HIPDNN_KNOB_TYPE_REDUCTION_MODE",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",                               {"HIPDNN_KNOB_TYPE_CTA_SPLIT_K_MODE",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_K_SLC",                                    {"HIPDNN_KNOB_TYPE_SPLIT_K_SLC",                                    "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_IDX_MODE",                                       {"HIPDNN_KNOB_TYPE_IDX_MODE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SLICED",                                         {"HIPDNN_KNOB_TYPE_SLICED",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_RS",                                       {"HIPDNN_KNOB_TYPE_SPLIT_RS",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SINGLEBUFFER",                                   {"HIPDNN_KNOB_TYPE_SINGLEBUFFER",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LDGC",                                           {"HIPDNN_KNOB_TYPE_LDGC",                                           "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPECFILT",                                       {"HIPDNN_KNOB_TYPE_SPECFILT",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_KERNEL_CFG",                                     {"HIPDNN_KNOB_TYPE_KERNEL_CFG",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_WORKSPACE",                                      {"HIPDNN_KNOB_TYPE_WORKSPACE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_CGA",                                       {"HIPDNN_KNOB_TYPE_TILE_CGA",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_CGA_M",                                     {"HIPDNN_KNOB_TYPE_TILE_CGA_M",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_CGA_N",                                     {"HIPDNN_KNOB_TYPE_TILE_CGA_N",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_BLOCK_SIZE",                                     {"HIPDNN_KNOB_TYPE_BLOCK_SIZE",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_OCCUPANCY",                                      {"HIPDNN_KNOB_TYPE_OCCUPANCY",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD",                          {"HIPDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK",                                {"HIPDNN_KNOB_TYPE_NUM_C_PER_BLOCK",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_SPLIT_COLS",                                     {"HIPDNN_KNOB_TYPE_SPLIT_COLS",                                     "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_ROWS",                                      {"HIPDNN_KNOB_TYPE_TILE_ROWS",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_TILE_COLS",                                      {"HIPDNN_KNOB_TYPE_TILE_COLS",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_LOAD_SIZE",                                      {"HIPDNN_KNOB_TYPE_LOAD_SIZE",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_KNOB_TYPE_COUNTS",                                         {"HIPDNN_KNOB_TYPE_COUNTS",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendHeurMode_t",                                         {"hipdnnBackendHeurMode_t",                                         "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_INSTANT",                                        {"HIPDNN_HEUR_MODE_INSTANT",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_B",                                              {"HIPDNN_HEUR_MODE_B",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_FALLBACK",                                       {"HIPDNN_HEUR_MODE_FALLBACK",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_HEUR_MODE_A",                                              {"HIPDNN_HEUR_MODE_A",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_HEUR_MODES_COUNT",                                         {"HIPDNN_HEUR_MODES_COUNT",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnNormMode_t",                                                {"hipdnnNormMode_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_PER_ACTIVATION",                                      {"HIPDNN_NORM_PER_ACTIVATION",                                      "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_PER_CHANNEL",                                         {"HIPDNN_NORM_PER_CHANNEL",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnNormAlgo_t",                                                {"hipdnnNormAlgo_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_ALGO_STANDARD",                                       {"HIPDNN_NORM_ALGO_STANDARD",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_ALGO_PERSIST",                                        {"HIPDNN_NORM_ALGO_PERSIST",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnNormOps_t",                                                 {"hipdnnNormOps_t",                                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM",                                            {"HIPDNN_NORM_OPS_NORM",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM_ACTIVATION",                                 {"HIPDNN_NORM_OPS_NORM_ACTIVATION",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_OPS_NORM_ADD_ACTIVATION",                             {"HIPDNN_NORM_OPS_NORM_ADD_ACTIVATION",                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBnFinalizeStatsMode_t",                                     {"hipdnnBnFinalizeStatsMode_t",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BN_FINALIZE_STATISTICS_TRAINING",                          {"HIPDNN_BN_FINALIZE_STATISTICS_TRAINING",                          "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         {"HIPDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendBehaviorNote_t",                                     {"hipdnnBackendBehaviorNote_t",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION",                        {"HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION",                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 0
  {"CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER",            {"HIPDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER",            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 1
  {"CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER",              {"HIPDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER",              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},  // 2
  {"CUDNN_BEHAVIOR_NOTE_TYPE_COUNT",                                 {"HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnResampleMode_t",                                            {"hipdnnResampleMode_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_NEAREST",                                         {"HIPDNN_RESAMPLE_NEAREST",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_BILINEAR",                                        {"HIPDNN_RESAMPLE_BILINEAR",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_AVGPOOL",                                         {"HIPDNN_RESAMPLE_AVGPOOL",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING",                         {"HIPDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING",                         {"HIPDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING",                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RESAMPLE_MAXPOOL",                                         {"HIPDNN_RESAMPLE_MAXPOOL",                                         "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnSignalMode_t",                                              {"hipdnnSignalMode_t",                                              "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SIGNAL_SET",                                               {"HIPDNN_SIGNAL_SET",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_SIGNAL_WAIT",                                              {"HIPDNN_SIGNAL_WAIT",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendTensorReordering_t",                                 {"hipdnnBackendTensorReordering_t",                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TENSOR_REORDERING_NONE",                                   {"HIPDNN_TENSOR_REORDERING_NONE",                                   "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TENSOR_REORDERING_INT8x32",                                {"HIPDNN_TENSOR_REORDERING_INT8x32",                                "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_TENSOR_REORDERING_F16x16",                                 {"HIPDNN_TENSOR_REORDERING_F16x16",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnPaddingMode_t",                                             {"hipdnnPaddingMode_t",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_ZERO_PAD",                                                 {"HIPDNN_ZERO_PAD",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NEG_INF_PAD",                                              {"HIPDNN_NEG_INF_PAD",                                              "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_EDGE_VAL_PAD",                                             {"HIPDNN_EDGE_VAL_PAD",                                             "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendNormMode_t",                                         {"hipdnnBackendNormMode_t",                                         "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_LAYER_NORM",                                               {"HIPDNN_LAYER_NORM",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_INSTANCE_NORM",                                            {"HIPDNN_INSTANCE_NORM",                                            "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_BATCH_NORM",                                               {"HIPDNN_BATCH_NORM",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_GROUP_NORM",                                               {"HIPDNN_GROUP_NORM",                                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RMS_NORM",                                                 {"HIPDNN_RMS_NORM",                                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendNormFwdPhase_t",                                     {"hipdnnBackendNormFwdPhase_t",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_FWD_INFERENCE",                                       {"HIPDNN_NORM_FWD_INFERENCE",                                       "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_NORM_FWD_TRAINING",                                        {"HIPDNN_NORM_FWD_TRAINING",                                        "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"cudnnRngDistribution_t",                                         {"hipdnnRngDistribution_t",                                         "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNG_DISTRIBUTION_BERNOULLI",                               {"HIPDNN_RNG_DISTRIBUTION_BERNOULLI",                               "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNG_DISTRIBUTION_UNIFORM",                                 {"HIPDNN_RNG_DISTRIBUTION_UNIFORM",                                 "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},
  {"CUDNN_RNG_DISTRIBUTION_NORMAL",                                  {"HIPDNN_RNG_DISTRIBUTION_NORMAL",                                  "",                                                                CONV_NUMERIC_LITERAL, API_DNN, 1, UNSUPPORTED}},

  // cuDNN types
  {"cudnnContext",                                                   {"hipdnnContext",                                                   "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnHandle_t",                                                  {"hipdnnHandle_t",                                                  "miopenHandle_t",                                                  CONV_TYPE, API_DNN, 1}},
  {"cudnnTensorStruct",                                              {"hipdnnTensorStruct",                                              "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnTensorDescriptor_t",                                        {"hipdnnTensorDescriptor_t",                                        "miopenTensorDescriptor_t",                                        CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionStruct",                                         {"hipdnnConvolutionStruct",                                         "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnConvolutionDescriptor_t",                                   {"hipdnnConvolutionDescriptor_t",                                   "miopenConvolutionDescriptor_t",                                   CONV_TYPE, API_DNN, 1}},
  {"cudnnPoolingStruct",                                             {"hipdnnPoolingStruct",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnPoolingDescriptor_t",                                       {"hipdnnPoolingDescriptor_t",                                       "miopenPoolingDescriptor_t",                                       CONV_TYPE, API_DNN, 1}},
  {"cudnnFilterStruct",                                              {"hipdnnFilterStruct",                                              "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  // NOTE: both cudnnFilterDescriptor_t and cudnnTensorDescriptor_t are mapped to miopenTensorDescriptor_t
  {"cudnnFilterDescriptor_t",                                        {"hipdnnFilterDescriptor_t",                                        "miopenTensorDescriptor_t",                                        CONV_TYPE, API_DNN, 1}},
  {"cudnnLRNStruct",                                                 {"hipdnnLRNStruct",                                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnLRNDescriptor_t",                                           {"hipdnnLRNDescriptor_t",                                           "miopenLRNDescriptor_t",                                           CONV_TYPE, API_DNN, 1}},
  {"cudnnActivationStruct",                                          {"hipdnnActivationStruct",                                          "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnActivationDescriptor_t",                                    {"hipdnnActivationDescriptor_t",                                    "miopenActivationDescriptor_t",                                    CONV_TYPE, API_DNN, 1}},
  {"cudnnSpatialTransformerStruct",                                  {"hipdnnSpatialTransformerStruct",                                  "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnSpatialTransformerDescriptor_t",                            {"hipdnnSpatialTransformerDescriptor_t",                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnOpTensorStruct",                                            {"hipdnnOpTensorStruct",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnOpTensorDescriptor_t",                                      {"hipdnnOpTensorDescriptor_t",                                      "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"cudnnReduceTensorStruct",                                        {"hipdnnReduceTensorStruct",                                        "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnReduceTensorDescriptor_t",                                  {"hipdnnReduceTensorDescriptor_t",                                  "miopenReduceTensorDescriptor_t",                                  CONV_TYPE, API_DNN, 1}},
  {"cudnnCTCLossStruct",                                             {"hipdnnCTCLossStruct",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnCTCLossDescriptor_t",                                       {"hipdnnCTCLossDescriptor_t",                                       "miopenCTCLossDescriptor_t",                                       CONV_TYPE, API_DNN, 1, HIP_UNSUPPORTED}},
  {"cudnnTensorTransformStruct",                                     {"hipdnnTensorTransformStruct",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnTensorTransformDescriptor_t",                               {"hipdnnTensorTransformDescriptor_t",                               "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnConvolutionFwdAlgoPerf_t",                                  {"hipdnnConvolutionFwdAlgoPerf_t",                                  "miopenConvAlgoPerf_t",                                            CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionFwdAlgoPerfStruct",                              {"hipdnnConvolutionFwdAlgoPerf_t",                                  "miopenConvAlgoPerf_t",                                            CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionBwdFilterAlgoPerf_t",                            {"hipdnnConvolutionBwdFilterAlgoPerf_t",                            "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"cudnnConvolutionBwdFilterAlgoPerfStruct",                        {"hipdnnConvolutionBwdFilterAlgoPerf_t",                            "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"cudnnConvolutionBwdDataAlgoPerf_t",                              {"hipdnnConvolutionBwdDataAlgoPerf_t",                              "miopenConvAlgoPerf_t",                                            CONV_TYPE, API_DNN, 1}},
  {"cudnnConvolutionBwdDataAlgoPerfStruct",                          {"hipdnnConvolutionBwdDataAlgoPerf_t",                              "miopenConvAlgoPerf_t",                                            CONV_TYPE, API_DNN, 1}},
  {"cudnnDropoutStruct",                                             {"hipdnnDropoutStruct",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnDropoutDescriptor_t",                                       {"hipdnnDropoutDescriptor_t",                                       "miopenDropoutDescriptor_t",                                       CONV_TYPE, API_DNN, 1}},
  {"cudnnAlgorithmStruct",                                           {"hipdnnAlgorithmStruct",                                           "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAlgorithmDescriptor_t",                                     {"hipdnnAlgorithmDescriptor_t",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAlgorithmPerformanceStruct",                                {"hipdnnAlgorithmPerformanceStruct",                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAlgorithmPerformance_t",                                    {"hipdnnAlgorithmPerformance_t",                                    "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnRNNStruct",                                                 {"hipdnnRNNStruct",                                                 "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnRNNDescriptor_t",                                           {"hipdnnRNNDescriptor_t",                                           "miopenRNNDescriptor_t",                                           CONV_TYPE, API_DNN, 1}},
  {"cudnnPersistentRNNPlan",                                         {"hipdnnPersistentRNNPlan",                                         "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnPersistentRNNPlan_t",                                       {"hipdnnPersistentRNNPlan_t",                                       "",                                                                CONV_TYPE, API_DNN, 1, ROC_UNSUPPORTED}},
  {"cudnnAlgorithm_t",                                               {"hipdnnAlgorithm_t",                                               "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAlgorithmUnionStruct",                                      {"hipdnnAlgorithm_t",                                               "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnDebug_t",                                                   {"hipdnnDebug_t",                                                   "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnDebugStruct",                                               {"hipdnnDebug_t",                                                   "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnCallback_t",                                                {"hipdnnCallback_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnRNNDataStruct",                                             {"hipdnnRNNDataStruct",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnRNNDataDescriptor_t",                                       {"hipdnnRNNDataDescriptor_t",                                       "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnSeqDataStruct",                                             {"hipdnnSeqDataStruct",                                             "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnSeqDataDescriptor_t",                                       {"hipdnnSeqDataDescriptor_t",                                       "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAttnStruct",                                                {"hipdnnAttnStruct",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnAttnDescriptor_t",                                          {"hipdnnAttnDescriptor_t",                                          "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsConstParamStruct",                                  {"hipdnnFusedOpsConstParamStruct",                                  "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsConstParamPack_t",                                  {"hipdnnFusedOpsConstParamPack_t",                                  "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamStruct",                                {"hipdnnFusedOpsVariantParamStruct",                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsVariantParamPack_t",                                {"hipdnnFusedOpsVariantParamPack_t",                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsPlanStruct",                                        {"hipdnnFusedOpsPlanStruct",                                        "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFusedOpsPlan_t",                                            {"hipdnnFusedOpsPlan_t",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnBackendDescriptor_t",                                       {"hipdnnBackendDescriptor_t",                                       "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"libraryPropertyType",                                            {"hipdnnLibraryPropertyType",                                       "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"libraryPropertyType_t",                                          {"hipdnnLibraryPropertyType_t",                                     "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFractionStruct",                                            {"hipdnnFractionStruct",                                            "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
  {"cudnnFraction_t",                                                {"hipdnnFraction_t",                                                "",                                                                CONV_TYPE, API_DNN, 1, UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_TYPE_NAME_VER_MAP {
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
  {"cudnnConvolutionBwdFilterAlgoPerf_t",                            {CUDNN_30,  CUDA_0,   CUDA_0  }},
  {"cudnnContext",                                                   {CUDNN_10,  CUDA_0,   CUDA_0  }},
  {"cudnnHandle_t",                                                  {CUDNN_10,  CUDA_0,   CUDA_0  }},
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
  {"CUDNN_ACTIVATION_SWISH",                                         {CUDNN_820, CUDA_0,   CUDA_0  }},
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
  {"cudnnCTCLossStruct",                                             {CUDNN_705, CUDA_0,   CUDA_0  }},
  {"cudnnCallback_t",                                                {CUDNN_713, CUDA_0,   CUDA_0  }},
  {"cudnnBnFinalizeStatsMode_t",                                     {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BN_FINALIZE_STATISTICS_TRAINING",                          {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"CUDNN_BN_FINALIZE_STATISTICS_INFERENCE",                         {CUDNN_810, CUDA_0,   CUDA_0  }},
  {"cudnnAlgorithmUnionStruct",                                      {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"cudnnDebugStruct",                                               {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdFilterAlgoPerfStruct",                        {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionFwdAlgoPerfStruct",                              {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"cudnnConvolutionBwdDataAlgoPerfStruct",                          {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE",                                {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC",                  {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC",                  {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC",                {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC",              {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC",                     {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC",                    {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",             {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC",              {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC",           {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC",            {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS",                    {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_BEHAVIOR_NOTE",                                       {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC",             {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"cudnnBackendBehaviorNote_t",                                     {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION",                        {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_BEHAVIOR_NOTE_TYPE_COUNT",                                 {CUDNN_820, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_BOOLEAN",                                             {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ADD_SQUARE",                                     {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_DIV",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_MOD",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_POW",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SUB",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ABS",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CEIL",                                           {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_COS",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_EXP",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_FLOOR",                                          {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_LOG",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_NEG",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_RSQRT",                                          {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_SIN",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_TAN",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_ERF",                                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_IDENTITY",                                       {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_GELU_APPROX_TANH_FWD",                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_GELU_APPROX_TANH_BWD",                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_EQ",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_NEQ",                                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_GT",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_GE",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_LT",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_CMP_LE",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_LOGICAL_AND",                                    {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_LOGICAL_OR",                                     {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_LOGICAL_NOT",                                    {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_GEN_INDEX",                                      {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_BINARY_SELECT",                                  {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"cudnnFractionStruct",                                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"cudnnFraction_t",                                                {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"cudnnResampleMode_t",                                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_NEAREST",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_BILINEAR",                                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_AVGPOOL",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_MAXPOOL",                                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"cudnnSignalMode_t",                                              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_SIGNAL_SET",                                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_SIGNAL_WAIT",                                              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_POINTWISE_AXIS",                                      {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION",                  {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_POINTWISE_TDESC",                           {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_REORDERING_MODE",                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_MODE",                                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_COMP_TYPE",                                  {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS",                               {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_POST_PADDINGS",                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_PRE_PADDINGS",                               {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_STRIDES",                                    {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_WINDOW_DIMS",                                {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION",                            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RESAMPLE_PADDING_MODE",                               {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC",                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC",                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC",                      {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA",                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC",                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC",                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC",                      {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA",                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONCAT_AXIS",                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS",                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX",                      {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC",                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_SIGNAL_MODE",                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC",                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_SIGNAL_VALUE",                              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_SIGNAL_XDESC",                              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_SIGNAL_YDESC",                              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_MODE",                             {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_PHASE",                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_XDESC",                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC",                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC",                {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC",                       {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC",                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC",                     {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC",              {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC",          {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC",           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC",         {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC",          {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_YDESC",                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_FWD_PEER_STAT_DESCS",                  {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_MODE",                             {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_XDESC",                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC",                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC",                {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC",                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC",                       {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC",                     {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC",                      {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC",                       {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC",                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_NORM_BWD_PEER_STAT_DESCS",                  {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_TENSOR_REORDERING_MODE",                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_RESAMPLE_MODE",                                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_PADDING_MODE",                                        {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_INT32",                                               {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_CHAR",                                                {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_SIGNAL_MODE",                                         {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_FRACTION",                                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_NORM_MODE",                                           {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_NORM_FWD_PHASE",                                      {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_RESAMPLE_DESCRIPTOR",                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR",                {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR",                {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR",                      {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR",                      {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR",                {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR",               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6",                         {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13",                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER",            {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER",              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_WORKSPACE",                                      {CUDNN_840, CUDA_0,   CUDA_0  }},
  {"CUDNN_HEUR_MODE_FALLBACK",                                       {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_HEUR_MODE_A",                                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"cudnnBackendTensorReordering_t",                                 {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_REORDERING_NONE",                                   {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_REORDERING_INT8x32",                                {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"cudnnPaddingMode_t",                                             {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_ZERO_PAD",                                                 {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_NEG_INF_PAD",                                              {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"CUDNN_EDGE_VAL_PAD",                                             {CUDNN_830, CUDA_0,   CUDA_0  }},
  {"cudnnBackendNormMode_t",                                         {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_LAYER_NORM",                                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_INSTANCE_NORM",                                            {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_BATCH_NORM",                                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_GROUP_NORM",                                               {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"cudnnBackendNormFwdPhase_t",                                     {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_FWD_INFERENCE",                                       {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_NORM_FWD_TRAINING",                                        {CUDNN_850, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING",                         {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING",                         {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_CGA",                                       {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_CGA_M",                                     {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_CGA_N",                                     {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_FP8_E4M3",                                            {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_FP8_E5M2",                                            {CUDNN_860, CUDA_0,   CUDA_0  }},
  {"CUDNN_DATA_FAST_FLOAT_FOR_FP8",                                  {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"cudnnRngDistribution_t",                                         {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNG_DISTRIBUTION_BERNOULLI",                               {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNG_DISTRIBUTION_UNIFORM",                                 {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_RNG_DISTRIBUTION_NORMAL",                                  {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC",               {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC",               {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC",               {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC",                        {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC",                        {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESHAPE_XDESC",                             {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RESHAPE_YDESC",                             {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_DISTRIBUTION",                                    {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_NORMAL_DIST_MEAN",                                {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION",                  {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM",                            {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM",                            {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY",                      {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RNG_YDESC",                                 {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RNG_SEED",                                  {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RNG_DESC",                                  {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_TYPE_RNG_DISTRIBUTION",                                    {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR",                     {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_RNG_DESCRIPTOR",                                   {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR",                         {CUDNN_870, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC",                           {CUDNN_880, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_BLOCK_SIZE",                                     {CUDNN_880, CUDA_0,   CUDA_0  }},
  {"CUDNN_TENSOR_REORDERING_F16x16",                                 {CUDNN_880, CUDA_0,   CUDA_0  }},
  {"CUDNN_POINTWISE_RECIPROCAL",                                     {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC",                           {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_MATMUL_PADDING_VALUE",                                {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_OCCUPANCY",                                      {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD",                          {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK",                                {CUDNN_890, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET",                          {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_ATTR_ENGINE_SM_COUNT_TARGET",                              {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_SPLIT_COLS",                                     {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_ROWS",                                      {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_TILE_COLS",                                      {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_KNOB_TYPE_LOAD_SIZE",                                      {CUDNN_895, CUDA_0,   CUDA_0  }},
  {"CUDNN_RMS_NORM",                                                 {CUDNN_896, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_DNN_TYPE_NAME_VER_MAP {
};
