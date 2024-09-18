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

using SEC = blas::BLAS_API_SECTIONS;

// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_TYPE_NAME_MAP {
  // Blas operations
  {"cublasOperation_t",                                              {"hipblasOperation_t",                                                "rocblas_operation",                                        CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_OP_N",                                                    {"HIPBLAS_OP_N",                                                      "rocblas_operation_none",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_OP_T",                                                    {"HIPBLAS_OP_T",                                                      "rocblas_operation_transpose",                              CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_OP_C",                                                    {"HIPBLAS_OP_C",                                                      "rocblas_operation_conjugate_transpose",                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_OP_HERMITAN",                                             {"HIPBLAS_OP_C",                                                      "rocblas_operation_conjugate_transpose",                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_OP_CONJG",                                                {"HIPBLAS_OP_CONJG",                                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},

  // Blas statuses
  {"cublasStatus",                                                   {"hipblasStatus_t",                                                   "rocblas_status",                                           CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"cublasStatus_t",                                                 {"hipblasStatus_t",                                                   "rocblas_status",                                           CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_SUCCESS",                                          {"HIPBLAS_STATUS_SUCCESS",                                            "rocblas_status_success",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_NOT_INITIALIZED",                                  {"HIPBLAS_STATUS_NOT_INITIALIZED",                                    "rocblas_status_invalid_handle",                            CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_ALLOC_FAILED",                                     {"HIPBLAS_STATUS_ALLOC_FAILED",                                       "rocblas_status_not_implemented",                           CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_INVALID_VALUE",                                    {"HIPBLAS_STATUS_INVALID_VALUE",                                      "rocblas_status_invalid_value",                             CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_MAPPING_ERROR",                                    {"HIPBLAS_STATUS_MAPPING_ERROR",                                      "rocblas_status_invalid_size",                              CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_EXECUTION_FAILED",                                 {"HIPBLAS_STATUS_EXECUTION_FAILED",                                   "rocblas_status_memory_error",                              CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_INTERNAL_ERROR",                                   {"HIPBLAS_STATUS_INTERNAL_ERROR",                                     "rocblas_status_internal_error",                            CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_NOT_SUPPORTED",                                    {"HIPBLAS_STATUS_NOT_SUPPORTED",                                      "rocblas_status_perf_degraded",                             CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_ARCH_MISMATCH",                                    {"HIPBLAS_STATUS_ARCH_MISMATCH",                                      "rocblas_status_arch_mismatch",                             CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_STATUS_LICENSE_ERROR",                                    {"HIPBLAS_STATUS_UNKNOWN",                                            "rocblas_status_not_implemented",                           CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}},

  // Blas Fill Modes
  {"cublasFillMode_t",                                               {"hipblasFillMode_t",                                                 "rocblas_fill",                                             CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_FILL_MODE_LOWER",                                         {"HIPBLAS_FILL_MODE_LOWER",                                           "rocblas_fill_lower",                                       CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_FILL_MODE_UPPER",                                         {"HIPBLAS_FILL_MODE_UPPER",                                           "rocblas_fill_upper",                                       CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_FILL_MODE_FULL",                                          {"HIPBLAS_FILL_MODE_FULL",                                            "rocblas_fill_full",                                        CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},

  // Blas Diag Types
  {"cublasDiagType_t",                                               {"hipblasDiagType_t",                                                 "rocblas_diagonal",                                         CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_DIAG_NON_UNIT",                                           {"HIPBLAS_DIAG_NON_UNIT",                                             "rocblas_diagonal_non_unit",                                CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_DIAG_UNIT",                                               {"HIPBLAS_DIAG_UNIT",                                                 "rocblas_diagonal_unit",                                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},

  // Blas Side Modes
  {"cublasSideMode_t",                                               {"hipblasSideMode_t",                                                 "rocblas_side",                                             CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_SIDE_LEFT",                                               {"HIPBLAS_SIDE_LEFT",                                                 "rocblas_side_left",                                        CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_SIDE_RIGHT",                                              {"HIPBLAS_SIDE_RIGHT",                                                "rocblas_side_right",                                       CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},

  // Blas Pointer Modes
  {"cublasPointerMode_t",                                            {"hipblasPointerMode_t",                                              "rocblas_pointer_mode",                                     CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_POINTER_MODE_HOST",                                       {"HIPBLAS_POINTER_MODE_HOST",                                         "rocblas_pointer_mode_host",                                CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_POINTER_MODE_DEVICE",                                     {"HIPBLAS_POINTER_MODE_DEVICE",                                       "rocblas_pointer_mode_device",                              CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},

  // Blas Atomics Modes
  {"cublasAtomicsMode_t",                                            {"hipblasAtomicsMode_t",                                              "rocblas_atomics_mode",                                     CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_ATOMICS_NOT_ALLOWED",                                     {"HIPBLAS_ATOMICS_NOT_ALLOWED",                                       "rocblas_atomics_not_allowed",                              CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_ATOMICS_ALLOWED",                                         {"HIPBLAS_ATOMICS_ALLOWED",                                           "rocblas_atomics_allowed",                                  CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},

  // Blas Math mode/tensor operation
  {"cublasMath_t",                                                   {"hipblasMath_t",                                                     "rocblas_math_mode",                                        CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_DEFAULT_MATH",                                            {"HIPBLAS_DEFAULT_MATH",                                              "rocblas_default_math",                                     CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}}, // 0
  {"CUBLAS_TENSOR_OP_MATH",                                          {"HIPBLAS_TENSOR_OP_MATH",                                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED | CUDA_DEPRECATED}}, // 1
  {"CUBLAS_PEDANTIC_MATH",                                           {"HIPBLAS_PEDANTIC_MATH",                                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 2
  {"CUBLAS_TF32_TENSOR_OP_MATH",                                     {"HIPBLAS_TF32_TENSOR_OP_MATH",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 3
  {"CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",               {"HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 16

  // Blass different GEMM algorithms
  {"cublasGemmAlgo_t",                                               {"hipblasGemmAlgo_t",                                                 "rocblas_gemm_algo",                                        CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_GEMM_DFALT",                                              {"HIPBLAS_GEMM_DEFAULT",                                              "rocblas_gemm_algo_standard",                               CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},  //  -1 // 160 // 0b0000000000
  {"CUBLAS_GEMM_DEFAULT",                                            {"HIPBLAS_GEMM_DEFAULT",                                              "rocblas_gemm_algo_standard",                               CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}},  //  -1 // 160 // 0b0000000000
  {"CUBLAS_GEMM_ALGO0",                                              {"HIPBLAS_GEMM_ALGO0",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   0
  {"CUBLAS_GEMM_ALGO1",                                              {"HIPBLAS_GEMM_ALGO1",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   1
  {"CUBLAS_GEMM_ALGO2",                                              {"HIPBLAS_GEMM_ALGO2",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   2
  {"CUBLAS_GEMM_ALGO3",                                              {"HIPBLAS_GEMM_ALGO3",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   3
  {"CUBLAS_GEMM_ALGO4",                                              {"HIPBLAS_GEMM_ALGO4",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   4
  {"CUBLAS_GEMM_ALGO5",                                              {"HIPBLAS_GEMM_ALGO5",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   5
  {"CUBLAS_GEMM_ALGO6",                                              {"HIPBLAS_GEMM_ALGO6",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   6
  {"CUBLAS_GEMM_ALGO7",                                              {"HIPBLAS_GEMM_ALGO7",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   7
  {"CUBLAS_GEMM_ALGO8",                                              {"HIPBLAS_GEMM_ALGO8",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   8
  {"CUBLAS_GEMM_ALGO9",                                              {"HIPBLAS_GEMM_ALGO9",                                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //   9
  {"CUBLAS_GEMM_ALGO10",                                             {"HIPBLAS_GEMM_ALGO10",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  10
  {"CUBLAS_GEMM_ALGO11",                                             {"HIPBLAS_GEMM_ALGO11",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  11
  {"CUBLAS_GEMM_ALGO12",                                             {"HIPBLAS_GEMM_ALGO12",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  12
  {"CUBLAS_GEMM_ALGO13",                                             {"HIPBLAS_GEMM_ALGO13",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  13
  {"CUBLAS_GEMM_ALGO14",                                             {"HIPBLAS_GEMM_ALGO14",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  14
  {"CUBLAS_GEMM_ALGO15",                                             {"HIPBLAS_GEMM_ALGO15",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  15
  {"CUBLAS_GEMM_ALGO16",                                             {"HIPBLAS_GEMM_ALGO16",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  16
  {"CUBLAS_GEMM_ALGO17",                                             {"HIPBLAS_GEMM_ALGO17",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  17
  {"CUBLAS_GEMM_ALGO18",                                             {"HIPBLAS_GEMM_ALGO18",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  18
  {"CUBLAS_GEMM_ALGO19",                                             {"HIPBLAS_GEMM_ALGO19",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  19
  {"CUBLAS_GEMM_ALGO20",                                             {"HIPBLAS_GEMM_ALGO20",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  20
  {"CUBLAS_GEMM_ALGO21",                                             {"HIPBLAS_GEMM_ALGO21",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  21
  {"CUBLAS_GEMM_ALGO22",                                             {"HIPBLAS_GEMM_ALGO22",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  22
  {"CUBLAS_GEMM_ALGO23",                                             {"HIPBLAS_GEMM_ALGO23",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  23
  {"CUBLAS_GEMM_DEFAULT_TENSOR_OP",                                  {"HIPBLAS_GEMM_DEFAULT_TENSOR_OP",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  99
  {"CUBLAS_GEMM_DFALT_TENSOR_OP",                                    {"HIPBLAS_GEMM_DFALT_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  //  99
  {"CUBLAS_GEMM_ALGO0_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO0_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 100
  {"CUBLAS_GEMM_ALGO1_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO1_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 101
  {"CUBLAS_GEMM_ALGO2_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO2_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 102
  {"CUBLAS_GEMM_ALGO3_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO3_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 103
  {"CUBLAS_GEMM_ALGO4_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO4_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 104
  {"CUBLAS_GEMM_ALGO5_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO5_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 105
  {"CUBLAS_GEMM_ALGO6_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO6_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 106
  {"CUBLAS_GEMM_ALGO7_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO7_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 107
  {"CUBLAS_GEMM_ALGO8_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO8_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 108
  {"CUBLAS_GEMM_ALGO9_TENSOR_OP",                                    {"HIPBLAS_GEMM_ALGO9_TENSOR_OP",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 109
  {"CUBLAS_GEMM_ALGO10_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO10_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 110
  {"CUBLAS_GEMM_ALGO11_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO11_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 111
  {"CUBLAS_GEMM_ALGO12_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO12_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 112
  {"CUBLAS_GEMM_ALGO13_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO13_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 113
  {"CUBLAS_GEMM_ALGO14_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO14_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 114
  {"CUBLAS_GEMM_ALGO15_TENSOR_OP",                                   {"HIPBLAS_GEMM_ALGO15_TENSOR_OP",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, UNSUPPORTED}},  // 115

  // CUDA Library Data Types

  {"cublasDataType_t",                                               {"hipDataType",                                                       "rocblas_datatype",                                         CONV_TYPE, API_BLAS, SEC::CUDA_DATA_TYPES}},
  {"cudaDataType_t",                                                 {"hipDataType",                                                       "rocblas_datatype_",                                        CONV_TYPE, API_BLAS, SEC::CUDA_DATA_TYPES}},
  {"cudaDataType",                                                   {"hipDataType",                                                       "rocblas_datatype",                                         CONV_TYPE, API_BLAS, SEC::CUDA_DATA_TYPES}},
  {"CUDA_R_16F",                                                     {"HIP_R_16F",                                                         "rocblas_datatype_f16_r",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  2 // 150
  {"CUDA_C_16F",                                                     {"HIP_C_16F",                                                         "rocblas_datatype_f16_c",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  6 // 153
  {"CUDA_R_32F",                                                     {"HIP_R_32F",                                                         "rocblas_datatype_f32_r",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  0 // 151
  {"CUDA_C_32F",                                                     {"HIP_C_32F",                                                         "rocblas_datatype_f32_c",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  4 // 154
  {"CUDA_R_64F",                                                     {"HIP_R_64F",                                                         "rocblas_datatype_f64_r",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  1 // 152
  {"CUDA_C_64F",                                                     {"HIP_C_64F",                                                         "rocblas_datatype_f64_c",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  5 // 155
  {"CUDA_R_8I",                                                      {"HIP_R_8I",                                                          "rocblas_datatype_i8_r",                                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  3 // 160
  {"CUDA_C_8I",                                                      {"HIP_C_8I",                                                          "rocblas_datatype_i8_c",                                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  7 // 164
  {"CUDA_R_8U",                                                      {"HIP_R_8U",                                                          "rocblas_datatype_u8_r",                                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  8 // 161
  {"CUDA_C_8U",                                                      {"HIP_C_8U",                                                          "rocblas_datatype_u8_c",                                    CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, //  9 // 165
  {"CUDA_R_32I",                                                     {"HIP_R_32I",                                                         "rocblas_datatype_i32_r",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 10 // 162
  {"CUDA_C_32I",                                                     {"HIP_C_32I",                                                         "rocblas_datatype_i32_c",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 11 // 166
  {"CUDA_R_32U",                                                     {"HIP_R_32U",                                                         "rocblas_datatype_u32_r",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 12 // 163
  {"CUDA_C_32U",                                                     {"HIP_C_32U",                                                         "rocblas_datatype_u32_c",                                   CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 13 // 167
  {"CUDA_R_16BF",                                                    {"HIP_R_16BF",                                                        "rocblas_datatype_bf16_r",                                  CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 14 // 168
  {"CUDA_C_16BF",                                                    {"HIP_C_16BF",                                                        "rocblas_datatype_bf16_c",                                  CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES}}, // 15 // 169
  {"CUDA_R_4I",                                                      {"HIPBLAS_R_4I",                                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 16
  {"CUDA_C_4I",                                                      {"HIPBLAS_C_4I",                                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 17
  {"CUDA_R_4U",                                                      {"HIPBLAS_R_4U",                                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 18
  {"CUDA_C_4U",                                                      {"HIPBLAS_C_4U",                                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 19
  {"CUDA_R_16I",                                                     {"HIPBLAS_R_16I",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 20
  {"CUDA_C_16I",                                                     {"HIPBLAS_C_16I",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 21
  {"CUDA_R_16U",                                                     {"HIPBLAS_R_16U",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 22
  {"CUDA_C_16U",                                                     {"HIPBLAS_C_16U",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 23
  {"CUDA_R_64I",                                                     {"HIPBLAS_R_64I",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 24
  {"CUDA_C_64I",                                                     {"HIPBLAS_C_64I",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 25
  {"CUDA_R_64U",                                                     {"HIPBLAS_R_64U",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 26
  {"CUDA_C_64U",                                                     {"HIPBLAS_C_64U",                                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 27
  {"CUDA_R_8F_E4M3",                                                 {"HIPBLAS_R_8F_E4M3",                                                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 28
  {"CUDA_R_8F_E5M2",                                                 {"HIPBLAS_R_8F_E5M2",                                                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::CUDA_DATA_TYPES, UNSUPPORTED}}, // 29

  // CUBLAS Data Types

  {"cublasHandle_t",                                                 {"hipblasHandle_t",                                                   "rocblas_handle",                                           CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  // TODO: dereferencing: typedef struct cublasContext *cublasHandle_t;
  {"cublasContext",                                                  {"hipblasContext",                                                    "_rocblas_handle",                                          CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES, HIP_UNSUPPORTED}},

  {"cublasComputeType_t",                                            {"hipblasComputeType_t",                                              "rocblas_computetype",                                      CONV_TYPE, API_BLAS, SEC::BLAS_DATA_TYPES}},
  {"CUBLAS_COMPUTE_16F",                                             {"HIPBLAS_COMPUTE_16F",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 64
  {"CUBLAS_COMPUTE_16F_PEDANTIC",                                    {"HIPBLAS_COMPUTE_16F_PEDANTIC",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 65
  {"CUBLAS_COMPUTE_32F",                                             {"HIPBLAS_COMPUTE_32F",                                               "rocblas_compute_type_f32",                                 CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES}}, // 68
  {"CUBLAS_COMPUTE_32F_PEDANTIC",                                    {"HIPBLAS_COMPUTE_32F_PEDANTIC",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 69
  {"CUBLAS_COMPUTE_32F_FAST_16F",                                    {"HIPBLAS_COMPUTE_32F_FAST_16F",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 74
  {"CUBLAS_COMPUTE_32F_FAST_16BF",                                   {"HIPBLAS_COMPUTE_32F_FAST_16BF",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 75
  {"CUBLAS_COMPUTE_32F_FAST_TF32",                                   {"HIPBLAS_COMPUTE_32F_FAST_TF32",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 77
  {"CUBLAS_COMPUTE_64F",                                             {"HIPBLAS_COMPUTE_64F",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 70
  {"CUBLAS_COMPUTE_64F_PEDANTIC",                                    {"HIPBLAS_COMPUTE_64F_PEDANTIC",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 71
  {"CUBLAS_COMPUTE_32I",                                             {"HIPBLAS_COMPUTE_32I",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 72
  {"CUBLAS_COMPUTE_32I_PEDANTIC",                                    {"HIPBLAS_COMPUTE_32I_PEDANTIC",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_DATA_TYPES, ROC_UNSUPPORTED}}, // 73

  // cuBLASLt Data Types

  {"cublasLtHandle_t",                                               {"hipblasLtHandle_t",                                                 "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  // TODO: dereferencing: typedef struct cublasLtContext *cublasLtHandle_t;
  {"cublasLtContext",                                                {"hipblasLtContext",                                                  "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  // NOTE: hipblasLtMatrixLayoutOpaque_t contains uint64_t data[4], whereas cublasLtMatrixLayoutOpaque_t contains uint64_t data[8]
  {"cublasLtMatrixLayoutOpaque_t",                                   {"hipblasLtMatrixLayoutOpaque_t",                                     "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  // NOTE: cublasLtMatrixLayoutStruct is the former name for cublasLtMatrixLayoutOpaque_t, that has been alive for 10.1.0 <= CUDA <= 10.2.0
  {"cublasLtMatrixLayoutStruct",                                     {"hipblasLtMatrixLayoutOpaque_t",                                     "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatrixLayout_t",                                         {"hipblasLtMatrixLayout_t",                                           "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  // NOTE: Aren't they compatible?
  {"cublasLtMatmulAlgo_t",                                           {"hipblasLtMatmulAlgo_t",                                             "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  // NOTE: Aren't they compatible?
  {"cublasLtMatmulDescOpaque_t",                                     {"hipblasLtMatmulDescOpaque_t",                                       "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatmulDesc_t",                                           {"hipblasLtMatmulDesc_t",                                             "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatrixTransformDescOpaque_t",                            {"hipblasLtMatrixTransformDescOpaque_t",                              "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatrixTransformDesc_t",                                  {"hipblasLtMatrixTransformDesc_t",                                    "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatmulPreferenceOpaque_t",                               {"hipblasLtMatmulPreferenceOpaque_t",                                 "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatmulPreference_t",                                     {"hipblasLtMatmulPreference_t",                                       "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtMatmulTile_t",                                           {"hipblasLtMatmulTile_t",                                             "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_UNDEFINED",                                 {"HIPBLASLT_MATMUL_TILE_UNDEFINED",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_8x8",                                       {"HIPBLASLT_MATMUL_TILE_8x8",                                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_8x16",                                      {"HIPBLASLT_MATMUL_TILE_8x16",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_16x8",                                      {"HIPBLASLT_MATMUL_TILE_16x8",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_8x32",                                      {"HIPBLASLT_MATMUL_TILE_8x32",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_16x16",                                     {"HIPBLASLT_MATMUL_TILE_16x16",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x8",                                      {"HIPBLASLT_MATMUL_TILE_32x8",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_8x64",                                      {"HIPBLASLT_MATMUL_TILE_8x64",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_16x32",                                     {"HIPBLASLT_MATMUL_TILE_16x32",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x16",                                     {"HIPBLASLT_MATMUL_TILE_32x16",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x8",                                      {"HIPBLASLT_MATMUL_TILE_64x8",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x32",                                     {"HIPBLASLT_MATMUL_TILE_32x32",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x64",                                     {"HIPBLASLT_MATMUL_TILE_32x64",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x32",                                     {"HIPBLASLT_MATMUL_TILE_64x32",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x128",                                    {"HIPBLASLT_MATMUL_TILE_32x128",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x64",                                     {"HIPBLASLT_MATMUL_TILE_64x64",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x32",                                    {"HIPBLASLT_MATMUL_TILE_128x32",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x128",                                    {"HIPBLASLT_MATMUL_TILE_64x128",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x64",                                    {"HIPBLASLT_MATMUL_TILE_128x64",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x256",                                    {"HIPBLASLT_MATMUL_TILE_64x256",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x128",                                   {"HIPBLASLT_MATMUL_TILE_128x128",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_256x64",                                    {"HIPBLASLT_MATMUL_TILE_256x64",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x512",                                    {"HIPBLASLT_MATMUL_TILE_64x512",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x256",                                   {"HIPBLASLT_MATMUL_TILE_128x256",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_256x128",                                   {"HIPBLASLT_MATMUL_TILE_256x128",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_512x64",                                    {"HIPBLASLT_MATMUL_TILE_512x64",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_64x96",                                     {"HIPBLASLT_MATMUL_TILE_64x96",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_96x64",                                     {"HIPBLASLT_MATMUL_TILE_96x64",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_96x128",                                    {"HIPBLASLT_MATMUL_TILE_96x128",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x160",                                   {"HIPBLASLT_MATMUL_TILE_128x160",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_160x128",                                   {"HIPBLASLT_MATMUL_TILE_160x128",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_192x128",                                   {"HIPBLASLT_MATMUL_TILE_192x128",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x192",                                   {"HIPBLASLT_MATMUL_TILE_128x192",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_128x96",                                    {"HIPBLASLT_MATMUL_TILE_128x96",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_32x256",                                    {"HIPBLASLT_MATMUL_TILE_32x256",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_256x32",                                    {"HIPBLASLT_MATMUL_TILE_256x32",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_TILE_END",                                       {"HIPBLASLT_MATMUL_TILE_END",                                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulStages_t",                                         {"hipblasLtMatmulStages_t",                                           "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_UNDEFINED",                               {"HIPBLASLT_MATMUL_STAGES_UNDEFINED",                                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x1",                                    {"HIPBLASLT_MATMUL_STAGES_16x1",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x2",                                    {"HIPBLASLT_MATMUL_STAGES_16x2",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x3",                                    {"HIPBLASLT_MATMUL_STAGES_16x3",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x4",                                    {"HIPBLASLT_MATMUL_STAGES_16x4",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x5",                                    {"HIPBLASLT_MATMUL_STAGES_16x5",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x6",                                    {"HIPBLASLT_MATMUL_STAGES_16x6",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x1",                                    {"HIPBLASLT_MATMUL_STAGES_32x1",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x2",                                    {"HIPBLASLT_MATMUL_STAGES_32x2",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x3",                                    {"HIPBLASLT_MATMUL_STAGES_32x3",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x4",                                    {"HIPBLASLT_MATMUL_STAGES_32x4",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x5",                                    {"HIPBLASLT_MATMUL_STAGES_32x5",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x6",                                    {"HIPBLASLT_MATMUL_STAGES_32x6",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x1",                                    {"HIPBLASLT_MATMUL_STAGES_64x1",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x2",                                    {"HIPBLASLT_MATMUL_STAGES_64x2",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x3",                                    {"HIPBLASLT_MATMUL_STAGES_64x3",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x4",                                    {"HIPBLASLT_MATMUL_STAGES_64x4",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x5",                                    {"HIPBLASLT_MATMUL_STAGES_64x5",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64x6",                                    {"HIPBLASLT_MATMUL_STAGES_64x6",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x1",                                   {"HIPBLASLT_MATMUL_STAGES_128x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x2",                                   {"HIPBLASLT_MATMUL_STAGES_128x2",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x3",                                   {"HIPBLASLT_MATMUL_STAGES_128x3",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x4",                                   {"HIPBLASLT_MATMUL_STAGES_128x4",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x5",                                   {"HIPBLASLT_MATMUL_STAGES_128x5",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128x6",                                   {"HIPBLASLT_MATMUL_STAGES_128x6",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32x10",                                   {"HIPBLASLT_MATMUL_STAGES_32x10",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_8x4",                                     {"HIPBLASLT_MATMUL_STAGES_8x4",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16x10",                                   {"HIPBLASLT_MATMUL_STAGES_16x10",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_8x5",                                     {"HIPBLASLT_MATMUL_STAGES_8x5",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_8x3",                                     {"HIPBLASLT_MATMUL_STAGES_8x3",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_8xAUTO",                                  {"HIPBLASLT_MATMUL_STAGES_8xAUTO",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_16xAUTO",                                 {"HIPBLASLT_MATMUL_STAGES_16xAUTO",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_32xAUTO",                                 {"HIPBLASLT_MATMUL_STAGES_32xAUTO",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_64xAUTO",                                 {"HIPBLASLT_MATMUL_STAGES_64xAUTO",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_128xAUTO",                                {"HIPBLASLT_MATMUL_STAGES_128xAUTO",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_STAGES_END",                                     {"HIPBLASLT_MATMUL_STAGES_END",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtClusterShape_t",                                         {"hipblasLtClusterShape_t",                                           "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_AUTO",                                    {"HIPBLASLT_CLUSTER_SHAPE_AUTO",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_4x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_4x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_4x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_4x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x4x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x4x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x4x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x4x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_4x4x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_4x4x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_8x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_8x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x8x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x8x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_8x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_8x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x8x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x8x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_16x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_16x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x16x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x16x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_3x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_3x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_5x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_5x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_6x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_6x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_7x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_7x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_9x1x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_9x1x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_10x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_10x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_11x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_11x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_12x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_12x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_13x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_13x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_14x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_14x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_15x1x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_15x1x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_3x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_3x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_5x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_5x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_6x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_6x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_7x2x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_7x2x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x3x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x3x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x3x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x3x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_3x3x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_3x3x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_4x3x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_4x3x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_5x3x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_5x3x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_3x4x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_3x4x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x5x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x5x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x5x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x5x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_3x5x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_3x5x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x6x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x6x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x6x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x6x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x7x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x7x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_2x7x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_2x7x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x9x1",                                   {"HIPBLASLT_CLUSTER_SHAPE_1x9x1",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x10x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x10x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x11x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x11x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x12x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x12x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x13x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x13x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x14x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x14x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_1x15x1",                                  {"HIPBLASLT_CLUSTER_SHAPE_1x15x1",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_CLUSTER_SHAPE_END",                                     {"HIPBLASLT_CLUSTER_SHAPE_END",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulInnerShape_t",                                     {"hipblasLtMatmulInnerShape_t",                                       "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED",                          {"HIPBLASLT_MATMUL_INNER_SHAPE_UNDEFINED",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA884",                             {"HIPBLASLT_MATMUL_INNER_SHAPE_MMA884",                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA1684",                            {"HIPBLASLT_MATMUL_INNER_SHAPE_MMA1684",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA1688",                            {"HIPBLASLT_MATMUL_INNER_SHAPE_MMA1688",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA16816",                           {"HIPBLASLT_MATMUL_INNER_SHAPE_MMA16816",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_INNER_SHAPE_END",                                {"HIPBLASLT_MATMUL_INNER_SHAPE_END",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtPointerMode_t",                                          {"hipblasLtPointerMode_t",                                            "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_HOST",                                     {"HIPBLASLT_POINTER_MODE_HOST",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_DEVICE",                                   {"HIPBLASLT_POINTER_MODE_DEVICE",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_DEVICE_VECTOR",                            {"HIPBLASLT_POINTER_MODE_DEVICE_VECTOR",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",            {"HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",            {"HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"cublasLtPointerModeMask_t",                                      {"hipblasLtPointerModeMask_t",                                        "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_MASK_HOST",                                {"HIPBLASLT_POINTER_MODE_MASK_HOST",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_MASK_DEVICE",                              {"HIPBLASLT_POINTER_MODE_MASK_DEVICE",                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR",                       {"HIPBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR",                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO",       {"HIPLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO",          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST",       {"HIPBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST",         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA",                              {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_FMA",                                "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA",                             {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_HMMA",                               "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA",                             {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_IMMA",                               "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA",                             {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_DMMA",                               "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK",                   {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK",                     "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK",                     {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK",                       "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F",                  {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F",                    "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F",                  {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F",                    "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F",                  {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F",                    "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I",                  {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I",                    "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK",            {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK",              "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F",                        {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F",                          "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF",                       {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF",                         "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32",                       {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32",                         "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F",                        {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F",                          "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F",                        {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F",                          "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I",                         {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I",                           "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3",                    {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3",                      "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2",                    {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2",                      "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK",               {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK",                 "",                                                         CONV_DEFINE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN",                         {"HIPBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN",                           "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtNumericalImplFlags_t",                                   {"hipblasLtNumericalImplFlags_t",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtOrder_t",                                                {"hipblasLtOrder_t",                                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_ORDER_COL",                                             {"HIPBLASLT_ORDER_COL",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_ORDER_ROW",                                             {"HIPBLASLT_ORDER_ROW",                                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_ORDER_COL32",                                           {"HIPBLASLT_ORDER_COL32",                                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ORDER_COL4_4R2_8C",                                     {"HIPBLASLT_ORDER_COL4_4R2_8C",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ORDER_COL32_2R_4R4",                                    {"HIPBLASLT_ORDER_COL32_2R_4R4",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatrixLayoutAttribute_t",                                {"hipblasLtMatrixLayoutAttribute_t",                                  "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_TYPE",                                    {"HIPBLASLT_MATRIX_LAYOUT_TYPE",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_ORDER",                                   {"HIPBLASLT_MATRIX_LAYOUT_ORDER",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_ROWS",                                    {"HIPBLASLT_MATRIX_LAYOUT_ROWS",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_COLS",                                    {"HIPBLASLT_MATRIX_LAYOUT_COLS",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_LD",                                      {"HIPBLASLT_MATRIX_LAYOUT_LD",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT",                             {"HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT",                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",                    {"HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET",                            {"HIPBLASLT_MATRIX_LAYOUT_PLANE_OFFSET",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulDescAttributes_t",                                 {"hipblasLtMatmulDescAttributes_t",                                   "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_COMPUTE_TYPE",                              {"HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE",                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_SCALE_TYPE",                                {"HIPBLASLT_MATMUL_DESC_SCALE_TYPE",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_POINTER_MODE",                              {"HIPBLASLT_MATMUL_DESC_POINTER_MODE",                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_TRANSA",                                    {"HIPBLASLT_MATMUL_DESC_TRANSA",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_TRANSB",                                    {"HIPBLASLT_MATMUL_DESC_TRANSB",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_TRANSC",                                    {"HIPBLASLT_MATMUL_DESC_TRANSC",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_FILL_MODE",                                 {"HIPBLASLT_MATMUL_DESC_FILL_MODE",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE",                                  {"HIPBLASLT_MATMUL_DESC_EPILOGUE",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_BIAS_POINTER",                              {"HIPBLASLT_MATMUL_DESC_BIAS_POINTER",                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE",                         {"HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE",                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",                      {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",                           {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",                 {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE",                 {"HIPBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE",                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET",                           {"HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_A_SCALE_POINTER",                           {"HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_B_SCALE_POINTER",                           {"HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_C_SCALE_POINTER",                           {"HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_D_SCALE_POINTER",                           {"HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_AMAX_D_POINTER",                            {"HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE",                    {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE",                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",                {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER",                 {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER",                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_FAST_ACCUM",                                {"HIPBLASLT_MATMUL_DESC_FAST_ACCUM",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",                            {"HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS",             {"HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS",               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS",             {"HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS",               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER",           {"HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER",             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER",          {"HIPBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER",            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatrixTransformDescAttributes_t",                        {"hipblasLtMatrixTransformDescAttributes_t",                          "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",                      {"HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",                    {"HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",                          {"HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",                          {"HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"cublasLtReductionScheme_t",                                      {"hipblasLtReductionScheme_t",                                        "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_REDUCTION_SCHEME_NONE",                                 {"HIPBLASLT_REDUCTION_SCHEME_NONE",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_REDUCTION_SCHEME_INPLACE",                              {"HIPBLASLT_REDUCTION_SCHEME_INPLACE",                                "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE",                         {"HIPBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE",                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",                          {"HIPBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_REDUCTION_SCHEME_MASK",                                 {"HIPBLASLT_REDUCTION_SCHEME_MASK",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtEpilogue_t",                                             {"hipblasLtEpilogue_t",                                               "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_DEFAULT",                                      {"HIPBLASLT_EPILOGUE_DEFAULT",                                        "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_RELU",                                         {"HIPBLASLT_EPILOGUE_RELU",                                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_RELU_AUX",                                     {"HIPBLASLT_EPILOGUE_RELU_AUX",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_BIAS",                                         {"HIPBLASLT_EPILOGUE_BIAS",                                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_RELU_BIAS",                                    {"HIPBLASLT_EPILOGUE_RELU_BIAS",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_RELU_AUX_BIAS",                                {"HIPBLASLT_EPILOGUE_RELU_AUX_BIAS",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_DRELU",                                        {"HIPBLASLT_EPILOGUE_DRELU",                                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_DRELU_BGRAD",                                  {"HIPBLASLT_EPILOGUE_DRELU_BGRAD",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_GELU",                                         {"HIPBLASLT_EPILOGUE_GELU",                                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_GELU_AUX",                                     {"HIPBLASLT_EPILOGUE_GELU_AUX",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_GELU_BIAS",                                    {"HIPBLASLT_EPILOGUE_GELU_BIAS",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_GELU_AUX_BIAS",                                {"HIPBLASLT_EPILOGUE_GELU_AUX_BIAS",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_DGELU",                                        {"HIPBLASLT_EPILOGUE_DGELU",                                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_DGELU_BGRAD",                                  {"HIPBLASLT_EPILOGUE_DGELU_BGRAD",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_BGRADA",                                       {"HIPBLASLT_EPILOGUE_BGRADA",                                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_EPILOGUE_BGRADB",                                       {"HIPBLASLT_EPILOGUE_BGRADB",                                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"cublasLtMatmulSearch_t",                                         {"hipblasLtMatmulSearch_t",                                           "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_BEST_FIT",                                       {"HIPBLASLT_SEARCH_BEST_FIT",                                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID",                             {"HIPBLASLT_SEARCH_LIMITED_BY_ALGO_ID",                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_RESERVED_02",                                    {"HIPBLASLT_SEARCH_RESERVED_02",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_RESERVED_03",                                    {"HIPBLASLT_SEARCH_RESERVED_03",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_RESERVED_04",                                    {"HIPBLASLT_SEARCH_RESERVED_04",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_SEARCH_RESERVED_05",                                    {"HIPBLASLT_SEARCH_RESERVED_05",                                      "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulPreferenceAttributes_t",                           {"hipblasLtMatmulPreferenceAttributes_t",                             "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_SEARCH_MODE",                               {"HIPBLASLT_MATMUL_PREF_SEARCH_MODE",                                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",                       {"HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",                         "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",                     {"HIPBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",                     {"HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",                     {"HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",                     {"HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",                     {"HIPBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT",                           {"HIPBLASLT_MATMUL_PREF_MAX_WAVES_COUNT",                             "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_MATMUL_PREF_IMPL_MASK",                                 {"HIPBLASLT_MATMUL_PREF_IMPL_MASK",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulHeuristicResult_t",                                {"hipblasLtMatmulHeuristicResult_t",                                  "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, ROC_UNSUPPORTED}},
  {"cublasLtMatmulAlgoCapAttributes_t",                              {"hipblasLtMatmulAlgoCapAttributes_t",                                "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_SPLITK_SUPPORT",                               {"HIPBLASLT_ALGO_CAP_SPLITK_SUPPORT",                                 "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK",                        {"HIPBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT",                        {"HIPBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT",                        {"HIPBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT",                  {"HIPBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT",                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_UPLO_SUPPORT",                                 {"HIPBLASLT_ALGO_CAP_UPLO_SUPPORT",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_TILE_IDS",                                     {"HIPBLASLT_ALGO_CAP_TILE_IDS",                                       "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX",                            {"HIPBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER",                          {"HIPBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_POINTER_MODE_MASK",                            {"HIPBLASLT_ALGO_CAP_POINTER_MODE_MASK",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_EPILOGUE_MASK",                                {"HIPBLASLT_ALGO_CAP_EPILOGUE_MASK",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_STAGES_IDS",                                   {"HIPBLASLT_ALGO_CAP_STAGES_IDS",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_LD_NEGATIVE",                                  {"HIPBLASLT_ALGO_CAP_LD_NEGATIVE",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS",                         {"HIPBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS",                           "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES",                        {"HIPBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES",                        {"HIPBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES",                        {"HIPBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES",                        {"HIPBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES",                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CAP_ATOMIC_SYNC",                                  {"HIPBLASLT_ALGO_CAP_ATOMIC_SYNC",                                    "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtMatmulAlgoConfigAttributes_t",                           {"hipblasLtMatmulAlgoConfigAttributes_t",                             "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_ID",                                        {"HIPBLASLT_ALGO_CONFIG_ID",                                          "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_TILE_ID",                                   {"HIPBLASLT_ALGO_CONFIG_TILE_ID",                                     "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_SPLITK_NUM",                                {"HIPBLASLT_ALGO_CONFIG_SPLITK_NUM",                                  "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",                          {"HIPBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",                             {"HIPBLASLT_ALGO_CONFIG_CTA_SWIZZLING",                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",                             {"HIPBLASLT_ALGO_CONFIG_CUSTOM_OPTION",                               "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_STAGES_ID",                                 {"HIPBLASLT_ALGO_CONFIG_STAGES_ID",                                   "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",                            {"HIPBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",                              "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",                          {"HIPBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",                            "",                                                         CONV_NUMERIC_LITERAL, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
  {"cublasLtLoggerCallback_t",                                       {"hipblasLtLoggerCallback_t",                                         "",                                                         CONV_TYPE, API_BLAS, SEC::BLAS_LT_DATA_TYPES, UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_BLAS_TYPE_NAME_VER_MAP {
  {"CUBLAS_OP_CONJG",                                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLAS_OP_HERMITAN",                                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLAS_FILL_MODE_FULL",                                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasDataType_t",                                               {CUDA_75,  CUDA_0,   CUDA_0  }},
  {"cublasMath_t",                                                   {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_DEFAULT_MATH",                                            {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_TENSOR_OP_MATH",                                          {CUDA_90,  CUDA_110, CUDA_0  }},
  {"CUBLAS_PEDANTIC_MATH",                                           {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_TF32_TENSOR_OP_MATH",                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",               {CUDA_110, CUDA_0,   CUDA_0  }},
  {"cublasGemmAlgo_t",                                               {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_DFALT",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_DEFAULT",                                            {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO0",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO1",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO2",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO3",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO4",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO5",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO6",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO7",                                              {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO15_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO14_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO13_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO12_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO11_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO10_TENSOR_OP",                                   {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO9_TENSOR_OP",                                    {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO8_TENSOR_OP",                                    {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO7_TENSOR_OP",                                    {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO6_TENSOR_OP",                                    {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO5_TENSOR_OP",                                    {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO4_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO3_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO2_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO1_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO0_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_DFALT_TENSOR_OP",                                    {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_DEFAULT_TENSOR_OP",                                  {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO23",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO22",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO21",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO20",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO19",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO18",                                             {CUDA_92,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO17",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO16",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO15",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO14",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO13",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO12",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO11",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO10",                                             {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO9",                                              {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"CUBLAS_GEMM_ALGO8",                                              {CUDA_90,  CUDA_0,   CUDA_0  }},
  {"cudaDataType_t",                                                 {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cudaDataType",                                                   {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_16F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_16F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_32F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_32F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_64F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_64F",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_8I",                                                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_8I",                                                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_8U",                                                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_8U",                                                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_32I",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_32I",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_R_32U",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"CUDA_C_32U",                                                     {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"cublasComputeType_t",                                            {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_16F",                                             {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_16F_PEDANTIC",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32F",                                             {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32F_PEDANTIC",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32F_FAST_16F",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32F_FAST_16BF",                                   {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32F_FAST_TF32",                                   {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_64F",                                             {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_64F_PEDANTIC",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32I",                                             {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUBLAS_COMPUTE_32I_PEDANTIC",                                    {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_4I",                                                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_4I",                                                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_4U",                                                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_4U",                                                      {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_16I",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_16I",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_16U",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_16U",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_64I",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_64I",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_64U",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_C_64U",                                                     {CUDA_110, CUDA_0,   CUDA_0  }},
  {"CUDA_R_8F_E4M3",                                                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUDA_R_8F_E5M2",                                                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cublasLtHandle_t",                                               {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtContext",                                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatrixLayoutOpaque_t",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatrixLayoutStruct",                                     {CUDA_101, CUDA_0,   CUDA_102}},
  {"cublasLtMatrixLayout_t",                                         {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulAlgo_t",                                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulDescOpaque_t",                                     {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatmulDesc_t",                                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatrixTransformDescOpaque_t",                            {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatrixTransformDesc_t",                                  {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulPreferenceOpaque_t",                               {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatmulPreference_t",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulTile_t",                                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_UNDEFINED",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_8x8",                                       {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_8x16",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_16x8",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_8x32",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_16x16",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x8",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_8x64",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_16x32",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x16",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x8",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x32",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x64",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x32",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x128",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x64",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_128x32",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x128",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_128x64",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x256",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_128x128",                                   {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_256x64",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x512",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_128x256",                                   {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_256x128",                                   {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_512x64",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_64x96",                                     {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_96x64",                                     {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_96x128",                                    {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_128x160",                                   {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_160x128",                                   {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_192x128",                                   {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_128x192",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_128x96",                                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_TILE_32x256",                                    {CUDA_121, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12011, CUBLAS_VERSION 120103, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 1 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_256x32",                                    {CUDA_121, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12011, CUBLAS_VERSION 120103, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 1 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_TILE_END",                                       {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulStages_t",                                         {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_UNDEFINED",                               {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x1",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x2",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x3",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x4",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x5",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x6",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x1",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x2",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x3",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x4",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x5",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x6",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x1",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x2",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x3",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x4",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x5",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_64x6",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x1",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x2",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x3",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x4",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x5",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_128x6",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_32x10",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_8x4",                                     {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_16x10",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_MATMUL_STAGES_8x5",                                     {CUDA_112, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11021, CUBLAS_VERSION 11401, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 4 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_STAGES_8x3",                                     {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_8xAUTO",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_16xAUTO",                                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_32xAUTO",                                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_64xAUTO",                                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_128xAUTO",                                {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_STAGES_END",                                     {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtClusterShape_t",                                         {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_AUTO",                                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_4x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_4x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x4x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x4x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_4x4x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_8x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x8x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_8x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x8x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_16x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x16x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_3x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_5x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_6x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_7x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_9x1x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_10x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_11x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_12x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_13x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_14x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_15x1x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_3x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_5x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_6x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_7x2x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x3x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x3x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_3x3x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_4x3x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_5x3x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_3x4x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x5x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x5x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_3x5x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x6x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x6x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x7x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_2x7x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x9x1",                                   {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x10x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x11x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x12x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x13x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x14x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_1x15x1",                                  {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_CLUSTER_SHAPE_END",                                     {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulInnerShape_t",                                     {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED",                          {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA884",                             {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA1684",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA1688",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_MMA16816",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_INNER_SHAPE_END",                                {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cublasLtPointerMode_t",                                          {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_HOST",                                     {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_DEVICE_VECTOR",                            {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",            {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",            {CUDA_114, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11042, CUBLAS_VERSION 11601, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 6 CUBLAS_VER_PATCH 1
  {"cublasLtPointerModeMask_t",                                      {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_MASK_HOST",                                {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_MASK_DEVICE",                              {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR",                       {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO",       {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST",       {CUDA_114, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11042, CUBLAS_VERSION 11601, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 6 CUBLAS_VER_PATCH 1
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA",                              {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA",                             {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA",                             {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA",                             {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK",                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK",                     {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F",                  {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F",                  {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F",                  {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I",                  {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK",            {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF",                       {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32",                       {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I",                         {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3",                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2",                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK",               {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN",                         {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtNumericalImplFlags_t",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtOrder_t",                                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ORDER_COL",                                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ORDER_ROW",                                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ORDER_COL32",                                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ORDER_COL4_4R2_8C",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ORDER_COL32_2R_4R4",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatrixLayoutAttribute_t",                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_TYPE",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_ORDER",                                   {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_ROWS",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_COLS",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_LD",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT",                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET",                            {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulDescAttributes_t",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_COMPUTE_TYPE",                              {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_SCALE_TYPE",                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_POINTER_MODE",                              {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_TRANSA",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_TRANSB",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_TRANSC",                                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_FILL_MODE",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE",                                  {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_MATMUL_DESC_BIAS_POINTER",                              {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE",                         {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",                      {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",                           {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",                 {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE",                 {CUDA_114, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11042, CUBLAS_VERSION 11601, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 6 CUBLAS_VER_PATCH 1
  {"CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET",                           {CUDA_115, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11051, CUBLAS_VERSION 11704, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 7 CUBLAS_VER_PATCH 4
  {"CUBLASLT_MATMUL_DESC_A_SCALE_POINTER",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_B_SCALE_POINTER",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_C_SCALE_POINTER",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_D_SCALE_POINTER",                           {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_AMAX_D_POINTER",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE",                    {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",                {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER",                 {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_FAST_ACCUM",                                {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS",             {CUDA_122, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12022, CUBLAS_VERSION 120205, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 2 CUBLAS_VER_PATCH 5
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS",             {CUDA_122, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12022, CUBLAS_VERSION 120205, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 2 CUBLAS_VER_PATCH 5
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_IN_COUNTERS_POINTER",           {CUDA_122, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12022, CUBLAS_VERSION 120205, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 2 CUBLAS_VER_PATCH 5
  {"CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_OUT_COUNTERS_POINTER",          {CUDA_122, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12022, CUBLAS_VERSION 120205, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 2 CUBLAS_VER_PATCH 5
  {"cublasLtMatrixTransformDescAttributes_t",                        {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",                    {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtReductionScheme_t",                                      {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_REDUCTION_SCHEME_NONE",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_REDUCTION_SCHEME_INPLACE",                              {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE",                         {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_REDUCTION_SCHEME_MASK",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtEpilogue_t",                                             {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_EPILOGUE_DEFAULT",                                      {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_EPILOGUE_RELU",                                         {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_EPILOGUE_RELU_AUX",                                     {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_BIAS",                                         {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_EPILOGUE_RELU_BIAS",                                    {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_EPILOGUE_RELU_AUX_BIAS",                                {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_DRELU",                                        {CUDA_116, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_EPILOGUE_DRELU_BGRAD",                                  {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_GELU",                                         {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_GELU_AUX",                                     {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_GELU_BIAS",                                    {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_GELU_AUX_BIAS",                                {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_DGELU",                                        {CUDA_116, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_EPILOGUE_DGELU_BGRAD",                                  {CUDA_113, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11031, CUBLAS_VERSION 11501, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 5 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_BGRADA",                                       {CUDA_114, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11042, CUBLAS_VERSION 11601, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 6 CUBLAS_VER_PATCH 1
  {"CUBLASLT_EPILOGUE_BGRADB",                                       {CUDA_114, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11042, CUBLAS_VERSION 11601, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 6 CUBLAS_VER_PATCH 1
  {"cublasLtMatmulSearch_t",                                         {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_SEARCH_BEST_FIT",                                       {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID",                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_SEARCH_RESERVED_02",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_SEARCH_RESERVED_03",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_SEARCH_RESERVED_04",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_SEARCH_RESERVED_05",                                    {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatmulPreferenceAttributes_t",                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_SEARCH_MODE",                               {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",                       {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT",                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_MATMUL_PREF_IMPL_MASK",                                 {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"cublasLtMatmulHeuristicResult_t",                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"cublasLtMatmulAlgoCapAttributes_t",                              {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_SPLITK_SUPPORT",                               {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK",                        {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT",                        {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT",                        {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT",                  {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_UPLO_SUPPORT",                                 {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_TILE_IDS",                                     {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX",                            {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER",                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CAP_POINTER_MODE_MASK",                            {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_ALGO_CAP_EPILOGUE_MASK",                                {CUDA_101, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 10011, CUBLAS_VERSION 10020, CUBLAS_VER_MAJOR 10 CUBLAS_VER_MINOR 2
  {"CUBLASLT_ALGO_CAP_STAGES_IDS",                                   {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_LD_NEGATIVE",                                  {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS",                         {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES",                        {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CAP_ATOMIC_SYNC",                                  {CUDA_122, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 12022, CUBLAS_VERSION 120205, CUBLAS_VER_MAJOR 12 CUBLAS_VER_MINOR 2 CUBLAS_VER_PATCH 5
  {"cublasLtMatmulAlgoConfigAttributes_t",                           {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_ID",                                        {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_TILE_ID",                                   {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_SPLITK_NUM",                                {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",                          {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",                             {CUDA_101, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_STAGES_ID",                                 {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11001, CUBLAS_VERSION 11000, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 0
  {"CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",                            {CUDA_118, CUDA_0,   CUDA_0  }},
  {"CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",                          {CUDA_118, CUDA_0,   CUDA_0  }},
  {"cublasLtLoggerCallback_t",                                       {CUDA_110, CUDA_0,   CUDA_0  }}, // A: CUDA_VERSION 11003, CUBLAS_VERSION 11200, CUBLAS_VER_MAJOR 11 CUBLAS_VER_MINOR 2
};

const std::map<llvm::StringRef, hipAPIversions> HIP_BLAS_TYPE_NAME_VER_MAP {
  {"hipblasOperation_t",                                             {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_OP_N",                                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_OP_T",                                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_OP_C",                                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasGemmAlgo_t",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_GEMM_DEFAULT",                                           {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasStatus_t",                                                {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_SUCCESS",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_NOT_INITIALIZED",                                 {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_ALLOC_FAILED",                                    {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_INVALID_VALUE",                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_MAPPING_ERROR",                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_EXECUTION_FAILED",                                {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_INTERNAL_ERROR",                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_NOT_SUPPORTED",                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_STATUS_ARCH_MISMATCH",                                   {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasFillMode_t",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_FILL_MODE_LOWER",                                        {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_FILL_MODE_UPPER",                                        {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_FILL_MODE_FULL",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasDiagType_t",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_DIAG_NON_UNIT",                                          {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_DIAG_UNIT",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasSideMode_t",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_SIDE_LEFT",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_SIDE_RIGHT",                                             {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasPointerMode_t",                                           {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_POINTER_MODE_HOST",                                      {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_POINTER_MODE_DEVICE",                                    {HIP_1082, HIP_0,    HIP_0   }},
  {"hipblasAtomicsMode_t",                                           {HIP_3100, HIP_0,    HIP_0   }},
  {"HIPBLAS_ATOMICS_NOT_ALLOWED",                                    {HIP_3100, HIP_0,    HIP_0   }},
  {"HIPBLAS_ATOMICS_ALLOWED",                                        {HIP_3100, HIP_0,    HIP_0   }},
  {"hipblasDatatype_t",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_16F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_16F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_32F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_32F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_64F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_64F",                                                  {HIP_1082, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_8I",                                                   {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_8I",                                                   {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_8U",                                                   {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_8U",                                                   {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_32I",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_32I",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_32U",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_32U",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_R_16B",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"HIPBLAS_C_16B",                                                  {HIP_3000, HIP_0,    HIP_0   }},
  {"hipblasHandle_t",                                                {HIP_3000, HIP_0,    HIP_0   }},
  {"hipDataType",                                                    {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_16F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_16F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_32F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_32F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_64F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_64F",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_8I",                                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_8I",                                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_8U",                                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_8U",                                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_32I",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_32I",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_32U",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_32U",                                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_R_16BF",                                                     {HIP_5070, HIP_0,    HIP_0   }},
  {"HIP_C_16BF",                                                     {HIP_5070, HIP_0,    HIP_0   }},
  {"hipblasComputeType_t",                                           {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_16F",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_16F_PEDANTIC",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32F",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32F_PEDANTIC",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32F_FAST_16F",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32F_FAST_16BF",                                  {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32F_FAST_TF32",                                  {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_64F",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_64F_PEDANTIC",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32I",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLAS_COMPUTE_32I_PEDANTIC",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasMath_t",                                                  {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLAS_DEFAULT_MATH",                                           {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLAS_TENSOR_OP_MATH",                                         {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLAS_PEDANTIC_MATH",                                          {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLAS_TF32_TENSOR_OP_MATH",                                    {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION",              {HIP_6010, HIP_0,    HIP_0   }},
  {"hipblasLtHandle_t",                                              {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulAlgo_t",                                          {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulDescOpaque_t",                                    {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulDesc_t",                                          {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatrixTransformDescOpaque_t",                           {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasLtMatrixTransformDesc_t",                                 {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulPreferenceOpaque_t",                              {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulPreference_t",                                    {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtPointerMode_t",                                         {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_POINTER_MODE_HOST",                                    {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_POINTER_MODE_DEVICE",                                  {HIP_6010, HIP_0,    HIP_0   }},
  {"HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",           {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasLtOrder_t",                                               {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_ORDER_COL",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_ORDER_ROW",                                            {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasLtMatrixLayoutAttribute_t",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_TYPE",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_ORDER",                                  {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_ROWS",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_COLS",                                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_LD",                                     {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT",                            {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",                   {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulDescAttributes_t",                                {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_POINTER_MODE",                             {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_TRANSA",                                   {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_TRANSB",                                   {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_EPILOGUE",                                 {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_BIAS_POINTER",                             {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",                     {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",                          {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",                {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER",                          {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER",                          {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER",                          {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER",                          {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",               {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",                           {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatrixTransformDescAttributes_t",                       {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",                     {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",                   {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",                         {HIP_6000, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",                         {HIP_6000, HIP_0,    HIP_0   }},
  {"hipblasLtEpilogue_t",                                            {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_DEFAULT",                                     {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_RELU",                                        {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_BIAS",                                        {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_RELU_BIAS",                                   {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_GELU",                                        {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_GELU_AUX",                                    {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_GELU_BIAS",                                   {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_GELU_AUX_BIAS",                               {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_DGELU",                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_DGELU_BGRAD",                                 {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_BGRADA",                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"HIPBLASLT_EPILOGUE_BGRADB",                                      {HIP_5070, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulPreferenceAttributes_t",                          {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_PREF_SEARCH_MODE",                              {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",                      {HIP_5050, HIP_0,    HIP_0   }},
  {"hipblasLtMatmulHeuristicResult_t",                               {HIP_5050, HIP_0,    HIP_0   }},
  {"HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER",                           {HIP_6020, HIP_0,    HIP_0,  }},

  {"rocblas_handle",                                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"_rocblas_handle",                                                {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_operation",                                              {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_operation_none",                                         {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_operation_transpose",                                    {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_operation_conjugate_transpose",                          {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_fill",                                                   {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_fill_upper",                                             {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_fill_lower",                                             {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_fill_full",                                              {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_diagonal",                                               {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_diagonal_non_unit",                                      {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_diagonal_unit",                                          {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_side",                                                   {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_side_left",                                              {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_side_right",                                             {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status",                                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_success",                                         {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_invalid_handle",                                  {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_not_implemented",                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_invalid_pointer",                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_invalid_size",                                    {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_memory_error",                                    {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_internal_error",                                  {HIP_1050, HIP_0,    HIP_0   }},
  {"rocblas_status_perf_degraded",                                   {HIP_3050, HIP_0,    HIP_0   }},
  {"rocblas_status_size_query_mismatch",                             {HIP_3050, HIP_0,    HIP_0   }},
  {"rocblas_status_arch_mismatch",                                   {HIP_5070, HIP_0,    HIP_0   }},
  {"rocblas_status_invalid_value",                                   {HIP_3050, HIP_0,    HIP_0   }},
  {"rocblas_datatype",                                               {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f16_r",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f32_r",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f64_r",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f16_c",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f32_c",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_f64_c",                                         {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_datatype_i8_r",                                          {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_u8_r",                                          {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_i32_r",                                         {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_u32_r",                                         {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_i8_c",                                          {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_u8_c",                                          {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_i32_c",                                         {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_u32_c",                                         {HIP_2000, HIP_0,    HIP_0   }},
  {"rocblas_datatype_bf16_r",                                        {HIP_3050, HIP_0,    HIP_0   }},
  {"rocblas_datatype_bf16_c",                                        {HIP_3050, HIP_0,    HIP_0   }},
  {"rocblas_pointer_mode",                                           {HIP_1060, HIP_0,    HIP_0   }},
  {"rocblas_pointer_mode_host",                                      {HIP_1060, HIP_0,    HIP_0   }},
  {"rocblas_pointer_mode_device",                                    {HIP_1060, HIP_0,    HIP_0   }},
  {"rocblas_atomics_mode",                                           {HIP_3080, HIP_0,    HIP_0   }},
  {"rocblas_atomics_not_allowed",                                    {HIP_3080, HIP_0,    HIP_0   }},
  {"rocblas_atomics_allowed",                                        {HIP_3080, HIP_0,    HIP_0   }},
  {"rocblas_gemm_algo",                                              {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_gemm_algo_standard",                                     {HIP_1082, HIP_0,    HIP_0   }},
  {"rocblas_math_mode",                                              {HIP_5070, HIP_0,    HIP_0   }},
  {"rocblas_default_math",                                           {HIP_5070, HIP_0,    HIP_0   }},
  {"rocblas_xf32_xdl_math_op",                                       {HIP_5070, HIP_0,    HIP_0   }},
  {"rocblas_computetype",                                            {HIP_5070, HIP_0,    HIP_0   }},
  {"rocblas_compute_type_f32",                                       {HIP_5070, HIP_0,    HIP_0   }},
};
