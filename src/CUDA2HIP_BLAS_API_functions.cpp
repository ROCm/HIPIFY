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
const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_FUNCTION_MAP {

  // Blas management functions
  {"cublasInit",                     {"hipblasInit",                     "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasShutdown",                 {"hipblasShutdown",                 "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetVersion",               {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetError",                 {"hipblasGetError",                 "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasAlloc",                    {"hipblasAlloc",                    "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasFree",                     {"hipblasFree",                     "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasSetKernelStream",          {"hipblasSetKernelStream",          "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetMathMode",              {"hipblasGetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasSetMathMode",              {"hipblasSetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasMigrateComputeType",       {"hipblasMigrateComputeType",       "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},

  // Blas logging
  {"cublasLogCallback",              {"hipblasLogCallback",              "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasLoggerConfigure",          {"hipblasLoggerConfigure",          "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasSetLoggerCallback",        {"hipblasSetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetLoggerCallback",        {"hipblasGetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},

  // Blas1 (v1) Routines
  {"cublasCreate",                   {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasDestroy",                  {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetStream",                {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetStream",                {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetPointerMode",           {"hipblasSetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetPointerMode",           {"hipblasGetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetVector",                {"hipblasSetVector",                "rocblas_set_vector",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetVector",                {"hipblasGetVector",                "rocblas_get_vector",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetVectorAsync",           {"hipblasSetVectorAsync",           "rocblas_set_vector_async",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetVectorAsync",           {"hipblasGetVectorAsync",           "rocblas_get_vector_async",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetMatrix",                {"hipblasSetMatrix",                "rocblas_set_matrix",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetMatrix",                {"hipblasGetMatrix",                "rocblas_get_matrix",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           "rocblas_set_matrix_async",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           "rocblas_get_matrix_async",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasXerbla",                   {"hipblasXerbla",                   "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},

  // Blas2 (v2) Routines
  {"cublasCreate_v2",                {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasDestroy_v2",               {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetVersion_v2",            {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasGetProperty",              {"hipblasGetProperty",              "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},
  {"cublasSetStream_v2",             {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetStream_v2",             {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS, 4}},
  {"cublasGetCudartVersion",         {"hipblasGetCudartVersion",         "",                                         CONV_LIB_FUNC, API_BLAS, 4, UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2",                    {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDnrm2",                    {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasScnrm2",                   {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDznrm2",                   {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasNrm2Ex",                   {"hipblasNrm2Ex",                   "",                                         CONV_LIB_FUNC, API_BLAS, 5, UNSUPPORTED}},

  // DOT
  {"cublasSdot",                     {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDdot",                     {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCdotu",                    {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCdotc",                    {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdotu",                    {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdotc",                    {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // SCAL
  {"cublasSscal",                    {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDscal",                    {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCscal",                    {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCsscal",                   {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZscal",                    {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdscal",                   {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // AXPY
  {"cublasSaxpy",                    {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDaxpy",                    {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCaxpy",                    {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZaxpy",                    {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // COPY
  {"cublasScopy",                    {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDcopy",                    {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCcopy",                    {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZcopy",                    {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // SWAP
  {"cublasSswap",                    {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDswap",                    {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCswap",                    {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZswap",                    {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // AMAX
  {"cublasIsamax",                   {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIdamax",                   {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIcamax",                   {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIzamax",                   {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // AMIN
  {"cublasIsamin",                   {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIdamin",                   {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIcamin",                   {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIzamin",                   {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // ASUM
  {"cublasSasum",                    {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDasum",                    {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasScasum",                   {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDzasum",                   {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // ROT
  {"cublasSrot",                     {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrot",                     {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCrot",                     {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCsrot",                    {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZrot",                     {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdrot",                    {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTG
  {"cublasSrotg",                    {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotg",                    {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCrotg",                    {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZrotg",                    {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTM
  {"cublasSrotm",                    {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotm",                    {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTMG
  {"cublasSrotmg",                   {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotmg",                   {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // GEMV
  {"cublasSgemv",                    {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDgemv",                    {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgemv",                    {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgemv",                    {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // GBMV
  {"cublasSgbmv",                    {"hipblasSgbmv",                    "rocblas_sgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDgbmv",                    {"hipblasDgbmv",                    "rocblas_dgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgbmv",                    {"hipblasCgbmv",                    "rocblas_cgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgbmv",                    {"hipblasZgbmv",                    "rocblas_zgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TRMV
  {"cublasStrmv",                    {"hipblasStrmv",                    "rocblas_strmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtrmv",                    {"hipblasDtrmv",                    "rocblas_dtrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtrmv",                    {"hipblasCtrmv",                    "rocblas_ctrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtrmv",                    {"hipblasZtrmv",                    "rocblas_ztrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TBMV
  {"cublasStbmv",                    {"hipblasStbmv",                    "rocblas_stbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtbmv",                    {"hipblasDtbmv",                    "rocblas_dtbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtbmv",                    {"hipblasCtbmv",                    "rocblas_ctbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtbmv",                    {"hipblasZtbmv",                    "rocblas_ztbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TPMV
  {"cublasStpmv",                    {"hipblasStpmv",                    "rocblas_stpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtpmv",                    {"hipblasDtpmv",                    "rocblas_dtpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtpmv",                    {"hipblasCtpmv",                    "rocblas_ctpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtpmv",                    {"hipblasZtpmv",                    "rocblas_ztpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TRSV
  {"cublasStrsv",                    {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtrsv",                    {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtrsv",                    {"hipblasCtrsv",                    "rocblas_ctrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtrsv",                    {"hipblasZtrsv",                    "rocblas_ztrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TPSV
  {"cublasStpsv",                    {"hipblasStpsv",                    "rocblas_stpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtpsv",                    {"hipblasDtpsv",                    "rocblas_dtpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtpsv",                    {"hipblasCtpsv",                    "rocblas_ctpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtpsv",                    {"hipblasZtpsv",                    "rocblas_ztpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TBSV
  {"cublasStbsv",                    {"hipblasStbsv",                    "rocblas_stbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtbsv",                    {"hipblasDtbsv",                    "rocblas_dtbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtbsv",                    {"hipblasCtbsv",                    "rocblas_ctbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtbsv",                    {"hipblasZtbsv",                    "rocblas_ztbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SYMV/HEMV
  {"cublasSsymv",                    {"hipblasSsymv",                    "rocblas_ssymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsymv",                    {"hipblasDsymv",                    "rocblas_dsymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsymv",                    {"hipblasCsymv",                    "rocblas_csymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsymv",                    {"hipblasZsymv",                    "rocblas_zsymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChemv",                    {"hipblasChemv",                    "rocblas_chemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhemv",                    {"hipblasZhemv",                    "rocblas_zhemv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SBMV/HBMV
  {"cublasSsbmv",                    {"hipblasSsbmv",                    "rocblas_ssbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsbmv",                    {"hipblasDsbmv",                    "rocblas_dsbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChbmv",                    {"hipblasChbmv",                    "rocblas_chbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhbmv",                    {"hipblasZhbmv",                    "rocblas_zhbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SPMV/HPMV
  {"cublasSspmv",                    {"hipblasSspmv",                    "rocblas_sspmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspmv",                    {"hipblasDspmv",                    "rocblas_dspmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpmv",                    {"hipblasChpmv",                    "rocblas_chpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpmv",                    {"hipblasZhpmv",                    "rocblas_zhpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // GER
  {"cublasSger",                     {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDger",                     {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgeru",                    {"hipblasCgeru",                    "rocblas_cgeru",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgerc",                    {"hipblasCgerc",                    "rocblas_cgerc",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgeru",                    {"hipblasZgeru",                    "rocblas_zgeru",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgerc",                    {"hipblasZgerc",                    "rocblas_zgerc",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SYR/HER
  {"cublasSsyr",                     {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsyr",                     {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsyr",                     {"hipblasCsyr",                     "rocblas_csyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsyr",                     {"hipblasZsyr",                     "rocblas_zsyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCher",                     {"hipblasCher",                     "rocblas_cher",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZher",                     {"hipblasZher",                     "rocblas_zher",                             CONV_LIB_FUNC, API_BLAS, 6}},

  // SPR/HPR
  {"cublasSspr",                     {"hipblasSspr",                     "rocblas_sspr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspr",                     {"hipblasDspr",                     "rocblas_dspr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpr",                     {"hipblasChpr",                     "rocblas_chpr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpr",                     {"hipblasZhpr",                     "rocblas_zhpr",                             CONV_LIB_FUNC, API_BLAS, 6}},

  // SYR2/HER2
  {"cublasSsyr2",                    {"hipblasSsyr2",                    "rocblas_ssyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsyr2",                    {"hipblasDsyr2",                    "rocblas_dsyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsyr2",                    {"hipblasCsyr2",                    "rocblas_csyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsyr2",                    {"hipblasZsyr2",                    "rocblas_zsyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCher2",                    {"hipblasCher2",                    "rocblas_cher2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZher2",                    {"hipblasZher2",                    "rocblas_zher2",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SPR2/HPR2
  {"cublasSspr2",                    {"hipblasSspr2",                    "rocblas_sspr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspr2",                    {"hipblasDspr2",                    "rocblas_dspr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpr2",                    {"hipblasChpr2",                    "rocblas_chpr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpr2",                    {"hipblasZhpr2",                    "rocblas_zhpr2",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // Blas3 (v1) Routines
  // GEMM
  {"cublasSgemm",                    {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDgemm",                    {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemm",                    {"hipblasCgemm",                    "rocblas_cgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZgemm",                    {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasHgemm",                    {"hipblasHgemm",                    "rocblas_hgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // BATCH GEMM
  {"cublasSgemmBatched",             {"hipblasSgemmBatched",             "rocblas_sgemm_batched",                    CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDgemmBatched",             {"hipblasDgemmBatched",             "rocblas_dgemm_batched",                    CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasHgemmBatched",             {"hipblasHgemmBatched",             "rocblas_hgemm_batched",                    CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasSgemmStridedBatched",      {"hipblasSgemmStridedBatched",      "rocblas_sgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDgemmStridedBatched",      {"hipblasDgemmStridedBatched",      "rocblas_dgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemmBatched",             {"hipblasCgemmBatched",             "rocblas_cgemm_batched",                    CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemm3mBatched",           {"hipblasCgemm3mBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, 7, UNSUPPORTED}},
  {"cublasZgemmBatched",             {"hipblasZgemmBatched",             "rocblas_zgemm_batched",                    CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemmStridedBatched",      {"hipblasCgemmStridedBatched",      "rocblas_cgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemm3mStridedBatched",    {"hipblasCgemm3mStridedBatched",    "",                                         CONV_LIB_FUNC, API_BLAS, 7, UNSUPPORTED}},
  {"cublasZgemmStridedBatched",      {"hipblasZgemmStridedBatched",      "rocblas_zgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasHgemmStridedBatched",      {"hipblasHgemmStridedBatched",      "rocblas_hgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS, 7}},

  // SYRK
  {"cublasSsyrk",                    {"hipblasSsyrk",                    "rocblas_ssyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsyrk",                    {"hipblasDsyrk",                    "rocblas_dsyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsyrk",                    {"hipblasCsyrk",                    "rocblas_csyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsyrk",                    {"hipblasZsyrk",                    "rocblas_zsyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // HERK
  {"cublasCherk",                    {"hipblasCherk",                    "rocblas_cherk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZherk",                    {"hipblasZherk",                    "rocblas_zherk",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // SYR2K
  {"cublasSsyr2k",                   {"hipblasSsyr2k",                   "rocblas_ssyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsyr2k",                   {"hipblasDsyr2k",                   "rocblas_dsyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsyr2k",                   {"hipblasCsyr2k",                   "rocblas_csyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsyr2k",                   {"hipblasZsyr2k",                   "rocblas_zsyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // SYRKX - eXtended SYRK
  {"cublasSsyrkx",                   {"hipblasSsyrkx",                   "rocblas_ssyrkx",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsyrkx",                   {"hipblasDsyrkx",                   "rocblas_dsyrkx",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsyrkx",                   {"hipblasCsyrkx",                   "rocblas_csyrkx",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsyrkx",                   {"hipblasZsyrkx",                   "rocblas_zsyrkx",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // HER2K
  {"cublasCher2k",                   {"hipblasCher2k",                   "rocblas_cher2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZher2k",                   {"hipblasZher2k",                   "rocblas_zher2k",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // HERKX - eXtended HERK
  {"cublasCherkx",                   {"hipblasCherkx",                   "rocblas_cherkx",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZherkx",                   {"hipblasZherkx",                   "rocblas_zherkx",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // SYMM
  {"cublasSsymm",                    {"hipblasSsymm",                    "rocblas_ssymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsymm",                    {"hipblasDsymm",                    "rocblas_dsymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsymm",                    {"hipblasCsymm",                    "rocblas_csymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsymm",                    {"hipblasZsymm",                    "rocblas_zsymm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // HEMM
  {"cublasChemm",                    {"hipblasChemm",                    "rocblas_chemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZhemm",                    {"hipblasZhemm",                    "rocblas_zhemm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // TRSM
  {"cublasStrsm",                    {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDtrsm",                    {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCtrsm",                    {"hipblasCtrsm",                    "rocblas_ctrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZtrsm",                    {"hipblasZtrsm",                    "rocblas_ztrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // TRMM
  {"cublasStrmm",                    {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDtrmm",                    {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCtrmm",                    {"hipblasCtrmm",                    "rocblas_ctrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZtrmm",                    {"hipblasZtrmm",                    "rocblas_ztrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
  // GEAM
  {"cublasSgeam",                    {"hipblasSgeam",                    "rocblas_sgeam",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasDgeam",                    {"hipblasDgeam",                    "rocblas_dgeam",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasCgeam",                    {"hipblasCgeam",                    "rocblas_cgeam",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasZgeam",                    {"hipblasZgeam",                    "rocblas_zgeam",                            CONV_LIB_FUNC, API_BLAS, 8}},

  // GETRF - Batched LU
  {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},

  // Batched inversion based on LU factorization from getrf
  {"cublasSgetriBatched",            {"hipblasSgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasDgetriBatched",            {"hipblasDgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasCgetriBatched",            {"hipblasCgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasZgetriBatched",            {"hipblasZgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},

  // Batched solver based on LU factorization from getrf
  {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             "rocblas_strsm_batched",                    CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             "rocblas_dtrsm_batched",                    CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             "rocblas_ctrsm_batched",                    CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             "rocblas_ztrsm_batched",                    CONV_LIB_FUNC, API_BLAS, 8}},

  // MATINV - Batched
  {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // Batch QR Factorization
  {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},
  {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, ROC_UNSUPPORTED}},

  // Least Square Min only m >= n and Non-transpose supported
  {"cublasSgelsBatched",             {"hipblasSgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasDgelsBatched",             {"hipblasDgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasCgelsBatched",             {"hipblasCgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasZgelsBatched",             {"hipblasZgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // DGMM
  {"cublasSdgmm",                    {"hipblasSdgmm",                    "rocblas_sdgmm",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasDdgmm",                    {"hipblasDdgmm",                    "rocblas_ddgmm",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasCdgmm",                    {"hipblasCdgmm",                    "rocblas_cdgmm",                            CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasZdgmm",                    {"hipblasZdgmm",                    "rocblas_zdgmm",                            CONV_LIB_FUNC, API_BLAS, 8}},

  // TPTTR - Triangular Pack format to Triangular format
  {"cublasStpttr",                   {"hipblasStpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasDtpttr",                   {"hipblasDtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasCtpttr",                   {"hipblasCtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasZtpttr",                   {"hipblasZtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // TRTTP - Triangular format to Triangular Pack format
  {"cublasStrttp",                   {"hipblasStrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasDtrttp",                   {"hipblasDtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasCtrttp",                   {"hipblasCtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasZtrttp",                   {"hipblasZtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // GEMV
  {"cublasSgemv_v2",                 {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDgemv_v2",                 {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgemv_v2",                 {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgemv_v2",                 {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // GBMV
  {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    "rocblas_sgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    "rocblas_dgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    "rocblas_cgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    "rocblas_zgbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TRMV
  {"cublasStrmv_v2",                 {"hipblasStrmv",                    "rocblas_strmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    "rocblas_dtrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    "rocblas_ctrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    "rocblas_ztrmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TBMV
  {"cublasStbmv_v2",                 {"hipblasStbmv",                    "rocblas_stbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    "rocblas_dtbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    "rocblas_ctbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    "rocblas_ztbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TPMV
  {"cublasStpmv_v2",                 {"hipblasStpmv",                    "rocblas_stpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    "rocblas_dtpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    "rocblas_ctpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    "rocblas_ztpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TRSV
  {"cublasStrsv_v2",                 {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    "rocblas_ctrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    "rocblas_ztrsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TPSV
  {"cublasStpsv_v2",                 {"hipblasStpsv",                    "rocblas_stpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    "rocblas_dtpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    "rocblas_ctpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    "rocblas_ztpsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // TBSV
  {"cublasStbsv_v2",                 {"hipblasStbsv",                    "rocblas_stbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    "rocblas_dtbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    "rocblas_ctbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    "rocblas_ztbsv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SYMV/HEMV
  {"cublasSsymv_v2",                 {"hipblasSsymv",                    "rocblas_ssymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsymv_v2",                 {"hipblasDsymv",                    "rocblas_dsymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsymv_v2",                 {"hipblasCsymv",                    "rocblas_csymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsymv_v2",                 {"hipblasZsymv",                    "rocblas_zsymv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChemv_v2",                 {"hipblasChemv",                    "rocblas_chemv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhemv_v2",                 {"hipblasZhemv",                    "rocblas_zhemv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SBMV/HBMV
  {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    "rocblas_ssbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsbmv_v2",                 {"hipblasDsbmv",                    "rocblas_dsbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChbmv_v2",                 {"hipblasChbmv",                    "rocblas_chbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    "rocblas_zhbmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SPMV/HPMV
  {"cublasSspmv_v2",                 {"hipblasSspmv",                    "rocblas_sspmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspmv_v2",                 {"hipblasDspmv",                    "rocblas_dspmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpmv_v2",                 {"hipblasChpmv",                    "rocblas_chpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    "rocblas_zhpmv",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // GER
  {"cublasSger_v2",                  {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDger_v2",                  {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgeru_v2",                 {"hipblasCgeru",                    "rocblas_cgeru",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCgerc_v2",                 {"hipblasCgerc",                    "rocblas_cgerc",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgeru_v2",                 {"hipblasZgeru",                    "rocblas_zgeru",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZgerc_v2",                 {"hipblasZgerc",                    "rocblas_zgerc",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SYR/HER
  {"cublasSsyr_v2",                  {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsyr_v2",                  {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsyr_v2",                  {"hipblasCsyr",                     "rocblas_csyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsyr_v2",                  {"hipblasZsyr",                     "rocblas_zsyr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCher_v2",                  {"hipblasCher",                     "rocblas_cher",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZher_v2",                  {"hipblasZher",                     "rocblas_zher",                             CONV_LIB_FUNC, API_BLAS, 6}},

  // SPR/HPR
  {"cublasSspr_v2",                  {"hipblasSspr",                     "rocblas_sspr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspr_v2",                  {"hipblasDspr",                     "rocblas_dspr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpr_v2",                  {"hipblasChpr",                     "rocblas_chpr",                             CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpr_v2",                  {"hipblasZhpr",                     "rocblas_zhpr",                             CONV_LIB_FUNC, API_BLAS, 6}},

  // SYR2/HER2
  {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    "rocblas_ssyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    "rocblas_dsyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    "rocblas_csyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    "rocblas_zsyr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasCher2_v2",                 {"hipblasCher2",                    "rocblas_cher2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZher2_v2",                 {"hipblasZher2",                    "rocblas_zher2",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // SPR2/HPR2
  {"cublasSspr2_v2",                 {"hipblasSspr2",                    "rocblas_sspr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasDspr2_v2",                 {"hipblasDspr2",                    "rocblas_dspr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasChpr2_v2",                 {"hipblasChpr2",                    "rocblas_chpr2",                            CONV_LIB_FUNC, API_BLAS, 6}},
  {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    "rocblas_zhpr2",                            CONV_LIB_FUNC, API_BLAS, 6}},

  // Blas3 (v2) Routines
  // GEMM
  {"cublasSgemm_v2",                 {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDgemm_v2",                 {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemm_v2",                 {"hipblasCgemm",                    "rocblas_cgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCgemm3m",                  {"hipblasCgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, 7, UNSUPPORTED}},
  {"cublasCgemm3mEx",                {"hipblasCgemm3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, 7, UNSUPPORTED}},
  {"cublasZgemm_v2",                 {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZgemm3m",                  {"hipblasZgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, 7, UNSUPPORTED}},

  //IO in FP16 / FP32, computation in float
  {"cublasSgemmEx",                  {"hipblasSgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasGemmEx",                   {"hipblasGemmEx",                   "rocblas_gemm_ex",                          CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasGemmBatchedEx",            {"hipblasGemmBatchedEx",            "rocblas_gemm_batched_ex",                  CONV_LIB_FUNC, API_BLAS, 8}},
  {"cublasGemmStridedBatchedEx",     {"hipblasGemmStridedBatchedEx",     "rocblas_gemm_strided_batched_ex",          CONV_LIB_FUNC, API_BLAS, 8}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCgemmEx",                  {"hipblasCgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasUint8gemmBias",            {"hipblasUint8gemmBias",            "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    "rocblas_ssyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    "rocblas_dsyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    "rocblas_csyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    "rocblas_zsyrk",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCsyrkEx",                  {"hipblasCsyrkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCsyrk3mEx",                {"hipblasCsyrk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  // HERK
  {"cublasCherk_v2",                 {"hipblasCherk",                    "rocblas_cherkx",                           CONV_LIB_FUNC, API_BLAS, 7}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCherkEx",                  {"hipblasCherkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCherk3mEx",                {"hipblasCherk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasZherk_v2",                 {"hipblasZherk",                    "rocblas_zherk",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // SYR2K
  {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   "rocblas_ssyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   "rocblas_dsyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   "rocblas_csyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   "rocblas_zsyr2k",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // HER2K
  {"cublasCher2k_v2",                {"hipblasCher2k",                   "rocblas_cher2k",                           CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZher2k_v2",                {"hipblasZher2k",                   "rocblas_zher2k",                           CONV_LIB_FUNC, API_BLAS, 7}},

  // SYMM
  {"cublasSsymm_v2",                 {"hipblasSsymm",                    "rocblas_ssymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDsymm_v2",                 {"hipblasDsymm",                    "rocblas_dsymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCsymm_v2",                 {"hipblasCsymm",                    "rocblas_csymm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZsymm_v2",                 {"hipblasZsymm",                    "rocblas_zsymm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // HEMM
  {"cublasChemm_v2",                 {"hipblasChemm",                    "rocblas_chemm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZhemm_v2",                 {"hipblasZhemm",                    "rocblas_zhemm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // TRSM
  {"cublasStrsm_v2",                 {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    "rocblas_ctrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    "rocblas_ztrsm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // TRMM
  {"cublasStrmm_v2",                 {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    "rocblas_ctrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},
  {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    "rocblas_ztrmm",                            CONV_LIB_FUNC, API_BLAS, 7}},

  // NRM2
  {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasScnrm2_v2",                {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDznrm2_v2",                {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // DOT
  {"cublasDotEx",                    {"hipblasDotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasDotcEx",                   {"hipblasDotcEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},

  {"cublasSdot_v2",                  {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDdot_v2",                  {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS, 5}},

  {"cublasCdotu_v2",                 {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCdotc_v2",                 {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdotu_v2",                 {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdotc_v2",                 {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // SCAL
  {"cublasScalEx",                   {"hipblasScalEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSscal_v2",                 {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDscal_v2",                 {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCscal_v2",                 {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCsscal_v2",                {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZscal_v2",                 {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdscal_v2",                {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // AXPY
  {"cublasAxpyEx",                   {"hipblasAxpyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // COPY
  {"cublasCopyEx",                   {"hipblasCopyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasScopy_v2",                 {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDcopy_v2",                 {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCcopy_v2",                 {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZcopy_v2",                 {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // SWAP
  {"cublasSwapEx",                   {"hipblasSwapEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSswap_v2",                 {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDswap_v2",                 {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCswap_v2",                 {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZswap_v2",                 {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // AMAX
  {"cublasIamaxEx",                  {"hipblasIamaxEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasIsamax_v2",                {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIdamax_v2",                {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIcamax_v2",                {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIzamax_v2",                {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // AMIN
  {"cublasIaminEx",                  {"hipblasIaminEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasIsamin_v2",                {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIdamin_v2",                {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIcamin_v2",                {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasIzamin_v2",                {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // ASUM
  {"cublasAsumEx",                   {"hipblasAsumEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSasum_v2",                 {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDasum_v2",                 {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasScasum_v2",                {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDzasum_v2",                {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS, 5}},

  // ROT
  {"cublasRotEx",                    {"hipblasRotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSrot_v2",                  {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrot_v2",                  {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCrot_v2",                  {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCsrot_v2",                 {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZrot_v2",                  {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZdrot_v2",                 {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTG
  {"cublasRotgEx",                   {"hipblasRotgEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSrotg_v2",                 {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotg_v2",                 {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasCrotg_v2",                 {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasZrotg_v2",                 {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTM
  {"cublasRotmEx",                   {"hipblasRotmEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSrotm_v2",                 {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotm_v2",                 {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS, 5}},

  // ROTMG
  {"cublasRotmgEx",                  {"hipblasRotmgEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, 8, UNSUPPORTED}},
  {"cublasSrotmg_v2",                {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS, 5}},
  {"cublasDrotmg_v2",                {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS, 5}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_BLAS_FUNCTION_VER_MAP {
  {"cublasGetMathMode",                          {CUDA_90,  CUDA_0, CUDA_0}},
  {"cublasMigrateComputeType",                   {CUDA_110, CUDA_0, CUDA_0}},
  {"cublasLogCallback",                          {CUDA_92,  CUDA_0, CUDA_0}},
  {"cublasLoggerConfigure",                      {CUDA_92,  CUDA_0, CUDA_0}},
  {"cublasSetLoggerCallback",                    {CUDA_92,  CUDA_0, CUDA_0}},
  {"cublasGetLoggerCallback",                    {CUDA_92,  CUDA_0, CUDA_0}},
  {"cublasGetCudartVersion",                     {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasNrm2Ex",                               {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasHgemm",                                {CUDA_75,  CUDA_0, CUDA_0}},
  {"cublasHgemmBatched",                         {CUDA_90,  CUDA_0, CUDA_0}},
  {"cublasSgemmStridedBatched",                  {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasDgemmStridedBatched",                  {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCgemm3mBatched",                       {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCgemmStridedBatched",                  {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCgemm3mStridedBatched",                {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasZgemmStridedBatched",                  {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasHgemmStridedBatched",                  {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCgemm3m",                              {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCgemm3mEx",                            {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasZgemm3m",                              {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasSgemmEx",                              {CUDA_75,  CUDA_0, CUDA_0}},
  {"cublasGemmEx",                               {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasGemmBatchedEx",                        {CUDA_91,  CUDA_0, CUDA_0}},
  {"cublasGemmStridedBatchedEx",                 {CUDA_91,  CUDA_0, CUDA_0}},
  {"cublasCgemmEx",                              {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasUint8gemmBias",                        {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCsyrkEx",                              {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCsyrk3mEx",                            {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCherkEx",                              {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCherk3mEx",                            {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasDotEx",                                {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasDotcEx",                               {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasScalEx",                               {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasAxpyEx",                               {CUDA_80,  CUDA_0, CUDA_0}},
  {"cublasCopyEx",                               {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasSwapEx",                               {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasIamaxEx",                              {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasIaminEx",                              {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasAsumEx",                               {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasRotEx",                                {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasRotgEx",                               {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasRotmEx",                               {CUDA_101, CUDA_0, CUDA_0}},
  {"cublasRotmgEx",                              {CUDA_101, CUDA_0, CUDA_0}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_BLAS_API_SECTION_MAP {
  {2, "CUBLAS Data types"},
  {3, "CUDA Datatypes Reference"},
  {4, "CUBLAS Helper Function Reference"},
  {5, "CUBLAS Level-1 Function Reference"},
  {6, "CUBLAS Level-2 Function Reference"},
  {7, "CUBLAS Level-3 Function Reference"},
  {8, "BLAS-like Extension"},
};
