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
  {"cublasInit",                     {"hipblasInit",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasShutdown",                 {"hipblasShutdown",                 "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetVersion",               {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetError",                 {"hipblasGetError",                 "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasAlloc",                    {"hipblasAlloc",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasFree",                     {"hipblasFree",                     "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetKernelStream",          {"hipblasSetKernelStream",          "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetAtomicsMode",           {"hipblasGetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetAtomicsMode",           {"hipblasSetAtomicsMode",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetMathMode",              {"hipblasGetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetMathMode",              {"hipblasSetMathMode",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasMigrateComputeType",       {"hipblasMigrateComputeType",       "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas logging
  {"cublasLogCallback",              {"hipblasLogCallback",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasLoggerConfigure",          {"hipblasLoggerConfigure",          "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetLoggerCallback",        {"hipblasSetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetLoggerCallback",        {"hipblasGetLoggerCallback",        "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas1 (v1) Routines
  {"cublasCreate",                   {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy",                  {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetStream",                {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream",                {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode",           {"hipblasSetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode",           {"hipblasGetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVector",                {"hipblasSetVector",                "rocblas_set_vector",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVector",                {"hipblasGetVector",                "rocblas_get_vector",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetVectorAsync",           {"hipblasSetVectorAsync",           "rocblas_set_vector_async",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVectorAsync",           {"hipblasGetVectorAsync",           "rocblas_get_vector_async",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetMatrix",                {"hipblasSetMatrix",                "rocblas_set_matrix",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetMatrix",                {"hipblasGetMatrix",                "rocblas_get_matrix",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetMatrixAsync",           {"hipblasSetMatrixAsync",           "rocblas_set_matrix_async",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetMatrixAsync",           {"hipblasGetMatrixAsync",           "rocblas_get_matrix_async",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasXerbla",                   {"hipblasXerbla",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // NRM2
  {"cublasSnrm2",                    {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2",                    {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2",                   {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDznrm2",                   {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasNrm2Ex",                   {"hipblasNrm2Ex",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // DOT
  {"cublasSdot",                     {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot",                     {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotu",                    {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotc",                    {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotu",                    {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotc",                    {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS}},

  // SCAL
  {"cublasSscal",                    {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal",                    {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal",                    {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsscal",                   {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZscal",                    {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdscal",                   {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS}},

  // AXPY
  {"cublasSaxpy",                    {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy",                    {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy",                    {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZaxpy",                    {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS}},

  // COPY
  {"cublasScopy",                    {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy",                    {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy",                    {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZcopy",                    {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS}},

  // SWAP
  {"cublasSswap",                    {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDswap",                    {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCswap",                    {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZswap",                    {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS}},

  // AMAX
  {"cublasIsamax",                   {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax",                   {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax",                   {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamax",                   {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS}},

  // AMIN
  {"cublasIsamin",                   {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamin",                   {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamin",                   {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamin",                   {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS}},

  // ASUM
  {"cublasSasum",                    {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum",                    {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum",                   {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDzasum",                   {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS}},

  // ROT
  {"cublasSrot",                     {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrot",                     {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrot",                     {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsrot",                    {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrot",                     {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdrot",                    {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTG
  {"cublasSrotg",                    {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotg",                    {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrotg",                    {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrotg",                    {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTM
  {"cublasSrotm",                    {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotm",                    {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTMG
  {"cublasSrotmg",                   {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotmg",                   {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS}},

  // GEMV
  {"cublasSgemv",                    {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS}},
  // NOTE: there is no such a function in CUDA
  {"cublasSgemvBatched",             {"hipblasSgemvBatched",             "rocblas_sgemv_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemv",                    {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv",                    {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemv",                    {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS}},

  // GBMV
  {"cublasSgbmv",                    {"hipblasSgbmv",                    "rocblas_sgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgbmv",                    {"hipblasDgbmv",                    "rocblas_dgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgbmv",                    {"hipblasCgbmv",                    "rocblas_cgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgbmv",                    {"hipblasZgbmv",                    "rocblas_zgbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TRMV
  {"cublasStrmv",                    {"hipblasStrmv",                    "rocblas_strmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrmv",                    {"hipblasDtrmv",                    "rocblas_dtrmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrmv",                    {"hipblasCtrmv",                    "rocblas_ctrmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrmv",                    {"hipblasZtrmv",                    "rocblas_ztrmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TBMV
  {"cublasStbmv",                    {"hipblasStbmv",                    "rocblas_stbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtbmv",                    {"hipblasDtbmv",                    "rocblas_dtbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtbmv",                    {"hipblasCtbmv",                    "rocblas_ctbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtbmv",                    {"hipblasZtbmv",                    "rocblas_ztbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TPMV
  {"cublasStpmv",                    {"hipblasStpmv",                    "rocblas_stpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtpmv",                    {"hipblasDtpmv",                    "rocblas_dtpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtpmv",                    {"hipblasCtpmv",                    "rocblas_ctpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtpmv",                    {"hipblasZtpmv",                    "rocblas_ztpmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TRSV
  {"cublasStrsv",                    {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsv",                    {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsv",                    {"hipblasCtrsv",                    "rocblas_ctrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrsv",                    {"hipblasZtrsv",                    "rocblas_ztrsv",                            CONV_LIB_FUNC, API_BLAS}},

  // TPSV
  {"cublasStpsv",                    {"hipblasStpsv",                    "rocblas_stpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtpsv",                    {"hipblasDtpsv",                    "rocblas_dtpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtpsv",                    {"hipblasCtpsv",                    "rocblas_ctpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtpsv",                    {"hipblasZtpsv",                    "rocblas_ztpsv",                            CONV_LIB_FUNC, API_BLAS}},

  // TBSV
  {"cublasStbsv",                    {"hipblasStbsv",                    "rocblas_stbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtbsv",                    {"hipblasDtbsv",                    "rocblas_dtbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtbsv",                    {"hipblasCtbsv",                    "rocblas_ctbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtbsv",                    {"hipblasZtbsv",                    "rocblas_ztbsv",                            CONV_LIB_FUNC, API_BLAS}},

  // SYMV/HEMV
  {"cublasSsymv",                    {"hipblasSsymv",                    "rocblas_ssymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsymv",                    {"hipblasDsymv",                    "rocblas_dsymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsymv",                    {"hipblasCsymv",                    "rocblas_csymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsymv",                    {"hipblasZsymv",                    "rocblas_zsymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChemv",                    {"hipblasChemv",                    "rocblas_chemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhemv",                    {"hipblasZhemv",                    "rocblas_zhemv",                            CONV_LIB_FUNC, API_BLAS}},

  // SBMV/HBMV
  {"cublasSsbmv",                    {"hipblasSsbmv",                    "rocblas_ssbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsbmv",                    {"hipblasDsbmv",                    "rocblas_dsbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChbmv",                    {"hipblasChbmv",                    "rocblas_chbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhbmv",                    {"hipblasZhbmv",                    "rocblas_zhbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // SPMV/HPMV
  {"cublasSspmv",                    {"hipblasSspmv",                    "rocblas_sspmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspmv",                    {"hipblasDspmv",                    "rocblas_dspmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpmv",                    {"hipblasChpmv",                    "rocblas_chpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpmv",                    {"hipblasZhpmv",                    "rocblas_zhpmv",                            CONV_LIB_FUNC, API_BLAS}},

  // GER
  {"cublasSger",                     {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger",                     {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru",                    {"hipblasCgeru",                    "rocblas_cgeru",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgerc",                    {"hipblasCgerc",                    "rocblas_cgerc",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgeru",                    {"hipblasZgeru",                    "rocblas_zgeru",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgerc",                    {"hipblasZgerc",                    "rocblas_zgerc",                            CONV_LIB_FUNC, API_BLAS}},

  // SYR/HER
  {"cublasSsyr",                     {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr",                     {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr",                     {"hipblasCsyr",                     "rocblas_csyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr",                     {"hipblasZsyr",                     "rocblas_zsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCher",                     {"hipblasCher",                     "rocblas_cher",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher",                     {"hipblasZher",                     "rocblas_zher",                             CONV_LIB_FUNC, API_BLAS}},

  // SPR/HPR
  {"cublasSspr",                     {"hipblasSspr",                     "rocblas_sspr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspr",                     {"hipblasDspr",                     "rocblas_dspr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpr",                     {"hipblasChpr",                     "rocblas_chpr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpr",                     {"hipblasZhpr",                     "rocblas_zhpr",                             CONV_LIB_FUNC, API_BLAS}},

  // SYR2/HER2
  {"cublasSsyr2",                    {"hipblasSsyr2",                    "rocblas_ssyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr2",                    {"hipblasDsyr2",                    "rocblas_dsyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr2",                    {"hipblasCsyr2",                    "rocblas_csyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr2",                    {"hipblasZsyr2",                    "rocblas_zsyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCher2",                    {"hipblasCher2",                    "rocblas_cher2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher2",                    {"hipblasZher2",                    "rocblas_zher2",                            CONV_LIB_FUNC, API_BLAS}},

  // SPR2/HPR2
  {"cublasSspr2",                    {"hipblasSspr2",                    "rocblas_sspr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspr2",                    {"hipblasDspr2",                    "rocblas_dspr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpr2",                    {"hipblasChpr2",                    "rocblas_chpr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpr2",                    {"hipblasZhpr2",                    "rocblas_zhpr2",                            CONV_LIB_FUNC, API_BLAS}},

  // Blas3 (v1) Routines
  // GEMM
  {"cublasSgemm",                    {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm",                    {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm",                    {"hipblasCgemm",                    "rocblas_cgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemm",                    {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemm",                    {"hipblasHgemm",                    "rocblas_hgemm",                            CONV_LIB_FUNC, API_BLAS}},

  // BATCH GEMM
  {"cublasSgemmBatched",             {"hipblasSgemmBatched",             "rocblas_sgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemmBatched",             {"hipblasDgemmBatched",             "rocblas_dgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemmBatched",             {"hipblasHgemmBatched",             "rocblas_hgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasSgemmStridedBatched",      {"hipblasSgemmStridedBatched",      "rocblas_sgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemmStridedBatched",      {"hipblasDgemmStridedBatched",      "rocblas_dgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemmBatched",             {"hipblasCgemmBatched",             "rocblas_cgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3mBatched",           {"hipblasCgemm3mBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemmBatched",             {"hipblasZgemmBatched",             "rocblas_zgemm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemmStridedBatched",      {"hipblasCgemmStridedBatched",      "rocblas_cgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3mStridedBatched",    {"hipblasCgemm3mStridedBatched",    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemmStridedBatched",      {"hipblasZgemmStridedBatched",      "rocblas_zgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},
  {"cublasHgemmStridedBatched",      {"hipblasHgemmStridedBatched",      "rocblas_hgemm_strided_batched",            CONV_LIB_FUNC, API_BLAS}},

  // SYRK
  {"cublasSsyrk",                    {"hipblasSsyrk",                    "rocblas_ssyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyrk",                    {"hipblasDsyrk",                    "rocblas_dsyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyrk",                    {"hipblasCsyrk",                    "rocblas_csyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyrk",                    {"hipblasZsyrk",                    "rocblas_zsyrk",                            CONV_LIB_FUNC, API_BLAS}},

  // HERK
  {"cublasCherk",                    {"hipblasCherk",                    "rocblas_cherk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZherk",                    {"hipblasZherk",                    "rocblas_zherk",                            CONV_LIB_FUNC, API_BLAS}},

  // SYR2K
  {"cublasSsyr2k",                   {"hipblasSsyr2k",                   "rocblas_ssyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr2k",                   {"hipblasDsyr2k",                   "rocblas_dsyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr2k",                   {"hipblasCsyr2k",                   "rocblas_csyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr2k",                   {"hipblasZsyr2k",                   "rocblas_zsyr2k",                           CONV_LIB_FUNC, API_BLAS}},

  // SYRKX - eXtended SYRK
  {"cublasSsyrkx",                   {"hipblasSsyrkx",                   "rocblas_ssyrkx",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyrkx",                   {"hipblasDsyrkx",                   "rocblas_dsyrkx",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyrkx",                   {"hipblasCsyrkx",                   "rocblas_csyrkx",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyrkx",                   {"hipblasZsyrkx",                   "rocblas_zsyrkx",                           CONV_LIB_FUNC, API_BLAS}},

  // HER2K
  {"cublasCher2k",                   {"hipblasCher2k",                   "rocblas_cher2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher2k",                   {"hipblasZher2k",                   "rocblas_zher2k",                           CONV_LIB_FUNC, API_BLAS}},

  // HERKX - eXtended HERK
  {"cublasCherkx",                   {"hipblasCherkx",                   "rocblas_cherkx",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZherkx",                   {"hipblasZherkx",                   "rocblas_zherkx",                           CONV_LIB_FUNC, API_BLAS}},

  // SYMM
  {"cublasSsymm",                    {"hipblasSsymm",                    "rocblas_ssymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsymm",                    {"hipblasDsymm",                    "rocblas_dsymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsymm",                    {"hipblasCsymm",                    "rocblas_csymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsymm",                    {"hipblasZsymm",                    "rocblas_zsymm",                            CONV_LIB_FUNC, API_BLAS}},

  // HEMM
  {"cublasChemm",                    {"hipblasChemm",                    "rocblas_chemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhemm",                    {"hipblasZhemm",                    "rocblas_zhemm",                            CONV_LIB_FUNC, API_BLAS}},

  // TRSM
  {"cublasStrsm",                    {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm",                    {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm",                    {"hipblasCtrsm",                    "rocblas_ctrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrsm",                    {"hipblasZtrsm",                    "rocblas_ztrsm",                            CONV_LIB_FUNC, API_BLAS}},

  // TRMM
  {"cublasStrmm",                    {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrmm",                    {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrmm",                    {"hipblasCtrmm",                    "rocblas_ctrmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrmm",                    {"hipblasZtrmm",                    "rocblas_ztrmm",                            CONV_LIB_FUNC, API_BLAS}},

  // ------------------------ CUBLAS BLAS - like extension (cublas_api.h)
  // GEAM
  {"cublasSgeam",                    {"hipblasSgeam",                    "rocblas_sgeam",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgeam",                    {"hipblasDgeam",                    "rocblas_dgeam",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeam",                    {"hipblasCgeam",                    "rocblas_cgeam",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgeam",                    {"hipblasZgeam",                    "rocblas_zgeam",                            CONV_LIB_FUNC, API_BLAS}},

  // GETRF - Batched LU
  {"cublasSgetrfBatched",            {"hipblasSgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasDgetrfBatched",            {"hipblasDgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasCgetrfBatched",            {"hipblasCgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasZgetrfBatched",            {"hipblasZgetrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},

  // Batched inversion based on LU factorization from getrf
  {"cublasSgetriBatched",            {"hipblasSgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasDgetriBatched",            {"hipblasDgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasCgetriBatched",            {"hipblasCgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasZgetriBatched",            {"hipblasZgetriBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},

  // Batched solver based on LU factorization from getrf
  {"cublasSgetrsBatched",            {"hipblasSgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasDgetrsBatched",            {"hipblasDgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasCgetrsBatched",            {"hipblasCgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasZgetrsBatched",            {"hipblasZgetrsBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},

  // TRSM - Batched Triangular Solver
  {"cublasStrsmBatched",             {"hipblasStrsmBatched",             "rocblas_strsm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsmBatched",             {"hipblasDtrsmBatched",             "rocblas_dtrsm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsmBatched",             {"hipblasCtrsmBatched",             "rocblas_ctrsm_batched",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrsmBatched",             {"hipblasZtrsmBatched",             "rocblas_ztrsm_batched",                    CONV_LIB_FUNC, API_BLAS}},

  // MATINV - Batched
  {"cublasSmatinvBatched",           {"hipblasSmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDmatinvBatched",           {"hipblasDmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCmatinvBatched",           {"hipblasCmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZmatinvBatched",           {"hipblasZmatinvBatched",           "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Batch QR Factorization
  {"cublasSgeqrfBatched",            {"hipblasSgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasDgeqrfBatched",            {"hipblasDgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasCgeqrfBatched",            {"hipblasCgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},
  {"cublasZgeqrfBatched",            {"hipblasZgeqrfBatched",            "",                                         CONV_LIB_FUNC, API_BLAS, ROC_UNSUPPORTED}},

  // Least Square Min only m >= n and Non-transpose supported
  {"cublasSgelsBatched",             {"hipblasSgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDgelsBatched",             {"hipblasDgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgelsBatched",             {"hipblasCgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgelsBatched",             {"hipblasZgelsBatched",             "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // DGMM
  {"cublasSdgmm",                    {"hipblasSdgmm",                    "rocblas_sdgmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdgmm",                    {"hipblasDdgmm",                    "rocblas_ddgmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdgmm",                    {"hipblasCdgmm",                    "rocblas_cdgmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdgmm",                    {"hipblasZdgmm",                    "rocblas_zdgmm",                            CONV_LIB_FUNC, API_BLAS}},

  // TPTTR - Triangular Pack format to Triangular format
  {"cublasStpttr",                   {"hipblasStpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtpttr",                   {"hipblasDtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtpttr",                   {"hipblasCtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtpttr",                   {"hipblasZtpttr",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // TRTTP - Triangular format to Triangular Pack format
  {"cublasStrttp",                   {"hipblasStrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDtrttp",                   {"hipblasDtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCtrttp",                   {"hipblasCtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZtrttp",                   {"hipblasZtrttp",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // Blas2 (v2) Routines
  {"cublasCreate_v2",                {"hipblasCreate",                   "rocblas_create_handle",                    CONV_LIB_FUNC, API_BLAS}},
  {"cublasDestroy_v2",               {"hipblasDestroy",                  "rocblas_destroy_handle",                   CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetVersion_v2",            {"hipblasGetVersion",               "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGetProperty",              {"hipblasGetProperty",              "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSetStream_v2",             {"hipblasSetStream",                "rocblas_set_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetStream_v2",             {"hipblasGetStream",                "rocblas_get_stream",                       CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetPointerMode_v2",        {"hipblasGetPointerMode",           "rocblas_set_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasSetPointerMode_v2",        {"hipblasSetPointerMode",           "rocblas_get_pointer_mode",                 CONV_LIB_FUNC, API_BLAS}},
  {"cublasGetCudartVersion",         {"hipblasGetCudartVersion",         "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // GEMV
  {"cublasSgemv_v2",                 {"hipblasSgemv",                    "rocblas_sgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemv_v2",                 {"hipblasDgemv",                    "rocblas_dgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemv_v2",                 {"hipblasCgemv",                    "rocblas_cgemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemv_v2",                 {"hipblasZgemv",                    "rocblas_zgemv",                            CONV_LIB_FUNC, API_BLAS}},

  // GBMV
  {"cublasSgbmv_v2",                 {"hipblasSgbmv",                    "rocblas_sgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgbmv_v2",                 {"hipblasDgbmv",                    "rocblas_dgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgbmv_v2",                 {"hipblasCgbmv",                    "rocblas_cgbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgbmv_v2",                 {"hipblasZgbmv",                    "rocblas_zgbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TRMV
  {"cublasStrmv_v2",                 {"hipblasStrmv",                    "rocblas_strmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrmv_v2",                 {"hipblasDtrmv",                    "rocblas_dtrmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrmv_v2",                 {"hipblasCtrmv",                    "rocblas_ctrmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrmv_v2",                 {"hipblasZtrmv",                    "rocblas_ztrmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TBMV
  {"cublasStbmv_v2",                 {"hipblasStbmv",                    "rocblas_stbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtbmv_v2",                 {"hipblasDtbmv",                    "rocblas_dtbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtbmv_v2",                 {"hipblasCtbmv",                    "rocblas_ctbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtbmv_v2",                 {"hipblasZtbmv",                    "rocblas_ztbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TPMV
  {"cublasStpmv_v2",                 {"hipblasStpmv",                    "rocblas_stpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtpmv_v2",                 {"hipblasDtpmv",                    "rocblas_dtpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtpmv_v2",                 {"hipblasCtpmv",                    "rocblas_ctpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtpmv_v2",                 {"hipblasZtpmv",                    "rocblas_ztpmv",                            CONV_LIB_FUNC, API_BLAS}},

  // TRSV
  {"cublasStrsv_v2",                 {"hipblasStrsv",                    "rocblas_strsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsv_v2",                 {"hipblasDtrsv",                    "rocblas_dtrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsv_v2",                 {"hipblasCtrsv",                    "rocblas_ctrsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrsv_v2",                 {"hipblasZtrsv",                    "rocblas_ztrsv",                            CONV_LIB_FUNC, API_BLAS}},

  // TPSV
  {"cublasStpsv_v2",                 {"hipblasStpsv",                    "rocblas_stpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtpsv_v2",                 {"hipblasDtpsv",                    "rocblas_dtpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtpsv_v2",                 {"hipblasCtpsv",                    "rocblas_ctpsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtpsv_v2",                 {"hipblasZtpsv",                    "rocblas_ztpsv",                            CONV_LIB_FUNC, API_BLAS}},

  // TBSV
  {"cublasStbsv_v2",                 {"hipblasStbsv",                    "rocblas_stbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtbsv_v2",                 {"hipblasDtbsv",                    "rocblas_dtbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtbsv_v2",                 {"hipblasCtbsv",                    "rocblas_ctbsv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtbsv_v2",                 {"hipblasZtbsv",                    "rocblas_ztbsv",                            CONV_LIB_FUNC, API_BLAS}},

  // SYMV/HEMV
  {"cublasSsymv_v2",                 {"hipblasSsymv",                    "rocblas_ssymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsymv_v2",                 {"hipblasDsymv",                    "rocblas_dsymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsymv_v2",                 {"hipblasCsymv",                    "rocblas_csymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsymv_v2",                 {"hipblasZsymv",                    "rocblas_zsymv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChemv_v2",                 {"hipblasChemv",                    "rocblas_chemv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhemv_v2",                 {"hipblasZhemv",                    "rocblas_zhemv",                            CONV_LIB_FUNC, API_BLAS}},

  // SBMV/HBMV
  {"cublasSsbmv_v2",                 {"hipblasSsbmv",                    "rocblas_ssbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsbmv_v2",                 {"hipblasDsbmv",                    "rocblas_dsbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChbmv_v2",                 {"hipblasChbmv",                    "rocblas_chbmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhbmv_v2",                 {"hipblasZhbmv",                    "rocblas_zhbmv",                            CONV_LIB_FUNC, API_BLAS}},

  // SPMV/HPMV
  {"cublasSspmv_v2",                 {"hipblasSspmv",                    "rocblas_sspmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspmv_v2",                 {"hipblasDspmv",                    "rocblas_dspmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpmv_v2",                 {"hipblasChpmv",                    "rocblas_chpmv",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpmv_v2",                 {"hipblasZhpmv",                    "rocblas_zhpmv",                            CONV_LIB_FUNC, API_BLAS}},

  // GER
  {"cublasSger_v2",                  {"hipblasSger",                     "rocblas_sger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDger_v2",                  {"hipblasDger",                     "rocblas_dger",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgeru_v2",                 {"hipblasCgeru",                    "rocblas_cgeru",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgerc_v2",                 {"hipblasCgerc",                    "rocblas_cgerc",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgeru_v2",                 {"hipblasZgeru",                    "rocblas_zgeru",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgerc_v2",                 {"hipblasZgerc",                    "rocblas_zgerc",                            CONV_LIB_FUNC, API_BLAS}},

  // SYR/HER
  {"cublasSsyr_v2",                  {"hipblasSsyr",                     "rocblas_ssyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr_v2",                  {"hipblasDsyr",                     "rocblas_dsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr_v2",                  {"hipblasCsyr",                     "rocblas_csyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr_v2",                  {"hipblasZsyr",                     "rocblas_zsyr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCher_v2",                  {"hipblasCher",                     "rocblas_cher",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher_v2",                  {"hipblasZher",                     "rocblas_zher",                             CONV_LIB_FUNC, API_BLAS}},

  // SPR/HPR
  {"cublasSspr_v2",                  {"hipblasSspr",                     "rocblas_sspr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspr_v2",                  {"hipblasDspr",                     "rocblas_dspr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpr_v2",                  {"hipblasChpr",                     "rocblas_chpr",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpr_v2",                  {"hipblasZhpr",                     "rocblas_zhpr",                             CONV_LIB_FUNC, API_BLAS}},

  // SYR2/HER2
  {"cublasSsyr2_v2",                 {"hipblasSsyr2",                    "rocblas_ssyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr2_v2",                 {"hipblasDsyr2",                    "rocblas_dsyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr2_v2",                 {"hipblasCsyr2",                    "rocblas_csyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr2_v2",                 {"hipblasZsyr2",                    "rocblas_zsyr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCher2_v2",                 {"hipblasCher2",                    "rocblas_cher2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher2_v2",                 {"hipblasZher2",                    "rocblas_zher2",                            CONV_LIB_FUNC, API_BLAS}},

  // SPR2/HPR2
  {"cublasSspr2_v2",                 {"hipblasSspr2",                    "rocblas_sspr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDspr2_v2",                 {"hipblasDspr2",                    "rocblas_dspr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasChpr2_v2",                 {"hipblasChpr2",                    "rocblas_chpr2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhpr2_v2",                 {"hipblasZhpr2",                    "rocblas_zhpr2",                            CONV_LIB_FUNC, API_BLAS}},

  // Blas3 (v2) Routines
  // GEMM
  {"cublasSgemm_v2",                 {"hipblasSgemm",                    "rocblas_sgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDgemm_v2",                 {"hipblasDgemm",                    "rocblas_dgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm_v2",                 {"hipblasCgemm",                    "rocblas_cgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCgemm3m",                  {"hipblasCgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasCgemm3mEx",                {"hipblasCgemm3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZgemm_v2",                 {"hipblasZgemm",                    "rocblas_zgemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZgemm3m",                  {"hipblasZgemm3m",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  //IO in FP16 / FP32, computation in float
  {"cublasSgemmEx",                  {"hipblasSgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasGemmEx",                   {"hipblasGemmEx",                   "rocblas_gemm_ex",                          CONV_LIB_FUNC, API_BLAS}},
  {"cublasGemmBatchedEx",            {"hipblasGemmBatchedEx",            "rocblas_gemm_batched_ex",                  CONV_LIB_FUNC, API_BLAS}},
  {"cublasGemmStridedBatchedEx",     {"hipblasGemmStridedBatchedEx",     "rocblas_gemm_strided_batched_ex",          CONV_LIB_FUNC, API_BLAS}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCgemmEx",                  {"hipblasCgemmEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasUint8gemmBias",            {"hipblasUint8gemmBias",            "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // SYRK
  {"cublasSsyrk_v2",                 {"hipblasSsyrk",                    "rocblas_ssyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyrk_v2",                 {"hipblasDsyrk",                    "rocblas_dsyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyrk_v2",                 {"hipblasCsyrk",                    "rocblas_csyrk",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyrk_v2",                 {"hipblasZsyrk",                    "rocblas_zsyrk",                            CONV_LIB_FUNC, API_BLAS}},

  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCsyrkEx",                  {"hipblasCsyrkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCsyrk3mEx",                {"hipblasCsyrk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  // HERK
  {"cublasCherk_v2",                 {"hipblasCherk",                    "rocblas_cherkx",                           CONV_LIB_FUNC, API_BLAS}},
  // IO in Int8 complex/cuComplex, computation in cuComplex
  {"cublasCherkEx",                  {"hipblasCherkEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  // IO in Int8 complex/cuComplex, computation in cuComplex, Gaussian math
  {"cublasCherk3mEx",                {"hipblasCherk3mEx",                "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasZherk_v2",                 {"hipblasZherk",                    "rocblas_zherk",                            CONV_LIB_FUNC, API_BLAS}},

  // SYR2K
  {"cublasSsyr2k_v2",                {"hipblasSsyr2k",                   "rocblas_ssyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsyr2k_v2",                {"hipblasDsyr2k",                   "rocblas_dsyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsyr2k_v2",                {"hipblasCsyr2k",                   "rocblas_csyr2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsyr2k_v2",                {"hipblasZsyr2k",                   "rocblas_zsyr2k",                           CONV_LIB_FUNC, API_BLAS}},

  // HER2K
  {"cublasCher2k_v2",                {"hipblasCher2k",                   "rocblas_cher2k",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZher2k_v2",                {"hipblasZher2k",                   "rocblas_zher2k",                           CONV_LIB_FUNC, API_BLAS}},

  // SYMM
  {"cublasSsymm_v2",                 {"hipblasSsymm",                    "rocblas_ssymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDsymm_v2",                 {"hipblasDsymm",                    "rocblas_dsymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsymm_v2",                 {"hipblasCsymm",                    "rocblas_csymm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZsymm_v2",                 {"hipblasZsymm",                    "rocblas_zsymm",                            CONV_LIB_FUNC, API_BLAS}},

  // HEMM
  {"cublasChemm_v2",                 {"hipblasChemm",                    "rocblas_chemm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZhemm_v2",                 {"hipblasZhemm",                    "rocblas_zhemm",                            CONV_LIB_FUNC, API_BLAS}},

  // TRSM
  {"cublasStrsm_v2",                 {"hipblasStrsm",                    "rocblas_strsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrsm_v2",                 {"hipblasDtrsm",                    "rocblas_dtrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrsm_v2",                 {"hipblasCtrsm",                    "rocblas_ctrsm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrsm_v2",                 {"hipblasZtrsm",                    "rocblas_ztrsm",                            CONV_LIB_FUNC, API_BLAS}},

  // TRMM
  {"cublasStrmm_v2",                 {"hipblasStrmm",                    "rocblas_strmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDtrmm_v2",                 {"hipblasDtrmm",                    "rocblas_dtrmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCtrmm_v2",                 {"hipblasCtrmm",                    "rocblas_ctrmm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZtrmm_v2",                 {"hipblasZtrmm",                    "rocblas_ztrmm",                            CONV_LIB_FUNC, API_BLAS}},

  // NRM2
  {"cublasSnrm2_v2",                 {"hipblasSnrm2",                    "rocblas_snrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDnrm2_v2",                 {"hipblasDnrm2",                    "rocblas_dnrm2",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScnrm2_v2",                {"hipblasScnrm2",                   "rocblas_scnrm2",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDznrm2_v2",                {"hipblasDznrm2",                   "rocblas_dznrm2",                           CONV_LIB_FUNC, API_BLAS}},

  // DOT
  {"cublasDotEx",                    {"hipblasDotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasDotcEx",                   {"hipblasDotcEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},

  {"cublasSdot_v2",                  {"hipblasSdot",                     "rocblas_sdot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDdot_v2",                  {"hipblasDdot",                     "rocblas_ddot",                             CONV_LIB_FUNC, API_BLAS}},

  {"cublasCdotu_v2",                 {"hipblasCdotu",                    "rocblas_cdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCdotc_v2",                 {"hipblasCdotc",                    "rocblas_cdotc",                            CONV_LIB_FUNC, API_BLAS,}},
  {"cublasZdotu_v2",                 {"hipblasZdotu",                    "rocblas_zdotu",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdotc_v2",                 {"hipblasZdotc",                    "rocblas_zdotc",                            CONV_LIB_FUNC, API_BLAS}},

  // SCAL
  {"cublasScalEx",                   {"hipblasScalEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSscal_v2",                 {"hipblasSscal",                    "rocblas_sscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDscal_v2",                 {"hipblasDscal",                    "rocblas_dscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCscal_v2",                 {"hipblasCscal",                    "rocblas_cscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsscal_v2",                {"hipblasCsscal",                   "rocblas_csscal",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasZscal_v2",                 {"hipblasZscal",                    "rocblas_zscal",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdscal_v2",                {"hipblasZdscal",                   "rocblas_zdscal",                           CONV_LIB_FUNC, API_BLAS}},

  // AXPY
  {"cublasAxpyEx",                   {"hipblasAxpyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSaxpy_v2",                 {"hipblasSaxpy",                    "rocblas_saxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDaxpy_v2",                 {"hipblasDaxpy",                    "rocblas_daxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCaxpy_v2",                 {"hipblasCaxpy",                    "rocblas_caxpy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZaxpy_v2",                 {"hipblasZaxpy",                    "rocblas_zaxpy",                            CONV_LIB_FUNC, API_BLAS}},

  // COPY
  {"cublasCopyEx",                   {"hipblasCopyEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasScopy_v2",                 {"hipblasScopy",                    "rocblas_scopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDcopy_v2",                 {"hipblasDcopy",                    "rocblas_dcopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCcopy_v2",                 {"hipblasCcopy",                    "rocblas_ccopy",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZcopy_v2",                 {"hipblasZcopy",                    "rocblas_zcopy",                            CONV_LIB_FUNC, API_BLAS}},

  // SWAP
  {"cublasSwapEx",                   {"hipblasSwapEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSswap_v2",                 {"hipblasSswap",                    "rocblas_sswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDswap_v2",                 {"hipblasDswap",                    "rocblas_dswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCswap_v2",                 {"hipblasCswap",                    "rocblas_cswap",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZswap_v2",                 {"hipblasZswap",                    "rocblas_zswap",                            CONV_LIB_FUNC, API_BLAS}},

  // AMAX
  {"cublasIamaxEx",                  {"hipblasIamaxEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasIsamax_v2",                {"hipblasIsamax",                   "rocblas_isamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamax_v2",                {"hipblasIdamax",                   "rocblas_idamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamax_v2",                {"hipblasIcamax",                   "rocblas_icamax",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamax_v2",                {"hipblasIzamax",                   "rocblas_izamax",                           CONV_LIB_FUNC, API_BLAS}},

  // AMIN
  {"cublasIaminEx",                  {"hipblasIaminEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasIsamin_v2",                {"hipblasIsamin",                   "rocblas_isamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIdamin_v2",                {"hipblasIdamin",                   "rocblas_idamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIcamin_v2",                {"hipblasIcamin",                   "rocblas_icamin",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasIzamin_v2",                {"hipblasIzamin",                   "rocblas_izamin",                           CONV_LIB_FUNC, API_BLAS}},

  // ASUM
  {"cublasAsumEx",                   {"hipblasAsumEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSasum_v2",                 {"hipblasSasum",                    "rocblas_sasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDasum_v2",                 {"hipblasDasum",                    "rocblas_dasum",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasScasum_v2",                {"hipblasScasum",                   "rocblas_scasum",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDzasum_v2",                {"hipblasDzasum",                   "rocblas_dzasum",                           CONV_LIB_FUNC, API_BLAS}},

  // ROT
  {"cublasRotEx",                    {"hipblasRotEx",                    "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrot_v2",                  {"hipblasSrot",                     "rocblas_srot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrot_v2",                  {"hipblasDrot",                     "rocblas_drot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrot_v2",                  {"hipblasCrot",                     "rocblas_crot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasCsrot_v2",                 {"hipblasCsrot",                    "rocblas_csrot",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrot_v2",                  {"hipblasZrot",                     "rocblas_zrot",                             CONV_LIB_FUNC, API_BLAS}},
  {"cublasZdrot_v2",                 {"hipblasZdrot",                    "rocblas_zdrot",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTG
  {"cublasRotgEx",                   {"hipblasRotgEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotg_v2",                 {"hipblasSrotg",                    "rocblas_srotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotg_v2",                 {"hipblasDrotg",                    "rocblas_drotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasCrotg_v2",                 {"hipblasCrotg",                    "rocblas_crotg",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasZrotg_v2",                 {"hipblasZrotg",                    "rocblas_zrotg",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTM
  {"cublasRotmEx",                   {"hipblasRotmEx",                   "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotm_v2",                 {"hipblasSrotm",                    "rocblas_srotm",                            CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotm_v2",                 {"hipblasDrotm",                    "rocblas_drotm",                            CONV_LIB_FUNC, API_BLAS}},

  // ROTMG
  {"cublasRotmgEx",                  {"hipblasRotmgEx",                  "",                                         CONV_LIB_FUNC, API_BLAS, UNSUPPORTED}},
  {"cublasSrotmg_v2",                {"hipblasSrotmg",                   "rocblas_srotmg",                           CONV_LIB_FUNC, API_BLAS}},
  {"cublasDrotmg_v2",                {"hipblasDrotmg",                   "rocblas_drotmg",                           CONV_LIB_FUNC, API_BLAS}},
};
