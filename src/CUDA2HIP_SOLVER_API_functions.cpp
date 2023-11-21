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
const std::map<llvm::StringRef, hipCounter> CUDA_SOLVER_FUNCTION_MAP {
  {"cusolverDnCreate",                                    {"hipsolverDnCreate",                                    "rocblas_create_handle",                                          CONV_LIB_FUNC, API_SOLVER, 2, HIP_EXPERIMENTAL}},
  {"cusolverDnDestroy",                                   {"hipsolverDnDestroy",                                   "rocblas_destroy_handle",                                         CONV_LIB_FUNC, API_SOLVER, 2, HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnDgetrf -> rocsolver_dgetrf + harness of other API calls
  {"cusolverDnDgetrf",                                    {"hipsolverDnDgetrf",                                    "rocsolver_dgetrf",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnDgetrf_bufferSize -> rocsolver_dgetrf + harness of other API calls
  {"cusolverDnDgetrf_bufferSize",                         {"hipsolverDnDgetrf_bufferSize",                         "rocsolver_dgetrf",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnSgetrf -> rocsolver_sgetrf + harness of other API calls
  {"cusolverDnSgetrf",                                    {"hipsolverDnSgetrf",                                    "rocsolver_sgetrf",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnSgetrf_bufferSize -> rocsolver_sgetrf + harness of other API calls
  {"cusolverDnSgetrf_bufferSize",                         {"hipsolverDnSgetrf_bufferSize",                         "rocsolver_sgetrf",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnDgetrs -> rocsolver_dgetrs + harness of other API calls
  {"cusolverDnDgetrs",                                    {"hipsolverDnDgetrs",                                    "rocsolver_dgetrs",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  // [HIPIFY feature] TODO: cusolverDnSgetrs -> rocsolver_sgetrs + harness of other API calls
  {"cusolverDnSgetrs",                                    {"hipsolverDnSgetrs",                                    "rocsolver_sgetrs",                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED | HIP_EXPERIMENTAL}},
  {"cusolverDnXgetrf",                                    {"hipsolverDnXgetrf",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnXgetrf_bufferSize",                         {"hipsolverDnXgetrf_bufferSize",                         "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnXgetrs",                                    {"hipsolverDnXgetrs",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnCreateParams",                              {"hipsolverDnCreateParams",                              "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnSetAdvOptions",                             {"hipsolverDnSetAdvOptions",                             "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnSetStream",                                 {"hipsolverSetStream",                                   "rocblas_set_stream",                                             CONV_LIB_FUNC, API_SOLVER, 2, HIP_EXPERIMENTAL}},
  {"cusolverDnGetStream",                                 {"hipsolverGetStream",                                   "rocblas_get_stream",                                             CONV_LIB_FUNC, API_SOLVER, 2, HIP_EXPERIMENTAL}},
  {"cusolverDnSetDeterministicMode",                      {"hipsolverDnSetDeterministicMode",                      "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnGetDeterministicMode",                      {"hipsolverDnGetDeterministicMode",                      "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsCreate",                           {"hipsolverDnIRSParamsCreate",                           "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsDestroy",                          {"hipsolverDnIRSParamsDestroy",                          "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetRefinementSolver",              {"hipsolverDnIRSParamsSetRefinementSolver",              "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetSolverMainPrecision",           {"hipsolverDnIRSParamsSetSolverMainPrecision",           "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetSolverLowestPrecision",         {"hipsolverDnIRSParamsSetSolverLowestPrecision",         "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetSolverPrecisions",              {"hipsolverDnIRSParamsSetSolverPrecisions",              "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetTol",                           {"hipsolverDnIRSParamsSetTol",                           "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetTolInner",                      {"hipsolverDnIRSParamsSetTolInner",                      "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetMaxIters",                      {"hipsolverDnIRSParamsSetMaxIters",                      "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsSetMaxItersInner",                 {"hipsolverDnIRSParamsSetMaxItersInner",                 "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsGetMaxIters",                      {"hipsolverDnIRSParamsGetMaxIters",                      "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsEnableFallback",                   {"hipsolverDnIRSParamsEnableFallback",                   "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSParamsDisableFallback",                  {"hipsolverDnIRSParamsDisableFallback",                  "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosCreate",                            {"hipsolverDnIRSInfosCreate",                            "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosDestroy",                           {"hipsolverDnIRSInfosDestroy",                           "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosGetNiters",                         {"hipsolverDnIRSInfosGetNiters",                         "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosGetOuterNiters",                    {"hipsolverDnIRSInfosGetOuterNiters",                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosRequestResidual",                   {"hipsolverDnIRSInfosRequestResidual",                   "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosGetResidualHistory",                {"hipsolverDnIRSInfosGetResidualHistory",                "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnIRSInfosGetMaxIters",                       {"hipsolverDnIRSInfosGetMaxIters",                       "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  // NOTE: rocsolver_zgesv has a harness of rocblas_set_workspace, hipsolverZZgesv_bufferSize, and rocsolver_zgesv_outofplace
  {"cusolverDnZZgesv",                                    {"hipsolverDnZZgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED}},
  {"cusolverDnZCgesv",                                    {"hipsolverDnZCgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnZKgesv",                                    {"hipsolverDnZKgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnZEgesv",                                    {"hipsolverDnZEgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnZYgesv",                                    {"hipsolverDnZYgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  // NOTE: rocsolver_cgesv has a harness of rocblas_set_workspace, hipsolverCCgesv_bufferSize, and rocsolver_cgesv_outofplace
  {"cusolverDnCCgesv",                                    {"hipsolverDnCCgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED}},
  {"cusolverDnCEgesv",                                    {"hipsolverDnCEgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnCKgesv",                                    {"hipsolverDnCKgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnCYgesv",                                    {"hipsolverDnCYgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  // NOTE: rocsolver_dgesv has a harness of rocblas_set_workspace, hipsolverDDgesv_bufferSize, and rocsolver_dgesv_outofplace
  {"cusolverDnDDgesv",                                    {"hipsolverDnDDgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED}},
  {"cusolverDnDSgesv",                                    {"hipsolverDnDSgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnDHgesv",                                    {"hipsolverDnDHgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnDBgesv",                                    {"hipsolverDnDBgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnDXgesv",                                    {"hipsolverDnDXgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  // NOTE: rocsolver_sgesv has a harness of rocblas_set_workspace, hipsolverSSgesv_bufferSize, and rocsolver_sgesv_outofplace
  {"cusolverDnSSgesv",                                    {"hipsolverDnSSgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, ROC_UNSUPPORTED}},
  {"cusolverDnSHgesv",                                    {"hipsolverDnSHgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnSBgesv",                                    {"hipsolverDnSBgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
  {"cusolverDnSXgesv",                                    {"hipsolverDnSXgesv",                                    "",                                                               CONV_LIB_FUNC, API_SOLVER, 2, UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SOLVER_FUNCTION_VER_MAP {
  {"cusolverDnCreateParams",                              {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnSetAdvOptions",                             {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnXgetrf",                                    {CUDA_111,  CUDA_0, CUDA_0}},
  {"cusolverDnXgetrf_bufferSize",                         {CUDA_111,  CUDA_0, CUDA_0}},
  {"cusolverDnXgetrs",                                    {CUDA_111,  CUDA_0, CUDA_0}},
  {"cusolverDnSetDeterministicMode",                      {CUDA_122,  CUDA_0, CUDA_0}},
  {"cusolverDnGetDeterministicMode",                      {CUDA_122,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsCreate",                           {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsDestroy",                          {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetRefinementSolver",              {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetSolverMainPrecision",           {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetSolverLowestPrecision",         {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetSolverPrecisions",              {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetTol",                           {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetTolInner",                      {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetMaxIters",                      {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsSetMaxItersInner",                 {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsGetMaxIters",                      {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsEnableFallback",                   {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSParamsDisableFallback",                  {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosCreate",                            {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosDestroy",                           {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosGetNiters",                         {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosGetOuterNiters",                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosRequestResidual",                   {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosGetResidualHistory",                {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnIRSInfosGetMaxIters",                       {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnZZgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnZCgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnZKgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnZEgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnZYgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnCCgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnCEgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnCKgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnCYgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnDDgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnDSgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnDHgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnDBgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnDXgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnSSgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnSHgesv",                                    {CUDA_102,  CUDA_0, CUDA_0}},
  {"cusolverDnSBgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
  {"cusolverDnSXgesv",                                    {CUDA_110,  CUDA_0, CUDA_0}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_SOLVER_FUNCTION_VER_MAP {
  {"hipsolverDnCreate",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnDestroy",                                  {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnDgetrf",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnDgetrf_bufferSize",                        {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnSgetrf",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnSgetrf_bufferSize",                        {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnDgetrs",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnSgetrs",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverSetStream",                                  {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverGetStream",                                  {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnZZgesv",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnCCgesv",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnDDgesv",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverDnSSgesv",                                   {HIP_5010, HIP_0,    HIP_0,  HIP_LATEST}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_SOLVER_API_SECTION_MAP {
  {1, "CUSOLVER Data types"},
  {2, "CUSOLVER Function Reference"},
};
