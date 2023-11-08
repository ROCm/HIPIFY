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
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SOLVER_FUNCTION_VER_MAP {
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
};
