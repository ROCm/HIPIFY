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
const std::map<llvm::StringRef, hipCounter> CUDA_SOLVER_TYPE_NAME_MAP {
  {"cusolverStatus_t",                                         {"hipsolverStatus_t",                                         "rocblas_status",                                                   CONV_TYPE, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_SUCCESS",                                  {"HIPSOLVER_STATUS_SUCCESS",                                  "rocblas_status_success",                                           CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_NOT_INITIALIZED",                          {"HIPSOLVER_STATUS_NOT_INITIALIZED",                          "rocblas_status_invalid_handle",                                    CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_ALLOC_FAILED",                             {"HIPSOLVER_STATUS_ALLOC_FAILED",                             "rocblas_status_memory_error",                                      CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_INVALID_VALUE",                            {"HIPSOLVER_STATUS_INVALID_VALUE",                            "rocblas_status_invalid_value",                                     CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_ARCH_MISMATCH",                            {"HIPSOLVER_STATUS_ARCH_MISMATCH",                            "rocblas_status_arch_mismatch",                                     CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_MAPPING_ERROR",                            {"HIPSOLVER_STATUS_MAPPING_ERROR",                            "rocblas_status_not_implemented",                                   CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_EXECUTION_FAILED",                         {"HIPSOLVER_STATUS_EXECUTION_FAILED",                         "rocblas_status_not_implemented",                                   CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_INTERNAL_ERROR",                           {"HIPSOLVER_STATUS_INTERNAL_ERROR",                           "rocblas_status_internal_error",                                    CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED",                {"HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED",                "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_NOT_SUPPORTED",                            {"HIPSOLVER_STATUS_NOT_SUPPORTED",                            "rocblas_status_not_implemented",                                   CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_ZERO_PIVOT",                               {"HIPSOLVER_STATUS_ZERO_PIVOT",                               "rocblas_status_not_implemented",                                   CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_STATUS_INVALID_LICENSE",                          {"HIPSOLVER_STATUS_INVALID_LICENSE",                          "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED",               {"HIPSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED",               "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID",                       {"HIPSOLVER_STATUS_IRS_PARAMS_INVALID",                       "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC",                  {"HIPSOLVER_STATUS_IRS_PARAMS_INVALID_PREC",                  "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE",                {"HIPSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE",                "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER",               {"HIPSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER",               "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_INTERNAL_ERROR",                       {"HIPSOLVER_STATUS_IRS_INTERNAL_ERROR",                       "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_NOT_SUPPORTED",                        {"HIPSOLVER_STATUS_IRS_NOT_SUPPORTED",                        "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_OUT_OF_RANGE",                         {"HIPSOLVER_STATUS_IRS_OUT_OF_RANGE",                         "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",  {"HIPSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",  "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED",                {"HIPSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED",                "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED",                  {"HIPSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED",                  "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_IRS_MATRIX_SINGULAR",                      {"HIPSOLVER_STATUS_IRS_MATRIX_SINGULAR",                      "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"CUSOLVER_STATUS_INVALID_WORKSPACE",                        {"HIPSOLVER_STATUS_INVALID_WORKSPACE",                        "",                                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, UNSUPPORTED}},
  {"cusolverDnHandle_t",                                       {"hipsolverHandle_t",                                         "rocblas_handle",                                                   CONV_TYPE, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"cusolverEigType_t",                                        {"hipsolverEigType_t",                                        "rocblas_eform",                                                    CONV_TYPE, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_EIG_TYPE_1",                                      {"HIPSOLVER_EIG_TYPE_1",                                      "rocblas_eform_ax",                                                 CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_EIG_TYPE_2",                                      {"HIPSOLVER_EIG_TYPE_2",                                      "rocblas_eform_abx",                                                CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
  {"CUSOLVER_EIG_TYPE_3",                                      {"HIPSOLVER_EIG_TYPE_3",                                      "rocblas_eform_bax",                                                CONV_NUMERIC_LITERAL, API_SOLVER, 1, HIP_EXPERIMENTAL}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SOLVER_TYPE_NAME_VER_MAP {
  {"CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED",               {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID",                       {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC",                  {CUDA_110, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE",                {CUDA_110, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER",               {CUDA_110, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_INTERNAL_ERROR",                       {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_NOT_SUPPORTED",                        {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_OUT_OF_RANGE",                         {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",  {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED",                {CUDA_102, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED",                  {CUDA_110, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_IRS_MATRIX_SINGULAR",                      {CUDA_110, CUDA_0, CUDA_0}},
  {"CUSOLVER_STATUS_INVALID_WORKSPACE",                        {CUDA_110, CUDA_0, CUDA_0}},
  {"cusolverEigType_t",                                        {CUDA_80,  CUDA_0, CUDA_0}},
  {"CUSOLVER_EIG_TYPE_1",                                      {CUDA_80,  CUDA_0, CUDA_0}},
  {"CUSOLVER_EIG_TYPE_2",                                      {CUDA_80,  CUDA_0, CUDA_0}},
  {"CUSOLVER_EIG_TYPE_3",                                      {CUDA_80,  CUDA_0, CUDA_0}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_SOLVER_TYPE_NAME_VER_MAP {
  {"hipsolverStatus_t",                                        {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_SUCCESS",                                 {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_NOT_INITIALIZED",                         {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_ALLOC_FAILED",                            {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_INVALID_VALUE",                           {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_ARCH_MISMATCH",                           {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_MAPPING_ERROR",                           {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_EXECUTION_FAILED",                        {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_INTERNAL_ERROR",                          {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_STATUS_NOT_SUPPORTED",                           {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverHandle_t",                                        {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"hipsolverEigType_t",                                       {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_EIG_TYPE_1",                                     {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_EIG_TYPE_2",                                     {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"HIPSOLVER_EIG_TYPE_3",                                     {HIP_4050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status",                                           {HIP_3000, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_success",                                   {HIP_3000, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_invalid_handle",                            {HIP_5060, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_memory_error",                              {HIP_5060, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_invalid_value",                             {HIP_3050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_not_implemented",                           {HIP_1050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_internal_error",                            {HIP_1050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_status_arch_mismatch",                             {HIP_5070, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_handle",                                           {HIP_1050, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_eform",                                            {HIP_4020, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_eform_ax",                                         {HIP_4020, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_eform_abx",                                        {HIP_4020, HIP_0,    HIP_0,  HIP_LATEST}},
  {"rocblas_eform_bax",                                        {HIP_4020, HIP_0,    HIP_0,  HIP_LATEST}},
};
