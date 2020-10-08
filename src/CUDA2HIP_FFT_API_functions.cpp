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
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_FUNCTION_MAP {
  {"cufftPlan1d",                                         {"hipfftPlan1d",                                         "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftPlan2d",                                         {"hipfftPlan2d",                                         "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftPlan3d",                                         {"hipfftPlan3d",                                         "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftPlanMany",                                       {"hipfftPlanMany",                                       "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftMakePlan1d",                                     {"hipfftMakePlan1d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftMakePlan2d",                                     {"hipfftMakePlan2d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftMakePlan3d",                                     {"hipfftMakePlan3d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftMakePlanMany",                                   {"hipfftMakePlanMany",                                   "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftMakePlanMany64",                                 {"hipfftMakePlanMany64",                                 "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSizeMany64",                                  {"hipfftGetSizeMany64",                                  "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftEstimate1d",                                     {"hipfftEstimate1d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftEstimate2d",                                     {"hipfftEstimate2d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftEstimate3d",                                     {"hipfftEstimate3d",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftEstimateMany",                                   {"hipfftEstimateMany",                                   "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftCreate",                                         {"hipfftCreate",                                         "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSize1d",                                      {"hipfftGetSize1d",                                      "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSize2d",                                      {"hipfftGetSize2d",                                      "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSize3d",                                      {"hipfftGetSize3d",                                      "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSizeMany",                                    {"hipfftGetSizeMany",                                    "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetSize",                                        {"hipfftGetSize",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftSetWorkArea",                                    {"hipfftSetWorkArea",                                    "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftSetAutoAllocation",                              {"hipfftSetAutoAllocation",                              "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecC2C",                                        {"hipfftExecC2C",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecR2C",                                        {"hipfftExecR2C",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecC2R",                                        {"hipfftExecC2R",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecZ2Z",                                        {"hipfftExecZ2Z",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecD2Z",                                        {"hipfftExecD2Z",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftExecZ2D",                                        {"hipfftExecZ2D",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftSetStream",                                      {"hipfftSetStream",                                      "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftDestroy",                                        {"hipfftDestroy",                                        "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetVersion",                                     {"hipfftGetVersion",                                     "", CONV_LIB_FUNC, API_FFT, 2}},
  {"cufftGetProperty",                                    {"hipfftGetProperty",                                    "", CONV_LIB_FUNC, API_FFT, 2}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_FUNCTION_VER_MAP {
  {"cufftMakePlanMany64",                                 {CUDA_75,  CUDA_0, CUDA_0}},
  {"cufftGetSizeMany64",                                  {CUDA_75,  CUDA_0, CUDA_0}},
  {"cufftGetProperty",                                    {CUDA_80,  CUDA_0, CUDA_0}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_FFT_API_SECTION_MAP {
  {1, "CUFFT Data types"},
  {2, "CUFFT API functions"},
};
