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
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  m["cufftPlan1d"]                                      = {"hipfftPlan1d",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftPlan2d"]                                      = {"hipfftPlan2d",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftPlan3d"]                                      = {"hipfftPlan3d",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftPlanMany"]                                    = {"hipfftPlanMany",                                       "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftMakePlan1d"]                                  = {"hipfftMakePlan1d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftMakePlan2d"]                                  = {"hipfftMakePlan2d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftMakePlan3d"]                                  = {"hipfftMakePlan3d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftMakePlanMany"]                                = {"hipfftMakePlanMany",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftMakePlanMany64"]                              = {"hipfftMakePlanMany64",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSizeMany64"]                               = {"hipfftGetSizeMany64",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftEstimate1d"]                                  = {"hipfftEstimate1d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftEstimate2d"]                                  = {"hipfftEstimate2d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftEstimate3d"]                                  = {"hipfftEstimate3d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftEstimateMany"]                                = {"hipfftEstimateMany",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCreate"]                                      = {"hipfftCreate",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSize1d"]                                   = {"hipfftGetSize1d",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSize2d"]                                   = {"hipfftGetSize2d",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSize3d"]                                   = {"hipfftGetSize3d",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSizeMany"]                                 = {"hipfftGetSizeMany",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetSize"]                                     = {"hipfftGetSize",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftSetWorkArea"]                                 = {"hipfftSetWorkArea",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftSetAutoAllocation"]                           = {"hipfftSetAutoAllocation",                              "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecC2C"]                                     = {"hipfftExecC2C",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecR2C"]                                     = {"hipfftExecR2C",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecC2R"]                                     = {"hipfftExecC2R",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecZ2Z"]                                     = {"hipfftExecZ2Z",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecD2Z"]                                     = {"hipfftExecD2Z",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftExecZ2D"]                                     = {"hipfftExecZ2D",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftSetStream"]                                   = {"hipfftSetStream",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftDestroy"]                                     = {"hipfftDestroy",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetVersion"]                                  = {"hipfftGetVersion",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftGetProperty"]                                 = {"hipfftGetProperty",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtSetGPUs"]                                   = {"hipfftXtSetGPUs",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtMalloc"]                                    = {"hipfftXtMalloc",                                       "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtMemcpy"]                                    = {"hipfftXtMemcpy",                                       "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtFree"]                                      = {"hipfftXtFree",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtSetWorkArea"]                               = {"hipfftXtSetWorkArea",                                  "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftXtExecDescriptorC2C"]                         = {"hipfftXtExecDescriptorC2C",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptorR2C"]                         = {"hipfftXtExecDescriptorR2C",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptorC2R"]                         = {"hipfftXtExecDescriptorC2R",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptorZ2Z"]                         = {"hipfftXtExecDescriptorZ2Z",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptorD2Z"]                         = {"hipfftXtExecDescriptorD2Z",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptorZ2D"]                         = {"hipfftXtExecDescriptorZ2D",                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtQueryPlan"]                                 = {"hipfftXtQueryPlan",                                    "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftCallbackLoadC"]                               = {"hipfftCallbackLoadC",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackLoadZ"]                               = {"hipfftCallbackLoadZ",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackLoadR"]                               = {"hipfftCallbackLoadR",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackLoadD"]                               = {"hipfftCallbackLoadD",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackStoreC"]                              = {"hipfftCallbackStoreC",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackStoreZ"]                              = {"hipfftCallbackStoreZ",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackStoreR"]                              = {"hipfftCallbackStoreR",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftCallbackStoreD"]                              = {"hipfftCallbackStoreD",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtSetCallback"]                               = {"hipfftXtSetCallback",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtClearCallback"]                             = {"hipfftXtClearCallback",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtSetCallbackSharedSize"]                     = {"hipfftXtSetCallbackSharedSize",                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtMakePlanMany"]                              = {"hipfftXtMakePlanMany",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtGetSizeMany"]                               = {"hipfftXtGetSizeMany",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExec"]                                      = {"hipfftXtExec",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtExecDescriptor"]                            = {"hipfftXtExecDescriptor",                               "", CONV_LIB_FUNC, API_FFT, 2};
  m["cufftXtSetWorkAreaPolicy"]                         = {"hipfftXtSetWorkAreaPolicy",                            "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftXtSetDistribution"]                           = {"hipfftXtSetDistribution",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftSetPlanPropertyInt64"]                        = {"hipfftSetPlanPropertyInt64",                           "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftGetPlanPropertyInt64"]                        = {"hipfftGetPlanPropertyInt64",                           "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["cufftResetPlanProperty"]                           = {"hipfftResetPlanProperty",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_dft_1d"]                                 = {"fftw_plan_dft_1d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_2d"]                                 = {"fftw_plan_dft_2d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_3d"]                                 = {"fftw_plan_dft_3d",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft"]                                    = {"fftw_plan_dft",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_r2c_1d"]                             = {"fftw_plan_dft_r2c_1d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_r2c_2d"]                             = {"fftw_plan_dft_r2c_2d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_r2c_3d"]                             = {"fftw_plan_dft_r2c_3d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_r2c"]                                = {"fftw_plan_dft_r2c",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_c2r_1d"]                             = {"fftw_plan_dft_c2r_1d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_c2r_2d"]                             = {"fftw_plan_dft_c2r_2d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_c2r_3d"]                             = {"fftw_plan_dft_c2r_3d",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_dft_c2r"]                                = {"fftw_plan_dft_c2r",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_execute"]                                     = {"fftw_execute",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_1d"]                                = {"fftwf_plan_dft_1d",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_2d"]                                = {"fftwf_plan_dft_2d",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_3d"]                                = {"fftwf_plan_dft_3d",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft"]                                   = {"fftwf_plan_dft",                                       "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_r2c_1d"]                            = {"fftwf_plan_dft_r2c_1d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_r2c_2d"]                            = {"fftwf_plan_dft_r2c_2d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_r2c_3d"]                            = {"fftwf_plan_dft_r2c_3d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_r2c"]                               = {"fftwf_plan_dft_r2c",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_c2r_1d"]                            = {"fftwf_plan_dft_c2r_1d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_c2r_2d"]                            = {"fftwf_plan_dft_c2r_2d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_c2r_3d"]                            = {"fftwf_plan_dft_c2r_3d",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_dft_c2r"]                               = {"fftwf_plan_dft_c2r",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_execute"]                                    = {"fftwf_execute",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_print_plan"]                                  = {"fftw_print_plan",                                      "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_print_plan"]                                 = {"fftwf_print_plan",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_set_timelimit"]                               = {"fftw_set_timelimit",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_set_timelimit"]                              = {"fftwf_set_timelimit",                                  "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_cost"]                                        = {"fftw_cost",                                            "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_cost"]                                       = {"fftwf_cost",                                           "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_flops"]                                       = {"fftw_flops",                                           "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_flops"]                                      = {"fftwf_flops",                                          "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_destroy_plan"]                                = {"fftw_destroy_plan",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_destroy_plan"]                               = {"fftwf_destroy_plan",                                   "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_cleanup"]                                     = {"fftw_cleanup",                                         "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_cleanup"]                                    = {"fftwf_cleanup",                                        "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_plan_many_dft"]                               = {"fftw_plan_many_dft",                                   "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_many_dft_r2c"]                           = {"fftw_plan_many_dft_r2c",                               "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_many_dft_c2r"]                           = {"fftw_plan_many_dft_c2r",                               "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru_dft"]                               = {"fftw_plan_guru_dft",                                   "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru_dft_r2c"]                           = {"fftw_plan_guru_dft_r2c",                               "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru_dft_c2r"]                           = {"fftw_plan_guru_dft_c2r",                               "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru64_dft"]                             = {"fftw_plan_guru64_dft",                                 "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru64_dft_r2c"]                         = {"fftw_plan_guru64_dft_r2c",                             "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_plan_guru64_dft_c2r"]                         = {"fftw_plan_guru64_dft_c2r",                             "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_execute_dft"]                                 = {"fftw_execute_dft",                                     "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_execute_dft_r2c"]                             = {"fftw_execute_dft_r2c",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_execute_dft_c2r"]                             = {"fftw_execute_dft_c2r",                                 "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_plan_many_dft"]                              = {"fftwf_plan_many_dft",                                  "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_many_dft_r2c"]                          = {"fftwf_plan_many_dft_r2c",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_many_dft_c2r"]                          = {"fftwf_plan_many_dft_c2r",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru_dft"]                              = {"fftwf_plan_guru_dft",                                  "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru_dft_r2c"]                          = {"fftwf_plan_guru_dft_r2c",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru_dft_c2r"]                          = {"fftwf_plan_guru_dft_c2r",                              "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru64_dft"]                            = {"fftwf_plan_guru64_dft",                                "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru64_dft_r2c"]                        = {"fftwf_plan_guru64_dft_r2c",                            "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_plan_guru64_dft_c2r"]                        = {"fftwf_plan_guru64_dft_c2r",                            "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_execute_dft"]                                = {"fftwf_execute_dft",                                    "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_execute_dft_r2c"]                            = {"fftwf_execute_dft_r2c",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftwf_execute_dft_c2r"]                            = {"fftwf_execute_dft_c2r",                                "", CONV_LIB_FUNC, API_FFT, 2};
  m["fftw_export_wisdom_to_file"]                       = {"fftw_export_wisdom_to_file",                           "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_export_wisdom_to_file"]                      = {"fftwf_export_wisdom_to_file",                          "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftw_import_wisdom_from_file"]                     = {"fftw_import_wisdom_from_file",                         "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};
  m["fftwf_import_wisdom_from_file"]                    = {"fftwf_import_wisdom_from_file",                        "", CONV_LIB_FUNC, API_FFT, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cufftMakePlanMany64"]                                = {CUDA_75,  CUDA_0,   CUDA_0  };
  m["cufftGetSizeMany64"]                                 = {CUDA_75,  CUDA_0,   CUDA_0  };
  m["cufftGetProperty"]                                   = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtMakePlanMany"]                                = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtGetSizeMany"]                                 = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtExec"]                                        = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtExecDescriptor"]                              = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtSetWorkAreaPolicy"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cufftXtSetDistribution"]                             = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cufftSetPlanPropertyInt64"]                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cufftGetPlanPropertyInt64"]                          = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cufftResetPlanProperty"]                             = {CUDA_124, CUDA_0,   CUDA_0  };
  m["fftw_plan_guru64_dft"]                               = {CUDA_100, CUDA_0,   CUDA_0  };
  m["fftw_plan_guru64_dft_r2c"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["fftw_plan_guru64_dft_c2r"]                           = {CUDA_100, CUDA_0,   CUDA_0  };
  m["fftwf_plan_guru64_dft"]                              = {CUDA_100, CUDA_0,   CUDA_0  };
  m["fftwf_plan_guru64_dft_r2c"]                          = {CUDA_100, CUDA_0,   CUDA_0  };
  m["fftwf_plan_guru64_dft_c2r"]                          = {CUDA_100, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_FFT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipfftPlan1d"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftPlan2d"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftPlan3d"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftPlanMany"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftMakePlan1d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftMakePlan2d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftMakePlan3d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftMakePlanMany"]                                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftMakePlanMany64"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSizeMany64"]                                = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftEstimate1d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftEstimate2d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftEstimate3d"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftEstimateMany"]                                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftCreate"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSize1d"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSize2d"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSize3d"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSizeMany"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetSize"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftSetWorkArea"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftSetAutoAllocation"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecC2C"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecR2C"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecC2R"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecZ2Z"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecD2Z"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftExecZ2D"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftSetStream"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftDestroy"]                                      = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetVersion"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftGetProperty"]                                  = {HIP_2060, HIP_0,    HIP_0   };
  m["hipfftCallbackLoadC"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackLoadZ"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackLoadR"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackLoadD"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackStoreC"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackStoreZ"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackStoreR"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftCallbackStoreD"]                               = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftXtSetCallback"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftXtClearCallback"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftXtSetCallbackSharedSize"]                      = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftXtSetGPUs"]                                    = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtMalloc"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtMemcpy"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtFree"]                                       = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorC2C"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorR2C"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorC2R"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorZ2Z"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorD2Z"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptorZ2D"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtMakePlanMany"]                               = {HIP_5060, HIP_0,    HIP_0   };
  m["hipfftXtGetSizeMany"]                                = {HIP_5060, HIP_0,    HIP_0   };
  m["hipfftXtExec"]                                       = {HIP_5060, HIP_0,    HIP_0   };
  m["hipfftXtExecDescriptor"]                             = {HIP_6000, HIP_0,    HIP_0   };
  m["fftw_plan_dft_1d"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_2d"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_3d"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft"]                                      = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_r2c_1d"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_r2c_2d"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_r2c_3d"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_r2c"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_c2r_2d"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_c2r_3d"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan_dft_c2r"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_execute"]                                       = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_1d"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_2d"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_3d"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_r2c_1d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_r2c_2d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_r2c_3d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_r2c"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_c2r_1d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_c2r_2d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_c2r_3d"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan_dft_c2r"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_execute"]                                      = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_print_plan"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_print_plan"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_set_timelimit"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_set_timelimit"]                                = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_cost"]                                          = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_cost"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_flops"]                                         = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_flops"]                                        = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_destroy_plan"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_destroy_plan"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_cleanup"]                                       = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_cleanup"]                                      = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_execute_dft"]                                   = {HIP_7020, HIP_0,    HIP_0   };
  m["fftw_execute_dft_r2c"]                               = {HIP_7020, HIP_0,    HIP_0   };
  m["fftw_execute_dft_c2r"]                               = {HIP_7020, HIP_0,    HIP_0   };
  m["fftwf_execute_dft"]                                  = {HIP_7020, HIP_0,    HIP_0   };
  m["fftwf_execute_dft_r2c"]                              = {HIP_7020, HIP_0,    HIP_0   };
  m["fftwf_execute_dft_c2r"]                              = {HIP_7020, HIP_0,    HIP_0   };

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_FFT_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1] = "CUFFT Data types";
  m[2] = "CUFFT API functions";

  return m;
}();
