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
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_TYPE_NAME_MAP = []() {
  std::map<llvm::StringRef, hipCounter> m;

  // cuFFT defines
  m["CUFFT_FORWARD"]                                    = {"HIPFFT_FORWARD",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1};  // -1
  m["CUFFT_INVERSE"]                                    = {"HIPFFT_BACKWARD",                                  "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  1
  m["CUFFT_COMPATIBILITY_DEFAULT"]                      = {"HIPFFT_COMPATIBILITY_DEFAULT",                     "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  CUFFT_COMPATIBILITY_FFTW_PADDING
  m["MAX_CUFFT_ERROR"]                                  = {"HIPFFT_MAX_ERROR",                                 "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x11

  // cuFFT enums
  m["cufftResult_t"]                                    = {"hipfftResult_t",                                   "", CONV_TYPE, API_FFT, 1};
  m["cufftResult"]                                      = {"hipfftResult",                                     "", CONV_TYPE, API_FFT, 1};
  m["CUFFT_SUCCESS"]                                    = {"HIPFFT_SUCCESS",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x0  0
  m["CUFFT_INVALID_PLAN"]                               = {"HIPFFT_INVALID_PLAN",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x1  1
  m["CUFFT_ALLOC_FAILED"]                               = {"HIPFFT_ALLOC_FAILED",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x2  2
  m["CUFFT_INVALID_TYPE"]                               = {"HIPFFT_INVALID_TYPE",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x3  3
  m["CUFFT_INVALID_VALUE"]                              = {"HIPFFT_INVALID_VALUE",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x4  4
  m["CUFFT_INTERNAL_ERROR"]                             = {"HIPFFT_INTERNAL_ERROR",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x5  5
  m["CUFFT_EXEC_FAILED"]                                = {"HIPFFT_EXEC_FAILED",                               "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x6  6
  m["CUFFT_SETUP_FAILED"]                               = {"HIPFFT_SETUP_FAILED",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x7  7
  m["CUFFT_INVALID_SIZE"]                               = {"HIPFFT_INVALID_SIZE",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x8  8
  m["CUFFT_UNALIGNED_DATA"]                             = {"HIPFFT_UNALIGNED_DATA",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x9  9
  m["CUFFT_INCOMPLETE_PARAMETER_LIST"]                  = {"HIPFFT_INCOMPLETE_PARAMETER_LIST",                 "", CONV_NUMERIC_LITERAL, API_FFT, 1, CUDA_REMOVED};  //  0xA  10
  m["CUFFT_INVALID_DEVICE"]                             = {"HIPFFT_INVALID_DEVICE",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0xB  11
  m["CUFFT_PARSE_ERROR"]                                = {"HIPFFT_PARSE_ERROR",                               "", CONV_NUMERIC_LITERAL, API_FFT, 1, CUDA_REMOVED};  //  0xC  12
  m["CUFFT_NO_WORKSPACE"]                               = {"HIPFFT_NO_WORKSPACE",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0xD  13
  m["CUFFT_NOT_IMPLEMENTED"]                            = {"HIPFFT_NOT_IMPLEMENTED",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0xE  14
  m["CUFFT_LICENSE_ERROR"]                              = {"HIPFFT_LICENSE_ERROR",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED | CUDA_REMOVED};
  m["CUFFT_NOT_SUPPORTED"]                              = {"HIPFFT_NOT_SUPPORTED",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x10 16
  m["CUFFT_MISSING_DEPENDENCY"]                         = {"HIPFFT_MISSING_DEPENDENCY",                        "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x11 17
  m["CUFFT_NVRTC_FAILURE"]                              = {"HIPFFT_NVRTC_FAILURE",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x12 18
  m["CUFFT_NVJITLINK_FAILURE"]                          = {"HIPFFT_NVJITLINK_FAILURE",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x13 19
  m["CUFFT_NVSHMEM_FAILURE"]                            = {"HIPFFT_NVSHMEM_FAILURE",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x14 20

  m["cufftType_t"]                                      = {"hipfftType_t",                                     "", CONV_TYPE, API_FFT, 1};
  m["cufftType"]                                        = {"hipfftType",                                       "", CONV_TYPE, API_FFT, 1};
  m["CUFFT_R2C"]                                        = {"HIPFFT_R2C",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x2a
  m["CUFFT_C2R"]                                        = {"HIPFFT_C2R",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x2c
  m["CUFFT_C2C"]                                        = {"HIPFFT_C2C",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x29
  m["CUFFT_D2Z"]                                        = {"HIPFFT_D2Z",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x6a
  m["CUFFT_Z2D"]                                        = {"HIPFFT_Z2D",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x6c
  m["CUFFT_Z2Z"]                                        = {"HIPFFT_Z2Z",                                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x69

  m["cufftCompatibility_t"]                             = {"hipfftCompatibility_t",                            "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftCompatibility"]                               = {"hipfftCompatibility",                              "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["CUFFT_COMPATIBILITY_FFTW_PADDING"]                 = {"HIPFFT_COMPATIBILITY_FFTW_PADDING",                "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x01

  // cufftXt enums
  m["cufftXtSubFormat_t"]                               = {"hipfftXtSubFormat_t",                              "", CONV_TYPE, API_FFT, 1};
  m["cufftXtSubFormat"]                                 = {"hipfftXtSubFormat",                                "", CONV_TYPE, API_FFT, 1};
  m["CUFFT_XT_FORMAT_INPUT"]                            = {"HIPFFT_XT_FORMAT_INPUT",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x00
  m["CUFFT_XT_FORMAT_OUTPUT"]                           = {"HIPFFT_XT_FORMAT_OUTPUT",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x01
  m["CUFFT_XT_FORMAT_INPLACE"]                          = {"HIPFFT_XT_FORMAT_INPLACE",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x02
  m["CUFFT_XT_FORMAT_INPLACE_SHUFFLED"]                 = {"HIPFFT_XT_FORMAT_INPLACE_SHUFFLED",                "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x03
  m["CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED"]                = {"HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED",               "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x04
  m["CUFFT_XT_FORMAT_DISTRIBUTED_INPUT"]                = {"HIPFFT_XT_FORMAT_DISTRIBUTED_INPUT",               "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x05
  m["CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT"]               = {"HIPFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",              "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x06
  m["CUFFT_FORMAT_UNDEFINED"]                           = {"HIPFFT_FORMAT_UNDEFINED",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x07

  m["cufftXtCopyType_t"]                                = {"hipfftXtCopyType_t",                               "", CONV_TYPE, API_FFT, 1};
  m["cufftXtCopyType"]                                  = {"hipfftXtCopyType",                                 "", CONV_TYPE, API_FFT, 1};
  m["CUFFT_COPY_HOST_TO_DEVICE"]                        = {"HIPFFT_COPY_HOST_TO_DEVICE",                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x00
  m["CUFFT_COPY_DEVICE_TO_HOST"]                        = {"HIPFFT_COPY_DEVICE_TO_HOST",                       "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x01
  m["CUFFT_COPY_DEVICE_TO_DEVICE"]                      = {"HIPFFT_COPY_DEVICE_TO_DEVICE",                     "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x02
  m["CUFFT_COPY_UNDEFINED"]                             = {"HIPFFT_COPY_UNDEFINED",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x03

  m["cufftXtQueryType_t"]                               = {"hipfftXtQueryType_t",                              "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftXtQueryType"]                                 = {"hipfftXtQueryType",                                "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["CUFFT_QUERY_1D_FACTORS"]                           = {"HIPFFT_QUERY_1D_FACTORS",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x00
  m["CUFFT_QUERY_UNDEFINED"]                            = {"HIPFFT_QUERY_UNDEFINED",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x01

  m["cufftXtWorkAreaPolicy_t"]                          = {"hipfftXtWorkAreaPolicy_t",                         "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftXtWorkAreaPolicy"]                            = {"hipfftXtWorkAreaPolicy",                           "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["CUFFT_WORKAREA_MINIMAL"]                           = {"HIPFFT_WORKAREA_MINIMAL",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0
  m["CUFFT_WORKAREA_USER"]                              = {"HIPFFT_WORKAREA_USER",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  1
  m["CUFFT_WORKAREA_PERFORMANCE"]                       = {"HIPFFT_WORKAREA_PERFORMANCE",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  2

  m["cufftXtCallbackType_t"]                            = {"hipfftXtCallbackType_t",                           "", CONV_TYPE, API_FFT, 1};
  m["cufftXtCallbackType"]                              = {"hipfftXtCallbackType",                             "", CONV_TYPE, API_FFT, 1};
  m["CUFFT_CB_LD_COMPLEX"]                              = {"HIPFFT_CB_LD_COMPLEX",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x0
  m["CUFFT_CB_LD_COMPLEX_DOUBLE"]                       = {"HIPFFT_CB_LD_COMPLEX_DOUBLE",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x1
  m["CUFFT_CB_LD_REAL"]                                 = {"HIPFFT_CB_LD_REAL",                                "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x2
  m["CUFFT_CB_LD_REAL_DOUBLE"]                          = {"HIPFFT_CB_LD_REAL_DOUBLE",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x3
  m["CUFFT_CB_ST_COMPLEX"]                              = {"HIPFFT_CB_ST_COMPLEX",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x4
  m["CUFFT_CB_ST_COMPLEX_DOUBLE"]                       = {"HIPFFT_CB_ST_COMPLEX_DOUBLE",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x5
  m["CUFFT_CB_ST_REAL"]                                 = {"HIPFFT_CB_ST_REAL",                                "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x6
  m["CUFFT_CB_ST_REAL_DOUBLE"]                          = {"HIPFFT_CB_ST_REAL_DOUBLE",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x7
  m["CUFFT_CB_UNDEFINED"]                               = {"HIPFFT_CB_UNDEFINED",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1};  //  0x7

  m["cufftProperty_t"]                                  = {"hipfftProperty",                                   "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftProperty"]                                    = {"hipfftProperty",                                   "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT"]            = {"HIPFFT_PLAN_PROPERTY_INT64_PATIENT_JIT",           "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x1
  m["NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS"]   = {"HIPFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS",  "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED};  //  0x2

  // cuFFT types
  m["cufftReal"]                                        = {"hipfftReal",                                       "", CONV_TYPE, API_FFT, 1};
  m["cufftDoubleReal"]                                  = {"hipfftDoubleReal",                                 "", CONV_TYPE, API_FFT, 1};
  m["cufftComplex"]                                     = {"hipfftComplex",                                    "", CONV_TYPE, API_FFT, 1};
  m["cufftDoubleComplex"]                               = {"hipfftDoubleComplex",                              "", CONV_TYPE, API_FFT, 1};
  m["cufftHandle"]                                      = {"hipfftHandle",                                     "", CONV_TYPE, API_FFT, 1};
  m["cufftXt1dFactors_t"]                               = {"hipfftXt1dFactors_t",                              "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftXt1dFactors"]                                 = {"hipfftXt1dFactors",                                "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftBox3d_t"]                                     = {"hipfftBox3d_t",                                    "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cufftBox3d"]                                       = {"hipfftBox3d",                                      "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["cudaLibXtDesc_t"]                                  = {"hipLibXtDesc_t",                                   "", CONV_TYPE, API_FFT, 1};
  m["cudaLibXtDesc"]                                    = {"hipLibXtDesc",                                     "", CONV_TYPE, API_FFT, 1};

  // cuFFTw types
  m["FFTW_FORWARD"]                                     = {"FFTW_FORWARD",                                     "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_BACKWARD"]                                    = {"FFTW_BACKWARD",                                    "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_INVERSE"]                                     = {"FFTW_INVERSE",                                     "", CONV_DEFINE, API_FFT, 1, UNSUPPORTED};
  m["FFTW_ESTIMATE"]                                    = {"FFTW_ESTIMATE",                                    "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_MEASURE"]                                     = {"FFTW_MEASURE",                                     "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_PATIENT"]                                     = {"FFTW_PATIENT",                                     "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_EXHAUSTIVE"]                                  = {"FFTW_EXHAUSTIVE",                                  "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_WISDOM_ONLY"]                                 = {"FFTW_WISDOM_ONLY",                                 "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_DESTROY_INPUT"]                               = {"FFTW_DESTROY_INPUT",                               "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_PRESERVE_INPUT"]                              = {"FFTW_PRESERVE_INPUT",                              "", CONV_DEFINE, API_FFT, 1};
  m["FFTW_UNALIGNED"]                                   = {"FFTW_UNALIGNED",                                   "", CONV_DEFINE, API_FFT, 1};
  m["fftw_complex"]                                     = {"fftw_complex",                                     "", CONV_TYPE, API_FFT, 1};
  m["fftwf_complex"]                                    = {"fftwf_complex",                                    "", CONV_TYPE, API_FFT, 1};
  m["fftw_iodim"]                                       = {"fftw_iodim",                                       "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["fftwf_iodim"]                                      = {"fftwf_iodim",                                      "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["fftw_iodim64"]                                     = {"fftw_iodim64",                                     "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["fftwf_iodim64"]                                    = {"fftwf_iodim64",                                    "", CONV_TYPE, API_FFT, 1, UNSUPPORTED};
  m["fftw_plan"]                                        = {"fftw_plan",                                        "", CONV_TYPE, API_FFT, 1};
  m["fftwf_plan"]                                       = {"fftwf_plan",                                       "", CONV_TYPE, API_FFT, 1};
  
  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_TYPE_NAME_VER_MAP = []() {
  std::map<llvm::StringRef, cudaAPIversions> m;
  m["CUFFT_NOT_SUPPORTED"]                              = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["cufftXtWorkAreaPolicy_t"]                          = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["cufftXtWorkAreaPolicy"]                            = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["CUFFT_WORKAREA_MINIMAL"]                           = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["CUFFT_WORKAREA_USER"]                              = {CUDA_92,  CUDA_0,   CUDA_0  };
  m["CUFFT_XT_FORMAT_DISTRIBUTED_INPUT"]                = {CUDA_118, CUDA_0,   CUDA_0  };
  m["CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT"]               = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cufftBox3d_t"]                                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cufftBox3d"]                                       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["cufftProperty_t"]                                  = {CUDA_124, CUDA_0,   CUDA_0  };
  m["cufftProperty"]                                    = {CUDA_124, CUDA_0,   CUDA_0  };
  m["NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT"]            = {CUDA_124, CUDA_0,   CUDA_0  };
  m["NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS"]   = {CUDA_125, CUDA_0,   CUDA_0  };
  m["CUFFT_INCOMPLETE_PARAMETER_LIST"]                  = {CUDA_0,   CUDA_0,   CUDA_130};
  m["CUFFT_PARSE_ERROR"]                                = {CUDA_0,   CUDA_0,   CUDA_130};
  m["CUFFT_LICENSE_ERROR"]                              = {CUDA_0,   CUDA_0,   CUDA_130};
  m["CUFFT_MISSING_DEPENDENCY"]                         = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUFFT_NVRTC_FAILURE"]                              = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUFFT_NVJITLINK_FAILURE"]                          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["CUFFT_NVSHMEM_FAILURE"]                            = {CUDA_130, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_FFT_TYPE_NAME_VER_MAP = []() {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["HIPFFT_FORWARD"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_BACKWARD"]                                  = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftResult_t"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftResult"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_SUCCESS"]                                   = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INVALID_PLAN"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_ALLOC_FAILED"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INVALID_TYPE"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INVALID_VALUE"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INTERNAL_ERROR"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_EXEC_FAILED"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_SETUP_FAILED"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INVALID_SIZE"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_UNALIGNED_DATA"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INCOMPLETE_PARAMETER_LIST"]                 = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_INVALID_DEVICE"]                            = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_PARSE_ERROR"]                               = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_NO_WORKSPACE"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_NOT_IMPLEMENTED"]                           = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_NOT_SUPPORTED"]                             = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftType_t"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftType"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_R2C"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_C2R"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_C2C"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_D2Z"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_Z2D"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["HIPFFT_Z2Z"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftReal"]                                       = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftDoubleReal"]                                 = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftComplex"]                                    = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftDoubleComplex"]                              = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftHandle"]                                     = {HIP_1070, HIP_0,    HIP_0   };
  m["hipfftXtSubFormat_t"]                              = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtSubFormat"]                                = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_XT_FORMAT_INPUT"]                           = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_XT_FORMAT_OUTPUT"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_XT_FORMAT_INPLACE"]                         = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_XT_FORMAT_INPLACE_SHUFFLED"]                = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED"]               = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_FORMAT_UNDEFINED"]                          = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtCopyType_t"]                               = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtCopyType"]                                 = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_COPY_HOST_TO_DEVICE"]                       = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_COPY_DEVICE_TO_HOST"]                       = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_COPY_DEVICE_TO_DEVICE"]                     = {HIP_6000, HIP_0,    HIP_0   };
  m["HIPFFT_COPY_UNDEFINED"]                            = {HIP_6000, HIP_0,    HIP_0   };
  m["hipfftXtCallbackType_t"]                           = {HIP_4030, HIP_0,    HIP_0   };
  m["hipfftXtCallbackType"]                             = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_LD_COMPLEX"]                             = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_LD_COMPLEX_DOUBLE"]                      = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_LD_REAL"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_LD_REAL_DOUBLE"]                         = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_ST_COMPLEX"]                             = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_ST_COMPLEX_DOUBLE"]                      = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_ST_REAL"]                                = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_ST_REAL_DOUBLE"]                         = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPFFT_CB_UNDEFINED"]                              = {HIP_4030, HIP_0,    HIP_0   };
  m["hipLibXtDesc_t"]                                   = {HIP_6000, HIP_0,    HIP_0   };
  m["hipLibXtDesc"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["FFTW_FORWARD"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_BACKWARD"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_ESTIMATE"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_MEASURE"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_PATIENT"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_EXHAUSTIVE"]                                  = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_WISDOM_ONLY"]                                 = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_DESTROY_INPUT"]                               = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_PRESERVE_INPUT"]                              = {HIP_7010, HIP_0,    HIP_0   };
  m["FFTW_UNALIGNED"]                                   = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_complex"]                                     = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_complex"]                                    = {HIP_7010, HIP_0,    HIP_0   };
  m["fftw_plan"]                                        = {HIP_7010, HIP_0,    HIP_0   };
  m["fftwf_plan"]                                       = {HIP_7010, HIP_0,    HIP_0   };

  return m;
}();
