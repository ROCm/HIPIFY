/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

// Maps the names of CUDA RTC API functions to the corresponding HIP functions
const std::map<llvm::StringRef, hipCounter> CUDA_RTC_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  m["nvrtcGetErrorString"]                        = {"hiprtcGetErrorString",                         "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcVersion"]                               = {"hiprtcVersion",                                "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetNumSupportedArchs"]                  = {"hiprtcGetNumSupportedArchs",                   "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetSupportedArchs"]                     = {"hiprtcGetSupportedArchs",                      "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcCreateProgram"]                         = {"hiprtcCreateProgram",                          "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcDestroyProgram"]                        = {"hiprtcDestroyProgram",                         "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcCompileProgram"]                        = {"hiprtcCompileProgram",                         "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetPTXSize"]                            = {"hiprtcGetCodeSize",                            "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetPTX"]                                = {"hiprtcGetCode",                                "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetCUBINSize"]                          = {"hiprtcGetBitcodeSize",                         "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetCUBIN"]                              = {"hiprtcGetBitcode",                             "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetNVVMSize"]                           = {"hiprtcGetNVVMSize",                            "", CONV_LIB_FUNC, API_RTC, 2, CUDA_DEPRECATED | CUDA_REMOVED | UNSUPPORTED};
  m["nvrtcGetNVVM"]                               = {"hiprtcGetNVVM",                                "", CONV_LIB_FUNC, API_RTC, 2, CUDA_DEPRECATED | CUDA_REMOVED | UNSUPPORTED};
  m["nvrtcGetProgramLogSize"]                     = {"hiprtcGetProgramLogSize",                      "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetProgramLog"]                         = {"hiprtcGetProgramLog",                          "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcAddNameExpression"]                     = {"hiprtcAddNameExpression",                      "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetLoweredName"]                        = {"hiprtcGetLoweredName",                         "", CONV_LIB_FUNC, API_RTC, 2};
  m["nvrtcGetLTOIRSize"]                          = {"hiprtcGetLTOIRSize",                           "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetLTOIR"]                              = {"hiprtcGetLTOIR",                               "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetOptiXIRSize"]                        = {"hiprtcGetOptiXIRSize",                         "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetOptiXIR"]                            = {"hiprtcGetOptiXIR",                             "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetPCHHeapSize"]                        = {"hiprtcGetPCHHeapSize",                         "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcSetPCHHeapSize"]                        = {"hiprtcSetPCHHeapSize",                         "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetPCHCreateStatus"]                    = {"hiprtcGetPCHCreateStatus",                     "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetPCHHeapSizeRequired"]                = {"hiprtcGetPCHHeapSizeRequired",                 "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcSetFlowCallback"]                       = {"hiprtcSetFlowCallback",                        "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetTileIRSize"]                         = {"hiprtcGetTileIRSize",                          "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};
  m["nvrtcGetTileIR"]                             = {"hiprtcGetTileIR",                              "", CONV_LIB_FUNC, API_RTC, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RTC_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["nvrtcGetNumSupportedArchs"]                  = {CUDA_112, CUDA_0,   CUDA_0  };
  m["nvrtcGetSupportedArchs"]                     = {CUDA_112, CUDA_0,   CUDA_0  };
  m["nvrtcGetCUBINSize"]                          = {CUDA_111, CUDA_0,   CUDA_0  };
  m["nvrtcGetCUBIN"]                              = {CUDA_111, CUDA_0,   CUDA_0  };
  m["nvrtcGetNVVMSize"]                           = {CUDA_114, CUDA_120, CUDA_130};
  m["nvrtcGetNVVM"]                               = {CUDA_114, CUDA_120, CUDA_130};
  m["nvrtcAddNameExpression"]                     = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["nvrtcGetLoweredName"]                        = {CUDA_80,  CUDA_0,   CUDA_0  };
  m["nvrtcGetLTOIRSize"]                          = {CUDA_120, CUDA_0,   CUDA_0  };
  m["nvrtcGetLTOIR"]                              = {CUDA_120, CUDA_0,   CUDA_0  };
  m["nvrtcGetOptiXIRSize"]                        = {CUDA_120, CUDA_0,   CUDA_0  };
  m["nvrtcGetOptiXIR"]                            = {CUDA_120, CUDA_0,   CUDA_0  };
  m["nvrtcGetPCHHeapSize"]                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["nvrtcSetPCHHeapSize"]                        = {CUDA_128, CUDA_0,   CUDA_0  };
  m["nvrtcGetPCHCreateStatus"]                    = {CUDA_128, CUDA_0,   CUDA_0  };
  m["nvrtcGetPCHHeapSizeRequired"]                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["nvrtcSetFlowCallback"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["nvrtcGetTileIRSize"]                         = {CUDA_132, CUDA_0,   CUDA_0  };
  m["nvrtcGetTileIR"]                             = {CUDA_132, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_RTC_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hiprtcGetErrorString"]                       = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcVersion"]                              = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcCreateProgram"]                        = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcDestroyProgram"]                       = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcCompileProgram"]                       = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetCodeSize"]                          = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetCode"]                              = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetProgramLogSize"]                    = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetProgramLog"]                        = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcAddNameExpression"]                    = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetLoweredName"]                       = {HIP_2060, HIP_0,    HIP_0   };
  m["hiprtcGetBitcode"]                           = {HIP_5030, HIP_0,    HIP_0   };
  m["hiprtcGetBitcodeSize"]                       = {HIP_5030, HIP_0,    HIP_0   };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RTC_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIChangedVersions> m;

  m["nvrtcCreateProgram"]                         = {CUDA_80};
  m["nvrtcCompileProgram"]                        = {CUDA_80};

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_RTC_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIChangedVersions> m;

  m["hiprtcCreateProgram"]                        = {HIP_7000};
  m["hiprtcCompileProgram"]                       = {HIP_7000};

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_RTC_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1]                                            = "RTC Data types";
  m[2]                                            = "RTC API functions";

  return m;
}();
