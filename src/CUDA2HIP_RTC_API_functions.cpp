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
const std::map<llvm::StringRef, hipCounter> CUDA_RTC_FUNCTION_MAP {
  {"nvrtcGetErrorString",                         {"hiprtcGetErrorString",                         "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcVersion",                                {"hiprtcVersion",                                "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetNumSupportedArchs",                   {"hiprtcGetNumSupportedArchs",                   "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
  {"nvrtcGetSupportedArchs",                      {"hiprtcGetSupportedArchs",                      "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
  {"nvrtcCreateProgram",                          {"hiprtcCreateProgram",                          "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcDestroyProgram",                         {"hiprtcDestroyProgram",                         "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcCompileProgram",                         {"hiprtcCompileProgram",                         "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetPTXSize",                             {"hiprtcGetCodeSize",                            "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetPTX",                                 {"hiprtcGetCode",                                "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetCUBINSize",                           {"hiprtcGetBitcodeSize",                         "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetCUBIN",                               {"hiprtcGetBitcode",                             "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetNVVMSize",                            {"hiprtcGetNVVMSize",                            "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"nvrtcGetNVVM",                                {"hiprtcGetNVVM",                                "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED | CUDA_DEPRECATED}},
  {"nvrtcGetProgramLogSize",                      {"hiprtcGetProgramLogSize",                      "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetProgramLog",                          {"hiprtcGetProgramLog",                          "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcAddNameExpression",                      {"hiprtcAddNameExpression",                      "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetLoweredName",                         {"hiprtcGetLoweredName",                         "", CONV_LIB_FUNC, API_RTC, 2}},
  {"nvrtcGetLTOIRSize",                           {"hiprtcGetLTOIRSize",                           "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
  {"nvrtcGetLTOIR",                               {"hiprtcGetLTOIR",                               "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
  {"nvrtcGetOptiXIRSize",                         {"hiprtcGetOptiXIRSize",                         "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
  {"nvrtcGetOptiXIR",                             {"hiprtcGetOptiXIR",                             "", CONV_LIB_FUNC, API_RTC, 2, HIP_UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RTC_FUNCTION_VER_MAP {
  {"nvrtcGetNumSupportedArchs",                   {CUDA_112, CUDA_0,   CUDA_0  }},
  {"nvrtcGetSupportedArchs",                      {CUDA_112, CUDA_0,   CUDA_0  }},
  {"nvrtcGetCUBINSize",                           {CUDA_111, CUDA_0,   CUDA_0  }},
  {"nvrtcGetCUBIN",                               {CUDA_111, CUDA_0,   CUDA_0  }},
  {"nvrtcGetNVVMSize",                            {CUDA_114, CUDA_120, CUDA_0  }},
  {"nvrtcGetNVVM",                                {CUDA_114, CUDA_120, CUDA_0  }},
  {"nvrtcAddNameExpression",                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"nvrtcGetLoweredName",                         {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"nvrtcGetLTOIRSize",                           {CUDA_120, CUDA_0,   CUDA_0  }},
  {"nvrtcGetLTOIR",                               {CUDA_120, CUDA_0,   CUDA_0  }},
  {"nvrtcGetOptiXIRSize",                         {CUDA_120, CUDA_0,   CUDA_0  }},
  {"nvrtcGetOptiXIR",                             {CUDA_120, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_RTC_FUNCTION_VER_MAP {
  {"hiprtcGetErrorString",                        {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcVersion",                               {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcCreateProgram",                         {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcDestroyProgram",                        {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcCompileProgram",                        {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetCodeSize",                           {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetCode",                               {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetProgramLogSize",                     {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetProgramLog",                         {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcAddNameExpression",                     {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetLoweredName",                        {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcGetBitcode",                            {HIP_5030, HIP_0,    HIP_0   }},
  {"hiprtcGetBitcodeSize",                        {HIP_5030, HIP_0,    HIP_0   }},
};

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RTC_FUNCTION_CHANGED_VER_MAP {
  {"nvrtcCreateProgram",                          {CUDA_80}},
  {"nvrtcCompileProgram",                         {CUDA_80}},
};

const std::map<unsigned int, llvm::StringRef> CUDA_RTC_API_SECTION_MAP {
  {1, "RTC Data types"},
  {2, "RTC API functions"},
};
