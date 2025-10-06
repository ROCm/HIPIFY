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

// Maps the names of CUDA RTC API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_RTC_TYPE_NAME_MAP {
  {"nvrtcResult",                                                {"hiprtcResult",                                        "", CONV_TYPE, API_RTC, 1}},
  {"NVRTC_SUCCESS",                                              {"HIPRTC_SUCCESS",                                      "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 0
  {"NVRTC_ERROR_OUT_OF_MEMORY",                                  {"HIPRTC_ERROR_OUT_OF_MEMORY",                          "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 1
  {"NVRTC_ERROR_PROGRAM_CREATION_FAILURE",                       {"HIPRTC_ERROR_PROGRAM_CREATION_FAILURE",               "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 2
  {"NVRTC_ERROR_INVALID_INPUT",                                  {"HIPRTC_ERROR_INVALID_INPUT",                          "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 3
  {"NVRTC_ERROR_INVALID_PROGRAM",                                {"HIPRTC_ERROR_INVALID_PROGRAM",                        "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 4
  {"NVRTC_ERROR_INVALID_OPTION",                                 {"HIPRTC_ERROR_INVALID_OPTION",                         "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 5
  {"NVRTC_ERROR_COMPILATION",                                    {"HIPRTC_ERROR_COMPILATION",                            "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 6
  {"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",                      {"HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE",              "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 7
  {"NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",          {"HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",  "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 8
  {"NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",            {"HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",    "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 9
  {"NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",                      {"HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID",              "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 10
  {"NVRTC_ERROR_INTERNAL_ERROR",                                 {"HIPRTC_ERROR_INTERNAL_ERROR",                         "", CONV_NUMERIC_LITERAL, API_RTC, 1}}, // 11
  {"NVRTC_ERROR_TIME_FILE_WRITE_FAILED",                         {"HIPRTC_ERROR_TIME_FILE_WRITE_FAILED",                 "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 12
  {"NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED",                        {"HIPRTC_ERROR_NO_PCH_CREATE_ATTEMPTED",                "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 13
  {"NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",                      {"HIPRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",              "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 14
  {"NVRTC_ERROR_PCH_CREATE",                                     {"HIPRTC_ERROR_PCH_CREATE",                             "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 15
  {"NVRTC_ERROR_CANCELLED",                                      {"HIPRTC_ERROR_CANCELLED",                              "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 16
  {"NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED",                   {"HIPRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED",           "", CONV_NUMERIC_LITERAL, API_RTC, 1, UNSUPPORTED}}, // 17

  {"nvrtcProgram",                                               {"hiprtcProgram",                                       "", CONV_TYPE, API_RTC, 1}},
  {"_nvrtcProgram",                                              {"_hiprtcProgram",                                      "", CONV_TYPE, API_RTC, 1}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RTC_TYPE_NAME_VER_MAP {
  {"NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",          {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",            {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID",                      {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_INTERNAL_ERROR",                                 {CUDA_80,  CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_TIME_FILE_WRITE_FAILED",                         {CUDA_121, CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED",                        {CUDA_128, CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_PCH_CREATE",                                     {CUDA_128, CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_CANCELLED",                                      {CUDA_128, CUDA_0,   CUDA_0  }},
  {"NVRTC_ERROR_TIME_TRACE_FILE_WRITE_FAILED",                   {CUDA_130, CUDA_0,   CUDA_0  }},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_RTC_TYPE_NAME_VER_MAP {
  {"hiprtcResult",                                               {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_SUCCESS",                                             {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_OUT_OF_MEMORY",                                 {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_PROGRAM_CREATION_FAILURE",                      {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_INVALID_INPUT",                                 {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_INVALID_PROGRAM",                               {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_INVALID_OPTION",                                {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_COMPILATION",                                   {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE",                     {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION",         {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION",           {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID",                     {HIP_2060, HIP_0,    HIP_0   }},
  {"HIPRTC_ERROR_INTERNAL_ERROR",                                {HIP_2060, HIP_0,    HIP_0   }},
  {"hiprtcProgram",                                              {HIP_2060, HIP_0,    HIP_0   }},
  {"_hiprtcProgram",                                             {HIP_2060, HIP_0,    HIP_0   }},
};
