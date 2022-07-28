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

// Maps the names of CUDA CUB API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_CUB_NAMESPACE_MAP {
  {"cub",                                      {"hipcub",                                    "", CONV_TYPE, API_CUB, 1}},
};

// Maps the names of CUDA CUB API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_CUB_TYPE_NAME_MAP {
  // 5. Defines
  {"CUB_STDERR",                               {"HIPCUB_STDERR",                             "", CONV_DEFINE, API_CUB, 1}},
  {"CubDebug",                                 {"HipcubDebug",                               "", CONV_DEFINE, API_CUB, 1}},
  {"CubDebugExit",                             {"HipcubDebugExit",                           "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"_CubLog",                                  {"_HipcubLog",                                "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_RUNTIME_FUNCTION",                     {"HIPCUB_RUNTIME_FUNCTION",                   "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_PTX_WARP_THREADS",                     {"HIPCUB_WARP_THREADS",                       "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_PTX_ARCH",                             {"HIPCUB_ARCH",                               "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_NAMESPACE_BEGIN",                      {"BEGIN_HIPCUB_NAMESPACE",                    "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_NAMESPACE_END",                        {"END_HIPCUB_NAMESPACE",                      "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_USE_COOPERATIVE_GROUPS",               {"HIPCUB_USE_COOPERATIVE_GROUPS",             "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IS_DEVICE_CODE",                       {"HIPCUB_IS_DEVICE_CODE",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IS_HOST_CODE",                         {"HIPCUB_IS_HOST_CODE",                       "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_INCLUDE_DEVICE_CODE",                  {"HIPCUB_INCLUDE_DEVICE_CODE",                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_INCLUDE_HOST_CODE",                    {"HIPCUB_INCLUDE_HOST_CODE",                  "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_MAX_DEVICES",                          {"HIPCUB_MAX_DEVICES",                        "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_CPP_DIALECT",                          {"HIPCUB_CPP_DIALECT",                        "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_RUNTIME_ENABLED",                      {"HIPCUB_RUNTIME_ENABLED",                    "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_LOG_WARP_THREADS",                     {"HIPCUB_LOG_WARP_THREADS",                   "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PTX_LOG_WARP_THREADS",                 {"HIPCUB_LOG_WARP_THREADS",                   "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_LOG_SMEM_BANKS",                       {"HIPCUB_LOG_SMEM_BANKS",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_SMEM_BANKS",                           {"HIPCUB_SMEM_BANKS",                         "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PTX_LOG_SMEM_BANKS",                   {"HIPCUB_LOG_SMEM_BANKS",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PTX_SMEM_BANKS",                       {"HIPCUB_SMEM_BANKS",                         "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_SUBSCRIPTION_FACTOR",                  {"HIPCUB_SUBSCRIPTION_FACTOR",                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PTX_SUBSCRIPTION_FACTOR",              {"HIPCUB_SUBSCRIPTION_FACTOR",                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PREFER_CONFLICT_OVER_PADDING",         {"HIPCUB_PREFER_CONFLICT_OVER_PADDING",       "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PTX_PREFER_CONFLICT_OVER_PADDING",     {"HIPCUB_PREFER_CONFLICT_OVER_PADDING",       "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_MAX",                                  {"CUB_MAX",                                   "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_MIN",                                  {"CUB_MIN",                                   "", CONV_DEFINE, API_CUB, 1}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_CUB_TYPE_NAME_VER_MAP {
};

const std::map<llvm::StringRef, hipAPIversions> HIP_CUB_TYPE_NAME_VER_MAP {
  {"HIPCUB_STDERR",                          {HIP_2050, HIP_0,    HIP_0   }},
  {"HipcubDebug",                            {HIP_2050, HIP_0,    HIP_0   }},
  {"_HipcubLog",                             {HIP_2050, HIP_0,    HIP_0   }},
  {"HIPCUB_RUNTIME_FUNCTION",                {HIP_2050, HIP_0,    HIP_0   }},
  {"HIPCUB_WARP_THREADS",                    {HIP_2050, HIP_0,    HIP_0   }},
  {"HIPCUB_ARCH",                            {HIP_2050, HIP_0,    HIP_0   }},
  {"BEGIN_HIPCUB_NAMESPACE",                 {HIP_2050, HIP_0,    HIP_0   }},
  {"END_HIPCUB_NAMESPACE",                   {HIP_2050, HIP_0,    HIP_0   }},
  {"CUB_MIN",                                {HIP_2050, HIP_0,    HIP_0   }},
};

const std::map<unsigned int, llvm::StringRef> CUDA_CUB_API_SECTION_MAP {
  {1, "CUB Data types"},
};
