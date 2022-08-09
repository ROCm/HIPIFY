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
  // 1. Structs
  {"CubVector",                                {"HipcubVector",                              "", CONV_TYPE, API_CUB, 1, HIP_UNSUPPORTED}},

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
  {"__CUB_ALIGN_BYTES",                        {"__HIPCUB_ALIGN_BYTES",                      "", CONV_DEFINE, API_CUB, 1}},
  {"CUB_DEFINE_VECTOR_TYPE",                   {"HIPCUB_DEFINE_VECTOR_TYPE",                 "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEFINE_DETECT_NESTED_TYPE",            {"HIPCUB_DEFINE_DETECT_NESTED_TYPE",          "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"__CUB_LP64__",                             {"__HIPCUB_LP64__",                           "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"_CUB_ASM_PTR_",                            {"_HIPCUB_ASM_PTR_",                          "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"_CUB_ASM_PTR_SIZE_",                       {"_HIPCUB_ASM_PTR_SIZE_",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_USE_COOPERATIVE_GROUPS",               {"HIPCUB_USE_COOPERATIVE_GROUPS",             "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_ALIGN",                                {"HIPCUB_ALIGN",                              "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_PREVENT_MACRO_SUBSTITUTION",           {"HIPCUB_PREVENT_MACRO_SUBSTITUTION",         "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_QUOTIENT_FLOOR",                       {"HIPCUB_QUOTIENT_FLOOR",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_QUOTIENT_CEILING",                     {"HIPCUB_QUOTIENT_CEILING",                   "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_ROUND_UP_NEAREST",                     {"HIPCUB_ROUND_UP_NEAREST",                   "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_ROUND_DOWN_NEAREST",                   {"HIPCUB_ROUND_DOWN_NEAREST",                 "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_STATIC_ASSERT",                        {"HIPCUB_STATIC_ASSERT",                      "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_CAT",                                  {"HIPCUB_CAT",                                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_CAT_",                                 {"HIPCUB_CAT_",                               "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_API",                {"HIPCUB_IGNORE_DEPRECATED_API",              "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEPRECATED",                           {"HIPCUB_DEPRECATED",                         "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_HOST_COMPILER",                        {"HIPCUB_HOST_COMPILER",                      "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_HOST_COMPILER_MSVC",                   {"HIPCUB_HOST_COMPILER_MSVC",                 "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_HOST_COMPILER_CLANG",                  {"HIPCUB_HOST_COMPILER_CLANG",                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_HOST_COMPILER_GCC",                    {"HIPCUB_HOST_COMPILER_GCC",                  "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_HOST_COMPILER_UNKNOWN",                {"HIPCUB_HOST_COMPILER_UNKNOWN",              "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER",                      {"HIPCUB_DEVICE_COMPILER",                    "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER_MSVC",                 {"HIPCUB_DEVICE_COMPILER_MSVC",               "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER_CLANG",                {"HIPCUB_DEVICE_COMPILER_CLANG",              "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER_GCC",                  {"HIPCUB_DEVICE_COMPILER_GCC",                "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER_NVCC",                 {"HIPCUB_DEVICE_COMPILER_NVCC",               "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_DEVICE_COMPILER_UNKNOWN",              {"HIPCUB_DEVICE_COMPILER_UNKNOWN",            "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_DIALECT",            {"HIPCUB_IGNORE_DEPRECATED_DIALECT",          "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_CPP_DIALECT",        {"HIPCUB_IGNORE_DEPRECATED_CPP_DIALECT",      "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_CPP_11",             {"HIPCUB_IGNORE_DEPRECATED_CPP_11",           "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_CPP_11",             {"HIPCUB_IGNORE_DEPRECATED_CPP_11",           "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_IGNORE_DEPRECATED_COMPILER",           {"HIPCUB_IGNORE_DEPRECATED_COMPILER",         "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_MSVC_VERSION",                         {"HIPCUB_MSVC_VERSION",                       "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_MSVC_VERSION_FULL",                    {"HIPCUB_MSVC_VERSION_FULL",                  "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_CPLUSPLUS",                            {"HIPCUB_CPLUSPLUS",                          "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_COMP_DEPR_IMPL",                       {"HIPCUB_COMP_DEPR_IMPL",                     "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_COMP_DEPR_IMPL0",                      {"HIPCUB_COMP_DEPR_IMPL0",                    "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_COMP_DEPR_IMPL1",                      {"HIPCUB_COMP_DEPR_IMPL1",                    "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_COMPILER_DEPRECATION",                 {"HIPCUB_COMPILER_DEPRECATION",               "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
  {"CUB_COMPILER_DEPRECATION_SOFT",            {"HIPCUB_COMPILER_DEPRECATION_SOFT",          "", CONV_DEFINE, API_CUB, 1, HIP_UNSUPPORTED}},
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
  {"CUB_MAX",                                {HIP_4050, HIP_0,    HIP_0   }},
  {"CUB_MIN",                                {HIP_4050, HIP_0,    HIP_0   }},
  {"__HIPCUB_ALIGN_BYTES",                   {HIP_4050, HIP_0,    HIP_0   }},
};

const std::map<unsigned int, llvm::StringRef> CUDA_CUB_API_SECTION_MAP {
  {1, "CUB Data types"},
};
