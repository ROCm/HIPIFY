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

// Maps the names of CUDA Device/Host types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_DEVICE_TYPE_NAME_MAP = []() {
  std::map<llvm::StringRef, hipCounter> m;

  // float16 Precision Device types
  m["__half"]                              = {"__half",                                "rocblas_half",                            CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__half_raw"]                          = {"__half_raw",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__half2"]                             = {"__half2",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__half2_raw"]                         = {"__half2_raw",                           "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  // Bfloat16 Precision Device types
  m["__nv_bfloat16"]                       = {"__hip_bfloat16",                        "rocblas_bfloat16",                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["nv_bfloat16"]                         = {"hip_bfloat16",                          "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_bfloat16_raw"]                   = {"__hip_bfloat16_raw",                    "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_bfloat162"]                      = {"__hip_bfloat162",                       "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["nv_bfloat162"]                        = {"hip_bfloat162",                         "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["__nv_bfloat162_raw"]                  = {"__hip_bfloat162_raw",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  // float8 Precision Device types
  m["__nv_fp8_storage_t"]                  = {"__hip_fp8_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8x2_storage_t"]                = {"__hip_fp8x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8x4_storage_t"]                = {"__hip_fp8x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8_e5m2"]                       = {"__hip_fp8_e5m2_fnuz",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8x2_e5m2"]                     = {"__hip_fp8x2_e5m2_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8_e4m3"]                       = {"__hip_fp8_e4m3_fnuz",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8x2_e4m3"]                     = {"__hip_fp8x2_e4m3_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8x4_e4m3"]                     = {"__hip_fp8x4_e4m3_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_saturation_t"]                   = {"__hip_saturation_t",                    "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__NV_NOSAT"]                          = {"__HIP_NOSAT",                           "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__NV_SATFINITE"]                      = {"__HIP_SATFINITE",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__nv_fp8_interpretation_t"]           = {"__hip_fp8_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__NV_E4M3"]                           = {"__HIP_E4M3_FNUZ",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__NV_E5M2"]                           = {"__HIP_E5M2_FNUZ",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__nv_fp8x4_e5m2"]                     = {"__hip_fp8x4_e5m2_fnuz",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp8_e8m0"]                       = {"__hip_fp8_e8m0",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["__nv_fp8x2_e8m0"]                     = {"__hip_fp8x2_e8m0",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["__nv_fp8x4_e8m0"]                     = {"__hip_fp8x4_e8m0",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  // float6 Precision Device types
  m["__nv_fp6_storage_t"]                  = {"__hip_fp6_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x2_storage_t"]                = {"__hip_fp6x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x4_storage_t"]                = {"__hip_fp6x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6_interpretation_t"]           = {"__hip_fp6_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__NV_E2M3"]                           = {"__HIP_E2M3",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__NV_E3M2"]                           = {"__HIP_E3M2",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__nv_fp6_e3m2"]                       = {"__hip_fp6_e3m2",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x2_e3m2"]                     = {"__hip_fp6x2_e3m2",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x4_e3m2"]                     = {"__hip_fp6x4_e3m2",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6_e2m3"]                       = {"__hip_fp6_e2m3",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x2_e2m3"]                     = {"__hip_fp6x2_e2m3",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp6x4_e2m3"]                     = {"__hip_fp6x4_e2m3",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  // float4 Precision Device types
  m["__nv_fp4_storage_t"]                  = {"__hip_fp4_storage_t",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp4x2_storage_t"]                = {"__hip_fp4x2_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp4x4_storage_t"]                = {"__hip_fp4x4_storage_t",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp4_interpretation_t"]           = {"__hip_fp4_interpretation_t",            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__NV_E2M1"]                           = {"__HIP_E2M1",                            "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["__nv_fp4_e2m1"]                       = {"__hip_fp4_e2m1",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp4x2_e2m1"]                     = {"__hip_fp4x2_e2m1",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["__nv_fp4x4_e2m1"]                     = {"__hip_fp4x4_e2m1",                      "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  // defines
  m["CUDART_INF_FP16"]                     = {"HIPRT_INF_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_MAX_NORMAL_FP16"]              = {"HIPRT_MAX_NORMAL_FP16",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_MIN_DENORM_FP16"]              = {"HIPRT_MIN_DENORM_FP16",                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_NAN_FP16"]                     = {"HIPRT_NAN_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_NEG_ZERO_FP16"]                = {"HIPRT_NEG_ZERO_FP16",                   "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_ONE_FP16"]                     = {"HIPRT_ONE_FP16",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["CUDART_ZERO_FP16"]                    = {"HIPRT_ZERO_FP16",                       "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  // builtins
  m["cudaRoundMode"]                       = {"hipRoundMode",                          "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["cudaRoundNearest"]                    = {"hipRoundNearest",                       "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["cudaRoundZero"]                       = {"hipRoundZero",                          "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["cudaRoundPosInf"]                     = {"hipRoundPosInf",                        "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  m["cudaRoundMinInf"]                     = {"hipRoundMinInf",                        "",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, 2};
  // vestor types
  m["char1"]                               = {"char1",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["char2"]                               = {"char2",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["char3"]                               = {"char3",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["char4"]                               = {"char4",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uchar1"]                              = {"uchar1",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uchar2"]                              = {"uchar2",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uchar3"]                              = {"uchar3",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uchar4"]                              = {"uchar4",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["short1"]                              = {"short1",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["short2"]                              = {"short2",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["short3"]                              = {"short3",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["short4"]                              = {"short4",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ushort1"]                             = {"ushort1",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ushort2"]                             = {"ushort2",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ushort3"]                             = {"ushort3",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ushort4"]                             = {"ushort4",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["int1"]                                = {"int1",                                  "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["int2"]                                = {"int2",                                  "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["int3"]                                = {"int3",                                  "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["int4"]                                = {"int4",                                  "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uint1"]                               = {"uint1",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uint2"]                               = {"uint2",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uint3"]                               = {"uint3",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["uint4"]                               = {"uint4",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["long1"]                               = {"long1",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["long2"]                               = {"long2",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["long3"]                               = {"long3",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["long4"]                               = {"long4",                                 "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, CUDA_DEPRECATED};
  m["ulong1"]                              = {"ulong1",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulong2"]                              = {"ulong2",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulong3"]                              = {"ulong3",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulong4"]                              = {"ulong4",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, CUDA_DEPRECATED};
  m["long4_16a"]                           = {"long4_16a",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["ulong4_16a"]                          = {"ulong4_16a",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["long4_32a"]                           = {"long4_32a",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["ulong4_32a"]                          = {"ulong4_32a",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["float1"]                              = {"float1",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["float2"]                              = {"float2",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["float3"]                              = {"float3",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["float4"]                              = {"float4",                                "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["double1"]                             = {"double1",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["double2"]                             = {"double2",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["double3"]                             = {"double3",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["double4"]                             = {"double4",                               "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, CUDA_DEPRECATED};
  m["double4_16a"]                         = {"double4_16a",                           "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["double4_32a"]                         = {"double4_32a",                           "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["longlong1"]                           = {"longlong1",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["longlong2"]                           = {"longlong2",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["longlong3"]                           = {"longlong3",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["longlong4"]                           = {"longlong4",                             "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, CUDA_DEPRECATED};
  m["longlong4_16a"]                       = {"longlong4_16a",                         "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["longlong4_32a"]                       = {"longlong4_32a",                         "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["ulonglong1"]                          = {"ulonglong1",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulonglong2"]                          = {"ulonglong2",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulonglong3"]                          = {"ulonglong3",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2};
  m["ulonglong4"]                          = {"ulonglong4",                            "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, CUDA_DEPRECATED};
  m["ulonglong4_16a"]                      = {"ulonglong4_16a",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};
  m["ulonglong4_32a"]                      = {"ulonglong4_32a",                        "",                                        CONV_DEVICE_TYPE, API_RUNTIME, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_DEVICE_TYPE_NAME_VER_MAP = []() {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["__nv_bfloat16"]                       = {CUDA_110, CUDA_0,   CUDA_0  };
  m["nv_bfloat16"]                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["__nv_bfloat16_raw"]                   = {CUDA_110, CUDA_0,   CUDA_0  };
  m["__nv_bfloat162"]                      = {CUDA_110, CUDA_0,   CUDA_0  };
  m["nv_bfloat162"]                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["__nv_bfloat162_raw"]                  = {CUDA_110, CUDA_0,   CUDA_0  };
  m["__nv_fp8_storage_t"]                  = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x2_storage_t"]                = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x4_storage_t"]                = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8_e5m2"]                       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x2_e5m2"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8_e4m3"]                       = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x2_e4m3"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x4_e4m3"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_saturation_t"]                   = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__NV_NOSAT"]                          = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__NV_SATFINITE"]                      = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8_interpretation_t"]           = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__NV_E4M3"]                           = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__NV_E5M2"]                           = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp8x4_e5m2"]                     = {CUDA_118, CUDA_0,   CUDA_0  };
  m["__nv_fp6_storage_t"]                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x2_storage_t"]                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x4_storage_t"]                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6_interpretation_t"]           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__NV_E2M3"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__NV_E3M2"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6_e3m2"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x2_e3m2"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x4_e3m2"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6_e2m3"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x2_e2m3"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp6x4_e2m3"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4_storage_t"]                  = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4x2_storage_t"]                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4x4_storage_t"]                = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4_interpretation_t"]           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__NV_E2M1"]                           = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4_e2m1"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4x2_e2m1"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp4x4_e2m1"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp8_e8m0"]                       = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp8x2_e8m0"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["__nv_fp8x4_e8m0"]                     = {CUDA_128, CUDA_0,   CUDA_0  };
  m["CUDART_INF_FP16"]                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_MAX_NORMAL_FP16"]              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_MIN_DENORM_FP16"]              = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_NAN_FP16"]                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_NEG_ZERO_FP16"]                = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_ONE_FP16"]                     = {CUDA_122, CUDA_0,   CUDA_0  };
  m["CUDART_ZERO_FP16"]                    = {CUDA_122, CUDA_0,   CUDA_0  };
  m["long4"]                               = {CUDA_0,   CUDA_130, CUDA_0  };
  m["ulong4"]                              = {CUDA_0,   CUDA_130, CUDA_0  };
  m["long4_16a"]                           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["ulong4_16a"]                          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["long4_32a"]                           = {CUDA_130, CUDA_0,   CUDA_0  };
  m["ulong4_32a"]                          = {CUDA_130, CUDA_0,   CUDA_0  };
  m["double4"]                             = {CUDA_0,   CUDA_130, CUDA_0  };
  m["double4_16a"]                         = {CUDA_130, CUDA_0,   CUDA_0  };
  m["double4_32a"]                         = {CUDA_130, CUDA_0,   CUDA_0  };
  m["longlong4"]                           = {CUDA_0,   CUDA_130, CUDA_0  };
  m["longlong4_16a"]                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["longlong4_32a"]                       = {CUDA_130, CUDA_0,   CUDA_0  };
  m["ulonglong4"]                          = {CUDA_0,   CUDA_130, CUDA_0  };
  m["ulonglong4_16a"]                      = {CUDA_130, CUDA_0,   CUDA_0  };
  m["ulonglong4_32a"]                      = {CUDA_130, CUDA_0,   CUDA_0  };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_DEVICE_TYPE_NAME_VER_MAP = []() {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["__half"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["__half2"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["__half_raw"]                          = {HIP_1090, HIP_0,    HIP_0   };
  m["__half2_raw"]                         = {HIP_1090, HIP_0,    HIP_0   };
  m["__hip_bfloat16"]                      = {HIP_5070, HIP_0,    HIP_0   };
  m["__hip_fp8_e4m3_fnuz"]                 = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8_storage_t"]                 = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x2_storage_t"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x4_storage_t"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8_e5m2_fnuz"]                 = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x2_e5m2_fnuz"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x2_e4m3_fnuz"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x4_e4m3_fnuz"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_saturation_t"]                  = {HIP_6020, HIP_0,    HIP_0   };
  m["__HIP_NOSAT"]                         = {HIP_6020, HIP_0,    HIP_0   };
  m["__HIP_SATFINITE"]                     = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8_interpretation_t"]          = {HIP_6020, HIP_0,    HIP_0   };
  m["__HIP_E4M3_FNUZ"]                     = {HIP_6020, HIP_0,    HIP_0   };
  m["__HIP_E5M2_FNUZ"]                     = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_fp8x4_e5m2_fnuz"]               = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_bfloat16_raw"]                  = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_bfloat162_raw"]                 = {HIP_6020, HIP_0,    HIP_0   };
  m["__hip_bfloat162"]                     = {HIP_5070, HIP_0,    HIP_0   };
  m["hip_bfloat16"]                        = {HIP_3050, HIP_0,    HIP_0   };
  m["HIPRT_INF_FP16"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_MAX_NORMAL_FP16"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_MIN_DENORM_FP16"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_NAN_FP16"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_NEG_ZERO_FP16"]                 = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_ONE_FP16"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["HIPRT_ZERO_FP16"]                     = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4_storage_t"]                 = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4x2_storage_t"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4x4_storage_t"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4_interpretation_t"]          = {HIP_7000, HIP_0,    HIP_0   };
  m["__HIP_E2M1"]                          = {HIP_7000, HIP_0,    HIP_0   };
  m["hipRoundMode"]                        = {HIP_7000, HIP_0,    HIP_0   };
  m["hipRoundNearest"]                     = {HIP_7000, HIP_0,    HIP_0   };
  m["hipRoundZero"]                        = {HIP_7000, HIP_0,    HIP_0   };
  m["hipRoundPosInf"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["hipRoundMinInf"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4_e2m1"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4x2_e2m1"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp4x4_e2m1"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["char1"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["char2"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["char3"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["char4"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["uchar1"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["uchar2"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["uchar3"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["uchar4"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["short1"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["short2"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["short3"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["short4"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["ushort1"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["ushort2"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["ushort3"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["ushort4"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["int1"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["int2"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["int3"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["int4"]                                = {HIP_1060, HIP_0,    HIP_0   };
  m["uint1"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["uint2"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["uint3"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["uint4"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["long1"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["long2"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["long3"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["long4"]                               = {HIP_1060, HIP_0,    HIP_0   };
  m["ulong1"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["ulong2"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["ulong3"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["ulong4"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["float1"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["float2"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["float3"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["float4"]                              = {HIP_1060, HIP_0,    HIP_0   };
  m["double1"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["double2"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["double3"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["double4"]                             = {HIP_1060, HIP_0,    HIP_0   };
  m["longlong1"]                           = {HIP_1060, HIP_0,    HIP_0   };
  m["longlong2"]                           = {HIP_1060, HIP_0,    HIP_0   };
  m["longlong3"]                           = {HIP_1060, HIP_0,    HIP_0   };
  m["longlong4"]                           = {HIP_1060, HIP_0,    HIP_0   };
  m["ulonglong1"]                          = {HIP_1060, HIP_0,    HIP_0   };
  m["ulonglong2"]                          = {HIP_1060, HIP_0,    HIP_0   };
  m["ulonglong3"]                          = {HIP_1060, HIP_0,    HIP_0   };
  m["ulonglong4"]                          = {HIP_1060, HIP_0,    HIP_0   };
  m["__hip_fp6_storage_t"]                 = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x2_storage_t"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x4_storage_t"]               = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6_e2m3"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6_e3m2"]                      = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x2_e2m3"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x2_e3m2"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x4_e2m3"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6x4_e3m2"]                    = {HIP_7000, HIP_0,    HIP_0   };
  m["__hip_fp6_interpretation_t"]          = {HIP_7000, HIP_0,    HIP_0   };
  m["__HIP_E2M3"]                          = {HIP_7000, HIP_0,    HIP_0   };
  m["__HIP_E3M2"]                          = {HIP_7000, HIP_0,    HIP_0   };

  m["rocblas_half"]                        = {HIP_1050, HIP_0,    HIP_0   };
  m["rocblas_bfloat16"]                    = {HIP_3050, HIP_0,    HIP_0   };

  return m;
}();
