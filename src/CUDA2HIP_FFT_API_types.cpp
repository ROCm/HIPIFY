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
const std::map<llvm::StringRef, hipCounter> CUDA_FFT_TYPE_NAME_MAP {

  // cuFFT defines
  {"CUFFT_FORWARD",                               {"HIPFFT_FORWARD",                               "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  // -1
  {"CUFFT_INVERSE",                               {"HIPFFT_BACKWARD",                              "", CONV_NUMERIC_LITERAL, API_FFT, 1}}, //  1
  {"CUFFT_COMPATIBILITY_DEFAULT",                 {"HIPFFT_COMPATIBILITY_DEFAULT",                 "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  CUFFT_COMPATIBILITY_FFTW_PADDING
  {"MAX_CUFFT_ERROR",                             {"HIPFFT_MAX_ERROR",                             "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x11

  // cuFFT enums
  {"cufftResult_t",                               {"hipfftResult_t",                               "", CONV_TYPE, API_FFT, 1}},
  {"cufftResult",                                 {"hipfftResult",                                 "", CONV_TYPE, API_FFT, 1}},
  {"CUFFT_SUCCESS",                               {"HIPFFT_SUCCESS",                               "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x0  0
  {"CUFFT_INVALID_PLAN",                          {"HIPFFT_INVALID_PLAN",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x1  1
  {"CUFFT_ALLOC_FAILED",                          {"HIPFFT_ALLOC_FAILED",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x2  2
  {"CUFFT_INVALID_TYPE",                          {"HIPFFT_INVALID_TYPE",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x3  3
  {"CUFFT_INVALID_VALUE",                         {"HIPFFT_INVALID_VALUE",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x4  4
  {"CUFFT_INTERNAL_ERROR",                        {"HIPFFT_INTERNAL_ERROR",                        "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x5  5
  {"CUFFT_EXEC_FAILED",                           {"HIPFFT_EXEC_FAILED",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x6  6
  {"CUFFT_SETUP_FAILED",                          {"HIPFFT_SETUP_FAILED",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x7  7
  {"CUFFT_INVALID_SIZE",                          {"HIPFFT_INVALID_SIZE",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x8  8
  {"CUFFT_UNALIGNED_DATA",                        {"HIPFFT_UNALIGNED_DATA",                        "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x9  9
  {"CUFFT_INCOMPLETE_PARAMETER_LIST",             {"HIPFFT_INCOMPLETE_PARAMETER_LIST",             "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0xA  10
  {"CUFFT_INVALID_DEVICE",                        {"HIPFFT_INVALID_DEVICE",                        "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0xB  11
  {"CUFFT_PARSE_ERROR",                           {"HIPFFT_PARSE_ERROR",                           "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0xC  12
  {"CUFFT_NO_WORKSPACE",                          {"HIPFFT_NO_WORKSPACE",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0xD  13
  {"CUFFT_NOT_IMPLEMENTED",                       {"HIPFFT_NOT_IMPLEMENTED",                       "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0xE  14
  {"CUFFT_LICENSE_ERROR",                         {"HIPFFT_LICENSE_ERROR",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_NOT_SUPPORTED",                         {"HIPFFT_NOT_SUPPORTED",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x10 16

  {"cufftType_t",                                 {"hipfftType_t",                                 "", CONV_TYPE, API_FFT, 1}},
  {"cufftType",                                   {"hipfftType",                                   "", CONV_TYPE, API_FFT, 1}},
  {"CUFFT_R2C",                                   {"HIPFFT_R2C",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x2a
  {"CUFFT_C2R",                                   {"HIPFFT_C2R",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x2c
  {"CUFFT_C2C",                                   {"HIPFFT_C2C",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x29
  {"CUFFT_D2Z",                                   {"HIPFFT_D2Z",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x6a
  {"CUFFT_Z2D",                                   {"HIPFFT_Z2D",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x6c
  {"CUFFT_Z2Z",                                   {"HIPFFT_Z2Z",                                   "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x69

  {"cufftCompatibility_t",                        {"hipfftCompatibility_t",                        "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftCompatibility",                          {"hipfftCompatibility",                          "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_COMPATIBILITY_FFTW_PADDING",            {"HIPFFT_COMPATIBILITY_FFTW_PADDING",            "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x01

  {"cufftXtSubFormat_t",                          {"hipfftXtSubFormat_t",                          "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftXtSubFormat",                            {"hipfftXtSubFormat",                            "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_XT_FORMAT_INPUT",                       {"HIPFFT_XT_FORMAT_INPUT",                       "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x00
  {"CUFFT_XT_FORMAT_OUTPUT",                      {"HIPFFT_XT_FORMAT_OUTPUT",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x01
  {"CUFFT_XT_FORMAT_INPLACE",                     {"HIPFFT_XT_FORMAT_INPLACE",                     "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x02
  {"CUFFT_XT_FORMAT_INPLACE_SHUFFLED",            {"HIPFFT_XT_FORMAT_INPLACE_SHUFFLED",            "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x03
  {"CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED",           {"HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED",           "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x04
  {"CUFFT_XT_FORMAT_DISTRIBUTED_INPUT",           {"HIPFFT_XT_FORMAT_DISTRIBUTED_INPUT",           "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x05
  {"CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",          {"HIPFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",          "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x06
  {"CUFFT_FORMAT_UNDEFINED",                      {"HIPFFT_FORMAT_UNDEFINED",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x07

  {"cufftXtCopyType_t",                           {"hipfftXtCopyType_t",                           "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftXtCopyType",                             {"hipfftXtCopyType",                             "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_COPY_HOST_TO_DEVICE",                   {"HIPFFT_COPY_HOST_TO_DEVICE",                   "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x00
  {"CUFFT_COPY_DEVICE_TO_HOST",                   {"HIPFFT_COPY_DEVICE_TO_HOST",                   "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x01
  {"CUFFT_COPY_DEVICE_TO_DEVICE",                 {"HIPFFT_COPY_DEVICE_TO_DEVICE",                 "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x02
  {"CUFFT_COPY_UNDEFINED",                        {"HIPFFT_COPY_UNDEFINED",                        "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x03

  {"cufftXtQueryType_t",                          {"hipfftXtQueryType_t",                          "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftXtQueryType",                            {"hipfftXtQueryType",                            "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_QUERY_1D_FACTORS",                      {"HIPFFT_QUERY_1D_FACTORS",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x00
  {"CUFFT_QUERY_UNDEFINED",                       {"HIPFFT_QUERY_UNDEFINED",                       "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0x01

  {"cufftXtWorkAreaPolicy_t",                     {"hipfftXtWorkAreaPolicy_t",                     "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftXtWorkAreaPolicy",                       {"hipfftXtWorkAreaPolicy",                       "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"CUFFT_WORKAREA_MINIMAL",                      {"HIPFFT_WORKAREA_MINIMAL",                      "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  0
  {"CUFFT_WORKAREA_USER",                         {"HIPFFT_WORKAREA_USER",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  1
  {"CUFFT_WORKAREA_PERFORMANCE",                  {"HIPFFT_WORKAREA_PERFORMANCE",                  "", CONV_NUMERIC_LITERAL, API_FFT, 1, UNSUPPORTED}},  //  2

  {"cufftXtCallbackType_t",                       {"hipfftXtCallbackType_t",                       "", CONV_TYPE, API_FFT, 1}},
  {"cufftXtCallbackType",                         {"hipfftXtCallbackType",                         "", CONV_TYPE, API_FFT, 1}},
  {"CUFFT_CB_LD_COMPLEX",                         {"HIPFFT_CB_LD_COMPLEX",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x0
  {"CUFFT_CB_LD_COMPLEX_DOUBLE",                  {"HIPFFT_CB_LD_COMPLEX_DOUBLE",                  "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x1
  {"CUFFT_CB_LD_REAL",                            {"HIPFFT_CB_LD_REAL",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x2
  {"CUFFT_CB_LD_REAL_DOUBLE",                     {"HIPFFT_CB_LD_REAL_DOUBLE",                     "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x3
  {"CUFFT_CB_ST_COMPLEX",                         {"HIPFFT_CB_ST_COMPLEX",                         "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x4
  {"CUFFT_CB_ST_COMPLEX_DOUBLE",                  {"HIPFFT_CB_ST_COMPLEX_DOUBLE",                  "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x5
  {"CUFFT_CB_ST_REAL",                            {"HIPFFT_CB_ST_REAL",                            "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x6
  {"CUFFT_CB_ST_REAL_DOUBLE",                     {"HIPFFT_CB_ST_REAL_DOUBLE",                     "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x7
  {"CUFFT_CB_UNDEFINED",                          {"HIPFFT_CB_UNDEFINED",                          "", CONV_NUMERIC_LITERAL, API_FFT, 1}},  //  0x7

  // cuFFT types
  {"cufftReal",                                   {"hipfftReal",                                   "", CONV_TYPE, API_FFT, 1}},
  {"cufftDoubleReal",                             {"hipfftDoubleReal",                             "", CONV_TYPE, API_FFT, 1}},
  {"cufftComplex",                                {"hipfftComplex",                                "", CONV_TYPE, API_FFT, 1}},
  {"cufftDoubleComplex",                          {"hipfftDoubleComplex",                          "", CONV_TYPE, API_FFT, 1}},
  {"cufftHandle",                                 {"hipfftHandle",                                 "", CONV_TYPE, API_FFT, 1}},
  {"cufftXt1dFactors_t",                          {"hipfftXt1dFactors_t",                          "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftXt1dFactors",                            {"hipfftXt1dFactors",                            "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftBox3d_t",                                {"hipfftBox3d_t",                                "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
  {"cufftBox3d",                                  {"hipfftBox3d",                                  "", CONV_TYPE, API_FFT, 1, UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_TYPE_NAME_VER_MAP {
  {"CUFFT_NOT_SUPPORTED",                         {CUDA_80,  CUDA_0, CUDA_0}},
  {"cufftXtWorkAreaPolicy_t",                     {CUDA_92,  CUDA_0, CUDA_0}},
  {"cufftXtWorkAreaPolicy",                       {CUDA_92,  CUDA_0, CUDA_0}},
  {"CUFFT_WORKAREA_MINIMAL",                      {CUDA_92,  CUDA_0, CUDA_0}},
  {"CUFFT_WORKAREA_USER",                         {CUDA_92,  CUDA_0, CUDA_0}},
  {"CUFFT_XT_FORMAT_DISTRIBUTED_INPUT",           {CUDA_118, CUDA_0, CUDA_0}},
  {"CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT",          {CUDA_118, CUDA_0, CUDA_0}},
  {"cufftBox3d_t",                                {CUDA_118, CUDA_0, CUDA_0}},
  {"cufftBox3d",                                  {CUDA_118, CUDA_0, CUDA_0}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_FFT_TYPE_NAME_VER_MAP {
  {"HIPFFT_FORWARD",                              {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_BACKWARD",                             {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftResult_t",                              {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftResult",                                {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_SUCCESS",                              {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INVALID_PLAN",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_ALLOC_FAILED",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INVALID_TYPE",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INVALID_VALUE",                        {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INTERNAL_ERROR",                       {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_EXEC_FAILED",                          {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_SETUP_FAILED",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INVALID_SIZE",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_UNALIGNED_DATA",                       {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INCOMPLETE_PARAMETER_LIST",            {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_INVALID_DEVICE",                       {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_PARSE_ERROR",                          {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_NO_WORKSPACE",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_NOT_IMPLEMENTED",                      {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_NOT_SUPPORTED",                        {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftType_t",                                {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftType",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_R2C",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_C2R",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_C2C",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_D2Z",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_Z2D",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"HIPFFT_Z2Z",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftReal",                                  {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftDoubleReal",                            {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftComplex",                               {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftDoubleComplex",                         {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftHandle",                                {HIP_1070, HIP_0,    HIP_0   }},
  {"hipfftXtCallbackType_t",                      {HIP_4030, HIP_0,    HIP_0   }},
  {"hipfftXtCallbackType",                        {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_LD_COMPLEX",                        {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_LD_COMPLEX_DOUBLE",                 {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_LD_REAL",                           {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_LD_REAL_DOUBLE",                    {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_ST_COMPLEX",                        {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_ST_COMPLEX_DOUBLE",                 {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_ST_REAL",                           {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_ST_REAL_DOUBLE",                    {HIP_4030, HIP_0,    HIP_0   }},
  {"HIPFFT_CB_UNDEFINED",                         {HIP_4030, HIP_0,    HIP_0   }},
};
