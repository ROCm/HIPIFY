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
const std::map<llvm::StringRef, hipCounter> CUDA_RAND_FUNCTION_MAP {
  // RAND Host functions
  {"curandCreateGenerator",                         {"hiprandCreateGenerator",                         "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandCreateGeneratorHost",                     {"hiprandCreateGeneratorHost",                     "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandCreatePoissonDistribution",               {"hiprandCreatePoissonDistribution",               "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandDestroyDistribution",                     {"hiprandDestroyDistribution",                     "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandDestroyGenerator",                        {"hiprandDestroyGenerator",                        "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerate",                                {"hiprandGenerate",                                "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateLogNormal",                       {"hiprandGenerateLogNormal",                       "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateLogNormalDouble",                 {"hiprandGenerateLogNormalDouble",                 "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateLongLong",                        {"hiprandGenerateLongLong",                        "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGenerateNormal",                          {"hiprandGenerateNormal",                          "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateNormalDouble",                    {"hiprandGenerateNormalDouble",                    "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGeneratePoisson",                         {"hiprandGeneratePoisson",                         "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateSeeds",                           {"hiprandGenerateSeeds",                           "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateUniform",                         {"hiprandGenerateUniform",                         "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGenerateUniformDouble",                   {"hiprandGenerateUniformDouble",                   "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandGetDirectionVectors32",                   {"hiprandGetDirectionVectors32",                   "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGetDirectionVectors64",                   {"hiprandGetDirectionVectors64",                   "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGetProperty",                             {"hiprandGetProperty",                             "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGetScrambleConstants32",                  {"hiprandGetScrambleConstants32",                  "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGetScrambleConstants64",                  {"hiprandGetScrambleConstants64",                  "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandGetVersion",                              {"hiprandGetVersion",                              "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandSetGeneratorOffset",                      {"hiprandSetGeneratorOffset",                      "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandSetGeneratorOrdering",                    {"hiprandSetGeneratorOrdering",                    "", CONV_LIB_FUNC, API_RAND, 2, HIP_UNSUPPORTED}},
  {"curandSetPseudoRandomGeneratorSeed",            {"hiprandSetPseudoRandomGeneratorSeed",            "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandSetQuasiRandomGeneratorDimensions",       {"hiprandSetQuasiRandomGeneratorDimensions",       "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandSetStream",                               {"hiprandSetStream",                               "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandMakeMTGP32Constants",                     {"hiprandMakeMTGP32Constants",                     "", CONV_LIB_FUNC, API_RAND, 2}},
  {"curandMakeMTGP32KernelState",                   {"hiprandMakeMTGP32KernelState",                   "", CONV_LIB_FUNC, API_RAND, 2}},

  // RAND Device functions
  {"curand",                                        {"hiprand",                                        "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_init",                                   {"hiprand_init",                                   "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal",                             {"hiprand_log_normal",                             "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal_double",                      {"hiprand_log_normal_double",                      "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal2",                            {"hiprand_log_normal2",                            "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal2_double",                     {"hiprand_log_normal2_double",                     "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal4",                            {"hiprand_log_normal4",                            "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_log_normal4_double",                     {"hiprand_log_normal4_double",                     "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_mtgp32_single",                          {"hiprand_mtgp32_single",                          "", CONV_LIB_DEVICE_FUNC, API_RAND, 3, HIP_UNSUPPORTED}},
  {"curand_mtgp32_single_specific",                 {"hiprand_mtgp32_single_specific",                 "", CONV_LIB_DEVICE_FUNC, API_RAND, 3, HIP_UNSUPPORTED}},
  {"curand_mtgp32_specific",                        {"hiprand_mtgp32_specific",                        "", CONV_LIB_DEVICE_FUNC, API_RAND, 3, HIP_UNSUPPORTED}},
  {"curand_normal",                                 {"hiprand_normal",                                 "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_normal_double",                          {"hiprand_normal_double",                          "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_normal2",                                {"hiprand_normal2",                                "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_normal2_double",                         {"hiprand_normal2_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_normal4",                                {"hiprand_normal4",                                "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_normal4_double",                         {"hiprand_normal4_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_uniform",                                {"hiprand_uniform",                                "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_uniform_double",                         {"hiprand_uniform_double",                         "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_uniform2_double",                        {"hiprand_uniform2_double",                        "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_uniform4",                               {"hiprand_uniform4",                               "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_uniform4_double",                        {"hiprand_uniform4_double",                        "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_discrete",                               {"hiprand_discrete",                               "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_discrete4",                              {"hiprand_discrete4",                              "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_poisson",                                {"hiprand_poisson",                                "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_poisson4",                               {"hiprand_poisson4",                               "", CONV_LIB_DEVICE_FUNC, API_RAND, 3}},
  {"curand_Philox4x32_10",                          {"hiprand_Philox4x32_10",                          "", CONV_LIB_DEVICE_FUNC, API_RAND, 3, HIP_UNSUPPORTED}},
  // unchanged function names: skipahead, skipahead_sequence, skipahead_subsequence
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RAND_FUNCTION_VER_MAP {
};

const std::map<unsigned int, llvm::StringRef> CUDA_RAND_API_SECTION_MAP {
  {1, "CURAND Data types"},
  {2, "Host API Functions"},
  {3, "Device API Functions"},
};
