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
const std::map<llvm::StringRef, hipCounter> CUDA_RAND_TYPE_NAME_MAP {
  // RAND Host types
  {"curandStatus",                  {"hiprandStatus_t",                "", CONV_TYPE, API_RAND, 1}},
  {"curandStatus_t",                {"hiprandStatus_t",                "", CONV_TYPE, API_RAND, 1}},
  {"curandRngType",                 {"hiprandRngType_t",               "", CONV_TYPE, API_RAND, 1}},
  {"curandRngType_t",               {"hiprandRngType_t",               "", CONV_TYPE, API_RAND, 1}},
  {"curandGenerator_st",            {"hiprandGenerator_st",            "", CONV_TYPE, API_RAND, 1}},
  {"curandGenerator_t",             {"hiprandGenerator_t",             "", CONV_TYPE, API_RAND, 1}},
  {"curandDirectionVectorSet",      {"hiprandDirectionVectorSet_t",    "", CONV_TYPE, API_RAND, 1}},
  {"curandDirectionVectorSet_t",    {"hiprandDirectionVectorSet_t",    "", CONV_TYPE, API_RAND, 1}},
  {"curandOrdering",                {"hiprandOrdering_t",              "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandOrdering_t",              {"hiprandOrdering_t",              "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistribution_st",         {"hiprandDistribution_st",         "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2V_st",         {"hiprandHistogramM2V_st",         "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistribution_t",          {"hiprandDistribution_t",          "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2V_t",          {"hiprandDistribution_t",          "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistributionShift_st",    {"hiprandDistributionShift_st",    "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistributionShift_t",     {"hiprandDistributionShift_t",     "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistributionM2Shift_st",  {"hiprandDistributionM2Shift_st",  "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDistributionM2Shift_t",   {"hiprandDistributionM2Shift_t",   "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2_st",          {"hiprandHistogramM2_st",          "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2_t",           {"hiprandHistogramM2_t",           "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2K_st",         {"hiprandHistogramM2K_st",         "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandHistogramM2K_t",          {"hiprandHistogramM2K_t",          "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDiscreteDistribution_st", {"hiprandDiscreteDistribution_st", "", CONV_TYPE, API_RAND, 1}},
  {"curandDiscreteDistribution_t",  {"hiprandDiscreteDistribution_t",  "", CONV_TYPE, API_RAND, 1}},
  {"curandMethod",                  {"hiprandMethod_t",                "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandMethod_t",                {"hiprandMethod_t",                "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandDirectionVectors32_t",    {"hiprandDirectionVectors32_t",    "", CONV_TYPE, API_RAND, 1}},
  {"curandDirectionVectors64_t",    {"hiprandDirectionVectors64_t",    "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},

  // RAND types for Device functions
  {"curandStateMtgp32",             {"hiprandStateMtgp32",             "", CONV_TYPE, API_RAND, 1}},
  {"curandStateMtgp32_t",           {"hiprandStateMtgp32_t",           "", CONV_TYPE, API_RAND, 1}},
  {"curandStateScrambledSobol64",   {"hiprandStateScrambledSobol64",   "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateScrambledSobol64_t", {"hiprandStateScrambledSobol64_t", "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateSobol64",            {"hiprandStateSobol64",            "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateSobol64_t",          {"hiprandStateSobol64_t",          "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateScrambledSobol32",   {"hiprandStateScrambledSobol32",   "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateScrambledSobol32_t", {"hiprandStateScrambledSobol32_t", "", CONV_TYPE, API_RAND, 1, HIP_UNSUPPORTED}},
  {"curandStateSobol32",            {"hiprandStateSobol32",            "", CONV_TYPE, API_RAND, 1}},
  {"curandStateSobol32_t",          {"hiprandStateSobol32_t",          "", CONV_TYPE, API_RAND, 1}},
  {"curandStateMRG32k3a",           {"hiprandStateMRG32k3a",           "", CONV_TYPE, API_RAND, 1}},
  {"curandStateMRG32k3a_t",         {"hiprandStateMRG32k3a_t",         "", CONV_TYPE, API_RAND, 1}},
  {"curandStatePhilox4_32_10",      {"hiprandStatePhilox4_32_10",      "", CONV_TYPE, API_RAND, 1}},
  {"curandStatePhilox4_32_10_t",    {"hiprandStatePhilox4_32_10_t",    "", CONV_TYPE, API_RAND, 1}},
  {"curandStateXORWOW",             {"hiprandStateXORWOW",             "", CONV_TYPE, API_RAND, 1}},
  {"curandStateXORWOW_t",           {"hiprandStateXORWOW_t",           "", CONV_TYPE, API_RAND, 1}},
  {"curandState",                   {"hiprandState",                   "", CONV_TYPE, API_RAND, 1}},
  {"curandState_t",                 {"hiprandState_t",                 "", CONV_TYPE, API_RAND, 1}},

  // RAND function call status types (enum curandStatus)
  {"CURAND_STATUS_SUCCESS",                         {"HIPRAND_STATUS_SUCCESS",                         "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_VERSION_MISMATCH",                {"HIPRAND_STATUS_VERSION_MISMATCH",                "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_NOT_INITIALIZED",                 {"HIPRAND_STATUS_NOT_INITIALIZED",                 "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_ALLOCATION_FAILED",               {"HIPRAND_STATUS_ALLOCATION_FAILED",               "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_TYPE_ERROR",                      {"HIPRAND_STATUS_TYPE_ERROR",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_OUT_OF_RANGE",                    {"HIPRAND_STATUS_OUT_OF_RANGE",                    "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_LENGTH_NOT_MULTIPLE",             {"HIPRAND_STATUS_LENGTH_NOT_MULTIPLE",             "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",       {"HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",       "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_LAUNCH_FAILURE",                  {"HIPRAND_STATUS_LAUNCH_FAILURE",                  "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_PREEXISTING_FAILURE",             {"HIPRAND_STATUS_PREEXISTING_FAILURE",             "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_INITIALIZATION_FAILED",           {"HIPRAND_STATUS_INITIALIZATION_FAILED",           "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_ARCH_MISMATCH",                   {"HIPRAND_STATUS_ARCH_MISMATCH",                   "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_STATUS_INTERNAL_ERROR",                  {"HIPRAND_STATUS_INTERNAL_ERROR",                  "", CONV_NUMERIC_LITERAL, API_RAND, 1}},

  // RAND generator types (enum curandRngType)
  {"CURAND_RNG_TEST",                               {"HIPRAND_RNG_TEST",                               "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_DEFAULT",                     {"HIPRAND_RNG_PSEUDO_DEFAULT",                     "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_XORWOW",                      {"HIPRAND_RNG_PSEUDO_XORWOW",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_MRG32K3A",                    {"HIPRAND_RNG_PSEUDO_MRG32K3A",                    "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_MTGP32",                      {"HIPRAND_RNG_PSEUDO_MTGP32",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_MT19937",                     {"HIPRAND_RNG_PSEUDO_MT19937",                     "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_PSEUDO_PHILOX4_32_10",               {"HIPRAND_RNG_PSEUDO_PHILOX4_32_10",               "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_QUASI_DEFAULT",                      {"HIPRAND_RNG_QUASI_DEFAULT",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_QUASI_SOBOL32",                      {"HIPRAND_RNG_QUASI_SOBOL32",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL32",            {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32",            "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_QUASI_SOBOL64",                      {"HIPRAND_RNG_QUASI_SOBOL64",                      "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL64",            {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64",            "", CONV_NUMERIC_LITERAL, API_RAND, 1}},

  // RAND ordering of results in memory (enum curandOrdering)
  {"CURAND_ORDERING_PSEUDO_BEST",                   {"HIPRAND_ORDERING_PSEUDO_BEST",                   "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_DEFAULT",                {"HIPRAND_ORDERING_PSEUDO_DEFAULT",                "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_SEEDED",                 {"HIPRAND_ORDERING_PSEUDO_SEEDED",                 "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_LEGACY",                 {"HIPRAND_ORDERING_PSEUDO_LEGACY",                 "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_PSEUDO_DYNAMIC",                {"HIPRAND_ORDERING_PSEUDO_DYNAMIC",                "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ORDERING_QUASI_DEFAULT",                 {"HIPRAND_ORDERING_QUASI_DEFAULT",                 "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},

  // RAND choice of direction vector set (enum curandDirectionVectorSet)
  {"CURAND_DIRECTION_VECTORS_32_JOEKUO6",           {"HIPRAND_DIRECTION_VECTORS_32_JOEKUO6",           "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_DIRECTION_VECTORS_64_JOEKUO6",           {"HIPRAND_DIRECTION_VECTORS_64_JOEKUO6",           "", CONV_NUMERIC_LITERAL, API_RAND, 1}},
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", "", CONV_NUMERIC_LITERAL, API_RAND, 1}},

  // RAND method (enum curandMethod)
  {"CURAND_CHOOSE_BEST",                            {"HIPRAND_CHOOSE_BEST",                            "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_ITR",                                    {"HIPRAND_ITR",                                    "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_KNUTH",                                  {"HIPRAND_KNUTH",                                  "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_HITR",                                   {"HIPRAND_HITR",                                   "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_M1",                                     {"HIPRAND_M1",                                     "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_M2",                                     {"HIPRAND_M2",                                     "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_BINARY_SEARCH",                          {"HIPRAND_BINARY_SEARCH",                          "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_DISCRETE_GAUSS",                         {"HIPRAND_DISCRETE_GAUSS",                         "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_REJECTION",                              {"HIPRAND_REJECTION",                              "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_DEVICE_API",                             {"HIPRAND_DEVICE_API",                             "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_FAST_REJECTION",                         {"HIPRAND_FAST_REJECTION",                         "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_3RD",                                    {"HIPRAND_3RD",                                    "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_DEFINITION",                             {"HIPRAND_DEFINITION",                             "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
  {"CURAND_POISSON",                                {"HIPRAND_POISSON",                                "", CONV_NUMERIC_LITERAL, API_RAND, 1, HIP_UNSUPPORTED}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_RAND_TYPE_NAME_VER_MAP {
  {"CURAND_ORDERING_PSEUDO_LEGACY",                 {CUDA_110, CUDA_0, CUDA_0}},
  {"CURAND_ORDERING_PSEUDO_DYNAMIC",                {CUDA_115, CUDA_0, CUDA_0}},
};

const std::map<llvm::StringRef, hipAPIversions> HIP_RAND_TYPE_NAME_VER_MAP {
  {"hiprandStatus_t",                                {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandRngType_t",                               {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandGenerator_st",                            {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandGenerator_t",                             {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandDiscreteDistribution_st",                 {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandDiscreteDistribution_t",                  {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandDirectionVectors32_t",                    {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandStateMtgp32",                             {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStateMtgp32_t",                           {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandStateSobol32",                            {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStateSobol32_t",                          {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandStateMRG32k3a",                           {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStateMRG32k3a_t",                         {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandStatePhilox4_32_10",                      {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStatePhilox4_32_10_t",                    {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStateXORWOW",                             {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandStateXORWOW_t",                           {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandState",                                   {HIP_1080, HIP_0,    HIP_0   }},
  {"hiprandState_t",                                 {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_SUCCESS",                         {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_VERSION_MISMATCH",                {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_NOT_INITIALIZED",                 {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_ALLOCATION_FAILED",               {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_TYPE_ERROR",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_OUT_OF_RANGE",                    {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_LENGTH_NOT_MULTIPLE",             {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED",       {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_LAUNCH_FAILURE",                  {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_PREEXISTING_FAILURE",             {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_INITIALIZATION_FAILED",           {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_ARCH_MISMATCH",                   {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_STATUS_INTERNAL_ERROR",                  {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_TEST",                               {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_DEFAULT",                     {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_XORWOW",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_MRG32K3A",                    {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_MTGP32",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_MT19937",                     {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_PSEUDO_PHILOX4_32_10",               {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_QUASI_DEFAULT",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_QUASI_SOBOL32",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32",            {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_QUASI_SOBOL64",                      {HIP_1050, HIP_0,    HIP_0   }},
  {"HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64",            {HIP_1050, HIP_0,    HIP_0   }},
  {"hiprandDirectionVectorSet_t",                    {HIP_6000, HIP_0,    HIP_0,  }},
  {"HIPRAND_DIRECTION_VECTORS_32_JOEKUO6",           {HIP_6000, HIP_0,    HIP_0,  }},
  {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", {HIP_6000, HIP_0,    HIP_0,  }},
  {"HIPRAND_DIRECTION_VECTORS_64_JOEKUO6",           {HIP_6000, HIP_0,    HIP_0,  }},
  {"HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", {HIP_6000, HIP_0,    HIP_0,  }},
};
