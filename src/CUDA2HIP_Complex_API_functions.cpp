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

// Maps the names of CUDA Complex API functions to the corresponding HIP functions
const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_FUNCTION_MAP {
  {"cuCrealf",               {"hipCrealf",               "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCimagf",               {"hipCimagf",               "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"make_cuFloatComplex",    {"make_hipFloatComplex",    "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuConjf",                {"hipConjf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCaddf",                {"hipCaddf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCsubf",                {"hipCsubf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCmulf",                {"hipCmulf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCdivf",                {"hipCdivf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCabsf",                {"hipCabsf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCreal",                {"hipCreal",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCimag",                {"hipCimag",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"make_cuDoubleComplex",   {"make_hipDoubleComplex",   "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuConj",                 {"hipConj",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCadd",                 {"hipCadd",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCsub",                 {"hipCsub",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCmul",                 {"hipCmul",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCdiv",                 {"hipCdiv",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCabs",                 {"hipCabs",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"make_cuComplex",         {"make_hipComplex",         "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuComplexFloatToDouble", {"hipComplexFloatToDouble", "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuComplexDoubleToFloat", {"hipComplexDoubleToFloat", "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCfmaf",                {"hipCfmaf",                "", CONV_COMPLEX, API_COMPLEX, 2}},
  {"cuCfma",                 {"hipCfma",                 "", CONV_COMPLEX, API_COMPLEX, 2}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_COMPLEX_FUNCTION_VER_MAP {
};

const std::map<llvm::StringRef, hipAPIversions> HIP_COMPLEX_FUNCTION_VER_MAP {
  {"hipCrealf",               {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCimagf",               {HIP_1060, HIP_0,    HIP_0   }},
  {"make_hipFloatComplex",    {HIP_1060, HIP_0,    HIP_0   }},
  {"hipConjf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCaddf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCsubf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCmulf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCdivf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCabsf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCreal",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCimag",                {HIP_1060, HIP_0,    HIP_0   }},
  {"make_hipDoubleComplex",   {HIP_1060, HIP_0,    HIP_0   }},
  {"hipConj",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCadd",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCsub",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCmul",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCdiv",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCabs",                 {HIP_1060, HIP_0,    HIP_0   }},
  {"make_hipComplex",         {HIP_1060, HIP_0,    HIP_0   }},
  {"hipComplexFloatToDouble", {HIP_1060, HIP_0,    HIP_0   }},
  {"hipComplexDoubleToFloat", {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCfmaf",                {HIP_1060, HIP_0,    HIP_0   }},
  {"hipCfma",                 {HIP_1060, HIP_0,    HIP_0   }},
};

const std::map<unsigned int, llvm::StringRef> CUDA_COMPLEX_API_SECTION_MAP {
  {1, "cuComplex Data types"},
  {2, "cuComplex API functions"},
};
