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
const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_FUNCTION_MAP = []() {
  std::map<llvm::StringRef, hipCounter> m;

  m["cuCrealf"]                = {"hipCrealf",               "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCimagf"]                = {"hipCimagf",               "", CONV_COMPLEX, API_COMPLEX, 2};
  m["make_cuFloatComplex"]     = {"make_hipFloatComplex",    "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuConjf"]                 = {"hipConjf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCaddf"]                 = {"hipCaddf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCsubf"]                 = {"hipCsubf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCmulf"]                 = {"hipCmulf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCdivf"]                 = {"hipCdivf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCabsf"]                 = {"hipCabsf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCreal"]                 = {"hipCreal",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCimag"]                 = {"hipCimag",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["make_cuDoubleComplex"]    = {"make_hipDoubleComplex",   "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuConj"]                  = {"hipConj",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCadd"]                  = {"hipCadd",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCsub"]                  = {"hipCsub",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCmul"]                  = {"hipCmul",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCdiv"]                  = {"hipCdiv",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCabs"]                  = {"hipCabs",                 "", CONV_COMPLEX, API_COMPLEX, 2};
  m["make_cuComplex"]          = {"make_hipComplex",         "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuComplexFloatToDouble"]  = {"hipComplexFloatToDouble", "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuComplexDoubleToFloat"]  = {"hipComplexDoubleToFloat", "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCfmaf"]                 = {"hipCfmaf",                "", CONV_COMPLEX, API_COMPLEX, 2};
  m["cuCfma"]                  = {"hipCfma",                 "", CONV_COMPLEX, API_COMPLEX, 2};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_COMPLEX_FUNCTION_VER_MAP = []() {
  std::map<llvm::StringRef, cudaAPIversions> m;
  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_COMPLEX_FUNCTION_VER_MAP = []() {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipCrealf"]               = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCimagf"]               = {HIP_1060, HIP_0,    HIP_0   };
  m["make_hipFloatComplex"]    = {HIP_1060, HIP_0,    HIP_0   };
  m["hipConjf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCaddf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCsubf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCmulf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCdivf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCabsf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCreal"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCimag"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["make_hipDoubleComplex"]   = {HIP_1060, HIP_0,    HIP_0   };
  m["hipConj"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCadd"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCsub"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCmul"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCdiv"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCabs"]                 = {HIP_1060, HIP_0,    HIP_0   };
  m["make_hipComplex"]         = {HIP_1060, HIP_0,    HIP_0   };
  m["hipComplexFloatToDouble"] = {HIP_1060, HIP_0,    HIP_0   };
  m["hipComplexDoubleToFloat"] = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCfmaf"]                = {HIP_1060, HIP_0,    HIP_0   };
  m["hipCfma"]                 = {HIP_1060, HIP_0,    HIP_0   };

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_COMPLEX_API_SECTION_MAP = []() {
  std::map<unsigned int, llvm::StringRef> m;

  m[1]                         = "cuComplex Data types";
  m[2]                         = "cuComplex API functions";

  return m;
}();
