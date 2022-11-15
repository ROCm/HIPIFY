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

// Maps the names of CUDA Complex API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_TYPE_NAME_MAP {
  {"cuFloatComplex",  {"hipFloatComplex",  "rocblas_float_complex",  CONV_TYPE, API_COMPLEX, 1}},
  {"cuDoubleComplex", {"hipDoubleComplex", "rocblas_double_complex", CONV_TYPE, API_COMPLEX, 1}},
  {"cuComplex",       {"hipComplex",       "rocblas_float_complex",  CONV_TYPE, API_COMPLEX, 1}},
};

const std::map<llvm::StringRef, cudaAPIversions> CUDA_COMPLEX_TYPE_NAME_VER_MAP {
};

const std::map<llvm::StringRef, hipAPIversions> HIP_COMPLEX_TYPE_NAME_VER_MAP {
  {"hipFloatComplex", {HIP_1060, HIP_0,    HIP_0   }},
  {"hipDoubleComplex",{HIP_1060, HIP_0,    HIP_0   }},
  {"hipComplex",      {HIP_1060, HIP_0,    HIP_0   }},
};
