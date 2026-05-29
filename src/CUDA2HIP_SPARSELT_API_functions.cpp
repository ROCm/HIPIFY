/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// Maps the names of CUDA SPARSELt API functions to the corresponding HIP functions
const std::map<llvm::StringRef, hipCounter> CUDA_SPARSELT_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  m["cusparseLtInit"]                                                 = {"hipsparseLtInit",                                    "", CONV_LIB_FUNC, API_SPARSELT, 2};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSELT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cusparseLtInit"]                                                 = {CUSPARSELT_001, CUDA_0      , CUDA_0      };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSELT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipsparseLtInit"]                                                = {HIP_7020, HIP_0,    HIP_0   };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_SPARSELT_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIChangedVersions> m;

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_SPARSELT_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIChangedVersions> m;

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_SPARSELT_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1]                                                              = "CUSPARSELT Data types";
  m[2]                                                              = "CUSPARSELT Function Reference";

  return m;
}();
