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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGE  S OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Maps the names of CUDA SPARSELt API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_SPARSELT_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  // 1. Structs
  m["cusparseLtHandle_t"]                                                = {"hipsparseLtHandle_t",                           "", CONV_TYPE, API_SPARSELT, 1};
  m["cusparseLtMatDescriptor_t"]                                         = {"hipsparseLtMatDescriptor_t",                    "", CONV_TYPE, API_SPARSELT, 1};
  m["cusparseLtMatmulDescriptor_t"]                                      = {"hipsparseLtMatmulDescriptor_t",                 "", CONV_TYPE, API_SPARSELT, 1};
  m["cusparseLtMatmulAlgSelection_t"]                                    = {"hipsparseLtMatmulAlgSelection_t",               "", CONV_TYPE, API_SPARSELT, 1};
  m["cusparseLtMatmulPlan_t"]                                            = {"hipsparseLtMatmulPlan_t",                       "", CONV_TYPE, API_SPARSELT, 1};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSELT_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cusparseLtHandle_t"]                                                = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatDescriptor_t"]                                         = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulDescriptor_t"]                                      = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgSelection_t"]                                    = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulPlan_t"]                                            = {CUSPARSELT_001, CUDA_0      , CUDA_0      };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSELT_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipsparseLtHandle_t"]                                               = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatDescriptor_t"]                                        = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulDescriptor_t"]                                     = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulAlgSelection_t"]                                   = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulPlan_t"]                                           = {HIP_7020, HIP_0,    HIP_0    };

  return m;
}();
