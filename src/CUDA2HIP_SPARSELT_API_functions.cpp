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
  m["cusparseLtDestroy"]                                              = {"hipsparseLtDestroy",                                 "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtDenseDescriptorInit"]                                  = {"hipsparseLtDenseDescriptorInit",                     "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtStructuredDescriptorInit"]                             = {"hipsparseLtStructuredDescriptorInit",                "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulDescriptorInit"]                                 = {"hipsparseLtMatmulDescriptorInit",                    "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulAlgSelectionInit"]                               = {"hipsparseLtMatmulAlgSelectionInit",                  "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulAlgSetAttribute"]                                = {"hipsparseLtMatmulAlgSetAttribute",                   "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulAlgGetAttribute"]                                = {"hipsparseLtMatmulAlgGetAttribute",                   "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulGetWorkspace"]                                   = {"hipsparseLtMatmulGetWorkspace",                      "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulPlanInit"]                                       = {"hipsparseLtMatmulPlanInit",                          "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulPlanDestroy"]                                    = {"hipsparseLtMatmulPlanDestroy",                       "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmul"]                                               = {"hipsparseLtMatmul",                                  "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulSearch"]                                         = {"hipsparseLtMatmulSearch",                            "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMAPrune"]                                           = {"hipsparseLtSpMMAPrune",                              "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMAPruneCheck"]                                      = {"hipsparseLtSpMMAPruneCheck",                         "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMACompressedSize"]                                  = {"hipsparseLtSpMMACompressedSize",                     "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMACompress"]                                        = {"hipsparseLtSpMMACompress",                           "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatDescriptorDestroy"]                                 = {"hipsparseLtMatDescriptorDestroy",                    "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMAPrune2"]                                          = {"hipsparseLtSpMMAPrune2",                             "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMAPruneCheck2"]                                     = {"hipsparseLtSpMMAPruneCheck2",                        "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMACompressedSize2"]                                 = {"hipsparseLtSpMMACompressedSize2",                    "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtSpMMACompress2"]                                       = {"hipsparseLtSpMMACompress2",                          "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatDescSetAttribute"]                                  = {"hipsparseLtMatDescSetAttribute",                     "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatDescGetAttribute"]                                  = {"hipsparseLtMatDescGetAttribute",                     "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulDescSetAttribute"]                               = {"hipsparseLtMatmulDescSetAttribute",                  "", CONV_LIB_FUNC, API_SPARSELT, 2};
  m["cusparseLtMatmulDescGetAttribute"]                               = {"hipsparseLtMatmulDescGetAttribute",                  "", CONV_LIB_FUNC, API_SPARSELT, 2};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSELT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cusparseLtInit"]                                                 = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtDestroy"]                                              = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtDenseDescriptorInit"]                                  = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtStructuredDescriptorInit"]                             = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulDescriptorInit"]                                 = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgSelectionInit"]                               = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgSetAttribute"]                                = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgGetAttribute"]                                = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulGetWorkspace"]                                   = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulPlanInit"]                                       = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulPlanDestroy"]                                    = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmul"]                                               = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulSearch"]                                         = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMAPrune"]                                           = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMAPruneCheck"]                                      = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMACompressedSize"]                                  = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMACompress"]                                        = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatDescriptorDestroy"]                                 = {CUSPARSELT_010, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMAPrune2"]                                          = {CUSPARSELT_010, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMAPruneCheck2"]                                     = {CUSPARSELT_010, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMACompressedSize2"]                                 = {CUSPARSELT_010, CUDA_0      , CUDA_0      };
  m["cusparseLtSpMMACompress2"]                                       = {CUSPARSELT_010, CUDA_0      , CUDA_0      };
  m["cusparseLtMatDescSetAttribute"]                                  = {CUSPARSELT_020, CUDA_0      , CUDA_0      };
  m["cusparseLtMatDescGetAttribute"]                                  = {CUSPARSELT_020, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulDescSetAttribute"]                               = {CUSPARSELT_020, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulDescGetAttribute"]                               = {CUSPARSELT_020, CUDA_0      , CUDA_0      };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSELT_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipsparseLtInit"]                                                = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtDestroy"]                                             = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtDenseDescriptorInit"]                                 = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtStructuredDescriptorInit"]                            = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulDescriptorInit"]                                = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulAlgSelectionInit"]                              = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulAlgSetAttribute"]                               = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulAlgGetAttribute"]                               = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulGetWorkspace"]                                  = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulPlanInit"]                                      = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulPlanDestroy"]                                   = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmul"]                                              = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulSearch"]                                        = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMAPrune"]                                          = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMAPruneCheck"]                                     = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMACompressedSize"]                                 = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMACompress"]                                       = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatDescriptorDestroy"]                                = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMAPrune2"]                                         = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMAPruneCheck2"]                                    = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMACompressedSize2"]                                = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtSpMMACompress2"]                                      = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatDescSetAttribute"]                                 = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatDescGetAttribute"]                                 = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulDescSetAttribute"]                              = {HIP_7100, HIP_0,    HIP_0   };
  m["hipsparseLtMatmulDescGetAttribute"]                              = {HIP_7100, HIP_0,    HIP_0   };

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
