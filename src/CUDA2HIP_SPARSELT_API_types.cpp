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
  m["cusparseLtSparsity_t"]                                              = {"hipsparseLtSparsity_t",                         "", CONV_TYPE, API_SPARSELT, 1};
  m["CUSPARSELT_SPARSITY_50_PERCENT"]                                    = {"HIPSPARSELT_SPARSITY_50_PERCENT",               "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["cusparseComputeType"]                                               = {"hipsparseLtComputetype_t",                      "", CONV_TYPE, API_SPARSELT, 1};
  m["CUSPARSE_COMPUTE_16F"]                                              = {"HIPSPARSELT_COMPUTE_16F",                       "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["CUSPARSE_COMPUTE_32I"]                                              = {"HIPSPARSELT_COMPUTE_32I",                       "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["cusparseLtMatmulAlg_t"]                                             = {"hipsparseLtMatmulAlg_t",                        "", CONV_TYPE, API_SPARSELT, 1};
  m["CUSPARSELT_MATMUL_ALG_DEFAULT"]                                     = {"HIPSPARSELT_MATMUL_ALG_DEFAULT",                "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["cusparseLtMatmulAlgAttribute_t"]                                    = {"hipsparseLtMatmulAlgAttribute_t",               "", CONV_TYPE, API_SPARSELT, 1};
  m["CUSPARSELT_MATMUL_ALG_CONFIG_ID"]                                   = {"HIPSPARSELT_MATMUL_ALG_CONFIG_ID",              "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID"]                               = {"HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID",          "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["CUSPARSELT_MATMUL_SEARCH_ITERATIONS"]                               = {"HIPSPARSELT_MATMUL_SEARCH_ITERATIONS",          "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["cusparseLtPruneAlg_t"]                                              = {"hipsparseLtPruneAlg_t",                         "", CONV_TYPE, API_SPARSELT, 1};
  m["CUSPARSELT_PRUNE_SPMMA_TILE"]                                       = {"HIPSPARSELT_PRUNE_SPMMA_TILE",                  "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};
  m["CUSPARSELT_PRUNE_SPMMA_STRIP"]                                      = {"HIPSPARSELT_PRUNE_SPMMA_STRIP",                 "", CONV_NUMERIC_LITERAL, API_SPARSELT, 1};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSELT_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cusparseLtHandle_t"]                                                = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatDescriptor_t"]                                         = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulDescriptor_t"]                                      = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgSelection_t"]                                    = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulPlan_t"]                                            = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtSparsity_t"]                                              = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_SPARSITY_50_PERCENT"]                                    = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseComputeType"]                                               = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSE_COMPUTE_16F"]                                              = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSE_COMPUTE_32I"]                                              = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlg_t"]                                             = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_MATMUL_ALG_DEFAULT"]                                     = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtMatmulAlgAttribute_t"]                                    = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_MATMUL_ALG_CONFIG_ID"]                                   = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID"]                               = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_MATMUL_SEARCH_ITERATIONS"]                               = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["cusparseLtPruneAlg_t"]                                              = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_PRUNE_SPMMA_TILE"]                                       = {CUSPARSELT_001, CUDA_0      , CUDA_0      };
  m["CUSPARSELT_PRUNE_SPMMA_STRIP"]                                      = {CUSPARSELT_001, CUDA_0      , CUDA_0      };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSELT_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipsparseLtHandle_t"]                                               = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatDescriptor_t"]                                        = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulDescriptor_t"]                                     = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulAlgSelection_t"]                                   = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulPlan_t"]                                           = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtSparsity_t"]                                             = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_SPARSITY_50_PERCENT"]                                   = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtComputetype_t"]                                          = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_COMPUTE_16F"]                                           = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_COMPUTE_32I"]                                           = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulAlg_t"]                                            = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_MATMUL_ALG_DEFAULT"]                                    = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtMatmulAlgAttribute_t"]                                   = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_MATMUL_ALG_CONFIG_ID"]                                  = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_MATMUL_ALG_CONFIG_MAX_ID"]                              = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_MATMUL_SEARCH_ITERATIONS"]                              = {HIP_7020, HIP_0,    HIP_0    };
  m["hipsparseLtPruneAlg_t"]                                             = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_PRUNE_SPMMA_TILE"]                                      = {HIP_7020, HIP_0,    HIP_0    };
  m["HIPSPARSELT_PRUNE_SPMMA_STRIP"]                                     = {HIP_7020, HIP_0,    HIP_0    };

  return m;
}();
