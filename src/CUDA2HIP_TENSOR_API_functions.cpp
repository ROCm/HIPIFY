/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

const std::map<llvm::StringRef, hipCounter> CUDA_TENSOR_FUNCTION_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  m["cutensorCreate"]                                               = {"hiptensorCreate",                                     "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorDestroy"]                                              = {"hiptensorDestroy",                                    "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorHandleResizePlanCache"]                                = {"hiptensorHandleResizePlanCache",                      "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorHandleWritePlanCacheToFile"]                           = {"hiptensorHandleWritePlanCacheToFile",                 "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorHandleReadPlanCacheFromFile"]                          = {"hiptensorHandleReadPlanCacheFromFile",                "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorWriteKernelCacheToFile"]                               = {"hiptensorWriteKernelCacheToFile",                     "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorReadKernelCacheFromFile"]                              = {"hiptensorReadKernelCacheFromFile",                    "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreateTensorDescriptor"]                               = {"hiptensorCreateTensorDescriptor",                     "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorInitTensorDescriptor"]                                 = {"hiptensorInitTensorDescriptor",                       "", CONV_LIB_FUNC, API_TENSOR, 2, HIP_REMOVED};
  m["cutensorDestroyTensorDescriptor"]                              = {"hiptensorDestroyTensorDescriptor",                    "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreateElementwiseTrinary"]                             = {"hiptensorCreateElementwiseTrinary",                   "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorElementwiseTrinaryExecute"]                            = {"hiptensorElementwiseTrinaryExecute",                  "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreateElementwiseBinary"]                              = {"hiptensorCreateElementwiseBinary",                    "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorElementwiseBinaryExecute"]                             = {"hiptensorElementwiseBinaryExecute",                   "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreatePermutation"]                                    = {"hiptensorCreatePermutation",                          "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorPermutation"]                                          = {"hiptensorPermutation",                                "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorPermute"]                                              = {"hiptensorPermute",                                    "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreateContraction"]                                    = {"hiptensorCreateContraction",                          "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorContraction"]                                          = {"hiptensorContraction",                                "", CONV_LIB_FUNC, API_TENSOR, 2, HIP_REMOVED};
  m["cutensorDestroyOperationDescriptor"]                           = {"hiptensorDestroyOperationDescriptor",                 "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorOperationDescriptorSetAttribute"]                      = {"hiptensorOperationDescriptorSetAttribute",            "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorOperationDescriptorGetAttribute"]                      = {"hiptensorOperationDescriptorGetAttribute",            "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreatePlanPreference"]                                 = {"hiptensorCreatePlanPreference",                       "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorDestroyPlanPreference"]                                = {"hiptensorDestroyPlanPreference",                      "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorPlanPreferenceSetAttribute"]                           = {"hiptensorPlanPreferenceSetAttribute",                 "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorPlanPreferenceGetAttribute"]                           = {"hiptensorPlanPreferenceGetAttribute",                 "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorPlanGetAttribute"]                                     = {"hiptensorPlanGetAttribute",                           "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorEstimateWorkspaceSize"]                                = {"hiptensorEstimateWorkspaceSize",                      "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreatePlan"]                                           = {"hiptensorCreatePlan",                                 "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorDestroyPlan"]                                          = {"hiptensorDestroyPlan",                                "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorContract"]                                             = {"hiptensorContract",                                   "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorReduction"]                                            = {"hiptensorReduction",                                  "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorCreateReduction"]                                      = {"hiptensorCreateReduction",                            "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorReduce"]                                               = {"hiptensorReduce",                                     "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorGetErrorString"]                                       = {"hiptensorGetErrorString",                             "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorGetVersion"]                                           = {"hiptensorGetVersion",                                 "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorGetCudartVersion"]                                     = {"hiptensorGetHiprtVersion",                            "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerSetCallback"]                                    = {"hiptensorLoggerSetCallback",                          "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerSetFile"]                                        = {"hiptensorLoggerSetFile",                              "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerOpenFile"]                                       = {"hiptensorLoggerOpenFile",                             "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerSetLevel"]                                       = {"hiptensorLoggerSetLevel",                             "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerSetMask"]                                        = {"hiptensorLoggerSetMask",                              "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorLoggerForceDisable"]                                   = {"hiptensorLoggerForceDisable",                         "", CONV_LIB_FUNC, API_TENSOR, 2};
  m["cutensorMgCreate"]                                             = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroy"]                                            = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateTensorDescriptor"]                             = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyTensorDescriptor"]                            = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateCopyDescriptor"]                               = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyCopyDescriptor"]                              = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCopyGetWorkspace"]                                   = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateCopyPlan"]                                     = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyCopyPlan"]                                    = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCopy"]                                               = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateContractionFind"]                              = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyContractionFind"]                             = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgContractionFindSetAttribute"]                        = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateContractionDescriptor"]                        = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyContractionDescriptor"]                       = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgContractionGetWorkspace"]                            = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgCreateContractionPlan"]                              = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgDestroyContractionPlan"]                             = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorMgContraction"]                                        = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorCreateContractionTrinary"]                             = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorContractTrinary"]                                      = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorCreateBlockSparseTensorDescriptor"]                    = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorDestroyBlockSparseTensorDescriptor"]                   = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorCreateBlockSparseContraction"]                         = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};
  m["cutensorBlockSparseContract"]                                  = {"",                                                    "", CONV_LIB_FUNC, API_TENSOR, 2, UNSUPPORTED};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_TENSOR_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cutensorCreate"]                                               = {CUTENSOR_1700, CUDA_0,        CUDA_0       };
  m["cutensorDestroy"]                                              = {CUTENSOR_1700, CUDA_0,        CUDA_0       };
  m["cutensorHandleResizePlanCache"]                                = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorHandleWritePlanCacheToFile"]                           = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorHandleReadPlanCacheFromFile"]                          = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorWriteKernelCacheToFile"]                               = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorReadKernelCacheFromFile"]                              = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreateTensorDescriptor"]                               = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorInitTensorDescriptor"]                                 = {CUTENSOR_1010, CUDA_0,        CUTENSOR_2000};
  m["cutensorDestroyTensorDescriptor"]                              = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreateElementwiseTrinary"]                             = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorElementwiseTrinaryExecute"]                            = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreateElementwiseBinary"]                              = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorElementwiseBinaryExecute"]                             = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreatePermutation"]                                    = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorPermutation"]                                          = {CUTENSOR_1010, CUDA_0,        CUTENSOR_2000};
  m["cutensorPermute"]                                              = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreateContraction"]                                    = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorContraction"]                                          = {CUTENSOR_1010, CUDA_0,        CUTENSOR_2000};
  m["cutensorDestroyOperationDescriptor"]                           = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorOperationDescriptorSetAttribute"]                      = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorOperationDescriptorGetAttribute"]                      = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreatePlanPreference"]                                 = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorDestroyPlanPreference"]                                = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorPlanPreferenceSetAttribute"]                           = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorPlanPreferenceGetAttribute"]                           = {CUTENSOR_2400, CUDA_0,        CUDA_0       };
  m["cutensorPlanGetAttribute"]                                     = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorEstimateWorkspaceSize"]                                = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreatePlan"]                                           = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorDestroyPlan"]                                          = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorContract"]                                             = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorCreateReduction"]                                      = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorReduction"]                                            = {CUTENSOR_1010, CUDA_0,        CUTENSOR_2000};
  m["cutensorReduce"]                                               = {CUTENSOR_2000, CUDA_0,        CUDA_0       };
  m["cutensorGetErrorString"]                                       = {CUTENSOR_1010, CUDA_0,        CUDA_0       };
  m["cutensorGetVersion"]                                           = {CUTENSOR_1010, CUDA_0,        CUDA_0       };
  m["cutensorGetCudartVersion"]                                     = {CUTENSOR_1010, CUDA_0,        CUDA_0       };
  m["cutensorLoggerSetCallback"]                                    = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorLoggerSetFile"]                                        = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorLoggerOpenFile"]                                       = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorLoggerSetLevel"]                                       = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorLoggerSetMask"]                                        = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorLoggerForceDisable"]                                   = {CUTENSOR_1320, CUDA_0,        CUDA_0       };
  m["cutensorMgCreate"]                                             = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroy"]                                            = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateTensorDescriptor"]                             = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyTensorDescriptor"]                            = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateCopyDescriptor"]                               = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyCopyDescriptor"]                              = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCopyGetWorkspace"]                                   = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateCopyPlan"]                                     = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyCopyPlan"]                                    = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCopy"]                                               = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateContractionFind"]                              = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyContractionFind"]                             = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgContractionFindSetAttribute"]                        = {CUTENSOR_1500, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateContractionDescriptor"]                        = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyContractionDescriptor"]                       = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgContractionGetWorkspace"]                            = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgCreateContractionPlan"]                              = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgDestroyContractionPlan"]                             = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorMgContraction"]                                        = {CUTENSOR_1400, CUDA_0,        CUDA_0       };
  m["cutensorCreateContractionTrinary"]                             = {CUTENSOR_2200, CUDA_0,        CUDA_0       };
  m["cutensorContractTrinary"]                                      = {CUTENSOR_2200, CUDA_0,        CUDA_0       };
  m["cutensorCreateBlockSparseTensorDescriptor"]                    = {CUTENSOR_2300, CUDA_0,        CUDA_0       };
  m["cutensorDestroyBlockSparseTensorDescriptor"]                   = {CUTENSOR_2300, CUDA_0,        CUDA_0       };
  m["cutensorCreateBlockSparseContraction"]                         = {CUTENSOR_2300, CUDA_0,        CUDA_0       };
  m["cutensorBlockSparseContract"]                                  = {CUTENSOR_2300, CUDA_0,        CUDA_0       };

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_TENSOR_FUNCTION_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hiptensorCreate"]                                              = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorDestroy"]                                             = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorInitTensorDescriptor"]                                = {HIP_5070,      HIP_0,         HIP_7000     };
  m["hiptensorPermutation"]                                         = {HIP_6010,      HIP_0,         HIP_0        };
  m["hiptensorContraction"]                                         = {HIP_6010,      HIP_0,         HIP_7000     };
  m["hiptensorReduction"]                                           = {HIP_6030,      HIP_0,         HIP_0        };
  m["hiptensorGetErrorString"]                                      = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorGetHiprtVersion"]                                     = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerSetCallback"]                                   = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerSetFile"]                                       = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerOpenFile"]                                      = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerSetLevel"]                                      = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerSetMask"]                                       = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorLoggerForceDisable"]                                  = {HIP_5070,      HIP_0,         HIP_0        };
  m["hiptensorHandleResizePlanCache"]                               = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorHandleWritePlanCacheToFile"]                          = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorHandleReadPlanCacheFromFile"]                         = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorWriteKernelCacheToFile"]                              = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorReadKernelCacheFromFile"]                             = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreateTensorDescriptor"]                              = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorDestroyTensorDescriptor"]                             = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreateContraction"]                                   = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorDestroyOperationDescriptor"]                          = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorOperationDescriptorSetAttribute"]                     = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorOperationDescriptorGetAttribute"]                     = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreatePlanPreference"]                                = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorDestroyPlanPreference"]                               = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorPlanPreferenceSetAttribute"]                          = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorPlanGetAttribute"]                                    = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorEstimateWorkspaceSize"]                               = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreatePermutation"]                                   = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreatePlan"]                                          = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorDestroyPlan"]                                         = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorContract"]                                            = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorPermute"]                                             = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreateElementwiseBinary"]                             = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorElementwiseBinaryExecute"]                            = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreateElementwiseTrinary"]                            = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorElementwiseTrinaryExecute"]                           = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorCreateReduction"]                                     = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorReduce"]                                              = {HIP_7000,      HIP_0,         HIP_0        };
  m["hiptensorGetVersion"]                                          = {HIP_7020,      HIP_0,         HIP_0        };

  return m;
}();

const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_TENSOR_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIChangedVersions> m;

  m["cutensorCreate"]                                               = {CUTENSOR_2000};
  m["cutensorDestroy"]                                              = {CUTENSOR_2000};

  return m;
}();

const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_TENSOR_FUNCTION_CHANGED_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIChangedVersions> m;

  m["hiptensorCreate"]                                              = {HIP_7000};
  m["hiptensorDestroy"]                                             = {HIP_7000};

  return m;
}();

const std::map<unsigned int, llvm::StringRef> CUDA_TENSOR_API_SECTION_MAP = [] {
  std::map<unsigned int, llvm::StringRef> m;

  m[1]                                                              = "CUTENSOR Data types";
  m[2]                                                              = "CUTENSOR Function Reference";

  return m;
}();
