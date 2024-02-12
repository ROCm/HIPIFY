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

#include <algorithm>
#include <set>
#include "HipifyAction.h"
#include "CUDA2HIP_Scripting.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/HeaderSearch.h"
#include "LLVMCompat.h"
#include "CUDA2HIP.h"
#include "StringUtils.h"
#include "ArgParse.h"

using namespace hipify;

const std::string sHIP = "HIP";
const std::string sROC = "ROC";
const std::string sCub = "cub";
const std::string sHipcub = "hipcub";
const std::string sHIP_KERNEL_NAME = "HIP_KERNEL_NAME";
std::string sHIP_SYMBOL = "HIP_SYMBOL";
std::string s_reinterpret_cast = "reinterpret_cast<const void*>";
std::string s_reinterpret_cast_size_t = "reinterpret_cast<size_t*>";
std::string s_int32_t = "int32_t";
std::string s_int64_t = "int64_t";
const std::string sHipLaunchKernelGGL = "hipLaunchKernelGGL";
const std::string sDim3 = "dim3(";
const std::string s_hiprand_kernel_h = "hiprand/hiprand_kernel.h";
const std::string s_hiprand_h = "hiprand/hiprand.h";
const std::string sOnce = "once";
const std::string s_string_literal = "[string literal]";
// CUDA identifiers, used in matchers
const std::string sCudaMemcpyToSymbol = "cudaMemcpyToSymbol";
const std::string sCudaMemcpyToSymbolAsync = "cudaMemcpyToSymbolAsync";
const std::string sCudaGetSymbolSize = "cudaGetSymbolSize";
const std::string sCudaGetSymbolAddress = "cudaGetSymbolAddress";
const std::string sCudaMemcpyFromSymbol = "cudaMemcpyFromSymbol";
const std::string sCudaMemcpyFromSymbolAsync = "cudaMemcpyFromSymbolAsync";
const std::string sCudaGraphAddMemcpyNodeToSymbol = "cudaGraphAddMemcpyNodeToSymbol";
const std::string sCudaGraphAddMemcpyNodeFromSymbol = "cudaGraphAddMemcpyNodeFromSymbol";
const std::string sCudaGraphMemcpyNodeSetParamsToSymbol = "cudaGraphMemcpyNodeSetParamsToSymbol";
const std::string sCudaGraphMemcpyNodeSetParamsFromSymbol = "cudaGraphMemcpyNodeSetParamsFromSymbol";
const std::string sCudaGraphExecMemcpyNodeSetParamsToSymbol = "cudaGraphExecMemcpyNodeSetParamsToSymbol";
const std::string sCudaGraphExecMemcpyNodeSetParamsFromSymbol = "cudaGraphExecMemcpyNodeSetParamsFromSymbol";
const std::string sCuOccupancyMaxPotentialBlockSize = "cuOccupancyMaxPotentialBlockSize";
const std::string sCuOccupancyMaxPotentialBlockSizeWithFlags = "cuOccupancyMaxPotentialBlockSizeWithFlags";
const std::string sCudaGetTextureReference = "cudaGetTextureReference";
const std::string sCudnnGetConvolutionForwardWorkspaceSize = "cudnnGetConvolutionForwardWorkspaceSize";
const std::string sCudnnGetConvolutionBackwardDataWorkspaceSize = "cudnnGetConvolutionBackwardDataWorkspaceSize";
const std::string sCudnnFindConvolutionForwardAlgorithmEx = "cudnnFindConvolutionForwardAlgorithmEx";
const std::string sCudnnSetPooling2dDescriptor = "cudnnSetPooling2dDescriptor";
const std::string sCudnnGetPooling2dDescriptor = "cudnnGetPooling2dDescriptor";
const std::string sCudnnSetPoolingNdDescriptor = "cudnnSetPoolingNdDescriptor";
const std::string sCudnnGetPoolingNdDescriptor = "cudnnGetPoolingNdDescriptor";
const std::string sCudnnSetLRNDescriptor = "cudnnSetLRNDescriptor";
const std::string sCudnnGetRNNDescriptor_v6 = "cudnnGetRNNDescriptor_v6";
const std::string sCudnnSetRNNDescriptor_v6 = "cudnnSetRNNDescriptor_v6";
const std::string sCudnnSoftmaxForward = "cudnnSoftmaxForward";
const std::string sCudnnSoftmaxBackward = "cudnnSoftmaxBackward";
const std::string sCudnnConvolutionForward = "cudnnConvolutionForward";
const std::string sCudnnConvolutionBackwardData = "cudnnConvolutionBackwardData";
const std::string sCudnnRNNBackwardWeights = "cudnnRNNBackwardWeights";
const std::string sCusparseZgpsvInterleavedBatch = "cusparseZgpsvInterleavedBatch";
const std::string sCusparseCgpsvInterleavedBatch = "cusparseCgpsvInterleavedBatch";
const std::string sCusparseDgpsvInterleavedBatch = "cusparseDgpsvInterleavedBatch";
const std::string sCusparseSgpsvInterleavedBatch = "cusparseSgpsvInterleavedBatch";
const std::string sCusparseZgpsvInterleavedBatch_bufferSizeExt = "cusparseZgpsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseCgpsvInterleavedBatch_bufferSizeExt = "cusparseCgpsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseDgpsvInterleavedBatch_bufferSizeExt = "cusparseDgpsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseSgpsvInterleavedBatch_bufferSizeExt = "cusparseSgpsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseZgtsvInterleavedBatch = "cusparseZgtsvInterleavedBatch";
const std::string sCusparseCgtsvInterleavedBatch = "cusparseCgtsvInterleavedBatch";
const std::string sCusparseDgtsvInterleavedBatch = "cusparseDgtsvInterleavedBatch";
const std::string sCusparseSgtsvInterleavedBatch = "cusparseSgtsvInterleavedBatch";
const std::string sCusparseZgtsvInterleavedBatch_bufferSizeExt = "cusparseZgtsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseCgtsvInterleavedBatch_bufferSizeExt = "cusparseCgtsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseDgtsvInterleavedBatch_bufferSizeExt = "cusparseDgtsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseSgtsvInterleavedBatch_bufferSizeExt = "cusparseSgtsvInterleavedBatch_bufferSizeExt";
const std::string sCusparseZcsrilu02 = "cusparseZcsrilu02";
const std::string sCusparseCcsrilu02 = "cusparseCcsrilu02";
const std::string sCusparseDcsrilu02 = "cusparseDcsrilu02";
const std::string sCusparseScsrilu02 = "cusparseScsrilu02";
const std::string sCusparseZcsrilu02_analysis = "cusparseZcsrilu02_analysis";
const std::string sCusparseCcsrilu02_analysis = "cusparseCcsrilu02_analysis";
const std::string sCusparseDcsrilu02_analysis = "cusparseDcsrilu02_analysis";
const std::string sCusparseScsrilu02_analysis = "cusparseScsrilu02_analysis";
const std::string sCusparseZcsric02_analysis = "cusparseZcsric02_analysis";
const std::string sCusparseCcsric02_analysis = "cusparseCcsric02_analysis";
const std::string sCusparseDcsric02_analysis = "cusparseDcsric02_analysis";
const std::string sCusparseScsric02_analysis = "cusparseScsric02_analysis";
const std::string sCusparseZcsric02_bufferSize = "cusparseZcsric02_bufferSize";
const std::string sCusparseCcsric02_bufferSize = "cusparseCcsric02_bufferSize";
const std::string sCusparseDcsric02_bufferSize = "cusparseDcsric02_bufferSize";
const std::string sCusparseScsric02_bufferSize = "cusparseScsric02_bufferSize";
const std::string sCusparseZbsrilu02 = "cusparseZbsrilu02";
const std::string sCusparseCbsrilu02 = "cusparseCbsrilu02";
const std::string sCusparseDbsrilu02 = "cusparseDbsrilu02";
const std::string sCusparseSbsrilu02 = "cusparseSbsrilu02";
const std::string sCusparseZbsrilu02_analysis = "cusparseZbsrilu02_analysis";
const std::string sCusparseCbsrilu02_analysis = "cusparseCbsrilu02_analysis";
const std::string sCusparseDbsrilu02_analysis = "cusparseDbsrilu02_analysis";
const std::string sCusparseSbsrilu02_analysis = "cusparseSbsrilu02_analysis";
const std::string sCusparseZbsric02 = "cusparseZbsric02";
const std::string sCusparseCbsric02 = "cusparseCbsric02";
const std::string sCusparseDbsric02 = "cusparseDbsric02";
const std::string sCusparseSbsric02 = "cusparseSbsric02";
const std::string sCusparseZbsric02_analysis = "cusparseZbsric02_analysis";
const std::string sCusparseCbsric02_analysis = "cusparseCbsric02_analysis";
const std::string sCusparseDbsric02_analysis = "cusparseDbsric02_analysis";
const std::string sCusparseSbsric02_analysis = "cusparseSbsric02_analysis";
const std::string sCusparseZbsric02_bufferSize = "cusparseZbsric02_bufferSize";
const std::string sCusparseCbsric02_bufferSize = "cusparseCbsric02_bufferSize";
const std::string sCusparseDbsric02_bufferSize = "cusparseDbsric02_bufferSize";
const std::string sCusparseSbsric02_bufferSize = "cusparseSbsric02_bufferSize";
const std::string sCusparseZbsrsm2_bufferSize = "cusparseZbsrsm2_bufferSize";
const std::string sCusparseCbsrsm2_bufferSize = "cusparseCbsrsm2_bufferSize";
const std::string sCusparseDbsrsm2_bufferSize = "cusparseDbsrsm2_bufferSize";
const std::string sCusparseSbsrsm2_bufferSize = "cusparseSbsrsm2_bufferSize";
const std::string sCusparseZcsrsm2_solve = "cusparseZcsrsm2_solve";
const std::string sCusparseCcsrsm2_solve = "cusparseCcsrsm2_solve";
const std::string sCusparseDcsrsm2_solve = "cusparseDcsrsm2_solve";
const std::string sCusparseScsrsm2_solve = "cusparseScsrsm2_solve";
const std::string sCusparseZcsrsm2_analysis = "cusparseZcsrsm2_analysis";
const std::string sCusparseCcsrsm2_analysis = "cusparseCcsrsm2_analysis";
const std::string sCusparseDcsrsm2_analysis = "cusparseDcsrsm2_analysis";
const std::string sCusparseScsrsm2_analysis = "cusparseScsrsm2_analysis";
const std::string sCusparseScsrsm2_bufferSizeExt = "cusparseScsrsm2_bufferSizeExt";
const std::string sCusparseDcsrsm2_bufferSizeExt = "cusparseDcsrsm2_bufferSizeExt";
const std::string sCusparseCcsrsm2_bufferSizeExt = "cusparseCcsrsm2_bufferSizeExt";
const std::string sCusparseZcsrsm2_bufferSizeExt = "cusparseZcsrsm2_bufferSizeExt";
const std::string sCusparseZgemvi_bufferSize = "cusparseZgemvi_bufferSize";
const std::string sCusparseCgemvi_bufferSize = "cusparseCgemvi_bufferSize";
const std::string sCusparseDgemvi_bufferSize = "cusparseDgemvi_bufferSize";
const std::string sCusparseSgemvi_bufferSize = "cusparseSgemvi_bufferSize";
const std::string sCusparseZcsrsv2_solve = "cusparseZcsrsv2_solve";
const std::string sCusparseCcsrsv2_solve = "cusparseCcsrsv2_solve";
const std::string sCusparseDcsrsv2_solve = "cusparseDcsrsv2_solve";
const std::string sCusparseScsrsv2_solve = "cusparseScsrsv2_solve";
const std::string sCusparseZcsrsv2_analysis = "cusparseZcsrsv2_analysis";
const std::string sCusparseCcsrsv2_analysis = "cusparseCcsrsv2_analysis";
const std::string sCusparseDcsrsv2_analysis = "cusparseDcsrsv2_analysis";
const std::string sCusparseScsrsv2_analysis = "cusparseScsrsv2_analysis";
const std::string sCusparseZcsrmv = "cusparseZcsrmv";
const std::string sCusparseCcsrmv = "cusparseCcsrmv";
const std::string sCusparseDcsrmv = "cusparseDcsrmv";
const std::string sCusparseScsrmv = "cusparseScsrmv";
const std::string sCusparseZbsrsv2_solve = "cusparseZbsrsv2_solve";
const std::string sCusparseCbsrsv2_solve = "cusparseCbsrsv2_solve";
const std::string sCusparseDbsrsv2_solve = "cusparseDbsrsv2_solve";
const std::string sCusparseSbsrsv2_solve = "cusparseSbsrsv2_solve";
const std::string sCusparseSbsrsv2_analysis = "cusparseSbsrsv2_analysis";
const std::string sCusparseDbsrsv2_analysis = "cusparseDbsrsv2_analysis";
const std::string sCusparseCbsrsv2_analysis = "cusparseCbsrsv2_analysis";
const std::string sCusparseZbsrsv2_analysis = "cusparseZbsrsv2_analysis";
const std::string sCusparseZcsrmm = "cusparseZcsrmm";
const std::string sCusparseCcsrmm = "cusparseCcsrmm";
const std::string sCusparseDcsrmm = "cusparseDcsrmm";
const std::string sCusparseScsrmm = "cusparseScsrmm";
const std::string sCusparseZcsrgeam2 = "cusparseZcsrgeam2";
const std::string sCusparseCcsrgeam2 = "cusparseCcsrgeam2";
const std::string sCusparseDcsrgeam2 = "cusparseDcsrgeam2";
const std::string sCusparseScsrgeam2 = "cusparseScsrgeam2";
const std::string sCusparseZbsrsv2_bufferSize = "cusparseZbsrsv2_bufferSize";
const std::string sCusparseCbsrsv2_bufferSize = "cusparseCbsrsv2_bufferSize";
const std::string sCusparseDbsrsv2_bufferSize = "cusparseDbsrsv2_bufferSize";
const std::string sCusparseSbsrsv2_bufferSize = "cusparseSbsrsv2_bufferSize";
const std::string sCusparseZcsrsv2_bufferSize = "cusparseZcsrsv2_bufferSize";
const std::string sCusparseCcsrsv2_bufferSize = "cusparseCcsrsv2_bufferSize";
const std::string sCusparseDcsrsv2_bufferSize = "cusparseDcsrsv2_bufferSize";
const std::string sCusparseScsrsv2_bufferSize = "cusparseScsrsv2_bufferSize";
const std::string sCusparseZcsrgemm2 = "cusparseZcsrgemm2";
const std::string sCusparseCcsrgemm2 = "cusparseCcsrgemm2";
const std::string sCusparseDcsrgemm2 = "cusparseDcsrgemm2";
const std::string sCusparseScsrgemm2 = "cusparseScsrgemm2";
const std::string sCusparseZcsrilu02_bufferSize = "cusparseZcsrilu02_bufferSize";
const std::string sCusparseCcsrilu02_bufferSize = "cusparseCcsrilu02_bufferSize";
const std::string sCusparseDcsrilu02_bufferSize = "cusparseDcsrilu02_bufferSize";
const std::string sCusparseScsrilu02_bufferSize = "cusparseScsrilu02_bufferSize";
const std::string sCusparseZbsrilu02_bufferSize = "cusparseZbsrilu02_bufferSize";
const std::string sCusparseCbsrilu02_bufferSize = "cusparseCbsrilu02_bufferSize";
const std::string sCusparseDbsrilu02_bufferSize = "cusparseDbsrilu02_bufferSize";
const std::string sCusparseSbsrilu02_bufferSize = "cusparseSbsrilu02_bufferSize";
const std::string sCusparseCsr2cscEx2_bufferSize = "cusparseCsr2cscEx2_bufferSize";
const std::string sCusparseSparseToDense = "cusparseSparseToDense";
const std::string sCusparseSparseToDense_bufferSize = "cusparseSparseToDense_bufferSize";
const std::string sCusparseDenseToSparse_bufferSize = "cusparseDenseToSparse_bufferSize";
const std::string sCusparseDenseToSparse_analysis = "cusparseDenseToSparse_analysis";
const std::string sCusparseSpMM_bufferSize = "cusparseSpMM_bufferSize";
const std::string sCusparseSpSM_analysis = "cusparseSpSM_analysis";
const std::string sCusparseSpSM_solve = "cusparseSpSM_solve";
const std::string sCusparseXcsrgeam2Nnz = "cusparseXcsrgeam2Nnz";
const std::string sCudaMallocHost = "cudaMallocHost";
const std::string sCusparseSpMM = "cusparseSpMM";
const std::string sCusparseSpVV = "cusparseSpVV";
const std::string sCusparseSpVV_bufferSize = "cusparseSpVV_bufferSize";
const std::string sCusparseSpMV = "cusparseSpMV";
const std::string sCusparseSpMV_bufferSize = "cusparseSpMV_bufferSize";
const std::string sCusparseSpMM_preprocess = "cusparseSpMM_preprocess";
const std::string sCusparseSpSV_bufferSize = "cusparseSpSV_bufferSize";

// CUDA_OVERLOADED
const std::string sCudaEventCreate = "cudaEventCreate";
const std::string sCudaGraphInstantiate = "cudaGraphInstantiate";
// Matchers' names
const StringRef sCudaLaunchKernel = "cudaLaunchKernel";
const StringRef sCudaHostFuncCall = "cudaHostFuncCall";
const StringRef sCudaOverloadedHostFuncCall = "cudaOverloadedHostFuncCall";
const StringRef sCudaDeviceFuncCall = "cudaDeviceFuncCall";
const StringRef sCubNamespacePrefix = "cubNamespacePrefix";
const StringRef sCubFunctionTemplateDecl = "cubFunctionTemplateDecl";
const StringRef sCubUsingNamespaceDecl = "cubUsingNamespaceDecl";
const StringRef sHalf2Member = "half2Member";
const StringRef sDataTypeSelection = "dataTypeSelection";

std::string getCastType(hipify::CastTypes c) {
  switch (c) {
    case e_HIP_SYMBOL: return sHIP_SYMBOL;
    case e_reinterpret_cast: return s_reinterpret_cast;
    case e_reinterpret_cast_size_t: return s_reinterpret_cast_size_t;
    case e_int32_t: return s_int32_t;
    case e_int64_t: return s_int64_t;
    case e_remove_argument: return "";
    case e_add_const_argument: return "";
    case e_add_var_argument: return "";
    case e_move_argument: return "";
    case e_replace_argument_with_const: return "";
    default: return "";
  }
}

std::map<std::string, std::string> TypeOverloads {
  {"enum cudaDataType_t", "hipDataType"},
  {"cudaDataType_t", "hipDataType"},
  {"cudaDataType", "hipDataType"},
};

std::map<std::string, hipify::FuncOverloadsStruct> FuncOverloads {
  {sCudaEventCreate,
    {
      {
        {1, {{"hipEventCreate", "", CONV_EVENT, API_RUNTIME, runtime::CUDA_RUNTIME_API_SECTIONS::EVENT}, ot_arguments_number, ow_None}},
        {2, {{"hipEventCreateWithFlags", "", CONV_EVENT, API_RUNTIME, runtime::CUDA_RUNTIME_API_SECTIONS::EVENT}, ot_arguments_number, ow_None}},
      }
    }
  },
  {sCudaGraphInstantiate,
    {
      {
        {5, {{"hipGraphInstantiate", "", CONV_GRAPH, API_RUNTIME, runtime::CUDA_RUNTIME_API_SECTIONS::GRAPH}, ot_arguments_number, ow_None}},
        {3, {{"hipGraphInstantiateWithFlags", "", CONV_GRAPH, API_RUNTIME, runtime::CUDA_RUNTIME_API_SECTIONS::GRAPH}, ot_arguments_number, ow_None}},
      }
    }
  },
};

std::map<std::string, std::vector<ArgCastStruct>> FuncArgCasts {
  {sCudaMallocHost,
    {
      {
        {
          {2, {e_add_const_argument, cw_None, "hipHostMallocDefault"}}
        }
      }
    }
  },
  {sCudaMemcpyToSymbol,
    {
      {
        {
          {0, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaMemcpyToSymbolAsync,
    {
      {
        {
          {0, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGetSymbolSize,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGetSymbolAddress,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaMemcpyFromSymbol,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaMemcpyFromSymbolAsync,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphAddMemcpyNodeToSymbol,
    {
      {
        {
          {4, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphAddMemcpyNodeFromSymbol,
    {
      {
        {
          {5, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphMemcpyNodeSetParamsToSymbol,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphMemcpyNodeSetParamsFromSymbol,
    {
      {
        {
          {2, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphExecMemcpyNodeSetParamsToSymbol,
    {
      {
        {
          {2, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGraphExecMemcpyNodeSetParamsFromSymbol,
    {
      {
        {
          {3, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCudaGetTextureReference,
    {
      {
        {
          {1, {e_HIP_SYMBOL, cw_None}}
        }
      }
    }
  },
  {sCuOccupancyMaxPotentialBlockSize,
    {
      {
        {
          {3, {e_remove_argument, cw_DataLoss}}
        }
      }
    }
  },
  {sCuOccupancyMaxPotentialBlockSizeWithFlags,
    {
      {
        {
          {3, {e_remove_argument, cw_DataLoss}}
        }
      }
    }
  },
  {sCudnnGetConvolutionForwardWorkspaceSize,
    {
      {
        {
          {1, {e_move_argument, cw_None, "", 2}},
          {2, {e_move_argument, cw_None, "", 1}},
          {5, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnGetConvolutionBackwardDataWorkspaceSize,
    {
      {
        {
          {1, {e_move_argument, cw_None, "", 2}},
          {2, {e_move_argument, cw_None, "", 1}},
          {5, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnFindConvolutionForwardAlgorithmEx,
    {
      {
        {
          {13, {e_add_const_argument, cw_None, "true"}}
        },
        true,
        true
      }
    }
  },
  {sCudnnSetPooling2dDescriptor,
    {
      {
        {
          {2, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnGetPooling2dDescriptor,
    {
      {
        {
          {2, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnSetPoolingNdDescriptor,
    {
      {
        {
          {2, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnGetPoolingNdDescriptor,
    {
      {
        {
          {3, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnSetLRNDescriptor,
    {
      {
        {
          {1, {e_add_const_argument, cw_None, "miopenLRNCrossChannel"}}
        },
        true,
        true
      }
    }
  },
  {sCudnnGetRNNDescriptor_v6,
    {
      {
        {
          {0, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnSetRNNDescriptor_v6,
    {
      {
        {
          {0, {e_remove_argument, cw_None}}
        },
        true,
        true
      }
    }
  },
  {sCudnnSoftmaxForward,
    {
      {
        {
          {1, {e_move_argument, cw_None, "", 9, 2}},
        },
        true,
        true
      }
    }
  },
  {sCudnnSoftmaxBackward,
    {
      {
        {
          {1, {e_move_argument, cw_None, "", 11, 2}},
        },
        true,
        true
      }
    }
  },
  {sCudnnConvolutionForward,
    {
      {
        {
          {8, {e_move_argument, cw_None, "", 13, 2}},
        },
        true,
        true
      }
    }
  },
  {sCudnnConvolutionBackwardData,
    {
      {
        {
          {2, {e_move_argument, cw_None, "", 4, 2}},
          {4, {e_move_argument, cw_None, "", 2, 2}},
          {8, {e_move_argument, cw_None, "", 13, 2}},
        },
        true,
        true
      }
    }
  },
  {sCudnnRNNBackwardWeights,
    {
      {
        {
          {9, {e_move_argument, cw_None, "", 11, 2}},
          {11, {e_move_argument, cw_None, "", 9, 2}},
        },
        true,
        true
      }
    }
  },
  {sCusparseZgpsvInterleavedBatch,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCgpsvInterleavedBatch,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDgpsvInterleavedBatch,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSgpsvInterleavedBatch,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZgpsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCgpsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDgpsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSgpsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {9, {e_add_var_argument, cw_None, "", 10}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZgtsvInterleavedBatch,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCgtsvInterleavedBatch,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDgtsvInterleavedBatch,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSgtsvInterleavedBatch,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZgtsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCgtsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDgtsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSgtsvInterleavedBatch_bufferSizeExt,
    {
      {
        {
          {7, {e_add_var_argument, cw_None, "", 8}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrilu02,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrilu02,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrilu02,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrilu02,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrilu02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrilu02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrilu02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrilu02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsric02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsric02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsric02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsric02_analysis,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {9, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsric02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsric02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsric02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsric02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrilu02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrilu02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrilu02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrilu02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrilu02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrilu02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrilu02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrilu02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsric02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsric02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsric02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsric02,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsric02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsric02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsric02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsric02_analysis,
    {
      {
        {
          {10, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {11, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsric02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsric02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsric02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsric02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrsm2_bufferSize,
    {
      {
        {
          {13, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrsm2_bufferSize,
    {
      {
        {
          {13, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrsm2_bufferSize,
    {
      {
        {
          {13, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrsm2_bufferSize,
    {
      {
        {
          {13, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsm2_solve,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsm2_solve,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsm2_solve,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsm2_solve,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsm2_analysis,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {16, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsm2_analysis,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {16, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsm2_analysis,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {16, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsm2_analysis,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {16, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsm2_bufferSizeExt,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsm2_bufferSizeExt,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsm2_bufferSizeExt,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsm2_bufferSizeExt,
    {
      {
        {
          {15, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZgemvi_bufferSize,
    {
      {
        {
          {5, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCgemvi_bufferSize,
    {
      {
        {
          {5, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDgemvi_bufferSize,
    {
      {
        {
          {5, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSgemvi_bufferSize,
    {
      {
        {
          {5, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsv2_solve,
    {
      {
        {
          {12, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsv2_solve,
    {
      {
        {
          {12, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsv2_solve,
    {
      {
        {
          {12, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsv2_solve,
    {
      {
        {
          {12, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsv2_analysis,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {10, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsv2_analysis,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {10, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsv2_analysis,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {10, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsv2_analysis,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {10, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrmv,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrmv,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrmv,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrmv,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrsv2_solve,
    {
      {
        {
          {14, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrsv2_solve,
    {
      {
        {
          {14, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrsv2_solve,
    {
      {
        {
          {14, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrsv2_solve,
    {
      {
        {
          {14, {e_replace_argument_with_const, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrsv2_analysis,
    {
      {
        {
          {11, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {12, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrsv2_analysis,
    {
      {
        {
          {11, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {12, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrsv2_analysis,
    {
      {
        {
          {11, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {12, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrsv2_analysis,
    {
      {
        {
          {11, {e_replace_argument_with_const, cw_None, "rocsparse_analysis_policy_force"}},
          {12, {e_add_const_argument, cw_None, "rocsparse_solve_policy_auto"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrmm,
    {
      {
        {
          {2, {e_add_const_argument, cw_None, "rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrmm,
    {
      {
        {
          {2, {e_add_const_argument, cw_None, "rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrmm,
    {
      {
        {
          {2, {e_add_const_argument, cw_None, "rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrmm,
    {
      {
        {
          {2, {e_add_const_argument, cw_None, "rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrgeam2,
    {
      {
        {
          {19, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrgeam2,
    {
      {
        {
          {19, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrgeam2,
    {
      {
        {
          {19, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrgeam2,
    {
      {
        {
          {19, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrsv2_bufferSize,
    {
      {
        {
          {11, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrsv2_bufferSize,
    {
      {
        {
          {11, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrsv2_bufferSize,
    {
      {
        {
          {11, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrsv2_bufferSize,
    {
      {
        {
          {11, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrsv2_bufferSize,
    {
      {
        {
          {9, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrsv2_bufferSize,
    {
      {
        {
          {9, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrsv2_bufferSize,
    {
      {
        {
          {9, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrsv2_bufferSize,
    {
      {
        {
          {9, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrgemm2,
    {
      {
        {
          {1, {e_add_const_argument, cw_None, "rocsparse_operation_none, rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrgemm2,
    {
      {
        {
          {1, {e_add_const_argument, cw_None, "rocsparse_operation_none, rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrgemm2,
    {
      {
        {
          {1, {e_add_const_argument, cw_None, "rocsparse_operation_none, rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrgemm2,
    {
      {
        {
          {1, {e_add_const_argument, cw_None, "rocsparse_operation_none, rocsparse_operation_none"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZcsrilu02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCcsrilu02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDcsrilu02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseScsrilu02_bufferSize,
    {
      {
        {
          {8, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseZbsrilu02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCbsrilu02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDbsrilu02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSbsrilu02_bufferSize,
    {
      {
        {
          {10, {e_reinterpret_cast_size_t, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseCsr2cscEx2_bufferSize,
    {
      {
        {
          {4, {e_remove_argument, cw_None}},
          {7, {e_remove_argument, cw_None}},
          {8, {e_remove_argument, cw_None}},
          {9, {e_remove_argument, cw_None}},
          {10, {e_remove_argument, cw_None}},
          {12, {e_remove_argument, cw_None}},
          {13, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSparseToDense,
    {
      {
        {
          {4, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSparseToDense_bufferSize,
    {
      {
        {
          {5, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDenseToSparse_bufferSize,
    {
      {
        {
          {5, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseDenseToSparse_analysis,
    {
      {
        {
          {4, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpMM_bufferSize,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "rocsparse_spmm_stage_compute"}},
          {12, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpSM_analysis,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_spsm_stage_compute"}},
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpSM_solve,
    {
      {
        {
          {9, {e_replace_argument_with_const, cw_None, "rocsparse_spsm_stage_compute"}},
          {10, {e_add_const_argument, cw_None, "nullptr"}},
          {11, {e_add_const_argument, cw_None, "nullptr"}},
        },
        true,
        false
      },
      {
        {
          {10, {e_add_const_argument, cw_None, "nullptr"}}
        }
      }
    }
  },
  {sCusparseXcsrgeam2Nnz,
    {
      {
        {
          {14, {e_remove_argument, cw_None}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpMM,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "rocsparse_spmm_stage_compute, nullptr"}},
        },
        true,
        false
      }
    }
  },
  {sCusparseSpVV_bufferSize,
    {
      {
        {
          {7, {e_add_const_argument, cw_None, "nullptr"}},
        },
        true,
        false
      }
    }
  },
  {sCusparseSpVV,
    {
      {
        {
          {6, {e_add_const_argument, cw_None, "nullptr"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpMV_bufferSize,
    {
      {
        {
          {9, {e_add_const_argument, cw_None, "rocsparse_spmv_stage_buffer_size"}},
          {11, {e_add_const_argument, cw_None, "nullptr"}},
        },
        true,
        false
      }
    }
  },
  {sCusparseSpMV,
    {
      {
        {
          {9, {e_add_const_argument, cw_None, "rocsparse_spmv_stage_compute"}}
        },
        true,
        false
      }
    }
  },
  {sCusparseSpMM_preprocess,
    {
      {
        {
          {10, {e_add_const_argument, cw_None, "rocsparse_spmm_stage_preprocess, nullptr"}},
        },
        true,
        false
      }
    }
  },
  {sCusparseSpSV_bufferSize,
    {
      {
        {
          {8, {e_replace_argument_with_const, cw_None, "rocsparse_spsv_stage_buffer_size"}},
          {10, {e_add_const_argument, cw_None, "nullptr"}},
        },
        true,
        false
      }
    }
  },
};

void HipifyAction::RewriteString(StringRef s, clang::SourceLocation start) {
  auto &SM = getCompilerInstance().getSourceManager();
  size_t begin = 0;
  while ((begin = s.find("cu", begin)) != StringRef::npos) {
    const size_t end = s.find_first_of(" ", begin + 4);
    StringRef name = s.slice(begin, end);
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found != CUDA_RENAMES_MAP().end()) {
      StringRef repName = Statistics::isToRoc(found->second) ? found->second.rocName : found->second.hipName;
      hipCounter counter = {s_string_literal, "", ConvTypes::CONV_LITERAL, ApiTypes::API_RUNTIME, found->second.supportDegree};
      Statistics::current().incrementCounter(counter, name.str());
      if (!Statistics::isUnsupported(counter)) {
        clang::SourceLocation sl = start.getLocWithOffset(begin + 1);
        ct::Replacement Rep(SM, sl, name.size(), repName.str());
        clang::FullSourceLoc fullSL(sl, SM);
        insertReplacement(Rep, fullSL);
      }
    }
    if (end == StringRef::npos) break;
    begin = end + 1;
  }
}

clang::SourceLocation HipifyAction::GetSubstrLocation(const std::string &str, const clang::SourceRange &sr) {
  clang::SourceLocation sl(sr.getBegin());
  clang::SourceLocation end(sr.getEnd());
  auto &SM = getCompilerInstance().getSourceManager();
  size_t length = SM.getCharacterData(end) - SM.getCharacterData(sl);
  StringRef sfull = StringRef(SM.getCharacterData(sl), length);
  size_t offset = sfull.find(str);
  if (offset > 0) {
    sl = sl.getLocWithOffset(offset);
  }
  return sl;
}

/**
  * Look at, and consider altering, a given token.
  *
  * If it's not a CUDA identifier, nothing happens.
  * If it's an unsupported CUDA identifier, a warning is emitted.
  * Otherwise, the source file is updated with the corresponding hipification.
  */
void HipifyAction::RewriteToken(const clang::Token &t) {
  if (!HipifyAMAP) {
    clang::SourceRange sr(t.getLocation());
    for (const auto& skipped : SkippedSourceRanges) {
      if (skipped.fullyContains(sr))
        return;
    }
  }
  // String literals containing CUDA references need fixing.
  if (t.is(clang::tok::string_literal)) {
    StringRef s(t.getLiteralData(), t.getLength());
    RewriteString(unquoteStr(s), t.getLocation());
    return;
  } else if (!t.isAnyIdentifier()) {
    // If it's neither a string nor an identifier, we don't care.
    return;
  }
  StringRef name = t.getRawIdentifier();
  clang::SourceLocation sl = t.getLocation();
  FindAndReplace(name, sl, CUDA_RENAMES_MAP());
}

void HipifyAction::FindAndReplace(StringRef name,
                                  clang::SourceLocation sl,
                                  const std::map<StringRef, hipCounter> &repMap,
                                  bool bReplace) {
  const auto found = repMap.find(name);
  if (found == repMap.end()) {
    // So it's an identifier, but not CUDA? Boring.
    return;
  }
  Statistics::current().incrementCounter(found->second, name.str());
  clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
  // Warn about the deprecated identifier in CUDA but hipify it.
  if (Statistics::isCudaDeprecated(found->second)) {
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is deprecated in CUDA.");
    DE.Report(sl, ID) << found->first;
  }
  // Warn about the unsupported experimental identifier.
  if (Statistics::isHipExperimental(found->second) &&!Experimental) {
    std::string sWarn;
    Statistics::isToRoc(found->second) ? sWarn = sROC : sWarn = sHIP;
    sWarn = "" + sWarn;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is experimental in '%1'; to hipify it, use the '--experimental' option.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  // Warn about the identifier which is supported only for _v2 version of it
  // [NOTE]: Currently, only cuBlas is tracked for versioning and only for _v2;
  // cublas_v2.h has to be included in the source cuda file for hipification.
  if (Statistics::isHipSupportedV2Only(found->second) && found->second.apiType == API_BLAS && !insertedBLASHeader_V2) {
    std::string sWarn;
    Statistics::isToRoc(found->second) ? sWarn = sROC : sWarn = sHIP;
    sWarn = "" + sWarn;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "Only '%0_v2' version of '%0' is supported in '%1'; to hipify it, include 'cublas_v2.h' in the source.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  // Warn about the unsupported identifier.
  if (Statistics::isUnsupported(found->second)) {
    std::string sWarn;
    Statistics::isToRoc(found->second) ? sWarn = sROC : sWarn = sHIP;
    sWarn = "" + sWarn;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  if (!bReplace) {
    return;
  }
  StringRef repName = Statistics::isToRoc(found->second) ? (found->second.rocName.empty() ? found->second.hipName : found->second.rocName) : found->second.hipName;
  auto &SM = getCompilerInstance().getSourceManager();
  ct::Replacement Rep(SM, sl, name.size(), repName.str());
  clang::FullSourceLoc fullSL(sl, SM);
  insertReplacement(Rep, fullSL);
}

namespace {

clang::SourceRange getReadRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  bool beginSafe = !SM.isMacroBodyExpansion(begin) || clang::Lexer::isAtStartOfMacroExpansion(begin, SM, clang::LangOptions{});
  bool endSafe = !SM.isMacroBodyExpansion(end) || clang::Lexer::isAtEndOfMacroExpansion(end, SM, clang::LangOptions{});
  if (beginSafe && endSafe) {
    return {SM.getFileLoc(begin), SM.getFileLoc(end)};
  } else {
    return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
  }
}

clang::SourceRange getWriteRange(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  clang::SourceLocation begin = exprRange.getBegin();
  clang::SourceLocation end = exprRange.getEnd();
  // If the range is contained within a macro, update the macro definition.
  // Otherwise, use the file location and hope for the best.
  if (!SM.isMacroBodyExpansion(begin) || !SM.isMacroBodyExpansion(end)) {
    return {SM.getExpansionLoc(begin), SM.getExpansionLoc(end)};
  }
  return {SM.getSpellingLoc(begin), SM.getSpellingLoc(end)};
}

StringRef readSourceText(clang::SourceManager &SM, const clang::SourceRange &exprRange) {
  return clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(getReadRange(SM, exprRange)), SM, clang::LangOptions(), nullptr);
}

/**
  * Get a string representation of the expression `arg`, unless it's a defaulting function
  * call argument, in which case get a 0. Used for building argument lists to kernel calls.
  */
std::string stringifyZeroDefaultedArg(clang::SourceManager &SM, const clang::Expr *arg) {
  if (clang::isa<clang::CXXDefaultArgExpr>(arg)) return "0";
  else return std::string(readSourceText(SM, arg->getSourceRange()));
}

} // anonymous namespace

bool HipifyAction::Exclude(const hipCounter &hipToken) {
  switch (hipToken.type) {
    case CONV_INCLUDE_CUDA_MAIN_H:
      switch (hipToken.apiType) {
        case API_DRIVER:
        case API_RUNTIME:
          if (insertedRuntimeHeader) return true;
          insertedRuntimeHeader = true;
          return false;
        case API_BLAS:
          if (insertedBLASHeader) return true;
          insertedBLASHeader = true;
          return false;
        case API_RAND:
          if (hipToken.hipName == s_hiprand_kernel_h) {
            if (insertedRAND_kernelHeader) return true;
            insertedRAND_kernelHeader = true;
            return false;
          } else if (hipToken.hipName == s_hiprand_h) {
            if (insertedRANDHeader) return true;
            insertedRANDHeader = true;
            return false;
          }
        case API_DNN:
          if (insertedDNNHeader) return true;
          insertedDNNHeader = true;
          return false;
        case API_FFT:
          if (insertedFFTHeader) return true;
          insertedFFTHeader = true;
          return false;
        case API_COMPLEX:
          if (insertedComplexHeader) return true;
          insertedComplexHeader = true;
          return false;
        case API_SPARSE:
          if (insertedSPARSEHeader) return true;
          insertedSPARSEHeader = true;
          return false;
        case API_SOLVER:
          if (insertedSOLVERHeader) return true;
          insertedSOLVERHeader = true;
          return false;
        default:
          return false;
      }
      return false;
    case CONV_INCLUDE_CUDA_MAIN_V2_H:
      switch (hipToken.apiType) {
        case API_BLAS:
          if (insertedBLASHeader_V2) return true;
          insertedBLASHeader_V2 = true;
          if (insertedBLASHeader) return true;
          return false;
        case API_SPARSE:
          if (insertedSPARSEHeader_V2) return true;
          insertedSPARSEHeader_V2 = true;
          if (insertedSPARSEHeader) return true;
          return false;
        default:
          return false;
      }
      return false;
    case CONV_INCLUDE:
      if (hipToken.hipName.empty()) return true;
      switch (hipToken.apiType) {
        case API_RAND:
          if (hipToken.hipName == s_hiprand_kernel_h) {
            if (insertedRAND_kernelHeader) return true;
            insertedRAND_kernelHeader = true;
          }
          return false;
        default:
          return false;
      }
      return false;
    default:
      return false;
  }
}

void HipifyAction::InclusionDirective(clang::SourceLocation hash_loc,
                                      const clang::Token&,
                                      StringRef file_name,
                                      bool is_angled,
                                      clang::CharSourceRange filename_range,
                                      const clang::FileEntry*, StringRef,
                                      StringRef, const clang::Module*) {
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(hash_loc)) return;
  if (!firstHeader) {
    firstHeader = true;
    firstHeaderLoc = hash_loc;
  }
  const auto found = CUDA_INCLUDE_MAP.find(file_name);
  if (found == CUDA_INCLUDE_MAP.end()) return;
  bool exclude = Exclude(found->second);
  Statistics::current().incrementCounter(found->second, file_name.str());
  clang::SourceLocation sl = filename_range.getBegin();
  if (Statistics::isUnsupported(found->second)) {
    clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
    std::string sWarn;
    Statistics::isToRoc(found->second) ? sWarn = sROC : sWarn = sHIP;
    const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "'%0' is unsupported header in '%1'.");
    DE.Report(sl, ID) << found->first << sWarn;
    return;
  }
  clang::StringRef newInclude;
  // Keep the same include type that the user gave.
  if (!exclude) {
    clang::SmallString<128> includeBuffer;
    llvm::StringRef name = Statistics::isToRoc(found->second) ? (found->second.rocName.empty() ? found->second.hipName : found->second.rocName) : found->second.hipName;
    if (is_angled) newInclude = llvm::Twine("<" + name+ ">").toStringRef(includeBuffer);
    else           newInclude = llvm::Twine("\"" + name + "\"").toStringRef(includeBuffer);
  } else {
    // hashLoc is location of the '#', thus replacing the whole include directive by empty newInclude starting with '#'.
    sl = hash_loc;
  }
  const char *B = SM.getCharacterData(sl);
  const char *E = SM.getCharacterData(filename_range.getEnd());
  ct::Replacement Rep(SM, sl, E - B, newInclude.str());
  insertReplacement(Rep, clang::FullSourceLoc{sl, SM});
}

void HipifyAction::PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) {
  if (pragmaOnce) return;
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(Loc)) return;
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  clang::Token tok;
  PP.Lex(tok);
  StringRef Text(SM.getCharacterData(tok.getLocation()), tok.getLength());
  if (Text == sOnce) {
    pragmaOnce = true;
    pragmaOnceLoc = tok.getEndLoc();
  }
}

bool HipifyAction::cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result) {
  auto *launchKernel = Result.Nodes.getNodeAs<clang::CUDAKernelCallExpr>(sCudaLaunchKernel);
  if (!launchKernel) return false;
  auto *calleeExpr = launchKernel->getCallee();
  if (!calleeExpr) return false;
  auto *caleeDecl = launchKernel->getDirectCallee();
  if (!caleeDecl) return false;
  auto *config = launchKernel->getConfig();
  if (!config) return false;
  if (CudaKernelExecutionSyntax && !HipKernelExecutionSyntax) return false;
  clang::SmallString<40> XStr;
  llvm::raw_svector_ostream OS(XStr);
  clang::LangOptions DefaultLangOptions;
  auto *SM = Result.SourceManager;
  clang::SourceRange sr = calleeExpr->getSourceRange();
  std::string kern = readSourceText(*SM, sr).str();
  OS << sHipLaunchKernelGGL << "(";
  if (caleeDecl->isTemplateInstantiation()) {
    OS << sHIP_KERNEL_NAME << "(";
    std::string cub = sCub + "::";
    std::string hipcub;
    const auto found = CUDA_CUB_NAMESPACE_MAP.find(sCub);
    if (found != CUDA_CUB_NAMESPACE_MAP.end()) {
      hipcub = found->second.hipName.str() + "::";
    } else {
      hipcub = sHipcub + "::";
    }
    size_t pos = kern.find(cub);
    while (pos != std::string::npos) {
      kern.replace(pos, cub.size(), hipcub);
      pos = kern.find(cub, pos + hipcub.size());
    }
  }
  OS << kern;
  if (caleeDecl->isTemplateInstantiation()) OS << ")";
  OS << ", ";
  // Next up are the four kernel configuration parameters, the last two of which are optional and default to zero.
  // Copy the two dimensional arguments verbatim.
  for (unsigned int i = 0; i < 2; ++i) {
    const std::string sArg = readSourceText(*SM, config->getArg(i)->getSourceRange()).str();
    bool bDim3 = std::equal(sDim3.begin(), sDim3.end(), sArg.c_str());
    OS << (bDim3 ? "" : sDim3) << sArg << (bDim3 ? "" : ")") << ", ";
  }
  // The stream/memory arguments default to zero if omitted.
  OS << stringifyZeroDefaultedArg(*SM, config->getArg(2)) << ", ";
  OS << stringifyZeroDefaultedArg(*SM, config->getArg(3));
  // If there are ordinary arguments to the kernel, just copy them verbatim into our new call.
  int numArgs = launchKernel->getNumArgs();
  if (numArgs > 0) {
    OS << ", ";
    // Start of the first argument.
    clang::SourceLocation argStart = llcompat::getBeginLoc(launchKernel->getArg(0));
    // End of the last argument.
    clang::SourceLocation argEnd = llcompat::getEndLoc(launchKernel->getArg(numArgs - 1));
    OS << readSourceText(*SM, {argStart, argEnd});
  }
  OS << ")";
  clang::SourceLocation launchKernelExprLocBeg = launchKernel->getExprLoc();
  clang::SourceLocation launchKernelExprLocEnd = launchKernelExprLocBeg.isMacroID() ? llcompat::getEndOfExpansionRangeForLoc(*SM, launchKernelExprLocBeg) : llcompat::getEndLoc(launchKernel);
  clang::SourceLocation launchKernelEnd = llcompat::getEndLoc(launchKernel);
  clang::BeforeThanCompare<clang::SourceLocation> isBefore(*SM);
  launchKernelExprLocEnd = isBefore(launchKernelEnd, launchKernelExprLocEnd) ? launchKernelExprLocEnd : launchKernelEnd;
  clang::SourceRange replacementRange = getWriteRange(*SM, {launchKernelExprLocBeg, launchKernelExprLocEnd});
  clang::SourceLocation launchBeg = replacementRange.getBegin();
  clang::SourceLocation launchEnd = replacementRange.getEnd();
  if (isBefore(launchBeg, launchEnd)) {
    size_t length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(launchEnd, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(launchBeg);
    ct::Replacement Rep(*SM, launchBeg, length, OS.str());
    clang::FullSourceLoc fullSL(launchBeg, *SM);
    insertReplacement(Rep, fullSL);
    hipCounter counter = {sHipLaunchKernelGGL, "", ConvTypes::CONV_KERNEL_LAUNCH, ApiTypes::API_RUNTIME};
    Statistics::current().incrementCounter(counter, sCudaLaunchKernel.str());
    return true;
  }
  return false;
}

bool HipifyAction::cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result) {
  if (const clang::CallExpr *call = Result.Nodes.getNodeAs<clang::CallExpr>(sCudaDeviceFuncCall)) {
    auto *funcDcl = call->getDirectCallee();
    if (!funcDcl) return false;
    FindAndReplace(funcDcl->getDeclName().getAsString(), llcompat::getBeginLoc(call), CUDA_DEVICE_FUNCTION_MAP, false);
    return true;
  }
  return false;
}

bool HipifyAction::cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::TypedefNameDecl>(sCubNamespacePrefix)) {
    clang::QualType QT = decl->getUnderlyingType();
    auto *t = QT.getTypePtr();
    if (!t) return false;
    const clang::ElaboratedType *et = t->getAs<clang::ElaboratedType>();
    if (!et) return false;
    const clang::NestedNameSpecifier *nns = et->getQualifier();
    if (!nns) return false;
    const clang::NamespaceDecl *nsd = nns->getAsNamespace();
    if (!nsd) return false;
    const clang::TypeSourceInfo *si = decl->getTypeSourceInfo();
    const clang::TypeLoc tloc = si->getTypeLoc();
    const clang::SourceRange sr = tloc.getSourceRange();
    std::string name = nsd->getDeclName().getAsString();
    FindAndReplace(name, GetSubstrLocation(name, sr), CUDA_CUB_NAMESPACE_MAP);
    return true;
  }
  return false;
}

bool HipifyAction::cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::UsingDirectiveDecl>(sCubUsingNamespaceDecl)) {
    if (auto nsd = decl->getNominatedNamespace()) {
      FindAndReplace(nsd->getDeclName().getAsString(), decl->getIdentLocation(), CUDA_CUB_NAMESPACE_MAP);
      return true;
    }
  }
  return false;
}

bool HipifyAction::cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result) {
  if (auto *decl = Result.Nodes.getNodeAs<clang::FunctionTemplateDecl>(sCubFunctionTemplateDecl)) {
    auto *Tparams = decl->getTemplateParameters();
    bool ret = false;
    for (size_t I = 0; I < Tparams->size(); ++I) {
      const clang::ValueDecl *valueDecl = dyn_cast<clang::ValueDecl>(Tparams->getParam(I));
      if (!valueDecl) continue;
      clang::QualType QT = valueDecl->getType();
      auto *t = QT.getTypePtr();
      if (!t) continue;
      const clang::ElaboratedType *et = t->getAs<clang::ElaboratedType>();
      if (!et) continue;
      const clang::NestedNameSpecifier *nns = et->getQualifier();
      if (!nns) continue;
      const clang::NamespaceDecl *nsd = nns->getAsNamespace();
      if (!nsd) continue;
      const clang::SourceRange sr = valueDecl->getSourceRange();
      std::string name = nsd->getDeclName().getAsString();
      FindAndReplace(name, GetSubstrLocation(name, sr), CUDA_CUB_NAMESPACE_MAP);
      ret = true;
    }
    return ret;
  }
  return false;
}

bool HipifyAction::cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result) {
  if (auto *call = Result.Nodes.getNodeAs<clang::CallExpr>(sCudaHostFuncCall)) {
    if (!call->getNumArgs()) return false;
    auto *funcDcl = call->getDirectCallee();
    if (!funcDcl) return false;
    std::string sName = funcDcl->getDeclName().getAsString();
    auto it = FuncArgCasts.find(sName);
    if (it == FuncArgCasts.end()) return false;
    auto castStructs = it->second;
    for (auto cc : castStructs) {
      if (cc.isToMIOpen != TranslateToMIOpen || cc.isToRoc != TranslateToRoc) continue;
      clang::LangOptions DefaultLangOptions;
      for (auto c : cc.castMap) {
        size_t length = 0;
        unsigned int argNum = c.first;
        clang::SmallString<40> XStr;
        llvm::raw_svector_ostream OS(XStr);
        auto *SM = Result.SourceManager;
        clang::SourceRange sr, replacementRange;
        clang::SourceLocation s, e;
        if (argNum < call->getNumArgs()) {
          sr = call->getArg(argNum)->getSourceRange();
          replacementRange = getWriteRange(*SM, { sr.getBegin(), sr.getEnd() });
          s = replacementRange.getBegin();
          e = replacementRange.getEnd();
        } else {
          s = e = call->getEndLoc();
        }
        switch (c.second.castType) {
          case e_remove_argument:
          {
            OS << "";
            if (argNum < call->getNumArgs() - 1) {
              e = call->getArg(argNum + 1)->getBeginLoc();
            }
            else {
              e = call->getEndLoc();
              if (call->getNumArgs() > 1) {
                auto prevComma = clang::Lexer::findNextToken(call->getArg(argNum - 1)->getSourceRange().getEnd(), *SM, DefaultLangOptions);
                if (!prevComma)
                  s = call->getEndLoc();
                s = prevComma->getLocation();
              }
            }
            length = SM->getCharacterData(e) - SM->getCharacterData(s);
            break;
          }
          case e_move_argument:
          {
            std::string sArg;
            clang::SmallString<40> dst_XStr;
            llvm::raw_svector_ostream dst_OS(dst_XStr);
            if (c.second.numberToMoveOrCopy > 1) {
              if ((argNum + c.second.numberToMoveOrCopy - 1) >= call->getNumArgs())
                continue;
              sr = call->getArg(argNum + c.second.numberToMoveOrCopy - 1)->getSourceRange();
              sr.setBegin(call->getArg(argNum)->getBeginLoc());
            }
            sArg = readSourceText(*SM, sr).str();
            if (c.second.moveOrCopyTo < call->getNumArgs())
              dst_OS << sArg << ", ";
            else
              dst_OS << ", " << sArg;
            clang::SourceLocation dst_s;
            if (c.second.moveOrCopyTo < call->getNumArgs())
              dst_s = call->getArg(c.second.moveOrCopyTo)->getBeginLoc();
            else
              dst_s = call->getEndLoc();
            ct::Replacement dst_Rep(*SM, dst_s, 0, dst_OS.str());
            clang::FullSourceLoc dst_fullSL(dst_s, *SM);
            insertReplacement(dst_Rep, dst_fullSL);
            OS << "";
            if (argNum < call->getNumArgs())
              e = call->getArg(argNum + c.second.numberToMoveOrCopy)->getBeginLoc();
            else
              e = call->getEndLoc();
            length = SM->getCharacterData(e) - SM->getCharacterData(s);
            break;
          }
          case e_add_const_argument:
          {
            if (argNum < call->getNumArgs())
              OS << c.second.constValToAddOrReplace << ", ";
            else
              OS << ", " << c.second.constValToAddOrReplace;
            break;
          }
          case e_add_var_argument:
          {
            if (argNum >= call->getNumArgs())
              continue;
            sr = call->getArg(argNum)->getSourceRange();
            sr.setBegin(call->getArg(argNum)->getBeginLoc());
            std::string sArg = readSourceText(*SM, sr).str();
            if (c.second.moveOrCopyTo < call->getNumArgs()) {
              OS << sArg << ", ";
              s = call->getArg(c.second.moveOrCopyTo)->getBeginLoc();
            }
            else {
              OS << ", " << sArg;
              s = call->getEndLoc();
            }
            break;
          }
          case e_replace_argument_with_const:
          {
            if (argNum >= call->getNumArgs())
              break;
            OS << c.second.constValToAddOrReplace;
            length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(e, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(s);
            break;
          }
          default:
            OS << getCastType(c.second.castType) << "(" << readSourceText(*SM, sr) << ")";
            length = SM->getCharacterData(clang::Lexer::getLocForEndOfToken(e, 0, *SM, DefaultLangOptions)) - SM->getCharacterData(s);
            break;
        }
        ct::Replacement Rep(*SM, s, length, OS.str());
        clang::FullSourceLoc fullSL(s, *SM);
        insertReplacement(Rep, fullSL);
        switch (c.second.castWarn) {
          case cw_DataLoss: {
            clang::DiagnosticsEngine &DE = getCompilerInstance().getDiagnostics();
            const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "Possible data loss in %0 argument of '%1'.");
            DE.Report(fullSL, ID) << argNum+1 << sName;
            break;
          }
          case cw_None:
          default: break;
        }
      }
    }
    return true;
  }
  return false;
}

bool HipifyAction::cudaOverloadedHostFuncCall(const mat::MatchFinder::MatchResult& Result) {
  if (auto* call = Result.Nodes.getNodeAs<clang::CallExpr>(sCudaOverloadedHostFuncCall)) {
    if (!call->getNumArgs()) return false;
    auto* funcDcl = call->getDirectCallee();
    if (!funcDcl) return false;
    std::string name = funcDcl->getDeclName().getAsString();
    const auto found = CUDA_RENAMES_MAP().find(name);
    if (found == CUDA_RENAMES_MAP().end()) return false;
    if (!Statistics::isCudaOverloaded(found->second)) return false;
    auto it = FuncOverloads.find(name);
    if (it == FuncOverloads.end()) return false;
    auto FuncOverloadsStruct = it->second;
    if (FuncOverloadsStruct.isToMIOpen != TranslateToMIOpen || FuncOverloadsStruct.isToRoc != TranslateToRoc) return false;
    unsigned numArgs = call->getNumArgs();
    auto itNumArgs = FuncOverloadsStruct.overloadMap.find(numArgs);
    if (itNumArgs == FuncOverloadsStruct.overloadMap.end()) return false;
    auto overrideInfo = itNumArgs->second;
    auto counter = overrideInfo.counter;
    // check if SUPPORTED
    auto* SM = Result.SourceManager;
    clang::SourceLocation s;
    switch (overrideInfo.overloadType) {
      case ot_arguments_number:
      default:
      {
        s = call->getBeginLoc();
        ct::Replacement Rep(*SM, s, name.size(), counter.hipName.str());
        clang::FullSourceLoc fullSL(s, *SM);
        insertReplacement(Rep, fullSL);
        break;
      }
    }
    return true;
  }
  return false;
}

bool HipifyAction::half2Member(const mat::MatchFinder::MatchResult &Result) {
  if (auto *expr = Result.Nodes.getNodeAs<clang::MemberExpr>(sHalf2Member)) {
    auto *baseExpr = expr->getBase();
    if (!baseExpr) return false;
    clang::QualType QT = baseExpr->getType();
    if (QT.getAsString() != "half2") return false;
    auto *val = expr->getMemberDecl();
    if (!val) return false;
    std::string valName = val->getNameAsString();
    if (valName != "x" && valName != "y") return false;
    const clang::SourceRange sr = expr->getSourceRange();
    std::string exprName = readSourceText(*Result.SourceManager, sr).str();
    clang::SmallString<40> XStr;
    llvm::raw_svector_ostream OS(XStr);
    OS << "reinterpret_cast<half&>(" << exprName << ")";
    ct::Replacement Rep(*Result.SourceManager, sr.getBegin(), exprName.size(), OS.str());
    clang::FullSourceLoc fullSL(sr.getBegin(), *Result.SourceManager);
    insertReplacement(Rep, fullSL);
    if (!NoWarningsUndocumented) {
      clang::DiagnosticsEngine& DE = getCompilerInstance().getDiagnostics();
      const auto ID = DE.getCustomDiagID(clang::DiagnosticsEngine::Warning, "Undocumented feature. CUDA API does not explicitly define the 'x' and 'y' members of 'half2' and the access through the dot operator, while in practice, nvcc supports it and treats them as 'half'. AMD HIP does define the 'x' and 'y' members of 'half2' as 'unsigned short'. Thus, without 'reinterpret_cast' to 'half' of the 'half2' members, the resulting values in the hipified code are incorrect and differ from CUDA ones. The '%0' will be transformed to 'reinterpret_cast<half&>(%0)'.");
      DE.Report(fullSL, ID) << exprName;
    }
    return true;
  }
  return false;
}

bool HipifyAction::dataTypeSelection(const mat::MatchFinder::MatchResult& Result) {
  if (auto *vardecl = Result.Nodes.getNodeAs<clang::VarDecl>(sDataTypeSelection)) {
    clang::QualType QT = vardecl->getType();
    std::string name = QT.getAsString();
    const auto found = TypeOverloads.find(name);
    if (found == TypeOverloads.end()) return false;
    if (name.find("enum ") == 0)
      name.erase(0, 5);
    std::string correct_name = found->second;
    const clang::TypeSourceInfo *si = vardecl->getTypeSourceInfo();
    const clang::TypeLoc tloc = si->getTypeLoc();
    const clang::SourceRange sr = tloc.getSourceRange();
    const clang::SourceLocation sl = sr.getBegin();
    auto *SM = Result.SourceManager;
    ct::Replacement Rep(*SM, sl, name.size(), correct_name);
    clang::FullSourceLoc fullSL(sl, *SM);
    insertReplacement(Rep, fullSL);
  }
  return false;
}

void HipifyAction::insertReplacement(const ct::Replacement &rep, const clang::FullSourceLoc &fullSL) {
  llcompat::insertReplacement(*replacements, rep);
  if (PrintStats || PrintStatsCSV) {
    rep.getLength();
    Statistics::current().lineTouched(fullSL.getExpansionLineNumber());
    Statistics::current().bytesChanged(rep.getLength());
  }
}

std::unique_ptr<clang::ASTConsumer> HipifyAction::CreateASTConsumer(clang::CompilerInstance &CI, StringRef) {
  Finder.reset(new mat::MatchFinder);
  // Replace the <<<...>>> language extension with a hip kernel launch
  Finder->addMatcher(mat::cudaKernelCallExpr(mat::isExpansionInMainFile()).bind(sCudaLaunchKernel), this);
  if (!NoUndocumented) {
    Finder->addMatcher(
      mat::memberExpr(
        mat::isExpansionInMainFile(),
        mat::unless(
          mat::hasParent(
            mat::cxxReinterpretCastExpr(
              mat::hasDestinationType(
                mat::referenceType()
              )
            )
          )
        )
      ).bind(sHalf2Member),
      this
    );
  }
  Finder->addMatcher(
    mat::callExpr(
      mat::isExpansionInMainFile(),
      mat::callee(
        mat::functionDecl(
          mat::hasAnyName(
            sCudaGetSymbolAddress,
            sCudaGetSymbolSize,
            sCudaMemcpyFromSymbol,
            sCudaMemcpyFromSymbolAsync,
            sCudaMemcpyToSymbol,
            sCudaMemcpyToSymbolAsync,
            sCudaGraphAddMemcpyNodeToSymbol,
            sCudaGraphAddMemcpyNodeFromSymbol,
            sCudaGraphMemcpyNodeSetParamsToSymbol,
            sCudaGraphMemcpyNodeSetParamsFromSymbol,
            sCudaGraphExecMemcpyNodeSetParamsToSymbol,
            sCudaGraphExecMemcpyNodeSetParamsFromSymbol,
            sCuOccupancyMaxPotentialBlockSize,
            sCuOccupancyMaxPotentialBlockSizeWithFlags,
            sCudaGetTextureReference,
            sCudnnGetConvolutionForwardWorkspaceSize,
            sCudnnGetConvolutionBackwardDataWorkspaceSize,
            sCudnnFindConvolutionForwardAlgorithmEx,
            sCudnnSetPooling2dDescriptor,
            sCudnnGetPooling2dDescriptor,
            sCudnnSetPoolingNdDescriptor,
            sCudnnGetPoolingNdDescriptor,
            sCudnnSetLRNDescriptor,
            sCudnnGetRNNDescriptor_v6,
            sCudnnSetRNNDescriptor_v6,
            sCudnnSoftmaxForward,
            sCudnnSoftmaxBackward,
            sCudnnConvolutionForward,
            sCudnnConvolutionBackwardData,
            sCudnnRNNBackwardWeights,
            sCusparseZgpsvInterleavedBatch,
            sCusparseCgpsvInterleavedBatch,
            sCusparseDgpsvInterleavedBatch,
            sCusparseSgpsvInterleavedBatch,
            sCusparseZgpsvInterleavedBatch_bufferSizeExt,
            sCusparseCgpsvInterleavedBatch_bufferSizeExt,
            sCusparseDgpsvInterleavedBatch_bufferSizeExt,
            sCusparseSgpsvInterleavedBatch_bufferSizeExt,
            sCusparseZgtsvInterleavedBatch,
            sCusparseCgtsvInterleavedBatch,
            sCusparseDgtsvInterleavedBatch,
            sCusparseSgtsvInterleavedBatch,
            sCusparseZgtsvInterleavedBatch_bufferSizeExt,
            sCusparseCgtsvInterleavedBatch_bufferSizeExt,
            sCusparseDgtsvInterleavedBatch_bufferSizeExt,
            sCusparseSgtsvInterleavedBatch_bufferSizeExt,
            sCusparseZcsrilu02,
            sCusparseCcsrilu02,
            sCusparseDcsrilu02,
            sCusparseScsrilu02,
            sCusparseZcsrilu02_analysis,
            sCusparseCcsrilu02_analysis,
            sCusparseDcsrilu02_analysis,
            sCusparseScsrilu02_analysis,
            sCusparseZcsric02_analysis,
            sCusparseCcsric02_analysis,
            sCusparseDcsric02_analysis,
            sCusparseScsric02_analysis,
            sCusparseZcsric02_bufferSize,
            sCusparseCcsric02_bufferSize,
            sCusparseDcsric02_bufferSize,
            sCusparseScsric02_bufferSize,
            sCusparseZbsrilu02,
            sCusparseCbsrilu02,
            sCusparseDbsrilu02,
            sCusparseSbsrilu02,
            sCusparseZbsrilu02_analysis,
            sCusparseCbsrilu02_analysis,
            sCusparseDbsrilu02_analysis,
            sCusparseSbsrilu02_analysis,
            sCusparseZbsric02,
            sCusparseCbsric02,
            sCusparseDbsric02,
            sCusparseSbsric02,
            sCusparseZbsric02_analysis,
            sCusparseCbsric02_analysis,
            sCusparseDbsric02_analysis,
            sCusparseSbsric02_analysis,
            sCusparseZbsric02_bufferSize,
            sCusparseCbsric02_bufferSize,
            sCusparseDbsric02_bufferSize,
            sCusparseSbsric02_bufferSize,
            sCusparseZbsrsm2_bufferSize,
            sCusparseCbsrsm2_bufferSize,
            sCusparseDbsrsm2_bufferSize,
            sCusparseSbsrsm2_bufferSize,
            sCusparseZcsrsm2_solve,
            sCusparseCcsrsm2_solve,
            sCusparseDcsrsm2_solve,
            sCusparseScsrsm2_solve,
            sCusparseZcsrsm2_analysis,
            sCusparseCcsrsm2_analysis,
            sCusparseDcsrsm2_analysis,
            sCusparseScsrsm2_analysis,
            sCusparseScsrsm2_bufferSizeExt,
            sCusparseDcsrsm2_bufferSizeExt,
            sCusparseCcsrsm2_bufferSizeExt,
            sCusparseZcsrsm2_bufferSizeExt,
            sCusparseZgemvi_bufferSize,
            sCusparseCgemvi_bufferSize,
            sCusparseDgemvi_bufferSize,
            sCusparseSgemvi_bufferSize,
            sCusparseZcsrsv2_solve,
            sCusparseCcsrsv2_solve,
            sCusparseDcsrsv2_solve,
            sCusparseScsrsv2_solve,
            sCusparseZcsrsv2_analysis,
            sCusparseCcsrsv2_analysis,
            sCusparseDcsrsv2_analysis,
            sCusparseScsrsv2_analysis,
            sCusparseZcsrmv,
            sCusparseCcsrmv,
            sCusparseDcsrmv,
            sCusparseScsrmv,
            sCusparseZbsrsv2_solve,
            sCusparseCbsrsv2_solve,
            sCusparseDbsrsv2_solve,
            sCusparseSbsrsv2_solve,
            sCusparseSbsrsv2_analysis,
            sCusparseDbsrsv2_analysis,
            sCusparseCbsrsv2_analysis,
            sCusparseZbsrsv2_analysis,
            sCusparseZcsrmm,
            sCusparseCcsrmm,
            sCusparseDcsrmm,
            sCusparseScsrmm,
            sCusparseZcsrgeam2,
            sCusparseCcsrgeam2,
            sCusparseDcsrgeam2,
            sCusparseScsrgeam2,
            sCusparseZbsrsv2_bufferSize,
            sCusparseCbsrsv2_bufferSize,
            sCusparseDbsrsv2_bufferSize,
            sCusparseSbsrsv2_bufferSize,
            sCusparseZcsrsv2_bufferSize,
            sCusparseCcsrsv2_bufferSize,
            sCusparseDcsrsv2_bufferSize,
            sCusparseScsrsv2_bufferSize,
            sCusparseZcsrgemm2,
            sCusparseCcsrgemm2,
            sCusparseDcsrgemm2,
            sCusparseScsrgemm2,
            sCusparseZcsrilu02_bufferSize,
            sCusparseCcsrilu02_bufferSize,
            sCusparseDcsrilu02_bufferSize,
            sCusparseScsrilu02_bufferSize,
            sCusparseZbsrilu02_bufferSize,
            sCusparseCbsrilu02_bufferSize,
            sCusparseDbsrilu02_bufferSize,
            sCusparseSbsrilu02_bufferSize,
            sCusparseCsr2cscEx2_bufferSize,
            sCusparseSparseToDense,
            sCusparseSparseToDense_bufferSize,
            sCusparseDenseToSparse_bufferSize,
            sCusparseDenseToSparse_analysis,
            sCusparseSpMM,
            sCusparseSpMM_bufferSize,
            sCusparseSpSM_analysis,
            sCusparseSpSM_solve,
            sCusparseXcsrgeam2Nnz,
            sCudaMallocHost,
            sCusparseSpVV,
            sCusparseSpVV_bufferSize,
            sCusparseSpMV,
            sCusparseSpMV_bufferSize,
            sCusparseSpMM_preprocess,
            sCusparseSpSV_bufferSize
          )
        )
      )
    ).bind(sCudaHostFuncCall),
    this
  );
  Finder->addMatcher(
    mat::callExpr(
      mat::isExpansionInMainFile(),
      mat::callee(
        mat::functionDecl(
          mat::hasAnyName(
            sCudaEventCreate,
            sCudaGraphInstantiate
          )
        )
      )
    ).bind(sCudaOverloadedHostFuncCall),
    this
  );
  Finder->addMatcher(
    mat::callExpr(
      mat::isExpansionInMainFile(),
      mat::callee(
        mat::functionDecl(
          mat::anyOf(
            mat::hasAttr(clang::attr::CUDADevice),
            mat::hasAttr(clang::attr::CUDAGlobal)
          ),
          mat::unless(mat::hasAttr(clang::attr::CUDAHost))
        )
      )
    ).bind(sCudaDeviceFuncCall),
    this
  );
  Finder->addMatcher(
    mat::typedefDecl(
      mat::isExpansionInMainFile(),
      mat::hasType(
        mat::elaboratedType(
          mat::hasQualifier(
            mat::specifiesNamespace(
              mat::hasName(sCub)
            )
          )
        )
       )
    ).bind(sCubNamespacePrefix),
    this
  );
  // TODO: Maybe worth to make it more concrete based on final cubFunctionTemplateDecl
  Finder->addMatcher(
    mat::functionTemplateDecl(
      mat::isExpansionInMainFile()
    ).bind(sCubFunctionTemplateDecl),
    this
  );
  // TODO: Maybe worth to make it more concrete
  Finder->addMatcher(
    mat::usingDirectiveDecl(
      mat::isExpansionInMainFile()
    ).bind(sCubUsingNamespaceDecl),
    this
  );
  Finder->addMatcher(
    mat::varDecl(
      mat::isExpansionInMainFile(),
      mat::hasType(
        mat::qualType(
          mat::hasCanonicalType(mat::enumType())
        )
      )
    ).bind(sDataTypeSelection),
    this
  );
  // Ownership is transferred to the caller.
  return Finder->newASTConsumer();
}

void HipifyAction::Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) {
  auto &SM = getCompilerInstance().getSourceManager();
  if (!SM.isWrittenInMainFile(Loc)) return;
  StringRef Text(SM.getCharacterData(MacroNameTok.getLocation()), MacroNameTok.getLength());
  Ifndefs.insert(std::make_pair(Text.str(), MacroNameTok.getEndLoc()));
}

void HipifyAction::EndSourceFileAction() {
  // Insert the hip header, if we didn't already do it by accident during substitution.
  if (!insertedRuntimeHeader) {
    // It's not sufficient to just replace CUDA headers with hip ones, because numerous CUDA headers are
    // implicitly included by the compiler. Instead, we _delete_ CUDA headers, and unconditionally insert
    // one copy of the hip include into every file.
    bool placeForIncludeCalculated = false;
    clang::SourceLocation sl, controllingMacroLoc;
    auto &CI = getCompilerInstance();
    auto &SM = CI.getSourceManager();
    const clang::IdentifierInfo *controllingMacro = llcompat::getControllingMacro(CI);
    if (controllingMacro) {
      auto found = Ifndefs.find(controllingMacro->getName().str());
      if (found != Ifndefs.end()) {
        controllingMacroLoc = found->second;
        placeForIncludeCalculated = true;
      }
    }
    if (pragmaOnce) {
      if (placeForIncludeCalculated) sl = pragmaOnceLoc < controllingMacroLoc ? pragmaOnceLoc : controllingMacroLoc;
      else                           sl = pragmaOnceLoc;
      placeForIncludeCalculated = true;
    }
    if (!placeForIncludeCalculated) {
      if (firstHeader)               sl = firstHeaderLoc;
      else                           sl = SM.getLocForStartOfFile(SM.getMainFileID());
    }
    clang::FullSourceLoc fullSL(sl, SM);
    ct::Replacement Rep(SM, sl, 0, "\n#include <hip/hip_runtime.h>\n");
    insertReplacement(Rep, fullSL);
  }
  clang::ASTFrontendAction::EndSourceFileAction();
}

namespace {

/**
  * A silly little class to proxy PPCallbacks back to the HipifyAction class.
  */
class PPCallbackProxy : public clang::PPCallbacks {
  HipifyAction &hipifyAction;

public:
  explicit PPCallbackProxy(HipifyAction &action): hipifyAction(action) {}
  void InclusionDirective(clang::SourceLocation hash_loc, const clang::Token &include_token,
                          StringRef file_name, bool is_angled, clang::CharSourceRange filename_range,
#if LLVM_VERSION_MAJOR < 15
                          const clang::FileEntry *file,
#elif LLVM_VERSION_MAJOR == 15
                          Optional<clang::FileEntryRef> file,
#else
                          clang::OptionalFileEntryRef file,
#endif
                          StringRef search_path, StringRef relative_path,
#if LLVM_VERSION_MAJOR < 19
                          const clang::Module *SuggestedModule
#else
                          const clang::Module *SuggestedModule,
                          bool ModuleImported
#endif
#if LLVM_VERSION_MAJOR > 6
                        , clang::SrcMgr::CharacteristicKind FileType
#endif
                         ) override {
#if LLVM_VERSION_MAJOR < 15
    auto f = file;
#else
    auto f = &file->getFileEntry();
#endif
    hipifyAction.InclusionDirective(hash_loc, include_token, file_name, is_angled, filename_range, f, search_path, relative_path, SuggestedModule);
  }

  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer) override {
    hipifyAction.PragmaDirective(Loc, Introducer);
  }

  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD) override {
    hipifyAction.Ifndef(Loc, MacroNameTok, MD);
  }

  void SourceRangeSkipped(clang::SourceRange Range, clang::SourceLocation EndifLoc) override {
    hipifyAction.AddSkippedSourceRange(Range);
  }
};
}

bool HipifyAction::BeginInvocation(clang::CompilerInstance &CI) {
  llcompat::RetainExcludedConditionalBlocks(CI);
  return true;
}

void HipifyAction::ExecuteAction() {
  clang::Preprocessor &PP = getCompilerInstance().getPreprocessor();
  // Register yourself as the preprocessor callback, by proxy.
  PP.addPPCallbacks(std::unique_ptr<PPCallbackProxy>(new PPCallbackProxy(*this)));
  // Now we're done futzing with the lexer, have the subclass proceeed with Sema and AST matching.
  clang::ASTFrontendAction::ExecuteAction();
  auto &SM = getCompilerInstance().getSourceManager();
  // Start lexing the specified input file.
  llcompat::Memory_Buffer FromFile = llcompat::getMemoryBuffer(SM);
  clang::Lexer RawLex(SM.getMainFileID(), FromFile, SM, PP.getLangOpts());
  RawLex.SetKeepWhitespaceMode(true);
  // Perform a token-level rewrite of CUDA identifiers to hip ones. The raw-mode lexer gives us enough
  // information to tell the difference between identifiers, string literals, and "other stuff". It also
  // ignores preprocessor directives, so this transformation will operate inside preprocessor-deleted code.
  clang::Token RawTok;
  RawLex.LexFromRawLexer(RawTok);
  while (RawTok.isNot(clang::tok::eof)) {
    RewriteToken(RawTok);
    RawLex.LexFromRawLexer(RawTok);
  }
}

void HipifyAction::AddSkippedSourceRange(clang::SourceRange Range) {
  SkippedSourceRanges.push_back(Range);
}

void HipifyAction::run(const mat::MatchFinder::MatchResult &Result) {
  if (cudaLaunchKernel(Result)) return;
  if (cudaHostFuncCall(Result)) return;
  if (cudaOverloadedHostFuncCall(Result)) return;
  if (cudaDeviceFuncCall(Result)) return;
  if (cubNamespacePrefix(Result)) return;
  if (cubFunctionTemplateDecl(Result)) return;
  if (cubUsingNamespaceDecl(Result)) return;
  if (UseHipDataType && dataTypeSelection(Result)) return;
  if (!NoUndocumented && half2Member(Result)) return;
}
