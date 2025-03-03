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

#pragma once

#include "llvm/ADT/StringRef.h"
#include <set>
#include <map>
#include "Statistics.h"

const std::string sHIP_version = Statistics::getHipVersion(HIP_LATEST);

// Maps CUDA header names to HIP header names
extern const std::map<llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP;
// Maps the names of CUDA DRIVER API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_TYPE_NAME_MAP;
// Maps the names of CUDA DRIVER API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_COMPLEX_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP;
// Maps the names of CUDA BLAS API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_TYPE_NAME_MAP;
// Maps the names of CUDA BLAS API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_BLAS_FUNCTION_MAP;
// Maps the names of CUDA RAND API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_RAND_TYPE_NAME_MAP;
// Maps the names of CUDA RAND API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_RAND_FUNCTION_MAP;
// Maps the names of CUDA DNN API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_DNN_TYPE_NAME_MAP;
// Maps the names of CUDA DNN API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_DNN_FUNCTION_MAP;
// Maps the names of CUDA FFT API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_FFT_TYPE_NAME_MAP;
// Maps the names of CUDA FFT API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_FFT_FUNCTION_MAP;
// Maps the names of CUDA SPARSE API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_SPARSE_TYPE_NAME_MAP;
// Maps the names of CUDA SPARSE API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_SPARSE_FUNCTION_MAP;
// Maps the names of CUDA CAFFE2 API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_CAFFE2_TYPE_NAME_MAP;
// Maps the names of CUDA CAFFE2 API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_CAFFE2_FUNCTION_MAP;
// Maps the names of CUDA Device types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_DEVICE_TYPE_NAME_MAP;
// Maps the names of CUDA Device functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_DEVICE_FUNCTION_MAP;
// Maps the names of CUDA CUB API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_CUB_TYPE_NAME_MAP;
// Maps the names of CUDA CUB API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_CUB_FUNCTION_MAP;
// Maps the names of CUDA CUB namespaces to the corresponding HIP namespaces
extern const std::map<llvm::StringRef, hipCounter> CUDA_CUB_NAMESPACE_MAP;
// Maps the names of CUDA RTC API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_RTC_TYPE_NAME_MAP;
// Maps the names of CUDA RTC API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_RTC_FUNCTION_MAP;
// Maps the names of CUDA SOLVER API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_SOLVER_TYPE_NAME_MAP;
// Maps the names of CUDA SOLVER API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_SOLVER_FUNCTION_MAP;
// Maps the names of CUDA TENSOR API types to the corresponding HIP types
extern const std::map<llvm::StringRef, hipCounter> CUDA_TENSOR_TYPE_NAME_MAP;
// Maps the names of CUDA TENSOR API functions to the corresponding HIP functions
extern const std::map<llvm::StringRef, hipCounter> CUDA_TENSOR_FUNCTION_MAP;

/**
  * The union of all the above maps, except includes.
  *
  * This should be used rarely, but is still needed to convert macro definitions (which can
  * contain any combination of the above things). AST walkers can usually get away with just
  * looking in the lookup table for the type of element they are processing, however, saving
  * a great deal of time.
  */
const std::map<llvm::StringRef, hipCounter> &CUDA_RENAMES_MAP();

extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DRIVER_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DRIVER_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RUNTIME_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RUNTIME_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_COMPLEX_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_COMPLEX_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_BLAS_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_BLAS_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RAND_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RAND_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DNN_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_FFT_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSE_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSE_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_CAFFE2_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_CAFFE2_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DEVICE_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_DEVICE_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_CUB_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_CUB_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RTC_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_RTC_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_SOLVER_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_SOLVER_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_TENSOR_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIversions> CUDA_TENSOR_FUNCTION_VER_MAP;

/**
  * The union of all the above CUDA maps.
  *
  */
const std::map<llvm::StringRef, cudaAPIversions> &CUDA_VERSIONS_MAP();

extern const std::map<llvm::StringRef, hipAPIversions> HIP_DRIVER_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_DRIVER_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_DRIVER_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RUNTIME_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RUNTIME_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RUNTIME_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_COMPLEX_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_COMPLEX_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_BLAS_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_BLAS_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_BLAS_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_BLAS_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RAND_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RAND_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_DNN_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_DNN_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_FFT_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_FFT_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSE_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSE_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIChangedVersions> HIP_SPARSE_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_SPARSE_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_CAFFE2_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_CAFFE2_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_DEVICE_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_DEVICE_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_DEVICE_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_CUB_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_CUB_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RTC_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_RTC_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, cudaAPIChangedVersions> CUDA_RTC_FUNCTION_CHANGED_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_SOLVER_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_SOLVER_FUNCTION_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_TENSOR_TYPE_NAME_VER_MAP;
extern const std::map<llvm::StringRef, hipAPIversions> HIP_TENSOR_FUNCTION_VER_MAP;

/**
  * The union of all the above HIP maps.
  *
  */
const std::map<llvm::StringRef, hipAPIversions> &HIP_VERSIONS_MAP();

extern const std::map<unsigned int, llvm::StringRef> CUDA_DRIVER_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_RUNTIME_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_COMPLEX_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_BLAS_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_RAND_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_DNN_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_FFT_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_SPARSE_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_DEVICE_FUNCTION_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_RTC_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_CUB_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_SOLVER_API_SECTION_MAP;
extern const std::map<unsigned int, llvm::StringRef> CUDA_TENSOR_API_SECTION_MAP;

namespace driver {
  enum CUDA_DRIVER_API_SECTIONS {
    DATA_TYPES = 1,
    ERROR = 2,
    INIT = 3,
    VERSION = 4,
    DEVICE = 5,
    DEVICE_DEPRECATED = 6,
    PRIMARY_CONTEXT = 7,
    CONTEXT = 8,
    CONTEXT_DEPRECATED = 9,
    MODULE = 10,
    MODULE_DEPRECATED = 11,
    LIBRARY = 12,
    MEMORY = 13,
    VIRTUAL_MEMORY = 14,
    ORDERED_MEMORY = 15,
    MULTICAST = 16,
    UNIFIED = 17,
    STREAM = 18,
    EVENT = 19,
    EXTERNAL_RES = 20,
    STREAM_MEMORY = 21,
    EXECUTION = 22,
    EXECUTION_DEPRECATED = 23,
    GRAPH = 24,
    OCCUPANCY = 25,
    TEXTURE_DEPRECATED = 26,
    SURFACE_DEPRECATED = 27,
    TEXTURE = 28,
    SURFACE = 29,
    TENSOR = 30,
    PEER = 31,
    GRAPHICS = 32,
    DRIVER_ENTRY_POINT = 33,
    COREDUMP = 34,
    GREEN_CONTEXT = 35,
    PROFILER_DEPRECATED = 36,
    PROFILER = 37,
    OPENGL = 38,
    D3D9 = 39,
    D3D10 = 40,
    D3D11 = 41,
    VDPAU = 42,
    EGL = 43,
  };
}

namespace runtime {
  enum CUDA_RUNTIME_API_SECTIONS {
    DEVICE = 1,
    DEVICE_DEPRECATED = 2,
    THREAD_DEPRECATED = 3,
    ERROR = 4,
    STREAM = 5,
    EVENT = 6,
    EXTERNAL_RES = 7,
    EXECUTION = 8,
    EXECUTION_DEPRECATED = 9,
    OCCUPANCY = 10,
    MEMORY = 11,
    MEMORY_DEPRECATED = 12,
    ORDERED_MEMORY = 13,
    UNIFIED = 14,
    PEER = 15,
    OPENGL = 16,
    OPENGL_DEPRECATED = 17,
    D3D9 = 18,
    D3D9_DEPRECATED = 19,
    D3D10 = 20,
    D3D10_DEPRECATED = 21,
    D3D11 = 22,
    D3D11_DEPRECATED = 23,
    VDPAU = 24,
    EGL = 25,
    GRAPHICS = 26,
    TEXTURE = 27,
    SURFACE = 28,
    VERSION = 29,
    GRAPH = 30,
    DRIVER_ENTRY_POINT = 31,
    CPP = 32,
    DRIVER_INTERACT = 33,
    PROFILER = 34,
    DATA_TYPES = 35,
    EXECUTION_REMOVED = 36,
    TEXTURE_REMOVED = 37,
    SURFACE_REMOVED = 38,
    PROFILER_REMOVED = 39,
  };
}

namespace blas {
  enum BLAS_API_SECTIONS {
    BLAS_DATA_TYPES = 1,
    CUDA_DATA_TYPES = 2,
    BLAS_LT_DATA_TYPES = 3,
    BLAS_HELPER = 4,
    BLAS_LEVEL_1 = 5,
    BLAS_LEVEL_2 = 6,
    BLAS_LEVEL_3 = 7,
    BLAS_EXT = 8,
    BLAS_LT = 9,
  };
}
