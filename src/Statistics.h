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

#include <chrono>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <list>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

namespace chr = std::chrono;

enum ConvTypes {
  //  driver::ERROR
  // runtime::ERROR
  CONV_ERROR = 0,
  //  driver::INIT
  CONV_INIT,
  //  driver::VERSION
  // runtime::VERSION
  CONV_VERSION,
  //  driver::DEVICE
  //  driver::DEVICE_DEPRECATED
  // runtime::DEVICE
  CONV_DEVICE,
  //  driver::PRIMARY_CONTEXT
  //  driver::CONTEXT
  //  driver::CONTEXT_DEPRECATED
  CONV_CONTEXT,
  //  driver::MODULE
  //  driver::MODULE_DEPRECATED
  CONV_MODULE,
  //  driver::LIBRARY
  CONV_LIBRARY,
  //  driver::MEMORY
  // runtime::MEMORY
  // runtime::MEMORY_DEPRECATED
  CONV_MEMORY,
  //  driver::VIRTUAL_MEMORY
  CONV_VIRTUAL_MEMORY,
  //  driver::ORDERED_MEMORY
  // runtime::ORDERED_MEMORY
  CONV_ORDERED_MEMORY,
  //  driver::MULTICAST
  CONV_MULTICAST,
  //  driver::UNIFIED
  // runtime::UNIFIED
  CONV_UNIFIED,
  //  driver::STREAM
  // runtime::STREAM
  CONV_STREAM,
  //  driver::EVENT
  // runtime::EVENT
  CONV_EVENT,
  //  driver::EXTERNAL_RES
  // runtime::EXTERNAL_RES
  CONV_EXTERNAL_RES,
  //  driver::STREAM_MEMORY
  CONV_STREAM_MEMORY,
  //  driver::EXECUTION
  //  driver::EXECUTION_DEPRECATED
  // runtime::EXECUTION
  // runtime::EXECUTION_REMOVED
  CONV_EXECUTION,
  //  driver::GRAPH
  // runtime::GRAPH
  CONV_GRAPH,
  //  driver::OCCUPANCY
  // runtime::OCCUPANCY
  CONV_OCCUPANCY,
  //  driver::TEXTURE_DEPRECATED
  //  driver::TEXTURE
  // runtime::TEXTURE
  // runtime::TEXTURE_REMOVED
  CONV_TEXTURE,
  //  driver::SURFACE_DEPRECATED
  //  driver::SURFACE
  // runtime::SURFACE
  // runtime::SURFACE_REMOVED
  CONV_SURFACE,
  //  driver::TENSOR
  CONV_TENSOR,
  //  driver::PEER
  // runtime::PEER
  CONV_PEER,
  //  driver::GRAPHICS
  // runtime::GRAPHICS
  CONV_GRAPHICS,
  //  driver::DRIVER_ENTRY_POINT
  // runtime::DRIVER_ENTRY_POINT
  CONV_DRIVER_ENTRY_POINT,
  // runtime::CPP
  CONV_CPP,
  //  driver::COREDUMP
  CONV_COREDUMP,
  // runtime::DRIVER_INTERACT
  CONV_DRIVER_INTERACT,
  //  driver::PROFILER_DEPRECATED
  //  driver::PROFILER
  // runtime::PROFILER
  // runtime::PROFILER_REMOVED
  CONV_PROFILER,
  //  driver::OPENGL
  // runtime::OPENGL
  // runtime::OPENGL_DEPRECATED
  CONV_OPENGL,
  //  driver::D3D9
  // runtime::D3D9
  // runtime::D3D9_DEPRECATED
  CONV_D3D9,
  //  driver::D3D10
  // runtime::D3D10
  // runtime::D3D10_DEPRECATED
  CONV_D3D10,
  //  driver::D3D11
  // runtime::D3D11
  // runtime::D3D11_DEPRECATED
  CONV_D3D11,
  //  driver::VDPAU
  // runtime::VDPAU
  CONV_VDPAU,
  //  driver::EGL
  // runtime::EGL
  CONV_EGL,
  // runtime::THREAD_DEPRECATED
  CONV_THREAD,
  CONV_COMPLEX,
  CONV_LIB_FUNC,
  CONV_LIB_DEVICE_FUNC,
  CONV_DEVICE_FUNC,
  CONV_DEVICE_TYPE,
  CONV_INCLUDE,
  CONV_INCLUDE_CUDA_MAIN_H,
  CONV_INCLUDE_CUDA_MAIN_V2_H,
  CONV_TYPE,
  CONV_LITERAL,
  CONV_NUMERIC_LITERAL,
  CONV_DEFINE,
  CONV_EXTERN_SHARED,
  CONV_KERNEL_LAUNCH,
  CONV_LAST
};
constexpr int NUM_CONV_TYPES = (int) ConvTypes::CONV_LAST;

enum ApiTypes {
  API_DRIVER = 0,
  API_RUNTIME,
  API_COMPLEX,
  API_BLAS,
  API_RAND,
  API_DNN,
  API_FFT,
  API_SPARSE,
  API_SOLVER,
  API_CUB,
  API_CAFFE2,
  API_RTC,
  API_LAST
};
constexpr int NUM_API_TYPES = (int) ApiTypes::API_LAST;

enum SupportDegree {
  FULL = 0x0,
  HIP_UNSUPPORTED = 0x1,
  ROC_UNSUPPORTED = 0x2,
  UNSUPPORTED = 0x4,
  CUDA_DEPRECATED = 0x8,
  HIP_DEPRECATED = 0x10,
  ROC_DEPRECATED = 0x20,
  DEPRECATED = 0x40,
  CUDA_REMOVED = 0x80,
  HIP_REMOVED = 0x100,
  ROC_REMOVED = 0x200,
  REMOVED = 0x400,
  HIP_EXPERIMENTAL = 0x800,
  HIP_SUPPORTED_V2_ONLY = 0x1000,
  ROC_MIOPEN_ONLY = 0x2000,
  CUDA_OVERLOADED = 0x4000
};

enum cudaVersions {
  CUDA_0 = 0, // Unknown version
  CUDA_10 = 1000,
  CUDA_11 = 1010,
  CUDA_20 = 2000,
  CUDA_21 = 2010,
  CUDA_22 = 2020,
  CUDA_23 = 2030,
  CUDA_30 = 3000,
  CUDA_31 = 3010,
  CUDA_32 = 3020,
  CUDA_40 = 4000,
  CUDA_41 = 4010,
  CUDA_42 = 4020,
  CUDA_50 = 5000,
  CUDA_55 = 5050,
  CUDA_60 = 6000,
  CUDA_65 = 6050,
  CUDA_70 = 7000,
  CUDA_75 = 7050,
  CUDA_80 = 8000,
  CUDA_90 = 9000,
  CUDA_91 = 9010,
  CUDA_92 = 9020,
  CUDA_100 = 10000,
  CUDA_101 = 10010,
  CUDA_102 = 10020,
  CUDA_110 = 11000,
  CUDA_111 = 11010,
  CUDA_112 = 11020,
  CUDA_113 = 11030,
  CUDA_114 = 11040,
  CUDA_115 = 11050,
  CUDA_116 = 11060,
  CUDA_117 = 11070,
  CUDA_118 = 11080,
  CUDA_120 = 12000,
  CUDA_121 = 12010,
  CUDA_122 = 12020,
  CUDA_123 = 12030,
  CUDA_LATEST = CUDA_123,
  CUDNN_10 = 100,
  CUDNN_20 = 200,
  CUDNN_30 = 300,
  CUDNN_40 = 400,
  CUDNN_50 = 500,
  CUDNN_51 = 510,
  CUDNN_60 = 600,
  CUDNN_704 = 704,
  CUDNN_705 = 705,
  CUDNN_712 = 712,
  CUDNN_713 = 713,
  CUDNN_714 = 714,
  CUDNN_721 = 721,
  CUDNN_730 = 730,
  CUDNN_731 = 731,
  CUDNN_741 = 741,
  CUDNN_742 = 742,
  CUDNN_750 = 750,
  CUDNN_751 = 751,
  CUDNN_760 = 760,
  CUDNN_761 = 761,
  CUDNN_762 = 762,
  CUDNN_763 = 763,
  CUDNN_764 = 764,
  CUDNN_765 = 765,
  CUDNN_801 = 801,
  CUDNN_802 = 802,
  CUDNN_803 = 803,
  CUDNN_804 = 804,
  CUDNN_805 = 805,
  CUDNN_810 = 810,
  CUDNN_811 = 811,
  CUDNN_820 = 820,
  CUDNN_830 = 830,
  CUDNN_840 = 840,
  CUDNN_850 = 850,
  CUDNN_860 = 860,
  CUDNN_870 = 870,
  CUDNN_880 = 880,
  CUDNN_881 = 881,
  CUDNN_890 = 890,
  CUDNN_891 = 891,
  CUDNN_892 = 892,
  CUDNN_893 = 893,
  CUDNN_894 = 894,
  CUDNN_895 = 895,
  CUDNN_896 = 896,
  CUDNN_897 = 897,
  CUDNN_LATEST = CUDNN_897,
};

enum hipVersions {
  HIP_0 = 0, // Unknown version
  HIP_1050 = 1050,
  HIP_1051 = 1051,
  HIP_1052 = 1052,
  HIP_1060 = 1060,
  HIP_1061 = 1061,
  HIP_1064 = 1064,
  HIP_1070 = 1070,
  HIP_1071 = 1071,
  HIP_1080 = 1080,
  HIP_1082 = 1082,
  HIP_1090 = 1090,
  HIP_1091 = 1091,
  HIP_1092 = 1092,
  HIP_2000 = 2000,
  HIP_2010 = 2010,
  HIP_2020 = 2020,
  HIP_2030 = 2030,
  HIP_2040 = 2040,
  HIP_2050 = 2050,
  HIP_2060 = 2060,
  HIP_2070 = 2070,
  HIP_2072 = 2072,
  HIP_2080 = 2080,
  HIP_2090 = 2090,
  HIP_2100 = 2100,
  HIP_3000 = 3000,
  HIP_3010 = 3010,
  HIP_3011 = 3011,
  HIP_3020 = 3020,
  HIP_3021 = 3021,
  HIP_3022 = 3022,
  HIP_3030 = 3030,
  HIP_3040 = 3040,
  HIP_3050 = 3050,
  HIP_3051 = 3051,
  HIP_3060 = 3060,
  HIP_3070 = 3070,
  HIP_3080 = 3080,
  HIP_3090 = 3090,
  HIP_3100 = 3100,
  HIP_4000 = 4000,
  HIP_4010 = 4010,
  HIP_4011 = 4011,
  HIP_4020 = 4020,
  HIP_4030 = 4030,
  HIP_4040 = 4040,
  HIP_4050 = 4050,
  HIP_4051 = 4051,
  HIP_4052 = 4052,
  HIP_5000 = 5000,
  HIP_5001 = 5001,
  HIP_5002 = 5002,
  HIP_5010 = 5010,
  HIP_5011 = 5011,
  HIP_5020 = 5020,
  HIP_5030 = 5030,
  HIP_5040 = 5040,
  HIP_5050 = 5050,
  HIP_5060 = 5060,
  HIP_5070 = 5070,
  HIP_6000 = 6000,
  HIP_6002 = 6002,
  HIP_6010 = 6010,
  HIP_LATEST = HIP_6010,
};

struct cudaAPIversions {
  cudaVersions appeared;
  cudaVersions deprecated;
  cudaVersions removed;
};

struct hipAPIversions {
  hipVersions appeared;
  hipVersions deprecated;
  hipVersions removed;
  hipVersions experimental = HIP_0;
};

typedef std::list<hipVersions> hipAPIChangedVersions;
typedef std::list<cudaVersions> cudaAPIChangedVersions;

// The names of various fields in in the statistics reports.
extern const char *counterNames[NUM_CONV_TYPES];
extern const char *counterTypes[NUM_CONV_TYPES];
extern const char *apiNames[NUM_API_TYPES];
extern const char *apiTypes[NUM_API_TYPES];

struct hipCounter {
  llvm::StringRef hipName;
  llvm::StringRef rocName;
  ConvTypes type;
  ApiTypes apiType;
  unsigned int apiSection;
  unsigned int supportDegree;
};

/**
  * Tracks a set of named counters, as well as counters for each of the type enums defined above.
  */
class StatCounter {
private:
  // Each thing we track is either "supported" or "unsupported"...
  std::map<std::string, int> counters;
  int apiCounters[NUM_API_TYPES] = {};
  int convTypeCounters[NUM_CONV_TYPES] = {};

public:
  void incrementCounter(const hipCounter &counter, const std::string &name);
  // Add the counters from `other` onto the counters of this object.
  void add(const StatCounter &other);
  int getConvSum();
  void print(std::ostream* csv, llvm::raw_ostream* printOut, const std::string &prefix);
};

/**
  * Tracks the statistics for a single input file.
  */
class Statistics {
  StatCounter supported;
  StatCounter unsupported;
  std::string fileName;
  std::set<int> touchedLinesSet = {};
  unsigned touchedLines = 0;
  unsigned totalLines = 0;
  unsigned touchedBytes = 0;
  unsigned totalBytes = 0;
  chr::steady_clock::time_point startTime;
  chr::steady_clock::time_point completionTime;

public:
  Statistics(const std::string &name);
  void incrementCounter(const hipCounter &counter, const std::string &name);
  // Add the counters from `other` onto the counters of this object.
  void add(const Statistics &other);
  void lineTouched(unsigned int lineNumber);
  void bytesChanged(unsigned int bytes);
  // Set the completion timestamp to now.
  void markCompletion();

public:
  /**
    * Pretty-print the statistics stored in this object.
    *
    * @param csv Pointer to an output stream for the CSV to write. If null, no CSV is written
    * @param printOut Pointer to an output stream to print human-readable textual stats to. If null, no
    *                 such stats are produced.
    */
  void print(std::ostream* csv, llvm::raw_ostream* printOut, bool skipHeader = false);
  // Print aggregated statistics for all registered counters.
  static void printAggregate(std::ostream *csv, llvm::raw_ostream* printOut);
  // The Statistics for each input file.
  static std::map<std::string, Statistics> stats;
  // The Statistics objects for the currently-being-processed input file.
  static Statistics* currentStatistics;
  // Aggregate statistics over all entries in `stats` and return the resulting Statistics object.
  static Statistics getAggregate();
  /**
    * Convenient global entry point for updating the "active" Statistics. Since we operate single-threadedly
    * processing one file at a time, this allows us to simply expose the stats for the current file globally,
    * simplifying things.
    */
  static Statistics &current();
  /**
    * Set the active Statistics object to the named one, creating it if necessary, and write the completion
    * timestamp into the currently active one.
    */
  static void setActive(const std::string &name);
  // Check the counter and option TranslateToRoc whether it should be translated to Roc or not.
  static bool isToRoc(const hipCounter &counter);
  // Check whether the counter is HIP_EXPERIMENTAL or not.
  static bool isHipExperimental(const hipCounter &counter);
  // Check whether the counter is HIP_UNSUPPORTED or not.
  static bool isHipUnsupported(const hipCounter &counter);
  // Check whether the counter is ROC_UNSUPPORTED or not.
  static bool isRocUnsupported(const hipCounter &counter);
  // Check whether the counter is ROC_UNSUPPORTED/HIP_UNSUPPORTED/UNSUPPORTED or not.
  static bool isUnsupported(const hipCounter& counter);
  // Check whether the counter is CUDA_DEPRECATED or not.
  static bool isCudaDeprecated(const hipCounter& counter);
  // Check whether the counter is HIP_DEPRECATED or not.
  static bool isHipDeprecated(const hipCounter& counter);
  // Check whether the counter is ROC_DEPRECATED or not.
  static bool isRocDeprecated(const hipCounter& counter);
  // Check whether the counter is DEPRECATED or not.
  static bool isDeprecated(const hipCounter& counter);
  // Check whether the counter is CUDA_REMOVED or not.
  static bool isCudaRemoved(const hipCounter& counter);
  // Check whether the counter is HIP_REMOVED or not.
  static bool isHipRemoved(const hipCounter& counter);
  // Check whether the counter is ROC_REMOVED or not.
  static bool isRocRemoved(const hipCounter& counter);
  // Check whether the counter is REMOVED or not.
  static bool isRemoved(const hipCounter& counter);
  // Check whether the counter is HIP_SUPPORTED_V2_ONLY or not.
  static bool isHipSupportedV2Only(const hipCounter& counter);
  // Check whether the counter is ROC_MIOPEN_ONLY or not.
  static bool isRocMiopenOnly(const hipCounter& counter);
  // Check whether the counter is CUDA_OVERLOADED or not.
  static bool isCudaOverloaded(const hipCounter& counter);
  // Get string CUDA version.
  static std::string getCudaVersion(const cudaVersions &ver);
  // Get string HIP version.
  static std::string getHipVersion(const hipVersions &ver);
  // Set this flag in case of hipification errors.
  bool hasErrors = false;
};
