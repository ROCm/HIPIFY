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

#include "Statistics.h"
#include <assert.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "ArgParse.h"

const char *counterNames[NUM_CONV_TYPES] = {
  "error", // CONV_ERROR
  "init", // CONV_INIT
  "version", // CONV_VERSION
  "device", // CONV_DEVICE
  "context", // CONV_CONTEXT
  "module", // CONV_MODULE
  "library", // CONV_LIBRARY
  "memory", // CONV_MEMORY
  "virtual_memory", // CONV_VIRTUAL_MEMORY
  "ordered_memory", // CONV_ORDERED_MEMORY
  "multicast", // CONV_MULTICAST
  "unified", // CONV_UNIFIED
  "stream", // CONV_STREAM
  "event", // CONV_EVENT
  "external_resource", // CONV_EXTERNAL_RES
  "stream_memory", // CONV_STREAM_MEMORY
  "execution", // CONV_EXECUTION
  "graph", // CONV_GRAPH
  "occupancy", // CONV_OCCUPANCY
  "texture", // CONV_TEXTURE
  "surface", // CONV_SURFACE
  "tensor", // CONV_TENSOR
  "peer", // CONV_PEER
  "graphics", // CONV_GRAPHICS
  "driver_entry_point", // CONV_DRIVER_ENTRY_POINT
  "cpp", // CONV_CPP
  "coredump", // CONV_COREDUMP
  "driver_interact", // CONV_DRIVER_INTERACT
  "profiler", // CONV_PROFILER
  "openGL", // CONV_OPENGL
  "D3D9", // CONV_D3D9
  "D3D10", // CONV_D3D10
  "D3D11", // CONV_D3D11
  "VDPAU", // CONV_VDPAU
  "EGL", // CONV_EGL
  "thread", // CONV_THREAD
  "complex", // CONV_COMPLEX
  "library", // CONV_LIB_FUNC
  "device_library", // CONV_LIB_DEVICE_FUNC
  "device_function", // CONV_DEVICE_FUNC
  "device_type", // CONV_DEVICE_TYPE
  "include", // CONV_INCLUDE
  "include_cuda_main_header", // CONV_INCLUDE_CUDA_MAIN_H
  "include_cuda_main_header_v2", // CONV_INCLUDE_CUDA_MAIN_V2_H
  "type", // CONV_TYPE
  "literal", // CONV_LITERAL
  "numeric_literal", // CONV_NUMERIC_LITERAL
  "define", // CONV_DEFINE
  "extern_shared", // CONV_EXTERN_SHARED
  "kernel_launch" // CONV_KERNEL_LAUNCH
};

const char *counterTypes[NUM_CONV_TYPES] = {
  "CONV_ERROR",
  "CONV_INIT",
  "CONV_VERSION",
  "CONV_DEVICE",
  "CONV_CONTEXT",
  "CONV_MODULE",
  "CONV_LIBRARY",
  "CONV_MEMORY",
  "CONV_VIRTUAL_MEMORY",
  "CONV_ORDERED_MEMORY",
  "CONV_MULTICAST",
  "CONV_UNIFIED",
  "CONV_STREAM",
  "CONV_EVENT",
  "CONV_EXTERNAL_RES",
  "CONV_STREAM_MEMORY",
  "CONV_EXECUTION",
  "CONV_GRAPH",
  "CONV_OCCUPANCY",
  "CONV_TEXTURE",
  "CONV_SURFACE",
  "CONV_TENSOR",
  "CONV_PEER",
  "CONV_GRAPHICS",
  "CONV_DRIVER_ENTRY_POINT",
  "CONV_CPP",
  "CONV_COREDUMP",
  "CONV_DRIVER_INTERACT",
  "CONV_PROFILER",
  "CONV_OPENGL",
  "CONV_D3D9",
  "CONV_D3D10",
  "CONV_D3D11",
  "CONV_VDPAU",
  "CONV_EGL",
  "CONV_THREAD",
  "CONV_COMPLEX",
  "CONV_LIB_FUNC",
  "CONV_LIB_DEVICE_FUNC",
  "CONV_DEVICE_FUNC",
  "CONV_DEVICE_TYPE",
  "CONV_INCLUDE",
  "CONV_INCLUDE_CUDA_MAIN_H",
  "CONV_INCLUDE_CUDA_MAIN_V2_H",
  "CONV_TYPE",
  "CONV_LITERAL",
  "CONV_NUMERIC_LITERAL",
  "CONV_DEFINE",
  "CONV_EXTERN_SHARED",
  "CONV_KERNEL_LAUNCH"
};

const char *apiNames[NUM_API_TYPES] = {
  "CUDA Driver API",
  "CUDA RT API",
  "cuComplex API",
  "cuBLAS API",
  "cuRAND API",
  "cuDNN API",
  "cuFFT API",
  "cuSPARSE API",
  "cuSOLVER API",
  "CUB API",
  "CAFFE2 API",
  "RTC API"
};

const char *apiTypes[NUM_API_TYPES] = {
  "API_DRIVER",
  "API_RUNTIME",
  "API_COMPLEX",
  "API_BLAS",
  "API_RAND",
  "API_DNN",
  "API_FFT",
  "API_CUB",
  "API_SPARSE",
  "API_SOLVER",
  "API_CAFFE2",
  "API_RTC"
};

namespace {

template<typename ST, typename ST2>
void conditionalPrint(ST *stream1,
                      ST2 *stream2,
                      const std::string &s1,
                      const std::string &s2) {
  if (stream1) *stream1 << s1;
  if (stream2) *stream2 << s2;
}

// Print a named stat value to both the terminal and the CSV file.
template<typename T>
void printStat(std::ostream *csv, llvm::raw_ostream *printOut, const std::string &name, T value) {
  if (printOut)
    *printOut << "  " << name << ": " << value << "\n";
  if (csv)
    *csv << name << ";" << value << "\n";
}

} // Anonymous namespace

void StatCounter::incrementCounter(const hipCounter &counter, const std::string &name) {
  counters[name]++;
  apiCounters[(int) counter.apiType]++;
  convTypeCounters[(int) counter.type]++;
}

void StatCounter::add(const StatCounter &other) {
  for (const auto &p : other.counters)
    counters[p.first] += p.second;
  for (int i = 0; i < NUM_API_TYPES; ++i)
    apiCounters[i] += other.apiCounters[i];
  for (int i = 0; i < NUM_CONV_TYPES; ++i)
    convTypeCounters[i] += other.convTypeCounters[i];
}

int StatCounter::getConvSum() {
  int acc = 0;
  for (const int &i : convTypeCounters)
    acc += i;
  return acc;
}

void StatCounter::print(std::ostream *csv, llvm::raw_ostream *printOut, const std::string &prefix) {
  for (int i = 0; i < NUM_CONV_TYPES; ++i) {
    if (convTypeCounters[i] > 0) {
      conditionalPrint(csv, printOut, "\nCUDA ref type;Count\n", "[HIPIFY] info: " + prefix + " refs by type:\n");
      break;
    }
  }
  for (int i = 0; i < NUM_CONV_TYPES; ++i) {
    if (convTypeCounters[i] > 0) {
      printStat(csv, printOut, counterNames[i], convTypeCounters[i]);
    }
  }
  for (int i = 0; i < NUM_API_TYPES; ++i) {
    if (apiCounters[i] > 0) {
      conditionalPrint(csv, printOut, "\nCUDA API;Count\n", "[HIPIFY] info: " + prefix + " refs by API:\n");
      break;
    }
  }
  for (int i = 0; i < NUM_API_TYPES; ++i) {
    if (apiCounters[i] > 0) {
      printStat(csv, printOut, apiNames[i], apiCounters[i]);
    }
  }
  if (counters.size() > 0) {
    conditionalPrint(csv, printOut, "\nCUDA ref name;Count\n", "[HIPIFY] info: " + prefix + " refs by names:\n");
    for (const auto &it : counters) {
      printStat(csv, printOut, it.first, it.second);
    }
  }
}

Statistics::Statistics(const std::string &name): fileName(name) {
  // Compute the total bytes/lines in the input file.
  std::ifstream src_file(name, std::ios::binary | std::ios::ate);
  if (src_file.good()) {
    src_file.clear();
    src_file.seekg(0);
    totalLines = (unsigned)std::count(std::istreambuf_iterator<char>(src_file), std::istreambuf_iterator<char>(), '\n');
    totalBytes = (unsigned)src_file.tellg();
    if (totalBytes < 0) {
      totalBytes = 0;
    }
  }
  startTime = chr::steady_clock::now();
}

///////// Counter update routines //////////

void Statistics::incrementCounter(const hipCounter &counter, const std::string &name) {
  if (Statistics::isUnsupported(counter)) {
    unsupported.incrementCounter(counter, name);
  } else {
    supported.incrementCounter(counter, name);
  }
}

void Statistics::add(const Statistics &other) {
  supported.add(other.supported);
  unsupported.add(other.unsupported);
  touchedBytes += other.touchedBytes;
  totalBytes += other.totalBytes;
  touchedLines += other.touchedLines;
  totalLines += other.totalLines;
  if (other.hasErrors && !hasErrors) hasErrors = true;
  if (startTime > other.startTime)   startTime = other.startTime;
}

void Statistics::lineTouched(unsigned int lineNumber) {
  touchedLinesSet.insert(lineNumber);
  touchedLines = unsigned(touchedLinesSet.size());
}

void Statistics::bytesChanged(unsigned int bytes) {
  touchedBytes += bytes;
}

void Statistics::markCompletion() {
  completionTime = chr::steady_clock::now();
}

///////// Output functions //////////

void Statistics::print(std::ostream *csv, llvm::raw_ostream *printOut, bool skipHeader) {
  if (!skipHeader) {
    std::string str = "file \'" + fileName + "\' statistics:\n";
    conditionalPrint(csv, printOut, "\n" + str, "\n[HIPIFY] info: " + str);
  }
  if (hasErrors || totalBytes == 0 || totalLines == 0) {
    std::string str = "\n  ERROR: Statistics is invalid due to failed hipification.\n\n";
    conditionalPrint(csv, printOut, str, str);
  }
  std::stringstream stream;
  // Total number of (un)supported refs that were converted.
  int supportedSum = supported.getConvSum();
  int unsupportedSum = unsupported.getConvSum();
  int allSum = supportedSum + unsupportedSum;
  printStat(csv, printOut, "CONVERTED refs count", supportedSum);
  printStat(csv, printOut, "UNCONVERTED refs count", unsupportedSum);
  stream << std::fixed << std::setprecision(1) << 100 - (0 == allSum ? 100 : double(unsupportedSum) / double(allSum) * 100);
  printStat(csv, printOut, "CONVERSION %", stream.str());
  stream.str("");
  printStat(csv, printOut, "REPLACED bytes", touchedBytes);
  printStat(csv, printOut, "TOTAL bytes", totalBytes);
  printStat(csv, printOut, "CHANGED lines of code", touchedLines);
  printStat(csv, printOut, "TOTAL lines of code", totalLines);
  stream << std::fixed << std::setprecision(1) << (0 == totalBytes ? 0 : double(touchedBytes) / double(totalBytes) * 100);
  printStat(csv, printOut, "CODE CHANGED (in bytes) %", stream.str());
  stream.str("");
  stream << std::fixed << std::setprecision(1) << (0 == totalBytes ? 0 : double(touchedLines) / double(totalLines) * 100);
  printStat(csv, printOut, "CODE CHANGED (in lines) %", stream.str());
  stream.str("");
  typedef std::chrono::duration<double, std::milli> duration;
  duration elapsed = completionTime - startTime;
  stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
  printStat(csv, printOut, "TIME ELAPSED s", stream.str());
  supported.print(csv, printOut, "CONVERTED");
  unsupported.print(csv, printOut, "UNCONVERTED");
}

void Statistics::printAggregate(std::ostream *csv, llvm::raw_ostream *printOut) {
  Statistics globalStats = getAggregate();
  // A file is considered "converted" if we made any changes to it.
  int convertedFiles = 0;
  for (const auto &p : stats) {
    if (p.second.touchedLines && p.second.totalBytes &&
        p.second.totalLines && !p.second.hasErrors) {
      convertedFiles++;
    }
  }
  globalStats.markCompletion();
  globalStats.print(csv, printOut);
  std::string str = "TOTAL statistics:";
  conditionalPrint(csv, printOut, "\n" + str + "\n", "\n[HIPIFY] info: " + str + "\n");
  printStat(csv, printOut, "CONVERTED files", convertedFiles);
  printStat(csv, printOut, "PROCESSED files", stats.size());
}

//// Static state management ////

Statistics Statistics::getAggregate() {
  Statistics globalStats("GLOBAL");
  for (const auto &p : stats) {
    globalStats.add(p.second);
  }
  return globalStats;
}

Statistics &Statistics::current() {
  assert(Statistics::currentStatistics);
  return *Statistics::currentStatistics;
}

void Statistics::setActive(const std::string &name) {
  stats.emplace(std::make_pair(name, Statistics{name}));
  Statistics::currentStatistics = &stats.at(name);
}

bool Statistics::isToRoc(const hipCounter &counter) {
  return (counter.apiType == API_BLAS || counter.apiType == API_DNN || counter.apiType == API_SPARSE || counter.apiType == API_SOLVER ||
          counter.apiType == API_RUNTIME || counter.apiType == API_COMPLEX) &&
          ((TranslateToRoc && !TranslateToMIOpen && !isRocMiopenOnly(counter)) || TranslateToMIOpen);
}

bool Statistics::isHipExperimental(const hipCounter& counter) {
  return HIP_EXPERIMENTAL == (counter.supportDegree & HIP_EXPERIMENTAL);
}

bool Statistics::isHipUnsupported(const hipCounter &counter) {
  return HIP_UNSUPPORTED == (counter.supportDegree & HIP_UNSUPPORTED) ||
    UNSUPPORTED == (counter.supportDegree & UNSUPPORTED);
}

bool Statistics::isRocUnsupported(const hipCounter &counter) {
  return ROC_UNSUPPORTED == (counter.supportDegree & ROC_UNSUPPORTED) ||
    UNSUPPORTED == (counter.supportDegree & UNSUPPORTED);
}

bool Statistics::isUnsupported(const hipCounter& counter) {
  if (UNSUPPORTED == (counter.supportDegree & UNSUPPORTED)) return true;
  if (Statistics::isToRoc(counter)) return Statistics::isRocUnsupported(counter);
  else return Statistics::isHipUnsupported(counter);
}

bool Statistics::isCudaDeprecated(const hipCounter &counter) {
  return CUDA_DEPRECATED == (counter.supportDegree & CUDA_DEPRECATED) ||
         DEPRECATED == (counter.supportDegree & DEPRECATED);
}

bool Statistics::isHipDeprecated(const hipCounter &counter) {
  return HIP_DEPRECATED == (counter.supportDegree & HIP_DEPRECATED) ||
         DEPRECATED == (counter.supportDegree & DEPRECATED);
}

bool Statistics::isRocDeprecated(const hipCounter &counter) {
  return ROC_DEPRECATED == (counter.supportDegree & ROC_DEPRECATED) ||
         DEPRECATED == (counter.supportDegree & DEPRECATED);
}

bool Statistics::isDeprecated(const hipCounter &counter) {
  return DEPRECATED == (counter.supportDegree & DEPRECATED) || (
         CUDA_DEPRECATED == (counter.supportDegree & CUDA_DEPRECATED) &&
         HIP_DEPRECATED == (counter.supportDegree & HIP_DEPRECATED) &&
         ROC_DEPRECATED == (counter.supportDegree & ROC_DEPRECATED));
}

bool Statistics::isCudaRemoved(const hipCounter &counter) {
  return CUDA_REMOVED == (counter.supportDegree & CUDA_REMOVED) ||
         REMOVED == (counter.supportDegree & REMOVED);
}

bool Statistics::isHipRemoved(const hipCounter &counter) {
  return HIP_REMOVED == (counter.supportDegree & HIP_REMOVED) ||
         REMOVED == (counter.supportDegree & REMOVED);
}

bool Statistics::isRocRemoved(const hipCounter &counter) {
  return ROC_REMOVED == (counter.supportDegree & ROC_REMOVED) ||
         REMOVED == (counter.supportDegree & REMOVED);
}

bool Statistics::isRemoved(const hipCounter &counter) {
  return REMOVED == (counter.supportDegree & REMOVED) || (
         CUDA_REMOVED == (counter.supportDegree & CUDA_REMOVED) &&
         HIP_REMOVED == (counter.supportDegree & HIP_REMOVED) &&
         ROC_REMOVED == (counter.supportDegree & ROC_REMOVED));
}

bool Statistics::isHipSupportedV2Only(const hipCounter& counter) {
  return HIP_SUPPORTED_V2_ONLY == (counter.supportDegree & HIP_SUPPORTED_V2_ONLY);
}

bool Statistics::isRocMiopenOnly(const hipCounter& counter) {
  return ROC_MIOPEN_ONLY == (counter.supportDegree & ROC_MIOPEN_ONLY);
}

bool Statistics::isCudaOverloaded(const hipCounter& counter) {
  return CUDA_OVERLOADED == (counter.supportDegree & CUDA_OVERLOADED);
}

std::string Statistics::getCudaVersion(const cudaVersions& ver) {
  switch (ver) {
    case CUDA_0:
    default:       return "";
    case CUDA_10:  return "1.0";
    case CUDA_11:  return "1.1";
    case CUDA_20:  return "2.0";
    case CUDA_21:  return "2.1";
    case CUDA_22:  return "2.2";
    case CUDA_23:  return "2.3";
    case CUDA_30:  return "3.0";
    case CUDA_31:  return "3.1";
    case CUDA_32:  return "3.2";
    case CUDA_40:  return "4.0";
    case CUDA_41:  return "4.1";
    case CUDA_42:  return "4.2";
    case CUDA_50:  return "5.0";
    case CUDA_55:  return "5.5";
    case CUDA_60:  return "6.0";
    case CUDA_65:  return "6.5";
    case CUDA_70:  return "7.0";
    case CUDA_75:  return "7.5";
    case CUDA_80:  return "8.0";
    case CUDA_90:  return "9.0";
    case CUDA_91:  return "9.1";
    case CUDA_92:  return "9.2";
    case CUDA_100: return "10.0";
    case CUDA_101: return "10.1";
    case CUDA_102: return "10.2";
    case CUDA_110: return "11.0";
    case CUDA_111: return "11.1";
    case CUDA_112: return "11.2";
    case CUDA_113: return "11.3";
    case CUDA_114: return "11.4";
    case CUDA_115: return "11.5";
    case CUDA_116: return "11.6";
    case CUDA_117: return "11.7";
    case CUDA_118: return "11.8";
    case CUDA_120: return "12.0";
    case CUDA_121: return "12.1";
    case CUDA_122: return "12.2";
    case CUDA_123: return "12.3";
    case CUDNN_10: return "1.0.0";
    case CUDNN_20: return "2.0.0";
    case CUDNN_30: return "3.0.0";
    case CUDNN_40: return "4.0.0";
    case CUDNN_50: return "5.0.0";
    case CUDNN_51: return "5.1.0";
    case CUDNN_60: return "6.0.0";
    case CUDNN_704: return "7.0.4";
    case CUDNN_705: return "7.0.5";
    case CUDNN_712: return "7.1.2";
    case CUDNN_713: return "7.1.3";
    case CUDNN_714: return "7.1.4";
    case CUDNN_721: return "7.2.1";
    case CUDNN_730: return "7.3.0";
    case CUDNN_731: return "7.3.1";
    case CUDNN_741: return "7.4.1";
    case CUDNN_742: return "7.4.2";
    case CUDNN_750: return "7.5.0";
    case CUDNN_751: return "7.5.1";
    case CUDNN_760: return "7.6.0";
    case CUDNN_761: return "7.6.1";
    case CUDNN_762: return "7.6.2";
    case CUDNN_763: return "7.6.3";
    case CUDNN_764: return "7.6.4";
    case CUDNN_765: return "7.6.5";
    case CUDNN_801: return "8.0.1";
    case CUDNN_802: return "8.0.2";
    case CUDNN_803: return "8.0.3";
    case CUDNN_804: return "8.0.4";
    case CUDNN_805: return "8.0.5";
    case CUDNN_810: return "8.1.0";
    case CUDNN_811: return "8.1.1";
    case CUDNN_820: return "8.2.0";
    case CUDNN_830: return "8.3.0";
    case CUDNN_840: return "8.4.0";
    case CUDNN_850: return "8.5.0";
    case CUDNN_860: return "8.6.0";
    case CUDNN_870: return "8.7.0";
    case CUDNN_880: return "8.8.0";
    case CUDNN_881: return "8.8.1";
    case CUDNN_890: return "8.9.0";
    case CUDNN_891: return "8.9.1";
    case CUDNN_892: return "8.9.2";
    case CUDNN_893: return "8.9.3";
    case CUDNN_894: return "8.9.4";
    case CUDNN_895: return "8.9.5";
    case CUDNN_896: return "8.9.6";
    case CUDNN_897: return "8.9.7";
  }
  return "";
}

std::string Statistics::getHipVersion(const hipVersions& ver) {
  switch (ver) {
    case HIP_0:
    default:       return "";
    case HIP_1050: return "1.5.0";
    case HIP_1051: return "1.5.1";
    case HIP_1052: return "1.5.2";
    case HIP_1060: return "1.6.0";
    case HIP_1061: return "1.6.1";
    case HIP_1064: return "1.6.4";
    case HIP_1070: return "1.7.0";
    case HIP_1071: return "1.7.1";
    case HIP_1080: return "1.8.0";
    case HIP_1082: return "1.8.2";
    case HIP_1090: return "1.9.0";
    case HIP_1091: return "1.9.1";
    case HIP_1092: return "1.9.2";
    case HIP_2000: return "2.0.0";
    case HIP_2010: return "2.1.0";
    case HIP_2020: return "2.2.0";
    case HIP_2030: return "2.3.0";
    case HIP_2040: return "2.4.0";
    case HIP_2050: return "2.5.0";
    case HIP_2060: return "2.6.0";
    case HIP_2070: return "2.7.0";
    case HIP_2072: return "2.7.2";
    case HIP_2080: return "2.8.0";
    case HIP_2090: return "2.9.0";
    case HIP_2100: return "2.10.0";
    case HIP_3000: return "3.0.0";
    case HIP_3010: return "3.1.0";
    case HIP_3011: return "3.1.1";
    case HIP_3020: return "3.2.0";
    case HIP_3021: return "3.2.1";
    case HIP_3022: return "3.2.2";
    case HIP_3030: return "3.3.0";
    case HIP_3040: return "3.4.0";
    case HIP_3050: return "3.5.0";
    case HIP_3051: return "3.5.1";
    case HIP_3060: return "3.6.0";
    case HIP_3070: return "3.7.0";
    case HIP_3080: return "3.8.0";
    case HIP_3090: return "3.9.0";
    case HIP_3100: return "3.10.0";
    case HIP_4000: return "4.0.0";
    case HIP_4010: return "4.1.0";
    case HIP_4011: return "4.1.1";
    case HIP_4020: return "4.2.0";
    case HIP_4030: return "4.3.0";
    case HIP_4040: return "4.4.0";
    case HIP_4050: return "4.5.0";
    case HIP_4051: return "4.5.1";
    case HIP_4052: return "4.5.2";
    case HIP_5000: return "5.0.0";
    case HIP_5001: return "5.0.1";
    case HIP_5002: return "5.0.2";
    case HIP_5010: return "5.1.0";
    case HIP_5011: return "5.1.1";
    case HIP_5020: return "5.2.0";
    case HIP_5030: return "5.3.0";
    case HIP_5040: return "5.4.0";
    case HIP_5050: return "5.5.0";
    case HIP_5060: return "5.6.0";
    case HIP_5070: return "5.7.0";
    case HIP_6000: return "6.0.0";
    case HIP_6002: return "6.0.2";
    case HIP_6010: return "6.1.0";
  }
  return "";
}

std::map<std::string, Statistics> Statistics::stats = {};
Statistics *Statistics::currentStatistics = nullptr;
