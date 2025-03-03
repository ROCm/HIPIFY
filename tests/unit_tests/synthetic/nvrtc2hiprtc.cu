// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hiprtc.h"
#include "nvrtc.h"
// CHECK-NOT: #include "hip/hiprtc.h"
// CHECK-NOT: #include "nvrtc.h"

int main() {
  printf("25. CUDA nvRTC API to HIP hipRTC API synthetic test\n");

  int numOptions = 0;
  int numHeaders = 0;
  const char* pOptions = nullptr;
  const char* pHeadedrs = nullptr;
  const char* pIncludeNames = nullptr;
  char* pchSrc = nullptr;
  char* pchName = nullptr;

  // CHECK: hiprtcProgram rtcProgram = nullptr;
  nvrtcProgram rtcProgram = nullptr;

  // CHECK: hiprtcResult rtcResult;
  nvrtcResult rtcResult;

  // CUDA: nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options);
  // HIP: hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char** options);
  // CHECK: rtcResult = hiprtcCompileProgram(rtcProgram, numOptions, const_cast<const char**>(&pOptions));
  rtcResult = nvrtcCompileProgram(rtcProgram, numOptions, &pOptions);

  // CUDA: nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames);
  // HIP: hiprtcResult hiprtcCreateProgram(hiprtcProgram* prog, const char* src, const char* name, int numHeaders, const char** headers, const char** includeNames);
  // CHECK: rtcResult = hiprtcCreateProgram(&rtcProgram, pchSrc, pchName, numHeaders, const_cast<const char**>(&pHeadedrs), const_cast<const char**>(&pIncludeNames));
  rtcResult = nvrtcCreateProgram(&rtcProgram, pchSrc, pchName, numHeaders, &pHeadedrs, &pIncludeNames);

  return 0;
}
