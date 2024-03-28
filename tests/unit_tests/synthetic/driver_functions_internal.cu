// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#elif CUDA_VERSION <= 10000
  #include <GL/glew.h>
#endif
#include "cudaGL.h"

int main() {
  printf("13. CUDA Driver API Internal Functions synthetic test\n");

  size_t bytes = 0;

  // CHECK: hipTexRef texref;
  // CHECK-NEXT: HIP_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  // CHECK-NEXT: hipDeviceptr_t deviceptr;
  CUtexref texref;
  CUDA_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  CUdeviceptr deviceptr;

  // CUDA: CUresult CUDAAPI cuTexRefSetAddress2D_v2(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr, size_t Pitch);
  // CHECK: hipError_t result = hipTexRefSetAddress2D(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);
  CUresult result = cuTexRefSetAddress2D_v2(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);

  return 0;
}
