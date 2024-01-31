// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
  printf("07. CUDA Runtime API Typedefs synthetic test\n");

  // CHECK: hipStreamCallback_t StreamCallback_t;
  // CHECK-NEXT: hipSurfaceObject_t SurfaceObject_t;
  // CHECK-NEXT: hipTextureObject_t TextureObject_t;
  cudaStreamCallback_t StreamCallback_t;
  cudaSurfaceObject_t SurfaceObject_t;
  cudaTextureObject_t TextureObject_t;

  // CHECK: hipUUID uuid;
  cudaUUID_t uuid;

#if CUDA_VERSION >= 10000
  // CHECK: hipHostFn_t HostFn_t;
  cudaHostFn_t HostFn_t;
#endif

  return 0;
}
