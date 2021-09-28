// RUN: %run_test hipify "%s" "%t" --skip-excluded-preprocessor-conditional-blocks %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
  printf("07. CUDA Runtime API Typedefs synthetic test\n");

#if CUDA_VERSION > 9020
  // CHECK: hipHostFn_t HostFn_t;
  cudaHostFn_t HostFn_t;
#endif

  // CHECK: hipStreamCallback_t StreamCallback_t;
  // CHECK-NEXT: hipSurfaceObject_t SurfaceObject_t;
  // CHECK-NEXT: hipTextureObject_t TextureObject_t;
  cudaStreamCallback_t StreamCallback_t;
  cudaSurfaceObject_t SurfaceObject_t;
  cudaTextureObject_t TextureObject_t;

  return 0;
}
