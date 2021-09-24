// RUN: %run_test hipify "%s" "%t" %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {
  printf("07. CUDA Runtime API Typedefs synthetic test\n");

  // CHECK: hipHostFn_t HostFn_t;
  // CHECK-NEXT: hipStreamCallback_t StreamCallback_t;
  // CHECK-NEXT: hipSurfaceObject_t SurfaceObject_t;
  // CHECK-NEXT: hipTextureObject_t TextureObject_t;
  cudaHostFn_t HostFn_t;
  cudaStreamCallback_t StreamCallback_t;
  cudaSurfaceObject_t SurfaceObject_t;
  cudaTextureObject_t TextureObject_t;

  return 0;
}
