// RUN: %run_test hipify "%s" "%t" %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

int main() {
  printf("03. CUDA Driver API Typedefs synthetic test\n");

  // CHECK: hipDevice_t device;
  // CHECK-NEXT: hipDevice_t device_v1;
  CUdevice device;
  CUdevice_v1 device_v1;

  // CHECK: hipDeviceptr_t deviceptr;
  // CHECK-NEXT: hipDeviceptr_t deviceptr_v1;
  // CHECK-NEXT: hipDeviceptr_t deviceptr_v2;
  CUdeviceptr deviceptr;
  CUdeviceptr_v1 deviceptr_v1;
  CUdeviceptr_v2 deviceptr_v2;

  // CHECK: hipHostFn_t hostFn;
  CUhostFn hostFn;

  // CHECK: hipStreamCallback_t streamCallback;
  CUstreamCallback streamCallback;

  // CHECK: hipSurfaceObject_t surfObject;
  // CHECK-NEXT: hipSurfaceObject_t surfObject_v1;
  CUsurfObject surfObject;
  CUsurfObject_v1 surfObject_v1;

  // CHECK: hipTextureObject_t texObject;
  // CHECK-NEXT: hipTextureObject_t texObject_v1;
  CUtexObject texObject;
  CUtexObject_v1 texObject_v1;

  return 0;
}
