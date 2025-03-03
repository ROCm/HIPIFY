// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <stdio.h>

int main() {
  printf("03. CUDA Driver API Typedefs synthetic test\n");

  // CHECK: hipDevice_t device;
  CUdevice device;

  // CHECK: hipDeviceptr_t deviceptr;
  // CHECK-NEXT: hipDeviceptr_t deviceptr_v1;
  CUdeviceptr deviceptr;
  CUdeviceptr_v1 deviceptr_v1;

  // CHECK: hipStreamCallback_t streamCallback;
  CUstreamCallback streamCallback;

  // CHECK: hipSurfaceObject_t surfObject;
  CUsurfObject surfObject;

  // CHECK: hipTextureObject_t texObject;
  CUtexObject texObject;

  // CHECK: hipUUID uuid;
  CUuuid uuid;

#if CUDA_VERSION >= 10000
  // CHECK: hipHostFn_t hostFn;
  CUhostFn hostFn;

  // CHECK: HIP_MEMSET_NODE_PARAMS MEMSET_NODE_PARAMS_st;
  // CHECK-NEXT: HIP_MEMSET_NODE_PARAMS MEMSET_NODE_PARAMS;
  CUDA_MEMSET_NODE_PARAMS_st MEMSET_NODE_PARAMS_st;
  CUDA_MEMSET_NODE_PARAMS MEMSET_NODE_PARAMS;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipMemGenericAllocationHandle_t memGenericAllocationHandle_t;
  CUmemGenericAllocationHandle memGenericAllocationHandle_t;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipDevice_t device_v1;
  CUdevice_v1 device_v1;

  // CHECK: hipDeviceptr_t deviceptr_v2;
  CUdeviceptr_v2 deviceptr_v2;

  // CHECK: hipSurfaceObject_t surfObject_v1;
  CUsurfObject_v1 surfObject_v1;

  // CHECK: hipTextureObject_t texObject_v1;
  CUtexObject_v1 texObject_v1;

  // CHECK: hipMemGenericAllocationHandle_t memGenericAllocationHandle_v1;
  CUmemGenericAllocationHandle_v1 memGenericAllocationHandle_v1;

  // CHECK: HIP_MEMSET_NODE_PARAMS MEMSET_NODE_PARAMS_v1;
  CUDA_MEMSET_NODE_PARAMS_v1 MEMSET_NODE_PARAMS_v1;

  // CHECK: hipStreamBatchMemOpParams streamBatchMemOpParams_v1;
  CUstreamBatchMemOpParams_v1 streamBatchMemOpParams_v1;
#endif

  return 0;
}
