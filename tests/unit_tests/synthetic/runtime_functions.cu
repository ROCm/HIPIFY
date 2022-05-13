// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#endif
#include "cudaGL.h"

int main() {
  printf("12. CUDA Runtime API Functions synthetic test\n");

  size_t bytes = 0;
  int device = 0;
  void* deviceptr = nullptr;

  // CHECK: hipError_t result = hipSuccess;
  // CHECK: hipStream_t stream;
  cudaError result = cudaSuccess;
  cudaStream_t stream;

#if CUDA_VERSION >= 11020
  // CHECK: hipMemPool_t memPool_t;
  cudaMemPool_t memPool_t;
#endif

#if CUDA_VERSION >= 11020
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device);
  // HIP: hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
  // CHECK: result = hipDeviceGetDefaultMemPool(&memPool_t, device);
  result = cudaDeviceGetDefaultMemPool(&memPool_t, device);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
  // HIP: hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
  // CHECK: result = hipDeviceSetMemPool(device, memPool_t);
  result = cudaDeviceSetMemPool(device, memPool_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device);
  // HIP: hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device);
  // CHECK: result = hipDeviceGetMemPool(&memPool_t, device);
  result = cudaDeviceGetMemPool(&memPool_t, device);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocAsync(void **devPtr, size_t size, cudaStream_t hStream);
  // HIP: hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
  // CHECK: result = hipMallocAsync(&deviceptr, bytes, stream);
  result = cudaMallocAsync(&deviceptr, bytes, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeAsync(void *devPtr, cudaStream_t hStream);
  // HIP: hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream);
  // CHECK: result = hipFreeAsync(deviceptr, stream);
  result = cudaFreeAsync(deviceptr, stream);

#endif

  return 0;
}
