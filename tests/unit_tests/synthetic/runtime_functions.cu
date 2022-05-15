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
  void* image = nullptr;

#if defined(_WIN32)
  unsigned long long ull = 0;
#else
  unsigned long ull = 0;
#endif

  // CHECK: hipError_t result = hipSuccess;
  // CHECK-NEXT: hipStream_t stream;
  cudaError result = cudaSuccess;
  cudaStream_t stream;

#if CUDA_VERSION >= 10000
  // CHECK: hipHostFn_t hostFn;
  cudaHostFn_t hostFn;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
  // HIP: hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
  // CHECK: result = hipLaunchHostFunc(stream, hostFn, image);
  result = cudaLaunchHostFunc(stream, hostFn, image);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipStreamCaptureMode streamCaptureMode;
  cudaStreamCaptureMode streamCaptureMode;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode);
  // HIP: hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode);
  // CHECK: result = hipThreadExchangeStreamCaptureMode(&streamCaptureMode);
  result = cudaThreadExchangeStreamCaptureMode(&streamCaptureMode);
#endif

#if CUDA_VERSION >= 11020
  // CHECK: hipMemPoolAttr memPoolAttr;
  cudaMemPoolAttr memPoolAttr;
  // CHECK: hipMemAccessDesc memAccessDesc;
  cudaMemAccessDesc memAccessDesc;
  // CHECK: hipMemAccessFlags memAccessFlags;
  cudaMemAccessFlags memAccessFlags;
  // CHECK: hipMemLocation memLocation;
  cudaMemLocation memLocation;
  // CHECK: hipMemPoolProps memPoolProps;
  cudaMemPoolProps memPoolProps;
  // CHECK: hipMemPool_t memPool_t;
  cudaMemPool_t memPool_t;
  // CHECK: hipMemAllocationHandleType memAllocationHandleType;
  cudaMemAllocationHandleType memAllocationHandleType;

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

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t minBytesToKeep);
  // HIP: hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
  // CHECK: result = hipMemPoolTrimTo(memPool_t, bytes);
  result = cudaMemPoolTrimTo(memPool_t, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );
  // HIP: hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
  // CHECK: result = hipMemPoolSetAttribute(memPool_t, memPoolAttr, image);
  result = cudaMemPoolSetAttribute(memPool_t, memPoolAttr, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value );
  // HIP: hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
  // CHECK: result = hipMemPoolGetAttribute(memPool_t, memPoolAttr, image);
  result = cudaMemPoolGetAttribute(memPool_t, memPoolAttr, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList, size_t count);
  // HIP: hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count);
  // CHECK: result = hipMemPoolSetAccess(memPool_t, &memAccessDesc, bytes);
  result = cudaMemPoolSetAccess(memPool_t, &memAccessDesc, bytes);

  // CUDA: CUresult extern __host__ cudaError_t CUDARTAPI cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool, struct cudaMemLocation *location);
  // HIP: hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location);
  // CHECK: result = hipMemPoolGetAccess(&memAccessFlags, memPool_t, &memLocation);
  result = cudaMemPoolGetAccess(&memAccessFlags, memPool_t, &memLocation);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *poolProps);
  // HIP: hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
  // CHECK: result = hipMemPoolCreate(&memPool_t, &memPoolProps);
  result = cudaMemPoolCreate(&memPool_t, &memPoolProps);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t memPool);
  // HIP: hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
  // CHECK: result = hipMemPoolDestroy(memPool_t);
  result = cudaMemPoolDestroy(memPool_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool, cudaStream_t stream);
  // HIP: hipError_t hipMallocFromPoolAsync(void** dev_ptr, size_t size, hipMemPool_t mem_pool, hipStream_t stream);
  // CHECK: result = hipMallocFromPoolAsync(&deviceptr, bytes, memPool_t, stream);
  result = cudaMallocFromPoolAsync(&deviceptr, bytes, memPool_t, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(void* shareableHandle, cudaMemPool_t memPool, enum cudaMemAllocationHandleType handleType, unsigned int flags);
  // HIP: hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags);
  // CHECK: result = hipMemPoolExportToShareableHandle(image, memPool_t, memAllocationHandleType, ull);
  result = cudaMemPoolExportToShareableHandle(image, memPool_t, memAllocationHandleType, ull);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(cudaMemPool_t* memPool, void* shareableHandle, enum cudaMemAllocationHandleType handleType, unsigned int flags);
  // HIP: hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags);
  // CHECK: result = hipMemPoolImportFromShareableHandle(&memPool_t, image, memAllocationHandleType, ull);
  result = cudaMemPoolImportFromShareableHandle(&memPool_t, image, memAllocationHandleType, ull);

  // CHECK: hipMemPoolPtrExportData memPoolPtrExportData;
  cudaMemPoolPtrExportData memPoolPtrExportData;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolExportPointer(struct cudaMemPoolPtrExportData *exportData, void *ptr);
  // HIP: hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr);
  // CHECK: result = hipMemPoolExportPointer(&memPoolPtrExportData, deviceptr);
  result = cudaMemPoolExportPointer(&memPoolPtrExportData, deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPoolImportPointer(void **ptr, cudaMemPool_t memPool, struct cudaMemPoolPtrExportData *exportData);
  // HIP: hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData* export_data);
  // CHECK: result = hipMemPoolImportPointer(&deviceptr, memPool_t, &memPoolPtrExportData);
  result = cudaMemPoolImportPointer(&deviceptr, memPool_t, &memPoolPtrExportData);
#endif

  return 0;
}
