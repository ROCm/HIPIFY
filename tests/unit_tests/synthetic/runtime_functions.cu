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
  int deviceId = 0;
  int intVal = 0;
  unsigned int flags = 0;
  void* deviceptr = nullptr;
  void* image = nullptr;
  char* ch = nullptr;

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

#if CUDA_VERSION >= 11000
  // CHECK: hipKernelNodeAttrID kernelNodeAttrID;
  cudaKernelNodeAttrID kernelNodeAttrID;
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue;
  cudaKernelNodeAttrValue kernelNodeAttrValue;
  // CHECK: hipGraphNode_t graphNode;
  cudaGraphNode_t graphNode;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, const union cudaKernelNodeAttrValue* value);
  // HIP: hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cudaGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, union cudaKernelNodeAttrValue* value_out);
  // HIP: hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cudaGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
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

  // CHECK: hipDeviceProp_t DeviceProp;
  cudaDeviceProp DeviceProp;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
  // HIP: hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
  // CHECK: result = hipChooseDevice(&device, &DeviceProp);
  result = cudaChooseDevice(&device, &DeviceProp);

  // CHECK: hipDeviceAttribute_t DeviceAttr;
  cudaDeviceAttr DeviceAttr;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device);
  // HIP: hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
  // CHECK: result = hipDeviceGetAttribute(&device, DeviceAttr, deviceId);
  result = cudaDeviceGetAttribute(&device, DeviceAttr, deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
  // HIP: hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
  // CHECK: result = hipDeviceGetByPCIBusId(&device, ch);
  result = cudaDeviceGetByPCIBusId(&device, ch);

  // CHECK: hipFuncCache_t FuncCache;
  cudaFuncCache FuncCache;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig);
  // HIP: hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
  // CHECK: result = hipDeviceGetCacheConfig(&FuncCache);
  result = cudaDeviceGetCacheConfig(&FuncCache);

  // CHECK: hipLimit_t Limit;
  cudaLimit Limit;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit);
  // HIP: hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
  // CHECK: result = hipDeviceGetLimit(&bytes, Limit);
  result = cudaDeviceGetLimit(&bytes, Limit);

#if CUDA_VERSION >= 8000
  // CHECK: hipDeviceP2PAttr DeviceP2PAttr;
  cudaDeviceP2PAttr DeviceP2PAttr;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // HIP: hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // CHECK: result = hipDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);
  result = cudaDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);
#endif

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device);
  // HIP: hipError_t hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
  // CHECK: result = hipDeviceGetPCIBusId(ch, intVal, device);
  result = cudaDeviceGetPCIBusId(ch, intVal, device);

  // CHECK: hipSharedMemConfig SharedMemConfig;
  cudaSharedMemConfig SharedMemConfig;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig);
  // HIP: hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* pConfig);
  // CHECK: result = hipDeviceGetSharedMemConfig(&SharedMemConfig);
  result = cudaDeviceGetSharedMemConfig(&SharedMemConfig);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
  // HIP: hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
  // CHECK: result = hipDeviceGetStreamPriorityRange(&deviceId, &intVal);
  result = cudaDeviceGetStreamPriorityRange(&deviceId, &intVal);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceReset(void);
  // HIP: hipError_t hipError_t hipDeviceReset(void);
  // CHECK: result = hipDeviceReset();
  result = cudaDeviceReset();

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
  // HIP: hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
  // CHECK: result = hipDeviceSetCacheConfig(FuncCache);
  result = cudaDeviceSetCacheConfig(FuncCache);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
  // HIP: hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);
  // CHECK: result = hipDeviceSetSharedMemConfig(SharedMemConfig);
  result = cudaDeviceSetSharedMemConfig(SharedMemConfig);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void);
  // HIP: hipError_t hipDeviceSynchronize(void);
  // CHECK: result = hipDeviceSynchronize();
  result = cudaDeviceSynchronize();

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device);
  // HIP: hipError_t hipGetDevice(int* deviceId);
  // CHECK: result = hipGetDevice(&deviceId);
  result = cudaGetDevice(&deviceId);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count);
  // HIP: hipError_t hipGetDeviceCount(int* count);
  // CHECK: result = hipGetDeviceCount(&deviceId);
  result = cudaGetDeviceCount(&deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetDeviceFlags( unsigned int *flags );
  // HIP: hipError_t hipGetDeviceFlags(unsigned int* flags);
  // CHECK: result = hipGetDeviceFlags(&flags);
  result = cudaGetDeviceFlags(&flags);

  return 0;
}
