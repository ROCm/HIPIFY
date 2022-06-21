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
  float ms = 0;
  void* deviceptr = nullptr;
  void* image = nullptr;
  void* func = nullptr;
  char* ch = nullptr;
  const char* const_ch = nullptr;
  dim3 gridDim;
  dim3 blockDim;

#if defined(_WIN32)
  unsigned long long ull = 0;
#else
  unsigned long ull = 0;
#endif
  unsigned long long ull_2 = 0;

  // CHECK: hipError_t result = hipSuccess;
  // CHECK-NEXT: hipError_t Error_t;
  // CHECK-NEXT: hipStream_t stream;
  cudaError result = cudaSuccess;
  cudaError_t Error_t;
  cudaStream_t stream;

#if CUDA_VERSION >= 8000
  // CHECK: hipDeviceP2PAttr DeviceP2PAttr;
  cudaDeviceP2PAttr DeviceP2PAttr;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // HIP: hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // CHECK: result = hipDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);
  result = cudaDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipHostFn_t hostFn;
  cudaHostFn_t hostFn;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
  // HIP: hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
  // CHECK: result = hipLaunchHostFunc(stream, hostFn, image);
  result = cudaLaunchHostFunc(stream, hostFn, image);

  // CHECK: hipStreamCaptureMode StreamCaptureMode;
  cudaStreamCaptureMode StreamCaptureMode;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
  // HIP: hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
  // CHECK: result = hipStreamBeginCapture(stream, StreamCaptureMode);
  result = cudaStreamBeginCapture(stream, StreamCaptureMode);

  // CHECK: hipGraph_t Graph_t;
  cudaGraph_t Graph_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
  // HIP: hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
  // CHECK: result = hipStreamEndCapture(stream, &Graph_t);
  result = cudaStreamEndCapture(stream, &Graph_t);

  // CHECK: hipStreamCaptureStatus StreamCaptureStatus;
  cudaStreamCaptureStatus StreamCaptureStatus;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus);
  // HIP: hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
  // CHECK: result = hipStreamIsCapturing(stream, &StreamCaptureStatus);
  result = cudaStreamIsCapturing(stream, &StreamCaptureStatus);

  // CHECK: hipExternalMemory_t ExternalMemory_t;
  cudaExternalMemory_t ExternalMemory_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t extMem);
  // HIP: hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
  // CHECK: result = hipDestroyExternalMemory(ExternalMemory_t);
  result = cudaDestroyExternalMemory(ExternalMemory_t);

  // CHECK: hipExternalSemaphore_t ExternalSemaphore_t;
  cudaExternalSemaphore_t ExternalSemaphore_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem);
  // HIP: hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);
  // CHECK: result = hipDestroyExternalSemaphore(ExternalSemaphore_t);
  result = cudaDestroyExternalSemaphore(ExternalSemaphore_t);

  // CHECK: hipExternalMemoryBufferDesc ExternalMemoryBufferDesc;
  cudaExternalMemoryBufferDesc ExternalMemoryBufferDesc;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaExternalMemoryGetMappedBuffer(void **devPtr, cudaExternalMemory_t extMem, const struct cudaExternalMemoryBufferDesc *bufferDesc);
  // HIP: hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc);
  // CHECK: result = hipExternalMemoryGetMappedBuffer(&deviceptr, ExternalMemory_t, &ExternalMemoryBufferDesc);
  result = cudaExternalMemoryGetMappedBuffer(&deviceptr, ExternalMemory_t, &ExternalMemoryBufferDesc);

  // CHECK: hipExternalMemoryHandleDesc ExternalMemoryHandleDesc;
  cudaExternalMemoryHandleDesc ExternalMemoryHandleDesc;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaImportExternalMemory(cudaExternalMemory_t *extMem_out, const struct cudaExternalMemoryHandleDesc *memHandleDesc);
  // HIP: hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc);
  // CHECK: result = hipImportExternalMemory(&ExternalMemory_t, &ExternalMemoryHandleDesc);
  result = cudaImportExternalMemory(&ExternalMemory_t, &ExternalMemoryHandleDesc);

  // CHECK: hipExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;
  cudaExternalSemaphoreHandleDesc ExternalSemaphoreHandleDesc;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaImportExternalSemaphore(cudaExternalSemaphore_t *extSem_out, const struct cudaExternalSemaphoreHandleDesc *semHandleDesc);
  // HIP: hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out, const hipExternalSemaphoreHandleDesc* semHandleDesc);
  // CHECK: result = hipImportExternalSemaphore(&ExternalSemaphore_t, &ExternalSemaphoreHandleDesc);
  result = cudaImportExternalSemaphore(&ExternalSemaphore_t, &ExternalSemaphoreHandleDesc);

  // CHECK: hipExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;
  cudaExternalSemaphoreSignalParams ExternalSemaphoreSignalParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreSignalParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, hipStream_t stream);
  // CHECK: result = hipSignalExternalSemaphoresAsync(&ExternalSemaphore_t, &ExternalSemaphoreSignalParams, flags, stream);
  result = cudaSignalExternalSemaphoresAsync(&ExternalSemaphore_t, &ExternalSemaphoreSignalParams, flags, stream);

  // CHECK: hipExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;
  cudaExternalSemaphoreWaitParams ExternalSemaphoreWaitParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t *extSemArray, const struct cudaExternalSemaphoreWaitParams *paramsArray, unsigned int numExtSems, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, hipStream_t stream);
  // CHECK: result = hipWaitExternalSemaphoresAsync(&ExternalSemaphore_t, &ExternalSemaphoreWaitParams, flags, stream);
  result = cudaWaitExternalSemaphoresAsync(&ExternalSemaphore_t, &ExternalSemaphoreWaitParams, flags, stream);

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

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId);
  // HIP: hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus, unsigned long long* pId);
  // CHECK: result = hipStreamGetCaptureInfo(stream, &StreamCaptureStatus, &ull_2);
  result = cudaStreamGetCaptureInfo(stream, &StreamCaptureStatus, &ull_2);

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

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
  // HIP: hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId);
  // CHECK: result = hipGetDeviceProperties(&DeviceProp, deviceId);
  result = cudaGetDeviceProperties(&DeviceProp, deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr);
  // HIP: hipError_t hipError_t hipIpcCloseMemHandle(void* devPtr);
  // CHECK: result = hipIpcCloseMemHandle(deviceptr);
  result = cudaIpcCloseMemHandle(deviceptr);

  // CHECK: hipIpcEventHandle_t IpcEventHandle_t;
  cudaIpcEventHandle_t IpcEventHandle_t;

  // CHECK: hipEvent_t Event_t;
  // CHECK-Next: hipEvent_t Event_2;
  cudaEvent_t Event_t;
  cudaEvent_t Event_2;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event);
  // HIP: hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);
  // CHECK: result = hipIpcGetEventHandle(&IpcEventHandle_t, Event_t);
  result = cudaIpcGetEventHandle(&IpcEventHandle_t, Event_t);

  // CHECK: hipIpcMemHandle_t IpcMemHandle_t;
  cudaIpcMemHandle_t IpcMemHandle_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr);
  // HIP: hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
  // CHECK: result = hipIpcGetMemHandle(&IpcMemHandle_t, deviceptr);
  result = cudaIpcGetMemHandle(&IpcMemHandle_t, deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle);
  // HIP: hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);
  // CHECK: result = hipIpcOpenEventHandle(&Event_t, IpcEventHandle_t);
  result = cudaIpcOpenEventHandle(&Event_t, IpcEventHandle_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags);
  // HIP: hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
  // CHECK: result = hipIpcOpenMemHandle(&deviceptr, IpcMemHandle_t, flags);
  result = cudaIpcOpenMemHandle(&deviceptr, IpcMemHandle_t, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);
  // HIP: hipError_t hipSetDevice(int deviceId);
  // CHECK: result = hipSetDevice(deviceId);
  result = cudaSetDevice(deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags( unsigned int flags );
  // HIP: hipError_t hipSetDeviceFlags(unsigned flags);
  // CHECK: result = hipSetDeviceFlags(flags);
  result = cudaSetDeviceFlags(flags);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadExit(void);
  // HIP: hipError_t hipDeviceReset(void);
  // CHECK: result = hipDeviceReset();
  result = cudaThreadExit();

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig);
  // HIP: hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* cacheConfig);
  // CHECK: result = hipDeviceGetCacheConfig(&FuncCache);
  result = cudaThreadGetCacheConfig(&FuncCache);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig);
  // HIP: hipError_t hipError_t hipDeviceSetCacheConfig(hipFuncCache_t cacheConfig);
  // CHECK: result = hipDeviceSetCacheConfig(FuncCache);
  result = cudaThreadSetCacheConfig(FuncCache);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);
  // HIP: hipError_t hipError_t hipDeviceSynchronize(void);
  // CHECK: result = hipDeviceSynchronize();
  result = cudaThreadSynchronize();

  // CUDA: extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorName(cudaError_t error);
  // HIP: const char* hipGetErrorName(hipError_t hip_error);
  // CHECK: const_ch = hipGetErrorName(Error_t);
  const_ch = cudaGetErrorName(Error_t);

  // CUDA: extern __host__ __cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error);
  // HIP: const char* hipGetErrorString(hipError_t hipError);
  // CHECK: const_ch = hipGetErrorString(Error_t);
  const_ch = cudaGetErrorString(Error_t);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void);
  // HIP: hipError_t hipGetLastError(void);
  // CHECK: result = hipGetLastError();
  result = cudaGetLastError();

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaPeekAtLastError(void);
  // HIP: hipError_t hipPeekAtLastError(void);
  // CHECK: result = hipPeekAtLastError();
  result = cudaPeekAtLastError();

  // CHECK: hipStreamCallback_t StreamCallback_t;
  cudaStreamCallback_t StreamCallback_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
  // HIP: hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData, unsigned int flags);
  // CHECK: result = hipStreamAddCallback(stream, StreamCallback_t, image, flags);
  result = cudaStreamAddCallback(stream, StreamCallback_t, image, flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamAttachMemAsync(cudaStream_t stream, void *devPtr, size_t length __dv(0), unsigned int flags = cudaMemAttachSingle);
  // HIP: hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length __dparm(0), unsigned int flags __dparm(hipMemAttachSingle));
  // CHECK: result = hipStreamAttachMemAsync(stream, deviceptr, bytes, flags);
  result = cudaStreamAttachMemAsync(stream, deviceptr, bytes, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream);
  // HIP: hipError_t hipStreamCreate(hipStream_t* stream);
  // CHECK: result = hipStreamCreate(&stream);
  result = cudaStreamCreate(&stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
  // HIP: hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
  // CHECK: result = hipStreamCreateWithFlags(&stream, flags);
  result = cudaStreamCreateWithFlags(&stream, flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority);
  // HIP: hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags);
  // CHECK: result = hipStreamCreateWithPriority(&stream, flags, intVal);
  result = cudaStreamCreateWithPriority(&stream, flags, intVal);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
  // HIP: hipError_t hipStreamDestroy(hipStream_t stream);
  // CHECK: result = hipStreamDestroy(stream);
  result = cudaStreamDestroy(stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t hStream, unsigned int *flags);
  // HIP: hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
  // CHECK: result = hipStreamGetFlags(stream, &flags);
  result = cudaStreamGetFlags(stream, &flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t hStream, int *priority);
  // HIP: hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
  // CHECK: result = hipStreamGetPriority(stream, &intVal);
  result = cudaStreamGetPriority(stream, &intVal);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);
  // HIP: hipError_t hipStreamQuery(hipStream_t stream);
  // CHECK: result = hipStreamQuery(stream);
  result = cudaStreamQuery(stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
  // HIP: hipError_t hipStreamSynchronize(hipStream_t stream);
  // CHECK: result = hipStreamSynchronize(stream);
  result = cudaStreamSynchronize(stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags __dv(0));
  // HIP: hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
  // CHECK: result = hipStreamWaitEvent(stream, Event_t, flags);
  result = cudaStreamWaitEvent(stream, Event_t, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event);
  // HIP: hipError_t hipEventCreate(hipEvent_t* event);
  // CHECK: result = hipEventCreate(&Event_t);
  result = cudaEventCreate(&Event_t);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
  // HIP: hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
  // CHECK: result = hipEventCreateWithFlags(&Event_t, flags);
  result = cudaEventCreateWithFlags(&Event_t, flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);
  // HIP: hipError_t hipEventDestroy(hipEvent_t event);
  // CHECK: result = hipEventDestroy(Event_t);
  result = cudaEventDestroy(Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
  // HIP: hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
  // CHECK: result = hipEventElapsedTime(&ms, Event_t, Event_2);
  result = cudaEventElapsedTime(&ms, Event_t, Event_2);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);
  // HIP: hipError_t hipEventQuery(hipEvent_t event);
  // CHECK: result = hipEventQuery(Event_t);
  result = cudaEventQuery(Event_t);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
  // CHECK: result = hipEventRecord(Event_t, stream);
  result = cudaEventRecord(Event_t, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);
  // HIP: hipError_t hipEventSynchronize(hipEvent_t event);
  // CHECK: result = hipEventSynchronize(Event_t);
  result = cudaEventSynchronize(Event_t);

  // CHECK: hipFuncAttributes FuncAttributes;
  cudaFuncAttributes FuncAttributes;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
  // HIP: hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func);
  // CHECK: result = hipFuncGetAttributes(&FuncAttributes, func);
  result = cudaFuncGetAttributes(&FuncAttributes, func);

#if CUDA_VERSION >= 9000
  // CHECK: hipFuncAttribute FuncAttribute;
  cudaFuncAttribute FuncAttribute;

  // CHECK: hipLaunchParams LaunchParams;
  cudaLaunchParams LaunchParams;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
  // HIP: hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
  // CHECK: result = hipFuncSetAttribute(func, FuncAttribute, intVal);
  result = cudaFuncSetAttribute(func, FuncAttribute, intVal);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
  // HIP: hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream);
  // CHECK: result = hipLaunchCooperativeKernel(func, gridDim, blockDim, &image, flags, stream);
  result = cudaLaunchCooperativeKernel(func, gridDim, blockDim, &image, flags, stream);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags  __dv(0));
  // HIP: hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices, unsigned int flags);
  // CHECK: result = hipLaunchCooperativeKernelMultiDevice(&LaunchParams, intVal, flags);
  result = cudaLaunchCooperativeKernelMultiDevice(&LaunchParams, intVal, flags);
#endif

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig);
  // HIP: hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t config);
  // CHECK: result = hipFuncSetCacheConfig(func, FuncCache);
  result = cudaFuncSetCacheConfig(func, FuncCache);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config);
  // HIP: hipError_t hipFuncSetSharedMemConfig(const void* func, hipSharedMemConfig config);
  // CHECK: result = hipFuncSetSharedMemConfig(func, SharedMemConfig);
  result = cudaFuncSetSharedMemConfig(func, SharedMemConfig);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
  // HIP: hipError_t hipLaunchKernel(const void* function_address, dim3 numBlocks, dim3 dimBlocks, void** args, size_t sharedMemBytes __dparm(0), hipStream_t stream __dparm(0));
  // CHECK: result = hipLaunchKernel(func, gridDim, blockDim, &image, bytes, stream);
  result = cudaLaunchKernel(func, gridDim, blockDim, &image, bytes, stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize);
  // HIP: hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk);
  // CHECK: result = hipOccupancyMaxActiveBlocksPerMultiprocessor(&intVal, func, device, bytes);
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&intVal, func, device, bytes);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
  // HIP: hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags __dparm(hipOccupancyDefault));
  // CHECK: result = hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&intVal, func, intVal, bytes, flags);
  result = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&intVal, func, intVal, bytes, flags);

  // CUDA: template<class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0);
  // HIP: template <typename T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0);
  // CHECK: result = hipOccupancyMaxPotentialBlockSize(&intVal, &device, func, bytes, deviceId);
  result = cudaOccupancyMaxPotentialBlockSize(&intVal, &device, func, bytes, deviceId);

  // CUDA: template<class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeWithFlags(int* minGridSize, int* blockSize, T func, size_t dynamicSMemSize = 0, int blockSizeLimit = 0, unsigned int flags = 0);
  // HIP: template <typename T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0, unsigned int  flags = 0);
  // CHECK: result = hipOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);
  result = cudaOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);

  return 0;
}
