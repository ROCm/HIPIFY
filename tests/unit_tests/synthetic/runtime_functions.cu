// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#endif

#include "cuda_gl_interop.h"
// CHECK: #include "hip/hip_runtime_api.h"
#include "cuda_profiler_api.h"

int main() {
  printf("12. CUDA Runtime API Functions synthetic test\n");

  size_t bytes = 0;
  size_t width = 0;
  size_t height = 0;
  size_t wOffset = 0;
  size_t hOffset = 0;
  size_t pitch = 0;
  size_t pitch_2 = 0;
  int device = 0;
  int deviceId = 0;
  int intVal = 0;
  int x = 0;
  int y = 0;
  int z = 0;
  int w = 0;
  unsigned int flags = 0;
  unsigned int levels = 0;
  unsigned int count = 0;
  float ms = 0;
  void* deviceptr = nullptr;
  void* deviceptr_2 = nullptr;
  void* image = nullptr;
  void* func = nullptr;
  void* src = nullptr;
  void* dst = nullptr;
  char* ch = nullptr;
  const char* const_ch = nullptr;
  dim3 gridDim;
  dim3 blockDim;
  GLuint gl_uint = 0;
  GLenum gl_enum = 0;
  struct textureReference* texref = nullptr;
  std::string name = "str";

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

  // CHECK: hipEvent_t Event_t;
  // CHECK-Next: hipEvent_t Event_2;
  cudaEvent_t Event_t;
  cudaEvent_t Event_2;

  // CHECK: hipMemcpy3DParms Memcpy3DParms;
  cudaMemcpy3DParms Memcpy3DParms;

  // CHECK: hipMemcpyKind MemcpyKind;
  cudaMemcpyKind MemcpyKind;

  // CHECK: hipChannelFormatDesc ChannelFormatDesc;
  cudaChannelFormatDesc ChannelFormatDesc;

  // CHECK: hipMipmappedArray* MipmappedArray;
  // CHECK-NEXT: hipMipmappedArray_t MipmappedArray_t;
  // CHECK-NEXT: hipMipmappedArray_const_t MipmappedArray_const_t;
  cudaMipmappedArray* MipmappedArray;
  cudaMipmappedArray_t MipmappedArray_t;
  cudaMipmappedArray_const_t MipmappedArray_const_t;

  // CHECK: hipArray* Array;
  // CHECK-NEXT: hipArray_t Array_t;
  // CHECK-NEXT: hipArray_const_t Array_const_t;
  cudaArray* Array;
  cudaArray_t Array_t;
  cudaArray_const_t Array_const_t;

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

  // CUDA: static __inline__ __host__ cudaError_t cudaEventCreate(cudaEvent_t* event, unsigned int flags);
  // HIP: hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
  // CHECK: result = hipEventCreateWithFlags(&Event_t, flags);
  result = cudaEventCreate(&Event_t, flags);

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
  // HIP: template <typename T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, T f, size_t dynSharedMemPerBlk = 0, int blockSizeLimit = 0, unsigned int flags = 0);
  // CHECK: result = hipOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);
  result = cudaOccupancyMaxPotentialBlockSizeWithFlags(&intVal, &device, func, bytes, deviceId, flags);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
  // HIP: hipError_t hipFree(void* ptr);
  // CHECK: result = hipFree(deviceptr);
  result = cudaFree(deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array);
  // HIP: hipError_t hipFreeArray(hipArray* array);
  // CHECK: result = hipFreeArray(Array_t);
  result = cudaFreeArray(Array_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);
  // HIP: hipError_t hipHostFree(void* ptr);
  // CHECK: result = hipHostFree(deviceptr);
  result = cudaFreeHost(deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
  // HIP: hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);
  // CHECK: result = hipFreeMipmappedArray(MipmappedArray_t);
  result = cudaFreeMipmappedArray(MipmappedArray_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetMipmappedArrayLevel(cudaArray_t *levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int level);
  // HIP: hipError_t hipGetMipmappedArrayLevel(hipArray_t* levelArray, hipMipmappedArray_const_t mipmappedArray, unsigned int level);
  // CHECK: result = hipGetMipmappedArrayLevel(&Array_t, MipmappedArray_const_t, flags);
  result = cudaGetMipmappedArrayLevel(&Array_t, MipmappedArray_const_t, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol);
  // HIP: hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol);
  // CHECK: result = hipGetSymbolAddress(&deviceptr, HIP_SYMBOL(image));
  result = cudaGetSymbolAddress(&deviceptr, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol);
  // HIP: hipError_t hipGetSymbolSize(size_t* size, const void* symbol);
  // CHECK: result = hipGetSymbolSize(&bytes, HIP_SYMBOL(image));
  result = cudaGetSymbolSize(&bytes, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
  // HIP: DEPRECATED("use hipHostMalloc instead") hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
  // CHECK: result = hipHostAlloc(&deviceptr, bytes, flags);
  result = cudaHostAlloc(&deviceptr, bytes, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
  // HIP: hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
  // CHECK: result = hipHostGetDevicePointer(&deviceptr, image, flags);
  result = cudaHostGetDevicePointer(&deviceptr, image, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost);
  // HIP: hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
  // CHECK: result = hipHostGetFlags(&flags, image);
  result = cudaHostGetFlags(&flags, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags);
  // HIP: hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
  // CHECK: result = hipHostRegister(image, bytes, flags);
  result = cudaHostRegister(image, bytes, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaHostUnregister(void *ptr);
  // HIP: hipError_t hipHostUnregister(void* hostPtr);
  // CHECK: result = hipHostUnregister(image);
  result = cudaHostUnregister(image);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
  // HIP: hipError_t hipMalloc(void** ptr, size_t size);
  // CHECK: result = hipMalloc(&deviceptr, bytes);
  result = cudaMalloc(&deviceptr, bytes);

  // CHECK: hipPitchedPtr PitchedPtr;
  cudaPitchedPtr PitchedPtr;

  // CHECK: hipExtent Extent;
  cudaExtent Extent;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchedDevPtr, struct cudaExtent extent);
  // HIP: hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
  // CHECK: result = hipMalloc3D(&PitchedPtr, Extent);
  result = cudaMalloc3D(&PitchedPtr, Extent);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));
  // HIP: hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc, struct hipExtent extent, unsigned int flags);
  // CHECK: result = hipMalloc3DArray(&Array_t, &ChannelFormatDesc, Extent, flags);
  result = cudaMalloc3DArray(&Array_t, &ChannelFormatDesc, Extent, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(0), unsigned int flags __dv(0));
  // HIP: hipError_t hipMallocArray(hipArray** array, const hipChannelFormatDesc* desc, size_t width, size_t height __dparm(0), unsigned int flags __dparm(hipArrayDefault));
  // CHECK: result = hipMallocArray(&Array_t, &ChannelFormatDesc, width, height, flags);
  result = cudaMallocArray(&Array_t, &ChannelFormatDesc, width, height, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size);
  // HIP: hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
  // CHECK: result = hipHostMalloc(&deviceptr, bytes, hipHostMallocDefault);
  result = cudaMallocHost(&deviceptr, bytes);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
  // HIP: hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags __dparm(hipMemAttachGlobal));
  // CHECK: result = hipMallocManaged(&deviceptr, bytes, flags);
  result = cudaMallocManaged(&deviceptr, bytes, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int numLevels, unsigned int flags __dv(0));
  // HIP: hipError_t hipMallocMipmappedArray(hipMipmappedArray_t* mipmappedArray, const struct hipChannelFormatDesc* desc, struct hipExtent extent, unsigned int numLevels, unsigned int flags __dparm(0));
  // CHECK: result = hipMallocMipmappedArray(&MipmappedArray_t, &ChannelFormatDesc, Extent, levels, flags);
  result = cudaMallocMipmappedArray(&MipmappedArray_t, &ChannelFormatDesc, Extent, levels, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
  // HIP: hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
  // CHECK: result = hipMallocPitch(&deviceptr, &bytes, width, height);
  result = cudaMallocPitch(&deviceptr, &bytes, width, height);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);
  // CHECK: result = hipMemcpy(deviceptr, deviceptr_2, bytes, MemcpyKind);
  result = cudaMemcpy(deviceptr, deviceptr_2, bytes, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
  // CHECK: result = hipMemcpy2D(deviceptr, pitch, deviceptr_2, pitch_2, width, height, MemcpyKind);
  result = cudaMemcpy2D(deviceptr, pitch, deviceptr_2, pitch_2, width, height, MemcpyKind);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpy2DAsync(deviceptr, pitch, deviceptr_2, pitch_2, width, height, MemcpyKind, stream);
  result = cudaMemcpy2DAsync(deviceptr, pitch, deviceptr_2, pitch_2, width, height, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipMemcpy2DFromArray( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind);
  // CHECK: result = hipMemcpy2DFromArray(deviceptr, pitch, Array_const_t, wOffset, hOffset, width, height, MemcpyKind);
  result = cudaMemcpy2DFromArray(deviceptr, pitch, Array_const_t, wOffset, hOffset, width, height, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpy2DFromArrayAsync( void* dst, size_t dpitch, hipArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpy2DFromArrayAsync(deviceptr, pitch, Array_const_t, wOffset, hOffset, width, height, MemcpyKind, stream);
  result = cudaMemcpy2DFromArrayAsync(deviceptr, pitch, Array_const_t, wOffset, hOffset, width, height, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind);
  // CHECK: result = hipMemcpy2DToArray(Array_t, wOffset, hOffset, deviceptr_2, pitch, width, height, MemcpyKind);
  result = cudaMemcpy2DToArray(Array_t, wOffset, hOffset, deviceptr_2, pitch, width, height, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpy2DToArrayAsync(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpy2DToArrayAsync(Array_t, wOffset, hOffset, deviceptr_2, pitch, width, height, MemcpyKind, stream);
  result = cudaMemcpy2DToArrayAsync(Array_t, wOffset, hOffset, deviceptr_2, pitch, width, height, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
  // HIP: hipError_t hipMemcpy3D(const struct hipMemcpy3DParms* p);
  // CHECK: result = hipMemcpy3D(&Memcpy3DParms);
  result = cudaMemcpy3D(&Memcpy3DParms);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpy3DAsync(const struct hipMemcpy3DParms* p, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpy3DAsync(&Memcpy3DParms, stream);
  result = cudaMemcpy3DAsync(&Memcpy3DParms, stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyAsync(deviceptr, deviceptr_2, bytes, MemcpyKind, stream);
  result = cudaMemcpyAsync(deviceptr, deviceptr_2, bytes, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
  // HIP: hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t sizeBytes, size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyDeviceToHost));
  // CHECK: result = hipMemcpyFromSymbol(deviceptr, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind);
  result = cudaMemcpyFromSymbol(deviceptr, image, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyFromSymbolAsync(deviceptr, HIP_SYMBOL(image), bytes, wOffset, MemcpyKind, stream);
  result = cudaMemcpyFromSymbolAsync(deviceptr, image, bytes, wOffset, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
  // HIP: hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src, int srcDeviceId, size_t sizeBytes);
  // CHECK: result = hipMemcpyPeer(deviceptr, deviceId, deviceptr_2, device, bytes);
  result = cudaMemcpyPeer(deviceptr, deviceId, deviceptr_2, device, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice, size_t count, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src, int srcDevice, size_t sizeBytes, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyPeerAsync(deviceptr, deviceId, deviceptr_2, device, bytes, stream);
  result = cudaMemcpyPeerAsync(deviceptr, deviceId, deviceptr_2, device, bytes, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
  // HIP: hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes, size_t offset __dparm(0), hipMemcpyKind kind __dparm(hipMemcpyHostToDevice));
  // CHECK: result = hipMemcpyToSymbol(HIP_SYMBOL(image), deviceptr, bytes, wOffset, MemcpyKind);
  result = cudaMemcpyToSymbol(image, deviceptr, bytes, wOffset, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src, size_t sizeBytes, size_t offset, hipMemcpyKind kind, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyToSymbolAsync(HIP_SYMBOL(image), deviceptr, bytes, wOffset, MemcpyKind, stream);
  result = cudaMemcpyToSymbolAsync(image, deviceptr, bytes, wOffset, MemcpyKind, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total);
  // HIP: hipError_t hipMemGetInfo(size_t* free, size_t* total);
  // CHECK: result = hipMemGetInfo(&bytes, &wOffset);
  result = cudaMemGetInfo(&bytes, &wOffset);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count);
  // HIP: hipError_t hipMemset(void* dst, int value, size_t sizeBytes);
  // CHECK: result = hipMemset(deviceptr, intVal, bytes);
  result = cudaMemset(deviceptr, intVal, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height);
  // HIP: hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);
  // CHECK: result = hipMemset2D(deviceptr, pitch, intVal, width, height);
  result = cudaMemset2D(deviceptr, pitch, intVal, width, height);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height,hipStream_t stream __dparm(0));
  // CHECK: result = hipMemset2DAsync(deviceptr, pitch, intVal, width, height, stream);
  result = cudaMemset2DAsync(deviceptr, pitch, intVal, width, height, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
  // HIP: hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent );
  // CHECK: result = hipMemset3D(PitchedPtr, intVal, Extent);
  result = cudaMemset3D(PitchedPtr, intVal, Extent);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent ,hipStream_t stream __dparm(0));
  // CHECK: result = hipMemset3DAsync(PitchedPtr, intVal, Extent, stream);
  result = cudaMemset3DAsync(PitchedPtr, intVal, Extent, stream);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemsetAsync(deviceptr, intVal, bytes, stream);
  result = cudaMemsetAsync(deviceptr, intVal, bytes, stream);

  // CUDA: static __inline__ __host__ struct cudaExtent make_cudaExtent(size_t w, size_t h, size_t d);
  // HIP: static inline struct hipExtent make_hipExtent(size_t w, size_t h, size_t d);
  // CHECK: Extent = make_hipExtent(width, height, bytes);
  Extent = make_cudaExtent(width, height, bytes);

  // CUDA: static __inline__ __host__ struct cudaPitchedPtr make_cudaPitchedPtr(void *d, size_t p, size_t xsz, size_t ysz);
  // HIP: static inline struct hipPitchedPtr make_hipPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz);
  // CHECK: PitchedPtr = make_hipPitchedPtr(image, pitch, width, height);
  PitchedPtr = make_cudaPitchedPtr(image, pitch, width, height);

  // CHECK: hipPos Pos;
  cudaPos Pos;

  // CUDA: static __inline__ __host__ struct cudaPos make_cudaPos(size_t x, size_t y, size_t z);
  // HIP: static inline struct hipPos make_hipPos(size_t x, size_t y, size_t z);
  // CHECK: Pos = make_hipPos(width, height, bytes);
  Pos = make_cudaPos(width, height, bytes);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset, size_t hOffset, size_t count, hipMemcpyKind kind);
  // CHECK: result = hipMemcpyFromArray(deviceptr, Array_const_t, wOffset, hOffset, bytes, MemcpyKind);
  result = cudaMemcpyFromArray(deviceptr, Array_const_t, wOffset, hOffset, bytes, MemcpyKind);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset, const void* src, size_t count, hipMemcpyKind kind);
  // CHECK: result = hipMemcpyToArray(Array_t, wOffset, hOffset, deviceptr, bytes, MemcpyKind);
  result = cudaMemcpyToArray(Array_t, wOffset, hOffset, deviceptr, bytes, MemcpyKind);

  // CHECK: hipPointerAttribute_t PointerAttributes;
  cudaPointerAttributes PointerAttributes;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr);
  // HIP: hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);
  // CHECK: result = hipPointerGetAttributes(&PointerAttributes, deviceptr);
  result = cudaPointerGetAttributes(&PointerAttributes, deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice);
  // HIP: hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
  // CHECK: result = hipDeviceCanAccessPeer(&intVal, device, deviceId);
  result = cudaDeviceCanAccessPeer(&intVal, device, deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice);
  // HIP: hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);
  // CHECK: result = hipDeviceDisablePeerAccess(device);
  result = cudaDeviceDisablePeerAccess(device);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
  // HIP: hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
  // CHECK: result = hipDeviceEnablePeerAccess(device, flags);
  result = cudaDeviceEnablePeerAccess(device, flags);

  // CHECK: hipGLDeviceList GLDeviceList;
  cudaGLDeviceList GLDeviceList;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGLGetDevices(unsigned int *pCudaDeviceCount, int *pCudaDevices, unsigned int cudaDeviceCount, enum cudaGLDeviceList deviceList);
  // HIP: hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList);
  // CHECK: result = hipGLGetDevices(&flags, &intVal, count, GLDeviceList);
  result = cudaGLGetDevices(&flags, &intVal, count, GLDeviceList);

  // CHECK: hipGraphicsResource* GraphicsResource;
  // CHECK-NEXT: hipGraphicsResource_t GraphicsResource_t;
  cudaGraphicsResource* GraphicsResource;
  cudaGraphicsResource_t GraphicsResource_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterBuffer(struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags);
  // HIP: hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer, unsigned int flags);
  // CHECK: result = hipGraphicsGLRegisterBuffer(&GraphicsResource, gl_uint, flags);
  result = cudaGraphicsGLRegisterBuffer(&GraphicsResource, gl_uint, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsGLRegisterImage(struct cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int flags);
  // HIP: hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
  // CHECK: result = hipGraphicsGLRegisterImage(&GraphicsResource, gl_uint, gl_enum, flags);
  result = cudaGraphicsGLRegisterImage(&GraphicsResource, gl_uint, gl_enum, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsMapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream __dparm(0));
  // CHECK: result = hipGraphicsMapResources(intVal, &GraphicsResource, stream);
  result = cudaGraphicsMapResources(intVal, &GraphicsResource, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(void **devPtr, size_t *size, cudaGraphicsResource_t resource);
  // HIP: hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, hipGraphicsResource_t resource);
  // CHECK: result = hipGraphicsResourceGetMappedPointer(&deviceptr, &bytes, GraphicsResource);
  result = cudaGraphicsResourceGetMappedPointer(&deviceptr, &bytes, GraphicsResource);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count, cudaGraphicsResource_t *resources, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream __dparm(0));
  // CHECK: result = hipGraphicsUnmapResources(intVal, &GraphicsResource, stream);
  result = cudaGraphicsUnmapResources(intVal, &GraphicsResource, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
  // HIP: hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
  // CHECK: result = hipGraphicsUnregisterResource(GraphicsResource);
  result = cudaGraphicsUnregisterResource(GraphicsResource);

  // CHECK: hipChannelFormatKind ChannelFormatKind;
  cudaChannelFormatKind ChannelFormatKind;

  // CUDA: extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
  // HIP: HIP_PUBLIC_API hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f);
  // CHECK: ChannelFormatDesc = hipCreateChannelDesc(x, y, z, w, ChannelFormatKind);
  ChannelFormatDesc = cudaCreateChannelDesc(x, y, z, w, ChannelFormatKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, cudaArray_const_t array);
  // HIP: hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);
  // CHECK: result = hipGetChannelDesc(&ChannelFormatDesc, Array_const_t);
  result = cudaGetChannelDesc(&ChannelFormatDesc, Array_const_t);

  // CHECK: hipTextureObject_t TextureObject_t;
  cudaTextureObject_t TextureObject_t;

  // CHECK: hipResourceDesc ResourceDesc;
  cudaResourceDesc ResourceDesc;

  // CHECK: hipTextureDesc TextureDesc;
  cudaTextureDesc TextureDesc;

  // CHECK: hipResourceViewDesc ResourceViewDesc;
  cudaResourceViewDesc ResourceViewDesc;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaCreateTextureObject(cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc, const struct cudaTextureDesc *pTexDesc, const struct cudaResourceViewDesc *pResViewDesc);
  // HIP: hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject, const hipResourceDesc* pResDesc, const hipTextureDesc* pTexDesc, const struct hipResourceViewDesc* pResViewDesc);
  // CHECK: result = hipCreateTextureObject(&TextureObject_t, &ResourceDesc, &TextureDesc, &ResourceViewDesc);
  result = cudaCreateTextureObject(&TextureObject_t, &ResourceDesc, &TextureDesc, &ResourceViewDesc);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDestroyTextureObject(cudaTextureObject_t texObject);
  // HIP: hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);
  // CHECK: result = hipDestroyTextureObject(TextureObject_t);
  result = cudaDestroyTextureObject(TextureObject_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc, cudaTextureObject_t texObject);
  // HIP: hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc, hipTextureObject_t textureObject);
  // CHECK: result = hipGetTextureObjectResourceDesc(&ResourceDesc, TextureObject_t);
  result = cudaGetTextureObjectResourceDesc(&ResourceDesc, TextureObject_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectResourceViewDesc(struct cudaResourceViewDesc *pResViewDesc, cudaTextureObject_t texObject);
  // HIP: hipError_t hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc, hipTextureObject_t textureObject);
  // CHECK: result = hipGetTextureObjectResourceViewDesc(&ResourceViewDesc, TextureObject_t);
  result = cudaGetTextureObjectResourceViewDesc(&ResourceViewDesc, TextureObject_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGetTextureObjectTextureDesc(struct cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject);
  // HIP: hipError_t hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc, hipTextureObject_t textureObject);
  // CHECK: result = hipGetTextureObjectTextureDesc(&TextureDesc, TextureObject_t);
  result = cudaGetTextureObjectTextureDesc(&TextureDesc, TextureObject_t);

  // CHECK: hipSurfaceObject_t SurfaceObject_t;
  cudaSurfaceObject_t SurfaceObject_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject, const struct cudaResourceDesc *pResDesc);
  // HIP: hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject, const hipResourceDesc* pResDesc);
  // CHECK: result = hipCreateSurfaceObject(&SurfaceObject_t, &ResourceDesc);
  result = cudaCreateSurfaceObject(&SurfaceObject_t, &ResourceDesc);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject);
  // HIP: hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);
  // CHECK: result = hipDestroySurfaceObject(SurfaceObject_t);
  result = cudaDestroySurfaceObject(SurfaceObject_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion);
  // HIP: hipError_t hipDriverGetVersion(int* driverVersion);
  // CHECK: result = hipDriverGetVersion(&intVal);
  result = cudaDriverGetVersion(&intVal);

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion);
  // HIP: hipError_t hipRuntimeGetVersion(int* runtimeVersion);
  // CHECK: result = hipRuntimeGetVersion(&intVal);
  result = cudaRuntimeGetVersion(&intVal);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaProfilerStart(void);
  // HIP: hipError_t hipProfilerStart();
  // CHECK: result = hipProfilerStart();
  result = cudaProfilerStart();

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaProfilerStop(void);
  // HIP: hipError_t hipProfilerStop();
  // CHECK: result = hipProfilerStop();
  result = cudaProfilerStop();

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
  // HIP: hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value);
  // CHECK: result = hipDeviceSetLimit(Limit, bytes);
  result = cudaDeviceSetLimit(Limit, bytes);

  // TODO
  // CUDA: template<typename UnaryFunction, class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMem(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0);
  // HIP: template<typename UnaryFunction, class T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeVariableSMem(int* min_grid_size, int* block_size, T func, UnaryFunction block_size_to_dynamic_smem_size, int block_size_limit = 0);

  // TODO
  // CUDA: template<typename UnaryFunction, class T> static __inline__ __host__ CUDART_DEVICE cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* minGridSize, int* blockSize, T func, UnaryFunction blockSizeToDynamicSMemSize, int blockSizeLimit = 0, unsigned int flags = 0);
  // HIP:  template<typename UnaryFunction, class T> static hipError_t __host__ inline hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(int* min_grid_size, int* block_size, T func, UnaryFunction block_size_to_dynamic_smem_size, int block_size_limit = 0, unsigned int flags = 0);

#if CUDA_VERSION >= 8000
  // CHECK: hipDeviceP2PAttr DeviceP2PAttr;
  cudaDeviceP2PAttr DeviceP2PAttr;

  // CHECK: hipMemoryAdvise MemoryAdvise;
  cudaMemoryAdvise MemoryAdvise;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // HIP: hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // CHECK: result = hipDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);
  result = cudaDeviceGetP2PAttribute(&intVal, DeviceP2PAttr, device, deviceId);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
  // HIP: hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device);
  // CHECK: result = hipMemAdvise(deviceptr, bytes, MemoryAdvise, device);
  result = cudaMemAdvise(deviceptr, bytes, MemoryAdvise, device);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream __dv(0));
  // HIP: hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemPrefetchAsync(deviceptr, bytes, device, stream);
  result = cudaMemPrefetchAsync(deviceptr, bytes, device, stream);

  // CHECK: hipMemRangeAttribute MemRangeAttribute;
  cudaMemRangeAttribute MemRangeAttribute;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttribute(void *data, size_t dataSize, enum cudaMemRangeAttribute attribute, const void *devPtr, size_t count);
  // HIP: hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute, const void* dev_ptr, size_t count);
  // CHECK: result = hipMemRangeGetAttribute(deviceptr, bytes, MemRangeAttribute, deviceptr_2, wOffset);
  result = cudaMemRangeGetAttribute(deviceptr, bytes, MemRangeAttribute, deviceptr_2, wOffset);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaMemRangeGetAttributes(void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes, size_t numAttributes, const void *devPtr, size_t count);
  // HIP: hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes, hipMemRangeAttribute* attributes, size_t num_attributes, const void* dev_ptr, size_t count);
  // CHECK: result = hipMemRangeGetAttributes(&deviceptr, &bytes, &MemRangeAttribute, wOffset, deviceptr_2, hOffset);
  result = cudaMemRangeGetAttributes(&deviceptr, &bytes, &MemRangeAttribute, wOffset, deviceptr_2, hOffset);
#endif

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

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernelMultiDevice(struct cudaLaunchParams *launchParamsList, unsigned int numDevices, unsigned int flags __dv(0));
  // HIP: hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices, unsigned int flags);
  // CHECK: result = hipLaunchCooperativeKernelMultiDevice(&LaunchParams, intVal, flags);
  result = cudaLaunchCooperativeKernelMultiDevice(&LaunchParams, intVal, flags);
#endif

#if CUDA_VERSION <= 10000
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
  // HIP: hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dparm(0), hipStream_t stream __dparm(0));
  // CHECK: result = hipConfigureCall(gridDim, blockDim, bytes, stream);
  result = cudaConfigureCall(gridDim, blockDim, bytes, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunch(const void *func);
  // HIP: hipError_t hipLaunchByPtr(const void* func);
  // CHECK: result = hipLaunchByPtr(deviceptr);
  result = cudaLaunch(deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset);
  // HIP: hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
  // CHECK: result = hipSetupArgument(deviceptr, bytes, wOffset);
  result = cudaSetupArgument(deviceptr, bytes, wOffset);
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipHostFn_t hostFn;
  cudaHostFn_t hostFn;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData);
  // HIP: hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
  // CHECK: result = hipLaunchHostFunc(stream, hostFn, image);
  result = cudaLaunchHostFunc(stream, hostFn, image);

  // CHECK: hipGraph_t Graph_t, Graph_t_2;
  cudaGraph_t Graph_t, Graph_t_2;

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

  // CHECK: hipGraphNode_t graphNode, graphNode_2;
  cudaGraphNode_t graphNode, graphNode_2;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddChildGraphNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaGraph_t childGraph);
  // HIP: hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipGraph_t childGraph);
  // CHECK: result = hipGraphAddChildGraphNode(&graphNode, Graph_t, &graphNode_2, bytes, Graph_t_2);
  result = cudaGraphAddChildGraphNode(&graphNode, Graph_t, &graphNode_2, bytes, Graph_t_2);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
  // HIP: hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies);
  // CHECK: result = hipGraphAddDependencies(Graph_t, &graphNode, &graphNode_2, bytes);
  result = cudaGraphAddDependencies(Graph_t, &graphNode, &graphNode_2, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddEmptyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies);
  // HIP: hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies);
  // CHECK: result = hipGraphAddEmptyNode(&graphNode, Graph_t, &graphNode_2, bytes);
  result = cudaGraphAddEmptyNode(&graphNode, Graph_t, &graphNode_2, bytes);

  // CHECK: hipHostNodeParams HostNodeParams;
  cudaHostNodeParams HostNodeParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddHostNode(&graphNode, Graph_t, &graphNode_2, bytes, &HostNodeParams);
  result = cudaGraphAddHostNode(&graphNode, Graph_t, &graphNode_2, bytes, &HostNodeParams);

  // CHECK: hipKernelNodeParams KernelNodeParams;
  cudaKernelNodeParams KernelNodeParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddKernelNode(&graphNode, Graph_t, &graphNode_2, bytes, &KernelNodeParams);
  result = cudaGraphAddKernelNode(&graphNode, Graph_t, &graphNode_2, bytes, &KernelNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemcpy3DParms *pCopyParams);
  // HIP: hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipMemcpy3DParms* pCopyParams);
  // CHECK: result = hipGraphAddMemcpyNode(&graphNode, Graph_t, &graphNode_2, bytes, &Memcpy3DParms);
  result = cudaGraphAddMemcpyNode(&graphNode, Graph_t, &graphNode_2, bytes, &Memcpy3DParms);

  // CHECK: hipMemsetParams MemsetParams;
  cudaMemsetParams MemsetParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemsetNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaMemsetParams *pMemsetParams);
  // HIP: hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipMemsetParams* pMemsetParams);
  // CHECK: result = hipGraphAddMemsetNode(&graphNode, Graph_t, &graphNode_2, bytes, &MemsetParams);
  result = cudaGraphAddMemsetNode(&graphNode, Graph_t, &graphNode_2, bytes, &MemsetParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphChildGraphNodeGetGraph(cudaGraphNode_t node, cudaGraph_t *pGraph);
  // HIP: hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);
  // CHECK: result = hipGraphChildGraphNodeGetGraph(graphNode, &Graph_t);
  result = cudaGraphChildGraphNodeGetGraph(graphNode, &Graph_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphClone(cudaGraph_t *pGraphClone, cudaGraph_t originalGraph);
  // HIP: hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);
  // CHECK: result = hipGraphClone(&Graph_t, Graph_t_2);
  result = cudaGraphClone(&Graph_t, Graph_t_2);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphCreate(cudaGraph_t *pGraph, unsigned int flags);
  // HIP: hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
  // CHECK: result = hipGraphCreate(&Graph_t, flags);
  result = cudaGraphCreate(&Graph_t, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphDestroy(cudaGraph_t graph);
  // HIP: hipError_t hipGraphDestroy(hipGraph_t graph);
  // CHECK: result = hipGraphDestroy(Graph_t);
  result = cudaGraphDestroy(Graph_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphDestroyNode(cudaGraphNode_t node);
  // HIP: hipError_t hipGraphDestroyNode(hipGraphNode_t node);
  // CHECK: result = hipGraphDestroyNode(graphNode);
  result = cudaGraphDestroyNode(graphNode);

  // CHECK: hipGraphExec_t GraphExec_t;
  cudaGraphExec_t GraphExec_t;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec);
  // HIP: hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);
  // CHECK: result = hipGraphExecDestroy(GraphExec_t);
  result = cudaGraphExecDestroy(GraphExec_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphGetEdges(cudaGraph_t graph, cudaGraphNode_t *from, cudaGraphNode_t *to, size_t *numEdges);
  // HIP: hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges);
  // CHECK: result = hipGraphGetEdges(Graph_t, &graphNode, &graphNode_2, &bytes);
  result = cudaGraphGetEdges(Graph_t, &graphNode, &graphNode_2, &bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t *nodes, size_t *numNodes);
  // HIP: hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);
  // CHECK: result = hipGraphGetNodes(Graph_t, &graphNode, &bytes);
  result = cudaGraphGetNodes(Graph_t, &graphNode, &bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t *pRootNodes, size_t *pNumRootNodes);
  // HIP: hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes);
  // CHECK: result = hipGraphGetRootNodes(Graph_t, &graphNode, &bytes);
  result = cudaGraphGetRootNodes(Graph_t, &graphNode, &bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeGetParams(cudaGraphNode_t node, struct cudaHostNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphHostNodeGetParams(graphNode, &HostNodeParams);
  result = cudaGraphHostNodeGetParams(graphNode, &HostNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphHostNodeSetParams(cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphHostNodeSetParams(graphNode, &HostNodeParams);
  result = cudaGraphHostNodeSetParams(graphNode, &HostNodeParams);

  char* name_ = const_cast<char*>(name.c_str());

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, cudaGraphNode_t *pErrorNode, char *pLogBuffer, size_t bufferSize);
  // HIP: hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph, hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
  // CHECK: result = hipGraphInstantiate(&GraphExec_t, Graph_t, &graphNode, name_, bytes);
  result = cudaGraphInstantiate(&GraphExec_t, Graph_t, &graphNode, name_, bytes);

  // CHECK: hipGraphNodeType GraphNodeType;
  cudaGraphNodeType GraphNodeType;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipStreamCaptureMode streamCaptureMode;
  cudaStreamCaptureMode streamCaptureMode;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
  // HIP: hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
  // CHECK: result = hipStreamBeginCapture(stream, streamCaptureMode);
  result = cudaStreamBeginCapture(stream, streamCaptureMode);

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

#if CUDA_VERSION >= 10020
  // CHECK: hipGraphExecUpdateResult GraphExecUpdateResult;
  cudaGraphExecUpdateResult GraphExecUpdateResult;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipKernelNodeAttrID kernelNodeAttrID;
  cudaKernelNodeAttrID kernelNodeAttrID;
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue;
  cudaKernelNodeAttrValue kernelNodeAttrValue;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, const union cudaKernelNodeAttrValue* value);
  // HIP: hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cudaGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetAttribute(cudaGraphNode_t hNode, enum cudaKernelNodeAttrID attr, union cudaKernelNodeAttrValue* value_out);
  // HIP: hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cudaGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphExecKernelNodeSetParams(GraphExec_t, graphNode, &KernelNodeParams);
  result = cudaGraphExecKernelNodeSetParams(GraphExec_t, graphNode, &KernelNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
  // HIP: hipError_t hipGraphExecMemcpyNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);
  // CHECK: result = hipGraphExecMemcpyNodeSetParams(GraphExec_t, graphNode, &Memcpy3DParms);
  result = cudaGraphExecMemcpyNodeSetParams(GraphExec_t, graphNode, &Memcpy3DParms);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
  // HIP: hipError_t hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipMemsetParams* pNodeParams);
  // CHECK: result = hipGraphExecMemsetNodeSetParams(GraphExec_t, graphNode, &MemsetParams);
  result = cudaGraphExecMemsetNodeSetParams(GraphExec_t, graphNode, &MemsetParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecHostNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaHostNodeParams *pNodeParams);
  // HIP: hipError_t hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphExecHostNodeSetParams(GraphExec_t, graphNode, &HostNodeParams);
  result = cudaGraphExecHostNodeSetParams(GraphExec_t, graphNode, &HostNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph, cudaGraphNode_t *hErrorNode_out, enum cudaGraphExecUpdateResult *updateResult_out);
  // HIP: hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t* hErrorNode_out, hipGraphExecUpdateResult* updateResult_out);
  // CHECK: result = hipGraphExecUpdate(GraphExec_t, Graph_t, &graphNode, &GraphExecUpdateResult);
  result = cudaGraphExecUpdate(GraphExec_t, Graph_t, &graphNode, &GraphExecUpdateResult);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphKernelNodeGetParams(graphNode, &KernelNodeParams);
  result = cudaGraphKernelNodeGetParams(graphNode, &KernelNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams *pNodeParams);
  // HIP: hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphKernelNodeSetParams(graphNode, &KernelNodeParams);
  result = cudaGraphKernelNodeSetParams(graphNode, &KernelNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
  // HIP: hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
  // CHECK: result = hipGraphLaunch(GraphExec_t, stream);
  result = cudaGraphLaunch(GraphExec_t, stream);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node, struct cudaMemcpy3DParms *pNodeParams);
  // HIP: hipError_t hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);
  // CHECK: result = hipGraphMemcpyNodeGetParams(graphNode, &Memcpy3DParms);
  result = cudaGraphMemcpyNodeGetParams(graphNode, &Memcpy3DParms);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams(cudaGraphNode_t node, const struct cudaMemcpy3DParms *pNodeParams);
  // HIP: hipError_t hipGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);
  // CHECK: result = hipGraphMemcpyNodeSetParams(graphNode, &Memcpy3DParms);
  result = cudaGraphMemcpyNodeSetParams(graphNode, &Memcpy3DParms);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeGetParams(cudaGraphNode_t node, struct cudaMemsetParams *pNodeParams);
  // HIP: hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);
  // CHECK: result = hipGraphMemsetNodeGetParams(graphNode, &MemsetParams);
  result = cudaGraphMemsetNodeGetParams(graphNode, &MemsetParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemsetNodeSetParams(cudaGraphNode_t node, const struct cudaMemsetParams *pNodeParams);
  // HIP: hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);
  // CHECK: result = hipGraphMemsetNodeSetParams(graphNode, &MemsetParams);
  result = cudaGraphMemsetNodeSetParams(graphNode, &MemsetParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeFindInClone(cudaGraphNode_t *pNode, cudaGraphNode_t originalNode, cudaGraph_t clonedGraph);
  // HIP: hipError_t hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph);
  // CHECK: result = hipGraphNodeFindInClone(&graphNode, graphNode_2, Graph_t);
  result = cudaGraphNodeFindInClone(&graphNode, graphNode_2, Graph_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependencies(cudaGraphNode_t node, cudaGraphNode_t *pDependencies, size_t *pNumDependencies);
  // HIP: hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies, size_t* pNumDependencies);
  // CHECK: result = hipGraphNodeGetDependencies(graphNode, &graphNode_2, &bytes);
  result = cudaGraphNodeGetDependencies(graphNode, &graphNode_2, &bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetDependentNodes(cudaGraphNode_t node, cudaGraphNode_t *pDependentNodes, size_t *pNumDependentNodes);
  // HIP: hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes, size_t* pNumDependentNodes);
  // CHECK: result = hipGraphNodeGetDependentNodes(graphNode, &graphNode_2, &bytes);
  result = cudaGraphNodeGetDependentNodes(graphNode, &graphNode_2, &bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType *pType);
  // HIP: hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);
  // CHECK: result = hipGraphNodeGetType(graphNode, &GraphNodeType);
  result = cudaGraphNodeGetType(graphNode, &GraphNodeType);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphRemoveDependencies(cudaGraph_t graph, const cudaGraphNode_t *from, const cudaGraphNode_t *to, size_t numDependencies);
  // HIP: hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies);
  // CHECK: result = hipGraphRemoveDependencies(Graph_t, &graphNode, &graphNode, bytes);
  result = cudaGraphRemoveDependencies(Graph_t, &graphNode, &graphNode, bytes);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphKernelNodeCopyAttributes(cudaGraphNode_t hSrc, cudaGraphNode_t hDst);
  // HIP: hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst);
  // CHECK: result = hipGraphKernelNodeCopyAttributes(graphNode, graphNode_2);
  result = cudaGraphKernelNodeCopyAttributes(graphNode, graphNode_2);
#endif

#if CUDA_VERSION >= 11010
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, hipMemcpyKind kind);
  // CHECK: result = hipGraphAddMemcpyNode1D(&graphNode, Graph_t, &graphNode_2, width, dst, src, bytes, MemcpyKind);
  result = cudaGraphAddMemcpyNode1D(&graphNode, Graph_t, &graphNode_2, width, dst, src, bytes, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemcpyNodeSetParams1D(cudaGraphNode_t node, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void* dst, const void* src, size_t count, hipMemcpyKind kind);
  // CHECK: result = hipGraphMemcpyNodeSetParams1D(graphNode, dst, src, bytes, MemcpyKind);
  result = cudaGraphMemcpyNodeSetParams1D(graphNode, dst, src, bytes, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
  // HIP: hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event);
  // CHECK: result = hipGraphAddEventRecordNode(&graphNode, Graph_t, &graphNode_2, bytes, Event_t);
  result = cudaGraphAddEventRecordNode(&graphNode, Graph_t, &graphNode_2, bytes, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
  // HIP: hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
  // CHECK: result = hipGraphEventRecordNodeGetEvent(graphNode, &Event_t);
  result = cudaGraphEventRecordNodeGetEvent(graphNode, &Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphEventRecordNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
  // HIP: hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
  // CHECK: result = hipGraphEventRecordNodeSetEvent(graphNode, Event_t);
  result = cudaGraphEventRecordNodeSetEvent(graphNode, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event);
  // HIP: hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event);
  // CHECK: result = hipGraphAddEventWaitNode(&graphNode, Graph_t, &graphNode_2, bytes, Event_t);
  result = cudaGraphAddEventWaitNode(&graphNode, Graph_t, &graphNode_2, bytes, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node, cudaEvent_t *event_out);
  // HIP: hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
  // CHECK: result = hipGraphEventWaitNodeGetEvent(graphNode, &Event_t);
  result = cudaGraphEventWaitNodeGetEvent(graphNode, &Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphEventWaitNodeSetEvent(cudaGraphNode_t node, cudaEvent_t event);
  // HIP: hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
  // CHECK: result = hipGraphEventWaitNodeSetEvent(graphNode, Event_t);
  result = cudaGraphEventWaitNodeSetEvent(graphNode, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecMemcpyNodeSetParams1D(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
  // HIP: hipError_t hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void* dst, const void* src, size_t count, hipMemcpyKind kind);
  // CHECK: result = hipGraphExecMemcpyNodeSetParams1D(GraphExec_t, graphNode, dst, src, bytes, MemcpyKind);
  result = cudaGraphExecMemcpyNodeSetParams1D(GraphExec_t, graphNode, dst, src, bytes, MemcpyKind);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecChildGraphNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, cudaGraph_t childGraph);
  // HIP: hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph);
  // CHECK: result = hipGraphExecChildGraphNodeSetParams(GraphExec_t, graphNode, Graph_t);
  result = cudaGraphExecChildGraphNodeSetParams(GraphExec_t, graphNode, Graph_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventRecordNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
  // HIP: hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event);
  // CHECK: result = hipGraphExecEventRecordNodeSetEvent(GraphExec_t, graphNode, Event_t);
  result = cudaGraphExecEventRecordNodeSetEvent(GraphExec_t, graphNode, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecEventWaitNodeSetEvent(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, cudaEvent_t event);
  // HIP: hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event);
  // CHECK: result = hipGraphExecEventWaitNodeSetEvent(GraphExec_t, graphNode, Event_t);
  result = cudaGraphExecEventWaitNodeSetEvent(GraphExec_t, graphNode, Event_t);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
  // HIP: hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);
  // CHECK: result = hipGraphUpload(GraphExec_t, stream);
  result = cudaGraphUpload(GraphExec_t, stream);
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

  // CHECK: hipExternalSemaphoreSignalNodeParams ExternalSemaphoreSignalNodeParams;
  cudaExternalSemaphoreSignalNodeParams ExternalSemaphoreSignalNodeParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresSignalNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
  // HIP: hipError_t hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphAddExternalSemaphoresSignalNode(&graphNode, Graph_t, &graphNode_2, bytes, &ExternalSemaphoreSignalNodeParams);
  result = cudaGraphAddExternalSemaphoresSignalNode(&graphNode, Graph_t, &graphNode_2, bytes, &ExternalSemaphoreSignalNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreSignalNodeParams *params_out);
  // HIP: hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out);
  // CHECK: result = hipGraphExternalSemaphoresSignalNodeGetParams(graphNode, &ExternalSemaphoreSignalNodeParams);
  result = cudaGraphExternalSemaphoresSignalNodeGetParams(graphNode, &ExternalSemaphoreSignalNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresSignalNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
  // HIP: hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphExternalSemaphoresSignalNodeSetParams(graphNode, &ExternalSemaphoreSignalNodeParams);
  result = cudaGraphExternalSemaphoresSignalNodeSetParams(graphNode, &ExternalSemaphoreSignalNodeParams);

  // CHECK: hipExternalSemaphoreWaitNodeParams ExternalSemaphoreWaitNodeParams;
  cudaExternalSemaphoreWaitNodeParams ExternalSemaphoreWaitNodeParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddExternalSemaphoresWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
  // HIP: hipError_t hipGraphAddExternalSemaphoresWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphAddExternalSemaphoresWaitNode(&graphNode, Graph_t, &graphNode_2, bytes, &ExternalSemaphoreWaitNodeParams);
  result = cudaGraphAddExternalSemaphoresWaitNode(&graphNode, Graph_t, &graphNode_2, bytes, &ExternalSemaphoreWaitNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeGetParams(cudaGraphNode_t hNode, struct cudaExternalSemaphoreWaitNodeParams *params_out);
  // HIP: hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out);
  // CHECK: result = hipGraphExternalSemaphoresWaitNodeGetParams(graphNode, &ExternalSemaphoreWaitNodeParams);
  result = cudaGraphExternalSemaphoresWaitNodeGetParams(graphNode, &ExternalSemaphoreWaitNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExternalSemaphoresWaitNodeSetParams(cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
  // HIP: hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphExternalSemaphoresWaitNodeSetParams(graphNode, &ExternalSemaphoreWaitNodeParams);
  result = cudaGraphExternalSemaphoresWaitNodeSetParams(graphNode, &ExternalSemaphoreWaitNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresSignalNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreSignalNodeParams *nodeParams);
  // HIP: hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphExecExternalSemaphoresSignalNodeSetParams(GraphExec_t, graphNode, &ExternalSemaphoreSignalNodeParams);
  result = cudaGraphExecExternalSemaphoresSignalNodeSetParams(GraphExec_t, graphNode, &ExternalSemaphoreSignalNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphExecExternalSemaphoresWaitNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, const struct cudaExternalSemaphoreWaitNodeParams *nodeParams);
  // HIP: hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphExecExternalSemaphoresWaitNodeSetParams(GraphExec_t, graphNode, &ExternalSemaphoreWaitNodeParams);
  result = cudaGraphExecExternalSemaphoresWaitNodeSetParams(GraphExec_t, graphNode, &ExternalSemaphoreWaitNodeParams);
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipUserObject_t userObject;
  cudaUserObject_t userObject;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);
  // HIP: hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);
  // CHECK: result = hipUserObjectCreate(&userObject, image, hostFn, count, flags);
  result = cudaUserObjectCreate(&userObject, image, hostFn, count, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaUserObjectRelease(cudaUserObject_t object, unsigned int count __dv(1));
  // HIP: hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count);
  // CHECK: result = hipUserObjectRelease(userObject, count);
  result = cudaUserObjectRelease(userObject, count);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaUserObjectRetain(cudaUserObject_t object, unsigned int count __dv(1));
  // HIP: hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count);
  // CHECK: result = hipUserObjectRetain(userObject, count);
  result = cudaUserObjectRetain(userObject, count);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1), unsigned int flags __dv(0));
  // HIP: hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags);
  // CHECK: result = hipGraphRetainUserObject(Graph_t, userObject, count, flags);
  result = cudaGraphRetainUserObject(Graph_t, userObject, count, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphReleaseUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count __dv(1));
  // HIP: hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count);
  // CHECK: result = hipGraphReleaseUserObject(Graph_t, userObject, count);
  result = cudaGraphReleaseUserObject(Graph_t, userObject, count);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphDebugDotPrint(cudaGraph_t graph, const char *path, unsigned int flags);
  // HIP: hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags);
  // CHECK: result = hipGraphDebugDotPrint(Graph_t, name.c_str(), flags);
  result = cudaGraphDebugDotPrint(Graph_t, name.c_str(), flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags __dv(0));
  // HIP: hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies, unsigned int flags __dparm(0));
  // CHECK: result = hipStreamUpdateCaptureDependencies(stream, &graphNode, bytes, flags);
  result = cudaStreamUpdateCaptureDependencies(stream, &graphNode, bytes, flags);
#endif

#if CUDA_VERSION >= 11040
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags);
  // HIP: hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags);
  // CHECK: result = hipGraphInstantiateWithFlags(&GraphExec_t, Graph_t, ull);
  result = cudaGraphInstantiateWithFlags(&GraphExec_t, Graph_t, ull);

  // CHECK: hipGraphMemAttributeType GraphMemAttributeType;
  cudaGraphMemAttributeType GraphMemAttributeType;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
  // HIP: hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
  // CHECK: result = hipDeviceGetGraphMemAttribute(device, GraphMemAttributeType, image);
  result = cudaDeviceGetGraphMemAttribute(device, GraphMemAttributeType, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr, void* value);
  // HIP: hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
  // CHECK: result = hipDeviceSetGraphMemAttribute(device, GraphMemAttributeType, image);
  result = cudaDeviceSetGraphMemAttribute(device, GraphMemAttributeType, image);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaDeviceGraphMemTrim(int device);
  // HIP: hipError_t hipDeviceGraphMemTrim(int device);
  // CHECK: result = hipDeviceGraphMemTrim(device);
  result = cudaDeviceGraphMemTrim(device);

  // CHECK: hipMemAllocNodeParams MemAllocNodeParams;
  cudaMemAllocNodeParams MemAllocNodeParams;

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, struct cudaMemAllocNodeParams *nodeParams);
  // HIP: hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipMemAllocNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddMemAllocNode(&graphNode, Graph_t, &graphNode_2, bytes, &MemAllocNodeParams);
  result = cudaGraphAddMemAllocNode(&graphNode, Graph_t, &graphNode_2, bytes, &MemAllocNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node, struct cudaMemAllocNodeParams *params_out);
  // HIP: hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams);
  // CHECK: result = hipGraphMemAllocNodeGetParams(graphNode, &MemAllocNodeParams);
  result = cudaGraphMemAllocNodeGetParams(graphNode, &MemAllocNodeParams);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphAddMemFreeNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, void *dptr);
  // HIP: hipError_t hipGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dev_ptr);
  // CHECK: result = hipGraphAddMemFreeNode(&graphNode, Graph_t, &graphNode_2, bytes, deviceptr);
  result = cudaGraphAddMemFreeNode(&graphNode, Graph_t, &graphNode_2, bytes, deviceptr);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node, void *dptr_out);
  // HIP: hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr);
  // CHECK: result = hipGraphMemFreeNodeGetParams(graphNode, &deviceptr);
  result = cudaGraphMemFreeNodeGetParams(graphNode, &deviceptr);
#endif

#if CUDA_VERSION >= 11060
  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeSetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled);
  // HIP: hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int isEnabled);
  // CHECK: result = hipGraphNodeSetEnabled(GraphExec_t, graphNode, flags);
  result = cudaGraphNodeSetEnabled(GraphExec_t, graphNode, flags);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaGraphNodeGetEnabled(cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int *isEnabled);
  // HIP: hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int* isEnabled);
  // CHECK: result = hipGraphNodeGetEnabled(GraphExec_t, graphNode, &flags);
  result = cudaGraphNodeGetEnabled(GraphExec_t, graphNode, &flags);
#endif

#if CUDA_VERSION < 12000
  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipBindTexture(size_t* offset, const textureReference* tex, const void* devPtr, const hipChannelFormatDesc* desc, size_t size __dparm(UINT_MAX));
  // CHECK: result = hipBindTexture(&wOffset, texref, deviceptr, &ChannelFormatDesc, bytes);
  result = cudaBindTexture(&wOffset, texref, deviceptr, &ChannelFormatDesc, bytes);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTexture2D(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, size_t pitch);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipBindTexture2D(size_t* offset, const textureReference* tex, const void* devPtr, const hipChannelFormatDesc* desc, size_t width, size_t height, size_t pitch);
  // CHECK: result = hipBindTexture2D(&wOffset, texref, deviceptr, &ChannelFormatDesc, width, height, pitch);
  result = cudaBindTexture2D(&wOffset, texref, deviceptr, &ChannelFormatDesc, width, height, pitch);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipBindTextureToArray(const textureReference* tex, hipArray_const_t array, const hipChannelFormatDesc* desc);
  // CHECK: result = hipBindTextureToArray(texref, Array_const_t, &ChannelFormatDesc);
  result = cudaBindTextureToArray(texref, Array_const_t, &ChannelFormatDesc);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaBindTextureToMipmappedArray(const struct textureReference *texref, cudaMipmappedArray_const_t mipmappedArray, const struct cudaChannelFormatDesc *desc);
  // HIP: hipError_t hipBindTextureToMipmappedArray(const textureReference* tex, hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc* desc);
  // CHECK: result = hipBindTextureToMipmappedArray(texref, MipmappedArray_const_t, &ChannelFormatDesc);
  result = cudaBindTextureToMipmappedArray(texref, MipmappedArray_const_t, &ChannelFormatDesc);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
  // HIP: hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
  // CHECK: result = hipGetTextureAlignmentOffset(&wOffset, texref);
  result = cudaGetTextureAlignmentOffset(&wOffset, texref);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const void *symbol);
  // HIP: hipError_t hipGetTextureReference(const textureReference** texref, const void* symbol);
  // CHECK: result = hipGetTextureReference(const_cast<const textureReference**>(&texref), {{(HIP_SYMBOL\()?}}image{{(\))?}});
  result = cudaGetTextureReference(const_cast<const textureReference**>(&texref), image);

  // CUDA: extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipUnbindTexture(const textureReference* tex);
  // CHECK: result = hipUnbindTexture(texref);
  result = cudaUnbindTexture(texref);
#endif

  return 0;
}
