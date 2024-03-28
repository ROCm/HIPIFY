// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>
#if defined(_WIN32)
  #include "windows.h"
  #include <GL/glew.h>
#elif CUDA_VERSION <= 10000
  #include <GL/glew.h>
#endif
#include "cudaGL.h"
#include "cudaProfiler.h"

int main() {
  printf("09. CUDA Driver API Functions synthetic test\n");

  unsigned int flags = 0;
  unsigned int flags_2 = 0;
  int dim = 0;
  int count = 0;
  int iBlockSize = 0;
  int iBlockSize_2 = 0;
  size_t bytes = 0;
  size_t bytes_2 = 0;
  void* image = nullptr;
  std::string name = "str";
  uint32_t u_value = 0;
  float ms = 0;
  float ms_2 = 0;
  int* value = 0;
  int* value_2 = 0;
  GLuint gl_uint = 0;
  GLenum gl_enum = 0;
#if defined(_WIN32)
  unsigned long long ull = 0;
#else
  unsigned long ull = 0;
#endif
  unsigned long long ull_2 = 0;
  // CHECK: hipDevice_t device;
  // CHECK-NEXT: hipCtx_t context;
  // CHECK-NEXT: hipFuncCache_t func_cache;
  // CHECK-NEXT: hipLimit_t limit;
  // CHECK-NEXT: hipSharedMemConfig pconfig;
  // CHECK-NEXT: hipFunction_t function;
  // CHECK-NEXT: hipFunction_attribute function_attribute;
  // CHECK-NEXT: hipModule_t module_;
  // CHECK-NEXT: hipDeviceptr_t deviceptr;
  // CHECK-NEXT: hipDeviceptr_t deviceptr_2;
  // CHECK-NEXT: hipTexRef texref;
  // CHECK-NEXT: hipJitOption jit_option;
  // CHECK-NEXT: hipArray_t array_;
  // CHECK-NEXT: HIP_ARRAY3D_DESCRIPTOR ARRAY3D_DESCRIPTOR;
  // CHECK-NEXT: HIP_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  // CHECK-NEXT: hipIpcEventHandle_t ipcEventHandle;
  // CHECK-NEXT: hipEvent_t event_;
  // CHECK-NEXT: hipEvent_t event_start;
  // CHECK-NEXT: hipEvent_t event_end;
  // CHECK-NEXT: hipIpcMemHandle_t ipcMemHandle;
  // CHECK-NEXT: hip_Memcpy2D MEMCPY2D;
  // CHECK-NEXT: HIP_MEMCPY3D MEMCPY3D;
  // CHECK-NEXT: hipStream_t stream;
  // CHECK-NEXT: hipMipmappedArray_t mipmappedArray;
  // CHECK-NEXT: hipStreamCallback_t streamCallback;
  // CHECK-NEXT: hipPointer_attribute pointer_attribute;
  // CHECK-NEXT: void* occupancyB2DSize;
  // CHECK-NEXT: hipGraphicsResource_t graphicsResource;
  // CHECK-NEXT: hipUUID uuid;
  CUdevice device;
  CUcontext context;
  CUfunc_cache func_cache;
  CUlimit limit;
  CUsharedconfig pconfig;
  CUfunction function;
  CUfunction_attribute function_attribute;
  CUmodule module_;
  CUdeviceptr deviceptr;
  CUdeviceptr deviceptr_2;
  CUtexref texref;
  CUjit_option jit_option;
  CUarray array_;
  CUDA_ARRAY3D_DESCRIPTOR ARRAY3D_DESCRIPTOR;
  CUDA_ARRAY_DESCRIPTOR ARRAY_DESCRIPTOR;
  CUipcEventHandle ipcEventHandle;
  CUevent event_;
  CUevent event_start;
  CUevent event_end;
  CUipcMemHandle ipcMemHandle;
  CUDA_MEMCPY2D MEMCPY2D;
  CUDA_MEMCPY3D MEMCPY3D;
  CUstream stream;
  CUmipmappedArray mipmappedArray;
  CUstreamCallback streamCallback;
  CUpointer_attribute pointer_attribute;
  CUoccupancyB2DSize occupancyB2DSize;
  CUgraphicsResource graphicsResource;
  CUuuid uuid;

  // CHECK: hipError_t result;
  CUresult result;

  unsigned int gridDimX = 0, gridDimY = 0, gridDimZ = 0, blockDimX = 0, blockDimY = 0, blockDimZ = 0, sharedMemBytes = 0;
  void* kernelParams = nullptr, * extra = nullptr;

  // CUDA: CUresult CUDAAPI cuInit(unsigned int Flags);
  // HIP: hipError_t hipInit(unsigned int flags);
  // CHECK: result = hipInit(flags);
  result = cuInit(flags);

  int driverVersion = 0;
  // CUDA: CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);
  // HIP: hipError_t hipDriverGetVersion(int* driverVersion);
  // CHECK: result = hipDriverGetVersion(&driverVersion);
  result = cuDriverGetVersion(&driverVersion);

  int ordinal = 0;
  // CUDA: CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal);
  // HIP: hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);
  // CHECK: result = hipDeviceGet(&device, ordinal);
  result = cuDeviceGet(&device, ordinal);

  int pi = 0;
  // CHECK: hipDeviceAttribute_t device_attribute = hipDeviceAttributePciBusId;
  CUdevice_attribute device_attribute = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
  // CUDA: CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
  // HIP: hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId);
  // CHECK: result = hipDeviceGetAttribute(&pi, device_attribute, device);
  result = cuDeviceGetAttribute(&pi, device_attribute, device);

  // CUDA: CUresult CUDAAPI cuDeviceGetCount(int *count);
  // HIP: hipError_t hipGetDeviceCount(int* count);
  // CHECK: result = hipGetDeviceCount(&count);
  result = cuDeviceGetCount(&count);

  // CUDA: CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev);
  // HIP: hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);
  // CHECK: result = hipDeviceTotalMem(&bytes, device);
  // CHECK-NEXT: result = hipDeviceTotalMem(&bytes, device);
  result = cuDeviceTotalMem(&bytes, device);
  result = cuDeviceTotalMem_v2(&bytes, device);

  int major = 0, minor = 0;
  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuDeviceComputeCapability(int *major, int *minor, CUdevice dev);
  // HIP: hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);
  // CHECK: result = hipDeviceComputeCapability(&major, &minor, device);
  result = cuDeviceComputeCapability(&major, &minor, device);

  int active = 0;
  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active);
  // HIP: hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active);
  // CHECK: result = hipDevicePrimaryCtxGetState(device, &flags, &active);
  result = cuDevicePrimaryCtxGetState(device, &flags, &active);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxRelease(device);
  result = cuDevicePrimaryCtxRelease(device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset(device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxRetain(&context, device);
  result = cuDevicePrimaryCtxRetain(&context, device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
  // HIP: hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
  // CHECK: result = hipDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags(device, flags);

  // CUDA: CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxCreate(hipCtx_t *ctx, unsigned int flags, hipDevice_t device);
  // CHECK: result = hipCtxCreate(&context, flags, device);
  // CHECK-NEXT: result = hipCtxCreate(&context, flags, device);
  result = cuCtxCreate(&context, flags, device);
  result = cuCtxCreate_v2(&context, flags, device);

  // CUDA: CUresult CUDAAPI cuCtxDestroy(CUcontext ctx);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxDestroy(hipCtx_t ctx);
  // CHECK: result = hipCtxDestroy(context);
  // CHECK-NEXT: result = hipCtxDestroy(context);
  result = cuCtxDestroy(context);
  result = cuCtxDestroy_v2(context);

  unsigned int version = 0;
  // CUDA: CUresult CUDAAPI cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);
  // CHECK: result = hipCtxGetApiVersion(context, &version);
  result = cuCtxGetApiVersion(context, &version);

  // CUDA: CUresult CUDAAPI cuCtxGetCacheConfig(CUfunc_cache *pconfig);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig);
  // CHECK: result = hipCtxGetCacheConfig(&func_cache);
  result = cuCtxGetCacheConfig(&func_cache);

  // CUDA: CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetCurrent(hipCtx_t* ctx);
  // CHECK: result = hipCtxGetCurrent(&context);
  result = cuCtxGetCurrent(&context);

  // CUDA: CUresult CUDAAPI cuCtxGetDevice(CUdevice *device);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetDevice(hipDevice_t* device);
  // CHECK: result = hipCtxGetDevice(&device);
  result = cuCtxGetDevice(&device);

  // CUDA: CUresult CUDAAPI cuCtxGetFlags(unsigned int *flags);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetFlags(unsigned int* flags);
  // CHECK: result = hipCtxGetFlags(&flags);
  result = cuCtxGetFlags(&flags);

  size_t pvalue = 0;
  // CUDA: CUresult CUDAAPI cuCtxGetLimit(size_t *pvalue, CUlimit limit);
  // HIP: hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit);
  // CHECK: result = hipDeviceGetLimit(&pvalue, limit);
  result = cuCtxGetLimit(&pvalue, limit);

  // CUDA: CUresult CUDAAPI cuCtxGetSharedMemConfig(CUsharedconfig *pConfig);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);
  // CHECK: result = hipCtxGetSharedMemConfig(&pconfig);
  result = cuCtxGetSharedMemConfig(&pconfig);

  int leastPriority = 0, greatestPriority = 0;
  // CUDA: CUresult CUDAAPI cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority);
  // HIP: hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
  // CHECK: result = hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  result = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);

  // CUDA: CUresult CUDAAPI cuCtxPopCurrent(CUcontext *pctx);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxPopCurrent(hipCtx_t* ctx);
  // CHECK: result = hipCtxPopCurrent(&context);
  // CHECK-NEXT: result = hipCtxPopCurrent(&context);
  result = cuCtxPopCurrent(&context);
  result = cuCtxPopCurrent_v2(&context);

  // CUDA: CUresult CUDAAPI cuCtxPushCurrent(CUcontext ctx);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxPushCurrent(hipCtx_t ctx);
  // CHECK: result = hipCtxPushCurrent(context);
  // CHECK-NEXT: result = hipCtxPushCurrent(context);
  result = cuCtxPushCurrent(context);
  result = cuCtxPushCurrent_v2(context);

  // CUDA: CUresult CUDAAPI cuCtxSetCacheConfig(CUfunc_cache config);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);
  // CHECK: result = hipCtxSetCacheConfig(func_cache);
  result = cuCtxSetCacheConfig(func_cache);

  // CUDA: CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxSetCurrent(hipCtx_t ctx);
  // CHECK: result = hipCtxSetCurrent(context);
  result = cuCtxSetCurrent(context);

  // CUDA: CUresult CUDAAPI cuCtxSetSharedMemConfig(CUsharedconfig config);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);
  // CHECK: result = hipCtxSetSharedMemConfig(pconfig);
  result = cuCtxSetSharedMemConfig(pconfig);

  // CUDA: CUresult CUDAAPI cuCtxSynchronize(void);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxSynchronize(void);
  // CHECK: result = hipCtxSynchronize();
  result = cuCtxSynchronize();

  // CUDA: CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);
  // HIP: hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname);
  // CHECK: result = hipModuleGetFunction(&function, module_, name.c_str());
  result = cuModuleGetFunction(&function, module_, name.c_str());

  // CUDA: CUresult CUDAAPI cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
  // HIP: hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char* name);
  // CHECK: result = hipModuleGetGlobal(&deviceptr, &bytes, module_, name.c_str());
  // CHECK-NEXT: result = hipModuleGetGlobal(&deviceptr, &bytes, module_, name.c_str());
  result = cuModuleGetGlobal(&deviceptr, &bytes, module_, name.c_str());
  result = cuModuleGetGlobal_v2(&deviceptr, &bytes, module_, name.c_str());

  // CUDA: CUresult CUDAAPI cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name);
  // HIP: hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name);
  // CHECK: result = hipModuleGetTexRef(&texref, module_, name.c_str());
  result = cuModuleGetTexRef(&texref, module_, name.c_str());

  // CUDA: CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname);
  // HIP: hipError_t hipModuleLoad(hipModule_t* module, const char* fname);
  // CHECK: result = hipModuleLoad(&module_, name.c_str());
  result = cuModuleLoad(&module_, name.c_str());

  // CUDA: CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image);
  // HIP: hipError_t hipModuleLoadData(hipModule_t* module, const void* image);
  // CHECK: result = hipModuleLoadData(&module_, image);
  result = cuModuleLoadData(&module_, image);

  unsigned int numOptions = 0;
  void* optionValues = nullptr;
  // CUDA: CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
  // HIP: hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image, unsigned int numOptions, hipJitOption* options, void** optionValues);
  // CHECK: result = hipModuleLoadDataEx(&module_, image, numOptions, &jit_option, &optionValues);
  result = cuModuleLoadDataEx(&module_, image, numOptions, &jit_option, &optionValues);

  // CUDA: CUresult CUDAAPI cuModuleUnload(CUmodule hmod);
  // HIP: hipError_t hipModuleUnload(hipModule_t module);
  // CHECK: result = hipModuleUnload(module_);
  result = cuModuleUnload(module_);

  // CUDA: CUresult CUDAAPI cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray);
  // HIP: hipError_t hipArray3DCreate(hipArray** array, const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray);
  // CHECK: result = hipArray3DCreate(&array_, &ARRAY3D_DESCRIPTOR);
  // CHECK-NEXT: result = hipArray3DCreate(&array_, &ARRAY3D_DESCRIPTOR);
  result = cuArray3DCreate(&array_, &ARRAY3D_DESCRIPTOR);
  result = cuArray3DCreate_v2(&array_, &ARRAY3D_DESCRIPTOR);

  // CUDA: CUresult CUDAAPI cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
  // HIP: hipError_t hipArrayCreate(hipArray** pHandle, const HIP_ARRAY_DESCRIPTOR* pAllocateArray);
  // CHECK: result = hipArrayCreate(&array_, &ARRAY_DESCRIPTOR);
  // CHECK: result = hipArrayCreate(&array_, &ARRAY_DESCRIPTOR);
  result = cuArrayCreate(&array_, &ARRAY_DESCRIPTOR);
  result = cuArrayCreate_v2(&array_, &ARRAY_DESCRIPTOR);

  // CUDA: CUresult CUDAAPI cuArrayDestroy(CUarray hArray);
  // HIP: hipError_t hipArrayDestroy(hipArray* array);
  // CHECK: result = hipArrayDestroy(array_);
  result = cuArrayDestroy(array_);

  std::string pciBusId;
  // CUDA: CUresult CUDAAPI cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId);
  // HIP: hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);
  // CHECK: result = hipDeviceGetByPCIBusId(&device, pciBusId.c_str());
  result = cuDeviceGetByPCIBusId(&device, pciBusId.c_str());

  int len = 0;
  char* pciBusId_ = const_cast<char*>(pciBusId.c_str());
  // CUDA: CUresult CUDAAPI cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev);
  // HIP: hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, int device);
  // CHECK: result = hipDeviceGetPCIBusId(pciBusId_, len, device);
  result = cuDeviceGetPCIBusId(pciBusId_, len, device);

  // CUDA: CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr dptr);
  // HIP: hipError_t hipIpcCloseMemHandle(void* devPtr);
  // CHECK: result = hipIpcCloseMemHandle(deviceptr);
  result = cuIpcCloseMemHandle(deviceptr);

  // CUDA: CUresult CUDAAPI cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event);
  // HIP: hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);
  // CHECK: result = hipIpcGetEventHandle(&ipcEventHandle, event_);
  result = cuIpcGetEventHandle(&ipcEventHandle, event_);

  // CUDA: CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr);
  // HIP: hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
  // CHECK: result = hipIpcGetMemHandle(&ipcMemHandle, deviceptr);
  result = cuIpcGetMemHandle(&ipcMemHandle, deviceptr);

  // CUDA: CUresult CUDAAPI cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle);
  // HIP: hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);
  // CHECK: result = hipIpcOpenEventHandle(&event_, ipcEventHandle);
  result = cuIpcOpenEventHandle(&event_, ipcEventHandle);

  // CUDA: CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags);
  // HIP: hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags);
  // CHECK: result = hipIpcOpenMemHandle(&deviceptr, ipcMemHandle, flags);
  result = cuIpcOpenMemHandle(&deviceptr, ipcMemHandle, flags);

  // CUDA: CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
  // HIP: hipError_t hipMalloc(void** ptr, size_t size);
  // CHECK: result = hipMalloc(&deviceptr, bytes);
  // CHECK-NEXT: result = hipMalloc(&deviceptr, bytes);
  result = cuMemAlloc(&deviceptr, bytes);
  result = cuMemAlloc_v2(&deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize);
  // HIP: DEPRECATED("use hipHostMalloc instead") hipError_t hipMemAllocHost(void** ptr, size_t size);
  // CHECK: result = hipMemAllocHost(&image, bytes);
  // CHECK-NEXT: result = hipMemAllocHost(&image, bytes);
  result = cuMemAllocHost(&image, bytes);
  result = cuMemAllocHost_v2(&image, bytes);

  // CUDA: CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
  // HIP: hipError_t hipMallocManaged(void** dev_ptr, size_t size, unsigned int flags __dparm(hipMemAttachGlobal));
  // CHECK: result = hipMallocManaged(&deviceptr, bytes, flags);
  result = cuMemAllocManaged(&deviceptr, bytes, flags);

  size_t pitch = 0, width = 0, height = 0;
  // CUDA: CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
  // HIP: hipError_t hipMemAllocPitch(hipDeviceptr_t* dptr, size_t* pitch, size_t widthInBytes, size_t height, unsigned int elementSizeBytes);
  // CHECK: result = hipMemAllocPitch(&deviceptr, &pitch, width, height, bytes);
  // CHECK-NEXT: result = hipMemAllocPitch(&deviceptr, &pitch, width, height, bytes);
  result = cuMemAllocPitch(&deviceptr, &pitch, width, height, bytes);
  result = cuMemAllocPitch_v2(&deviceptr, &pitch, width, height, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpy2D(const CUDA_MEMCPY2D *pCopy);
  // HIP: hipError_t hipMemcpyParam2D(const hip_Memcpy2D* pCopy);
  // CHECK: result = hipMemcpyParam2D(&MEMCPY2D);
  // CHECK-NEXT: result = hipMemcpyParam2D(&MEMCPY2D);
  result = cuMemcpy2D(&MEMCPY2D);
  result = cuMemcpy2D_v2(&MEMCPY2D);

  // CUDA: CUresult CUDAAPI cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream);
  // HIP: hipError_t hipMemcpyParam2DAsync(const hip_Memcpy2D* pCopy, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemcpyParam2DAsync(&MEMCPY2D, stream);
  // CHECK-NEXT: result = hipMemcpyParam2DAsync(&MEMCPY2D, stream);
  result = cuMemcpy2DAsync(&MEMCPY2D, stream);
  result = cuMemcpy2DAsync_v2(&MEMCPY2D, stream);

  // CUDA: CUresult CUDAAPI cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy);
  // HIP: hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D* pCopy);
  // CHECK: result = hipDrvMemcpy2DUnaligned(&MEMCPY2D);
  // CHECK-NEXT: result = hipDrvMemcpy2DUnaligned(&MEMCPY2D);
  result = cuMemcpy2DUnaligned(&MEMCPY2D);
  result = cuMemcpy2DUnaligned_v2(&MEMCPY2D);

  // CUDA: CUresult CUDAAPI cuMemcpy3D(const CUDA_MEMCPY3D *pCopy);
  // HIP: hipError_t hipDrvMemcpy3D(const HIP_MEMCPY3D* pCopy);
  // CHECK: result = hipDrvMemcpy3D(&MEMCPY3D);
  // CHECK-NEXT: result = hipDrvMemcpy3D(&MEMCPY3D);
  result = cuMemcpy3D(&MEMCPY3D);
  result = cuMemcpy3D_v2(&MEMCPY3D);

  // CUDA: CUresult CUDAAPI cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream);
  // HIP: hipError_t hipDrvMemcpy3DAsync(const HIP_MEMCPY3D* pCopy, hipStream_t stream);
  // CHECK: result = hipDrvMemcpy3DAsync(&MEMCPY3D, stream);
  // CHECK-NEXT: result = hipDrvMemcpy3DAsync(&MEMCPY3D, stream);
  result = cuMemcpy3DAsync(&MEMCPY3D, stream);
  result = cuMemcpy3DAsync_v2(&MEMCPY3D, stream);

  void* dsthost = nullptr;
  size_t offset = 0;
  // CUDA: CUresult CUDAAPI cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
  // HIP: hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset, size_t count);
  // CHECK: result = hipMemcpyAtoH(dsthost, array_, offset, bytes);
  // CHECK-NEXT: result = hipMemcpyAtoH(dsthost, array_, offset, bytes);
  result = cuMemcpyAtoH(dsthost, array_, offset, bytes);
  result = cuMemcpyAtoH_v2(dsthost, array_, offset, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
  // HIP: hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes);
  // CHECK: result = hipMemcpyDtoD(deviceptr, deviceptr, bytes);
  // CHECK-NEXT: result = hipMemcpyDtoD(deviceptr, deviceptr, bytes);
  result = cuMemcpyDtoD(deviceptr, deviceptr, bytes);
  result = cuMemcpyDtoD_v2(deviceptr, deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  // HIP: hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
  // CHECK: result = hipMemcpyDtoDAsync(deviceptr, deviceptr, bytes, stream);
  // CHECK-NEXT: result = hipMemcpyDtoDAsync(deviceptr, deviceptr, bytes, stream);
  result = cuMemcpyDtoDAsync(deviceptr, deviceptr, bytes, stream);
  result = cuMemcpyDtoDAsync_v2(deviceptr, deviceptr, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  // HIP: hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
  // CHECK: result = hipMemcpyDtoH(dsthost, deviceptr, bytes);
  // CHECK-NEXT: result = hipMemcpyDtoH(dsthost, deviceptr, bytes);
  result = cuMemcpyDtoH(dsthost, deviceptr, bytes);
  result = cuMemcpyDtoH_v2(dsthost, deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  // HIP: hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t sizeBytes, hipStream_t stream);
  // CHECK: result = hipMemcpyDtoHAsync(dsthost, deviceptr, bytes, stream);
  // CHECK-NEXT: result = hipMemcpyDtoHAsync(dsthost, deviceptr, bytes, stream);
  result = cuMemcpyDtoHAsync(dsthost, deviceptr, bytes, stream);
  result = cuMemcpyDtoHAsync_v2(dsthost, deviceptr, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
  // HIP: hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost, size_t count);
  // CHECK: result = hipMemcpyHtoA(array_, offset, dsthost, bytes);
  // CHECK-NEXT: result = hipMemcpyHtoA(array_, offset, dsthost, bytes);
  result = cuMemcpyHtoA(array_, offset, dsthost, bytes);
  result = cuMemcpyHtoA_v2(array_, offset, dsthost, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  // HIP: hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t sizeBytes);
  // CHECK: result = hipMemcpyHtoD(deviceptr, dsthost, bytes);
  // CHECK-NEXT: result = hipMemcpyHtoD(deviceptr, dsthost, bytes);
  result = cuMemcpyHtoD(deviceptr, dsthost, bytes);
  result = cuMemcpyHtoD_v2(deviceptr, dsthost, bytes);

  // CUDA: CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
  // HIP: hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t sizeBytes, hipStream_t stream);
  // CHECK: result = hipMemcpyHtoDAsync(deviceptr, dsthost, bytes, stream);
  // CHECK-NEXT: result = hipMemcpyHtoDAsync(deviceptr, dsthost, bytes, stream);
  result = cuMemcpyHtoDAsync(deviceptr, dsthost, bytes, stream);
  result = cuMemcpyHtoDAsync_v2(deviceptr, dsthost, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMemFree(CUdeviceptr dptr);
  // HIP: hipError_t hipFree(void* ptr);
  // CHECK: result = hipFree(deviceptr);
  // CHECK-NEXT: result = hipFree(deviceptr);
  result = cuMemFree(deviceptr);
  result = cuMemFree_v2(deviceptr);

  // CUDA: CUresult CUDAAPI cuMemFreeHost(void *p);
  // HIP: hipError_t hipHostFree(void* ptr);
  // CHECK: result = hipHostFree(image);
  result = cuMemFreeHost(image);

  // CUDA: CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr);
  // HIP: hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
  // CHECK: result = hipMemGetAddressRange(&deviceptr, &bytes, deviceptr_2);
  // CHECK-NEXT: result = hipMemGetAddressRange(&deviceptr, &bytes, deviceptr_2);
  result = cuMemGetAddressRange(&deviceptr, &bytes, deviceptr_2);
  result = cuMemGetAddressRange_v2(&deviceptr, &bytes, deviceptr_2);

  // CUDA: CUresult CUDAAPI cuMemGetInfo(size_t *free, size_t *total);
  // HIP: hipError_t hipMemGetInfo(size_t* free, size_t* total);
  // CHECK: result = hipMemGetInfo(&bytes, &bytes_2);
  // CHECK-NEXT: result = hipMemGetInfo(&bytes, &bytes_2);
  result = cuMemGetInfo(&bytes, &bytes_2);
  result = cuMemGetInfo_v2(&bytes, &bytes_2);

  // CUDA: CUresult CUDAAPI cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags);
  // HIP: DEPRECATED("use hipHostMalloc instead") hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
  // CHECK: result = hipHostAlloc(&image, bytes, flags);
  result = cuMemHostAlloc(&image, bytes, flags);

  // CUDA: CUresult CUDAAPI cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);
  // HIP: hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
  // CHECK: result = hipHostGetDevicePointer(&deviceptr, image, flags);
  // CHECK-NEXT: result = hipHostGetDevicePointer(&deviceptr, image, flags);
  result = cuMemHostGetDevicePointer(&deviceptr, image, flags);
  result = cuMemHostGetDevicePointer_v2(&deviceptr, image, flags);

  // CUDA: CUresult CUDAAPI cuMemHostGetFlags(unsigned int *pFlags, void *p);
  // HIP: hipError_t hipHostGetFlags(&flags, image);
  // CHECK: result = hipHostGetFlags(&flags, image);
  result = cuMemHostGetFlags(&flags, image);

  // CUDA: CUresult CUDAAPI cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags);
  // HIP: hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
  // CHECK: result = hipHostRegister(image, bytes, flags);
  // CHECK-NEXT: result = hipHostRegister(image, bytes, flags);
  result = cuMemHostRegister(image, bytes, flags);
  result = cuMemHostRegister_v2(image, bytes, flags);

  // CUDA: CUresult CUDAAPI cuMemHostUnregister(void *p);
  // HIP: hipError_t hipHostUnregister(void* hostPtr);
  // CHECK: result = hipHostUnregister(image);
  result = cuMemHostUnregister(image);

  unsigned short us = 0;
  // CUDA: CUresult CUDAAPI cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N);
  // HIP: hipError_t hipMemsetD16(hipDeviceptr_t dest, unsigned short value, size_t count);
  // CHECK: result = hipMemsetD16(deviceptr, us, bytes);
  // CHECK-NEXT: result = hipMemsetD16(deviceptr, us, bytes);
  result = cuMemsetD16(deviceptr, us, bytes);
  result = cuMemsetD16_v2(deviceptr, us, bytes);

  // CUDA: CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
  // HIP: hipError_t hipMemsetD16Async(hipDeviceptr_t dest, unsigned short value, size_t count, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemsetD16Async(deviceptr, us, bytes, stream);
  result = cuMemsetD16Async(deviceptr, us, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N)
  // HIP: hipError_t hipMemsetD32(hipDeviceptr_t dest, int value, size_t count);
  // CHECK: result = hipMemsetD32(deviceptr, flags, bytes);
  // CHECK-NEXT: result = hipMemsetD32(deviceptr, flags, bytes);
  result = cuMemsetD32(deviceptr, flags, bytes);
  result = cuMemsetD32_v2(deviceptr, flags, bytes);

  // CUDA: CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
  // HIP: hipError_t hipMemsetD32Async(hipDeviceptr_t dst, int value, size_t count, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemsetD32Async(deviceptr, flags, bytes, stream);
  result = cuMemsetD32Async(deviceptr, flags, bytes, stream);

  unsigned char uc = 0;
  // CUDA: CUresult CUDAAPI cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N);
  // HIP: hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t count);
  // CHECK: result = hipMemsetD8(deviceptr, uc, bytes);
  // CHECK-NEXT: result = hipMemsetD8(deviceptr, uc, bytes);
  result = cuMemsetD8(deviceptr, uc, bytes);
  result = cuMemsetD8_v2(deviceptr, uc, bytes);

  // CUDA: CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
  // HIP: hipError_t hipMemsetD8Async(hipDeviceptr_t dest, unsigned char value, size_t count, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemsetD8Async(deviceptr, uc, bytes, stream);
  result = cuMemsetD8Async(deviceptr, uc, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels);
  // HIP: hipError_t hipMipmappedArrayCreate(hipMipmappedArray_t* pHandle, HIP_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int numMipmapLevels);
  // CHECK: result = hipMipmappedArrayCreate(&mipmappedArray, &ARRAY3D_DESCRIPTOR, flags);
  result = cuMipmappedArrayCreate(&mipmappedArray, &ARRAY3D_DESCRIPTOR, flags);

  // CUDA: CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray);
  // HIP: hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);
  // CHECK: result = hipMipmappedArrayDestroy(mipmappedArray);
  result = cuMipmappedArrayDestroy(mipmappedArray);

  // CUDA: CUresult CUDAAPI cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
  // HIP: hipError_t hipMipmappedArrayGetLevel(hipArray_t* pLevelArray, hipMipmappedArray_t hMipMappedArray, unsigned int level);
  // CHECK: result = hipMipmappedArrayGetLevel(&array_, mipmappedArray, flags);
  result = cuMipmappedArrayGetLevel(&array_, mipmappedArray, flags);

  // CUDA: CUresult CUDAAPI cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr);
  // HIP: hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr);
  // CHECK: result = hipPointerGetAttribute(image, pointer_attribute, deviceptr);
  result = cuPointerGetAttribute(image, pointer_attribute, deviceptr);

  // CUDA: CUresult CUDAAPI cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr);
  // HIP: hipError_t hipDrvPointerGetAttributes(unsigned int numAttributes, hipPointer_attribute* attributes, void** data, hipDeviceptr_t ptr);
  // CHECK: result = hipDrvPointerGetAttributes(flags, &pointer_attribute, &image, deviceptr);
  result = cuPointerGetAttributes(flags, &pointer_attribute, &image, deviceptr);

  // CUDA: CUresult CUDAAPI cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags);
  // HIP: hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData, unsigned int flags);
  // CHECK: result = hipStreamAddCallback(stream, streamCallback, image, flags);
  result = cuStreamAddCallback(stream, streamCallback, image, flags);

  // CUDA: CUresult CUDAAPI cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
  // HIP: hipError_t hipStreamAttachMemAsync(hipStream_t stream, void* dev_ptr, size_t length __dparm(0), unsigned int flags __dparm(hipMemAttachSingle));
  // CHECK: result = hipStreamAttachMemAsync(stream, deviceptr, bytes, flags);
  result = cuStreamAttachMemAsync(stream, deviceptr, bytes, flags);

  // CUDA: CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags);
  // HIP: hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);
  // CHECK: result = hipStreamCreateWithFlags(&stream, flags);
  result = cuStreamCreate(&stream, flags);

  // CUDA: CUresult CUDAAPI cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority);
  // HIP: hipError_t hipStreamCreateWithPriority(hipStream_t* stream, unsigned int flags, int priority);
  // CHECK: result = hipStreamCreateWithPriority(&stream, flags, leastPriority);
  result = cuStreamCreateWithPriority(&stream, flags, leastPriority);

  // CUDA: CUresult CUDAAPI cuStreamDestroy(CUstream hStream);
  // HIP: hipError_t hipStreamDestroy(hipStream_t stream);
  // CHECK: result = hipStreamDestroy(stream);
  // CHECK-NEXT: result = hipStreamDestroy(stream);
  result = cuStreamDestroy(stream);
  result = cuStreamDestroy_v2(stream);

  // CUDA: CUresult CUDAAPI cuStreamGetFlags(CUstream hStream, unsigned int *flags);
  // HIP: hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int* flags);
  // CHECK: result = hipStreamGetFlags(stream, &flags);
  result = cuStreamGetFlags(stream, &flags);

  // CUDA: CUresult CUDAAPI cuStreamGetPriority(CUstream hStream, int *priority);
  // HIP: hipError_t hipStreamGetPriority(hipStream_t stream, int* priority);
  // CHECK: result = hipStreamGetPriority(stream, &leastPriority);
  result = cuStreamGetPriority(stream, &leastPriority);

  // CUDA: CUresult CUDAAPI cuStreamQuery(CUstream hStream);
  // HIP: hipError_t hipStreamQuery(hipStream_t stream);
  // CHECK: result = hipStreamQuery(stream);
  result = cuStreamQuery(stream);

  // CUDA: CUresult CUDAAPI cuStreamSynchronize(CUstream hStream);
  // HIP: hipError_t hipStreamSynchronize(hipStream_t stream);
  // CHECK: result = hipStreamSynchronize(stream);
  result = cuStreamSynchronize(stream);

  // CUDA: CUresult CUDAAPI cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags);
  // HIP: hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags);
  // CHECK: result = hipStreamWaitEvent(stream, event_, flags);
  result = cuStreamWaitEvent(stream, event_, flags);

  // CUDA: CUresult CUDAAPI cuEventCreate(CUevent *phEvent, unsigned int Flags);
  // HIP: hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags);
  // CHECK: result = hipEventCreateWithFlags(&event_, flags);
  result = cuEventCreate(&event_, flags);

  // CUDA: CUresult CUDAAPI cuEventDestroy(CUevent hEvent);
  // HIP: hipError_t hipEventDestroy(hipEvent_t event);
  // CHECK: result = hipEventDestroy(event_);
  // CHECK-NEXT: result = hipEventDestroy(event_);
  result = cuEventDestroy(event_);
  result = cuEventDestroy_v2(event_);

  // CUDA: CUresult CUDAAPI cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
  // HIP: hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);
  // CHECK: result = hipEventElapsedTime(&ms, event_start, event_end);
  result = cuEventElapsedTime(&ms, event_start, event_end);

  // CUDA: CUresult CUDAAPI cuEventRecord(CUevent hEvent, CUstream hStream);
  // HIP: hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream);
  // CHECK: result = hipEventRecord(event_, stream);
  result = cuEventRecord(event_, stream);

  // CUDA: CUresult CUDAAPI cuEventSynchronize(CUevent hEvent);
  // HIP: hipError_t hipEventSynchronize(hipEvent_t event);
  // CHECK: result = hipEventSynchronize(event_);
  result = cuEventSynchronize(event_);

  // CUDA: CUresult CUDAAPI cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc);
  // HIP: hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc);
  // CHECK: result = hipFuncGetAttribute(value, function_attribute, function);
  result = cuFuncGetAttribute(value, function_attribute, function);

  // CUDA: CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);
  // HIP: hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams, void** extra);
  // CHECK: result = hipModuleLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, &kernelParams, &extra);
  result = cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, &kernelParams, &extra);

  // CUDA: CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
  // HIP: hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk);
  // CHECK: result = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(value, function, iBlockSize, bytes);
  result = cuOccupancyMaxActiveBlocksPerMultiprocessor(value, function, iBlockSize, bytes);

  // CUDA: CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
  // HIP: hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit);
  // CHECK: result = hipModuleOccupancyMaxPotentialBlockSize(value, value_2, function, bytes, iBlockSize);
  result = cuOccupancyMaxPotentialBlockSize(value, value_2, function, occupancyB2DSize, bytes, iBlockSize);

  // CUDA: CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags);
  // HIP: hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize, hipFunction_t f, size_t dynSharedMemPerBlk, int blockSizeLimit, unsigned int flags);
  // CHECK: result = hipModuleOccupancyMaxPotentialBlockSizeWithFlags(value, value_2, function, bytes, iBlockSize, iBlockSize_2);
  result = cuOccupancyMaxPotentialBlockSizeWithFlags(value, value_2, function, occupancyB2DSize, bytes, iBlockSize, iBlockSize_2);

  // CUDA: CUresult CUDAAPI cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);
  // HIP: hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
  // CHECK: result = hipGraphicsGLRegisterImage(&graphicsResource, gl_uint, gl_enum, flags);
  result = cuGraphicsGLRegisterImage(&graphicsResource, gl_uint, gl_enum, flags);

  // CUDA: CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
  // HIP: hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
  // CHECK: result = hipGraphicsSubResourceGetMappedArray(&array_, graphicsResource, flags, flags_2);
  result = cuGraphicsSubResourceGetMappedArray(&array_, graphicsResource, flags, flags_2);

  // CUDA: CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
  // HIP: hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk, unsigned int flags);
  // CHECK: result = hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&iBlockSize, function, iBlockSize_2, bytes, flags);
  result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&iBlockSize, function, iBlockSize_2, bytes, flags);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetAddress(hipDeviceptr_t* dev_ptr, const textureReference* texRef);
  // CHECK: result = hipTexRefGetAddress(&deviceptr, texref);
  // CHECK-NEXT: result = hipTexRefGetAddress(&deviceptr, texref);
  result = cuTexRefGetAddress(&deviceptr, texref);
  result = cuTexRefGetAddress_v2(&deviceptr, texref);

  // CHECK: HIPaddress_mode address_mode;
  CUaddress_mode address_mode;

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetAddressMode(enum hipTextureAddressMode* pam, const textureReference* texRef, int dim);
  // CHECK: result = hipTexRefGetAddressMode(&address_mode, texref, dim);
  result = cuTexRefGetAddressMode(&address_mode, texref, dim);

  // CHECK: HIPfilter_mode filter_mode;
  CUfilter_mode filter_mode;

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef);
  // CHECK: result = hipTexRefGetFilterMode(&filter_mode, texref);
  result = cuTexRefGetFilterMode(&filter_mode, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetFlags(unsigned int* pFlags, const textureReference* texRef);
  // CHECK: result = hipTexRefGetFlags(&flags, texref);
  result = cuTexRefGetFlags(&flags, texref);

  // CHECK: hipArray_Format array_format;
  CUarray_format array_format;

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetFormat(hipArray_Format* pFormat, int* pNumChannels, const textureReference* texRef);
  // CHECK: result = hipTexRefGetFormat(&array_format, &iBlockSize, texref);
  result = cuTexRefGetFormat(&array_format, &iBlockSize, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetMaxAnisotropy(int* pmaxAnsio, const textureReference* texRef);
  // CHECK: result = hipTexRefGetMaxAnisotropy(&iBlockSize, texref);
  result = cuTexRefGetMaxAnisotropy(&iBlockSize, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetMipmapFilterMode(enum hipTextureFilterMode* pfm, const textureReference* texRef);
  // CHECK: result = hipTexRefGetMipmapFilterMode(&filter_mode, texref);
  result = cuTexRefGetMipmapFilterMode(&filter_mode, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetMipmapLevelBias(float* pbias, const textureReference* texRef);
  // CHECK: result = hipTexRefGetMipmapLevelBias(&ms, texref);
  result = cuTexRefGetMipmapLevelBias(&ms, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetMipmapLevelClamp(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, const textureReference* texRef);
  // CHECK: result = hipTexRefGetMipmapLevelClamp(&ms, &ms_2, texref);
  result = cuTexRefGetMipmapLevelClamp(&ms, &ms_2, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefGetMipMappedArray(hipMipmappedArray_t* pArray, const textureReference* texRef);
  // CHECK: result = hipTexRefGetMipMappedArray(&mipmappedArray, texref);
  result = cuTexRefGetMipmappedArray(&mipmappedArray, texref);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
  // HIP: hipError_t hipTexRefSetAddress(size_t* ByteOffset, textureReference* texRef, hipDeviceptr_t dptr, size_t bytes);
  // CHECK: result = hipTexRefSetAddress(&bytes, texref, deviceptr, bytes_2);
  // CHECK-NEXT: result = hipTexRefSetAddress(&bytes, texref, deviceptr, bytes_2);
  result = cuTexRefSetAddress(&bytes, texref, deviceptr, bytes_2);
  result = cuTexRefSetAddress_v2(&bytes, texref, deviceptr, bytes_2);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefSetAddress2D(textureReference* texRef, const HIP_ARRAY_DESCRIPTOR* desc, hipDeviceptr_t dptr, size_t Pitch);
  // CHECK: result = hipTexRefSetAddress2D(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);
  // CHECK-NEXT: result = hipTexRefSetAddress2D(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);
  result = cuTexRefSetAddress2D(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);
  result = cuTexRefSetAddress2D_v3(texref, &ARRAY_DESCRIPTOR, deviceptr, bytes);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am);
  // HIP: hipError_t hipTexRefSetAddressMode(textureReference* texRef, int dim, enum hipTextureAddressMode am);
  // CHECK: result = hipTexRefSetAddressMode(texref, dim, address_mode);
  result = cuTexRefSetAddressMode(texref, dim, address_mode);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
  // HIP: hipError_t hipTexRefSetArray(textureReference* tex, hipArray_const_t array, unsigned int flags);
  // CHECK: result = hipTexRefSetArray(texref, array_, flags);
  result = cuTexRefSetArray(texref, array_, flags);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm);
  // HIP: hipError_t hipError_t hipTexRefSetFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
  // CHECK: result = hipTexRefSetFilterMode(texref, filter_mode);
  result = cuTexRefSetFilterMode(texref, filter_mode);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags);
  // HIP: hipError_t hipTexRefSetFlags(textureReference* texRef, unsigned int Flags);
  // CHECK: result = hipTexRefSetFlags(texref, flags);
  result = cuTexRefSetFlags(texref, flags);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
  // HIP: hipError_t hipTexRefSetFormat(textureReference* texRef, hipArray_Format fmt, int NumPackedComponents);
  // CHECK: result = hipTexRefSetFormat(texref, array_format, iBlockSize);
  result = cuTexRefSetFormat(texref, array_format, iBlockSize);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefSetMaxAnisotropy(textureReference* texRef, unsigned int maxAniso);
  // CHECK: result = hipTexRefSetMaxAnisotropy(texref, flags);
  result = cuTexRefSetMaxAnisotropy(texref, flags);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm);
  // HIP: hipError_t hipTexRefSetMipmapFilterMode(textureReference* texRef, enum hipTextureFilterMode fm);
  // CHECK: result = hipTexRefSetMipmapFilterMode(texref, filter_mode);
  result = cuTexRefSetMipmapFilterMode(texref, filter_mode);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias);
  // HIP: hipError_t hipTexRefSetMipmapLevelBias(textureReference* texRef, float bias);
  // CHECK: result = hipTexRefSetMipmapLevelBias(texref, ms);
  result = cuTexRefSetMipmapLevelBias(texref, ms);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
  // HIP: hipError_t hipTexRefSetMipmapLevelClamp(textureReference* texRef, float minMipMapLevelClamp, float maxMipMapLevelClamp);
  // CHECK: result = hipTexRefSetMipmapLevelClamp(texref, ms, ms_2);
  result = cuTexRefSetMipmapLevelClamp(texref, ms, ms_2);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
  // HIP: hipError_t hipTexRefSetMipmappedArray(textureReference* texRef, struct hipMipmappedArray* mipmappedArray, unsigned int Flags);
  // CHECK: result = hipTexRefSetMipmappedArray(texref, mipmappedArray, flags);
  result = cuTexRefSetMipmappedArray(texref, mipmappedArray, flags);

  // CHECK: hipTextureObject_t texObject;
  CUtexObject texObject;

  // CHECK: HIP_RESOURCE_DESC res_descr;
  CUDA_RESOURCE_DESC res_descr;

  // CHECK: HIP_TEXTURE_DESC tex_descr;
  CUDA_TEXTURE_DESC tex_descr;

  // CHECK: HIP_RESOURCE_VIEW_DESC res_view_descr;
  CUDA_RESOURCE_VIEW_DESC res_view_descr;

  // CUDA: CUresult CUDAAPI cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc);
  // HIP: hipError_t hipTexObjectCreate(hipTextureObject_t* pTexObject, const HIP_RESOURCE_DESC* pResDesc, const HIP_TEXTURE_DESC* pTexDesc, const HIP_RESOURCE_VIEW_DESC* pResViewDesc);
  // CHECK: result = hipTexObjectCreate(&texObject, &res_descr, &tex_descr, &res_view_descr);
  result = cuTexObjectCreate(&texObject, &res_descr, &tex_descr, &res_view_descr);

  // CUDA: CUresult CUDAAPI cuTexObjectDestroy(CUtexObject texObject);
  // HIP: hipError_t hipTexObjectDestroy(hipTextureObject_t texObject);
  // CHECK: result = hipTexObjectDestroy(texObject);
  result = cuTexObjectDestroy(texObject);

  // CUDA: CUresult CUDAAPI cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject);
  // HIP: hipError_t hipTexObjectGetResourceDesc(HIP_RESOURCE_DESC* pResDesc, hipTextureObject_t texObject);
  // CHECK: result = hipTexObjectGetResourceDesc(&res_descr, texObject);
  result = cuTexObjectGetResourceDesc(&res_descr, texObject);

  // CUDA: CUresult CUDAAPI cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject);
  // HIP: hipError_t hipTexObjectGetResourceViewDesc(HIP_RESOURCE_VIEW_DESC* pResViewDesc, hipTextureObject_t texObject);
  // CHECK: result = hipTexObjectGetResourceViewDesc(&res_view_descr, texObject);
  result = cuTexObjectGetResourceViewDesc(&res_view_descr, texObject);

  // CUDA: CUresult CUDAAPI cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject);
  // HIP: hipError_t hipTexObjectGetTextureDesc(HIP_TEXTURE_DESC* pTexDesc, hipTextureObject_t texObject);
  // CHECK: result = hipTexObjectGetTextureDesc(&tex_descr, texObject);
  result = cuTexObjectGetTextureDesc(&tex_descr, texObject);

  // CUDA: CUresult CUDAAPI cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);
  // CHECK: result = hipCtxEnablePeerAccess(context, flags);
  result = cuCtxEnablePeerAccess(context, flags);

  // CUDA: CUresult CUDAAPI cuCtxDisablePeerAccess(CUcontext peerContext);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);
  // CHECK: result = hipCtxDisablePeerAccess(context);
  result = cuCtxDisablePeerAccess(context);

  // CUDA: CUresult CUDAAPI cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev);
  // HIP: hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);
  // CHECK: result = hipDeviceCanAccessPeer(value, device, dim);
  result = cuDeviceCanAccessPeer(value, device, dim);

  // CUDA: CUresult CUDAAPI cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
  // HIP: hipError_t hipGraphicsMapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream __dparm(0));
  // CHECK: result = hipGraphicsMapResources(iBlockSize, &graphicsResource, stream);
  result = cuGraphicsMapResources(iBlockSize, &graphicsResource, stream);

  // CUDA: CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource);
  // HIP: hipError_t hipGraphicsResourceGetMappedPointer(void** devPtr, size_t* size, hipGraphicsResource_t resource);
  // CHECK: result = hipGraphicsResourceGetMappedPointer(&deviceptr, &bytes, graphicsResource);
  // CHECK-NEXT: result = hipGraphicsResourceGetMappedPointer(&deviceptr, &bytes, graphicsResource);
  result = cuGraphicsResourceGetMappedPointer(&deviceptr, &bytes, graphicsResource);
  result = cuGraphicsResourceGetMappedPointer_v2(&deviceptr, &bytes, graphicsResource);

  // CUDA: CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
  // HIP: hipError_t hipGraphicsSubResourceGetMappedArray(hipArray_t* array, hipGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
  // CHECK: result = hipGraphicsSubResourceGetMappedArray(&array_, graphicsResource, flags, flags_2);
  result = cuGraphicsSubResourceGetMappedArray(&array_, graphicsResource, flags, flags_2);

  // CUDA: CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream);
  // HIP: hipError_t hipGraphicsUnmapResources(int count, hipGraphicsResource_t* resources, hipStream_t stream __dparm(0));
  // CHECK: result = hipGraphicsUnmapResources(iBlockSize, &graphicsResource, stream);
  result = cuGraphicsUnmapResources(iBlockSize, &graphicsResource, stream);

  // CUDA: CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource);
  // HIP: hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);
  // CHECK: result = hipGraphicsUnregisterResource(graphicsResource);
  result = cuGraphicsUnregisterResource(graphicsResource);

  // CUDA: CUresult CUDAAPI cuProfilerStart(void);
  // HIP: DEPRECATED("use roctracer/rocTX instead") hipError_t hipProfilerStart();
  // CHECK: result = hipProfilerStart();
  result = cuProfilerStart();

  // CUDA: CUresult CUDAAPI cuProfilerStop(void);
  // HIP: DEPRECATED("use roctracer/rocTX instead") hipError_t hipProfilerStop();
  // CHECK: result = hipProfilerStop();
  result = cuProfilerStop();

  // CHECK: hipGLDeviceList GLDeviceList;
  CUGLDeviceList GLDeviceList;

  // CUDA: CUresult CUDAAPI cuGLGetDevices(unsigned int *pCudaDeviceCount, CUdevice *pCudaDevices, unsigned int cudaDeviceCount, CUGLDeviceList deviceList);
  // HIP: hipError_t hipGLGetDevices(unsigned int* pHipDeviceCount, int* pHipDevices, unsigned int hipDeviceCount, hipGLDeviceList deviceList);
  // CHECK: result = hipGLGetDevices(&flags, &device, flags_2, GLDeviceList);
  result = cuGLGetDevices(&flags, &device, flags_2, GLDeviceList);

  // CUDA: CUresult CUDAAPI cuGraphicsGLRegisterBuffer(CUgraphicsResource *pCudaResource, GLuint buffer, unsigned int Flags);
  // HIP: hipError_t hipGraphicsGLRegisterBuffer(hipGraphicsResource** resource, GLuint buffer, unsigned int flags);
  // CHECK: result = hipGraphicsGLRegisterBuffer(&graphicsResource, gl_uint, flags);
  result = cuGraphicsGLRegisterBuffer(&graphicsResource, gl_uint, flags);

  // CUDA: CUresult CUDAAPI cuGraphicsGLRegisterImage(CUgraphicsResource *pCudaResource, GLuint image, GLenum target, unsigned int Flags);
  // HIP: hipError_t hipGraphicsGLRegisterImage(hipGraphicsResource** resource, GLuint image, GLenum target, unsigned int flags);
  // CHECK: result = hipGraphicsGLRegisterImage(&graphicsResource, gl_uint, gl_enum, flags);
  result = cuGraphicsGLRegisterImage(&graphicsResource, gl_uint, gl_enum, flags);

  // CUDA: CUresult CUDAAPI cuCtxSetLimit(CUlimit limit, size_t value);
  // HIP: hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value);
  // CHECK: result = hipDeviceSetLimit(limit, bytes);
  result = cuCtxSetLimit(limit, bytes);

  // CHECK: hipError_t result_2;
  CUresult result_2;
  const char* ret = NULL;

  // CUDA: CUresult CUDAAPI cuGetErrorName(CUresult error, const char **pStr);
  // HIP: hipError_t hipDrvGetErrorName(hipError_t hipError, const char** errorString);
  // CHECK: result = hipDrvGetErrorName(result_2, &ret);
  result = cuGetErrorName(result_2, &ret);

  // CUDA: CUresult CUDAAPI cuGetErrorString(CUresult error, const char **pStr);
  // HIP: hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString);
  // CHECK: result = hipDrvGetErrorString(result_2, &ret);
  result = cuGetErrorString(result_2, &ret);

  // CUDA: CUresult CUDAAPI cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr);
  // HIP: hipError_t hipPointerSetAttribute(const void* value, hipPointer_attribute attribute, hipDeviceptr_t ptr);
  // CHECK: result = hipPointerSetAttribute(image, pointer_attribute, deviceptr);
  result = cuPointerSetAttribute(image, pointer_attribute, deviceptr);

#if CUDA_VERSION >= 8000
  // CHECK: hipMemRangeAttribute MemoryRangeAttribute;
  // CHECK-NEXT: hipMemoryAdvise MemoryAdvise;
  CUmem_range_attribute MemoryRangeAttribute;
  CUmem_advise MemoryAdvise;

  // CUDA: CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
  // HIP: hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device);
  // CHECK: result = hipMemAdvise(deviceptr, bytes, MemoryAdvise, device);
  result = cuMemAdvise(deviceptr, bytes, MemoryAdvise, device);

  // CUDA: CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
  // HIP: hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream __dparm(0));
  // CHECK: result = hipMemPrefetchAsync(deviceptr, bytes, device, stream);
  result = cuMemPrefetchAsync(deviceptr, bytes, device, stream);

  // CUDA: CUresult CUDAAPI cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
  // HIP: hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute, const void* dev_ptr, size_t count);
  // CHECK: result = hipMemRangeGetAttribute(image, bytes, MemoryRangeAttribute, deviceptr, bytes);
  result = cuMemRangeGetAttribute(image, bytes, MemoryRangeAttribute, deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count);
  // HIP: hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes, hipMemRangeAttribute* attributes, size_t num_attributes, const void* dev_ptr, size_t count);
  // CHECK: result = hipMemRangeGetAttributes(&image, &bytes, &MemoryRangeAttribute, bytes, deviceptr, bytes);
  result = cuMemRangeGetAttributes(&image, &bytes, &MemoryRangeAttribute, bytes, deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  // CHECK: result = hipStreamWaitValue32(stream, deviceptr, u_value, flags);
  result = cuStreamWaitValue32(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  // CHECK: result = hipStreamWriteValue32(stream, deviceptr, u_value, flags);
  result = cuStreamWriteValue32(stream, deviceptr, u_value, flags);

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor);
  // HIP: DEPRECATED(DEPRECATED_MSG) hipError_t hipTexRefSetBorderColor(textureReference* texRef, float* pBorderColor);
  // CHECK: result = hipTexRefSetBorderColor(texref, &ms);
  result = cuTexRefSetBorderColor(texref, &ms);

  // CHECK: hipDeviceP2PAttr deviceP2PAttribute;
  CUdevice_P2PAttribute deviceP2PAttribute;

  // CUDA: CUresult CUDAAPI cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
  // HIP: hipError_t hipDeviceGetP2PAttribute(int* value, hipDeviceP2PAttr attr, int srcDevice, int dstDevice);
  // CHECK: result = hipDeviceGetP2PAttribute(value, deviceP2PAttribute, iBlockSize, iBlockSize_2);
  result = cuDeviceGetP2PAttribute(value, deviceP2PAttribute, iBlockSize, iBlockSize_2);
#endif

#if CUDA_VERSION >= 9000
  // CUDA: CUresult CUDAAPI cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  // CHECK: result = hipStreamWaitValue64(stream, deviceptr, u_value, flags);
  result = cuStreamWaitValue64(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  // CHECK: result = hipStreamWriteValue64(stream, deviceptr, u_value, flags);
  result = cuStreamWriteValue64(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams);
  // HIP: hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, hipStream_t stream, void** kernelParams);
  // CHECK: result = hipModuleLaunchCooperativeKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, &kernelParams);
  result = cuLaunchCooperativeKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, &kernelParams);

  // CHECK: hipFunctionLaunchParams_t LAUNCH_PARAMS_st;
  // CHECK-NEXT: hipFunctionLaunchParams LAUNCH_PARAMS;
  CUDA_LAUNCH_PARAMS_st LAUNCH_PARAMS_st;
  CUDA_LAUNCH_PARAMS LAUNCH_PARAMS;

  // CUDA: __CUDA_DEPRECATED CUresult CUDAAPI cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags);
  // HIP: hipError_t hipModuleLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList, unsigned int numDevices, unsigned int flags);
  // CHECK: result = hipModuleLaunchCooperativeKernelMultiDevice(&LAUNCH_PARAMS, flags_2, flags);
  result = cuLaunchCooperativeKernelMultiDevice(&LAUNCH_PARAMS, flags_2, flags);
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipGraph_t graph, graph2;
  // CHECK-NEXT: hipGraphNode_t graphNode, graphNode2;
  // CHECK-NEXT: const hipGraphNode_t *pGraphNode = nullptr;
  // CHECK-NEXT: hipKernelNodeParams KERNEL_NODE_PARAMS;
  // CHECK-NEXT: hipMemsetParams MEMSET_NODE_PARAMS;
  // CHECK-NEXT: hipGraphExec_t graphExec;
  // CHECK-NEXT: hipExternalMemory_t externalMemory;
  // CHECK-NEXT: hipExternalSemaphore_t externalSemaphore;
  // CHECK-NEXT: hipExternalMemoryBufferDesc EXTERNAL_MEMORY_BUFFER_DESC;
  // CHECK-NEXT: hipExternalMemoryHandleDesc EXTERNAL_MEMORY_HANDLE_DESC;
  // CHECK-NEXT: hipExternalSemaphoreHandleDesc EXTERNAL_SEMAPHORE_HANDLE_DESC;
  // CHECK-NEXT: hipExternalSemaphoreSignalParams EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
  // CHECK-NEXT: hipExternalSemaphoreWaitParams EXTERNAL_SEMAPHORE_WAIT_PARAMS;
  // CHECK-NEXT: hipStreamCaptureStatus streamCaptureStatus;
  // CHECK-NEXT: hipGraphNodeType graphNodeType;
  // CHECK-NEXT: hipHostNodeParams host_node_params;
  CUgraph graph, graph2;
  CUgraphNode graphNode, graphNode2;
  const CUgraphNode *pGraphNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS KERNEL_NODE_PARAMS;
  CUDA_MEMSET_NODE_PARAMS MEMSET_NODE_PARAMS;
  CUgraphExec graphExec;
  CUexternalMemory externalMemory;
  CUexternalSemaphore externalSemaphore;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC EXTERNAL_MEMORY_BUFFER_DESC;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC EXTERNAL_MEMORY_HANDLE_DESC;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC EXTERNAL_SEMAPHORE_HANDLE_DESC;
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS EXTERNAL_SEMAPHORE_WAIT_PARAMS;
  CUstreamCaptureStatus streamCaptureStatus;
  CUgraphNodeType graphNodeType;
  CUDA_HOST_NODE_PARAMS host_node_params;

  // CUDA: CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev);
  // HIP: hipError_t hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device);
  // CHECK: result = hipDeviceGetUuid(&uuid, device);
  result = cuDeviceGetUuid(&uuid, device);

  // CHECK: hipHostFn_t hostFn;
  CUhostFn hostFn;

  // CUDA: CUresult CUDAAPI cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData);
  // HIP: hipError_t hipLaunchHostFunc(hipStream_t stream, hipHostFn_t fn, void* userData);
  // CHECK: result = hipLaunchHostFunc(stream, hostFn, image);
  result = cuLaunchHostFunc(stream, hostFn, image);

  // CUDA: CUresult CUDAAPI cuStreamEndCapture(CUstream hStream, CUgraph *phGraph);
  // HIP: hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph);
  // CHECK: result = hipStreamEndCapture(stream, &graph);
  result = cuStreamEndCapture(stream, &graph);

  // CUDA: CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory extMem);
  // HIP: hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);
  // CHECK: result = hipDestroyExternalMemory(externalMemory);
  result = cuDestroyExternalMemory(externalMemory);

  // CUDA: CUresult CUDAAPI cuDestroyExternalSemaphore(CUexternalSemaphore extSem);
  // HIP: hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);
  // CHECK: result = hipDestroyExternalSemaphore(externalSemaphore);
  result = cuDestroyExternalSemaphore(externalSemaphore);

  // CUDA: CUresult CUDAAPI cuExternalMemoryGetMappedBuffer(CUdeviceptr *devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc);
  // HIP: hipError_t hipExternalMemoryGetMappedBuffer(void **devPtr, hipExternalMemory_t extMem, const hipExternalMemoryBufferDesc *bufferDesc);
  // CHECK: result = hipExternalMemoryGetMappedBuffer(&deviceptr, externalMemory, &EXTERNAL_MEMORY_BUFFER_DESC);
  result = cuExternalMemoryGetMappedBuffer(&deviceptr, externalMemory, &EXTERNAL_MEMORY_BUFFER_DESC);

  // CUDA: CUresult CUDAAPI cuImportExternalMemory(CUexternalMemory *extMem_out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc);
  // HIP: hipError_t hipImportExternalMemory(hipExternalMemory_t* extMem_out, const hipExternalMemoryHandleDesc* memHandleDesc);
  // CHECK: result = hipImportExternalMemory(&externalMemory, &EXTERNAL_MEMORY_HANDLE_DESC);
  result = cuImportExternalMemory(&externalMemory, &EXTERNAL_MEMORY_HANDLE_DESC);

  // CUDA: CUresult CUDAAPI cuImportExternalSemaphore(CUexternalSemaphore *extSem_out, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc);
  // HIP: hipError_t hipImportExternalSemaphore(hipExternalSemaphore_t* extSem_out, const hipExternalSemaphoreHandleDesc* semHandleDesc);
  // CHECK: result = hipImportExternalSemaphore(&externalSemaphore, &EXTERNAL_SEMAPHORE_HANDLE_DESC);
  result = cuImportExternalSemaphore(&externalSemaphore, &EXTERNAL_SEMAPHORE_HANDLE_DESC);

  // CUDA: CUresult CUDAAPI cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
  // HIP: hipError_t hipSignalExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreSignalParams* paramsArray, unsigned int numExtSems, hipStream_t stream);
  // CHECK: result = hipSignalExternalSemaphoresAsync(&externalSemaphore, &EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, flags, stream);
  result = cuSignalExternalSemaphoresAsync(&externalSemaphore, &EXTERNAL_SEMAPHORE_SIGNAL_PARAMS, flags, stream);

  // CUDA: CUresult CUDAAPI cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSemArray, const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray, unsigned int numExtSems, CUstream stream);
  // HIP: hipError_t hipWaitExternalSemaphoresAsync(const hipExternalSemaphore_t* extSemArray, const hipExternalSemaphoreWaitParams* paramsArray, unsigned int numExtSems, hipStream_t stream);
  // CHECK: result = hipWaitExternalSemaphoresAsync(&externalSemaphore, &EXTERNAL_SEMAPHORE_WAIT_PARAMS, flags, stream);
  result = cuWaitExternalSemaphoresAsync(&externalSemaphore, &EXTERNAL_SEMAPHORE_WAIT_PARAMS, flags, stream);

  // CUDA: CUresult CUDAAPI cuGraphAddDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
  // HIP: hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies);
  // CHECK: result = hipGraphAddDependencies(graph, &graphNode, &graphNode2, bytes);
  result = cuGraphAddDependencies(graph, &graphNode, &graphNode2, bytes);

  // CUDA: CUresult CUDAAPI cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies);
  // HIP: hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies);
  // CHECK: result = hipGraphAddEmptyNode(&graphNode, graph, &graphNode2, bytes);
  result = cuGraphAddEmptyNode(&graphNode, graph, &graphNode2, bytes);

  // CUDA: CUresult CUDAAPI cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddKernelNode(&graphNode, graph, &graphNode2, bytes, &KERNEL_NODE_PARAMS);
  result = cuGraphAddKernelNode(&graphNode, graph, &graphNode2, bytes, &KERNEL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphCreate(CUgraph *phGraph, unsigned int flags);
  // HIP: hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags);
  // CHECK: result = hipGraphCreate(&graph, flags);
  result = cuGraphCreate(&graph, flags);

  // CUDA: CUresult CUDAAPI cuGraphDestroy(CUgraph hGraph);
  // HIP: hipError_t hipGraphDestroy(hipGraph_t graph);
  // CHECK: result = hipGraphDestroy(graph);
  result = cuGraphDestroy(graph);

  // CUDA: CUresult CUDAAPI cuGraphExecDestroy(CUgraphExec hGraphExec);
  // HIP: hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec);
  // CHECK: result = hipGraphExecDestroy(graphExec);
  result = cuGraphExecDestroy(graphExec);

  // CUDA: CUresult CUDAAPI cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
  // HIP: hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes);
  // CHECK: result = hipGraphGetNodes(graph, &graphNode, &bytes);
  result = cuGraphGetNodes(graph, &graphNode, &bytes);

  // CUDA: CUresult CUDAAPI cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes, size_t *numRootNodes);
  // HIP: hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes);
  // CHECK: result = hipGraphGetRootNodes(graph, &graphNode, &bytes);
  result = cuGraphGetRootNodes(graph, &graphNode, &bytes);

  // CUDA: CUresult CUDAAPI cuGraphKernelNodeGetParams(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphKernelNodeGetParams(hipGraphNode_t node, hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphKernelNodeGetParams(graphNode, &KERNEL_NODE_PARAMS);
  result = cuGraphKernelNodeGetParams(graphNode, &KERNEL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphKernelNodeSetParams(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphKernelNodeSetParams(hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphKernelNodeSetParams(graphNode, &KERNEL_NODE_PARAMS);
  result = cuGraphKernelNodeSetParams(graphNode, &KERNEL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream);
  // HIP: hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream);
  // CHECK: result = hipGraphLaunch(graphExec, stream);
  result = cuGraphLaunch(graphExec, stream);

  // CUDA: CUresult CUDAAPI cuGraphMemcpyNodeGetParams(CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams);
  // HIP: hipError_t hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms* pNodeParams);
  // CHECK: result = hipDrvGraphMemcpyNodeGetParams(graphNode, &MEMCPY3D);
  result = cuGraphMemcpyNodeGetParams(graphNode, &MEMCPY3D);

  // CUDA: CUresult CUDAAPI cuGraphMemcpyNodeSetParams(CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams);
  // HIP: hipError_t hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t node, const hipMemcpy3DParms* pNodeParams);
  // CHECK: result = hipDrvGraphMemcpyNodeSetParams(graphNode, &MEMCPY3D);
  result = cuGraphMemcpyNodeSetParams(graphNode, &MEMCPY3D);

  // CUDA: CUresult CUDAAPI cuGraphMemsetNodeGetParams(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams* pNodeParams);
  // CHECK: result = hipGraphMemsetNodeGetParams(graphNode, &MEMSET_NODE_PARAMS);
  result = cuGraphMemsetNodeGetParams(graphNode, &MEMSET_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphMemsetNodeSetParams(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphMemsetNodeSetParams(hipGraphNode_t node, const hipMemsetParams* pNodeParams);
  // CHECK: result = hipGraphMemsetNodeSetParams(graphNode, &MEMSET_NODE_PARAMS);
  result = cuGraphMemsetNodeSetParams(graphNode, &MEMSET_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphGetEdges(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to, size_t *numEdges);
  // HIP: hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges);
  // CHECK: result = hipGraphGetEdges(graph, &graphNode, &graphNode2, &bytes);
  result = cuGraphGetEdges(graph, &graphNode, &graphNode2, &bytes);

  // CUDA: CUresult CUDAAPI cuGraphNodeGetDependencies(CUgraphNode hNode, CUgraphNode *dependencies, size_t *numDependencies);
  // HIP: hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies, size_t* pNumDependencies);
  // CHECK: result = hipGraphNodeGetDependencies(graphNode, &graphNode2, &bytes);
  result = cuGraphNodeGetDependencies(graphNode, &graphNode2, &bytes);

  // CUDA: CUresult CUDAAPI cuGraphRemoveDependencies(CUgraph hGraph, const CUgraphNode *from, const CUgraphNode *to, size_t numDependencies);
  // HIP: hipError_t hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t* from, const hipGraphNode_t* to, size_t numDependencies);
  // CHECK: result = hipGraphRemoveDependencies(graph, &graphNode, &graphNode2, bytes);
  result = cuGraphRemoveDependencies(graph, &graphNode, &graphNode2, bytes);

  // CUDA: CUresult CUDAAPI cuGraphNodeGetDependentNodes(CUgraphNode hNode, CUgraphNode *dependentNodes, size_t *numDependentNodes);
  // HIP: hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes, size_t* pNumDependentNodes);
  // CHECK: result = hipGraphNodeGetDependentNodes(graphNode, &graphNode2, &bytes);
  result = cuGraphNodeGetDependentNodes(graphNode, &graphNode2, &bytes);

  // CUDA: CUresult CUDAAPI cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);
  // HIP: hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType);
  // CHECK: result = hipGraphNodeGetType(graphNode, &graphNodeType);
  result = cuGraphNodeGetType(graphNode, &graphNodeType);

  // CUDA: CUresult CUDAAPI cuGraphDestroyNode(CUgraphNode hNode);
  // HIP: hipError_t hipGraphDestroyNode(hipGraphNode_t node);
  // CHECK: result = hipGraphDestroyNode(graphNode);
  result = cuGraphDestroyNode(graphNode);

  // CUDA: CUresult CUDAAPI cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_HOST_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphAddHostNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddHostNode(&graphNode, graph, &graphNode2, bytes, &host_node_params);
  result = cuGraphAddHostNode(&graphNode, graph, &graphNode2, bytes, &host_node_params);

  // CUDA: CUresult CUDAAPI cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
  // HIP: hipError_t hipGraphNodeFindInClone(hipGraphNode_t* pNode, hipGraphNode_t originalNode, hipGraph_t clonedGraph);
  // CHECK: result = hipGraphNodeFindInClone(&graphNode2, graphNode, graph);
  result = cuGraphNodeFindInClone(&graphNode2, graphNode, graph);

  // CUDA: CUresult CUDAAPI cuStreamIsCapturing(CUstream hStream, CUstreamCaptureStatus *captureStatus);
  // HIP: hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus);
  // CHECK: result = hipStreamIsCapturing(stream, &streamCaptureStatus);
  result = cuStreamIsCapturing(stream, &streamCaptureStatus);

  // CUDA: CUresult CUDAAPI cuGraphHostNodeGetParams(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphHostNodeGetParams(hipGraphNode_t node, hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphHostNodeGetParams(graphNode, &host_node_params);
  result = cuGraphHostNodeGetParams(graphNode, &host_node_params);

  // CUDA: CUresult CUDAAPI cuGraphHostNodeSetParams(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphHostNodeSetParams(hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphHostNodeSetParams(graphNode, &host_node_params);
  result = cuGraphHostNodeSetParams(graphNode, &host_node_params);

  // CUDA: CUresult CUDAAPI cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUgraph childGraph);
  // HIP: hipError_t hipGraphAddChildGraphNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipGraph_t childGraph);
  // CHECK: result = hipGraphAddChildGraphNode(&graphNode, graph, &graphNode2, bytes, graph2);
  result = cuGraphAddChildGraphNode(&graphNode, graph, &graphNode2, bytes, graph2);

  // CUDA: CUresult CUDAAPI cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph);
  // HIP: hipError_t hipGraphChildGraphNodeGetGraph(hipGraphNode_t node, hipGraph_t* pGraph);
  // CHECK: result = hipGraphChildGraphNodeGetGraph(graphNode, &graph);
  result = cuGraphChildGraphNodeGetGraph(graphNode, &graph);

  // CUDA: CUresult CUDAAPI cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph);
  // HIP: hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph);
  // CHECK: result = hipGraphClone(&graph, graph2);
  result = cuGraphClone(&graph, graph2);

  // CUDA: CUresult CUDAAPI cuGraphAddMemcpyNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEMCPY3D *copyParams, CUcontext ctx);
  // HIP: hipError_t hipDrvGraphAddMemcpyNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph, const hipGraphNode_t* dependencies, size_t numDependencies, const HIP_MEMCPY3D* copyParams, hipCtx_t ctx);
  // CHECK: result = hipDrvGraphAddMemcpyNode(&graphNode, graph, &graphNode2, bytes, &MEMCPY3D, context);
  result = cuGraphAddMemcpyNode(&graphNode, graph, &graphNode2, bytes, &MEMCPY3D, context);
#endif

#if CUDA_VERSION >= 10000 && CUDA_VERSION < 12000
  // CUDA: CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
  // HIP: hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph, hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize);
  // CHECK: result = hipGraphInstantiate(&graphExec, graph, &graphNode, nullptr, bytes);
  result = cuGraphInstantiate(&graphExec, graph, &graphNode, nullptr, bytes);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipStreamCaptureMode streamCaptureMode;
  CUstreamCaptureMode streamCaptureMode;

  // CUDA: CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode);
  // HIP: hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode);
  // CHECK: result = hipThreadExchangeStreamCaptureMode(&streamCaptureMode);
  result = cuThreadExchangeStreamCaptureMode(&streamCaptureMode);

  // CUDA: CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream, CUstreamCaptureMode mode);
  // HIP: hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode);
  // CHECK: result = hipStreamBeginCapture(stream, streamCaptureMode);
  // CHECK-NEXT: result = hipStreamBeginCapture(stream, streamCaptureMode);
  result = cuStreamBeginCapture(stream, streamCaptureMode);
  result = cuStreamBeginCapture_v2(stream, streamCaptureMode);

  // CUDA: CUresult CUDAAPI cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipKernelNodeParams* pNodeParams);
  // CHECK: result = hipGraphExecKernelNodeSetParams(graphExec, graphNode, &KERNEL_NODE_PARAMS);
  result = cuGraphExecKernelNodeSetParams(graphExec, graphNode, &KERNEL_NODE_PARAMS);
#endif

#if CUDA_VERSION >= 10010 && CUDA_VERSION < 12000
  // CUDA: CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
  // HIP: hipError_t hipStreamGetCaptureInfo(hipStream_t stream, hipStreamCaptureStatus* pCaptureStatus, unsigned long long* pId);
  // CHECK: result = hipStreamGetCaptureInfo(stream, &streamCaptureStatus, &ull);
  result = cuStreamGetCaptureInfo(stream, &streamCaptureStatus, &ull);
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipGraphExecUpdateResult graphExecUpdateResult;
  CUgraphExecUpdateResult graphExecUpdateResult;

  // CHECK: hipMemLocation memLocation_st;
  // CHECK-NEXT: hipMemLocation memLocation;
  // CHECK-NEXT: hipMemAllocationHandleType memAllocationHandleType;
  CUmemLocation_st memLocation_st;
  CUmemLocation memLocation;
  CUmemAllocationHandleType memAllocationHandleType;

  // CUDA: CUresult CUDAAPI cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipError_t hipGraphExecHostNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const hipHostNodeParams* pNodeParams);
  // CHECK: result = hipGraphExecHostNodeSetParams(graphExec, graphNode, &host_node_params);
  result = cuGraphExecHostNodeSetParams(graphExec, graphNode, &host_node_params);

  // CHECK: hipMemAllocationProp memAllocationProp;
  CUmemAllocationProp memAllocationProp;
  // CHECK: hipMemGenericAllocationHandle_t memGenericAllocationHandle_t;
  CUmemGenericAllocationHandle memGenericAllocationHandle_t;
  // CHECK: hipMemAllocationGranularity_flags memAllocationGranularity_flags;
  CUmemAllocationGranularity_flags memAllocationGranularity_flags;
  // CHECK: hipMemAccessDesc memAccessDesc;
  CUmemAccessDesc memAccessDesc;

  // CUDA: CUresult CUDAAPI cuMemAddressFree(CUdeviceptr ptr, size_t size);
  // HIP: hipError_t hipMemAddressFree(void* devPtr, size_t size);
  // CHECK: result = hipMemAddressFree(deviceptr, bytes);
  result = cuMemAddressFree(deviceptr, bytes);

  // CUDA: CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
  // HIP: hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags);
  // CHECK: result = hipMemAddressReserve(&deviceptr, bytes, bytes_2, deviceptr_2, ull);
  result = cuMemAddressReserve(&deviceptr, bytes, bytes_2, deviceptr_2, ull);

  // CUDA: CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags);
  // HIP: hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t size, const hipMemAllocationProp* prop, unsigned long long flags);
  // CHECK: result = hipMemCreate(&memGenericAllocationHandle_t, bytes, &memAllocationProp, ull);
  result = cuMemCreate(&memGenericAllocationHandle_t, bytes, &memAllocationProp, ull);

  // CUDA: CUresult CUDAAPI cuMemExportToShareableHandle(void *shareableHandle, CUmemGenericAllocationHandle handle, CUmemAllocationHandleType handleType, unsigned long long flags);
  // HIP: hipError_t hipMemExportToShareableHandle(void* shareableHandle, hipMemGenericAllocationHandle_t handle, hipMemAllocationHandleType handleType, unsigned long long flags);
  // CHECK: result = hipMemExportToShareableHandle(image, memGenericAllocationHandle_t, memAllocationHandleType, ull);
  result = cuMemExportToShareableHandle(image, memGenericAllocationHandle_t, memAllocationHandleType, ull);

  // CUDA: CUresult CUDAAPI cuMemGetAccess(unsigned long long *flags, const CUmemLocation *location, CUdeviceptr ptr);
  // HIP: hipError_t hipMemGetAccess(unsigned long long* flags, const hipMemLocation* location, void* ptr);
  // CHECK: result = hipMemGetAccess(&ull_2, &memLocation, deviceptr);
  result = cuMemGetAccess(&ull_2, &memLocation, deviceptr);

  // CUDA: CUresult CUDAAPI cuMemGetAllocationGranularity(size_t *granularity, const CUmemAllocationProp *prop, CUmemAllocationGranularity_flags option);
  // HIP: hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop, hipMemAllocationGranularity_flags option);
  // CHECK: result = hipMemGetAllocationGranularity(&bytes, &memAllocationProp, memAllocationGranularity_flags);
  result = cuMemGetAllocationGranularity(&bytes, &memAllocationProp, memAllocationGranularity_flags);

  // CUDA: CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop, CUmemGenericAllocationHandle handle);
  // HIP: hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop, hipMemGenericAllocationHandle_t handle);
  // CHECK: result = hipMemGetAllocationPropertiesFromHandle(&memAllocationProp, memGenericAllocationHandle_t);
  result = cuMemGetAllocationPropertiesFromHandle(&memAllocationProp, memGenericAllocationHandle_t);

  // CUDA: CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle, void *osHandle, CUmemAllocationHandleType shHandleType);
  // HIP: hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle, hipMemAllocationHandleType shHandleType);
  // CHECK: result = hipMemImportFromShareableHandle(&memGenericAllocationHandle_t, image, memAllocationHandleType);
  result = cuMemImportFromShareableHandle(&memGenericAllocationHandle_t, image, memAllocationHandleType);

  // CUDA: CUresult CUDAAPI cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
  // HIP: hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle, unsigned long long flags);
  // CHECK: result = hipMemMap(deviceptr, bytes, bytes_2, memGenericAllocationHandle_t, ull);
  result = cuMemMap(deviceptr, bytes, bytes_2, memGenericAllocationHandle_t, ull);

  // CUDA: CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle);
  // HIP: hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);
  // CHECK: result = hipMemRelease(memGenericAllocationHandle_t);
  result = cuMemRelease(memGenericAllocationHandle_t);

  // CUDA: CUresult CUDAAPI cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count);
  // HIP: hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);
  // CHECK: result = hipMemSetAccess(deviceptr, bytes, &memAccessDesc, bytes_2);
  result = cuMemSetAccess(deviceptr, bytes, &memAccessDesc, bytes_2);

  // CUDA: CUresult CUDAAPI cuMemUnmap(CUdeviceptr ptr, size_t size);
  // HIP: hipError_t hipMemUnmap(void* ptr, size_t size);
  // CHECK: result = hipMemUnmap(deviceptr, bytes);
  result = cuMemUnmap(deviceptr, bytes);
#endif

#if CUDA_VERSION >= 10020 && CUDA_VERSION < 12000
  // CUDA: CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out);
  // HIP: hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph, hipGraphNode_t* hErrorNode_out, hipGraphExecUpdateResult* updateResult_out);
  // CHECK: result = hipGraphExecUpdate(graphExec, graph, &graphNode, &graphExecUpdateResult);
  result = cuGraphExecUpdate(graphExec, graph, &graphNode, &graphExecUpdateResult);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: result = hipDevicePrimaryCtxRelease(device);
  result = cuDevicePrimaryCtxRelease_v2(device);

  // CHECK: result = hipDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset_v2(device);

  // CHECK: result = hipDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags_v2(device, flags);

  // CUDA: CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr);
  // HIP: hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* addr);
  // CHECK: result = hipMemRetainAllocationHandle(&memGenericAllocationHandle_t, image);
  result = cuMemRetainAllocationHandle(&memGenericAllocationHandle_t, image);

  // CHECK: hipKernelNodeAttrID kernelNodeAttrID;
  CUkernelNodeAttrID kernelNodeAttrID;
  // CHECK: hipKernelNodeAttrValue kernelNodeAttrValue;
  CUkernelNodeAttrValue kernelNodeAttrValue;

  // CUDA: CUresult CUDAAPI cuGraphKernelNodeSetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, const CUkernelNodeAttrValue* value);
  // HIP: hipError_t hipGraphKernelNodeSetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, const hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cuGraphKernelNodeSetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);

  // CUDA: CUresult CUDAAPI cuGraphKernelNodeGetAttribute(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out);
  // HIP: hipError_t hipGraphKernelNodeGetAttribute(hipGraphNode_t hNode, hipKernelNodeAttrID attr, hipKernelNodeAttrValue* value);
  // CHECK: result = hipGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);
  result = cuGraphKernelNodeGetAttribute(graphNode, kernelNodeAttrID, &kernelNodeAttrValue);

  // CUDA: CUresult CUDAAPI cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src);
  // HIP: hipError_t hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst);
  // CHECK: result = hipGraphKernelNodeCopyAttributes(graphNode, graphNode2);
  result = cuGraphKernelNodeCopyAttributes(graphNode, graphNode2);
#endif

#if CUDA_VERSION >= 11000 && CUDA_VERSION < 12000
  // CHECK: result = hipGraphInstantiate(&graphExec, graph, &graphNode, nullptr, bytes);
  result = cuGraphInstantiate_v2(&graphExec, graph, &graphNode, nullptr, bytes);
#endif

#if CUDA_VERSION >= 11010
  // CUDA: CUresult CUDAAPI cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
  // HIP: hipError_t hipGraphExecChildGraphNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, hipGraph_t childGraph);
  // CHECK: result = hipGraphExecChildGraphNodeSetParams(graphExec, graphNode, graph);
  result = cuGraphExecChildGraphNodeSetParams(graphExec, graphNode, graph);

  // CUDA: CUresult CUDAAPI cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
  // HIP: hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event);
  // CHECK: result = hipGraphAddEventRecordNode(&graphNode, graph, &graphNode2, bytes, event_);
  result = cuGraphAddEventRecordNode(&graphNode, graph, &graphNode2, bytes, event_);

  // CUDA: CUresult CUDAAPI cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
  // HIP: hipError_t hipGraphEventRecordNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
  // CHECK: result = hipGraphEventRecordNodeGetEvent(graphNode, &event_);
  result = cuGraphEventRecordNodeGetEvent(graphNode, &event_);

  // CUDA: CUresult CUDAAPI cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event);
  // HIP: hipError_t hipGraphEventRecordNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
  // CHECK: result = hipGraphEventRecordNodeSetEvent(graphNode, event_);
  result = cuGraphEventRecordNodeSetEvent(graphNode, event_);

  // CUDA: CUresult CUDAAPI cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
  // HIP: hipError_t hipGraphExecEventRecordNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event);
  // CHECK: result = hipGraphExecEventRecordNodeSetEvent(graphExec, graphNode, event_);
  result = cuGraphExecEventRecordNodeSetEvent(graphExec, graphNode, event_);

  // CUDA: CUresult CUDAAPI cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUevent event);
  // HIP: hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipEvent_t event);
  // CHECK: result = hipGraphAddEventWaitNode(&graphNode, graph, &graphNode2, bytes, event_);
  result = cuGraphAddEventWaitNode(&graphNode, graph, &graphNode2, bytes, event_);

  // CUDA: CUresult CUDAAPI cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out);
  // HIP: hipError_t hipGraphEventWaitNodeGetEvent(hipGraphNode_t node, hipEvent_t* event_out);
  // CHECK: result = hipGraphEventWaitNodeGetEvent(graphNode, &event_);
  result = cuGraphEventWaitNodeGetEvent(graphNode, &event_);

  // CUDA: CUresult CUDAAPI cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event);
  // HIP: hipError_t hipGraphEventWaitNodeSetEvent(hipGraphNode_t node, hipEvent_t event);
  // CHECK: result = hipGraphEventWaitNodeSetEvent(graphNode, event_);
  result = cuGraphEventWaitNodeSetEvent(graphNode, event_);

  // CUDA: CUresult CUDAAPI cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
  // HIP: hipError_t hipGraphExecEventWaitNodeSetEvent(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, hipEvent_t event);
  // CHECK: result = hipGraphExecEventWaitNodeSetEvent(graphExec, graphNode, event_);
  result = cuGraphExecEventWaitNodeSetEvent(graphExec, graphNode, event_);

  // CHECK: hipArrayMapInfo arrayMapInfo;
  CUarrayMapInfo arrayMapInfo;

  // CUDA: CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count, CUstream hStream);
  // HIP: hipError_t hipMemMapArrayAsync(hipArrayMapInfo* mapInfoList, unsigned int count, hipStream_t stream);
  // CHECK: result = hipMemMapArrayAsync(&arrayMapInfo, flags, stream);
  result = cuMemMapArrayAsync(&arrayMapInfo, flags, stream);

  // CUDA: CUresult CUDAAPI cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream);
  // HIP: hipError_t hipGraphUpload(hipGraphExec_t graphExec, hipStream_t stream);
  // CHECK: result = hipGraphUpload(graphExec, stream);
  result = cuGraphUpload(graphExec, stream);
#endif

#if CUDA_VERSION >= 11020
  // CHECK: hipMemPool_t memPool_t;
  // CHECK-NEXT: hipMemPoolProps memPoolProps_st;
  // CHECK-NEXT: hipMemPoolProps memPoolProps;
  CUmemoryPool memPool_t;
  CUmemPoolProps_st memPoolProps_st;
  CUmemPoolProps memPoolProps;

  // CUDA: CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev);
  // HIP: hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* mem_pool, int device);
  // CHECK: result = hipDeviceGetDefaultMemPool(&memPool_t, device);
  result = cuDeviceGetDefaultMemPool(&memPool_t, device);

  // CUDA: CUresult CUDAAPI cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool);
  // HIP: hipError_t hipDeviceSetMemPool(int device, hipMemPool_t mem_pool);
  // CHECK: result = hipDeviceSetMemPool(device, memPool_t);
  result = cuDeviceSetMemPool(device, memPool_t);

  // CUDA: CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev);
  // HIP: hipError_t hipDeviceGetMemPool(hipMemPool_t* mem_pool, int device);
  // CHECK: result = hipDeviceGetMemPool(&memPool_t, device);
  result = cuDeviceGetMemPool(&memPool_t, device);

  // CUDA: CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream);
  // HIP: hipError_t hipMallocAsync(void** dev_ptr, size_t size, hipStream_t stream);
  // CHECK: result = hipMallocAsync(&deviceptr, bytes, stream);
  result = cuMemAllocAsync(&deviceptr, bytes, stream);

  // CUDA: CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);
  // HIP: hipError_t hipFreeAsync(void* dev_ptr, hipStream_t stream);
  // CHECK: result = hipFreeAsync(deviceptr, stream);
  result = cuMemFreeAsync(deviceptr, stream);

  // CUDA: CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep);
  // HIP: hipError_t hipMemPoolTrimTo(hipMemPool_t mem_pool, size_t min_bytes_to_hold);
  // CHECK: result = hipMemPoolTrimTo(memPool_t, bytes);
  result = cuMemPoolTrimTo(memPool_t, bytes);

  // CHECK: hipMemPoolAttr memPoolAttr;
  CUmemPool_attribute memPoolAttr;

  // CUDA: CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
  // HIP: hipError_t hipMemPoolSetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
  // CHECK: result = hipMemPoolSetAttribute(memPool_t, memPoolAttr, image);
  result = cuMemPoolSetAttribute(memPool_t, memPoolAttr, image);

  // CUDA: CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value);
  // HIP: hipError_t hipMemPoolGetAttribute(hipMemPool_t mem_pool, hipMemPoolAttr attr, void* value);
  // CHECK: result = hipMemPoolGetAttribute(memPool_t, memPoolAttr, image);
  result = cuMemPoolGetAttribute(memPool_t, memPoolAttr, image);

  // CUDA: CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map, size_t count);
  // HIP: hipError_t hipMemPoolSetAccess(hipMemPool_t mem_pool, const hipMemAccessDesc* desc_list, size_t count);
  // CHECK: result = hipMemPoolSetAccess(memPool_t, &memAccessDesc, bytes);
  result = cuMemPoolSetAccess(memPool_t, &memAccessDesc, bytes);

  // CHECK: hipMemAccessFlags memAccessFlags;
  CUmemAccess_flags memAccessFlags;

  // CUDA: CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool, CUmemLocation *location);
  // HIP: hipError_t hipMemPoolGetAccess(hipMemAccessFlags* flags, hipMemPool_t mem_pool, hipMemLocation* location);
  // CHECK: result = hipMemPoolGetAccess(&memAccessFlags, memPool_t, &memLocation);
  result = cuMemPoolGetAccess(&memAccessFlags, memPool_t, &memLocation);

  // CUDA: CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps);
  // HIP: hipError_t hipMemPoolCreate(hipMemPool_t* mem_pool, const hipMemPoolProps* pool_props);
  // CHECK: result = hipMemPoolCreate(&memPool_t, &memPoolProps);
  result = cuMemPoolCreate(&memPool_t, &memPoolProps);

  // CUDA: CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool);
  // HIP: hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);
  // CHECK: result = hipMemPoolDestroy(memPool_t);
  result = cuMemPoolDestroy(memPool_t);

  // CUDA: CUresult CUDAAPI cuMemPoolExportToShareableHandle(void *handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
  // HIP: hipError_t hipMemPoolExportToShareableHandle(void* shared_handle, hipMemPool_t mem_pool, hipMemAllocationHandleType handle_type, unsigned int flags);
  // CHECK: result = hipMemPoolExportToShareableHandle(image, memPool_t, memAllocationHandleType, ull);
  result = cuMemPoolExportToShareableHandle(image, memPool_t, memAllocationHandleType, ull);

  // CUDA: CUresult CUDAAPI cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags);
  // HIP: hipError_t hipMemPoolImportFromShareableHandle(hipMemPool_t* mem_pool, void* shared_handle, hipMemAllocationHandleType handle_type, unsigned int flags);
  // CHECK: result = hipMemPoolImportFromShareableHandle(&memPool_t, image, memAllocationHandleType, ull);
  result = cuMemPoolImportFromShareableHandle(&memPool_t, image, memAllocationHandleType, ull);

  // CHECK: hipMemPoolPtrExportData memPoolPtrExportData;
  CUmemPoolPtrExportData memPoolPtrExportData;

  // CUDA: CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out, CUdeviceptr ptr);
  // HIP: hipError_t hipMemPoolExportPointer(hipMemPoolPtrExportData* export_data, void* dev_ptr);
  // CHECK: result = hipMemPoolExportPointer(&memPoolPtrExportData, deviceptr);
  result = cuMemPoolExportPointer(&memPoolPtrExportData, deviceptr);

  // CUDA: CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData *shareData);
  // HIP: hipError_t hipMemPoolImportPointer(void** dev_ptr, hipMemPool_t mem_pool, hipMemPoolPtrExportData* export_data);
  // CHECK: result = hipMemPoolImportPointer(&deviceptr, memPool_t, &memPoolPtrExportData);
  result = cuMemPoolImportPointer(&deviceptr, memPool_t, &memPoolPtrExportData);

  // CHECK: hipExternalSemaphoreSignalNodeParams EXT_SEM_SIGNAL_NODE_PARAMS;
  CUDA_EXT_SEM_SIGNAL_NODE_PARAMS EXT_SEM_SIGNAL_NODE_PARAMS;

  // CUDA: CUresult CUDAAPI cuGraphAddExternalSemaphoresSignalNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphAddExternalSemaphoresSignalNode(&graphNode, graph, &graphNode2, bytes, &EXT_SEM_SIGNAL_NODE_PARAMS);
  result = cuGraphAddExternalSemaphoresSignalNode(&graphNode, graph, &graphNode2, bytes, &EXT_SEM_SIGNAL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out);
  // HIP: hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreSignalNodeParams* params_out);
  // CHECK: result = hipGraphExternalSemaphoresSignalNodeGetParams(graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);
  result = cuGraphExternalSemaphoresSignalNodeGetParams(graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExternalSemaphoresSignalNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphExternalSemaphoresSignalNodeSetParams(graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);
  result = cuGraphExternalSemaphoresSignalNodeSetParams(graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);

  // CHECK: hipExternalSemaphoreWaitNodeParams EXT_SEM_WAIT_NODE_PARAMS;
  CUDA_EXT_SEM_WAIT_NODE_PARAMS EXT_SEM_WAIT_NODE_PARAMS;

  // CUDA: CUresult CUDAAPI cuGraphAddExternalSemaphoresWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphAddExternalSemaphoresWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphAddExternalSemaphoresWaitNode(&graphNode, graph, &graphNode2, bytes, &EXT_SEM_WAIT_NODE_PARAMS);
  result = cuGraphAddExternalSemaphoresWaitNode(&graphNode, graph, &graphNode2, bytes, &EXT_SEM_WAIT_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeGetParams(CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out);
  // HIP: hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode, hipExternalSemaphoreWaitNodeParams* params_out);
  // CHECK: result = hipGraphExternalSemaphoresWaitNodeGetParams(graphNode, &EXT_SEM_WAIT_NODE_PARAMS);
  result = cuGraphExternalSemaphoresWaitNodeGetParams(graphNode, &EXT_SEM_WAIT_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExternalSemaphoresWaitNodeSetParams(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphExternalSemaphoresWaitNodeSetParams(graphNode, &EXT_SEM_WAIT_NODE_PARAMS);
  result = cuGraphExternalSemaphoresWaitNodeSetParams(graphNode, &EXT_SEM_WAIT_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExecExternalSemaphoresSignalNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreSignalNodeParams* nodeParams);
  // CHECK: result = hipGraphExecExternalSemaphoresSignalNodeSetParams(graphExec, graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);
  result = cuGraphExecExternalSemaphoresSignalNodeSetParams(graphExec, graphNode, &EXT_SEM_SIGNAL_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphExecExternalSemaphoresWaitNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipExternalSemaphoreWaitNodeParams* nodeParams);
  // CHECK: result = hipGraphExecExternalSemaphoresWaitNodeSetParams(graphExec, graphNode, &EXT_SEM_WAIT_NODE_PARAMS);
  result = cuGraphExecExternalSemaphoresWaitNodeSetParams(graphExec, graphNode, &EXT_SEM_WAIT_NODE_PARAMS);
#endif

#if CUDA_VERSION >= 11030
  // CUDA < 12000: CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
  // CUDA:         CUresult CUDAAPI cuStreamGetCaptureInfo_v2(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
  // HIP: hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out __dparm(0), hipGraph_t* graph_out __dparm(0), const hipGraphNode_t** dependencies_out __dparm(0), size_t* numDependencies_out __dparm(0));
  // CHECK: result = hipStreamGetCaptureInfo_v2(stream, &streamCaptureStatus, &ull, &graph, &pGraphNode, &bytes);
  result = cuStreamGetCaptureInfo_v2(stream, &streamCaptureStatus, &ull, &graph, &pGraphNode, &bytes);

  // CUDA: CUresult CUDAAPI cuStreamUpdateCaptureDependencies(CUstream hStream, CUgraphNode *dependencies, size_t numDependencies, unsigned int flags);
  // HIP: hipError_t hipStreamUpdateCaptureDependencies(hipStream_t stream, hipGraphNode_t* dependencies, size_t numDependencies, unsigned int flags __dparm(0));
  // CHECK: result = hipStreamUpdateCaptureDependencies(stream, &graphNode, bytes, flags);
  result = cuStreamUpdateCaptureDependencies(stream, &graphNode, bytes, flags);

  // CHECK: hipUserObject_t userObject;
  CUuserObject userObject;

  // CUDA: CUresult CUDAAPI cuUserObjectCreate(CUuserObject *object_out, void *ptr, CUhostFn destroy, unsigned int initialRefcount, unsigned int flags);
  // HIP: hipError_t hipUserObjectCreate(hipUserObject_t* object_out, void* ptr, hipHostFn_t destroy, unsigned int initialRefcount, unsigned int flags);
  // CHECK: result = hipUserObjectCreate(&userObject, image, hostFn, count, flags);
  result = cuUserObjectCreate(&userObject, image, hostFn, count, flags);

  // CUDA: CUresult CUDAAPI cuUserObjectRelease(CUuserObject object, unsigned int count);
  // HIP: hipError_t hipUserObjectRelease(hipUserObject_t object, unsigned int count);
  // CHECK: result = hipUserObjectRelease(userObject, count);
  result = cuUserObjectRelease(userObject, count);

  // CUDA: CUresult CUDAAPI cuUserObjectRetain(CUuserObject object, unsigned int count);
  // HIP: hipError_t hipUserObjectRetain(hipUserObject_t object, unsigned int count);
  // CHECK: result = hipUserObjectRetain(userObject, count);
  result = cuUserObjectRetain(userObject, count);

  // CUDA: CUresult CUDAAPI cuGraphRetainUserObject(CUgraph graph, CUuserObject object, unsigned int count, unsigned int flags);
  // HIP: hipError_t hipGraphRetainUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count, unsigned int flags);
  // CHECK: result = hipGraphRetainUserObject(graph, userObject, count, flags);
  result = cuGraphRetainUserObject(graph, userObject, count, flags);

  // CUDA: CUresult CUDAAPI cuGraphReleaseUserObject(CUgraph graph, CUuserObject object, unsigned int count);
  // HIP: hipError_t hipGraphReleaseUserObject(hipGraph_t graph, hipUserObject_t object, unsigned int count);
  // CHECK: result = hipGraphReleaseUserObject(graph, userObject, count);
  result = cuGraphReleaseUserObject(graph, userObject, count);

  // CUDA: CUresult CUDAAPI cuGraphDebugDotPrint(CUgraph hGraph, const char *path, unsigned int flags);
  // HIP: hipError_t hipGraphDebugDotPrint(hipGraph_t graph, const char* path, unsigned int flags);
  // CHECK: result = hipGraphDebugDotPrint(graph, name.c_str(), flags);
  result = cuGraphDebugDotPrint(graph, name.c_str(), flags);
#endif

#if CUDA_VERSION >= 11040
  // CUDA: CUresult CUDAAPI cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
  // HIP: hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t* pGraphExec, hipGraph_t graph, unsigned long long flags);
  // CHECK: result = hipGraphInstantiateWithFlags(&graphExec, graph, ull);
  result = cuGraphInstantiateWithFlags(&graphExec, graph, ull);

  // CHECK: hipGraphMemAttributeType graphMem_attribute;
  CUgraphMem_attribute graphMem_attribute;

  // CUDA: CUresult CUDAAPI cuDeviceGetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
  // HIP: hipError_t hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
  // CHECK: result = hipDeviceGetGraphMemAttribute(device, graphMem_attribute, image);
  result = cuDeviceGetGraphMemAttribute(device, graphMem_attribute, image);

  // CUDA: CUresult CUDAAPI cuDeviceSetGraphMemAttribute(CUdevice device, CUgraphMem_attribute attr, void* value);
  // HIP: hipError_t hipDeviceSetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value);
  // CHECK: result = hipDeviceSetGraphMemAttribute(device, graphMem_attribute, image);
  result = cuDeviceSetGraphMemAttribute(device, graphMem_attribute, image);

  // CUDA: CUresult CUDAAPI cuDeviceGraphMemTrim(CUdevice device);
  // HIP: hipError_t hipDeviceGraphMemTrim(int device);
  // CHECK: result = hipDeviceGraphMemTrim(device);
  result = cuDeviceGraphMemTrim(device);

  // CHECK: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS;
  CUDA_MEM_ALLOC_NODE_PARAMS MEM_ALLOC_NODE_PARAMS;

  // CUDA: CUresult CUDAAPI cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams);
  // HIP: hipError_t hipGraphAddMemAllocNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, hipMemAllocNodeParams* pNodeParams);
  // CHECK: result = hipGraphAddMemAllocNode(&graphNode, graph, &graphNode2, bytes, &MEM_ALLOC_NODE_PARAMS);
  result = cuGraphAddMemAllocNode(&graphNode, graph, &graphNode2, bytes, &MEM_ALLOC_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphMemAllocNodeGetParams(CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out);
  // HIP: hipError_t hipGraphMemAllocNodeGetParams(hipGraphNode_t node, hipMemAllocNodeParams* pNodeParams);
  // CHECK: result = hipGraphMemAllocNodeGetParams(graphNode, &MEM_ALLOC_NODE_PARAMS);
  result = cuGraphMemAllocNodeGetParams(graphNode, &MEM_ALLOC_NODE_PARAMS);

  // CUDA: CUresult CUDAAPI cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, CUdeviceptr dptr);
  // HIP: hipError_t hipDrvGraphAddMemFreeNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const hipGraphNode_t* pDependencies, size_t numDependencies, void* dev_ptr);
  // CHECK: result = hipDrvGraphAddMemFreeNode(&graphNode, graph, &graphNode2, bytes, deviceptr);
  result = cuGraphAddMemFreeNode(&graphNode, graph, &graphNode2, bytes, deviceptr);

  // CUDA: CUresult CUDAAPI cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out);
  // HIP: hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void* dev_ptr);
  // CHECK: result = hipGraphMemFreeNodeGetParams(graphNode, &deviceptr);
  result = cuGraphMemFreeNodeGetParams(graphNode, &deviceptr);
#endif

#if CUDA_VERSION >= 11040 && CUDA_VERSION < 12020
  // CHECK: hipMemAllocNodeParams MEM_ALLOC_NODE_PARAMS_st;
  CUDA_MEM_ALLOC_NODE_PARAMS_st MEM_ALLOC_NODE_PARAMS_st;
#endif

#if CUDA_VERSION >= 11060
  // CUDA: CUresult CUDAAPI cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int isEnabled);
  // HIP: hipError_t hipGraphNodeSetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int isEnabled);
  // CHECK: result = hipGraphNodeSetEnabled(graphExec, graphNode, flags);
  result = cuGraphNodeSetEnabled(graphExec, graphNode, flags);

  // CUDA: CUresult CUDAAPI cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode, unsigned int *isEnabled);
  // HIP: hipError_t hipGraphNodeGetEnabled(hipGraphExec_t hGraphExec, hipGraphNode_t hNode, unsigned int* isEnabled);
  // CHECK: result = hipGraphNodeGetEnabled(graphExec, graphNode, &flags);
  result = cuGraphNodeGetEnabled(graphExec, graphNode, &flags);
#endif

#if CUDA_VERSION >= 11070
  // CUDA: CUresult CUDAAPI cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  // CHECK: result = hipStreamWaitValue32(stream, deviceptr, u_value, flags);
  result = cuStreamWaitValue32_v2(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  // CHECK: result = hipStreamWaitValue64(stream, deviceptr, u_value, flags);
  result = cuStreamWaitValue64_v2(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask __dparm(0xFFFFFFFF));
  // CHECK: result = hipStreamWriteValue32(stream, deviceptr, u_value, flags);
  result = cuStreamWriteValue32_v2(stream, deviceptr, u_value, flags);

  // CUDA: CUresult CUDAAPI cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
  // HIP: hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags, uint64_t mask __dparm(0xFFFFFFFFFFFFFFFF));
  // CHECK: result = hipStreamWriteValue64(stream, deviceptr, u_value, flags);
  result = cuStreamWriteValue64_v2(stream, deviceptr, u_value, flags);
#endif

#if CUDA_VERSION >= 12000
  // TODO: https://github.com/ROCm-Developer-Tools/HIPIFY/issues/782 - Introduce 1-to-N conditional matcher
  //       Implement "conditional" matching in hipify-clang, based on CUDA_VERSION first;
  //       below the transformation cuStreamGetCaptureInfo -> hipStreamGetCaptureInfo_v2 should be applied for CUDA_VERSION >= 12000,
  //       otherwise, cuStreamGetCaptureInfo -> hipStreamGetCaptureInfo should be applied
  // CUDA < 12000: CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out);
  // CUDA:         CUresult CUDAAPI cuStreamGetCaptureInfo(CUstream hStream, CUstreamCaptureStatus *captureStatus_out, cuuint64_t *id_out, CUgraph *graph_out, const CUgraphNode **dependencies_out, size_t *numDependencies_out);
  // HIP: hipError_t hipStreamGetCaptureInfo_v2(hipStream_t stream, hipStreamCaptureStatus* captureStatus_out, unsigned long long* id_out __dparm(0), hipGraph_t* graph_out __dparm(0), const hipGraphNode_t** dependencies_out __dparm(0), size_t* numDependencies_out __dparm(0));
  //
  result = cuStreamGetCaptureInfo(stream, &streamCaptureStatus, &ull, &graph, &pGraphNode, &bytes);

  // NOTE: not implemented yet in HIP
  // CUDA < 12000: CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out, CUgraphExecUpdateResult *updateResult_out);
  // CUDA:         CUresult CUDAAPI cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph, CUgraphExecUpdateResultInfo *resultInfo);
  // HIP:

  // NOTE: not implemented yet in HIP
  // CUDA < 12000: CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, CUgraphNode *phErrorNode, char *logBuffer, size_t bufferSize);
  // CUDA:         CUresult CUDAAPI cuGraphInstantiate(CUgraphExec *phGraphExec, CUgraph hGraph, unsigned long long flags);
  // HIP:
#endif

  return 0;
}
