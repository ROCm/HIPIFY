// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --skip-excluded-preprocessor-conditional-blocks %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>
#include <stdio.h>

int main() {
  printf("09. CUDA Driver API Functions synthetic test\n");

  unsigned int flags = 0;
  size_t bytes = 0;
  size_t bytes_2 = 0;
  void* image = nullptr;
  std::string name = "str";
  // CHECK: hipDevice_t device;
  // CHECK-NEXT: hipCtx_t context;
  // CHECK-NEXT: hipFuncCache_t func_cache;
  // CHECK-NEXT: hipLimit_t limit;
  // CHECK-NEXT: hipSharedMemConfig pconfig;
  // CHECK-NEXT: hipFunction_t function;
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
  // CHECK-NEXT: hipIpcMemHandle_t ipcMemHandle;
  // CHECK-NEXT: hip_Memcpy2D MEMCPY2D;
  // CHECK-NEXT: HIP_MEMCPY3D MEMCPY3D;
  // CHECK-NEXT: hipStream_t stream;
  // CHECK-NEXT: hipMipmappedArray_t mipmappedArray;
  CUdevice device;
  CUcontext context;
  CUfunc_cache func_cache;
  CUlimit limit;
  CUsharedconfig pconfig;
  CUfunction function;
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
  CUipcMemHandle ipcMemHandle;
  CUDA_MEMCPY2D MEMCPY2D;
  CUDA_MEMCPY3D MEMCPY3D;
  CUstream stream;
  CUmipmappedArray mipmappedArray;

  // CUDA: CUresult CUDAAPI cuInit(unsigned int Flags);
  // HIP: hipError_t hipInit(unsigned int flags);
  // CHECK: hipError_t result = hipInit(flags);
  CUresult result = cuInit(flags);

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

  int count = 0;
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
#if CUDA_VERSION > 10020
  // CHECK: result = hipDevicePrimaryCtxRelease(device);
  result = cuDevicePrimaryCtxRelease_v2(device);
#endif

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset(device);
#if CUDA_VERSION > 10020
  // CHECK: result = hipDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset_v2(device);
#endif

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxRetain(&context, device);
  result = cuDevicePrimaryCtxRetain(&context, device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
  // HIP: hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
  // CHECK: result = hipDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags(device, flags);
#if CUDA_VERSION > 10020
  // CHECK: result = hipDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags_v2(device, flags);
#endif

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

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////// TODO: Get rid of additional attribute 'unsigned int flags' used by HIP without a default value ///////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // CUDA: CUresult CUDAAPI cuMemAllocHost(void **pp, size_t bytesize);
  // HIP: DEPRECATED("use hipHostMalloc instead") hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
  // TODO: should be hipHostAlloc(&image, bytes, 0);
  // CHECK: result = hipHostAlloc(&image, bytes);
  // CHECK-NEXT: result = hipHostAlloc(&image, bytes);
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

  return 0;
}
