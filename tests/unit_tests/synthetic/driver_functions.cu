// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
#include <string>

int main() {
  printf("09. CUDA Driver API Functions synthetic test\n");

  unsigned int flags = 0;
  size_t bytes = 0;
  void* image = nullptr;
  std::string name = "function";
  // CHECK: hipDevice_t device;
  // CHECK-NEXT: hipCtx_t context;
  // CHECK-NEXT: hipFuncCache_t func_cache;
  // CHECK-NEXT: hipLimit_t limit;
  // CHECK-NEXT: hipSharedMemConfig pconfig;
  // CHECK-NEXT: hipFunction_t function;
  // CHECK-NEXT: hipModule_t module_;
  // CHECK-NEXT: hipDeviceptr_t deviceptr;
  // CHECK-NEXT: hipTexRef texref;
  // CHECK-NEXT: hipJitOption jit_option;
  CUdevice device;
  CUcontext context;
  CUfunc_cache func_cache;
  CUlimit limit;
  CUsharedconfig pconfig;
  CUfunction function;
  CUmodule module_;
  CUdeviceptr deviceptr;
  CUtexref texref;
  CUjit_option jit_option;

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
  // CHECK-NEXT: result = hipDevicePrimaryCtxRelease(device);
  result = cuDevicePrimaryCtxRelease(device);
  result = cuDevicePrimaryCtxRelease_v2(device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxReset(device);
  // CHECK-NEXT: result = hipDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset(device);
  result = cuDevicePrimaryCtxReset_v2(device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev);
  // HIP: hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);
  // CHECK: result = hipDevicePrimaryCtxRetain(&context, device);
  result = cuDevicePrimaryCtxRetain(&context, device);

  // CUDA: CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags);
  // HIP: hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);
  // CHECK: result = hipDevicePrimaryCtxSetFlags(device, flags);
  // CHECK-NEXT: result = hipDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags(device, flags);
  result = cuDevicePrimaryCtxSetFlags_v2(device, flags);

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
  // CHECK: result = hipModuleGetGlobal(&deviceptr, &bytes, module_, name.c_str());
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

  return 0;
}
