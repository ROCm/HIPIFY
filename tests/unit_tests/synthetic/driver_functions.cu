// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

int main() {
  printf("09. CUDA Driver API Functions synthetic test\n");

  unsigned int flags = 0;
  // CHECK: hipDevice_t device;
  // CHECK-NEXT: hipCtx_t context;
  // CHECK-NEXT: hipFuncCache_t func_cache;
  CUdevice device;
  CUcontext context;
  CUfunc_cache func_cache;

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

  size_t bytes = 0;
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

  return 0;
}
