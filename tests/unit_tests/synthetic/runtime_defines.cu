// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime_api.h>
// CHECK-NEXT: #include <hip/device_functions.h>
// CHECK-NEXT: #include <hip/hip_math_constants.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <math_constants.h>
#include <stdio.h>

  __global__ __constant__ float INF_F;
  __global__ __constant__ float NAN_F;
  __global__ __constant__ float MIN_DENORM_F;
  __global__ __constant__ float MAX_NORMAL_F;
  __global__ __constant__ float NEG_ZERO_F;
  __global__ __constant__ float ZERO_F;
  __global__ __constant__ float ONE_F;
  __global__ __constant__ float SQRT_HALF_F;
  __global__ __constant__ float SQRT_HALF_HI_F;
  __global__ __constant__ float SQRT_HALF_LO_F;
  __global__ __constant__ float SQRT_TWO_F;
  __global__ __constant__ float THIRD_F;
  __global__ __constant__ float PIO4_F;
  __global__ __constant__ float PIO2_F;
  __global__ __constant__ float _3PIO4_F;
  __global__ __constant__ float _2_OVER_PI_F;
  __global__ __constant__ float SQRT_2_OVER_PI_F;
  __global__ __constant__ float PI_F;
  __global__ __constant__ float L2E_F;
  __global__ __constant__ float L2T_F;
  __global__ __constant__ float LG2_F;
  __global__ __constant__ float LGE_F;
  __global__ __constant__ float LN2_F;
  __global__ __constant__ float LNT_F;
  __global__ __constant__ float LNPI_F;
  __global__ __constant__ float TWO_TO_M126_F;
  __global__ __constant__ float TWO_TO_126_F;
  __global__ __constant__ float NORM_HUGE_F;
  __global__ __constant__ float TWO_TO_23_F;
  __global__ __constant__ float TWO_TO_24_F;
  __global__ __constant__ float TWO_TO_31_F;
  __global__ __constant__ float TWO_TO_32_F;
  __global__ __constant__ float REMQUO_BITS_F;
  __global__ __constant__ float REMQUO_MASK_F;
  __global__ __constant__ double TRIG_PLOSS_F;
  __global__ __constant__ double INF;
  __global__ __constant__ double NAN_;
  __global__ __constant__ double NEG_ZERO;
  __global__ __constant__ double MIN_DENORM;
  __global__ __constant__ double ZERO;
  __global__ __constant__ double ONE;
  __global__ __constant__ double SQRT_TWO;
  __global__ __constant__ double SQRT_HALF;
  __global__ __constant__ double SQRT_HALF_HI;
  __global__ __constant__ double SQRT_HALF_LO;
  __global__ __constant__ double THIRD;
  __global__ __constant__ double TWOTHIRD;
  __global__ __constant__ double PIO4;
  __global__ __constant__ double PIO4_HI;
  __global__ __constant__ double PIO4_LO;
  __global__ __constant__ double PIO2;
  __global__ __constant__ double PIO2_HI;
  __global__ __constant__ double PIO2_LO;
  __global__ __constant__ double _3PIO4;
  __global__ __constant__ double _2_OVER_PI;
  __global__ __constant__ double _PI;
  __global__ __constant__ double PI_HI;
  __global__ __constant__ double PI_LO;
  __global__ __constant__ double SQRT_2PI;
  __global__ __constant__ double SQRT_2PI_HI;
  __global__ __constant__ double SQRT_2PI_LO;
  __global__ __constant__ double SQRT_PIO2;
  __global__ __constant__ double SQRT_PIO2_HI;
  __global__ __constant__ double SQRT_PIO2_LO;
  __global__ __constant__ double SQRT_2OPI;
  __global__ __constant__ double L2E;
  __global__ __constant__ double L2E_HI;
  __global__ __constant__ double L2E_LO;
  __global__ __constant__ double L2T;
  __global__ __constant__ double LG2;
  __global__ __constant__ double LG2_HI;
  __global__ __constant__ double LG2_LO;

__global__ void init() {
  // CHECK: INF_F = HIP_INF_F;
  // CHECK-NEXT: NAN_F = HIP_NAN_F;
  // CHECK-NEXT: MIN_DENORM_F = HIP_MIN_DENORM_F;
  // CHECK-NEXT: MAX_NORMAL_F = HIP_MAX_NORMAL_F;
  // CHECK-NEXT: NEG_ZERO_F = HIP_NEG_ZERO_F;
  // CHECK-NEXT: ZERO_F = HIP_ZERO_F;
  // CHECK-NEXT: ONE_F = HIP_ONE_F;
  // CHECK-NEXT: SQRT_HALF_F = HIP_SQRT_HALF_F;
  // CHECK-NEXT: SQRT_HALF_HI_F = HIP_SQRT_HALF_HI_F;
  // CHECK-NEXT: SQRT_HALF_LO_F = HIP_SQRT_HALF_LO_F;
  // CHECK-NEXT: SQRT_TWO_F = HIP_SQRT_TWO_F;
  // CHECK-NEXT: THIRD_F = HIP_THIRD_F;
  // CHECK-NEXT: PIO4_F = HIP_PIO4_F;
  // CHECK-NEXT: PIO2_F = HIP_PIO2_F;
  // CHECK-NEXT: _3PIO4_F = HIP_3PIO4_F;
  // CHECK-NEXT: _2_OVER_PI_F = HIP_2_OVER_PI_F;
  // CHECK-NEXT: SQRT_2_OVER_PI_F = HIP_SQRT_2_OVER_PI_F;
  // CHECK-NEXT: PI_F = HIP_PI_F;
  // CHECK-NEXT: L2E_F = HIP_L2E_F;
  // CHECK-NEXT: L2T_F = HIP_L2T_F;
  // CHECK-NEXT: LG2_F = HIP_LG2_F;
  // CHECK-NEXT: LGE_F = HIP_LGE_F;
  // CHECK-NEXT: LN2_F = HIP_LN2_F;
  // CHECK-NEXT: LNT_F = HIP_LNT_F;
  // CHECK-NEXT: LNPI_F = HIP_LNPI_F;
  // CHECK-NEXT: TWO_TO_M126_F = HIP_TWO_TO_M126_F;
  // CHECK-NEXT: TWO_TO_126_F = HIP_TWO_TO_126_F;
  // CHECK-NEXT: NORM_HUGE_F = HIP_NORM_HUGE_F;
  // CHECK-NEXT: TWO_TO_23_F = HIP_TWO_TO_23_F;
  // CHECK-NEXT: TWO_TO_24_F = HIP_TWO_TO_24_F;
  // CHECK-NEXT: TWO_TO_31_F = HIP_TWO_TO_31_F;
  // CHECK-NEXT: TWO_TO_32_F = HIP_TWO_TO_32_F;
  // CHECK-NEXT: REMQUO_BITS_F = HIP_REMQUO_BITS_F;
  // CHECK-NEXT: REMQUO_MASK_F = HIP_REMQUO_MASK_F;
  // CHECK-NEXT: TRIG_PLOSS_F = HIP_TRIG_PLOSS_F;
  // CHECK-NEXT: INF = HIP_INF;
  // CHECK-NEXT: NAN_ = HIP_NAN;
  // CHECK-NEXT: NEG_ZERO = HIP_NEG_ZERO;
  // CHECK-NEXT: MIN_DENORM = HIP_MIN_DENORM;
  // CHECK-NEXT: ZERO = HIP_ZERO;
  // CHECK-NEXT: ONE = HIP_ONE;
  // CHECK-NEXT: SQRT_TWO = HIP_SQRT_TWO;
  // CHECK-NEXT: SQRT_HALF = HIP_SQRT_HALF;
  // CHECK-NEXT: SQRT_HALF_HI = HIP_SQRT_HALF_HI;
  // CHECK-NEXT: SQRT_HALF_LO = HIP_SQRT_HALF_LO;
  // CHECK-NEXT: THIRD = HIP_THIRD;
  // CHECK-NEXT: TWOTHIRD = HIP_TWOTHIRD;
  // CHECK-NEXT: PIO4 = HIP_PIO4;
  // CHECK-NEXT: PIO4_HI = HIP_PIO4_HI;
  // CHECK-NEXT: PIO4_LO = HIP_PIO4_LO;
  // CHECK-NEXT: PIO2 = HIP_PIO2;
  // CHECK-NEXT: PIO2_HI = HIP_PIO2_HI;
  // CHECK-NEXT: PIO2_LO = HIP_PIO2_LO;
  // CHECK-NEXT: _3PIO4 = HIP_3PIO4;
  // CHECK-NEXT: _2_OVER_PI = HIP_2_OVER_PI;
  // CHECK-NEXT: _PI = HIP_PI;
  // CHECK-NEXT: PI_HI = HIP_PI_HI;
  // CHECK-NEXT: PI_LO = HIP_PI_LO;
  // CHECK-NEXT: SQRT_2PI = HIP_SQRT_2PI;
  // CHECK-NEXT: SQRT_2PI_HI = HIP_SQRT_2PI_HI;
  // CHECK-NEXT: SQRT_2PI_LO = HIP_SQRT_2PI_LO;
  // CHECK-NEXT: SQRT_PIO2 = HIP_SQRT_PIO2;
  // CHECK-NEXT: SQRT_PIO2_HI = HIP_SQRT_PIO2_HI;
  // CHECK-NEXT: SQRT_PIO2_LO = HIP_SQRT_PIO2_LO;
  // CHECK-NEXT: SQRT_2OPI = HIP_SQRT_2OPI;
  // CHECK-NEXT: L2E = HIP_L2E;
  // CHECK-NEXT: L2E_HI = HIP_L2E_HI;
  // CHECK-NEXT: L2E_LO = HIP_L2E_LO;
  // CHECK-NEXT: L2T = HIP_L2T;
  // CHECK-NEXT: LG2 = HIP_LG2;
  // CHECK-NEXT: LG2_HI = HIP_LG2_HI;
  // CHECK-NEXT: LG2_LO = HIP_LG2_LO;
  INF_F = CUDART_INF_F;
  NAN_F = CUDART_NAN_F;
  MIN_DENORM_F = CUDART_MIN_DENORM_F;
  MAX_NORMAL_F = CUDART_MAX_NORMAL_F;
  NEG_ZERO_F = CUDART_NEG_ZERO_F;
  ZERO_F = CUDART_ZERO_F;
  ONE_F = CUDART_ONE_F;
  SQRT_HALF_F = CUDART_SQRT_HALF_F;
  SQRT_HALF_HI_F = CUDART_SQRT_HALF_HI_F;
  SQRT_HALF_LO_F = CUDART_SQRT_HALF_LO_F;
  SQRT_TWO_F = CUDART_SQRT_TWO_F;
  THIRD_F = CUDART_THIRD_F;
  PIO4_F = CUDART_PIO4_F;
  PIO2_F = CUDART_PIO2_F;
  _3PIO4_F = CUDART_3PIO4_F;
  _2_OVER_PI_F = CUDART_2_OVER_PI_F;
  SQRT_2_OVER_PI_F = CUDART_SQRT_2_OVER_PI_F;
  PI_F = CUDART_PI_F;
  L2E_F = CUDART_L2E_F;
  L2T_F = CUDART_L2T_F;
  LG2_F = CUDART_LG2_F;
  LGE_F = CUDART_LGE_F;
  LN2_F = CUDART_LN2_F;
  LNT_F = CUDART_LNT_F;
  LNPI_F = CUDART_LNPI_F;
  TWO_TO_M126_F = CUDART_TWO_TO_M126_F;
  TWO_TO_126_F = CUDART_TWO_TO_126_F;
  NORM_HUGE_F = CUDART_NORM_HUGE_F;
  TWO_TO_23_F = CUDART_TWO_TO_23_F;
  TWO_TO_24_F = CUDART_TWO_TO_24_F;
  TWO_TO_31_F = CUDART_TWO_TO_31_F;
  TWO_TO_32_F = CUDART_TWO_TO_32_F;
  REMQUO_BITS_F = CUDART_REMQUO_BITS_F;
  REMQUO_MASK_F = CUDART_REMQUO_MASK_F;
  TRIG_PLOSS_F = CUDART_TRIG_PLOSS_F;
  INF = CUDART_INF;
  NAN_ = CUDART_NAN;
  NEG_ZERO = CUDART_NEG_ZERO;
  MIN_DENORM = CUDART_MIN_DENORM;
  ZERO = CUDART_ZERO;
  ONE = CUDART_ONE;
  SQRT_TWO = CUDART_SQRT_TWO;
  SQRT_HALF = CUDART_SQRT_HALF;
  SQRT_HALF_HI = CUDART_SQRT_HALF_HI;
  SQRT_HALF_LO = CUDART_SQRT_HALF_LO;
  THIRD = CUDART_THIRD;
  TWOTHIRD = CUDART_TWOTHIRD;
  PIO4 = CUDART_PIO4;
  PIO4_HI = CUDART_PIO4_HI;
  PIO4_LO = CUDART_PIO4_LO;
  PIO2 = CUDART_PIO2;
  PIO2_HI = CUDART_PIO2_HI;
  PIO2_LO = CUDART_PIO2_LO;
  _3PIO4 = CUDART_3PIO4;
  _2_OVER_PI = CUDART_2_OVER_PI;
  _PI = CUDART_PI;
  PI_HI = CUDART_PI_HI;
  PI_LO = CUDART_PI_LO;
  SQRT_2PI = CUDART_SQRT_2PI;
  SQRT_2PI_HI = CUDART_SQRT_2PI_HI;
  SQRT_2PI_LO = CUDART_SQRT_2PI_LO;
  SQRT_PIO2 = CUDART_SQRT_PIO2;
  SQRT_PIO2_HI = CUDART_SQRT_PIO2_HI;
  SQRT_PIO2_LO = CUDART_SQRT_PIO2_LO;
  SQRT_2OPI = CUDART_SQRT_2OPI;
  L2E = CUDART_L2E;
  L2E_HI = CUDART_L2E_HI;
  L2E_LO = CUDART_L2E_LO;
  L2T = CUDART_L2T;
  LG2 = CUDART_LG2;
  LG2_HI = CUDART_LG2_HI;
  LG2_LO = CUDART_LG2_LO;
}

int main() {
  printf("08. CUDA Runtime API Defines synthetic test\n");

  // CHECK: int IPC_HANDLE_SIZE = HIP_IPC_HANDLE_SIZE;
  // CHECK-NEXT: int ArrayDefault = hipArrayDefault;
  // CHECK-NEXT: int ArrayLayered = hipArrayLayered;
  // CHECK-NEXT: int ArraySurfaceLoadStore = hipArraySurfaceLoadStore;
  // CHECK-NEXT: int ArrayCubemap = hipArrayCubemap;
  // CHECK-NEXT: int ArrayTextureGather = hipArrayTextureGather;
  // CHECK-NEXT: int DeviceBlockingSync = hipDeviceScheduleBlockingSync;
  // CHECK-NEXT: int DeviceLmemResizeToMax = hipDeviceLmemResizeToMax;
  // CHECK-NEXT: int DeviceMapHost = hipDeviceMapHost;
  // CHECK-NEXT: int DeviceScheduleAuto = hipDeviceScheduleAuto;
  // CHECK-NEXT: int DeviceScheduleSpin = hipDeviceScheduleSpin;
  // CHECK-NEXT: int DeviceScheduleYield = hipDeviceScheduleYield;
  // CHECK-NEXT: int DeviceScheduleBlockingSync = hipDeviceScheduleBlockingSync;
  // CHECK-NEXT: int DeviceScheduleMask = hipDeviceScheduleMask;
  // CHECK-NEXT: int EventDefault = hipEventDefault;
  // CHECK-NEXT: int EventBlockingSync = hipEventBlockingSync;
  // CHECK-NEXT: int EventDisableTiming = hipEventDisableTiming;
  // CHECK-NEXT: int EventInterprocess = hipEventInterprocess;
  // CHECK-NEXT: int HostAllocDefault = hipHostMallocDefault;
  // CHECK-NEXT: int HostAllocPortable = hipHostMallocPortable;
  // CHECK-NEXT: int HostAllocMapped = hipHostMallocMapped;
  // CHECK-NEXT: int HostAllocWriteCombined = hipHostMallocWriteCombined;
  // CHECK-NEXT: int HostRegisterDefault = hipHostRegisterDefault;
  // CHECK-NEXT: int HostRegisterPortable = hipHostRegisterPortable;
  // CHECK-NEXT: int HostRegisterMapped = hipHostRegisterMapped;
  // CHECK-NEXT: int IpcMemLazyEnablePeerAccess = hipIpcMemLazyEnablePeerAccess;
  // CHECK-NEXT: int MemAttachGlobal = hipMemAttachGlobal;
  // CHECK-NEXT: int MemAttachHost = hipMemAttachHost;
  // CHECK-NEXT: int MemAttachSingle = hipMemAttachSingle;
  // CHECK-NEXT: int TextureType1D = hipTextureType1D;
  // CHECK-NEXT: int TextureType2D = hipTextureType2D;
  // CHECK-NEXT: int TextureType3D = hipTextureType3D;
  // CHECK-NEXT: int TextureTypeCubemap = hipTextureTypeCubemap;
  // CHECK-NEXT: int TextureType1DLayered = hipTextureType1DLayered;
  // CHECK-NEXT: int TextureType2DLayered = hipTextureType2DLayered;
  // CHECK-NEXT: int TextureTypeCubemapLayered = hipTextureTypeCubemapLayered;
  // CHECK-NEXT: int OccupancyDefault = hipOccupancyDefault;
  // CHECK-NEXT: int OccupancyDisableCachingOverride = hipOccupancyDisableCachingOverride;
  // CHECK-NEXT: int StreamDefault = hipStreamDefault;
  // CHECK-NEXT: int StreamNonBlocking = hipStreamNonBlocking;
  // CHECK-NEXT: hipStream_t StreamPerThread = hipStreamPerThread;
  int IPC_HANDLE_SIZE = CUDA_IPC_HANDLE_SIZE;
  int ArrayDefault = cudaArrayDefault;
  int ArrayLayered = cudaArrayLayered;
  int ArraySurfaceLoadStore = cudaArraySurfaceLoadStore;
  int ArrayCubemap = cudaArrayCubemap;
  int ArrayTextureGather = cudaArrayTextureGather;
  int DeviceBlockingSync = cudaDeviceBlockingSync;
  int DeviceLmemResizeToMax = cudaDeviceLmemResizeToMax;
  int DeviceMapHost = cudaDeviceMapHost;
  int DeviceScheduleAuto = cudaDeviceScheduleAuto;
  int DeviceScheduleSpin = cudaDeviceScheduleSpin;
  int DeviceScheduleYield = cudaDeviceScheduleYield;
  int DeviceScheduleBlockingSync = cudaDeviceScheduleBlockingSync;
  int DeviceScheduleMask = cudaDeviceScheduleMask;
  int EventDefault = cudaEventDefault;
  int EventBlockingSync = cudaEventBlockingSync;
  int EventDisableTiming = cudaEventDisableTiming;
  int EventInterprocess = cudaEventInterprocess;
  int HostAllocDefault = cudaHostAllocDefault;
  int HostAllocPortable = cudaHostAllocPortable;
  int HostAllocMapped = cudaHostAllocMapped;
  int HostAllocWriteCombined = cudaHostAllocWriteCombined;
  int HostRegisterDefault = cudaHostRegisterDefault;
  int HostRegisterPortable = cudaHostRegisterPortable;
  int HostRegisterMapped = cudaHostRegisterMapped;
  int IpcMemLazyEnablePeerAccess = cudaIpcMemLazyEnablePeerAccess;
  int MemAttachGlobal = cudaMemAttachGlobal;
  int MemAttachHost = cudaMemAttachHost;
  int MemAttachSingle = cudaMemAttachSingle;
  int TextureType1D = cudaTextureType1D;
  int TextureType2D = cudaTextureType2D;
  int TextureType3D = cudaTextureType3D;
  int TextureTypeCubemap = cudaTextureTypeCubemap;
  int TextureType1DLayered = cudaTextureType1DLayered;
  int TextureType2DLayered = cudaTextureType2DLayered;
  int TextureTypeCubemapLayered = cudaTextureTypeCubemapLayered;
  int OccupancyDefault = cudaOccupancyDefault;
  int OccupancyDisableCachingOverride = cudaOccupancyDisableCachingOverride;
  int StreamDefault = cudaStreamDefault;
  int StreamNonBlocking = cudaStreamNonBlocking;
  cudaStream_t StreamPerThread = cudaStreamPerThread;

#if CUDA_VERSION >= 7050
  // CHECK: int HostRegisterIoMemory = hipHostRegisterIoMemory;
  int HostRegisterIoMemory = cudaHostRegisterIoMemory;
#endif

#if CUDA_VERSION >= 8000
  // CHECK: int CpuDeviceId = hipCpuDeviceId;
  // CHECK-NEXT: int InvalidDeviceId = hipInvalidDeviceId;
  int CpuDeviceId = cudaCpuDeviceId;
  int InvalidDeviceId = cudaInvalidDeviceId;
#endif

#if CUDA_VERSION >= 9000
  // CHECK: int CooperativeLaunchMultiDeviceNoPreSync = hipCooperativeLaunchMultiDeviceNoPreSync;
  // CHECK-NEXT: int CooperativeLaunchMultiDeviceNoPostSync = hipCooperativeLaunchMultiDeviceNoPostSync;
  int CooperativeLaunchMultiDeviceNoPreSync = cudaCooperativeLaunchMultiDeviceNoPreSync;
  int CooperativeLaunchMultiDeviceNoPostSync = cudaCooperativeLaunchMultiDeviceNoPostSync;

  // CHECK: hipStream_t StreamLegacy = hipStreamLegacy;
  cudaStream_t StreamLegacy = cudaStreamLegacy;
#endif

#if CUDA_VERSION >= 10000
  // CHECK: int EXTERNAL_MEMORY_DEDICATED = hipExternalMemoryDedicated;
  int EXTERNAL_MEMORY_DEDICATED = cudaExternalMemoryDedicated;
#endif

#if CUDA_VERSION >= 11010
  // CHECK: int HostRegisterReadOnly = hipHostRegisterReadOnly;
  int HostRegisterReadOnly = cudaHostRegisterReadOnly;
#endif

#if CUDA_VERSION >= 12030
  // CHECK: int GRAPH_KERNEL_NODE_PORT_DEFAULT = hipGraphKernelNodePortDefault;
  int GRAPH_KERNEL_NODE_PORT_DEFAULT = cudaGraphKernelNodePortDefault;

  // CHECK: int GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER = hipGraphKernelNodePortLaunchCompletion;
  int GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER = cudaGraphKernelNodePortLaunchCompletion;

  // CHECK: int GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC = hipGraphKernelNodePortProgrammatic;
  int GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC = cudaGraphKernelNodePortProgrammatic;
#endif

  return 0;
}
