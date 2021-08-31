// RUN: %run_test hipify "%s" "%t" %hipify_args -D__CUDA_API_VERSION_INTERNAL %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>

int main() {
  printf("08. CUDA Runtime API Defines synthetic test\n");

  // CHECK: int IPC_HANDLE_SIZE = HIP_IPC_HANDLE_SIZE;
  // CHECK-NEXT: int ArrayDefault = hipArrayDefault;
  // CHECK-NEXT: int ArrayLayered = hipArrayLayered;
  // CHECK-NEXT: int ArraySurfaceLoadStore = hipArraySurfaceLoadStore;
  // CHECK-NEXT: int ArrayCubemap = hipArrayCubemap;
  // CHECK-NEXT: int ArrayTextureGather = hipArrayTextureGather;
  // CHECK-NEXT: int CooperativeLaunchMultiDeviceNoPreSync = hipCooperativeLaunchMultiDeviceNoPreSync;
  // CHECK-NEXT: int CooperativeLaunchMultiDeviceNoPostSync = hipCooperativeLaunchMultiDeviceNoPostSync;
  // CHECK-NEXT: int CpuDeviceId = hipCpuDeviceId;
  // CHECK-NEXT: int InvalidDeviceId = hipInvalidDeviceId;
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
  int IPC_HANDLE_SIZE = CUDA_IPC_HANDLE_SIZE;
  int ArrayDefault = cudaArrayDefault;
  int ArrayLayered = cudaArrayLayered;
  int ArraySurfaceLoadStore = cudaArraySurfaceLoadStore;
  int ArrayCubemap = cudaArrayCubemap;
  int ArrayTextureGather = cudaArrayTextureGather;
  int CooperativeLaunchMultiDeviceNoPreSync = cudaCooperativeLaunchMultiDeviceNoPreSync;
  int CooperativeLaunchMultiDeviceNoPostSync = cudaCooperativeLaunchMultiDeviceNoPostSync;
  int CpuDeviceId = cudaCpuDeviceId;
  int InvalidDeviceId = cudaInvalidDeviceId;
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

  return 0;
}
