// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>

int main() {
  printf("12.9000. CUDA Runtime API Functions synthetic test for CUDA >= 9000\n");

  // CHECK: hipError_t result = hipSuccess;
  cudaError result = cudaSuccess;

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  int intVal = 0;
  unsigned int flags = 0;
  dim3 gridDim;
  dim3 blockDim;
  void* func = nullptr;
  void* image = nullptr;

#if CUDA_VERSION >= 9000
  // CHECK: hipFuncAttribute FuncAttribute;
  cudaFuncAttribute FuncAttribute;

  // CUDA: extern __host__ __cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);
  // HIP: hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value);
  // CHECK: result = hipFuncSetAttribute(reinterpret_cast<const void*>(func), FuncAttribute, intVal);
  result = cudaFuncSetAttribute(func, FuncAttribute, intVal);

  // CUDA: extern __host__ cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
  // HIP: hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream);
  // CHECK: result = hipLaunchCooperativeKernel(reinterpret_cast<const void*>(func), gridDim, blockDim, &image, flags, stream);
  result = cudaLaunchCooperativeKernel(func, gridDim, blockDim, &image, flags, stream);
#endif

  return 0;
}
