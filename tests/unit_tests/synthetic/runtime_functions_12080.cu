// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>

int main() {
  printf("12.12080. CUDA Runtime API Functions synthetic test for CUDA >= 12080\n");

  // CHECK: hipError_t result = hipSuccess;
  cudaError result = cudaSuccess;

  // CHECK: hipLibrary_t library;
  cudaLibrary_t library;
  // CHECK: hipKernel_t* kernelArray;
  cudaKernel_t* kernelArray;
  unsigned int numKernels = 0;

#if CUDA_VERSION >= 12000
  // CUDA:extern __host__cudaError_t cudaLibraryEnumerateKernels(cudaKernel_t* kernels, unsigned int numKernels, cudaLibrary_t lib)
  // HIP: hipError_t hipLibraryEnumerateKernels(hipKernel_t* kernels, unsigned int numKernels, hipLibrary_t lib)
  // CHECK: result = hipLibraryEnumerateKernels(kernelArray, numKernels, library);
  result = cudaLibraryEnumerateKernels(kernelArray, numKernels, library);
#endif

  return 0;
}
