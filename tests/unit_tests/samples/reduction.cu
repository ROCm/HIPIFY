// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <iomanip>
#include <cmath>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

void check_error(void) {
  // CHECK: hipError_t err = hipGetLastError();
  cudaError_t err = cudaGetLastError();
  // CHECK: if (err != hipSuccess) {
  if (err != cudaSuccess) {
    std::cerr
      << "Error: "
      // CHECK: << hipGetErrorString(err)
      << cudaGetErrorString(err)
      << std::endl;
      exit(err);
  }
}

__global__ void atomic_reduction_kernel(int *in, int *out, int ARRAYSIZE) {
  int sum = int(0);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < ARRAYSIZE; i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  // CHECK: atomicAdd(out, sum);
  atomicAdd(out, sum);
}

__global__ void atomic_reduction_kernel2(int *in, int *out, int ARRAYSIZE) {
  int sum = int(0);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx * 16; i < ARRAYSIZE; i += blockDim.x * gridDim.x * 16) {
    sum += in[i] + in[i+1] + in[i+2] + in[i+3] + in[i+4] + in[i+5] + in[i+6] + in[i+7] + in[i+8] + in[i+9] + in[i+10]
         + in[i+11] + in[i+12] + in[i+13] + in[i+14] + in[i+15];
  }
  // CHECK: atomicAdd(out, sum);
  atomicAdd(out, sum);
}

__global__ void atomic_reduction_kernel3(int *in, int *out, int ARRAYSIZE) {
    int sum = int(0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx * 4; i < ARRAYSIZE; i += blockDim.x * gridDim.x * 4) {
      sum += in[i] + in[i+1] + in[i+2] + in[i+3];
    }
    // CHECK: atomicAdd(out, sum);
    atomicAdd(out, sum);
}

int main(int argc, char **argv) {
  unsigned int ARRAYSIZE = 52428800;
  if (argc < 2) {
    printf("Usage: ./reduction num_of_elems\n");
    printf("using default value: %d\n", ARRAYSIZE);
  } else
    ARRAYSIZE = atoi(argv[1]);
  int N = 10;
  printf("ARRAYSIZE: %d\n", ARRAYSIZE);
  std::cout << "Array size: " << ARRAYSIZE * sizeof(int) / 1024.0 / 1024.0 << " MB" << std::endl;
  int *Array = (int*)malloc(ARRAYSIZE * sizeof(int));
  int checksum = 0; 
  for (int i = 0; i < ARRAYSIZE; ++i) {
    Array[i] = rand()%2;
    checksum += Array[i];
  }
  int *in, *out;
  std::chrono::high_resolution_clock::time_point t1, t2;
  long long size = sizeof(int) * ARRAYSIZE;
  // CHECK: hipMalloc(&in, size);
  cudaMalloc(&in, size);
  // CHECK: hipMalloc(&out, sizeof(int));
  cudaMalloc(&out, sizeof(int));
  check_error();
  // CHECK: hipMemcpy(in, Array, ARRAYSIZE * sizeof(int), hipMemcpyHostToDevice);
  cudaMemcpy(in, Array, ARRAYSIZE * sizeof(int), cudaMemcpyHostToDevice);
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();
  check_error();
  // CHECK: hipDeviceProp_t props;
  cudaDeviceProp props;
  // CHECK: hipGetDeviceProperties(&props, 0);
  cudaGetDeviceProperties(&props, 0);
  int threads = 256;
  int blocks = std::min((ARRAYSIZE + threads - 1) / threads, 2048u);
  t1 = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < N; ++i) {
    // CHECK: hipMemsetAsync(out, 0, sizeof(int));
    cudaMemsetAsync(out, 0, sizeof(int));
    // CHECK: atomic_reduction_kernel<<<blocks, threads>>>(in, out, ARRAYSIZE);
    atomic_reduction_kernel<<<blocks, threads>>>(in, out, ARRAYSIZE);
    // CHECK: atomic_reduction_kernel2<<<blocks, threads>>>(in, out, ARRAYSIZE);
    atomic_reduction_kernel2<<<blocks, threads>>>(in, out, ARRAYSIZE);
    // CHECK: atomic_reduction_kernel3<<<blocks, threads>>>(in, out, ARRAYSIZE);
    atomic_reduction_kernel3<<<blocks, threads>>>(in, out, ARRAYSIZE);
    check_error();
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
    check_error();
  }
  t2 = std::chrono::high_resolution_clock::now();
  double times = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
  float GB = (float)ARRAYSIZE * sizeof(int) * N;
  std::cout << "The average performance of reduction is " << 1.0E-09 * GB / times << " GBytes/sec" << std::endl;
  int sum;
  // CHECK: hipMemcpy(&sum, out, sizeof(int), hipMemcpyDeviceToHost);
  cudaMemcpy(&sum, out, sizeof(int), cudaMemcpyDeviceToHost);
  check_error();
  if(sum == checksum)
    std::cout << "VERIFICATION: result is CORRECT" << std::endl << std::endl;
  else
    std::cout << "VERIFICATION: result is INCORRECT!!" << std::endl << std::endl;
  // CHECK: hipFree(in);
  cudaFree(in);
  // CHECK: hipFree(out);
  cudaFree(out);
  check_error();
  free(Array);
}
