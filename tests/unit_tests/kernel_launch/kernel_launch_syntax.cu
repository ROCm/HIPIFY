// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --hip-kernel-execution-syntax %clang_args

#include <iostream>
#include <algorithm>

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>

template<typename T>
__global__ void axpy(T a, T *x, T *y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

template<typename T1, typename T2>
__global__ void axpy_2(T1 a, T2 *x, T2 *y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

template<typename T>
__global__ void axpy_empty() {
}

__global__ void empty() {
}

__global__ void nonempty(int x, int y, int z) {
}

int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  // Copy input data to device.
  float* device_x;
  float* device_y;

  // CHECK: hipMalloc(&device_x, kDataLen * sizeof(float));
  cudaMalloc(&device_x, kDataLen * sizeof(float));
  // CHECK: hipMalloc(&device_y, kDataLen * sizeof(float));
  cudaMalloc(&device_y, kDataLen * sizeof(float));
  // CHECK: hipMemcpy(device_x, host_x, kDataLen * sizeof(float), hipMemcpyHostToDevice);
  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  int x = 1, y = 2, z = 3;
  size_t N = 32;
  // CHECK: hipStream_t stream = NULL;
  cudaStream_t stream = NULL;
  // CHECK: hipStreamCreate(&stream);
  cudaStreamCreate(&stream);

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<float><<<1, kDataLen>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<float><<<dim3(1), kDataLen>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<float><<<1, dim3(kDataLen)>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), 0, 0, a, device_x, device_y);
  axpy<float><<<dim3(1), dim3(kDataLen)>>>(a, device_x, device_y);

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, 0, a, device_x, device_y);
  axpy<float><<<1, kDataLen, N>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, 0, a, device_x, device_y);
  axpy<float><<<dim3(1), kDataLen, N>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, 0, a, device_x, device_y);
  axpy<float><<<1, dim3(kDataLen), N>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, 0, a, device_x, device_y);
  axpy<float><<<dim3(1), dim3(kDataLen), N>>>(a, device_x, device_y);

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, stream, a, device_x, device_y);
  axpy<float><<<1, kDataLen, N, stream>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, stream, a, device_x, device_y);
  axpy<float><<<dim3(1), kDataLen, N, stream>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, stream, a, device_x, device_y);
  axpy<float><<<1, dim3(kDataLen), N, stream>>>(a, device_x, device_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy<float>), dim3(1), dim3(kDataLen), N, stream, a, device_x, device_y);
  axpy<float><<<dim3(1), dim3(kDataLen), N, stream>>>(a, device_x, device_y);

  double h_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  double h_y[kDataLen];

  // Copy input data to device.
  double* d_x;
  double* d_y;

  // CHECK: hipMalloc(&d_x, kDataLen * sizeof(double));
  cudaMalloc(&d_x, kDataLen * sizeof(double));
  // CHECK: hipMalloc(&d_y, kDataLen * sizeof(double));
  cudaMalloc(&d_y, kDataLen * sizeof(double));
  // CHECK: hipMemcpy(d_x, h_x, kDataLen * sizeof(double), hipMemcpyHostToDevice);
  cudaMemcpy(d_x, h_x, kDataLen * sizeof(double), cudaMemcpyHostToDevice);

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(1), dim3(kDataLen*2+10), N*N, stream, a, d_x, d_y);
  axpy_2<float,double><<<1, kDataLen*2+10, N*N, stream>>>(a, d_x, d_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(1,1,1), dim3(kDataLen*2+10), N*N, stream, a, d_x, d_y);
  axpy_2<float,double><<<dim3(1,1,1), kDataLen*2+10, N*N, stream>>>(a, d_x, d_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(1), dim3(kDataLen*2+10), N*N, stream, a, d_x, d_y);
  axpy_2<float,double><<<1, dim3(kDataLen*2+10), N*N, stream>>>(a, d_x, d_y);
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(1,1,1), dim3(kDataLen*2+10), N*N, stream, a, d_x, d_y);
  axpy_2<float,double><<<dim3(1,1,1), dim3(kDataLen*2+10), N*N, stream>>>(a, d_x, d_y);


  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), 0, 0);
  axpy_empty<float><<<1, kDataLen>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), 0, 0);
  axpy_empty<float><<<dim3(1), kDataLen>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), 0, 0);
  axpy_empty<float><<<1, dim3(kDataLen)>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), 0, 0);
  axpy_empty<float><<<dim3(1), dim3(kDataLen)>>>();

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, 0);
  axpy_empty<float><<<1, kDataLen, N>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, 0);
  axpy_empty<float><<<dim3(1), kDataLen, N>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, 0);
  axpy_empty<float><<<1, dim3(kDataLen), N>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, 0);
  axpy_empty<float><<<dim3(1), dim3(kDataLen), N>>>();

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, stream);
  axpy_empty<float><<<1, kDataLen, N, stream>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, stream);
  axpy_empty<float><<<dim3(1), kDataLen, N, stream>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, stream);
  axpy_empty<float><<<1, dim3(kDataLen), N, stream>>>();
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_empty<float>), dim3(1), dim3(kDataLen), N, stream);
  axpy_empty<float><<<dim3(1), dim3(kDataLen), N, stream>>>();

  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), 0, 0);
  empty<<<1, kDataLen>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), 0, 0);
  empty<<<dim3(1), kDataLen>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), 0, 0);
  empty<<<1, dim3(kDataLen)>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), 0, 0);
  empty<<<dim3(1), dim3(kDataLen)>>> ( );

  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, 0);
  empty<<<1, kDataLen, N>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, 0);
  empty<<<dim3(1), kDataLen, N>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, 0);
  empty<<<1, dim3(kDataLen), N>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, 0);
  empty<<<dim3(1), dim3(kDataLen), N>>> ( );

  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, stream);
  empty<<<1, kDataLen, N, stream>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, stream);
  empty<<<dim3(1), kDataLen, N, stream>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, stream);
  empty<<<1, dim3(kDataLen), N, stream>>> ( );
  // CHECK: hipLaunchKernelGGL(empty, dim3(1), dim3(kDataLen), N, stream);
  empty<<<dim3(1), dim3(kDataLen), N, stream>>> ( );

  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), 0, 0, x, y, z);
  nonempty<<<1, kDataLen>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), 0, 0, x, y, z);
  nonempty<<<dim3(1), kDataLen>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), 0, 0, x, y, z);
  nonempty<<<1, dim3(kDataLen)>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), 0, 0, x, y, z);
  nonempty<<<dim3(1), dim3(kDataLen)>>> (x, y, z);

  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, 0, x, y, z);
  nonempty<<<1, kDataLen, N>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, 0, x, y, z);
  nonempty<<<dim3(1), kDataLen, N>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, 0, x, y, z);
  nonempty<<<1, dim3(kDataLen), N>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, 0, x, y, z);
  nonempty<<<dim3(1), dim3(kDataLen), N>>> (x, y, z);

  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, stream, x, y, z);
  nonempty<<<1, kDataLen, N, stream>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, stream, x, y, z);
  nonempty<<<dim3(1), kDataLen, N, stream>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, stream, x, y, z);
  nonempty<<<1, dim3(kDataLen), N, stream>>> (x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(1), dim3(kDataLen), N, stream, x, y, z);
  nonempty<<<dim3(1), dim3(kDataLen), N, stream>>> (x, y, z);

  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(x,y,z), dim3(std::min(kDataLen*2+10,x)), std::min(x,y), stream, a, std::min(d_x,d_y), std::max(d_x,d_y));
  axpy_2<float,double><<<dim3(x,y,z), std::min(kDataLen*2+10,x), std::min(x,y), stream>>>(a, std::min(d_x,d_y), std::max(d_x,d_y));
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(x,y,z), dim3(std::min(kDataLen*2+10,x)), std::min(x,y), 0, a, std::min(d_x,d_y), std::max(d_x,d_y));
  axpy_2<float,double><<<dim3(x,y,z), std::min(kDataLen*2+10,x), std::min(x,y)>>>(a, std::min(d_x,d_y), std::max(d_x,d_y));
  // CHECK: hipLaunchKernelGGL(HIP_KERNEL_NAME(axpy_2<float,double>), dim3(x,y,z), dim3(std::min(kDataLen*2+10,x)), 0, 0, a, std::min(d_x,d_y), std::max(d_x,d_y));
  axpy_2<float,double><<<dim3(x,y,z), std::min(kDataLen*2+10,x)>>>(a, std::min(d_x,d_y), std::max(d_x,d_y));

  // CHECK: hipLaunchKernelGGL(nonempty, dim3(x,y,z), dim3(x,y,std::min(y,z)), 0, 0, x, y, z);
  nonempty<<<dim3(x,y,z), dim3(x,y,std::min(y,z))>>>(x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(x,y,z), dim3(x,y,std::min(std::max(x,y),z)), 0, 0, x, y, z);
  nonempty<<<dim3(x,y,z), dim3(x,y,std::min(std::max(x,y),z))>>>(x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(x,y,z), dim3(x,y,std::min(std::max(x,int(N)),z)), 0, 0, x, y, z);
  nonempty<<<dim3(x,y,z), dim3(x,y,std::min(std::max(x,int(N)),z))>>>(x, y, z);
  // CHECK: hipLaunchKernelGGL(nonempty, dim3(x,y,z), dim3(x,y,std::min(std::max(x,int(N+N -x/y + y*1)),z)), 0, 0, x, y, z);
  nonempty<<<dim3(x,y,z), dim3(x,y,std::min(std::max(x,int(N+N -x/y + y*1)),z))>>>(x, y, z);

  // Copy output data to host.
  // CHECK: hipDeviceSynchronize();
  cudaDeviceSynchronize();

  // CHECK: hipMemcpy(host_y, device_y, kDataLen * sizeof(float), hipMemcpyDeviceToHost);
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  // Print the results.
  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  // CHECK: hipDeviceReset();
  cudaDeviceReset();
  return 0;
}
