// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --hip-kernel-execution-syntax %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void my_kernel(int arg) {}

void test_launches() {
    struct { int x, y; } config = {1, 2};
    dim3 grid_arr[2] = {dim3(1), dim3(2)};
    int shared_mem = 0;

    // CHECK: hipStream_t stream = 0;
    cudaStream_t stream = 0;

    // 1. Array indexing
    // CHECK: hipLaunchKernelGGL(my_kernel, dim3(grid_arr[0]), dim3(grid_arr[1]), 0, 0, 0);
    my_kernel<<<grid_arr[0], grid_arr[1]>>>(0);

    // 2. Member access
    // CHECK: hipLaunchKernelGGL(my_kernel, dim3(config.x), dim3(config.y), 0, 0, 0);
    my_kernel<<<dim3(config.x), dim3(config.y)>>>(0);

    // 3. Math, Logical, and Ternary operators
    // CHECK: hipLaunchKernelGGL(my_kernel, dim3(config.x == 1 ? 2 : 1), dim3(256), 0, 0, 0);
    my_kernel<<<config.x == 1 ? 2 : 1, 256>>>(0);

    // 4. Address-of operator
    // CHECK: hipLaunchKernelGGL(my_kernel, dim3(1), dim3(256), shared_mem, &stream, 0);
    my_kernel<<<1, 256, shared_mem, &stream>>>(0);

    // 5. Std calls
    // CHECK: hipLaunchKernelGGL(my_kernel, dim3(1), dim3(std::max(config.x, config.y)), shared_mem, 0, std::min(config.x, config.y));
    my_kernel<<<1, std::max(config.x, config.y), shared_mem>>>(std::min(config.x, config.y));
}
