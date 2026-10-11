// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#ifndef SHARED_H
#define SHARED_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline void sync(){
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}
#endif