// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef COMMON2_H
#define COMMON2_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline void w() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}
#endif