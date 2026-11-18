// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef COMMON1_H
#define COMMON1_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline void g(){
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}
#endif