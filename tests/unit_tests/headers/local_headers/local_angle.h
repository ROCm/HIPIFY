// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#ifndef LOCAL_ANGLE_H
#define LOCAL_ANGLE_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline void z(){
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}
#endif