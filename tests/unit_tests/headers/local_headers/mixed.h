// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#ifndef MIXED_H
#define MIXED_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline int mul(int a,int b){ return a*b; }
inline void s(){ 
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize(); 
}
#endif