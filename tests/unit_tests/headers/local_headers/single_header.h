// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
#ifndef SINGLE_HEADER_H
#define SINGLE_HEADER_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
inline void alloc(void **p){
    // CHECK: hipMalloc(p, 16);
    cudaMalloc(p, 16); }
#endif