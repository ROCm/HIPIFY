// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef NON_CUDA_H
#define NON_CUDA_H
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
inline int add(int a,int b){ return a + b; }
#endif