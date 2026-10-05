// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef DUP_H
#define DUP_H
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
inline void sync(){ cudaDeviceSynchronize(); }
#endif