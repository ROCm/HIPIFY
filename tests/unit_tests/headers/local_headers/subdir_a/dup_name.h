// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef SUBDIR_A_DUP_NAME_H
#define SUBDIR_A_DUP_NAME_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline void subdir_a_sync() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}

#endif
