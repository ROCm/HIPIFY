// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef SHARED_DEP_H
#define SHARED_DEP_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline void shared_dep_sync() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}

#endif
