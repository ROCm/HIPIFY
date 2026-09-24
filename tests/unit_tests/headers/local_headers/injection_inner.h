// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef INJECTION_INNER_H
#define INJECTION_INNER_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline __device__ void inner_add(float3* data, int idx) {
    return;
}

#endif
