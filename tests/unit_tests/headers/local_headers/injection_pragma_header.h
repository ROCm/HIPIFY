// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

#pragma once

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline __device__ void pragma_add(float3* data, int idx) {
    return;
}
