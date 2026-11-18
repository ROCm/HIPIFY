// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef INJECTION_USES_CMATH_H
#define INJECTION_USES_CMATH_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline __device__ float compute_sqrt(float x) {
    return sqrtf(x);
}

inline __device__ float compute_magnitude(float x, float y) {
    return sqrtf(x * x + y * y);
}

#endif
