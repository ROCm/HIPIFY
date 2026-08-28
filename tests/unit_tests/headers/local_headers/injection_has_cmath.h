// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef INJECTION_HAS_CMATH_H
#define INJECTION_HAS_CMATH_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include <cmath>
#include <cuda_runtime.h>
#include <cmath>

inline float compute_value(float x) {
    return sqrtf(x) + 1.0f;
}

#endif
