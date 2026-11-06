// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef INJECTION_HELPER_H
#define INJECTION_HELPER_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline __device__ float3 add_vectors(float3 a, float3 b) {
    return make_float3(0.0f, 0.0f, 0.0f);
}

inline __device__ void accumulate(float3* sum, float3 val) {
    return;
}

inline __device__ float3 scale_and_diff(float3 a, float3 b, float s) {
    return make_float3(0.0f, 0.0f, 0.0f);
}

#endif
