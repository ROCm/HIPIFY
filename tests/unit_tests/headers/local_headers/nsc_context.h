// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef NSC_CONTEXT_H
#define NSC_CONTEXT_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

struct NscVec {
    float3 data;
    cudaError_t status;
};

#endif
