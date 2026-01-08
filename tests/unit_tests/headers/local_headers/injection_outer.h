// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers-recursive %clang_args

#ifndef INJECTION_OUTER_H
#define INJECTION_OUTER_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "injection_inner.h"
#include <cuda_runtime.h>
#include "injection_inner.h"

inline __device__ void outer_process(float3* data, int idx) {
    inner_add(data, idx);
}

#endif
