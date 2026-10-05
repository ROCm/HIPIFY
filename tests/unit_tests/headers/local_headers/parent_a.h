// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef PARENT_A_H
#define PARENT_A_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "shared_dep.h"
#include <cuda_runtime.h>
#include "shared_dep.h"

inline void parent_a_malloc(void **p) {
    // CHECK: hipMalloc(p, 32);
    cudaMalloc(p, 32);
}

#endif
