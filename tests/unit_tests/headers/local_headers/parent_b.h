// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef PARENT_B_H
#define PARENT_B_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "shared_dep.h"
#include <cuda_runtime.h>
#include "shared_dep.h"

inline void parent_b_free(void *p) {
    // CHECK: hipFree(p);
    cudaFree(p);
}

#endif
