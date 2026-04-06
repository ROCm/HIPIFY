// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef BLOCK_COMMENT_INCLUDE_H
#define BLOCK_COMMENT_INCLUDE_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline void block_comment_sync() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}

#endif
