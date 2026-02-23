// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>

inline __host__ __device__ void dummy_vector_op() { return; }

#endif
