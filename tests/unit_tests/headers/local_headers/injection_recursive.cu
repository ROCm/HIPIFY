// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "vector_math.h"
// CHECK: #include "injection_outer.h"
#include <cuda_runtime.h>
#include "vector_math.h"
#include "injection_outer.h"

__global__ void recursiveKernel(float3* data) {
    int idx = threadIdx.x;
    dummy_vector_op();
    outer_process(data, idx);
}

int main() { return 0; }
