// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "vector_math.h"
// CHECK: #include "injection_helper.h"
#include <cuda_runtime.h>
#include "vector_math.h"
#include "injection_helper.h"

__global__ void testKernel(float3* data) {
    int idx = threadIdx.x;
    dummy_vector_op();
    add_vectors(data[idx], data[idx + 1]);
}

int main() { return 0; }
