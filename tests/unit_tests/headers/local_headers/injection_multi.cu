// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include <cmath>
// CHECK: #include "vector_math.h"
// CHECK: #include "injection_helper.h"
// CHECK: #include "injection_uses_cmath.h"
#include <cuda_runtime.h>
#include <cmath>
#include "vector_math.h"
#include "injection_helper.h"
#include "injection_uses_cmath.h"

__global__ void multiKernel(float3* data, float* vals) {
    int idx = threadIdx.x;
    dummy_vector_op();
    add_vectors(data[idx], data[idx + 1]);
    vals[idx] = compute_sqrt(vals[idx]);
}

int main() { return 0; }
