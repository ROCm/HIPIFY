// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "vector_math.h"
// CHECK: #include "injection_pragma_header.h"
#include <cuda_runtime.h>
#include "vector_math.h"
#include "injection_pragma_header.h"

__global__ void pragmaKernel(float3* data) {
    int idx = threadIdx.x;
    dummy_vector_op();
    pragma_add(data, idx);
}

int main() { return 0; }
