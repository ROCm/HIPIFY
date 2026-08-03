// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --local-headers %clang_args

// nsc_dependent.h uses a type from nsc_context.h without including it, so it
// hipifies only if the includes preceding it here are replayed in front of it.

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "nsc_context.h"
// CHECK: #include "nsc_dependent.h"
#include <cuda_runtime.h>
#include "nsc_context.h"
#include "nsc_dependent.h"

__global__ void nscKernel(NscVec* vectors) {
    int idx = threadIdx.x;
    vectors[idx].data = make_float3(0.0f, 0.0f, 0.0f);
}

int main() {
    NscVec vec;
    nsc_fill(&vec, 1);
    return 0;
}
