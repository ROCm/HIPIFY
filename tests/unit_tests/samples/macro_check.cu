// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --hip-kernel-execution-syntax %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>

#define a0     -3.0124472f
#define a1      1.7383092f
#define a2     -0.2796695f
#define a3      0.0547837f
#define a4     -0.0073118f

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

__device__ __constant__ float coef[5];

int main() {
    const float h_coef[] = {a0, a1, a2, a3, a4};
    // CHECK: CHECK( hipMemcpyToSymbol( HIP_SYMBOL(coef), h_coef, 5 * sizeof(float) ));
    CHECK( cudaMemcpyToSymbol( coef, h_coef, 5 * sizeof(float) ));
}