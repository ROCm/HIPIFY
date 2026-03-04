// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include <cmath>
// CHECK: #include "injection_has_cmath.h"
#include <cuda_runtime.h>
#include <cmath>
#include "injection_has_cmath.h"

int main() {
    float x = compute_value(4.0f);
    return (int)x;
}
