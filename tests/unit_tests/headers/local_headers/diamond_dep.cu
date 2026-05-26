// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --local-headers-recursive %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "diamond_left.h"
// CHECK: #include "diamond_right.h"
#include <cuda_runtime.h>
#include "diamond_left.h"
#include "diamond_right.h"

int main() {
    return 0;
}
