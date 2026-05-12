// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "transitive_parent.h"
#include <cuda_runtime.h>
#include <algorithm>
#include "transitive_parent.h"

int main() { return 0; }
