// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers-recursive
// %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "common.h"
#include <cuda_runtime.h>
#include "common.h"
int main(){}