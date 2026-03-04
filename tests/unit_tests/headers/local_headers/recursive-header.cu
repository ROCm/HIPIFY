// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "rec_1.h"
#include <cuda_runtime.h>
#include "rec_1.h"
int main(){}