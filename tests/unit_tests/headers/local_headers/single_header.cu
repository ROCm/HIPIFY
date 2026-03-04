// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "single_header.h"
#include <cuda_runtime.h>
#include "single_header.h"
int main(){return 0; }