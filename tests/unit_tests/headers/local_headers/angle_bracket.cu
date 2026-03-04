// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK-NOT: local_angle.h.hip
#include <cuda_runtime.h>
#include <local_angle.h>  // treated as system-style; not hipified
int main(){}