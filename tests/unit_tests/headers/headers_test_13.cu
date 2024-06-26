// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_math_constants.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include <stdio.h>
#include <cuda.h>
#include <math_constants.h>
#include <stdio.h>
