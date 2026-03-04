// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "duplicate.h"
// CHECK: #include "mixed.h"
// CHECK: #include "non_cuda.h"
#include <cuda_runtime.h>
#include "duplicate.h"
#include "mixed.h"
#include "non_cuda.h"
int main(){return 0; }

