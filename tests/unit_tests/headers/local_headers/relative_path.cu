// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
#include "relative/./sub/../common1.h"
#include "relative/sub/common2.h"
int main(){return 0; }