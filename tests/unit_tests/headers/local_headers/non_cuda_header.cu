// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include "non_cuda.h"
#include <cuda_runtime.h>
#include "non_cuda.h"