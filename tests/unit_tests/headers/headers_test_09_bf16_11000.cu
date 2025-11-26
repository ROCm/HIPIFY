// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK-NOT: #include <hip/hip_runtime.h>
#include <memory>

#include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

#if CUDA_VERSION >= 11000
// CHECK: #include "hip/hip_bf16.h"
#include "cuda_bf16.h"
#endif
