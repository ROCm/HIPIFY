// RUN: %run_test hipify "%s" "%t" %hipify_args 2 --default-preprocessor --roc %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda.h>
// CHECK-NOT: #include <hip/hip_runtime.h>
#include <memory>

#include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

#if CUDA_VERSION >= 7050
// CHECK: #include "hip/hip_fp16.h"
#include "cuda_fp16.h"
#endif
