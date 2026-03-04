// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "common.h"
// CHECK: #include "single_header.h"
// CHECK: #include "shared.h"
#include <cuda_runtime.h>

#include "common.h"
#include "single_header.h"
#include "shared.h"