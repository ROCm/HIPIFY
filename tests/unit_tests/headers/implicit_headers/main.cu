// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "common.h"
// CHECK: #include "common_1.h"
// CHECK: #include "common_2.h"
#include <cuda_runtime.h>

#include "common.h"
#include "common_1.h"
#include "common_2.h"