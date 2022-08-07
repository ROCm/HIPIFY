// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas.h"
// CHECK-NOT: #include "cublas_v2.h"
// CHECK: #include <stdio.h>
#include "cublas.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "hipblas.h"
#include <stdio.h>
