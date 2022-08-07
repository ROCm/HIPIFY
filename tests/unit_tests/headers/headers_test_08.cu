// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <iostream>
// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas.h"
// CHECK-NOT: #include "cublas_v2.h"
// CHECK: #include <stdio.h>
// CHECK-NOT: #include <cuda.h>
#include <cuda.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>
#include <iostream>
#include "cublas.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "hipblas.h"
#include <stdio.h>
