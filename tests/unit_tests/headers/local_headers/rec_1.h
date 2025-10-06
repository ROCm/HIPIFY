// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers-recursive %clang_args

#ifndef REC_H
#define REC_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "relative/common1.h"
#include <cuda_runtime.h>
#include "relative/common1.h"
#endif