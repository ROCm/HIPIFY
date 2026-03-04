// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#ifndef COMMON_H
#define COMMON_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: cuda_runtime.h
// CHECK: #include <math.h>
// CHECK: #include <memory.h>
// CHECK: #include <stdio.h>
// CHECK: #include <stdlib.h>
// CHECK: #include <time.h>
// CHECK: #include "mixed.h"
#include <cuda_runtime.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mixed.h"

#endif