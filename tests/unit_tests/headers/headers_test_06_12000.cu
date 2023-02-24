// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --skip-excluded-preprocessor-conditional-blocks %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <hipblas.h>
// CHECK: #include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <stdio.h>
