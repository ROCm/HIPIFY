// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include "duplicate.h"
#include "duplicate.h"
#include "duplicate.h" // duplicate
#include "duplicate.h" // duplicate

int main(){}