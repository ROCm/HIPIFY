// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_cooperative_groups.h>
// CHECK-NEXT: #include <hip/cooperative_groups/hip_reduce.h>
// CHECK-NEXT: #include <hip/cooperative_groups/hip_scan.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
