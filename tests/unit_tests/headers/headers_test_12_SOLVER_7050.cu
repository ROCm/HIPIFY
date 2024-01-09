// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --skip-excluded-preprocessor-conditional-blocks %clang_args

#ifndef HEADERS_TEST_12_SOLVER_H
// CHECK: #pragma once
#pragma once
#define HEADERS_TEST_12_SOLVER_H
// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
#include <cuda_runtime.h>
static int counter = 0;

// CHECK: #include "hipsolver.h"
// CHECK-NOT: #include "hipsolver.h"
// CHECK-NOT: #include "cusolver_common.h"
// CHECK-NOT: #include "cusolverDn.h"
// CHECK-NOT: #include "cusolverRf.h"
// CHECK-NOT: #include "cusolverSp.h"
// CHECK-NOT: #include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "cusolver_common.h"
#include "cusolverDn.h"
#include "cusolverRf.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#endif // HEADERS_TEST_12_SOLVER_H
