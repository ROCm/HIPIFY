// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Checks that HIP header file is included after include guard controlling macro,
// which goes before #pragma once.
// CHECK: #ifndef HEADERS_TEST_12_SOLVER_H
// CHECK-NEXT: #include <hip/hip_runtime.h>
#ifndef HEADERS_TEST_12_SOLVER_H
// CHECK: #pragma once
#pragma once
// CHECK-NOT: #include <hip/hip_runtime.h>
#define HEADERS_TEST_12_SOLVER_H
#include <stdio.h>
static int counter = 0;

// CHECK: #include "hipsolver.h"
// CHECK-NOT: #include "hipsolver.h"
// CHECK-NOT: #include "cusolver_common.h"
// CHECK-NOT: #include "cusolverDn.h"
// CHECK-NOT: #include "cusolverRf.h"
// CHECK-NOT: #include "cusolverMg.h"
// CHECK-NOT: #include "cusolverSp.h"
// CHECK-NOT: #include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include "cusolver_common.h"
#include "cusolverDn.h"
#include "cusolverRf.h"
#include "cusolverMg.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"

#endif // HEADERS_TEST_12_SOLVER_H