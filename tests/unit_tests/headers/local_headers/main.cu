// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "common.h"
// CHECK: #include "single_header.h"
// CHECK: #include "shared.h"
// CHECK: #include "block_comment_include.h"
// CHECK: #include "parent_a.h"
// CHECK: #include "parent_b.h"
// CHECK: #include "subdir_a/dup_name.h"
// CHECK: #include "subdir_b/dup_name.h"
#include <cuda_runtime.h>

#include "common.h"
#include "single_header.h"
#include "shared.h"

#include "block_comment_include.h" /* block comment after include */

#include "parent_a.h"
#include "parent_b.h"

#include "subdir_a/dup_name.h"
#include "subdir_b/dup_name.h"
