// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// RUN: hipify "%s" -o="%t.warn" %cuda_path_arg %clang_res_arg -- %clang_args 2>&1 | FileCheck -check-prefix=WARN "%s"

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 11000
// libcu++ headers shipped by libhipcxx under the same names are left unchanged and not reported.
// CHECK: #include <cuda/std/atomic>
// WARN-NOT: 'cuda/std/atomic' is unsupported header
#include <cuda/std/atomic>
// CHECK: #include <cuda/std/cmath>
// WARN-NOT: 'cuda/std/cmath' is unsupported header
#include <cuda/std/cmath>

// libcu++ headers without a libhipcxx equivalent are left unchanged and reported.
// CHECK: #include <cuda/barrier>
// WARN: warning: 'cuda/barrier' is unsupported header in 'libhipcxx'.
#include <cuda/barrier>
// CHECK: #include <cuda/pipeline>
// WARN: warning: 'cuda/pipeline' is unsupported header in 'libhipcxx'.
#include <cuda/pipeline>
// CHECK: #include <cuda/std/semaphore>
// WARN: warning: 'cuda/std/semaphore' is unsupported header in 'libhipcxx'.
#include <cuda/std/semaphore>
#endif
