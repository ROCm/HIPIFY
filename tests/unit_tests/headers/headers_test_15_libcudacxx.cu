// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --skip-excluded-preprocessor-conditional-blocks %clang_args 2>&1 | FileCheck "%s" --check-prefix=WARN

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

// libcu++ headers that libhipcxx provides under the same names are left unchanged and not reported.
// libcu++ headers that libhipcxx lists as unsupported are left unchanged too, and hipify-clang warns about them.
// Each header is guarded by a CUDA version that ships it. The WARN lines cover the headers that every CUDA version the test runs on ships.
// <cuda/buffer> from CUDA 13.2 doesn't compile with clang 23, so it isn't included here.

#if CUDA_VERSION >= 11000
// CHECK: #include <cuda/std/atomic>
// WARN-NOT: 'cuda/std/atomic' is unsupported header
#include <cuda/std/atomic>
// CHECK: #include <cuda/barrier>
// WARN: warning: 'cuda/barrier' is unsupported header in 'HIP'.
#include <cuda/barrier>
// CHECK: #include <cuda/std/barrier>
// WARN: warning: 'cuda/std/barrier' is unsupported header in 'HIP'.
#include <cuda/std/barrier>
// CHECK: #include <cuda/std/latch>
// WARN: warning: 'cuda/std/latch' is unsupported header in 'HIP'.
#include <cuda/std/latch>
// CHECK: #include <cuda/std/semaphore>
// WARN: warning: 'cuda/std/semaphore' is unsupported header in 'HIP'.
#include <cuda/std/semaphore>
#endif

#if CUDA_VERSION >= 11010
// CHECK: #include <cuda/latch>
// WARN: warning: 'cuda/latch' is unsupported header in 'HIP'.
#include <cuda/latch>
// CHECK: #include <cuda/pipeline>
// WARN: warning: 'cuda/pipeline' is unsupported header in 'HIP'.
#include <cuda/pipeline>
// CHECK: #include <cuda/semaphore>
// WARN: warning: 'cuda/semaphore' is unsupported header in 'HIP'.
#include <cuda/semaphore>
#endif

#if CUDA_VERSION >= 11030
// CHECK: #include <cuda/std/cmath>
// WARN-NOT: 'cuda/std/cmath' is unsupported header
#include <cuda/std/cmath>
#endif

#if CUDA_VERSION >= 11050
// CHECK: #include <cuda/annotated_ptr>
#include <cuda/annotated_ptr>
#endif

#if CUDA_VERSION >= 12040
// CHECK: #include <cuda/ptx>
#include <cuda/ptx>
#endif

#if CUDA_VERSION >= 13010
// CHECK: #include <cuda/std/execution>
#include <cuda/std/execution>
#endif

#if CUDA_VERSION >= 13020
// CHECK: #include <cuda/tma>
#include <cuda/tma>
#endif
