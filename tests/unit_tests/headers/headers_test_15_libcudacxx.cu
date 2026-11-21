// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --skip-excluded-preprocessor-conditional-blocks %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>

// libcu++ headers that libhipcxx provides under the same names are left unchanged.
// libcu++ headers that libhipcxx lists as unsupported are left unchanged too, and hipify-clang warns about them.
// Each header is guarded by the CUDA version that first ships it.

#if CUDA_VERSION >= 11000
// CHECK: #include <cuda/std/atomic>
#include <cuda/std/atomic>
// CHECK: #include <cuda/barrier>
#include <cuda/barrier>
// CHECK: #include <cuda/std/barrier>
#include <cuda/std/barrier>
// CHECK: #include <cuda/std/latch>
#include <cuda/std/latch>
// CHECK: #include <cuda/std/semaphore>
#include <cuda/std/semaphore>
#endif

#if CUDA_VERSION >= 11010
// CHECK: #include <cuda/latch>
#include <cuda/latch>
// CHECK: #include <cuda/pipeline>
#include <cuda/pipeline>
// CHECK: #include <cuda/semaphore>
#include <cuda/semaphore>
#endif

#if CUDA_VERSION >= 11050
// CHECK: #include <cuda/annotated_ptr>
#include <cuda/annotated_ptr>
#endif

#if CUDA_VERSION >= 11060
// CHECK: #include <cuda/std/cmath>
#include <cuda/std/cmath>
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
// CHECK: #include <cuda/buffer>
#include <cuda/buffer>
// CHECK: #include <cuda/tma>
#include <cuda/tma>
#endif
