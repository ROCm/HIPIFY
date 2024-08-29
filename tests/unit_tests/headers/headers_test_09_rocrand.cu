// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --roc %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <memory>

// CHECK-NOT: #include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

// CHECK: #include "hip/hip_runtime_api.h"
// CHECK: #include "hip/channel_descriptor.h"
// CHECK: #include "hip/device_functions.h"
// CHECK: #include "hip/driver_types.h"
// CHECK: #include "hip/hip_complex.h"
// CHECK: #include "hip/hip_texture_types.h"
// CHECK: #include "hip/hip_vector_types.h"

// CHECK: #include <iostream>

// CHECK: #include <stdio.h>

// CHECK: #include "rocrand/rocrand.h"
// CHECK: #include "rocrand/rocrand_kernel.h"

// CHECK: #include <algorithm>

// CHECK: #include "rocrand/rocrand_discrete.h"
// CHECK: #include "rocrand/rocrand_common.h"
// CHECK: #include "rocrand/rocrand_log_normal.h"
// CHECK: #include "rocrand/rocrand_mrg32k3a.h"
// CHECK: #include "rocrand/rocrand_mtgp32.h"
// CHECK: #include "rocrand/rocrand_mtgp32_11213.h"
// CHECK: #include "rocrand/rocrand_normal.h"
// CHECK: #include "rocrand/rocrand_philox4x32_10.h"
// CHECK: #include "rocrand/rocrand_poisson.h"
// CHECK: #include "rocrand/rocrand_xorwow_precomputed.h"
// CHECK: #include "rocrand/rocrand_uniform.h"

// CHECK-NOT: #include "rocrand/rocrand.h"
// CHECK-NOT: #include "rocrand/rocrand_kernel.h"
// CHECK-NOT: #include "rocrand/rocrand_discrete.h"
// CHECK-NOT: #include "rocrand/rocrand_mtgp32.h"
// CHECK-NOT: #include "rocrand/rocrand_normal.h"

// CHECK-NOT: #include "curand_discrete.h"
// CHECK-NOT: #include "curand_discrete2.h"
// CHECK-NOT: #include "curand_globals.h"
// CHECK-NOT: #include "curand_lognormal.h"
// CHECK-NOT: #include "curand_mrg32k3a.h"
// CHECK-NOT: #include "curand_mtgp32.h"
// CHECK-NOT: #include "curand_mtgp32_host.h"
// CHECK-NOT: #include "curand_mtgp32_kernel.h"
// CHECK-NOT: #include "curand_mtgp32dc_p_11213.h"
// CHECK-NOT: #include "curand_normal.h"
// CHECK-NOT: #include "curand_normal_static.h"
// CHECK-NOT: #include "curand_philox4x32_x.h"
// CHECK-NOT: #include "curand_poisson.h"
// CHECK-NOT: #include "curand_precalc.h"
// CHECK-NOT: #include "curand_uniform.h"

// CHECK: #include <string>

// CHECK: #include "hipfft/hipfft.h"
// CHECK: #include "rocsparse.h"

#include <cuda.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

#include <memory>

#include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

#include "cuda_runtime_api.h"
#include "channel_descriptor.h"
#include "device_functions.h"
#include "driver_types.h"
#include "cuComplex.h"
#include "cuda_texture_types.h"
#include "vector_types.h"

#include <iostream>

#include <stdio.h>

#include "curand.h"
#include "curand_kernel.h"

#include <algorithm>

#include "curand_discrete.h"
#include "curand_discrete2.h"
#include "curand_globals.h"
#include "curand_lognormal.h"
#include "curand_mrg32k3a.h"
#include "curand_mtgp32.h"
#include "curand_mtgp32_host.h"
#include "curand_mtgp32_kernel.h"
#include "curand_mtgp32dc_p_11213.h"
#include "curand_normal.h"
#include "curand_normal_static.h"
#include "curand_philox4x32_x.h"
#include "curand_poisson.h"
#include "curand_precalc.h"
#include "curand_uniform.h"

#include <string>

#include "cufft.h"

#include "cusparse.h"
