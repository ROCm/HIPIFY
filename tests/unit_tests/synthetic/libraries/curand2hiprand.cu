// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hiprand/hiprand.h"
// CHECK-NEXT: #include "hiprand/hiprand_kernel.h"
#include "curand.h"
#include "curand_kernel.h"
// CHECK-NOT: #include "hiprand/hiprand.h"
// CHECK-NOT: #include "hiprand/hiprand_kernel.h"

#if defined(_WIN32) && CUDA_VERSION < 9000
  typedef signed   __int64 int64_t;
  typedef unsigned __int64 uint64_t;
#endif

int main() {
  printf("21. cuRAND API to hipRAND API synthetic test\n");

  // CHECK: hiprandOrdering randOrdering;
  // CHECK-NEXT: hiprandOrdering_t randOrdering_t;
  // CHECK-NEXT: hiprandOrdering_t RAND_ORDERING_PSEUDO_BEST = HIPRAND_ORDERING_PSEUDO_BEST;
  // CHECK-NEXT: hiprandOrdering_t RAND_ORDERING_PSEUDO_DEFAULT = HIPRAND_ORDERING_PSEUDO_DEFAULT;
  // CHECK-NEXT: hiprandOrdering_t RAND_ORDERING_PSEUDO_SEEDED = HIPRAND_ORDERING_PSEUDO_SEEDED;
  // CHECK-NEXT: hiprandOrdering_t RAND_ORDERING_QUASI_DEFAULT = HIPRAND_ORDERING_QUASI_DEFAULT;
  curandOrdering randOrdering;
  curandOrdering_t randOrdering_t;
  curandOrdering_t RAND_ORDERING_PSEUDO_BEST = CURAND_ORDERING_PSEUDO_BEST;
  curandOrdering_t RAND_ORDERING_PSEUDO_DEFAULT = CURAND_ORDERING_PSEUDO_DEFAULT;
  curandOrdering_t RAND_ORDERING_PSEUDO_SEEDED = CURAND_ORDERING_PSEUDO_SEEDED;
  curandOrdering_t RAND_ORDERING_QUASI_DEFAULT = CURAND_ORDERING_QUASI_DEFAULT;

  // CHECK: hiprandStatus randStatus;
  // CHECK-NEXT: hiprandStatus_t status;
  curandStatus randStatus;
  curandStatus_t status;

  // CHECK: hiprandGenerator_st *randGenerator_st = nullptr;
  // CHECK-NEXT: hiprandGenerator_t randGenerator;
  curandGenerator_st *randGenerator_st = nullptr;
  curandGenerator_t randGenerator;

  // CHECK: hiprandStateSobol64 randStateSobol64;
  // CHECK-NEXT: hiprandStateSobol64_t randStateSobol64_t;
  curandStateSobol64 randStateSobol64;
  curandStateSobol64_t randStateSobol64_t;

  // CHECK: hiprandStateScrambledSobol64 randStateScrambledSobol64;
  // CHECK-NEXT: hiprandStateScrambledSobol64_t randStateScrambledSobol64_t;
  curandStateScrambledSobol64 randStateScrambledSobol64;
  curandStateScrambledSobol64_t randStateScrambledSobol64_t;

  // CHECK: hiprandStateScrambledSobol32 randStateScrambledSobol32;
  // CHECK-NEXT: hiprandStateScrambledSobol32_t randStateScrambledSobol32_t;
  curandStateScrambledSobol32 randStateScrambledSobol32;
  curandStateScrambledSobol32_t randStateScrambledSobol32_t;

  // CUDA: curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOrdering(hiprandGenerator_t generator, hiprandOrdering_t order);
  // CHECK: status = hiprandSetGeneratorOrdering(randGenerator, randOrdering_t);
  status = curandSetGeneratorOrdering(randGenerator, randOrdering_t);

#if CUDA_VERSION >= 11000 && CURAND_VERSION >= 10200
  // CHECK: hiprandOrdering_t RAND_ORDERING_PSEUDO_LEGACY = HIPRAND_ORDERING_PSEUDO_LEGACY;
  curandOrdering_t RAND_ORDERING_PSEUDO_LEGACY = CURAND_ORDERING_PSEUDO_LEGACY;
#endif

#if CUDA_VERSION >= 11050 && CURAND_VERSION >= 10207
  // CHECK: hiprandOrdering_t RAND_ORDERING_PSEUDO_DYNAMIC = HIPRAND_ORDERING_PSEUDO_DYNAMIC;
  curandOrdering_t RAND_ORDERING_PSEUDO_DYNAMIC = CURAND_ORDERING_PSEUDO_DYNAMIC;
#endif

  return 0;
}
