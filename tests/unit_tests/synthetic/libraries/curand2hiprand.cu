// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --default-preprocessor --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

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

  unsigned int *outputPtr = nullptr;
  size_t num = 0;

  // CHECK: hiprandStatus randStatus;
  // CHECK-NEXT: hiprandStatus_t status;
  // CHECK-NEXT: hiprandStatus_t STATUS_SUCCESS = HIPRAND_STATUS_SUCCESS;
  // CHECK-NEXT: hiprandStatus_t STATUS_VERSION_MISMATCH = HIPRAND_STATUS_VERSION_MISMATCH;
  // CHECK-NEXT: hiprandStatus_t STATUS_NOT_INITIALIZED = HIPRAND_STATUS_NOT_INITIALIZED;
  // CHECK-NEXT: hiprandStatus_t STATUS_ALLOCATION_FAILED = HIPRAND_STATUS_ALLOCATION_FAILED;
  // CHECK-NEXT: hiprandStatus_t STATUS_TYPE_ERROR = HIPRAND_STATUS_TYPE_ERROR;
  // CHECK-NEXT: hiprandStatus_t STATUS_OUT_OF_RANGE = HIPRAND_STATUS_OUT_OF_RANGE;
  // CHECK-NEXT: hiprandStatus_t STATUS_LENGTH_NOT_MULTIPLE = HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
  // CHECK-NEXT: hiprandStatus_t STATUS_DOUBLE_PRECISION_REQUIRED = HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
  // CHECK-NEXT: hiprandStatus_t STATUS_LAUNCH_FAILURE = HIPRAND_STATUS_LAUNCH_FAILURE;
  // CHECK-NEXT: hiprandStatus_t STATUS_PREEXISTING_FAILURE = HIPRAND_STATUS_PREEXISTING_FAILURE;
  // CHECK-NEXT: hiprandStatus_t STATUS_INITIALIZATION_FAILED = HIPRAND_STATUS_INITIALIZATION_FAILED;
  // CHECK-NEXT: hiprandStatus_t STATUS_ARCH_MISMATCH = HIPRAND_STATUS_ARCH_MISMATCH;
  // CHECK-NEXT: hiprandStatus_t STATUS_INTERNAL_ERROR = HIPRAND_STATUS_INTERNAL_ERROR;
  curandStatus randStatus;
  curandStatus_t status;
  curandStatus_t STATUS_SUCCESS = CURAND_STATUS_SUCCESS;
  curandStatus_t STATUS_VERSION_MISMATCH = CURAND_STATUS_VERSION_MISMATCH;
  curandStatus_t STATUS_NOT_INITIALIZED = CURAND_STATUS_NOT_INITIALIZED;
  curandStatus_t STATUS_ALLOCATION_FAILED = CURAND_STATUS_ALLOCATION_FAILED;
  curandStatus_t STATUS_TYPE_ERROR = CURAND_STATUS_TYPE_ERROR;
  curandStatus_t STATUS_OUT_OF_RANGE = CURAND_STATUS_OUT_OF_RANGE;
  curandStatus_t STATUS_LENGTH_NOT_MULTIPLE = CURAND_STATUS_LENGTH_NOT_MULTIPLE;
  curandStatus_t STATUS_DOUBLE_PRECISION_REQUIRED = CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
  curandStatus_t STATUS_LAUNCH_FAILURE = CURAND_STATUS_LAUNCH_FAILURE;
  curandStatus_t STATUS_PREEXISTING_FAILURE = CURAND_STATUS_PREEXISTING_FAILURE;
  curandStatus_t STATUS_INITIALIZATION_FAILED = CURAND_STATUS_INITIALIZATION_FAILED;
  curandStatus_t STATUS_ARCH_MISMATCH = CURAND_STATUS_ARCH_MISMATCH;
  curandStatus_t STATUS_INTERNAL_ERROR = CURAND_STATUS_INTERNAL_ERROR;

  // CHECK: hiprandRngType_t randRngType;
  // CHECK-NEXT: hiprandRngType_t randRngType_t;
  // CHECK-NEXT: hiprandRngType_t RNG_TEST = HIPRAND_RNG_TEST;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_DEFAULT = HIPRAND_RNG_PSEUDO_DEFAULT;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_XORWOW = HIPRAND_RNG_PSEUDO_XORWOW;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_MRG32K3A = HIPRAND_RNG_PSEUDO_MRG32K3A;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_MTGP32 = HIPRAND_RNG_PSEUDO_MTGP32;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_MT19937 = HIPRAND_RNG_PSEUDO_MT19937;
  // CHECK-NEXT: hiprandRngType_t RNG_PSEUDO_PHILOX4_32_10 = HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
  // CHECK-NEXT: hiprandRngType_t RNG_QUASI_DEFAULT = HIPRAND_RNG_QUASI_DEFAULT;
  // CHECK-NEXT: hiprandRngType_t RNG_QUASI_SOBOL32 = HIPRAND_RNG_QUASI_SOBOL32;
  // CHECK-NEXT: hiprandRngType_t RNG_QUASI_SCRAMBLED_SOBOL32 = HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
  // CHECK-NEXT: hiprandRngType_t RNG_QUASI_SOBOL64 = HIPRAND_RNG_QUASI_SOBOL64;
  // CHECK-NEXT: hiprandRngType_t RNG_QUASI_SCRAMBLED_SOBOL64 = HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  curandRngType randRngType;
  curandRngType_t randRngType_t;
  curandRngType_t RNG_TEST = CURAND_RNG_TEST;
  curandRngType_t RNG_PSEUDO_DEFAULT = CURAND_RNG_PSEUDO_DEFAULT;
  curandRngType_t RNG_PSEUDO_XORWOW = CURAND_RNG_PSEUDO_XORWOW;
  curandRngType_t RNG_PSEUDO_MRG32K3A = CURAND_RNG_PSEUDO_MRG32K3A;
  curandRngType_t RNG_PSEUDO_MTGP32 = CURAND_RNG_PSEUDO_MTGP32;
  curandRngType_t RNG_PSEUDO_MT19937 = CURAND_RNG_PSEUDO_MT19937;
  curandRngType_t RNG_PSEUDO_PHILOX4_32_10 = CURAND_RNG_PSEUDO_PHILOX4_32_10;
  curandRngType_t RNG_QUASI_DEFAULT = CURAND_RNG_QUASI_DEFAULT;
  curandRngType_t RNG_QUASI_SOBOL32 = CURAND_RNG_QUASI_SOBOL32;
  curandRngType_t RNG_QUASI_SCRAMBLED_SOBOL32 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
  curandRngType_t RNG_QUASI_SOBOL64 = CURAND_RNG_QUASI_SOBOL64;
  curandRngType_t RNG_QUASI_SCRAMBLED_SOBOL64 = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;

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

  // CHECK: hiprandDirectionVectors32_t directions32;
  // CHECK-NEXT: hiprandDirectionVectors64_t directions64;
  // CHECK-NEXT: hiprandDirectionVectors64_t *pDirections64 = nullptr;
  curandDirectionVectors32_t directions32;
  curandDirectionVectors64_t directions64;
  curandDirectionVectors64_t *pDirections64 = nullptr;

  // CHECK: hiprandDirectionVectorSet_t directionVectorSet;
  // CHECK-NEXT: hiprandDirectionVectorSet_t directionVectorSet_t;
  // CHECK-NEXT: hiprandDirectionVectorSet_t DIRECTION_VECTORS_32_JOEKUO6 = HIPRAND_DIRECTION_VECTORS_32_JOEKUO6;
  // CHECK-NEXT: hiprandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
  // CHECK-NEXT: hiprandDirectionVectorSet_t DIRECTION_VECTORS_64_JOEKUO6 = HIPRAND_DIRECTION_VECTORS_64_JOEKUO6;
  // CHECK-NEXT: hiprandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
  curandDirectionVectorSet directionVectorSet;
  curandDirectionVectorSet_t directionVectorSet_t;
  curandDirectionVectorSet_t DIRECTION_VECTORS_32_JOEKUO6 = CURAND_DIRECTION_VECTORS_32_JOEKUO6;
  curandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
  curandDirectionVectorSet_t DIRECTION_VECTORS_64_JOEKUO6 = CURAND_DIRECTION_VECTORS_64_JOEKUO6;
  curandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;

  // CUDA: curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandCreateGenerator(hiprandGenerator_t* generator, hiprandRngType_t rng_type)
  // CHECK: status = hiprandCreateGenerator(&randGenerator, randRngType_t);
  status = curandCreateGenerator(&randGenerator, randRngType_t);

  // CUDA: curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandDestroyGenerator(hiprandGenerator_t generator);
  // CHECK: status = hiprandDestroyGenerator(randGenerator);
  status = curandDestroyGenerator(randGenerator);

  // CUDA: curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandCreateGeneratorHost(hiprandGenerator_t * generator, hiprandRngType_t rng_type);
  // CHECK: status = hiprandCreateGeneratorHost(&randGenerator, randRngType_t);
  status = curandCreateGeneratorHost(&randGenerator, randRngType_t);

  // CUDA: curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOrdering(hiprandGenerator_t generator, hiprandOrdering_t order);
  // CHECK: status = hiprandSetGeneratorOrdering(randGenerator, randOrdering_t);
  status = curandSetGeneratorOrdering(randGenerator, randOrdering_t);

  // CUDA: curandStatus_t CURANDAPI curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[], curandDirectionVectorSet_t set);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors64(hiprandDirectionVectors64_t** vectors, hiprandDirectionVectorSet_t set);
  // CHECK: status = hiprandGetDirectionVectors64(&pDirections64, directionVectorSet_t);
  status = curandGetDirectionVectors64(&pDirections64, directionVectorSet_t);

  // CUDA: curandStatus_t CURANDAPI curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num);
  // HIP: hiprandStatus_t HIPRANDAPI hiprandGenerate(hiprandGenerator_t generator, unsigned int * output_data, size_t n);
  // CHECK: status = hiprandGenerate(randGenerator, outputPtr, num);
  status = curandGenerate(randGenerator, outputPtr, num);

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
