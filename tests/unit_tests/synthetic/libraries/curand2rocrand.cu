// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --default-preprocessor --experimental --roc %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocrand/rocrand.h"
#include "curand.h"
// CHECK-NOT: #include "rocrand/rocrand.h"

int main() {
  printf("21.1. cuRAND API to rocRAND API synthetic test\n");

  unsigned int *outputPtr = nullptr;
  size_t num = 0;

  // CHECK: rocrand_status randStatus;
  // CHECK-NEXT: rocrand_status status;
  // CHECK-NEXT: rocrand_status STATUS_SUCCESS = ROCRAND_STATUS_SUCCESS;
  // CHECK-NEXT: rocrand_status STATUS_VERSION_MISMATCH = ROCRAND_STATUS_VERSION_MISMATCH;
  // CHECK-NEXT: rocrand_status STATUS_NOT_INITIALIZED = ROCRAND_STATUS_NOT_CREATED;
  // CHECK-NEXT: rocrand_status STATUS_ALLOCATION_FAILED = ROCRAND_STATUS_ALLOCATION_FAILED;
  // CHECK-NEXT: rocrand_status STATUS_TYPE_ERROR = ROCRAND_STATUS_TYPE_ERROR;
  // CHECK-NEXT: rocrand_status STATUS_OUT_OF_RANGE = ROCRAND_STATUS_OUT_OF_RANGE;
  // CHECK-NEXT: rocrand_status STATUS_LENGTH_NOT_MULTIPLE = ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
  // CHECK-NEXT: rocrand_status STATUS_DOUBLE_PRECISION_REQUIRED = ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
  // CHECK-NEXT: rocrand_status STATUS_LAUNCH_FAILURE = ROCRAND_STATUS_LAUNCH_FAILURE;
  // CHECK-NEXT: rocrand_status STATUS_INTERNAL_ERROR = ROCRAND_STATUS_INTERNAL_ERROR;
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
  curandStatus_t STATUS_INTERNAL_ERROR = CURAND_STATUS_INTERNAL_ERROR;


  // CHECK: rocrand_rng_type randRngType;
  // CHECK-NEXT: rocrand_rng_type randRngType_t;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_DEFAULT = ROCRAND_RNG_PSEUDO_DEFAULT;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_XORWOW = ROCRAND_RNG_PSEUDO_XORWOW;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_MRG32K3A = ROCRAND_RNG_PSEUDO_MRG32K3A;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_MTGP32 = ROCRAND_RNG_PSEUDO_MTGP32;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_MT19937 = ROCRAND_RNG_PSEUDO_MT19937;
  // CHECK-NEXT: rocrand_rng_type RNG_PSEUDO_PHILOX4_32_10 = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
  // CHECK-NEXT: rocrand_rng_type RNG_QUASI_DEFAULT = ROCRAND_RNG_QUASI_DEFAULT;
  // CHECK-NEXT: rocrand_rng_type RNG_QUASI_SOBOL32 = ROCRAND_RNG_QUASI_SOBOL32;
  // CHECK-NEXT: rocrand_rng_type RNG_QUASI_SCRAMBLED_SOBOL32 = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
  // CHECK-NEXT: rocrand_rng_type RNG_QUASI_SOBOL64 = ROCRAND_RNG_QUASI_SOBOL64;
  // CHECK-NEXT: rocrand_rng_type RNG_QUASI_SCRAMBLED_SOBOL64 = ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
  curandRngType randRngType;
  curandRngType_t randRngType_t;
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

  // CHECK: rocrand_ordering randOrdering;
  // CHECK-NEXT: rocrand_ordering RAND_ORDERING_PSEUDO_BEST = ROCRAND_ORDERING_PSEUDO_BEST;
  // CHECK-NEXT: rocrand_ordering RAND_ORDERING_PSEUDO_DEFAULT = ROCRAND_ORDERING_PSEUDO_DEFAULT;
  // CHECK-NEXT: rocrand_ordering RAND_ORDERING_PSEUDO_SEEDED = ROCRAND_ORDERING_PSEUDO_SEEDED;
  // CHECK-NEXT: rocrand_ordering RAND_ORDERING_QUASI_DEFAULT = ROCRAND_ORDERING_QUASI_DEFAULT;
  curandOrdering randOrdering;
  curandOrdering_t RAND_ORDERING_PSEUDO_BEST = CURAND_ORDERING_PSEUDO_BEST;
  curandOrdering_t RAND_ORDERING_PSEUDO_DEFAULT = CURAND_ORDERING_PSEUDO_DEFAULT;
  curandOrdering_t RAND_ORDERING_PSEUDO_SEEDED = CURAND_ORDERING_PSEUDO_SEEDED;
  curandOrdering_t RAND_ORDERING_QUASI_DEFAULT = CURAND_ORDERING_QUASI_DEFAULT;

  // CHECK: rocrand_direction_vector_set directionVectorSet;
  // CHECK-NEXT: rocrand_direction_vector_set directionVectorSet_t;
  // CHECK-NEXT: rocrand_direction_vector_set DIRECTION_VECTORS_32_JOEKUO6 = ROCRAND_DIRECTION_VECTORS_32_JOEKUO6;
  // CHECK-NEXT: rocrand_direction_vector_set SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
  // CHECK-NEXT: rocrand_direction_vector_set DIRECTION_VECTORS_64_JOEKUO6 = ROCRAND_DIRECTION_VECTORS_64_JOEKUO6;
  // CHECK-NEXT: rocrand_direction_vector_set SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
  curandDirectionVectorSet directionVectorSet;
  curandDirectionVectorSet_t directionVectorSet_t;
  curandDirectionVectorSet_t DIRECTION_VECTORS_32_JOEKUO6 = CURAND_DIRECTION_VECTORS_32_JOEKUO6;
  curandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
  curandDirectionVectorSet_t DIRECTION_VECTORS_64_JOEKUO6 = CURAND_DIRECTION_VECTORS_64_JOEKUO6;
  curandDirectionVectorSet_t SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;

  // CHECK: rocrand_generator_base_type *randGenerator_st = nullptr;
  // CHECK-NEXT: rocrand_generator randGenerator;
  curandGenerator_st *randGenerator_st = nullptr;
  curandGenerator_t randGenerator;

  // CUDA: curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);
  // ROC: rocrand_status ROCRANDAPI rocrand_create_generator(rocrand_generator * generator, rocrand_rng_type rng_type);
  // CHECK: status = rocrand_create_generator(&randGenerator, randRngType_t);
  status = curandCreateGenerator(&randGenerator, randRngType_t);

  // CUDA: curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t generator);
  // ROC: rocrand_status ROCRANDAPI rocrand_destroy_generator(rocrand_generator generator);
  // CHECK: status = rocrand_destroy_generator(randGenerator);
  status = curandDestroyGenerator(randGenerator);

  // CUDA: curandStatus_t CURANDAPI curandCreateGeneratorHost(curandGenerator_t *generator, curandRngType_t rng_type);
  // ROC: rocrand_status ROCRANDAPI rocrand_create_generator_host_blocking(rocrand_generator* generator, rocrand_rng_type rng_type);
  // CHECK: status = rocrand_create_generator_host_blocking(&randGenerator, randRngType_t);
  status = curandCreateGeneratorHost(&randGenerator, randRngType_t);

  // CUDA: curandStatus_t CURANDAPI curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate(rocrand_generator generator, unsigned int * output_data, size_t n);
  // CHECK: status = rocrand_generate(randGenerator, outputPtr, num);
  status = curandGenerate(randGenerator, outputPtr, num);

#if CUDA_VERSION >= 11000 && CURAND_VERSION >= 10200
  // CHECK: rocrand_ordering RAND_ORDERING_PSEUDO_LEGACY = ROCRAND_ORDERING_PSEUDO_LEGACY;
  curandOrdering_t RAND_ORDERING_PSEUDO_LEGACY = CURAND_ORDERING_PSEUDO_LEGACY;
#endif

#if CUDA_VERSION >= 11050 && CURAND_VERSION >= 10207
  // CHECK: rocrand_ordering RAND_ORDERING_PSEUDO_DYNAMIC = ROCRAND_ORDERING_PSEUDO_DYNAMIC;
  curandOrdering_t RAND_ORDERING_PSEUDO_DYNAMIC = CURAND_ORDERING_PSEUDO_DYNAMIC;
#endif

  return 0;
}
