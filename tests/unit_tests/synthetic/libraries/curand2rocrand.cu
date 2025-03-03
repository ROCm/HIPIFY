// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --amap --default-preprocessor --experimental --roc %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocrand/rocrand.h"
// CHECK-NEXT: #include "rocrand/rocrand_kernel.h"
#include "curand.h"
#include "curand_kernel.h"
// CHECK-NOT: #include "rocrand/rocrand.h"
// CHECK-NOT: #include "rocrand/rocrand_kernel.h"

int main() {
  printf("23. cuRAND API to rocRAND API synthetic test\n");

  unsigned int *outputPtr = nullptr;
  unsigned int *constants = nullptr;
  unsigned long long *constantsLL = nullptr;
  float *outputPtrFloat = nullptr;
  double *outputPtrDouble = nullptr;
  unsigned int num_dimensions = 0;
  unsigned long long *outputPtrUll = nullptr;
  unsigned long long offset = 0;
  int version = 0;
  size_t num = 0;
  float mean = 0.f;
  double dmean = 0.f;
  float stddev = 0.f;
  double dstddev = 0.f;
  double dlambda = 0.f;

  // CHECK: hipStream_t stream;
  cudaStream_t stream;

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

  // CHECK: rocrand_device::sobol64_engine<false> randStateSobol64;
  // CHECK-NEXT: rocrand_state_sobol64 randStateSobol64_t;
  curandStateSobol64 randStateSobol64;
  curandStateSobol64_t randStateSobol64_t;

  // CHECK: rocrand_device::scrambled_sobol64_engine<false> randStateScrambledSobol64;
  // CHECK-NEXT: rocrand_state_scrambled_sobol64 randStateScrambledSobol64_t;
  curandStateScrambledSobol64 randStateScrambledSobol64;
  curandStateScrambledSobol64_t randStateScrambledSobol64_t;

  // CHECK: rocrand_device::sobol32_engine<false> randStateSobol32;
  // CHECK-NEXT: rocrand_state_sobol32 randStateSobol32_t;
  curandStateSobol32 randStateSobol32;
  curandStateSobol32_t randStateSobol32_t;

  // CHECK: rocrand_device::scrambled_sobol32_engine<false> randStateScrambledSobol32;
  // CHECK-NEXT: rocrand_state_scrambled_sobol32 randStateScrambledSobol32_t;
  curandStateScrambledSobol32 randStateScrambledSobol32;
  curandStateScrambledSobol32_t randStateScrambledSobol32_t;

  // CHECK: rocrand_discrete_distribution_st *discreteDistribution_st = nullptr;
  // CHECK-NEXT: rocrand_discrete_distribution discreteDistribution_t = nullptr;
  curandDiscreteDistribution_st *discreteDistribution_st = nullptr;
  curandDiscreteDistribution_t discreteDistribution_t = nullptr;

  // CHECK: rocrand_device::mtgp32_engine stateMtgp32;
  // CHECK-NEXT: rocrand_state_mtgp32 stateMtgp32_t;
  curandStateMtgp32 stateMtgp32;
  curandStateMtgp32_t stateMtgp32_t;

  // CHECK: rocrand_device::mrg32k3a_engine stateMRG32k3a;
  // CHECK-NEXT: rocrand_state_mrg32k3a stateMRG32k3a_t;
  curandStateMRG32k3a stateMRG32k3a;
  curandStateMRG32k3a_t stateMRG32k3a_t;

  // CHECK: rocrand_device::philox4x32_10_engine statePhilox4_32_10;
  // CHECK-NEXT: rocrand_state_philox4x32_10 statePhilox4_32_10_t;
  curandStatePhilox4_32_10 statePhilox4_32_10;
  curandStatePhilox4_32_10_t statePhilox4_32_10_t;

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

  // CUDA: curandStatus_t CURANDAPI curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_log_normal(rocrand_generator generator, float * output_data, size_t n, float mean, float stddev);
  // CHECK: status = rocrand_generate_log_normal(randGenerator, outputPtrFloat, num, mean, stddev);
  status = curandGenerateLogNormal(randGenerator, outputPtrFloat, num, mean, stddev);

  // CUDA: curandStatus_t CURANDAPI curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_long_long(rocrand_generator generator, unsigned long long int* output_data, size_t n);
  // CHECK: status = rocrand_generate_long_long(randGenerator, outputPtrUll, num);
  status = curandGenerateLongLong(randGenerator, outputPtrUll, num);

  // CUDA: curandStatus_t CURANDAPI curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_normal(rocrand_generator generator, float * output_data, size_t n, float mean, float stddev);
  // CHECK: status = rocrand_generate_normal(randGenerator, outputPtrFloat, num, mean, stddev);
  status = curandGenerateNormal(randGenerator, outputPtrFloat, num, mean, stddev);

  // CUDA: curandStatus_t CURANDAPI curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_normal_double(rocrand_generator generator, double * output_data, size_t n, double mean, double stddev);
  // CHECK: status = rocrand_generate_normal_double(randGenerator, outputPtrDouble, num, dmean, dstddev);
  status = curandGenerateNormalDouble(randGenerator, outputPtrDouble, num, dmean, dstddev);

  // CUDA: curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_uniform(rocrand_generator generator, float * output_data, size_t n);
  // CHECK: status = rocrand_generate_uniform(randGenerator, outputPtrFloat, num);
  status = curandGenerateUniform(randGenerator, outputPtrFloat, num);

  // CUDA: curandStatus_t CURANDAPI curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_uniform_double(rocrand_generator generator, double * output_data, size_t n);
  // CHECK: status = rocrand_generate_uniform_double(randGenerator, outputPtrDouble, num);
  status = curandGenerateUniformDouble(randGenerator, outputPtrDouble, num);

  // CUDA: curandStatus_t CURANDAPI curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_log_normal_double(rocrand_generator generator, double * output_data, size_t n, double mean, double stddev);
  // CHECK: status = rocrand_generate_log_normal_double(randGenerator, outputPtrDouble, num, dmean, dstddev);
  status = curandGenerateLogNormalDouble(randGenerator, outputPtrDouble, num, dmean, dstddev);

  // CUDA: curandStatus_t CURANDAPI curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda);
  // ROC: rocrand_status ROCRANDAPI rocrand_generate_poisson(rocrand_generator generator, unsigned int * output_data, size_t n, double lambda);
  // CHECK: status = rocrand_generate_poisson(randGenerator, outputPtr, num, dlambda);
  status = curandGeneratePoisson(randGenerator, outputPtr, num, dlambda);

  // CUDA: curandStatus_t CURANDAPI curandGenerateSeeds(curandGenerator_t generator);
  // ROC: rocrand_status ROCRANDAPI rocrand_initialize_generator(rocrand_generator generator);
  // CHECK: status = rocrand_initialize_generator(randGenerator);
  status = curandGenerateSeeds(randGenerator);

  // CUDA: curandStatus_t CURANDAPI curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
  // ROC: rocrand_status ROCRANDAPI rocrand_set_offset(rocrand_generator generator, unsigned long long offset);
  // CHECK: status = rocrand_set_offset(randGenerator, offset);
  status = curandSetGeneratorOffset(randGenerator, offset);

  // CUDA: curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed);
  // ROC: rocrand_status ROCRANDAPI rocrand_set_seed(rocrand_generator generator, unsigned long long seed);
  // CHECK: status = rocrand_set_seed(randGenerator, offset);
  status = curandSetPseudoRandomGeneratorSeed(randGenerator, offset);

  // CUDA: curandStatus_t CURANDAPI curandSetStream(curandGenerator_t generator, cudaStream_t stream);
  // ROC: rocrand_status ROCRANDAPI rocrand_set_stream(rocrand_generator generator, hipStream_t stream);
  // CHECK: status = rocrand_set_stream(randGenerator, stream);
  status = curandSetStream(randGenerator, stream);

  // CUDA: curandStatus_t CURANDAPI curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *discrete_distribution);
  // ROC: rocrand_status ROCRANDAPI rocrand_create_poisson_distribution(double lambda, rocrand_discrete_distribution * discrete_distribution);
  // CHECK: status = rocrand_create_poisson_distribution(dlambda, &discreteDistribution_t);
  status = curandCreatePoissonDistribution(dlambda, &discreteDistribution_t);

  // CUDA: curandStatus_t CURANDAPI curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution);
  // ROC: rocrand_status ROCRANDAPI rocrand_destroy_discrete_distribution(rocrand_discrete_distribution discrete_distribution);
  // CHECK: status = rocrand_destroy_discrete_distribution(discreteDistribution_t);
  status = curandDestroyDistribution(discreteDistribution_t);

  // CUDA: curandStatus_t CURANDAPI curandGetScrambleConstants32(unsigned int * * constants);
  // ROC: rocrand_status ROCRANDAPI rocrand_get_scramble_constants32(const unsigned int** constants);
  // CHECK: status = rocrand_get_scramble_constants32(&constants);
  status = curandGetScrambleConstants32(&constants);

  // CUDA: curandStatus_t CURANDAPI curandGetScrambleConstants64(unsigned long long * * constants);
  // ROC: rocrand_status ROCRANDAPI rocrand_get_scramble_constants64(const unsigned long long** constants);
  // CHECK: status = rocrand_get_scramble_constants64(&constantsLL);
  status = curandGetScrambleConstants64(&constantsLL);

  // CUDA: curandStatus_t CURANDAPI curandGetVersion(int *version);
  // ROC: rocrand_status ROCRANDAPI rocrand_get_version(int * version);
  // CHECK: status = rocrand_get_version(&version);
  status = curandGetVersion(&version);

  // CUDA: curandStatus_t CURANDAPI curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
  // ROC: rocrand_status ROCRANDAPI rocrand_set_ordering(rocrand_generator generator, rocrand_ordering order);
  // CHECK: status = rocrand_set_ordering(randGenerator, randOrdering);
  status = curandSetGeneratorOrdering(randGenerator, randOrdering);

  // CUDA: curandStatus_t CURANDAPI curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);
  // ROC: rocrand_status ROCRANDAPI rocrand_set_quasi_random_generator_dimensions(rocrand_generator generator, unsigned int dimensions);
  // CHECK: status = rocrand_set_quasi_random_generator_dimensions(randGenerator, num_dimensions);
  status = curandSetQuasiRandomGeneratorDimensions(randGenerator, num_dimensions);

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
