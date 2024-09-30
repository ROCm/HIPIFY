# CURAND API supported by ROC

## **1. CURAND Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CURAND_3RD`| | | | | | | | | | |
|`CURAND_BINARY_SEARCH`| | | | | | | | | | |
|`CURAND_CHOOSE_BEST`| | | | | | | | | | |
|`CURAND_DEFINITION`| | | | | | | | | | |
|`CURAND_DEVICE_API`| | | | | | | | | | |
|`CURAND_DIRECTION_VECTORS_32_JOEKUO6`| | | | |`ROCRAND_DIRECTION_VECTORS_32_JOEKUO6`|6.0.0| | | | |
|`CURAND_DIRECTION_VECTORS_64_JOEKUO6`| | | | |`ROCRAND_DIRECTION_VECTORS_64_JOEKUO6`|6.0.0| | | | |
|`CURAND_DISCRETE_GAUSS`| | | | | | | | | | |
|`CURAND_FAST_REJECTION`| | | | | | | | | | |
|`CURAND_HITR`| | | | | | | | | | |
|`CURAND_ITR`| | | | | | | | | | |
|`CURAND_KNUTH`| | | | | | | | | | |
|`CURAND_M1`| | | | | | | | | | |
|`CURAND_M2`| | | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_BEST`| | | | |`ROCRAND_ORDERING_PSEUDO_BEST`|5.5.0| | | | |
|`CURAND_ORDERING_PSEUDO_DEFAULT`| | | | |`ROCRAND_ORDERING_PSEUDO_DEFAULT`|5.5.0| | | | |
|`CURAND_ORDERING_PSEUDO_DYNAMIC`|11.5| | | |`ROCRAND_ORDERING_PSEUDO_DYNAMIC`|5.5.0| | | | |
|`CURAND_ORDERING_PSEUDO_LEGACY`|11.0| | | |`ROCRAND_ORDERING_PSEUDO_LEGACY`|5.5.0| | | | |
|`CURAND_ORDERING_PSEUDO_SEEDED`| | | | |`ROCRAND_ORDERING_PSEUDO_SEEDED`|5.5.0| | | | |
|`CURAND_ORDERING_QUASI_DEFAULT`| | | | |`ROCRAND_ORDERING_QUASI_DEFAULT`|5.5.0| | | | |
|`CURAND_POISSON`| | | | | | | | | | |
|`CURAND_REJECTION`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_DEFAULT`| | | | |`ROCRAND_RNG_PSEUDO_DEFAULT`|1.5.0| | | | |
|`CURAND_RNG_PSEUDO_MRG32K3A`| | | | |`ROCRAND_RNG_PSEUDO_MRG32K3A`|1.5.0| | | | |
|`CURAND_RNG_PSEUDO_MT19937`| | | | |`ROCRAND_RNG_PSEUDO_MT19937`|5.5.0| | | | |
|`CURAND_RNG_PSEUDO_MTGP32`| | | | |`ROCRAND_RNG_PSEUDO_MTGP32`|1.5.0| | | | |
|`CURAND_RNG_PSEUDO_PHILOX4_32_10`| | | | |`ROCRAND_RNG_PSEUDO_PHILOX4_32_10`|1.5.0| | | | |
|`CURAND_RNG_PSEUDO_XORWOW`| | | | |`ROCRAND_RNG_PSEUDO_XORWOW`|1.5.0| | | | |
|`CURAND_RNG_QUASI_DEFAULT`| | | | |`ROCRAND_RNG_QUASI_DEFAULT`|1.5.0| | | | |
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`| | | | |`ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32`|5.4.0| | | | |
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`| | | | |`ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64`|5.4.0| | | | |
|`CURAND_RNG_QUASI_SOBOL32`| | | | |`ROCRAND_RNG_QUASI_SOBOL32`|1.5.0| | | | |
|`CURAND_RNG_QUASI_SOBOL64`| | | | |`ROCRAND_RNG_QUASI_SOBOL64`|4.5.0| | | | |
|`CURAND_RNG_TEST`| | | | | | | | | | |
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`| | | | |`ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`|6.0.0| | | | |
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`| | | | |`ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`|6.0.0| | | | |
|`CURAND_STATUS_ALLOCATION_FAILED`| | | | |`ROCRAND_STATUS_ALLOCATION_FAILED`|1.5.0| | | | |
|`CURAND_STATUS_ARCH_MISMATCH`| | | | | | | | | | |
|`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`| | | | |`ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED`|1.5.0| | | | |
|`CURAND_STATUS_INITIALIZATION_FAILED`| | | | | | | | | | |
|`CURAND_STATUS_INTERNAL_ERROR`| | | | |`ROCRAND_STATUS_INTERNAL_ERROR`|1.5.0| | | | |
|`CURAND_STATUS_LAUNCH_FAILURE`| | | | |`ROCRAND_STATUS_LAUNCH_FAILURE`|1.5.0| | | | |
|`CURAND_STATUS_LENGTH_NOT_MULTIPLE`| | | | |`ROCRAND_STATUS_LENGTH_NOT_MULTIPLE`|1.5.0| | | | |
|`CURAND_STATUS_NOT_INITIALIZED`| | | | |`ROCRAND_STATUS_NOT_CREATED`|1.5.0| | | | |
|`CURAND_STATUS_OUT_OF_RANGE`| | | | |`ROCRAND_STATUS_OUT_OF_RANGE`|1.5.0| | | | |
|`CURAND_STATUS_PREEXISTING_FAILURE`| | | | | | | | | | |
|`CURAND_STATUS_SUCCESS`| | | | |`ROCRAND_STATUS_SUCCESS`|1.5.0| | | | |
|`CURAND_STATUS_TYPE_ERROR`| | | | |`ROCRAND_STATUS_TYPE_ERROR`|1.5.0| | | | |
|`CURAND_STATUS_VERSION_MISMATCH`| | | | |`ROCRAND_STATUS_VERSION_MISMATCH`|1.5.0| | | | |
|`curandDirectionVectorSet`| | | | |`rocrand_direction_vector_set`|6.0.0| | | | |
|`curandDirectionVectorSet_t`| | | | |`rocrand_direction_vector_set`|6.0.0| | | | |
|`curandDirectionVectors32_t`| | | | | | | | | | |
|`curandDirectionVectors64_t`| | | | | | | | | | |
|`curandDiscreteDistribution_st`| | | | | | | | | | |
|`curandDiscreteDistribution_t`| | | | | | | | | | |
|`curandDistributionM2Shift_st`| | | | | | | | | | |
|`curandDistributionM2Shift_t`| | | | | | | | | | |
|`curandDistributionShift_st`| | | | | | | | | | |
|`curandDistributionShift_t`| | | | | | | | | | |
|`curandDistribution_st`| | | | | | | | | | |
|`curandDistribution_t`| | | | | | | | | | |
|`curandGenerator_st`| | | | |`rocrand_generator_base_type`|1.5.0| | | | |
|`curandGenerator_t`| | | | |`rocrand_generator`|1.5.0| | | | |
|`curandHistogramM2K_st`| | | | | | | | | | |
|`curandHistogramM2K_t`| | | | | | | | | | |
|`curandHistogramM2V_st`| | | | | | | | | | |
|`curandHistogramM2V_t`| | | | | | | | | | |
|`curandHistogramM2_st`| | | | | | | | | | |
|`curandHistogramM2_t`| | | | | | | | | | |
|`curandMethod`| | | | | | | | | | |
|`curandMethod_t`| | | | | | | | | | |
|`curandOrdering`| | | | |`rocrand_ordering`|5.5.0| | | | |
|`curandOrdering_t`| | | | |`rocrand_ordering`|5.5.0| | | | |
|`curandRngType`| | | | |`rocrand_rng_type`|1.5.0| | | | |
|`curandRngType_t`| | | | |`rocrand_rng_type`|1.5.0| | | | |
|`curandState`| | | | | | | | | | |
|`curandStateMRG32k3a`| | | | | | | | | | |
|`curandStateMRG32k3a_t`| | | | | | | | | | |
|`curandStateMtgp32`| | | | | | | | | | |
|`curandStateMtgp32_t`| | | | | | | | | | |
|`curandStatePhilox4_32_10`| | | | | | | | | | |
|`curandStatePhilox4_32_10_t`| | | | | | | | | | |
|`curandStateScrambledSobol32`| | | | | | | | | | |
|`curandStateScrambledSobol32_t`| | | | | | | | | | |
|`curandStateScrambledSobol64`| | | | | | | | | | |
|`curandStateScrambledSobol64_t`| | | | | | | | | | |
|`curandStateSobol32`| | | | | | | | | | |
|`curandStateSobol32_t`| | | | | | | | | | |
|`curandStateSobol64`| | | | | | | | | | |
|`curandStateSobol64_t`| | | | | | | | | | |
|`curandStateXORWOW`| | | | | | | | | | |
|`curandStateXORWOW_t`| | | | | | | | | | |
|`curandState_t`| | | | | | | | | | |
|`curandStatus`| | | | |`rocrand_status`|1.5.0| | | | |
|`curandStatus_t`| | | | |`rocrand_status`|1.5.0| | | | |

## **2. Host API Functions**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`curandCreateGenerator`| | | | |`rocrand_create_generator`|1.5.0| | | | |
|`curandCreateGeneratorHost`| | | | |`rocrand_create_generator_host_blocking`|6.2.0| | | | |
|`curandCreatePoissonDistribution`| | | | | | | | | | |
|`curandDestroyDistribution`| | | | | | | | | | |
|`curandDestroyGenerator`| | | | |`rocrand_destroy_generator`|1.5.0| | | | |
|`curandGenerate`| | | | |`rocrand_generate`|1.5.0| | | | |
|`curandGenerateLogNormal`| | | | |`rocrand_generate_log_normal`|1.5.0| | | | |
|`curandGenerateLogNormalDouble`| | | | |`rocrand_generate_log_normal_double`|1.5.0| | | | |
|`curandGenerateLongLong`| | | | |`rocrand_generate_long_long`|5.4.0| | | | |
|`curandGenerateNormal`| | | | |`rocrand_generate_normal`|1.5.0| | | | |
|`curandGenerateNormalDouble`| | | | |`rocrand_generate_normal_double`|1.5.0| | | | |
|`curandGeneratePoisson`| | | | |`rocrand_generate_poisson`|1.5.0| | | | |
|`curandGenerateSeeds`| | | | |`rocrand_initialize_generator`|1.5.0| | | | |
|`curandGenerateUniform`| | | | |`rocrand_generate_uniform`|1.5.0| | | | |
|`curandGenerateUniformDouble`| | | | |`rocrand_generate_uniform_double`|1.5.0| | | | |
|`curandGetDirectionVectors32`| | | | | | | | | | |
|`curandGetDirectionVectors64`| | | | | | | | | | |
|`curandGetProperty`|8.0| | | | | | | | | |
|`curandGetScrambleConstants32`| | | | | | | | | | |
|`curandGetScrambleConstants64`| | | | | | | | | | |
|`curandGetVersion`| | | | | | | | | | |
|`curandMakeMTGP32Constants`| | | | | | | | | | |
|`curandMakeMTGP32KernelState`| | | | | | | | | | |
|`curandSetGeneratorOffset`| | | | |`rocrand_set_offset`|1.5.0| | | | |
|`curandSetGeneratorOrdering`| | | | | | | | | | |
|`curandSetPseudoRandomGeneratorSeed`| | | | |`rocrand_set_seed`|1.5.0| | | | |
|`curandSetQuasiRandomGeneratorDimensions`| | | | | | | | | | |
|`curandSetStream`| | | | |`rocrand_set_stream`|1.5.0| | | | |

## **3. Device API Functions**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`__curand_umul`|11.5| | | | | | | | | |
|`curand`| | | | | | | | | | |
|`curand_Philox4x32_10`| | | | | | | | | | |
|`curand_discrete`| | | | | | | | | | |
|`curand_discrete4`| | | | | | | | | | |
|`curand_init`| | | | | | | | | | |
|`curand_log_normal`| | | | | | | | | | |
|`curand_log_normal2`| | | | | | | | | | |
|`curand_log_normal2_double`| | | | | | | | | | |
|`curand_log_normal4`| | | | | | | | | | |
|`curand_log_normal4_double`| | | | | | | | | | |
|`curand_log_normal_double`| | | | | | | | | | |
|`curand_mtgp32_single`| | | | | | | | | | |
|`curand_mtgp32_single_specific`| | | | | | | | | | |
|`curand_mtgp32_specific`| | | | | | | | | | |
|`curand_normal`| | | | | | | | | | |
|`curand_normal2`| | | | | | | | | | |
|`curand_normal2_double`| | | | | | | | | | |
|`curand_normal4`| | | | | | | | | | |
|`curand_normal4_double`| | | | | | | | | | |
|`curand_normal_double`| | | | | | | | | | |
|`curand_poisson`| | | | | | | | | | |
|`curand_poisson4`| | | | | | | | | | |
|`curand_uniform`| | | | | | | | | | |
|`curand_uniform2_double`| | | | | | | | | | |
|`curand_uniform4`| | | | | | | | | | |
|`curand_uniform4_double`| | | | | | | | | | |
|`curand_uniform_double`| | | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental