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
|`curandDiscreteDistribution_st`| | | | |`rocrand_discrete_distribution_st`|1.5.0| | | | |
|`curandDiscreteDistribution_t`| | | | |`rocrand_discrete_distribution`|1.5.0| | | | |
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
|`curandStateMRG32k3a`| | | | |`rocrand_device::mrg32k3a_engine`|1.5.0| | | | |
|`curandStateMRG32k3a_t`| | | | |`rocrand_state_mrg32k3a`|1.5.0| | | | |
|`curandStateMtgp32`| | | | |`rocrand_device::mtgp32_engine`|1.5.0| | | | |
|`curandStateMtgp32_t`| | | | |`rocrand_state_mtgp32`|1.5.0| | | | |
|`curandStatePhilox4_32_10`| | | | |`rocrand_device::philox4x32_10_engine`|1.5.0| | | | |
|`curandStatePhilox4_32_10_t`| | | | |`rocrand_state_philox4x32_10`|1.5.0| | | | |
|`curandStateScrambledSobol32`| | | | |`rocrand_device::scrambled_sobol32_engine<false>`|5.4.0| | | | |
|`curandStateScrambledSobol32_t`| | | | |`rocrand_state_scrambled_sobol32`|5.4.0| | | | |
|`curandStateScrambledSobol64`| | | | |`rocrand_device::scrambled_sobol64_engine<false>`|5.4.0| | | | |
|`curandStateScrambledSobol64_t`| | | | |`rocrand_state_scrambled_sobol64`|5.4.0| | | | |
|`curandStateSobol32`| | | | |`rocrand_device::sobol32_engine<false>`|1.5.0| | | | |
|`curandStateSobol32_t`| | | | |`rocrand_state_sobol32`|1.5.0| | | | |
|`curandStateSobol64`| | | | |`rocrand_device::sobol64_engine<false>`|4.5.0| | | | |
|`curandStateSobol64_t`| | | | |`rocrand_state_sobol64`|4.5.0| | | | |
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
|`curandCreatePoissonDistribution`| | | | |`rocrand_create_poisson_distribution`|1.5.0| | | | |
|`curandDestroyDistribution`| | | | |`rocrand_destroy_discrete_distribution`|1.5.0| | | | |
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
|`curandGetDirectionVectors32`| | | | |`rocrand_get_direction_vectors32`|6.0.0| | | | |
|`curandGetDirectionVectors64`| | | | |`rocrand_get_direction_vectors64`|6.0.0| | | | |
|`curandGetProperty`|8.0| | | | | | | | | |
|`curandGetScrambleConstants32`| | | | |`rocrand_get_scramble_constants32`|6.0.0| | | | |
|`curandGetScrambleConstants64`| | | | |`rocrand_get_scramble_constants64`|6.0.0| | | | |
|`curandGetVersion`| | | | |`rocrand_get_version`|1.5.0| | | | |
|`curandMakeMTGP32Constants`| | | | |`rocrand_make_constant`|1.5.0| | | | |
|`curandMakeMTGP32KernelState`| | | | |`rocrand_make_state_mtgp32`|1.5.0| | | | |
|`curandSetGeneratorOffset`| | | | |`rocrand_set_offset`|1.5.0| | | | |
|`curandSetGeneratorOrdering`| | | | |`rocrand_set_ordering`|5.5.0| | | | |
|`curandSetPseudoRandomGeneratorSeed`| | | | |`rocrand_set_seed`|1.5.0| | | | |
|`curandSetQuasiRandomGeneratorDimensions`| | | | |`rocrand_set_quasi_random_generator_dimensions`|1.5.0| | | | |
|`curandSetStream`| | | | |`rocrand_set_stream`|1.5.0| | | | |

## **3. Device API Functions**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`__curand_umul`|11.5| | | | | | | | | |
|`curand`| | | | |`rocrand`|1.5.0| | | | |
|`curand_Philox4x32_10`| | | | | | | | | | |
|`curand_discrete`| | | | |`rocrand_discrete`|1.5.0| | | | |
|`curand_discrete4`| | | | |`rocrand_discrete4`|1.5.0| | | | |
|`curand_init`| | | | |`rocrand_init`|1.5.0| | | | |
|`curand_log_normal`| | | | |`rocrand_log_normal`|1.5.0| | | | |
|`curand_log_normal2`| | | | |`rocrand_log_normal2`|1.5.0| | | | |
|`curand_log_normal2_double`| | | | |`rocrand_log_normal_double2`|1.5.0| | | | |
|`curand_log_normal4`| | | | |`rocrand_log_normal4`|1.5.0| | | | |
|`curand_log_normal4_double`| | | | |`rocrand_log_normal_double4`|1.5.0| | | | |
|`curand_log_normal_double`| | | | |`rocrand_log_normal_double`|1.5.0| | | | |
|`curand_mtgp32_single`| | | | | | | | | | |
|`curand_mtgp32_single_specific`| | | | | | | | | | |
|`curand_mtgp32_specific`| | | | | | | | | | |
|`curand_normal`| | | | |`rocrand_normal`|1.5.0| | | | |
|`curand_normal2`| | | | |`rocrand_normal2`|1.5.0| | | | |
|`curand_normal2_double`| | | | |`rocrand_normal_double2`|1.5.0| | | | |
|`curand_normal4`| | | | |`rocrand_normal4`|1.5.0| | | | |
|`curand_normal4_double`| | | | |`rocrand_normal_double4`|1.5.0| | | | |
|`curand_normal_double`| | | | |`rocrand_normal_double`|1.5.0| | | | |
|`curand_poisson`| | | | |`rocrand_poisson`|1.5.0| | | | |
|`curand_poisson4`| | | | |`rocrand_poisson4`|1.5.0| | | | |
|`curand_uniform`| | | | |`rocrand_uniform`|1.5.0| | | | |
|`curand_uniform2_double`| | | | |`rocrand_uniform_double2`|1.5.0| | | | |
|`curand_uniform4`| | | | |`rocrand_uniform4`|1.5.0| | | | |
|`curand_uniform4_double`| | | | |`rocrand_uniform_double4`|1.5.0| | | | |
|`curand_uniform_double`| | | | |`rocrand_uniform_double`|1.5.0| | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental