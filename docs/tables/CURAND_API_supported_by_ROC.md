# CURAND API supported by ROC

## **1. CURAND Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CURAND_3RD`| | | | | | | | | | |
|`CURAND_BINARY_SEARCH`| | | | | | | | | | |
|`CURAND_CHOOSE_BEST`| | | | | | | | | | |
|`CURAND_DEFINITION`| | | | | | | | | | |
|`CURAND_DEVICE_API`| | | | | | | | | | |
|`CURAND_DIRECTION_VECTORS_32_JOEKUO6`| | | | | | | | | | |
|`CURAND_DIRECTION_VECTORS_64_JOEKUO6`| | | | | | | | | | |
|`CURAND_DISCRETE_GAUSS`| | | | | | | | | | |
|`CURAND_FAST_REJECTION`| | | | | | | | | | |
|`CURAND_HITR`| | | | | | | | | | |
|`CURAND_ITR`| | | | | | | | | | |
|`CURAND_KNUTH`| | | | | | | | | | |
|`CURAND_M1`| | | | | | | | | | |
|`CURAND_M2`| | | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_BEST`| | | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_DEFAULT`| | | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_DYNAMIC`|11.5| | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_LEGACY`|11.0| | | | | | | | | |
|`CURAND_ORDERING_PSEUDO_SEEDED`| | | | | | | | | | |
|`CURAND_ORDERING_QUASI_DEFAULT`| | | | | | | | | | |
|`CURAND_POISSON`| | | | | | | | | | |
|`CURAND_REJECTION`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_DEFAULT`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_MRG32K3A`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_MT19937`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_MTGP32`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_PHILOX4_32_10`| | | | | | | | | | |
|`CURAND_RNG_PSEUDO_XORWOW`| | | | | | | | | | |
|`CURAND_RNG_QUASI_DEFAULT`| | | | | | | | | | |
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`| | | | | | | | | | |
|`CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`| | | | | | | | | | |
|`CURAND_RNG_QUASI_SOBOL32`| | | | | | | | | | |
|`CURAND_RNG_QUASI_SOBOL64`| | | | | | | | | | |
|`CURAND_RNG_TEST`| | | | | | | | | | |
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6`| | | | | | | | | | |
|`CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6`| | | | | | | | | | |
|`CURAND_STATUS_ALLOCATION_FAILED`| | | | | | | | | | |
|`CURAND_STATUS_ARCH_MISMATCH`| | | | | | | | | | |
|`CURAND_STATUS_DOUBLE_PRECISION_REQUIRED`| | | | | | | | | | |
|`CURAND_STATUS_INITIALIZATION_FAILED`| | | | | | | | | | |
|`CURAND_STATUS_INTERNAL_ERROR`| | | | | | | | | | |
|`CURAND_STATUS_LAUNCH_FAILURE`| | | | | | | | | | |
|`CURAND_STATUS_LENGTH_NOT_MULTIPLE`| | | | | | | | | | |
|`CURAND_STATUS_NOT_INITIALIZED`| | | | | | | | | | |
|`CURAND_STATUS_OUT_OF_RANGE`| | | | | | | | | | |
|`CURAND_STATUS_PREEXISTING_FAILURE`| | | | | | | | | | |
|`CURAND_STATUS_SUCCESS`| | | | | | | | | | |
|`CURAND_STATUS_TYPE_ERROR`| | | | | | | | | | |
|`CURAND_STATUS_VERSION_MISMATCH`| | | | | | | | | | |
|`curandDirectionVectorSet`| | | | | | | | | | |
|`curandDirectionVectorSet_t`| | | | | | | | | | |
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
|`curandGenerator_st`| | | | | | | | | | |
|`curandGenerator_t`| | | | | | | | | | |
|`curandHistogramM2K_st`| | | | | | | | | | |
|`curandHistogramM2K_t`| | | | | | | | | | |
|`curandHistogramM2V_st`| | | | | | | | | | |
|`curandHistogramM2V_t`| | | | | | | | | | |
|`curandHistogramM2_st`| | | | | | | | | | |
|`curandHistogramM2_t`| | | | | | | | | | |
|`curandMethod`| | | | | | | | | | |
|`curandMethod_t`| | | | | | | | | | |
|`curandOrdering`| | | | | | | | | | |
|`curandOrdering_t`| | | | | | | | | | |
|`curandRngType`| | | | | | | | | | |
|`curandRngType_t`| | | | | | | | | | |
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
|`curandStatus`| | | | |`rocrand_status`|1.5.1| | | | |
|`curandStatus_t`| | | | |`rocrand_status`|1.5.1| | | | |

## **2. Host API Functions**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`curandCreateGenerator`| | | | | | | | | | |
|`curandCreateGeneratorHost`| | | | | | | | | | |
|`curandCreatePoissonDistribution`| | | | | | | | | | |
|`curandDestroyDistribution`| | | | | | | | | | |
|`curandDestroyGenerator`| | | | | | | | | | |
|`curandGenerate`| | | | | | | | | | |
|`curandGenerateLogNormal`| | | | | | | | | | |
|`curandGenerateLogNormalDouble`| | | | | | | | | | |
|`curandGenerateLongLong`| | | | | | | | | | |
|`curandGenerateNormal`| | | | | | | | | | |
|`curandGenerateNormalDouble`| | | | | | | | | | |
|`curandGeneratePoisson`| | | | | | | | | | |
|`curandGenerateSeeds`| | | | | | | | | | |
|`curandGenerateUniform`| | | | | | | | | | |
|`curandGenerateUniformDouble`| | | | | | | | | | |
|`curandGetDirectionVectors32`| | | | | | | | | | |
|`curandGetDirectionVectors64`| | | | | | | | | | |
|`curandGetProperty`|8.0| | | | | | | | | |
|`curandGetScrambleConstants32`| | | | | | | | | | |
|`curandGetScrambleConstants64`| | | | | | | | | | |
|`curandGetVersion`| | | | | | | | | | |
|`curandMakeMTGP32Constants`| | | | | | | | | | |
|`curandMakeMTGP32KernelState`| | | | | | | | | | |
|`curandSetGeneratorOffset`| | | | | | | | | | |
|`curandSetGeneratorOrdering`| | | | | | | | | | |
|`curandSetPseudoRandomGeneratorSeed`| | | | | | | | | | |
|`curandSetQuasiRandomGeneratorDimensions`| | | | | | | | | | |
|`curandSetStream`| | | | | | | | | | |

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