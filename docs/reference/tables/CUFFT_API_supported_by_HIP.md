<head>
    <meta charset="UTF-8">
    <meta name="description" content="CUDA APIs supported by HIPIFY">
    <meta name="keywords" content="HIPIFY, HIP, ROCm, CUDA, CUDA2HIP, hipification, hipify-clang, hipify-perl, FFT, cuFFT, cuFFTXt, hipFFT, hipFFTXt">
</head>

# CUFFT API supported by HIP


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, `U`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **U** - Unsupported for CUDA version(s); **E** - Experimental

## **1. CUFFT Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`CUFFT_ALLOC_FAILED`| | | | |`HIPFFT_ALLOC_FAILED`|1.7.0| | | | | |
|`CUFFT_C2C`| | | | |`HIPFFT_C2C`|1.7.0| | | | | |
|`CUFFT_C2R`| | | | |`HIPFFT_C2R`|1.7.0| | | | | |
|`CUFFT_CB_LD_COMPLEX`| | | | |`HIPFFT_CB_LD_COMPLEX`|4.3.0| | | | | |
|`CUFFT_CB_LD_COMPLEX_DOUBLE`| | | | |`HIPFFT_CB_LD_COMPLEX_DOUBLE`|4.3.0| | | | | |
|`CUFFT_CB_LD_REAL`| | | | |`HIPFFT_CB_LD_REAL`|4.3.0| | | | | |
|`CUFFT_CB_LD_REAL_DOUBLE`| | | | |`HIPFFT_CB_LD_REAL_DOUBLE`|4.3.0| | | | | |
|`CUFFT_CB_ST_COMPLEX`| | | | |`HIPFFT_CB_ST_COMPLEX`|4.3.0| | | | | |
|`CUFFT_CB_ST_COMPLEX_DOUBLE`| | | | |`HIPFFT_CB_ST_COMPLEX_DOUBLE`|4.3.0| | | | | |
|`CUFFT_CB_ST_REAL`| | | | |`HIPFFT_CB_ST_REAL`|4.3.0| | | | | |
|`CUFFT_CB_ST_REAL_DOUBLE`| | | | |`HIPFFT_CB_ST_REAL_DOUBLE`|4.3.0| | | | | |
|`CUFFT_CB_UNDEFINED`| | | | |`HIPFFT_CB_UNDEFINED`|4.3.0| | | | | |
|`CUFFT_COMPATIBILITY_DEFAULT`| | | | | | | | | | | |
|`CUFFT_COMPATIBILITY_FFTW_PADDING`| | | | | | | | | | | |
|`CUFFT_COPY_DEVICE_TO_DEVICE`| | | | |`HIPFFT_COPY_DEVICE_TO_DEVICE`|6.0.0| | | | | |
|`CUFFT_COPY_DEVICE_TO_HOST`| | | | |`HIPFFT_COPY_DEVICE_TO_HOST`|6.0.0| | | | | |
|`CUFFT_COPY_HOST_TO_DEVICE`| | | | |`HIPFFT_COPY_HOST_TO_DEVICE`|6.0.0| | | | | |
|`CUFFT_COPY_UNDEFINED`| | | | |`HIPFFT_COPY_UNDEFINED`|6.0.0| | | | | |
|`CUFFT_D2Z`| | | | |`HIPFFT_D2Z`|1.7.0| | | | | |
|`CUFFT_DESC_C2C`|13.1| | | | | | | | | | |
|`CUFFT_DESC_C2R`|13.1| | | | | | | | | | |
|`CUFFT_DESC_DOUBLE`|13.1| | | | | | | | | | |
|`CUFFT_DESC_FORWARD`|13.1| | | | | | | | | | |
|`CUFFT_DESC_HALF`|13.1| | | | | | | | | | |
|`CUFFT_DESC_INVERSE`|13.1| | | | | | | | | | |
|`CUFFT_DESC_R2C`|13.1| | | | | | | | | | |
|`CUFFT_DESC_SINGLE`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_DIRECTION`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_ELEMENTS_PER_THREAD`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_PRECISION`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_SIZE`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_SM`|13.1| | | | | | | | | | |
|`CUFFT_DESC_TRAIT_TYPE`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FATBIN`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FUNC_TRAIT_ELEMENTS_PER_THREAD`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FUNC_TRAIT_SHARED_MEMORY_PER_FFT`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FUNC_TRAIT_STORAGE_SIZE`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FUNC_TRAIT_SUGGESTED_FFTS_PER_BLOCK`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_FUNC_TRAIT_THREADS_PER_FFT`|13.1| | | | | | | | | | |
|`CUFFT_DEVICE_LTOIR`|13.1| | | | | | | | | | |
|`CUFFT_ENABLE_EXPERIMENTAL_API`|13.1| | | | | | | | | | |
|`CUFFT_EXEC_FAILED`| | | | |`HIPFFT_EXEC_FAILED`|1.7.0| | | | | |
|`CUFFT_FORMAT_UNDEFINED`| | | | |`HIPFFT_FORMAT_UNDEFINED`|6.0.0| | | | | |
|`CUFFT_FORWARD`| | | | |`HIPFFT_FORWARD`|1.7.0| | | | | |
|`CUFFT_INCOMPLETE_PARAMETER_LIST`| | | |13.0|`HIPFFT_INCOMPLETE_PARAMETER_LIST`|1.7.0| | | | | |
|`CUFFT_INTERNAL_ERROR`| | | | |`HIPFFT_INTERNAL_ERROR`|1.7.0| | | | | |
|`CUFFT_INVALID_DEVICE`| | | | |`HIPFFT_INVALID_DEVICE`|1.7.0| | | | | |
|`CUFFT_INVALID_PLAN`| | | | |`HIPFFT_INVALID_PLAN`|1.7.0| | | | | |
|`CUFFT_INVALID_SIZE`| | | | |`HIPFFT_INVALID_SIZE`|1.7.0| | | | | |
|`CUFFT_INVALID_TYPE`| | | | |`HIPFFT_INVALID_TYPE`|1.7.0| | | | | |
|`CUFFT_INVALID_VALUE`| | | | |`HIPFFT_INVALID_VALUE`|1.7.0| | | | | |
|`CUFFT_INVERSE`| | | | |`HIPFFT_BACKWARD`|1.7.0| | | | | |
|`CUFFT_LICENSE_ERROR`| | | |13.0| | | | | | | |
|`CUFFT_MISSING_DEPENDENCY`|13.0| | | | | | | | | | |
|`CUFFT_NOT_IMPLEMENTED`| | | | |`HIPFFT_NOT_IMPLEMENTED`|1.7.0| | | | | |
|`CUFFT_NOT_SUPPORTED`|8.0| | | |`HIPFFT_NOT_SUPPORTED`|1.7.0| | | | | |
|`CUFFT_NO_WORKSPACE`| | | | |`HIPFFT_NO_WORKSPACE`|1.7.0| | | | | |
|`CUFFT_NVJITLINK_FAILURE`|13.0| | | | | | | | | | |
|`CUFFT_NVRTC_FAILURE`|13.0| | | | | | | | | | |
|`CUFFT_NVSHMEM_FAILURE`|13.0| | | | | | | | | | |
|`CUFFT_PARSE_ERROR`| | | |13.0|`HIPFFT_PARSE_ERROR`|1.7.0| | | | | |
|`CUFFT_QUERY_1D_FACTORS`| | | | | | | | | | | |
|`CUFFT_QUERY_UNDEFINED`| | | | | | | | | | | |
|`CUFFT_R2C`| | | | |`HIPFFT_R2C`|1.7.0| | | | | |
|`CUFFT_SETUP_FAILED`| | | | |`HIPFFT_SETUP_FAILED`|1.7.0| | | | | |
|`CUFFT_SUCCESS`| | | | |`HIPFFT_SUCCESS`|1.7.0| | | | | |
|`CUFFT_UNALIGNED_DATA`| | | | |`HIPFFT_UNALIGNED_DATA`|1.7.0| | | | | |
|`CUFFT_WORKAREA_MINIMAL`|9.2| | | | | | | | | | |
|`CUFFT_WORKAREA_PERFORMANCE`| | | | | | | | | | | |
|`CUFFT_WORKAREA_USER`|9.2| | | | | | | | | | |
|`CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED`| | | | |`HIPFFT_XT_FORMAT_1D_INPUT_SHUFFLED`|6.0.0| | | | | |
|`CUFFT_XT_FORMAT_DISTRIBUTED_INPUT`|11.8| | | | | | | | | | |
|`CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT`|11.8| | | | | | | | | | |
|`CUFFT_XT_FORMAT_INPLACE`| | | | |`HIPFFT_XT_FORMAT_INPLACE`|6.0.0| | | | | |
|`CUFFT_XT_FORMAT_INPLACE_SHUFFLED`| | | | |`HIPFFT_XT_FORMAT_INPLACE_SHUFFLED`|6.0.0| | | | | |
|`CUFFT_XT_FORMAT_INPUT`| | | | |`HIPFFT_XT_FORMAT_INPUT`|6.0.0| | | | | |
|`CUFFT_XT_FORMAT_OUTPUT`| | | | |`HIPFFT_XT_FORMAT_OUTPUT`|6.0.0| | | | | |
|`CUFFT_Z2D`| | | | |`HIPFFT_Z2D`|1.7.0| | | | | |
|`CUFFT_Z2Z`| | | | |`HIPFFT_Z2Z`|1.7.0| | | | | |
|`FFTW_BACKWARD`| | | | |`FFTW_BACKWARD`|7.1.0| | | | | |
|`FFTW_DESTROY_INPUT`| | | | |`FFTW_DESTROY_INPUT`|7.1.0| | | | | |
|`FFTW_ESTIMATE`| | | | |`FFTW_ESTIMATE`|7.1.0| | | | | |
|`FFTW_EXHAUSTIVE`| | | | |`FFTW_EXHAUSTIVE`|7.1.0| | | | | |
|`FFTW_FORWARD`| | | | |`FFTW_FORWARD`|7.1.0| | | | | |
|`FFTW_INVERSE`| | | | | | | | | | | |
|`FFTW_MEASURE`| | | | |`FFTW_MEASURE`|7.1.0| | | | | |
|`FFTW_PATIENT`| | | | |`FFTW_PATIENT`|7.1.0| | | | | |
|`FFTW_PRESERVE_INPUT`| | | | |`FFTW_PRESERVE_INPUT`|7.1.0| | | | | |
|`FFTW_UNALIGNED`| | | | |`FFTW_UNALIGNED`|7.1.0| | | | | |
|`FFTW_WISDOM_ONLY`| | | | |`FFTW_WISDOM_ONLY`|7.1.0| | | | | |
|`MAX_CUFFT_ERROR`| | | | | | | | | | | |
|`NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS`|12.5| | | | | | | | | | |
|`NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT`|12.4| | | | | | | | | | |
|`cudaLibXtDesc`| | | | |`hipLibXtDesc`|6.0.0| | | | | |
|`cudaLibXtDesc_t`| | | | |`hipLibXtDesc_t`|6.0.0| | | | | |
|`cufftBox3d`|11.8| | | | | | | | | | |
|`cufftBox3d_t`|11.8| | | | | | | | | | |
|`cufftCompatibility`| | | | | | | | | | | |
|`cufftCompatibility_t`| | | | | | | | | | | |
|`cufftComplex`| | | | |`hipfftComplex`|1.7.0| | | | | |
|`cufftDescriptionDirection`|13.1| | | | | | | | | | |
|`cufftDescriptionDirection_t`|13.1| | | | | | | | | | |
|`cufftDescriptionHandle`|13.1| | | | | | | | | | |
|`cufftDescriptionPrecision`|13.1| | | | | | | | | | |
|`cufftDescriptionPrecision_t`|13.1| | | | | | | | | | |
|`cufftDescriptionTrait`|13.1| | | | | | | | | | |
|`cufftDescriptionTrait_t`|13.1| | | | | | | | | | |
|`cufftDescriptionType`|13.1| | | | | | | | | | |
|`cufftDescriptionType_t`|13.1| | | | | | | | | | |
|`cufftDeviceCodeContainer`|13.1| | | | | | | | | | |
|`cufftDeviceCodeContainer_t`|13.1| | | | | | | | | | |
|`cufftDeviceFunctionHandle`|13.1| | | | | | | | | | |
|`cufftDeviceFunctionTrait`|13.1| | | | | | | | | | |
|`cufftDeviceFunctionTrait_t`|13.1| | | | | | | | | | |
|`cufftDeviceHandle`|13.1| | | | | | | | | | |
|`cufftDoubleComplex`| | | | |`hipfftDoubleComplex`|1.7.0| | | | | |
|`cufftDoubleReal`| | | | |`hipfftDoubleReal`|1.7.0| | | | | |
|`cufftHandle`| | | | |`hipfftHandle`|1.7.0| | | | | |
|`cufftProperty`|12.4| | | | | | | | | | |
|`cufftProperty_t`|12.4| | | | | | | | | | |
|`cufftReal`| | | | |`hipfftReal`|1.7.0| | | | | |
|`cufftResult`| | | | |`hipfftResult`|1.7.0| | | | | |
|`cufftResult_t`| | | | |`hipfftResult_t`|1.7.0| | | | | |
|`cufftType`| | | | |`hipfftType`|1.7.0| | | | | |
|`cufftType_t`| | | | |`hipfftType_t`|1.7.0| | | | | |
|`cufftXt1dFactors`| | | | | | | | | | | |
|`cufftXt1dFactors_t`| | | | | | | | | | | |
|`cufftXtCallbackType`| | | | |`hipfftXtCallbackType`|4.3.0| | | | | |
|`cufftXtCallbackType_t`| | | | |`hipfftXtCallbackType_t`|4.3.0| | | | | |
|`cufftXtCopyType`| | | | |`hipfftXtCopyType`|6.0.0| | | | | |
|`cufftXtCopyType_t`| | | | |`hipfftXtCopyType_t`|6.0.0| | | | | |
|`cufftXtQueryType`| | | | | | | | | | | |
|`cufftXtQueryType_t`| | | | | | | | | | | |
|`cufftXtSubFormat`| | | | |`hipfftXtSubFormat`|6.0.0| | | | | |
|`cufftXtSubFormat_t`| | | | |`hipfftXtSubFormat_t`|6.0.0| | | | | |
|`cufftXtWorkAreaPolicy`|9.2| | | | | | | | | | |
|`cufftXtWorkAreaPolicy_t`|9.2| | | | | | | | | | |
|`fftw_complex`| | | | |`fftw_complex`|7.1.0| | | | | |
|`fftw_iodim`| | | | |`fftw_iodim`|7.11.0| | | | | |
|`fftw_iodim64`| | | | |`fftw_iodim64`|7.11.0| | | | | |
|`fftw_plan`| | | | |`fftw_plan`|7.1.0| | | | | |
|`fftwf_complex`| | | | |`fftwf_complex`|7.1.0| | | | | |
|`fftwf_iodim`| | | | |`fftwf_iodim`|7.11.0| | | | | |
|`fftwf_iodim64`| | | | |`fftwf_iodim64`|7.11.0| | | | | |
|`fftwf_plan`| | | | |`fftwf_plan`|7.1.0| | | | | |

## **2. CUFFT API functions**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cufftCallbackLoadC`| | | | |`hipfftCallbackLoadC`|4.3.0| | | | | |
|`cufftCallbackLoadD`| | | | |`hipfftCallbackLoadD`|4.3.0| | | | | |
|`cufftCallbackLoadR`| | | | |`hipfftCallbackLoadR`|4.3.0| | | | | |
|`cufftCallbackLoadZ`| | | | |`hipfftCallbackLoadZ`|4.3.0| | | | | |
|`cufftCallbackStoreC`| | | | |`hipfftCallbackStoreC`|4.3.0| | | | | |
|`cufftCallbackStoreD`| | | | |`hipfftCallbackStoreD`|4.3.0| | | | | |
|`cufftCallbackStoreR`| | | | |`hipfftCallbackStoreR`|4.3.0| | | | | |
|`cufftCallbackStoreZ`| | | | |`hipfftCallbackStoreZ`|4.3.0| | | | | |
|`cufftCreate`| | | | |`hipfftCreate`|1.7.0| | | | | |
|`cufftDescriptionCreate`|13.1| | | | | | | | | | |
|`cufftDescriptionGetTraitInt64`|13.1| | | | | | | | | | |
|`cufftDescriptionSetTraitInt64`|13.1| | | | | | | | | | |
|`cufftDestroy`| | | | |`hipfftDestroy`|1.7.0| | | | | |
|`cufftDeviceCreate`|13.1| | | | | | | | | | |
|`cufftDeviceDestroy`|13.1| | | | | | | | | | |
|`cufftDeviceGetDatabaseStr`|13.1| | | | | | | | | | |
|`cufftDeviceGetDatabaseStrSize`|13.1| | | | | | | | | | |
|`cufftDeviceGetDeviceFunctionTraitInt64`|13.1| | | | | | | | | | |
|`cufftDeviceGetDeviceFunctions`|13.1| | | | | | | | | | |
|`cufftDeviceGetLTOIRSizes`|13.1| | | | | | | | | | |
|`cufftDeviceGetLTOIRs`|13.1| | | | | | | | | | |
|`cufftDeviceGetNumDeviceFunctions`|13.1| | | | | | | | | | |
|`cufftDeviceGetNumLTOIRs`|13.1| | | | | | | | | | |
|`cufftDeviceGetSemanticVersion`|13.2| | | | | | | | | | |
|`cufftDeviceGetVersion`|13.1| | | | | | | | | | |
|`cufftDeviceIsSupported`|13.1| | | | | | | | | | |
|`cufftEstimate1d`| | | | |`hipfftEstimate1d`|1.7.0| | | | | |
|`cufftEstimate2d`| | | | |`hipfftEstimate2d`|1.7.0| | | | | |
|`cufftEstimate3d`| | | | |`hipfftEstimate3d`|1.7.0| | | | | |
|`cufftEstimateMany`| | | | |`hipfftEstimateMany`|1.7.0| | | | | |
|`cufftExecC2C`| | | | |`hipfftExecC2C`|1.7.0| | | | | |
|`cufftExecC2R`| | | | |`hipfftExecC2R`|1.7.0| | | | | |
|`cufftExecD2Z`| | | | |`hipfftExecD2Z`|1.7.0| | | | | |
|`cufftExecR2C`| | | | |`hipfftExecR2C`|1.7.0| | | | | |
|`cufftExecZ2D`| | | | |`hipfftExecZ2D`|1.7.0| | | | | |
|`cufftExecZ2Z`| | | | |`hipfftExecZ2Z`|1.7.0| | | | | |
|`cufftGetPlanPropertyInt64`|12.4| | | | | | | | | | |
|`cufftGetProperty`|8.0| | | |`hipfftGetProperty`|2.6.0| | | | | |
|`cufftGetSize`| | | | |`hipfftGetSize`|1.7.0| | | | | |
|`cufftGetSize1d`| | | | |`hipfftGetSize1d`|1.7.0| | | | | |
|`cufftGetSize2d`| | | | |`hipfftGetSize2d`|1.7.0| | | | | |
|`cufftGetSize3d`| | | | |`hipfftGetSize3d`|1.7.0| | | | | |
|`cufftGetSizeMany`| | | | |`hipfftGetSizeMany`|1.7.0| | | | | |
|`cufftGetSizeMany64`|7.5| | | |`hipfftGetSizeMany64`|1.7.0| | | | | |
|`cufftGetVersion`| | | | |`hipfftGetVersion`|1.7.0| | | | | |
|`cufftMakePlan1d`| | | | |`hipfftMakePlan1d`|1.7.0| | | | | |
|`cufftMakePlan2d`| | | | |`hipfftMakePlan2d`|1.7.0| | | | | |
|`cufftMakePlan3d`| | | | |`hipfftMakePlan3d`|1.7.0| | | | | |
|`cufftMakePlanMany`| | | | |`hipfftMakePlanMany`|1.7.0| | | | | |
|`cufftMakePlanMany64`|7.5| | | |`hipfftMakePlanMany64`|1.7.0| | | | | |
|`cufftPlan1d`| | | | |`hipfftPlan1d`|1.7.0| | | | | |
|`cufftPlan2d`| | | | |`hipfftPlan2d`|1.7.0| | | | | |
|`cufftPlan3d`| | | | |`hipfftPlan3d`|1.7.0| | | | | |
|`cufftPlanMany`| | | | |`hipfftPlanMany`|1.7.0| | | | | |
|`cufftResetPlanProperty`|12.4| | | | | | | | | | |
|`cufftSetAutoAllocation`| | | | |`hipfftSetAutoAllocation`|1.7.0| | | | | |
|`cufftSetPlanPropertyInt64`|12.4| | | | | | | | | | |
|`cufftSetStream`| | | | |`hipfftSetStream`|1.7.0| | | | | |
|`cufftSetWorkArea`| | | | |`hipfftSetWorkArea`|1.7.0| | | | | |
|`cufftXtClearCallback`| | | | |`hipfftXtClearCallback`|4.3.0| | | | | |
|`cufftXtExec`|8.0| | | |`hipfftXtExec`|5.6.0| | | | | |
|`cufftXtExecDescriptor`|8.0| | | |`hipfftXtExecDescriptor`|6.0.0| | | | | |
|`cufftXtExecDescriptorC2C`| | | | |`hipfftXtExecDescriptorC2C`|6.0.0| | | | | |
|`cufftXtExecDescriptorC2R`| | | | |`hipfftXtExecDescriptorC2R`|6.0.0| | | | | |
|`cufftXtExecDescriptorD2Z`| | | | |`hipfftXtExecDescriptorD2Z`|6.0.0| | | | | |
|`cufftXtExecDescriptorR2C`| | | | |`hipfftXtExecDescriptorR2C`|6.0.0| | | | | |
|`cufftXtExecDescriptorZ2D`| | | | |`hipfftXtExecDescriptorZ2D`|6.0.0| | | | | |
|`cufftXtExecDescriptorZ2Z`| | | | |`hipfftXtExecDescriptorZ2Z`|6.0.0| | | | | |
|`cufftXtFree`| | | | |`hipfftXtFree`|6.0.0| | | | | |
|`cufftXtGetSizeMany`|8.0| | | |`hipfftXtGetSizeMany`|5.6.0| | | | | |
|`cufftXtMakePlanMany`|8.0| | | |`hipfftXtMakePlanMany`|5.6.0| | | | | |
|`cufftXtMalloc`| | | | |`hipfftXtMalloc`|6.0.0| | | | | |
|`cufftXtMemcpy`| | | | |`hipfftXtMemcpy`|6.0.0| | | | | |
|`cufftXtQueryPlan`| | | | | | | | | | | |
|`cufftXtSetCallback`| | | | |`hipfftXtSetCallback`|4.3.0| | | | | |
|`cufftXtSetCallbackSharedSize`| | | | |`hipfftXtSetCallbackSharedSize`|4.3.0| | | | | |
|`cufftXtSetDistribution`|11.8| | | | | | | | | | |
|`cufftXtSetGPUs`| | | | |`hipfftXtSetGPUs`|6.0.0| | | | | |
|`cufftXtSetWorkArea`| | | | | | | | | | | |
|`cufftXtSetWorkAreaPolicy`|9.2| | | | | | | | | | |
|`fftw_cleanup`| | | | |`fftw_cleanup`|7.1.0| | | | | |
|`fftw_cost`| | | | |`fftw_cost`|7.1.0| | | | | |
|`fftw_destroy_plan`| | | | |`fftw_destroy_plan`|7.1.0| | | | | |
|`fftw_execute`| | | | |`fftw_execute`|7.1.0| | | | | |
|`fftw_execute_dft`| | | | |`fftw_execute_dft`|7.2.0| | | | | |
|`fftw_execute_dft_c2r`| | | | |`fftw_execute_dft_c2r`|7.2.0| | | | | |
|`fftw_execute_dft_r2c`| | | | |`fftw_execute_dft_r2c`|7.2.0| | | | | |
|`fftw_export_wisdom_to_file`| | | | | | | | | | | |
|`fftw_flops`| | | | |`fftw_flops`|7.1.0| | | | | |
|`fftw_import_wisdom_from_file`| | | | | | | | | | | |
|`fftw_plan_dft`| | | | |`fftw_plan_dft`|7.1.0| | | | | |
|`fftw_plan_dft_1d`| | | | |`fftw_plan_dft_1d`|7.1.0| | | | | |
|`fftw_plan_dft_2d`| | | | |`fftw_plan_dft_2d`|7.1.0| | | | | |
|`fftw_plan_dft_3d`| | | | |`fftw_plan_dft_3d`|7.1.0| | | | | |
|`fftw_plan_dft_c2r`| | | | |`fftw_plan_dft_c2r`|7.1.0| | | | | |
|`fftw_plan_dft_c2r_1d`| | | | |`fftw_plan_dft_c2r_1d`| | | | | | |
|`fftw_plan_dft_c2r_2d`| | | | |`fftw_plan_dft_c2r_2d`|7.1.0| | | | | |
|`fftw_plan_dft_c2r_3d`| | | | |`fftw_plan_dft_c2r_3d`|7.1.0| | | | | |
|`fftw_plan_dft_r2c`| | | | |`fftw_plan_dft_r2c`|7.1.0| | | | | |
|`fftw_plan_dft_r2c_1d`| | | | |`fftw_plan_dft_r2c_1d`|7.1.0| | | | | |
|`fftw_plan_dft_r2c_2d`| | | | |`fftw_plan_dft_r2c_2d`|7.1.0| | | | | |
|`fftw_plan_dft_r2c_3d`| | | | |`fftw_plan_dft_r2c_3d`|7.1.0| | | | | |
|`fftw_plan_guru64_dft`|10.0| | | |`fftw_plan_guru64_dft`|7.11.0| | | | | |
|`fftw_plan_guru64_dft_c2r`|10.0| | | |`fftw_plan_guru64_dft_c2r`|7.11.0| | | | | |
|`fftw_plan_guru64_dft_r2c`|10.0| | | |`fftw_plan_guru64_dft_r2c`|7.11.0| | | | | |
|`fftw_plan_guru_dft`| | | | |`fftw_plan_guru_dft`|7.11.0| | | | | |
|`fftw_plan_guru_dft_c2r`| | | | |`fftw_plan_guru_dft_c2r`|7.11.0| | | | | |
|`fftw_plan_guru_dft_r2c`| | | | |`fftw_plan_guru_dft_r2c`|7.11.0| | | | | |
|`fftw_plan_many_dft`| | | | |`fftw_plan_many_dft`|7.10.0| | | | | |
|`fftw_plan_many_dft_c2r`| | | | |`fftw_plan_many_dft_c2r`|7.10.0| | | | | |
|`fftw_plan_many_dft_r2c`| | | | |`fftw_plan_many_dft_r2c`|7.10.0| | | | | |
|`fftw_print_plan`| | | | |`fftw_print_plan`|7.1.0| | | | | |
|`fftw_set_timelimit`| | | | |`fftw_set_timelimit`|7.1.0| | | | | |
|`fftwf_cleanup`| | | | |`fftwf_cleanup`|7.1.0| | | | | |
|`fftwf_cost`| | | | |`fftwf_cost`|7.1.0| | | | | |
|`fftwf_destroy_plan`| | | | |`fftwf_destroy_plan`|7.1.0| | | | | |
|`fftwf_execute`| | | | |`fftwf_execute`|7.1.0| | | | | |
|`fftwf_execute_dft`| | | | |`fftwf_execute_dft`|7.2.0| | | | | |
|`fftwf_execute_dft_c2r`| | | | |`fftwf_execute_dft_c2r`|7.2.0| | | | | |
|`fftwf_execute_dft_r2c`| | | | |`fftwf_execute_dft_r2c`|7.2.0| | | | | |
|`fftwf_export_wisdom_to_file`| | | | | | | | | | | |
|`fftwf_flops`| | | | |`fftwf_flops`|7.1.0| | | | | |
|`fftwf_import_wisdom_from_file`| | | | | | | | | | | |
|`fftwf_plan_dft`| | | | |`fftwf_plan_dft`|7.1.0| | | | | |
|`fftwf_plan_dft_1d`| | | | |`fftwf_plan_dft_1d`|7.1.0| | | | | |
|`fftwf_plan_dft_2d`| | | | |`fftwf_plan_dft_2d`|7.1.0| | | | | |
|`fftwf_plan_dft_3d`| | | | |`fftwf_plan_dft_3d`|7.1.0| | | | | |
|`fftwf_plan_dft_c2r`| | | | |`fftwf_plan_dft_c2r`|7.1.0| | | | | |
|`fftwf_plan_dft_c2r_1d`| | | | |`fftwf_plan_dft_c2r_1d`|7.1.0| | | | | |
|`fftwf_plan_dft_c2r_2d`| | | | |`fftwf_plan_dft_c2r_2d`|7.1.0| | | | | |
|`fftwf_plan_dft_c2r_3d`| | | | |`fftwf_plan_dft_c2r_3d`|7.1.0| | | | | |
|`fftwf_plan_dft_r2c`| | | | |`fftwf_plan_dft_r2c`|7.1.0| | | | | |
|`fftwf_plan_dft_r2c_1d`| | | | |`fftwf_plan_dft_r2c_1d`|7.1.0| | | | | |
|`fftwf_plan_dft_r2c_2d`| | | | |`fftwf_plan_dft_r2c_2d`|7.1.0| | | | | |
|`fftwf_plan_dft_r2c_3d`| | | | |`fftwf_plan_dft_r2c_3d`|7.1.0| | | | | |
|`fftwf_plan_guru64_dft`|10.0| | | |`fftwf_plan_guru64_dft`|7.11.0| | | | | |
|`fftwf_plan_guru64_dft_c2r`|10.0| | | |`fftwf_plan_guru64_dft_c2r`|7.11.0| | | | | |
|`fftwf_plan_guru64_dft_r2c`|10.0| | | |`fftwf_plan_guru64_dft_r2c`|7.11.0| | | | | |
|`fftwf_plan_guru_dft`| | | | |`fftwf_plan_guru_dft`|7.11.0| | | | | |
|`fftwf_plan_guru_dft_c2r`| | | | |`fftwf_plan_guru_dft_c2r`|7.11.0| | | | | |
|`fftwf_plan_guru_dft_r2c`| | | | |`fftwf_plan_guru_dft_r2c`|7.11.0| | | | | |
|`fftwf_plan_many_dft`| | | | |`fftwf_plan_many_dft`|7.10.0| | | | | |
|`fftwf_plan_many_dft_c2r`| | | | |`fftwf_plan_many_dft_c2r`|7.10.0| | | | | |
|`fftwf_plan_many_dft_r2c`| | | | |`fftwf_plan_many_dft_r2c`|7.10.0| | | | | |
|`fftwf_print_plan`| | | | |`fftwf_print_plan`|7.1.0| | | | | |
|`fftwf_set_timelimit`| | | | |`fftwf_set_timelimit`|7.1.0| | | | | |

