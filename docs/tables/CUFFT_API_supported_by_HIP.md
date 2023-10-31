# CUFFT API supported by HIP

## **1. CUFFT Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUFFT_ALLOC_FAILED`| | | | |`HIPFFT_ALLOC_FAILED`|1.7.0| | | | |
|`CUFFT_C2C`| | | | |`HIPFFT_C2C`|1.7.0| | | | |
|`CUFFT_C2R`| | | | |`HIPFFT_C2R`|1.7.0| | | | |
|`CUFFT_CB_LD_COMPLEX`| | | | |`HIPFFT_CB_LD_COMPLEX`|4.3.0| | | | |
|`CUFFT_CB_LD_COMPLEX_DOUBLE`| | | | |`HIPFFT_CB_LD_COMPLEX_DOUBLE`|4.3.0| | | | |
|`CUFFT_CB_LD_REAL`| | | | |`HIPFFT_CB_LD_REAL`|4.3.0| | | | |
|`CUFFT_CB_LD_REAL_DOUBLE`| | | | |`HIPFFT_CB_LD_REAL_DOUBLE`|4.3.0| | | | |
|`CUFFT_CB_ST_COMPLEX`| | | | |`HIPFFT_CB_ST_COMPLEX`|4.3.0| | | | |
|`CUFFT_CB_ST_COMPLEX_DOUBLE`| | | | |`HIPFFT_CB_ST_COMPLEX_DOUBLE`|4.3.0| | | | |
|`CUFFT_CB_ST_REAL`| | | | |`HIPFFT_CB_ST_REAL`|4.3.0| | | | |
|`CUFFT_CB_ST_REAL_DOUBLE`| | | | |`HIPFFT_CB_ST_REAL_DOUBLE`|4.3.0| | | | |
|`CUFFT_CB_UNDEFINED`| | | | |`HIPFFT_CB_UNDEFINED`|4.3.0| | | | |
|`CUFFT_COMPATIBILITY_DEFAULT`| | | | | | | | | | |
|`CUFFT_COMPATIBILITY_FFTW_PADDING`| | | | | | | | | | |
|`CUFFT_COPY_DEVICE_TO_DEVICE`| | | | | | | | | | |
|`CUFFT_COPY_DEVICE_TO_HOST`| | | | | | | | | | |
|`CUFFT_COPY_HOST_TO_DEVICE`| | | | | | | | | | |
|`CUFFT_COPY_UNDEFINED`| | | | | | | | | | |
|`CUFFT_D2Z`| | | | |`HIPFFT_D2Z`|1.7.0| | | | |
|`CUFFT_EXEC_FAILED`| | | | |`HIPFFT_EXEC_FAILED`|1.7.0| | | | |
|`CUFFT_FORMAT_UNDEFINED`| | | | | | | | | | |
|`CUFFT_FORWARD`| | | | |`HIPFFT_FORWARD`|1.7.0| | | | |
|`CUFFT_INCOMPLETE_PARAMETER_LIST`| | | | |`HIPFFT_INCOMPLETE_PARAMETER_LIST`|1.7.0| | | | |
|`CUFFT_INTERNAL_ERROR`| | | | |`HIPFFT_INTERNAL_ERROR`|1.7.0| | | | |
|`CUFFT_INVALID_DEVICE`| | | | |`HIPFFT_INVALID_DEVICE`|1.7.0| | | | |
|`CUFFT_INVALID_PLAN`| | | | |`HIPFFT_INVALID_PLAN`|1.7.0| | | | |
|`CUFFT_INVALID_SIZE`| | | | |`HIPFFT_INVALID_SIZE`|1.7.0| | | | |
|`CUFFT_INVALID_TYPE`| | | | |`HIPFFT_INVALID_TYPE`|1.7.0| | | | |
|`CUFFT_INVALID_VALUE`| | | | |`HIPFFT_INVALID_VALUE`|1.7.0| | | | |
|`CUFFT_INVERSE`| | | | |`HIPFFT_BACKWARD`|1.7.0| | | | |
|`CUFFT_LICENSE_ERROR`| | | | | | | | | | |
|`CUFFT_NOT_IMPLEMENTED`| | | | |`HIPFFT_NOT_IMPLEMENTED`|1.7.0| | | | |
|`CUFFT_NOT_SUPPORTED`|8.0| | | |`HIPFFT_NOT_SUPPORTED`|1.7.0| | | | |
|`CUFFT_NO_WORKSPACE`| | | | |`HIPFFT_NO_WORKSPACE`|1.7.0| | | | |
|`CUFFT_PARSE_ERROR`| | | | |`HIPFFT_PARSE_ERROR`|1.7.0| | | | |
|`CUFFT_QUERY_1D_FACTORS`| | | | | | | | | | |
|`CUFFT_QUERY_UNDEFINED`| | | | | | | | | | |
|`CUFFT_R2C`| | | | |`HIPFFT_R2C`|1.7.0| | | | |
|`CUFFT_SETUP_FAILED`| | | | |`HIPFFT_SETUP_FAILED`|1.7.0| | | | |
|`CUFFT_SUCCESS`| | | | |`HIPFFT_SUCCESS`|1.7.0| | | | |
|`CUFFT_UNALIGNED_DATA`| | | | |`HIPFFT_UNALIGNED_DATA`|1.7.0| | | | |
|`CUFFT_WORKAREA_MINIMAL`|9.2| | | | | | | | | |
|`CUFFT_WORKAREA_PERFORMANCE`| | | | | | | | | | |
|`CUFFT_WORKAREA_USER`|9.2| | | | | | | | | |
|`CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED`| | | | | | | | | | |
|`CUFFT_XT_FORMAT_DISTRIBUTED_INPUT`|11.8| | | | | | | | | |
|`CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT`|11.8| | | | | | | | | |
|`CUFFT_XT_FORMAT_INPLACE`| | | | | | | | | | |
|`CUFFT_XT_FORMAT_INPLACE_SHUFFLED`| | | | | | | | | | |
|`CUFFT_XT_FORMAT_INPUT`| | | | | | | | | | |
|`CUFFT_XT_FORMAT_OUTPUT`| | | | | | | | | | |
|`CUFFT_Z2D`| | | | |`HIPFFT_Z2D`|1.7.0| | | | |
|`CUFFT_Z2Z`| | | | |`HIPFFT_Z2Z`|1.7.0| | | | |
|`MAX_CUFFT_ERROR`| | | | | | | | | | |
|`cufftBox3d`|11.8| | | | | | | | | |
|`cufftBox3d_t`|11.8| | | | | | | | | |
|`cufftCompatibility`| | | | | | | | | | |
|`cufftCompatibility_t`| | | | | | | | | | |
|`cufftComplex`| | | | |`hipfftComplex`|1.7.0| | | | |
|`cufftDoubleComplex`| | | | |`hipfftDoubleComplex`|1.7.0| | | | |
|`cufftDoubleReal`| | | | |`hipfftDoubleReal`|1.7.0| | | | |
|`cufftHandle`| | | | |`hipfftHandle`|1.7.0| | | | |
|`cufftReal`| | | | |`hipfftReal`|1.7.0| | | | |
|`cufftResult`| | | | |`hipfftResult`|1.7.0| | | | |
|`cufftResult_t`| | | | |`hipfftResult_t`|1.7.0| | | | |
|`cufftType`| | | | |`hipfftType`|1.7.0| | | | |
|`cufftType_t`| | | | |`hipfftType_t`|1.7.0| | | | |
|`cufftXt1dFactors`| | | | | | | | | | |
|`cufftXt1dFactors_t`| | | | | | | | | | |
|`cufftXtCallbackType`| | | | |`hipfftXtCallbackType`|4.3.0| | | | |
|`cufftXtCallbackType_t`| | | | |`hipfftXtCallbackType_t`|4.3.0| | | | |
|`cufftXtCopyType`| | | | | | | | | | |
|`cufftXtCopyType_t`| | | | | | | | | | |
|`cufftXtQueryType`| | | | | | | | | | |
|`cufftXtQueryType_t`| | | | | | | | | | |
|`cufftXtSubFormat`| | | | | | | | | | |
|`cufftXtSubFormat_t`| | | | | | | | | | |
|`cufftXtWorkAreaPolicy`|9.2| | | | | | | | | |
|`cufftXtWorkAreaPolicy_t`|9.2| | | | | | | | | |

## **2. CUFFT API functions**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cufftCallbackLoadC`| | | | |`hipfftCallbackLoadC`|4.3.0| | | | |
|`cufftCallbackLoadD`| | | | |`hipfftCallbackLoadD`|4.3.0| | | | |
|`cufftCallbackLoadR`| | | | |`hipfftCallbackLoadR`|4.3.0| | | | |
|`cufftCallbackLoadZ`| | | | |`hipfftCallbackLoadZ`|4.3.0| | | | |
|`cufftCallbackStoreC`| | | | |`hipfftCallbackStoreC`|4.3.0| | | | |
|`cufftCallbackStoreD`| | | | |`hipfftCallbackStoreD`|4.3.0| | | | |
|`cufftCallbackStoreR`| | | | |`hipfftCallbackStoreR`|4.3.0| | | | |
|`cufftCallbackStoreZ`| | | | |`hipfftCallbackStoreZ`|4.3.0| | | | |
|`cufftCreate`| | | | |`hipfftCreate`|1.7.0| | | | |
|`cufftDestroy`| | | | |`hipfftDestroy`|1.7.0| | | | |
|`cufftEstimate1d`| | | | |`hipfftEstimate1d`|1.7.0| | | | |
|`cufftEstimate2d`| | | | |`hipfftEstimate2d`|1.7.0| | | | |
|`cufftEstimate3d`| | | | |`hipfftEstimate3d`|1.7.0| | | | |
|`cufftEstimateMany`| | | | |`hipfftEstimateMany`|1.7.0| | | | |
|`cufftExecC2C`| | | | |`hipfftExecC2C`|1.7.0| | | | |
|`cufftExecC2R`| | | | |`hipfftExecC2R`|1.7.0| | | | |
|`cufftExecD2Z`| | | | |`hipfftExecD2Z`|1.7.0| | | | |
|`cufftExecR2C`| | | | |`hipfftExecR2C`|1.7.0| | | | |
|`cufftExecZ2D`| | | | |`hipfftExecZ2D`|1.7.0| | | | |
|`cufftExecZ2Z`| | | | |`hipfftExecZ2Z`|1.7.0| | | | |
|`cufftGetProperty`|8.0| | | |`hipfftGetProperty`|2.6.0| | | | |
|`cufftGetSize`| | | | |`hipfftGetSize`|1.7.0| | | | |
|`cufftGetSize1d`| | | | |`hipfftGetSize1d`|1.7.0| | | | |
|`cufftGetSize2d`| | | | |`hipfftGetSize2d`|1.7.0| | | | |
|`cufftGetSize3d`| | | | |`hipfftGetSize3d`|1.7.0| | | | |
|`cufftGetSizeMany`| | | | |`hipfftGetSizeMany`|1.7.0| | | | |
|`cufftGetSizeMany64`|7.5| | | |`hipfftGetSizeMany64`|1.7.0| | | | |
|`cufftGetVersion`| | | | |`hipfftGetVersion`|1.7.0| | | | |
|`cufftMakePlan1d`| | | | |`hipfftMakePlan1d`|1.7.0| | | | |
|`cufftMakePlan2d`| | | | |`hipfftMakePlan2d`|1.7.0| | | | |
|`cufftMakePlan3d`| | | | |`hipfftMakePlan3d`|1.7.0| | | | |
|`cufftMakePlanMany`| | | | |`hipfftMakePlanMany`|1.7.0| | | | |
|`cufftMakePlanMany64`|7.5| | | |`hipfftMakePlanMany64`|1.7.0| | | | |
|`cufftPlan1d`| | | | |`hipfftPlan1d`|1.7.0| | | | |
|`cufftPlan2d`| | | | |`hipfftPlan2d`|1.7.0| | | | |
|`cufftPlan3d`| | | | |`hipfftPlan3d`|1.7.0| | | | |
|`cufftPlanMany`| | | | |`hipfftPlanMany`|1.7.0| | | | |
|`cufftSetAutoAllocation`| | | | |`hipfftSetAutoAllocation`|1.7.0| | | | |
|`cufftSetStream`| | | | |`hipfftSetStream`|1.7.0| | | | |
|`cufftSetWorkArea`| | | | |`hipfftSetWorkArea`|1.7.0| | | | |
|`cufftXtClearCallback`| | | | |`hipfftXtClearCallback`|4.3.0| | | | |
|`cufftXtExec`|8.0| | | | | | | | | |
|`cufftXtExecDescriptor`|8.0| | | | | | | | | |
|`cufftXtExecDescriptorC2C`| | | | | | | | | | |
|`cufftXtExecDescriptorC2R`| | | | | | | | | | |
|`cufftXtExecDescriptorD2Z`| | | | | | | | | | |
|`cufftXtExecDescriptorR2C`| | | | | | | | | | |
|`cufftXtExecDescriptorZ2D`| | | | | | | | | | |
|`cufftXtExecDescriptorZ2Z`| | | | | | | | | | |
|`cufftXtFree`| | | | | | | | | | |
|`cufftXtGetSizeMany`|8.0| | | | | | | | | |
|`cufftXtMakePlanMany`|8.0| | | | | | | | | |
|`cufftXtMalloc`| | | | | | | | | | |
|`cufftXtMemcpy`| | | | | | | | | | |
|`cufftXtQueryPlan`| | | | | | | | | | |
|`cufftXtSetCallback`| | | | |`hipfftXtSetCallback`|4.3.0| | | | |
|`cufftXtSetCallbackSharedSize`| | | | |`hipfftXtSetCallbackSharedSize`|4.3.0| | | | |
|`cufftXtSetDistribution`|11.8| | | | | | | | | |
|`cufftXtSetGPUs`| | | | | | | | | | |
|`cufftXtSetWorkArea`| | | | | | | | | | |
|`cufftXtSetWorkAreaPolicy`|9.2| | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental