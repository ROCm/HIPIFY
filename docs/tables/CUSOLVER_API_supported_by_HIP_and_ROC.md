# CUSOLVER API supported by HIP and ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_DIRECT_BACKWARD`|11.0| | | | | | | | | | | | | | | |
|`CUBLAS_DIRECT_FORWARD`|11.0| | | | | | | | | | | | | | | |
|`CUBLAS_STOREV_COLUMNWISE`|11.0| | | | | | | | | | | | | | | |
|`CUBLAS_STOREV_ROWWISE`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVERDN_GETRF`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVERDN_POTRF`|11.5| | | | | | | | | | | | | | | |
|`CUSOLVER_ALG_0`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_ALG_1`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_ALG_2`|11.5| | | | | | | | | | | | | | | |
|`CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS`|12.2| | | | | | | | | | | | | | | |
|`CUSOLVER_C_16BF`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_16F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_32F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_64F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_8I`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_8U`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_AP`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_C_TF32`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_DETERMINISTIC_RESULTS`|12.2| | | | | | | | | | | | | | | |
|`CUSOLVER_EIG_MODE_NOVECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_NOVECTOR`|4.5.0| | | |6.1.0|`rocblas_evect_none`|4.1.0| | | |6.1.0|
|`CUSOLVER_EIG_MODE_VECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_VECTOR`|4.5.0| | | |6.1.0|`rocblas_evect_original`|4.1.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_ALL`|10.1| | | |`HIPSOLVER_EIG_RANGE_ALL`|5.3.0| | | |6.1.0|`rocblas_erange_all`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_I`|10.1| | | |`HIPSOLVER_EIG_RANGE_I`|5.3.0| | | |6.1.0|`rocblas_erange_index`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_V`|10.1| | | |`HIPSOLVER_EIG_RANGE_V`|5.3.0| | | |6.1.0|`rocblas_erange_value`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`HIPSOLVER_EIG_TYPE_1`|4.5.0| | | |6.1.0|`rocblas_eform_ax`|4.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`HIPSOLVER_EIG_TYPE_2`|4.5.0| | | |6.1.0|`rocblas_eform_abx`|4.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`HIPSOLVER_EIG_TYPE_3`|4.5.0| | | |6.1.0|`rocblas_eform_bax`|4.2.0| | | |6.1.0|
|`CUSOLVER_FRO_NORM`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_INF_NORM`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_CLASSICAL`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_CLASSICAL_GMRES`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES_GMRES`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES_NOPCOND`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_NONE`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_IRS_REFINE_NOT_SET`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_MAX_NORM`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_ONE_NORM`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_PREC_DD`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_PREC_SHT`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_PREC_SS`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_R_16BF`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_16F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_32F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_64F`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_8I`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_8U`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_AP`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_R_TF32`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`HIPSOLVER_STATUS_ALLOC_FAILED`|4.5.0| | | |6.1.0|`rocblas_status_memory_error`|5.6.0| | | |6.1.0|
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`HIPSOLVER_STATUS_ARCH_MISMATCH`|4.5.0| | | |6.1.0|`rocblas_status_arch_mismatch`|5.7.0| | | |6.1.0|
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`HIPSOLVER_STATUS_EXECUTION_FAILED`|4.5.0| | | |6.1.0|`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | |6.1.0|`rocblas_status_internal_error`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | |6.1.0|`rocblas_status_invalid_value`|3.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_WORKSPACE`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INTERNAL_ERROR`|10.2| | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | |6.1.0| | | | | | |
|`CUSOLVER_STATUS_IRS_MATRIX_SINGULAR`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NOT_SUPPORTED`|10.2| | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | |6.1.0| | | | | | |
|`CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_OUT_OF_RANGE`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID`|10.2| | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | |6.1.0| | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`HIPSOLVER_STATUS_MAPPING_ERROR`|4.5.0| | | |6.1.0|`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`HIPSOLVER_STATUS_NOT_INITIALIZED`|4.5.0| | | |6.1.0|`rocblas_status_invalid_handle`|5.6.0| | | |6.1.0|
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | |6.1.0|`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_SUCCESS`| | | | |`HIPSOLVER_STATUS_SUCCESS`|4.5.0| | | |6.1.0|`rocblas_status_success`|3.0.0| | | |6.1.0|
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`HIPSOLVER_STATUS_ZERO_PIVOT`|5.6.0| | | |6.1.0|`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`cusolverAlgMode_t`|11.0| | | | | | | | | | | | | | | |
|`cusolverDeterministicMode_t`|12.2| | | | | | | | | | | | | | | |
|`cusolverDirectMode_t`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnContext`| | | | | | | | | | | | | | | | |
|`cusolverDnFunction_t`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnHandle_t`| | | | |`hipsolverHandle_t`|4.5.0| | | |6.1.0|`rocblas_handle`|1.5.0| | | |6.1.0|
|`cusolverDnIRSInfos`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfos_t`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParams`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParams_t`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnParams`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnParams_t`|11.0| | | | | | | | | | | | | | | |
|`cusolverEigMode_t`|8.0| | | |`hipsolverEigMode_t`|4.5.0| | | |6.1.0|`rocblas_evect`|4.1.0| | | |6.1.0|
|`cusolverEigRange_t`|10.1| | | |`hipsolverEigRange_t`|5.3.0| | | |6.1.0|`rocblas_erange`|5.2.0| | | |6.1.0|
|`cusolverEigType_t`|8.0| | | |`hipsolverEigType_t`|4.5.0| | | |6.1.0|`rocblas_eform`|4.2.0| | | |6.1.0|
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | | | | | | | |
|`cusolverPrecType_t`|11.0| | | | | | | | | | | | | | | |
|`cusolverStatus_t`| | | | |`hipsolverStatus_t`|4.5.0| | | |6.1.0|`rocblas_status`|3.0.0| | | |6.1.0|
|`cusolverStorevMode_t`|11.0| | | | | | | | | | | | | | | |
|`cusolver_int_t`|10.1| | | |`int`| | | | | |`rocblas_int`|3.0.0| | | |6.1.0|
|`gesvdjInfo`|9.0| | | | | | | | | | | | | | | |
|`gesvdjInfo_t`|9.0| | | |`hipsolverGesvdjInfo_t`|5.1.0| | | |6.1.0| | | | | | |
|`syevjInfo`|9.0| | | | | | | | | | | | | | | |
|`syevjInfo_t`|9.0| | | |`hipsolverSyevjInfo_t`|5.1.0| | | |6.1.0| | | | | | |

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCCgels`|11.0| | | |`hipsolverDnCCgels`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCCgels_bufferSize`|11.0| | | |`hipsolverDnCCgels_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCCgesv`|10.2| | | |`hipsolverDnCCgesv`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCCgesv_bufferSize`|10.2| | | |`hipsolverDnCCgesv_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCEgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCEgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCEgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCEgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCKgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCKgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCKgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnCKgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnCYgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCYgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCYgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCYgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCgeqrf`| | | | |`hipsolverDnCgeqrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCgeqrf_bufferSize`| | | | |`hipsolverDnCgeqrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCgetrf`| | | | |`hipsolverDnCgetrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCgetrf_bufferSize`| | | | |`hipsolverDnCgetrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCgetrs`| | | | |`hipsolverDnCgetrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnClaswp`| | | | | | | | | | | | | | | | |
|`cusolverDnClauum`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnClauum_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnCpotrf`| | | | |`hipsolverDnCpotrf`|5.1.0| | | |6.1.0|`rocsolver_cpotrf`|3.6.0| | | |6.1.0|
|`cusolverDnCpotrfBatched`|9.1| | | |`hipsolverDnCpotrfBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCpotrf_bufferSize`| | | | |`hipsolverDnCpotrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCpotri`|10.1| | | |`hipsolverDnCpotri`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCpotri_bufferSize`|10.1| | | |`hipsolverDnCpotri_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCpotrs`| | | | |`hipsolverDnCpotrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCpotrsBatched`|9.1| | | |`hipsolverDnCpotrsBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCreate`| | | | |`hipsolverDnCreate`|5.1.0| | | |6.1.0|`rocblas_create_handle`| | | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnCsytrf`| | | | |`hipsolverDnCsytrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCsytrf_bufferSize`| | | | |`hipsolverDnCsytrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCsytri`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnCsytri_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnCungqr`|8.0| | | |`hipsolverDnCungqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCungqr_bufferSize`|8.0| | | |`hipsolverDnCungqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCunmqr`| | | | |`hipsolverDnCunmqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnCunmqr_bufferSize`|8.0| | | |`hipsolverDnCunmqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDBgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDBgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDBgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDBgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDDgels`|11.0| | | |`hipsolverDnDDgels`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDDgels_bufferSize`|11.0| | | |`hipsolverDnDDgels_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDDgesv`|10.2| | | |`hipsolverDnDDgesv`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDDgesv_bufferSize`|10.2| | | |`hipsolverDnDDgesv_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDHgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDHgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDHgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnDHgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnDSgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDSgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDSgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnDSgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnDXgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDXgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDXgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDXgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDestroy`| | | | |`hipsolverDnDestroy`|5.1.0| | | |6.1.0|`rocblas_destroy_handle`| | | | | |
|`cusolverDnDgeqrf`| | | | |`hipsolverDnDgeqrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDgeqrf_bufferSize`| | | | |`hipsolverDnDgeqrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDgetrf`| | | | |`hipsolverDnDgetrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDgetrf_bufferSize`| | | | |`hipsolverDnDgetrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDgetrs`| | | | |`hipsolverDnDgetrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDlaswp`| | | | | | | | | | | | | | | | |
|`cusolverDnDlauum`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnDlauum_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnDorgqr`|8.0| | | |`hipsolverDnDorgqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDorgqr_bufferSize`|8.0| | | |`hipsolverDnDorgqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDormqr`| | | | |`hipsolverDnDormqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDormqr_bufferSize`|8.0| | | |`hipsolverDnDormqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotrf`| | | | |`hipsolverDnDpotrf`|5.1.0| | | |6.1.0|`rocsolver_dpotrf`|3.2.0| | | |6.1.0|
|`cusolverDnDpotrfBatched`|9.1| | | |`hipsolverDnDpotrfBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotrf_bufferSize`| | | | |`hipsolverDnDpotrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotri`|10.1| | | |`hipsolverDnDpotri`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotri_bufferSize`|10.1| | | |`hipsolverDnDpotri_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotrs`| | | | |`hipsolverDnDpotrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDpotrsBatched`|9.1| | | |`hipsolverDnDpotrsBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDsytrf`| | | | |`hipsolverDnDsytrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDsytrf_bufferSize`| | | | |`hipsolverDnDsytrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnDsytri`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnDsytri_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnGetDeterministicMode`|12.2| | | | | | | | | | | | | | | |
|`cusolverDnGetStream`| | | | |`hipsolverGetStream`|4.5.0| | | |6.1.0|`rocblas_get_stream`| | | | | |
|`cusolverDnIRSInfosCreate`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosDestroy`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosGetMaxIters`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosGetNiters`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosGetOuterNiters`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosGetResidualHistory`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSInfosRequestResidual`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsCreate`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsDestroy`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsDisableFallback`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsEnableFallback`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsGetMaxIters`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetMaxIters`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetMaxItersInner`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetRefinementSolver`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetSolverLowestPrecision`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetSolverMainPrecision`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetSolverPrecisions`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetTol`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSParamsSetTolInner`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSXgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnIRSXgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnIRSXgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnIRSXgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnSBgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSBgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSBgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSBgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSHgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSHgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSHgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnSHgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnSSgels`|11.0| | | |`hipsolverDnSSgels`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSSgels_bufferSize`|11.0| | | |`hipsolverDnSSgels_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSSgesv`|10.2| | | |`hipsolverDnSSgesv`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSSgesv_bufferSize`|10.2| | | |`hipsolverDnSSgesv_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSXgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSXgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSXgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSXgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSetDeterministicMode`|12.2| | | | | | | | | | | | | | | |
|`cusolverDnSetStream`| | | | |`hipsolverSetStream`|4.5.0| | | |6.1.0|`rocblas_set_stream`| | | | | |
|`cusolverDnSgeqrf`| | | | |`hipsolverDnSgeqrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSgeqrf_bufferSize`| | | | |`hipsolverDnSgeqrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSgetrf`| | | | |`hipsolverDnSgetrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSgetrf_bufferSize`| | | | |`hipsolverDnSgetrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSgetrs`| | | | |`hipsolverDnSgetrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSlaswp`| | | | | | | | | | | | | | | | |
|`cusolverDnSlauum`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnSlauum_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnSorgqr`|8.0| | | |`hipsolverDnSorgqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSorgqr_bufferSize`|8.0| | | |`hipsolverDnSorgqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSormqr`| | | | |`hipsolverDnSormqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSormqr_bufferSize`|8.0| | | |`hipsolverDnSormqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotrf`| | | | |`hipsolverDnSpotrf`|5.1.0| | | |6.1.0|`rocsolver_spotrf`|3.2.0| | | |6.1.0|
|`cusolverDnSpotrfBatched`|9.1| | | |`hipsolverDnSpotrfBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotrf_bufferSize`| | | | |`hipsolverDnSpotrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotri`|10.1| | | |`hipsolverDnSpotri`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotri_bufferSize`|10.1| | | |`hipsolverDnSpotri_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotrs`| | | | |`hipsolverDnSpotrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSpotrsBatched`|9.1| | | |`hipsolverDnSpotrsBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSsytrf`| | | | |`hipsolverDnSsytrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSsytrf_bufferSize`| | | | |`hipsolverDnSsytrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnSsytri`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnSsytri_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnXgetrf`|11.1| | | | | | | | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | | | | | | | |
|`cusolverDnXsytrs`|11.3| | | | | | | | | | | | | | | |
|`cusolverDnXsytrs_bufferSize`|11.3| | | | | | | | | | | | | | | |
|`cusolverDnXtrtri`|11.4| | | | | | | | | | | | | | | |
|`cusolverDnXtrtri_bufferSize`|11.4| | | | | | | | | | | | | | | |
|`cusolverDnZCgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZCgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZCgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnZCgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnZEgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZEgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZEgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZEgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZKgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZKgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZKgesv`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnZKgesv_bufferSize`|10.2| | | | | | | | | | | | | | | |
|`cusolverDnZYgels`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZYgels_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZYgesv`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZYgesv_bufferSize`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnZZgels`|11.0| | | |`hipsolverDnZZgels`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZZgels_bufferSize`|11.0| | | |`hipsolverDnZZgels_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZZgesv`|10.2| | | |`hipsolverDnZZgesv`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZZgesv_bufferSize`|10.2| | | |`hipsolverDnZZgesv_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZgeqrf`| | | | |`hipsolverDnZgeqrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZgeqrf_bufferSize`| | | | |`hipsolverDnZgeqrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZgetrf`| | | | |`hipsolverDnZgetrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZgetrf_bufferSize`| | | | |`hipsolverDnZgetrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZgetrs`| | | | |`hipsolverDnZgetrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZlaswp`| | | | | | | | | | | | | | | | |
|`cusolverDnZlauum`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnZlauum_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnZpotrf`| | | | |`hipsolverDnZpotrf`|5.1.0| | | |6.1.0|`rocsolver_zpotrf`|3.6.0| | | |6.1.0|
|`cusolverDnZpotrfBatched`|9.1| | | |`hipsolverDnZpotrfBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZpotrf_bufferSize`| | | | |`hipsolverDnZpotrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZpotri`|10.1| | | |`hipsolverDnZpotri`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZpotri_bufferSize`|10.1| | | |`hipsolverDnZpotri_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZpotrs`| | | | |`hipsolverDnZpotrs`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZpotrsBatched`|9.1| | | |`hipsolverDnZpotrsBatched`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZsytrf`| | | | |`hipsolverDnZsytrf`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZsytrf_bufferSize`| | | | |`hipsolverDnZsytrf_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZsytri`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnZsytri_bufferSize`|10.1| | | | | | | | | | | | | | | |
|`cusolverDnZungqr`|8.0| | | |`hipsolverDnZungqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZungqr_bufferSize`|8.0| | | |`hipsolverDnZungqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZunmqr`| | | | |`hipsolverDnZunmqr`|5.1.0| | | |6.1.0| | | | | | |
|`cusolverDnZunmqr_bufferSize`|8.0| | | |`hipsolverDnZunmqr_bufferSize`|5.1.0| | | |6.1.0| | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental