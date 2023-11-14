# CUSOLVER API supported by ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_DIRECT_BACKWARD`|11.0| | | | | | | | | |
|`CUBLAS_DIRECT_FORWARD`|11.0| | | | | | | | | |
|`CUSOLVER_EIG_MODE_NOVECTOR`|8.0| | | |`rocblas_evect_none`|4.1.0| | | |6.1.0|
|`CUSOLVER_EIG_MODE_VECTOR`|8.0| | | |`rocblas_evect_original`|4.1.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_ALL`|10.1| | | |`rocblas_erange_all`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_I`|10.1| | | |`rocblas_erange_index`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_V`|10.1| | | |`rocblas_erange_value`|5.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`rocblas_eform_ax`|4.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`rocblas_eform_abx`|4.2.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`rocblas_eform_bax`|4.2.0| | | |6.1.0|
|`CUSOLVER_FRO_NORM`|10.2| | | | | | | | | |
|`CUSOLVER_INF_NORM`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_CLASSICAL`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_CLASSICAL_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_GMRES_NOPCOND`|11.0| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_NONE`|10.2| | | | | | | | | |
|`CUSOLVER_IRS_REFINE_NOT_SET`|10.2| | | | | | | | | |
|`CUSOLVER_MAX_NORM`|10.2| | | | | | | | | |
|`CUSOLVER_ONE_NORM`|10.2| | | | | | | | | |
|`CUSOLVER_PREC_DD`|10.2| | | | | | | | | |
|`CUSOLVER_PREC_SHT`|10.2| | | | | | | | | |
|`CUSOLVER_PREC_SS`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`rocblas_status_memory_error`|5.6.0| | | |6.1.0|
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`rocblas_status_arch_mismatch`|5.7.0| | | |6.1.0|
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`rocblas_status_internal_error`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`rocblas_status_invalid_value`|3.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_WORKSPACE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INTERNAL_ERROR`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_MATRIX_SINGULAR`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NOT_SUPPORTED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_OUT_OF_RANGE`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`rocblas_status_invalid_handle`|5.6.0| | | |6.1.0|
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_SUCCESS`| | | | |`rocblas_status_success`|3.0.0| | | |6.1.0|
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.1.0|
|`cusolverDirectMode_t`|11.0| | | | | | | | | |
|`cusolverDnHandle_t`| | | | |`rocblas_handle`|1.5.0| | | |6.1.0|
|`cusolverEigMode_t`|8.0| | | |`rocblas_evect`|4.1.0| | | |6.1.0|
|`cusolverEigRange_t`|10.1| | | |`rocblas_erange`|5.2.0| | | |6.1.0|
|`cusolverEigType_t`|8.0| | | |`rocblas_eform`|4.2.0| | | |6.1.0|
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | |
|`cusolverStatus_t`| | | | |`rocblas_status`|3.0.0| | | |6.1.0|

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCreate`| | | | |`rocblas_create_handle`| | | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | |
|`cusolverDnDestroy`| | | | |`rocblas_destroy_handle`| | | | | |
|`cusolverDnDgetrf`| | | | | | | | | | |
|`cusolverDnDgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDgetrs`| | | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | |
|`cusolverDnSgetrf`| | | | | | | | | | |
|`cusolverDnSgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSgetrs`| | | | | | | | | | |
|`cusolverDnXgetrf`|11.1| | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental