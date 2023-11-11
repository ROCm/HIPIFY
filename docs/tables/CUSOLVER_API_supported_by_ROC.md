# CUSOLVER API supported by ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`rocblas_eform_ax`|4.2.0| | | |6.0.0|
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`rocblas_eform_abx`|4.2.0| | | |6.0.0|
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`rocblas_eform_bax`|4.2.0| | | |6.0.0|
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`rocblas_status_memory_error`|5.6.0| | | |6.0.0|
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`rocblas_status_arch_mismatch`|5.7.0| | | |6.0.0|
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`rocblas_status_internal_error`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`rocblas_status_invalid_value`|3.5.0| | | |6.0.0|
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
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`rocblas_status_invalid_handle`|5.6.0| | | |6.0.0|
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_SUCCESS`| | | | |`rocblas_status_success`|3.0.0| | | |6.0.0|
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`cusolverDnHandle_t`| | | | |`rocblas_handle`|1.5.0| | | |6.0.0|
|`cusolverEigType_t`|8.0| | | |`rocblas_eform`|4.2.0| | | |6.0.0|
|`cusolverStatus_t`| | | | |`rocblas_status`|3.0.0| | | |6.0.0|

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