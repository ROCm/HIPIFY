# CUSOLVER API supported by HIP and ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`HIPSOLVER_EIG_TYPE_1`|4.5.0| | | |6.0.0|`rocblas_eform_ax`|4.2.0| | | |6.0.0|
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`HIPSOLVER_EIG_TYPE_2`|4.5.0| | | |6.0.0|`rocblas_eform_abx`|4.2.0| | | |6.0.0|
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`HIPSOLVER_EIG_TYPE_3`|4.5.0| | | |6.0.0|`rocblas_eform_bax`|4.2.0| | | |6.0.0|
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`HIPSOLVER_STATUS_ALLOC_FAILED`|4.5.0| | | |6.0.0|`rocblas_status_memory_error`|5.6.0| | | |6.0.0|
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`HIPSOLVER_STATUS_ARCH_MISMATCH`|4.5.0| | | |6.0.0|`rocblas_status_arch_mismatch`|5.7.0| | | |6.0.0|
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`HIPSOLVER_STATUS_EXECUTION_FAILED`|4.5.0| | | |6.0.0|`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | |6.0.0|`rocblas_status_internal_error`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | |6.0.0|`rocblas_status_invalid_value`|3.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_INVALID_WORKSPACE`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INTERNAL_ERROR`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_MATRIX_SINGULAR`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NOT_SUPPORTED`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_OUT_OF_RANGE`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE`|11.0| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED`|10.2| | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`HIPSOLVER_STATUS_MAPPING_ERROR`|4.5.0| | | |6.0.0|`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`HIPSOLVER_STATUS_NOT_INITIALIZED`|4.5.0| | | |6.0.0|`rocblas_status_invalid_handle`|5.6.0| | | |6.0.0|
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | |6.0.0|`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`CUSOLVER_STATUS_SUCCESS`| | | | |`HIPSOLVER_STATUS_SUCCESS`|4.5.0| | | |6.0.0|`rocblas_status_success`|3.0.0| | | |6.0.0|
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`HIPSOLVER_STATUS_ZERO_PIVOT`| | | | | |`rocblas_status_not_implemented`|1.5.0| | | |6.0.0|
|`cusolverDnHandle_t`| | | | |`hipsolverHandle_t`|4.5.0| | | |6.0.0|`rocblas_handle`|1.5.0| | | |6.0.0|
|`cusolverEigType_t`|8.0| | | |`hipsolverEigType_t`|4.5.0| | | |6.0.0|`rocblas_eform`|4.2.0| | | |6.0.0|
|`cusolverStatus_t`| | | | |`hipsolverStatus_t`|4.5.0| | | |6.0.0|`rocblas_status`|3.0.0| | | |6.0.0|

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCreate`| | | | |`hipsolverDnCreate`|5.1.0| | | |6.0.0|`rocblas_create_handle`| | | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnDestroy`| | | | |`hipsolverDnDestroy`|5.1.0| | | |6.0.0|`rocblas_destroy_handle`| | | | | |
|`cusolverDnDgetrf`| | | | |`hipsolverDnDgetrf`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnDgetrf_bufferSize`| | | | |`hipsolverDnDgetrf_bufferSize`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnDgetrs`| | | | |`hipsolverDnDgetrs`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | | | | | | | |
|`cusolverDnSgetrf`| | | | |`hipsolverDnSgetrf`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnSgetrf_bufferSize`| | | | |`hipsolverDnSgetrf_bufferSize`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnSgetrs`| | | | |`hipsolverDnSgetrs`|5.1.0| | | |6.0.0| | | | | | |
|`cusolverDnXgetrf`|11.1| | | | | | | | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental