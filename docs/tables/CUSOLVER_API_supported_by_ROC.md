# CUSOLVER API supported by ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_DIRECT_BACKWARD`|11.0| | | | | | | | | |
|`CUBLAS_DIRECT_FORWARD`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_COLUMNWISE`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_ROWWISE`|11.0| | | | | | | | | |
|`CUSOLVERDN_GETRF`|11.0| | | | | | | | | |
|`CUSOLVERDN_POTRF`|11.5| | | | | | | | | |
|`CUSOLVER_ALG_0`|11.0| | | | | | | | | |
|`CUSOLVER_ALG_1`|11.0| | | | | | | | | |
|`CUSOLVER_ALG_2`|11.5| | | | | | | | | |
|`CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS`|12.2| | | | | | | | | |
|`CUSOLVER_C_16BF`|11.0| | | | | | | | | |
|`CUSOLVER_C_16F`|11.0| | | | | | | | | |
|`CUSOLVER_C_32F`|11.0| | | | | | | | | |
|`CUSOLVER_C_64F`|11.0| | | | | | | | | |
|`CUSOLVER_C_8I`|11.0| | | | | | | | | |
|`CUSOLVER_C_8U`|11.0| | | | | | | | | |
|`CUSOLVER_C_AP`|11.0| | | | | | | | | |
|`CUSOLVER_C_TF32`|11.0| | | | | | | | | |
|`CUSOLVER_DETERMINISTIC_RESULTS`|12.2| | | | | | | | | |
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
|`CUSOLVER_R_16BF`|11.0| | | | | | | | | |
|`CUSOLVER_R_16F`|11.0| | | | | | | | | |
|`CUSOLVER_R_32F`|11.0| | | | | | | | | |
|`CUSOLVER_R_64F`|11.0| | | | | | | | | |
|`CUSOLVER_R_8I`|11.0| | | | | | | | | |
|`CUSOLVER_R_8U`|11.0| | | | | | | | | |
|`CUSOLVER_R_AP`|11.0| | | | | | | | | |
|`CUSOLVER_R_TF32`|11.0| | | | | | | | | |
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
|`cusolverAlgMode_t`|11.0| | | | | | | | | |
|`cusolverDeterministicMode_t`|12.2| | | | | | | | | |
|`cusolverDirectMode_t`|11.0| | | | | | | | | |
|`cusolverDnContext`| | | | | | | | | | |
|`cusolverDnFunction_t`|11.0| | | | | | | | | |
|`cusolverDnHandle_t`| | | | |`rocblas_handle`|1.5.0| | | |6.1.0|
|`cusolverDnIRSInfos`|10.2| | | | | | | | | |
|`cusolverDnIRSInfos_t`|10.2| | | | | | | | | |
|`cusolverDnIRSParams`|10.2| | | | | | | | | |
|`cusolverDnIRSParams_t`|10.2| | | | | | | | | |
|`cusolverDnParams`|11.0| | | | | | | | | |
|`cusolverDnParams_t`|11.0| | | | | | | | | |
|`cusolverEigMode_t`|8.0| | | |`rocblas_evect`|4.1.0| | | |6.1.0|
|`cusolverEigRange_t`|10.1| | | |`rocblas_erange`|5.2.0| | | |6.1.0|
|`cusolverEigType_t`|8.0| | | |`rocblas_eform`|4.2.0| | | |6.1.0|
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | |
|`cusolverPrecType_t`|11.0| | | | | | | | | |
|`cusolverStatus_t`| | | | |`rocblas_status`|3.0.0| | | |6.1.0|
|`cusolverStorevMode_t`|11.0| | | | | | | | | |
|`cusolver_int_t`|10.1| | | |`rocblas_int`|3.0.0| | | |6.1.0|
|`gesvdjInfo`|9.0| | | | | | | | | |
|`gesvdjInfo_t`|9.0| | | | | | | | | |
|`syevjInfo`|9.0| | | | | | | | | |
|`syevjInfo_t`|9.0| | | | | | | | | |

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCCgesv`|10.2| | | | | | | | | |
|`cusolverDnCEgesv`|11.0| | | | | | | | | |
|`cusolverDnCKgesv`|10.2| | | | | | | | | |
|`cusolverDnCYgesv`|11.0| | | | | | | | | |
|`cusolverDnCreate`| | | | |`rocblas_create_handle`| | | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | |
|`cusolverDnDBgesv`|11.0| | | | | | | | | |
|`cusolverDnDDgesv`|10.2| | | | | | | | | |
|`cusolverDnDHgesv`|10.2| | | | | | | | | |
|`cusolverDnDSgesv`|10.2| | | | | | | | | |
|`cusolverDnDXgesv`|11.0| | | | | | | | | |
|`cusolverDnDestroy`| | | | |`rocblas_destroy_handle`| | | | | |
|`cusolverDnDgetrf`| | | | | | | | | | |
|`cusolverDnDgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDgetrs`| | | | | | | | | | |
|`cusolverDnGetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnGetStream`| | | | |`rocblas_get_stream`| | | | | |
|`cusolverDnIRSInfosCreate`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosDestroy`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosGetMaxIters`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosGetNiters`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosGetOuterNiters`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosGetResidualHistory`|10.2| | | | | | | | | |
|`cusolverDnIRSInfosRequestResidual`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsCreate`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsDestroy`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsDisableFallback`|11.0| | | | | | | | | |
|`cusolverDnIRSParamsEnableFallback`|11.0| | | | | | | | | |
|`cusolverDnIRSParamsGetMaxIters`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetMaxIters`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetMaxItersInner`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetRefinementSolver`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetSolverLowestPrecision`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetSolverMainPrecision`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetSolverPrecisions`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetTol`|10.2| | | | | | | | | |
|`cusolverDnIRSParamsSetTolInner`|10.2| | | | | | | | | |
|`cusolverDnSBgesv`|11.0| | | | | | | | | |
|`cusolverDnSHgesv`|10.2| | | | | | | | | |
|`cusolverDnSSgesv`|10.2| | | | | | | | | |
|`cusolverDnSXgesv`|11.0| | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | |
|`cusolverDnSetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnSetStream`| | | | |`rocblas_set_stream`| | | | | |
|`cusolverDnSgetrf`| | | | | | | | | | |
|`cusolverDnSgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSgetrs`| | | | | | | | | | |
|`cusolverDnXgetrf`|11.1| | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | |
|`cusolverDnZCgesv`|10.2| | | | | | | | | |
|`cusolverDnZEgesv`|11.0| | | | | | | | | |
|`cusolverDnZKgesv`|10.2| | | | | | | | | |
|`cusolverDnZYgesv`|11.0| | | | | | | | | |
|`cusolverDnZZgesv`|10.2| | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental