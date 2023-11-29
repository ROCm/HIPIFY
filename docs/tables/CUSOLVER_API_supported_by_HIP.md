# CUSOLVER API supported by HIP

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
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
|`CUSOLVER_EIG_MODE_NOVECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_NOVECTOR`|4.5.0| | | |6.1.0|
|`CUSOLVER_EIG_MODE_VECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_VECTOR`|4.5.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_ALL`|10.1| | | |`HIPSOLVER_EIG_RANGE_ALL`|5.3.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_I`|10.1| | | |`HIPSOLVER_EIG_RANGE_I`|5.3.0| | | |6.1.0|
|`CUSOLVER_EIG_RANGE_V`|10.1| | | |`HIPSOLVER_EIG_RANGE_V`|5.3.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`HIPSOLVER_EIG_TYPE_1`|4.5.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`HIPSOLVER_EIG_TYPE_2`|4.5.0| | | |6.1.0|
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`HIPSOLVER_EIG_TYPE_3`|4.5.0| | | |6.1.0|
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
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`HIPSOLVER_STATUS_ALLOC_FAILED`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`HIPSOLVER_STATUS_ARCH_MISMATCH`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`HIPSOLVER_STATUS_EXECUTION_FAILED`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_INVALID_WORKSPACE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INTERNAL_ERROR`|10.2| | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_IRS_MATRIX_SINGULAR`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NOT_SUPPORTED`|10.2| | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_OUT_OF_RANGE`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID`|10.2| | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`HIPSOLVER_STATUS_MAPPING_ERROR`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`HIPSOLVER_STATUS_NOT_INITIALIZED`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_SUCCESS`| | | | |`HIPSOLVER_STATUS_SUCCESS`|4.5.0| | | |6.1.0|
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`HIPSOLVER_STATUS_ZERO_PIVOT`|5.6.0| | | |6.1.0|
|`cusolverAlgMode_t`|11.0| | | | | | | | | |
|`cusolverDeterministicMode_t`|12.2| | | | | | | | | |
|`cusolverDirectMode_t`|11.0| | | | | | | | | |
|`cusolverDnContext`| | | | | | | | | | |
|`cusolverDnFunction_t`|11.0| | | | | | | | | |
|`cusolverDnHandle_t`| | | | |`hipsolverHandle_t`|4.5.0| | | |6.1.0|
|`cusolverDnIRSInfos`|10.2| | | | | | | | | |
|`cusolverDnIRSInfos_t`|10.2| | | | | | | | | |
|`cusolverDnIRSParams`|10.2| | | | | | | | | |
|`cusolverDnIRSParams_t`|10.2| | | | | | | | | |
|`cusolverDnParams`|11.0| | | | | | | | | |
|`cusolverDnParams_t`|11.0| | | | | | | | | |
|`cusolverEigMode_t`|8.0| | | |`hipsolverEigMode_t`|4.5.0| | | |6.1.0|
|`cusolverEigRange_t`|10.1| | | |`hipsolverEigRange_t`|5.3.0| | | |6.1.0|
|`cusolverEigType_t`|8.0| | | |`hipsolverEigType_t`|4.5.0| | | |6.1.0|
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | |
|`cusolverPrecType_t`|11.0| | | | | | | | | |
|`cusolverStatus_t`| | | | |`hipsolverStatus_t`|4.5.0| | | |6.1.0|
|`cusolverStorevMode_t`|11.0| | | | | | | | | |
|`cusolver_int_t`|10.1| | | |`int`| | | | | |
|`gesvdjInfo`|9.0| | | | | | | | | |
|`gesvdjInfo_t`|9.0| | | |`hipsolverGesvdjInfo_t`|5.1.0| | | |6.1.0|
|`syevjInfo`|9.0| | | | | | | | | |
|`syevjInfo_t`|9.0| | | |`hipsolverSyevjInfo_t`|5.1.0| | | |6.1.0|

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCCgels`|11.0| | | |`hipsolverDnCCgels`|5.1.0| | | |6.1.0|
|`cusolverDnCCgels_bufferSize`|11.0| | | |`hipsolverDnCCgels_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnCCgesv`|10.2| | | |`hipsolverDnCCgesv`|5.1.0| | | |6.1.0|
|`cusolverDnCCgesv_bufferSize`|10.2| | | |`hipsolverDnCCgesv_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnCEgels`|11.0| | | | | | | | | |
|`cusolverDnCEgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCEgesv`|11.0| | | | | | | | | |
|`cusolverDnCEgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCKgels`|11.0| | | | | | | | | |
|`cusolverDnCKgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCKgesv`|10.2| | | | | | | | | |
|`cusolverDnCKgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnCYgels`|11.0| | | | | | | | | |
|`cusolverDnCYgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCYgesv`|11.0| | | | | | | | | |
|`cusolverDnCYgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCpotrf`| | | | |`hipsolverDnCpotrf`|5.1.0| | | |6.1.0|
|`cusolverDnCpotrfBatched`|9.1| | | |`hipsolverDnCpotrfBatched`|5.1.0| | | |6.1.0|
|`cusolverDnCpotrf_bufferSize`| | | | |`hipsolverDnCpotrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnCpotrs`| | | | |`hipsolverDnCpotrs`|5.1.0| | | |6.1.0|
|`cusolverDnCpotrsBatched`|9.1| | | |`hipsolverDnCpotrsBatched`|5.1.0| | | |6.1.0|
|`cusolverDnCreate`| | | | |`hipsolverDnCreate`|5.1.0| | | |6.1.0|
|`cusolverDnCreateParams`|11.0| | | | | | | | | |
|`cusolverDnDBgels`|11.0| | | | | | | | | |
|`cusolverDnDBgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDBgesv`|11.0| | | | | | | | | |
|`cusolverDnDBgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDDgels`|11.0| | | |`hipsolverDnDDgels`|5.1.0| | | |6.1.0|
|`cusolverDnDDgels_bufferSize`|11.0| | | |`hipsolverDnDDgels_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnDDgesv`|10.2| | | |`hipsolverDnDDgesv`|5.1.0| | | |6.1.0|
|`cusolverDnDDgesv_bufferSize`|10.2| | | |`hipsolverDnDDgesv_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnDHgels`|11.0| | | | | | | | | |
|`cusolverDnDHgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDHgesv`|10.2| | | | | | | | | |
|`cusolverDnDHgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnDSgels`|11.0| | | | | | | | | |
|`cusolverDnDSgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDSgesv`|10.2| | | | | | | | | |
|`cusolverDnDSgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnDXgels`|11.0| | | | | | | | | |
|`cusolverDnDXgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDXgesv`|11.0| | | | | | | | | |
|`cusolverDnDXgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDestroy`| | | | |`hipsolverDnDestroy`|5.1.0| | | |6.1.0|
|`cusolverDnDgetrf`| | | | |`hipsolverDnDgetrf`|5.1.0| | | |6.1.0|
|`cusolverDnDgetrf_bufferSize`| | | | |`hipsolverDnDgetrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnDgetrs`| | | | |`hipsolverDnDgetrs`|5.1.0| | | |6.1.0|
|`cusolverDnDpotrf`| | | | |`hipsolverDnDpotrf`|5.1.0| | | |6.1.0|
|`cusolverDnDpotrfBatched`|9.1| | | |`hipsolverDnDpotrfBatched`|5.1.0| | | |6.1.0|
|`cusolverDnDpotrf_bufferSize`| | | | |`hipsolverDnDpotrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnDpotrs`| | | | |`hipsolverDnDpotrs`|5.1.0| | | |6.1.0|
|`cusolverDnDpotrsBatched`|9.1| | | |`hipsolverDnDpotrsBatched`|5.1.0| | | |6.1.0|
|`cusolverDnGetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnGetStream`| | | | |`hipsolverGetStream`|4.5.0| | | |6.1.0|
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
|`cusolverDnIRSXgels`|11.0| | | | | | | | | |
|`cusolverDnIRSXgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnIRSXgesv`|10.2| | | | | | | | | |
|`cusolverDnIRSXgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnSBgels`|11.0| | | | | | | | | |
|`cusolverDnSBgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSBgesv`|11.0| | | | | | | | | |
|`cusolverDnSBgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSHgels`|11.0| | | | | | | | | |
|`cusolverDnSHgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSHgesv`|10.2| | | | | | | | | |
|`cusolverDnSHgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnSSgels`|11.0| | | |`hipsolverDnSSgels`|5.1.0| | | |6.1.0|
|`cusolverDnSSgels_bufferSize`|11.0| | | |`hipsolverDnSSgels_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnSSgesv`|10.2| | | |`hipsolverDnSSgesv`|5.1.0| | | |6.1.0|
|`cusolverDnSSgesv_bufferSize`|10.2| | | |`hipsolverDnSSgesv_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnSXgels`|11.0| | | | | | | | | |
|`cusolverDnSXgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSXgesv`|11.0| | | | | | | | | |
|`cusolverDnSXgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | |
|`cusolverDnSetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnSetStream`| | | | |`hipsolverSetStream`|4.5.0| | | |6.1.0|
|`cusolverDnSgetrf`| | | | |`hipsolverDnSgetrf`|5.1.0| | | |6.1.0|
|`cusolverDnSgetrf_bufferSize`| | | | |`hipsolverDnSgetrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnSgetrs`| | | | |`hipsolverDnSgetrs`|5.1.0| | | |6.1.0|
|`cusolverDnSpotrf`| | | | |`hipsolverDnSpotrf`|5.1.0| | | |6.1.0|
|`cusolverDnSpotrfBatched`|9.1| | | |`hipsolverDnSpotrfBatched`|5.1.0| | | |6.1.0|
|`cusolverDnSpotrf_bufferSize`| | | | |`hipsolverDnSpotrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnSpotrs`| | | | |`hipsolverDnSpotrs`|5.1.0| | | |6.1.0|
|`cusolverDnSpotrsBatched`|9.1| | | |`hipsolverDnSpotrsBatched`|5.1.0| | | |6.1.0|
|`cusolverDnXgetrf`|11.1| | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | |
|`cusolverDnZCgels`|11.0| | | | | | | | | |
|`cusolverDnZCgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZCgesv`|10.2| | | | | | | | | |
|`cusolverDnZCgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnZEgels`|11.0| | | | | | | | | |
|`cusolverDnZEgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZEgesv`|11.0| | | | | | | | | |
|`cusolverDnZEgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZKgels`|11.0| | | | | | | | | |
|`cusolverDnZKgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZKgesv`|10.2| | | | | | | | | |
|`cusolverDnZKgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnZYgels`|11.0| | | | | | | | | |
|`cusolverDnZYgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZYgesv`|11.0| | | | | | | | | |
|`cusolverDnZYgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZZgels`|11.0| | | |`hipsolverDnZZgels`|5.1.0| | | |6.1.0|
|`cusolverDnZZgels_bufferSize`|11.0| | | |`hipsolverDnZZgels_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnZZgesv`|10.2| | | |`hipsolverDnZZgesv`|5.1.0| | | |6.1.0|
|`cusolverDnZZgesv_bufferSize`|10.2| | | |`hipsolverDnZZgesv_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnZpotrf`| | | | |`hipsolverDnZpotrf`|5.1.0| | | |6.1.0|
|`cusolverDnZpotrfBatched`|9.1| | | |`hipsolverDnZpotrfBatched`|5.1.0| | | |6.1.0|
|`cusolverDnZpotrf_bufferSize`| | | | |`hipsolverDnZpotrf_bufferSize`|5.1.0| | | |6.1.0|
|`cusolverDnZpotrs`| | | | |`hipsolverDnZpotrs`|5.1.0| | | |6.1.0|
|`cusolverDnZpotrsBatched`|9.1| | | |`hipsolverDnZpotrsBatched`|5.1.0| | | |6.1.0|


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental