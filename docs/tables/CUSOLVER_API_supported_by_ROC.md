# CUSOLVER API supported by ROC

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_DIRECT_BACKWARD`|11.0| | | | | | | | | |
|`CUBLAS_DIRECT_FORWARD`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_COLUMNWISE`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_ROWWISE`|11.0| | | | | | | | | |
|`CUDALIBMG_GRID_MAPPING_COL_MAJOR`|10.1| | | | | | | | | |
|`CUDALIBMG_GRID_MAPPING_ROW_MAJOR`|10.1| | | | | | | | | |
|`CUSOLVERDN_GETRF`|11.0| | | | | | | | | |
|`CUSOLVERDN_POTRF`|11.5| | | | | | | | | |
|`CUSOLVERRF_FACTORIZATION_ALG0`| | | | | | | | | | |
|`CUSOLVERRF_FACTORIZATION_ALG1`| | | | | | | | | | |
|`CUSOLVERRF_FACTORIZATION_ALG2`| | | | | | | | | | |
|`CUSOLVERRF_MATRIX_FORMAT_CSC`| | | | | | | | | | |
|`CUSOLVERRF_MATRIX_FORMAT_CSR`| | | | | | | | | | |
|`CUSOLVERRF_NUMERIC_BOOST_NOT_USED`| | | | | | | | | | |
|`CUSOLVERRF_NUMERIC_BOOST_USED`| | | | | | | | | | |
|`CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF`| | | | | | | | | | |
|`CUSOLVERRF_RESET_VALUES_FAST_MODE_ON`| | | | | | | | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG1`| | | | | | | | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG2`| | | | | | | | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG3`| | | | | | | | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L`| | | | | | | | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U`| | | | | | | | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_STORED_L`| | | | | | | | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_STORED_U`| | | | | | | | | | |
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
|`CUSOLVER_EIG_MODE_NOVECTOR`|8.0| | | |`rocblas_evect_none`|4.1.0| | | | |
|`CUSOLVER_EIG_MODE_VECTOR`|8.0| | | |`rocblas_evect_original`|4.1.0| | | | |
|`CUSOLVER_EIG_RANGE_ALL`|10.1| | | |`rocblas_erange_all`|5.2.0| | | | |
|`CUSOLVER_EIG_RANGE_I`|10.1| | | |`rocblas_erange_index`|5.2.0| | | | |
|`CUSOLVER_EIG_RANGE_V`|10.1| | | |`rocblas_erange_value`|5.2.0| | | | |
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`rocblas_eform_ax`|4.2.0| | | | |
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`rocblas_eform_abx`|4.2.0| | | | |
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`rocblas_eform_bax`|4.2.0| | | | |
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
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`rocblas_status_memory_error`|5.6.0| | | | |
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`rocblas_status_arch_mismatch`|5.7.0| | | | |
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | | |
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`rocblas_status_internal_error`|1.5.0| | | | |
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`rocblas_status_invalid_value`|3.5.0| | | | |
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
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`rocblas_status_not_implemented`|1.5.0| | | | |
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | |
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`rocblas_status_invalid_handle`|5.6.0| | | | |
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`rocblas_status_not_implemented`|1.5.0| | | | |
|`CUSOLVER_STATUS_SUCCESS`| | | | |`rocblas_status_success`|3.0.0| | | | |
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`rocblas_status_not_implemented`|1.5.0| | | | |
|`csrcholInfo`|7.5| | | | | | | | | |
|`csrcholInfoHost`|7.5| | | | | | | | | |
|`csrcholInfoHost_t`|7.5| | | | | | | | | |
|`csrcholInfo_t`|7.5| | | | | | | | | |
|`csrluInfoHost`|7.5| | | | | | | | | |
|`csrluInfoHost_t`|7.5| | | | | | | | | |
|`csrqrInfo`| | | | | | | | | | |
|`csrqrInfoHost`|7.5| | | | | | | | | |
|`csrqrInfoHost_t`|7.5| | | | | | | | | |
|`csrqrInfo_t`| | | | | | | | | | |
|`cudaLibMgGrid_t`|10.1| | | | | | | | | |
|`cudaLibMgMatrixDesc_t`|10.1| | | | | | | | | |
|`cusolverAlgMode_t`|11.0| | | | | | | | | |
|`cusolverDeterministicMode_t`|12.2| | | | | | | | | |
|`cusolverDirectMode_t`|11.0| | | | | | | | | |
|`cusolverDnContext`| | | | | | | | | | |
|`cusolverDnFunction_t`|11.0| | | | | | | | | |
|`cusolverDnHandle_t`| | | | |`rocblas_handle`|1.5.0| | | | |
|`cusolverDnIRSInfos`|10.2| | | | | | | | | |
|`cusolverDnIRSInfos_t`|10.2| | | | | | | | | |
|`cusolverDnIRSParams`|10.2| | | | | | | | | |
|`cusolverDnIRSParams_t`|10.2| | | | | | | | | |
|`cusolverDnLoggerCallback_t`|11.7| | | | | | | | | |
|`cusolverDnParams`|11.0| | | | | | | | | |
|`cusolverDnParams_t`|11.0| | | | | | | | | |
|`cusolverEigMode_t`|8.0| | | |`rocblas_evect`|4.1.0| | | | |
|`cusolverEigRange_t`|10.1| | | |`rocblas_erange`|5.2.0| | | | |
|`cusolverEigType_t`|8.0| | | |`rocblas_eform`|4.2.0| | | | |
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | |
|`cusolverMgContext`|10.1| | | | | | | | | |
|`cusolverMgGridMapping_t`|10.1| | | | | | | | | |
|`cusolverMgHandle_t`|10.1| | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | |
|`cusolverPrecType_t`|11.0| | | | | | | | | |
|`cusolverRfCommon`| | | | | | | | | | |
|`cusolverRfFactorization_t`| | | | | | | | | | |
|`cusolverRfHandle_t`| | | | | | | | | | |
|`cusolverRfMatrixFormat_t`| | | | | | | | | | |
|`cusolverRfNumericBoostReport_t`| | | | | | | | | | |
|`cusolverRfResetValuesFastMode_t`| | | | | | | | | | |
|`cusolverRfTriangularSolve_t`| | | | | | | | | | |
|`cusolverRfUnitDiagonal_t`| | | | | | | | | | |
|`cusolverSpContext`| | | | | | | | | | |
|`cusolverSpHandle_t`| | | | | | | | | | |
|`cusolverStatus_t`| | | | |`rocblas_status`|3.0.0| | | | |
|`cusolverStorevMode_t`|11.0| | | | | | | | | |
|`cusolver_int_t`|10.1| | | |`rocblas_int`|3.0.0| | | | |
|`gesvdjInfo`|9.0| | | | | | | | | |
|`gesvdjInfo_t`|9.0| | | | | | | | | |
|`syevjInfo`|9.0| | | | | | | | | |
|`syevjInfo_t`|9.0| | | | | | | | | |

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCCgels`|11.0| | | | | | | | | |
|`cusolverDnCCgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnCCgesv`|10.2| | | | | | | | | |
|`cusolverDnCCgesv_bufferSize`|10.2| | | | | | | | | |
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
|`cusolverDnCgebrd`| | | | | | | | | | |
|`cusolverDnCgebrd_bufferSize`| | | | | | | | | | |
|`cusolverDnCgeqrf`| | | | | | | | | | |
|`cusolverDnCgeqrf_bufferSize`| | | | | | | | | | |
|`cusolverDnCgesvd`| | | | | | | | | | |
|`cusolverDnCgesvd_bufferSize`| | | | | | | | | | |
|`cusolverDnCgesvdaStridedBatched`|10.1| | | | | | | | | |
|`cusolverDnCgesvdaStridedBatched_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCgesvdj`|9.0| | | | | | | | | |
|`cusolverDnCgesvdjBatched`|9.0| | | | | | | | | |
|`cusolverDnCgesvdjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnCgesvdj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnCgetrf`| | | | | | | | | | |
|`cusolverDnCgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnCgetrs`| | | | | | | | | | |
|`cusolverDnCheevd`|8.0| | | | | | | | | |
|`cusolverDnCheevd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnCheevdx`|10.1| | | | | | | | | |
|`cusolverDnCheevdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCheevj`|9.0| | | | | | | | | |
|`cusolverDnCheevjBatched`|9.0| | | | | | | | | |
|`cusolverDnCheevjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnCheevj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnChegvd`|8.0| | | | | | | | | |
|`cusolverDnChegvd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnChegvdx`|10.1| | | | | | | | | |
|`cusolverDnChegvdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnChegvj`|9.0| | | | | | | | | |
|`cusolverDnChegvj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnChetrd`|8.0| | | | | | | | | |
|`cusolverDnChetrd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnClaswp`| | | | | | | | | | |
|`cusolverDnClauum`|10.1| | | | | | | | | |
|`cusolverDnClauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCpotrf`| | | | |`rocsolver_cpotrf`|3.6.0| | | | |
|`cusolverDnCpotrfBatched`|9.1| | | | | | | | | |
|`cusolverDnCpotrf_bufferSize`| | | | | | | | | | |
|`cusolverDnCpotri`|10.1| | | | | | | | | |
|`cusolverDnCpotri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCpotrs`| | | | | | | | | | |
|`cusolverDnCpotrsBatched`|9.1| | | | | | | | | |
|`cusolverDnCreate`| | | | |`rocblas_create_handle`| | | | | |
|`cusolverDnCreateGesvdjInfo`|9.0| | | | | | | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | |
|`cusolverDnCreateSyevjInfo`|9.0| | | | | | | | | |
|`cusolverDnCsytrf`| | | | | | | | | | |
|`cusolverDnCsytrf_bufferSize`| | | | | | | | | | |
|`cusolverDnCsytri`|10.1| | | | | | | | | |
|`cusolverDnCsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCungbr`|8.0| | | | | | | | | |
|`cusolverDnCungbr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnCungqr`|8.0| | | | | | | | | |
|`cusolverDnCungqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnCungtr`|8.0| | | | | | | | | |
|`cusolverDnCungtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnCunmqr`| | | | | | | | | | |
|`cusolverDnCunmqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnCunmtr`|8.0| | | | | | | | | |
|`cusolverDnCunmtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDBgels`|11.0| | | | | | | | | |
|`cusolverDnDBgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDBgesv`|11.0| | | | | | | | | |
|`cusolverDnDBgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDDgels`|11.0| | | | | | | | | |
|`cusolverDnDDgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDDgesv`|10.2| | | | | | | | | |
|`cusolverDnDDgesv_bufferSize`|10.2| | | | | | | | | |
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
|`cusolverDnDestroy`| | | | |`rocblas_destroy_handle`| | | | | |
|`cusolverDnDestroyGesvdjInfo`|9.0| | | | | | | | | |
|`cusolverDnDestroyParams`|11.0| | | | | | | | | |
|`cusolverDnDestroySyevjInfo`|9.0| | | | | | | | | |
|`cusolverDnDgebrd`| | | | | | | | | | |
|`cusolverDnDgebrd_bufferSize`| | | | | | | | | | |
|`cusolverDnDgeqrf`| | | | | | | | | | |
|`cusolverDnDgeqrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDgesvd`| | | | | | | | | | |
|`cusolverDnDgesvd_bufferSize`| | | | | | | | | | |
|`cusolverDnDgesvdaStridedBatched`|10.1| | | | | | | | | |
|`cusolverDnDgesvdaStridedBatched_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDgesvdj`|9.0| | | | | | | | | |
|`cusolverDnDgesvdjBatched`|9.0| | | | | | | | | |
|`cusolverDnDgesvdjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnDgesvdj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnDgetrf`| | | | | | | | | | |
|`cusolverDnDgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDgetrs`| | | | | | | | | | |
|`cusolverDnDlaswp`| | | | | | | | | | |
|`cusolverDnDlauum`|10.1| | | | | | | | | |
|`cusolverDnDlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDorgbr`|8.0| | | | | | | | | |
|`cusolverDnDorgbr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDorgqr`|8.0| | | | | | | | | |
|`cusolverDnDorgqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDorgtr`|8.0| | | | | | | | | |
|`cusolverDnDorgtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDormqr`| | | | | | | | | | |
|`cusolverDnDormqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDormtr`|8.0| | | | | | | | | |
|`cusolverDnDormtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDpotrf`| | | | |`rocsolver_dpotrf`|3.2.0| | | | |
|`cusolverDnDpotrfBatched`|9.1| | | | | | | | | |
|`cusolverDnDpotrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDpotri`|10.1| | | | | | | | | |
|`cusolverDnDpotri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDpotrs`| | | | | | | | | | |
|`cusolverDnDpotrsBatched`|9.1| | | | | | | | | |
|`cusolverDnDsyevd`|8.0| | | | | | | | | |
|`cusolverDnDsyevd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDsyevdx`|10.1| | | | | | | | | |
|`cusolverDnDsyevdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDsyevj`|9.0| | | | | | | | | |
|`cusolverDnDsyevjBatched`|9.0| | | | | | | | | |
|`cusolverDnDsyevjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnDsyevj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnDsygvd`|8.0| | | | | | | | | |
|`cusolverDnDsygvd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDsygvdx`|10.1| | | | | | | | | |
|`cusolverDnDsygvdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDsygvj`|9.0| | | | | | | | | |
|`cusolverDnDsygvj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnDsytrd`| | | | | | | | | | |
|`cusolverDnDsytrd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnDsytrf`| | | | | | | | | | |
|`cusolverDnDsytrf_bufferSize`| | | | | | | | | | |
|`cusolverDnDsytri`|10.1| | | | | | | | | |
|`cusolverDnDsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnGeqrf`|11.0|11.1| | | | | | | | |
|`cusolverDnGeqrf_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnGesvd`|11.0|11.1| | | | | | | | |
|`cusolverDnGesvd_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnGetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnGetStream`| | | | |`rocblas_get_stream`| | | | | |
|`cusolverDnGetrf`|11.0|11.1| | | | | | | | |
|`cusolverDnGetrf_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnGetrs`|11.0|11.1| | | | | | | | |
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
|`cusolverDnLoggerForceDisable`|11.7| | | | | | | | | |
|`cusolverDnLoggerOpenFile`|11.7| | | | | | | | | |
|`cusolverDnLoggerSetCallback`|11.7| | | | | | | | | |
|`cusolverDnLoggerSetFile`|11.7| | | | | | | | | |
|`cusolverDnLoggerSetLevel`|11.7| | | | | | | | | |
|`cusolverDnLoggerSetMask`|11.7| | | | | | | | | |
|`cusolverDnPotrf`|11.0|11.1| | | | | | | | |
|`cusolverDnPotrf_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnPotrs`|11.0|11.1| | | | | | | | |
|`cusolverDnSBgels`|11.0| | | | | | | | | |
|`cusolverDnSBgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSBgesv`|11.0| | | | | | | | | |
|`cusolverDnSBgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSHgels`|11.0| | | | | | | | | |
|`cusolverDnSHgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSHgesv`|10.2| | | | | | | | | |
|`cusolverDnSHgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnSSgels`|11.0| | | | | | | | | |
|`cusolverDnSSgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSSgesv`|10.2| | | | | | | | | |
|`cusolverDnSSgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnSXgels`|11.0| | | | | | | | | |
|`cusolverDnSXgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSXgesv`|11.0| | | | | | | | | |
|`cusolverDnSXgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | |
|`cusolverDnSetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnSetStream`| | | | |`rocblas_set_stream`| | | | | |
|`cusolverDnSgebrd`| | | | | | | | | | |
|`cusolverDnSgebrd_bufferSize`| | | | | | | | | | |
|`cusolverDnSgeqrf`| | | | | | | | | | |
|`cusolverDnSgeqrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSgesvd`| | | | | | | | | | |
|`cusolverDnSgesvd_bufferSize`| | | | | | | | | | |
|`cusolverDnSgesvdaStridedBatched`|10.1| | | | | | | | | |
|`cusolverDnSgesvdaStridedBatched_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSgesvdj`|9.0| | | | | | | | | |
|`cusolverDnSgesvdjBatched`|9.0| | | | | | | | | |
|`cusolverDnSgesvdjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnSgesvdj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnSgetrf`| | | | | | | | | | |
|`cusolverDnSgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSgetrs`| | | | | | | | | | |
|`cusolverDnSlaswp`| | | | | | | | | | |
|`cusolverDnSlauum`|10.1| | | | | | | | | |
|`cusolverDnSlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSorgbr`|8.0| | | | | | | | | |
|`cusolverDnSorgbr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSorgqr`|8.0| | | | | | | | | |
|`cusolverDnSorgqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSorgtr`|8.0| | | | | | | | | |
|`cusolverDnSorgtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSormqr`| | | | | | | | | | |
|`cusolverDnSormqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSormtr`|8.0| | | | | | | | | |
|`cusolverDnSormtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSpotrf`| | | | |`rocsolver_spotrf`|3.2.0| | | | |
|`cusolverDnSpotrfBatched`|9.1| | | | | | | | | |
|`cusolverDnSpotrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSpotri`|10.1| | | | | | | | | |
|`cusolverDnSpotri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSpotrs`| | | | | | | | | | |
|`cusolverDnSpotrsBatched`|9.1| | | | | | | | | |
|`cusolverDnSsyevd`|8.0| | | | | | | | | |
|`cusolverDnSsyevd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSsyevdx`|10.1| | | | | | | | | |
|`cusolverDnSsyevdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSsyevj`|9.0| | | | | | | | | |
|`cusolverDnSsyevjBatched`|9.0| | | | | | | | | |
|`cusolverDnSsyevjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnSsyevj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnSsygvd`|8.0| | | | | | | | | |
|`cusolverDnSsygvd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSsygvdx`|10.1| | | | | | | | | |
|`cusolverDnSsygvdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSsygvj`|9.0| | | | | | | | | |
|`cusolverDnSsygvj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnSsytrd`| | | | | | | | | | |
|`cusolverDnSsytrd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnSsytrf`| | | | | | | | | | |
|`cusolverDnSsytrf_bufferSize`| | | | | | | | | | |
|`cusolverDnSsytri`|10.1| | | | | | | | | |
|`cusolverDnSsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSyevd`|11.0|11.1| | | | | | | | |
|`cusolverDnSyevd_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnSyevdx`|11.0|11.1| | | | | | | | |
|`cusolverDnSyevdx_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnXgeqrf`|11.1| | | | | | | | | |
|`cusolverDnXgeqrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgesvd`|11.1| | | | | | | | | |
|`cusolverDnXgesvd_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgesvdjGetResidual`|9.0| | | | | | | | | |
|`cusolverDnXgesvdjGetSweeps`|9.0| | | | | | | | | |
|`cusolverDnXgesvdjSetMaxSweeps`|9.0| | | | | | | | | |
|`cusolverDnXgesvdjSetSortEig`|9.0| | | | | | | | | |
|`cusolverDnXgesvdjSetTolerance`|9.0| | | | | | | | | |
|`cusolverDnXgesvdp`|11.1| | | | | | | | | |
|`cusolverDnXgesvdp_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgesvdr`|11.2| | | | | | | | | |
|`cusolverDnXgesvdr_bufferSize`|11.2| | | | | | | | | |
|`cusolverDnXgetrf`|11.1| | | | | | | | | |
|`cusolverDnXgetrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXgetrs`|11.1| | | | | | | | | |
|`cusolverDnXpotrf`|11.1| | | | | | | | | |
|`cusolverDnXpotrf_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXpotrs`|11.1| | | | | | | | | |
|`cusolverDnXsyevd`|11.1| | | | | | | | | |
|`cusolverDnXsyevd_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXsyevdx`|11.1| | | | | | | | | |
|`cusolverDnXsyevdx_bufferSize`|11.1| | | | | | | | | |
|`cusolverDnXsyevjGetResidual`|9.0| | | | | | | | | |
|`cusolverDnXsyevjGetSweeps`|9.0| | | | | | | | | |
|`cusolverDnXsyevjSetMaxSweeps`|9.0| | | | | | | | | |
|`cusolverDnXsyevjSetSortEig`|9.0| | | | | | | | | |
|`cusolverDnXsyevjSetTolerance`|9.0| | | | | | | | | |
|`cusolverDnXsytrs`|11.3| | | | | | | | | |
|`cusolverDnXsytrs_bufferSize`|11.3| | | | | | | | | |
|`cusolverDnXtrtri`|11.4| | | | | | | | | |
|`cusolverDnXtrtri_bufferSize`|11.4| | | | | | | | | |
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
|`cusolverDnZZgels`|11.0| | | | | | | | | |
|`cusolverDnZZgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnZZgesv`|10.2| | | | | | | | | |
|`cusolverDnZZgesv_bufferSize`|10.2| | | | | | | | | |
|`cusolverDnZgebrd`| | | | | | | | | | |
|`cusolverDnZgebrd_bufferSize`| | | | | | | | | | |
|`cusolverDnZgeqrf`| | | | | | | | | | |
|`cusolverDnZgeqrf_bufferSize`| | | | | | | | | | |
|`cusolverDnZgesvd`| | | | | | | | | | |
|`cusolverDnZgesvd_bufferSize`| | | | | | | | | | |
|`cusolverDnZgesvdaStridedBatched`|10.1| | | | | | | | | |
|`cusolverDnZgesvdaStridedBatched_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZgesvdj`|9.0| | | | | | | | | |
|`cusolverDnZgesvdjBatched`|9.0| | | | | | | | | |
|`cusolverDnZgesvdjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnZgesvdj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnZgetrf`| | | | | | | | | | |
|`cusolverDnZgetrf_bufferSize`| | | | | | | | | | |
|`cusolverDnZgetrs`| | | | | | | | | | |
|`cusolverDnZheevd`|8.0| | | | | | | | | |
|`cusolverDnZheevd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZheevdx`|10.1| | | | | | | | | |
|`cusolverDnZheevdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZheevj`|9.0| | | | | | | | | |
|`cusolverDnZheevjBatched`|9.0| | | | | | | | | |
|`cusolverDnZheevjBatched_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnZheevj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnZhegvd`|8.0| | | | | | | | | |
|`cusolverDnZhegvd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZhegvdx`|10.1| | | | | | | | | |
|`cusolverDnZhegvdx_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZhegvj`|9.0| | | | | | | | | |
|`cusolverDnZhegvj_bufferSize`|9.0| | | | | | | | | |
|`cusolverDnZhetrd`|8.0| | | | | | | | | |
|`cusolverDnZhetrd_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZlaswp`| | | | | | | | | | |
|`cusolverDnZlauum`|10.1| | | | | | | | | |
|`cusolverDnZlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZpotrf`| | | | |`rocsolver_zpotrf`|3.6.0| | | | |
|`cusolverDnZpotrfBatched`|9.1| | | | | | | | | |
|`cusolverDnZpotrf_bufferSize`| | | | | | | | | | |
|`cusolverDnZpotri`|10.1| | | | | | | | | |
|`cusolverDnZpotri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZpotrs`| | | | | | | | | | |
|`cusolverDnZpotrsBatched`|9.1| | | | | | | | | |
|`cusolverDnZsytrf`| | | | | | | | | | |
|`cusolverDnZsytrf_bufferSize`| | | | | | | | | | |
|`cusolverDnZsytri`|10.1| | | | | | | | | |
|`cusolverDnZsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZungbr`|8.0| | | | | | | | | |
|`cusolverDnZungbr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZungqr`|8.0| | | | | | | | | |
|`cusolverDnZungqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZungtr`|8.0| | | | | | | | | |
|`cusolverDnZungtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZunmqr`| | | | | | | | | | |
|`cusolverDnZunmqr_bufferSize`|8.0| | | | | | | | | |
|`cusolverDnZunmtr`|8.0| | | | | | | | | |
|`cusolverDnZunmtr_bufferSize`|8.0| | | | | | | | | |
|`cusolverMgCreate`|10.1| | | | | | | | | |
|`cusolverMgCreateDeviceGrid`|10.1| | | | | | | | | |
|`cusolverMgCreateMatrixDesc`|10.1| | | | | | | | | |
|`cusolverMgDestroy`|10.1| | | | | | | | | |
|`cusolverMgDestroyGrid`|10.1| | | | | | | | | |
|`cusolverMgDeviceSelect`|10.1| | | | | | | | | |
|`cusolverMgGetrf`|10.2| | | | | | | | | |
|`cusolverMgGetrf_bufferSize`|10.2| | | | | | | | | |
|`cusolverMgGetrs`|10.2| | | | | | | | | |
|`cusolverMgGetrs_bufferSize`|10.2| | | | | | | | | |
|`cusolverMgPotrf`|11.0| | | | | | | | | |
|`cusolverMgPotrf_bufferSize`|11.0| | | | | | | | | |
|`cusolverMgPotri`|11.0| | | | | | | | | |
|`cusolverMgPotri_bufferSize`|11.0| | | | | | | | | |
|`cusolverMgPotrs`|11.0| | | | | | | | | |
|`cusolverMgPotrs_bufferSize`|11.0| | | | | | | | | |
|`cusolverMgSyevd`|10.1| | | | | | | | | |
|`cusolverMgSyevd_bufferSize`|10.1| | | | | | | | | |
|`cusolverRfAccessBundledFactorsDevice`| | | | | | | | | | |
|`cusolverRfAnalyze`| | | | | | | | | | |
|`cusolverRfBatchAnalyze`| | | | | | | | | | |
|`cusolverRfBatchRefactor`| | | | | | | | | | |
|`cusolverRfBatchResetValues`| | | | | | | | | | |
|`cusolverRfBatchSetupHost`| | | | | | | | | | |
|`cusolverRfBatchSolve`| | | | | | | | | | |
|`cusolverRfBatchZeroPivot`| | | | | | | | | | |
|`cusolverRfCreate`| | | | | | | | | | |
|`cusolverRfDestroy`| | | | | | | | | | |
|`cusolverRfExtractBundledFactorsHost`| | | | | | | | | | |
|`cusolverRfExtractSplitFactorsHost`| | | | | | | | | | |
|`cusolverRfGetAlgs`| | | | | | | | | | |
|`cusolverRfGetMatrixFormat`| | | | | | | | | | |
|`cusolverRfGetNumericBoostReport`| | | | | | | | | | |
|`cusolverRfGetNumericProperties`| | | | | | | | | | |
|`cusolverRfGetResetValuesFastMode`| | | | | | | | | | |
|`cusolverRfRefactor`| | | | | | | | | | |
|`cusolverRfResetValues`| | | | | | | | | | |
|`cusolverRfSetAlgs`| | | | | | | | | | |
|`cusolverRfSetMatrixFormat`| | | | | | | | | | |
|`cusolverRfSetNumericProperties`| | | | | | | | | | |
|`cusolverRfSetResetValuesFastMode`| | | | | | | | | | |
|`cusolverRfSetupDevice`| | | | | | | | | | |
|`cusolverRfSetupHost`| | | | | | | | | | |
|`cusolverRfSolve`| | | | | | | | | | |
|`cusolverSpCcsrcholBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholDiag`|10.1| | | | | | | | | |
|`cusolverSpCcsrcholFactor`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholFactorHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholSolve`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholSolveHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpCcsrcholZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpCcsreigsHost`| | | | | | | | | | |
|`cusolverSpCcsreigvsi`| | | | | | | | | | |
|`cusolverSpCcsreigvsiHost`| | | | | | | | | | |
|`cusolverSpCcsrlsqvqrHost`| | | | | | | | | | |
|`cusolverSpCcsrlsvchol`| | | | | | | | | | |
|`cusolverSpCcsrlsvcholHost`| | | | | | | | | | |
|`cusolverSpCcsrlsvluHost`| | | | | | | | | | |
|`cusolverSpCcsrlsvqr`| | | | | | | | | | |
|`cusolverSpCcsrlsvqrHost`| | | | | | | | | | |
|`cusolverSpCcsrluBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrluExtractHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrluFactorHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrluSolveHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrluZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrBufferInfoBatched`| | | | | | | | | | |
|`cusolverSpCcsrqrBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrFactor`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrFactorHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrSetup`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrSetupHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrSolve`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrSolveHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpCcsrqrsvBatched`| | | | | | | | | | |
|`cusolverSpCcsrzfdHost`|9.2| | | | | | | | | |
|`cusolverSpCreate`| | | | | | | | | | |
|`cusolverSpCreateCsrcholInfo`|7.5| | | | | | | | | |
|`cusolverSpCreateCsrcholInfoHost`|7.5| | | | | | | | | |
|`cusolverSpCreateCsrluInfoHost`|7.5| | | | | | | | | |
|`cusolverSpCreateCsrqrInfo`| | | | | | | | | | |
|`cusolverSpCreateCsrqrInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholDiag`|10.1| | | | | | | | | |
|`cusolverSpDcsrcholFactor`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholFactorHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholSolve`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholSolveHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpDcsrcholZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpDcsreigsHost`| | | | | | | | | | |
|`cusolverSpDcsreigvsi`| | | | | | | | | | |
|`cusolverSpDcsreigvsiHost`| | | | | | | | | | |
|`cusolverSpDcsrlsqvqrHost`| | | | | | | | | | |
|`cusolverSpDcsrlsvchol`| | | | | | | | | | |
|`cusolverSpDcsrlsvcholHost`| | | | | | | | | | |
|`cusolverSpDcsrlsvluHost`| | | | | | | | | | |
|`cusolverSpDcsrlsvqr`| | | | | | | | | | |
|`cusolverSpDcsrlsvqrHost`| | | | | | | | | | |
|`cusolverSpDcsrluBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrluExtractHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrluFactorHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrluSolveHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrluZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrBufferInfoBatched`| | | | | | | | | | |
|`cusolverSpDcsrqrBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrFactor`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrFactorHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrSetup`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrSetupHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrSolve`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrSolveHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpDcsrqrsvBatched`| | | | | | | | | | |
|`cusolverSpDcsrzfdHost`|9.2| | | | | | | | | |
|`cusolverSpDestroy`| | | | | | | | | | |
|`cusolverSpDestroyCsrcholInfo`|7.5| | | | | | | | | |
|`cusolverSpDestroyCsrcholInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDestroyCsrluInfoHost`|7.5| | | | | | | | | |
|`cusolverSpDestroyCsrqrInfo`| | | | | | | | | | |
|`cusolverSpDestroyCsrqrInfoHost`|7.5| | | | | | | | | |
|`cusolverSpGetStream`| | | | | | | | | | |
|`cusolverSpScsrcholBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpScsrcholBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpScsrcholDiag`|10.1| | | | | | | | | |
|`cusolverSpScsrcholFactor`|7.5| | | | | | | | | |
|`cusolverSpScsrcholFactorHost`|7.5| | | | | | | | | |
|`cusolverSpScsrcholSolve`|7.5| | | | | | | | | |
|`cusolverSpScsrcholSolveHost`|7.5| | | | | | | | | |
|`cusolverSpScsrcholZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpScsrcholZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpScsreigsHost`| | | | | | | | | | |
|`cusolverSpScsreigvsi`| | | | | | | | | | |
|`cusolverSpScsreigvsiHost`| | | | | | | | | | |
|`cusolverSpScsrlsqvqrHost`| | | | | | | | | | |
|`cusolverSpScsrlsvchol`| | | | | | | | | | |
|`cusolverSpScsrlsvcholHost`| | | | | | | | | | |
|`cusolverSpScsrlsvluHost`| | | | | | | | | | |
|`cusolverSpScsrlsvqr`| | | | | | | | | | |
|`cusolverSpScsrlsvqrHost`| | | | | | | | | | |
|`cusolverSpScsrluBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpScsrluExtractHost`|7.5| | | | | | | | | |
|`cusolverSpScsrluFactorHost`|7.5| | | | | | | | | |
|`cusolverSpScsrluSolveHost`|7.5| | | | | | | | | |
|`cusolverSpScsrluZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpScsrqrBufferInfoBatched`| | | | | | | | | | |
|`cusolverSpScsrqrBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrFactor`|7.5| | | | | | | | | |
|`cusolverSpScsrqrFactorHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrSetup`|7.5| | | | | | | | | |
|`cusolverSpScsrqrSetupHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrSolve`|7.5| | | | | | | | | |
|`cusolverSpScsrqrSolveHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpScsrqrZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpScsrqrsvBatched`| | | | | | | | | | |
|`cusolverSpScsrzfdHost`|9.2| | | | | | | | | |
|`cusolverSpSetStream`| | | | | | | | | | |
|`cusolverSpXcsrcholAnalysis`|7.5| | | | | | | | | |
|`cusolverSpXcsrcholAnalysisHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrissymHost`| | | | | | | | | | |
|`cusolverSpXcsrluAnalysisHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrluNnzHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrmetisndHost`|9.2| | | | | | | | | |
|`cusolverSpXcsrpermHost`| | | | | | | | | | |
|`cusolverSpXcsrperm_bufferSizeHost`| | | | | | | | | | |
|`cusolverSpXcsrqrAnalysis`|7.5| | | | | | | | | |
|`cusolverSpXcsrqrAnalysisBatched`| | | | | | | | | | |
|`cusolverSpXcsrqrAnalysisHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrsymamdHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrsymmdqHost`|7.5| | | | | | | | | |
|`cusolverSpXcsrsymrcmHost`| | | | | | | | | | |
|`cusolverSpZcsrcholBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholDiag`|10.1| | | | | | | | | |
|`cusolverSpZcsrcholFactor`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholFactorHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholSolve`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholSolveHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpZcsrcholZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpZcsreigsHost`| | | | | | | | | | |
|`cusolverSpZcsreigvsi`| | | | | | | | | | |
|`cusolverSpZcsreigvsiHost`| | | | | | | | | | |
|`cusolverSpZcsrlsqvqrHost`| | | | | | | | | | |
|`cusolverSpZcsrlsvchol`| | | | | | | | | | |
|`cusolverSpZcsrlsvcholHost`| | | | | | | | | | |
|`cusolverSpZcsrlsvluHost`| | | | | | | | | | |
|`cusolverSpZcsrlsvqr`| | | | | | | | | | |
|`cusolverSpZcsrlsvqrHost`| | | | | | | | | | |
|`cusolverSpZcsrluBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrluExtractHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrluFactorHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrluSolveHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrluZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrBufferInfo`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrBufferInfoBatched`| | | | | | | | | | |
|`cusolverSpZcsrqrBufferInfoHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrFactor`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrFactorHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrSetup`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrSetupHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrSolve`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrSolveHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrZeroPivot`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrZeroPivotHost`|7.5| | | | | | | | | |
|`cusolverSpZcsrqrsvBatched`| | | | | | | | | | |
|`cusolverSpZcsrzfdHost`|9.2| | | | | | | | | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental