# CUSOLVER API supported by HIP

## **1. CUSOLVER Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUBLAS_DIRECT_BACKWARD`|11.0| | | | | | | | | |
|`CUBLAS_DIRECT_FORWARD`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_COLUMNWISE`|11.0| | | | | | | | | |
|`CUBLAS_STOREV_ROWWISE`|11.0| | | | | | | | | |
|`CUDALIBMG_GRID_MAPPING_COL_MAJOR`|10.1| | | | | | | | | |
|`CUDALIBMG_GRID_MAPPING_ROW_MAJOR`|10.1| | | | | | | | | |
|`CUSOLVERDN_GETRF`|11.0| | | | | | | | | |
|`CUSOLVERDN_POTRF`|11.5| | | | | | | | | |
|`CUSOLVERRF_FACTORIZATION_ALG0`| | | | |`HIPSOLVERRF_FACTORIZATION_ALG0`|5.6.0| | | | |
|`CUSOLVERRF_FACTORIZATION_ALG1`| | | | |`HIPSOLVERRF_FACTORIZATION_ALG1`|5.6.0| | | | |
|`CUSOLVERRF_FACTORIZATION_ALG2`| | | | |`HIPSOLVERRF_FACTORIZATION_ALG2`|5.6.0| | | | |
|`CUSOLVERRF_MATRIX_FORMAT_CSC`| | | | |`HIPSOLVERRF_MATRIX_FORMAT_CSC`|5.6.0| | | | |
|`CUSOLVERRF_MATRIX_FORMAT_CSR`| | | | |`HIPSOLVERRF_MATRIX_FORMAT_CSR`|5.6.0| | | | |
|`CUSOLVERRF_NUMERIC_BOOST_NOT_USED`| | | | |`HIPSOLVERRF_NUMERIC_BOOST_NOT_USED`|5.6.0| | | | |
|`CUSOLVERRF_NUMERIC_BOOST_USED`| | | | |`HIPSOLVERRF_NUMERIC_BOOST_USED`|5.6.0| | | | |
|`CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF`| | | | |`HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF`|5.6.0| | | | |
|`CUSOLVERRF_RESET_VALUES_FAST_MODE_ON`| | | | |`HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON`|5.6.0| | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG1`| | | | |`HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1`|5.6.0| | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG2`| | | | |`HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2`|5.6.0| | | | |
|`CUSOLVERRF_TRIANGULAR_SOLVE_ALG3`| | | | |`HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3`|5.6.0| | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L`| | | | |`HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L`|5.6.0| | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U`| | | | |`HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U`|5.6.0| | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_STORED_L`| | | | |`HIPSOLVERRF_UNIT_DIAGONAL_STORED_L`|5.6.0| | | | |
|`CUSOLVERRF_UNIT_DIAGONAL_STORED_U`| | | | |`HIPSOLVERRF_UNIT_DIAGONAL_STORED_U`|5.6.0| | | | |
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
|`CUSOLVER_EIG_MODE_NOVECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_NOVECTOR`|4.5.0| | | | |
|`CUSOLVER_EIG_MODE_VECTOR`|8.0| | | |`HIPSOLVER_EIG_MODE_VECTOR`|4.5.0| | | | |
|`CUSOLVER_EIG_RANGE_ALL`|10.1| | | |`HIPSOLVER_EIG_RANGE_ALL`|5.3.0| | | | |
|`CUSOLVER_EIG_RANGE_I`|10.1| | | |`HIPSOLVER_EIG_RANGE_I`|5.3.0| | | | |
|`CUSOLVER_EIG_RANGE_V`|10.1| | | |`HIPSOLVER_EIG_RANGE_V`|5.3.0| | | | |
|`CUSOLVER_EIG_TYPE_1`|8.0| | | |`HIPSOLVER_EIG_TYPE_1`|4.5.0| | | | |
|`CUSOLVER_EIG_TYPE_2`|8.0| | | |`HIPSOLVER_EIG_TYPE_2`|4.5.0| | | | |
|`CUSOLVER_EIG_TYPE_3`|8.0| | | |`HIPSOLVER_EIG_TYPE_3`|4.5.0| | | | |
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
|`CUSOLVER_STATUS_ALLOC_FAILED`| | | | |`HIPSOLVER_STATUS_ALLOC_FAILED`|4.5.0| | | | |
|`CUSOLVER_STATUS_ARCH_MISMATCH`| | | | |`HIPSOLVER_STATUS_ARCH_MISMATCH`|4.5.0| | | | |
|`CUSOLVER_STATUS_EXECUTION_FAILED`| | | | |`HIPSOLVER_STATUS_EXECUTION_FAILED`|4.5.0| | | | |
|`CUSOLVER_STATUS_INTERNAL_ERROR`| | | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | | |
|`CUSOLVER_STATUS_INVALID_LICENSE`| | | | | | | | | | |
|`CUSOLVER_STATUS_INVALID_VALUE`| | | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | | |
|`CUSOLVER_STATUS_INVALID_WORKSPACE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_INTERNAL_ERROR`|10.2| | | |`HIPSOLVER_STATUS_INTERNAL_ERROR`|4.5.0| | | | |
|`CUSOLVER_STATUS_IRS_MATRIX_SINGULAR`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_NOT_SUPPORTED`|10.2| | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | | |
|`CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_OUT_OF_RANGE`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID`|10.2| | | |`HIPSOLVER_STATUS_INVALID_VALUE`|4.5.0| | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE`|11.0| | | | | | | | | |
|`CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED`|10.2| | | | | | | | | |
|`CUSOLVER_STATUS_MAPPING_ERROR`| | | | |`HIPSOLVER_STATUS_MAPPING_ERROR`|4.5.0| | | | |
|`CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | |`HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED`|6.1.0| | | |6.1.0|
|`CUSOLVER_STATUS_NOT_INITIALIZED`| | | | |`HIPSOLVER_STATUS_NOT_INITIALIZED`|4.5.0| | | | |
|`CUSOLVER_STATUS_NOT_SUPPORTED`| | | | |`HIPSOLVER_STATUS_NOT_SUPPORTED`|4.5.0| | | | |
|`CUSOLVER_STATUS_SUCCESS`| | | | |`HIPSOLVER_STATUS_SUCCESS`|4.5.0| | | | |
|`CUSOLVER_STATUS_ZERO_PIVOT`| | | | |`HIPSOLVER_STATUS_ZERO_PIVOT`|5.6.0| | | | |
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
|`cusolverDnHandle_t`| | | | |`hipsolverHandle_t`|4.5.0| | | | |
|`cusolverDnIRSInfos`|10.2| | | | | | | | | |
|`cusolverDnIRSInfos_t`|10.2| | | | | | | | | |
|`cusolverDnIRSParams`|10.2| | | | | | | | | |
|`cusolverDnIRSParams_t`|10.2| | | | | | | | | |
|`cusolverDnLoggerCallback_t`|11.7| | | | | | | | | |
|`cusolverDnParams`|11.0| | | | | | | | | |
|`cusolverDnParams_t`|11.0| | | | | | | | | |
|`cusolverEigMode_t`|8.0| | | |`hipsolverEigMode_t`|4.5.0| | | | |
|`cusolverEigRange_t`|10.1| | | |`hipsolverEigRange_t`|5.3.0| | | | |
|`cusolverEigType_t`|8.0| | | |`hipsolverEigType_t`|4.5.0| | | | |
|`cusolverIRSRefinement_t`|10.2| | | | | | | | | |
|`cusolverMgContext`|10.1| | | | | | | | | |
|`cusolverMgGridMapping_t`|10.1| | | | | | | | | |
|`cusolverMgHandle_t`|10.1| | | | | | | | | |
|`cusolverNorm_t`|10.2| | | | | | | | | |
|`cusolverPrecType_t`|11.0| | | | | | | | | |
|`cusolverRfCommon`| | | | | | | | | | |
|`cusolverRfFactorization_t`| | | | |`hipsolverRfFactorization_t`|5.6.0| | | | |
|`cusolverRfHandle_t`| | | | |`hipsolverRfHandle_t`|5.6.0| | | | |
|`cusolverRfMatrixFormat_t`| | | | |`hipsolverRfMatrixFormat_t`|5.6.0| | | | |
|`cusolverRfNumericBoostReport_t`| | | | |`hipsolverRfNumericBoostReport_t`|5.6.0| | | | |
|`cusolverRfResetValuesFastMode_t`| | | | |`hipsolverRfResetValuesFastMode_t`|5.6.0| | | | |
|`cusolverRfTriangularSolve_t`| | | | |`hipsolverRfTriangularSolve_t`|5.6.0| | | | |
|`cusolverRfUnitDiagonal_t`| | | | |`hipsolverRfUnitDiagonal_t`|5.6.0| | | | |
|`cusolverSpContext`| | | | | | | | | | |
|`cusolverSpHandle_t`| | | | |`hipsolverSpHandle_t`|6.1.0| | | |6.1.0|
|`cusolverStatus_t`| | | | |`hipsolverStatus_t`|4.5.0| | | | |
|`cusolverStorevMode_t`|11.0| | | | | | | | | |
|`cusolver_int_t`|10.1| | | |`int`| | | | | |
|`gesvdjInfo`|9.0| | | | | | | | | |
|`gesvdjInfo_t`|9.0| | | |`hipsolverGesvdjInfo_t`|5.1.0| | | | |
|`syevjInfo`|9.0| | | | | | | | | |
|`syevjInfo_t`|9.0| | | |`hipsolverSyevjInfo_t`|5.1.0| | | | |

## **2. CUSOLVER Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusolverDnCCgels`|11.0| | | |`hipsolverDnCCgels`|5.1.0| | | | |
|`cusolverDnCCgels_bufferSize`|11.0| | | |`hipsolverDnCCgels_bufferSize`|5.1.0| | | | |
|`cusolverDnCCgesv`|10.2| | | |`hipsolverDnCCgesv`|5.1.0| | | | |
|`cusolverDnCCgesv_bufferSize`|10.2| | | |`hipsolverDnCCgesv_bufferSize`|5.1.0| | | | |
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
|`cusolverDnCgebrd`| | | | |`hipsolverDnCgebrd`|5.1.0| | | | |
|`cusolverDnCgebrd_bufferSize`| | | | |`hipsolverDnCgebrd_bufferSize`|5.1.0| | | | |
|`cusolverDnCgeqrf`| | | | |`hipsolverDnCgeqrf`|5.1.0| | | | |
|`cusolverDnCgeqrf_bufferSize`| | | | |`hipsolverDnCgeqrf_bufferSize`|5.1.0| | | | |
|`cusolverDnCgesvd`| | | | |`hipsolverDnCgesvd`|5.1.0| | | | |
|`cusolverDnCgesvd_bufferSize`| | | | |`hipsolverDnCgesvd_bufferSize`|5.1.0| | | | |
|`cusolverDnCgesvdaStridedBatched`|10.1| | | |`hipsolverDnCgesvdaStridedBatched`|5.4.0| | | | |
|`cusolverDnCgesvdaStridedBatched_bufferSize`|10.1| | | |`hipsolverDnCgesvdaStridedBatched_bufferSize`|5.4.0| | | | |
|`cusolverDnCgesvdj`|9.0| | | |`hipsolverDnCgesvdj`|5.1.0| | | | |
|`cusolverDnCgesvdjBatched`|9.0| | | |`hipsolverDnCgesvdjBatched`|5.1.0| | | | |
|`cusolverDnCgesvdjBatched_bufferSize`|9.0| | | |`hipsolverDnCgesvdjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnCgesvdj_bufferSize`|9.0| | | |`hipsolverDnCgesvdj_bufferSize`|5.1.0| | | | |
|`cusolverDnCgetrf`| | | | |`hipsolverDnCgetrf`|5.1.0| | | | |
|`cusolverDnCgetrf_bufferSize`| | | | |`hipsolverDnCgetrf_bufferSize`|5.1.0| | | | |
|`cusolverDnCgetrs`| | | | |`hipsolverDnCgetrs`|5.1.0| | | | |
|`cusolverDnCheevd`|8.0| | | |`hipsolverDnCheevd`|5.1.0| | | | |
|`cusolverDnCheevd_bufferSize`|8.0| | | |`hipsolverDnCheevd_bufferSize`|5.1.0| | | | |
|`cusolverDnCheevdx`|10.1| | | |`hipsolverDnCheevdx`|5.3.0| | | | |
|`cusolverDnCheevdx_bufferSize`|10.1| | | |`hipsolverDnCheevdx_bufferSize`|5.3.0| | | | |
|`cusolverDnCheevj`|9.0| | | |`hipsolverDnCheevj`|5.1.0| | | | |
|`cusolverDnCheevjBatched`|9.0| | | |`hipsolverDnCheevjBatched`|5.1.0| | | | |
|`cusolverDnCheevjBatched_bufferSize`|9.0| | | |`hipsolverDnCheevjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnCheevj_bufferSize`|9.0| | | |`hipsolverDnCheevj_bufferSize`|5.1.0| | | | |
|`cusolverDnChegvd`|8.0| | | |`hipsolverDnChegvd`|5.1.0| | | | |
|`cusolverDnChegvd_bufferSize`|8.0| | | |`hipsolverDnChegvd_bufferSize`|5.1.0| | | | |
|`cusolverDnChegvdx`|10.1| | | |`hipsolverDnChegvdx`|5.3.0| | | | |
|`cusolverDnChegvdx_bufferSize`|10.1| | | |`hipsolverDnChegvdx_bufferSize`|5.3.0| | | | |
|`cusolverDnChegvj`|9.0| | | |`hipsolverDnChegvj`|5.1.0| | | | |
|`cusolverDnChegvj_bufferSize`|9.0| | | |`hipsolverDnChegvj_bufferSize`|5.1.0| | | | |
|`cusolverDnChetrd`|8.0| | | |`hipsolverDnChetrd`|5.1.0| | | | |
|`cusolverDnChetrd_bufferSize`|8.0| | | |`hipsolverDnChetrd_bufferSize`|5.1.0| | | | |
|`cusolverDnClaswp`| | | | | | | | | | |
|`cusolverDnClauum`|10.1| | | | | | | | | |
|`cusolverDnClauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCpotrf`| | | | |`hipsolverDnCpotrf`|5.1.0| | | | |
|`cusolverDnCpotrfBatched`|9.1| | | |`hipsolverDnCpotrfBatched`|5.1.0| | | | |
|`cusolverDnCpotrf_bufferSize`| | | | |`hipsolverDnCpotrf_bufferSize`|5.1.0| | | | |
|`cusolverDnCpotri`|10.1| | | |`hipsolverDnCpotri`|5.1.0| | | | |
|`cusolverDnCpotri_bufferSize`|10.1| | | |`hipsolverDnCpotri_bufferSize`|5.1.0| | | | |
|`cusolverDnCpotrs`| | | | |`hipsolverDnCpotrs`|5.1.0| | | | |
|`cusolverDnCpotrsBatched`|9.1| | | |`hipsolverDnCpotrsBatched`|5.1.0| | | | |
|`cusolverDnCreate`| | | | |`hipsolverDnCreate`|5.1.0| | | | |
|`cusolverDnCreateGesvdjInfo`|9.0| | | |`hipsolverDnCreateGesvdjInfo`|5.1.0| | | | |
|`cusolverDnCreateParams`|11.0| | | | | | | | | |
|`cusolverDnCreateSyevjInfo`|9.0| | | |`hipsolverDnCreateSyevjInfo`|5.1.0| | | | |
|`cusolverDnCsytrf`| | | | |`hipsolverDnCsytrf`|5.1.0| | | | |
|`cusolverDnCsytrf_bufferSize`| | | | |`hipsolverDnCsytrf_bufferSize`|5.1.0| | | | |
|`cusolverDnCsytri`|10.1| | | | | | | | | |
|`cusolverDnCsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnCungbr`|8.0| | | |`hipsolverDnCungbr`|5.1.0| | | | |
|`cusolverDnCungbr_bufferSize`|8.0| | | |`hipsolverDnCungbr_bufferSize`|5.1.0| | | | |
|`cusolverDnCungqr`|8.0| | | |`hipsolverDnCungqr`|5.1.0| | | | |
|`cusolverDnCungqr_bufferSize`|8.0| | | |`hipsolverDnCungqr_bufferSize`|5.1.0| | | | |
|`cusolverDnCungtr`|8.0| | | |`hipsolverDnCungtr`|5.1.0| | | | |
|`cusolverDnCungtr_bufferSize`|8.0| | | |`hipsolverDnCungtr_bufferSize`|5.1.0| | | | |
|`cusolverDnCunmqr`| | | | |`hipsolverDnCunmqr`|5.1.0| | | | |
|`cusolverDnCunmqr_bufferSize`|8.0| | | |`hipsolverDnCunmqr_bufferSize`|5.1.0| | | | |
|`cusolverDnCunmtr`|8.0| | | |`hipsolverDnCunmtr`|5.1.0| | | | |
|`cusolverDnCunmtr_bufferSize`|8.0| | | |`hipsolverDnCunmtr_bufferSize`|5.1.0| | | | |
|`cusolverDnDBgels`|11.0| | | | | | | | | |
|`cusolverDnDBgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDBgesv`|11.0| | | | | | | | | |
|`cusolverDnDBgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnDDgels`|11.0| | | |`hipsolverDnDDgels`|5.1.0| | | | |
|`cusolverDnDDgels_bufferSize`|11.0| | | |`hipsolverDnDDgels_bufferSize`|5.1.0| | | | |
|`cusolverDnDDgesv`|10.2| | | |`hipsolverDnDDgesv`|5.1.0| | | | |
|`cusolverDnDDgesv_bufferSize`|10.2| | | |`hipsolverDnDDgesv_bufferSize`|5.1.0| | | | |
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
|`cusolverDnDestroy`| | | | |`hipsolverDnDestroy`|5.1.0| | | | |
|`cusolverDnDestroyGesvdjInfo`|9.0| | | |`hipsolverDnDestroyGesvdjInfo`|5.1.0| | | | |
|`cusolverDnDestroyParams`|11.0| | | | | | | | | |
|`cusolverDnDestroySyevjInfo`|9.0| | | |`hipsolverDnDestroySyevjInfo`|5.1.0| | | | |
|`cusolverDnDgebrd`| | | | |`hipsolverDnDgebrd`|5.1.0| | | | |
|`cusolverDnDgebrd_bufferSize`| | | | |`hipsolverDnDgebrd_bufferSize`|5.1.0| | | | |
|`cusolverDnDgeqrf`| | | | |`hipsolverDnDgeqrf`|5.1.0| | | | |
|`cusolverDnDgeqrf_bufferSize`| | | | |`hipsolverDnDgeqrf_bufferSize`|5.1.0| | | | |
|`cusolverDnDgesvd`| | | | |`hipsolverDnDgesvd`|5.1.0| | | | |
|`cusolverDnDgesvd_bufferSize`| | | | |`hipsolverDnDgesvd_bufferSize`|5.1.0| | | | |
|`cusolverDnDgesvdaStridedBatched`|10.1| | | |`hipsolverDnDgesvdaStridedBatched`|5.4.0| | | | |
|`cusolverDnDgesvdaStridedBatched_bufferSize`|10.1| | | |`hipsolverDnDgesvdaStridedBatched_bufferSize`|5.4.0| | | | |
|`cusolverDnDgesvdj`|9.0| | | |`hipsolverDnDgesvdj`|5.1.0| | | | |
|`cusolverDnDgesvdjBatched`|9.0| | | |`hipsolverDnDgesvdjBatched`|5.1.0| | | | |
|`cusolverDnDgesvdjBatched_bufferSize`|9.0| | | |`hipsolverDnDgesvdjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnDgesvdj_bufferSize`|9.0| | | |`hipsolverDnDgesvdj_bufferSize`|5.1.0| | | | |
|`cusolverDnDgetrf`| | | | |`hipsolverDnDgetrf`|5.1.0| | | | |
|`cusolverDnDgetrf_bufferSize`| | | | |`hipsolverDnDgetrf_bufferSize`|5.1.0| | | | |
|`cusolverDnDgetrs`| | | | |`hipsolverDnDgetrs`|5.1.0| | | | |
|`cusolverDnDlaswp`| | | | | | | | | | |
|`cusolverDnDlauum`|10.1| | | | | | | | | |
|`cusolverDnDlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnDorgbr`|8.0| | | |`hipsolverDnDorgbr`|5.1.0| | | | |
|`cusolverDnDorgbr_bufferSize`|8.0| | | |`hipsolverDnDorgbr_bufferSize`|5.1.0| | | | |
|`cusolverDnDorgqr`|8.0| | | |`hipsolverDnDorgqr`|5.1.0| | | | |
|`cusolverDnDorgqr_bufferSize`|8.0| | | |`hipsolverDnDorgqr_bufferSize`|5.1.0| | | | |
|`cusolverDnDorgtr`|8.0| | | |`hipsolverDnDorgtr`|5.1.0| | | | |
|`cusolverDnDorgtr_bufferSize`|8.0| | | |`hipsolverDnDorgtr_bufferSize`|5.1.0| | | | |
|`cusolverDnDormqr`| | | | |`hipsolverDnDormqr`|5.1.0| | | | |
|`cusolverDnDormqr_bufferSize`|8.0| | | |`hipsolverDnDormqr_bufferSize`|5.1.0| | | | |
|`cusolverDnDormtr`|8.0| | | |`hipsolverDnDormtr`|5.1.0| | | | |
|`cusolverDnDormtr_bufferSize`|8.0| | | |`hipsolverDnDormtr_bufferSize`|5.1.0| | | | |
|`cusolverDnDpotrf`| | | | |`hipsolverDnDpotrf`|5.1.0| | | | |
|`cusolverDnDpotrfBatched`|9.1| | | |`hipsolverDnDpotrfBatched`|5.1.0| | | | |
|`cusolverDnDpotrf_bufferSize`| | | | |`hipsolverDnDpotrf_bufferSize`|5.1.0| | | | |
|`cusolverDnDpotri`|10.1| | | |`hipsolverDnDpotri`|5.1.0| | | | |
|`cusolverDnDpotri_bufferSize`|10.1| | | |`hipsolverDnDpotri_bufferSize`|5.1.0| | | | |
|`cusolverDnDpotrs`| | | | |`hipsolverDnDpotrs`|5.1.0| | | | |
|`cusolverDnDpotrsBatched`|9.1| | | |`hipsolverDnDpotrsBatched`|5.1.0| | | | |
|`cusolverDnDsyevd`|8.0| | | |`hipsolverDnDsyevd`|5.1.0| | | | |
|`cusolverDnDsyevd_bufferSize`|8.0| | | |`hipsolverDnDsyevd_bufferSize`|5.1.0| | | | |
|`cusolverDnDsyevdx`|10.1| | | |`hipsolverDnDsyevdx`|5.3.0| | | | |
|`cusolverDnDsyevdx_bufferSize`|10.1| | | |`hipsolverDnDsyevdx_bufferSize`|5.3.0| | | | |
|`cusolverDnDsyevj`|9.0| | | |`hipsolverDnDsyevj`|5.1.0| | | | |
|`cusolverDnDsyevjBatched`|9.0| | | |`hipsolverDnDsyevjBatched`|5.1.0| | | | |
|`cusolverDnDsyevjBatched_bufferSize`|9.0| | | |`hipsolverDnDsyevjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnDsyevj_bufferSize`|9.0| | | |`hipsolverDnDsyevj_bufferSize`|5.1.0| | | | |
|`cusolverDnDsygvd`|8.0| | | |`hipsolverDnDsygvd`|5.1.0| | | | |
|`cusolverDnDsygvd_bufferSize`|8.0| | | |`hipsolverDnDsygvd_bufferSize`|5.1.0| | | | |
|`cusolverDnDsygvdx`|10.1| | | |`hipsolverDnDsygvdx`|5.3.0| | | | |
|`cusolverDnDsygvdx_bufferSize`|10.1| | | |`hipsolverDnDsygvdx_bufferSize`|5.3.0| | | | |
|`cusolverDnDsygvj`|9.0| | | |`hipsolverDnDsygvj`|5.1.0| | | | |
|`cusolverDnDsygvj_bufferSize`|9.0| | | |`hipsolverDnDsygvj_bufferSize`|5.1.0| | | | |
|`cusolverDnDsytrd`| | | | |`hipsolverDnDsytrd`|5.1.0| | | | |
|`cusolverDnDsytrd_bufferSize`|8.0| | | |`hipsolverDnDsytrd_bufferSize`|5.1.0| | | | |
|`cusolverDnDsytrf`| | | | |`hipsolverDnDsytrf`|5.1.0| | | | |
|`cusolverDnDsytrf_bufferSize`| | | | |`hipsolverDnDsytrf_bufferSize`|5.1.0| | | | |
|`cusolverDnDsytri`|10.1| | | | | | | | | |
|`cusolverDnDsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnGeqrf`|11.0|11.1| | | | | | | | |
|`cusolverDnGeqrf_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnGesvd`|11.0|11.1| | | | | | | | |
|`cusolverDnGesvd_bufferSize`|11.0|11.1| | | | | | | | |
|`cusolverDnGetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnGetStream`| | | | |`hipsolverGetStream`|4.5.0| | | | |
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
|`cusolverDnSSgels`|11.0| | | |`hipsolverDnSSgels`|5.1.0| | | | |
|`cusolverDnSSgels_bufferSize`|11.0| | | |`hipsolverDnSSgels_bufferSize`|5.1.0| | | | |
|`cusolverDnSSgesv`|10.2| | | |`hipsolverDnSSgesv`|5.1.0| | | | |
|`cusolverDnSSgesv_bufferSize`|10.2| | | |`hipsolverDnSSgesv_bufferSize`|5.1.0| | | | |
|`cusolverDnSXgels`|11.0| | | | | | | | | |
|`cusolverDnSXgels_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSXgesv`|11.0| | | | | | | | | |
|`cusolverDnSXgesv_bufferSize`|11.0| | | | | | | | | |
|`cusolverDnSetAdvOptions`|11.0| | | | | | | | | |
|`cusolverDnSetDeterministicMode`|12.2| | | | | | | | | |
|`cusolverDnSetStream`| | | | |`hipsolverSetStream`|4.5.0| | | | |
|`cusolverDnSgebrd`| | | | |`hipsolverDnSgebrd`|5.1.0| | | | |
|`cusolverDnSgebrd_bufferSize`| | | | |`hipsolverDnSgebrd_bufferSize`|5.1.0| | | | |
|`cusolverDnSgeqrf`| | | | |`hipsolverDnSgeqrf`|5.1.0| | | | |
|`cusolverDnSgeqrf_bufferSize`| | | | |`hipsolverDnSgeqrf_bufferSize`|5.1.0| | | | |
|`cusolverDnSgesvd`| | | | |`hipsolverDnSgesvd`|5.1.0| | | | |
|`cusolverDnSgesvd_bufferSize`| | | | |`hipsolverDnSgesvd_bufferSize`|5.1.0| | | | |
|`cusolverDnSgesvdaStridedBatched`|10.1| | | |`hipsolverDnSgesvdaStridedBatched`|5.4.0| | | | |
|`cusolverDnSgesvdaStridedBatched_bufferSize`|10.1| | | |`hipsolverDnSgesvdaStridedBatched_bufferSize`|5.4.0| | | | |
|`cusolverDnSgesvdj`|9.0| | | |`hipsolverDnSgesvdj`|5.1.0| | | | |
|`cusolverDnSgesvdjBatched`|9.0| | | |`hipsolverDnSgesvdjBatched`|5.1.0| | | | |
|`cusolverDnSgesvdjBatched_bufferSize`|9.0| | | |`hipsolverDnSgesvdjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnSgesvdj_bufferSize`|9.0| | | |`hipsolverDnSgesvdj_bufferSize`|5.1.0| | | | |
|`cusolverDnSgetrf`| | | | |`hipsolverDnSgetrf`|5.1.0| | | | |
|`cusolverDnSgetrf_bufferSize`| | | | |`hipsolverDnSgetrf_bufferSize`|5.1.0| | | | |
|`cusolverDnSgetrs`| | | | |`hipsolverDnSgetrs`|5.1.0| | | | |
|`cusolverDnSlaswp`| | | | | | | | | | |
|`cusolverDnSlauum`|10.1| | | | | | | | | |
|`cusolverDnSlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnSorgbr`|8.0| | | |`hipsolverDnSorgbr`|5.1.0| | | | |
|`cusolverDnSorgbr_bufferSize`|8.0| | | |`hipsolverDnSorgbr_bufferSize`|5.1.0| | | | |
|`cusolverDnSorgqr`|8.0| | | |`hipsolverDnSorgqr`|5.1.0| | | | |
|`cusolverDnSorgqr_bufferSize`|8.0| | | |`hipsolverDnSorgqr_bufferSize`|5.1.0| | | | |
|`cusolverDnSorgtr`|8.0| | | |`hipsolverDnSorgtr`|5.1.0| | | | |
|`cusolverDnSorgtr_bufferSize`|8.0| | | |`hipsolverDnSorgtr_bufferSize`|5.1.0| | | | |
|`cusolverDnSormqr`| | | | |`hipsolverDnSormqr`|5.1.0| | | | |
|`cusolverDnSormqr_bufferSize`|8.0| | | |`hipsolverDnSormqr_bufferSize`|5.1.0| | | | |
|`cusolverDnSormtr`|8.0| | | |`hipsolverDnSormtr`|5.1.0| | | | |
|`cusolverDnSormtr_bufferSize`|8.0| | | |`hipsolverDnSormtr_bufferSize`|5.1.0| | | | |
|`cusolverDnSpotrf`| | | | |`hipsolverDnSpotrf`|5.1.0| | | | |
|`cusolverDnSpotrfBatched`|9.1| | | |`hipsolverDnSpotrfBatched`|5.1.0| | | | |
|`cusolverDnSpotrf_bufferSize`| | | | |`hipsolverDnSpotrf_bufferSize`|5.1.0| | | | |
|`cusolverDnSpotri`|10.1| | | |`hipsolverDnSpotri`|5.1.0| | | | |
|`cusolverDnSpotri_bufferSize`|10.1| | | |`hipsolverDnSpotri_bufferSize`|5.1.0| | | | |
|`cusolverDnSpotrs`| | | | |`hipsolverDnSpotrs`|5.1.0| | | | |
|`cusolverDnSpotrsBatched`|9.1| | | |`hipsolverDnSpotrsBatched`|5.1.0| | | | |
|`cusolverDnSsyevd`|8.0| | | |`hipsolverDnSsyevd`|5.1.0| | | | |
|`cusolverDnSsyevd_bufferSize`|8.0| | | |`hipsolverDnSsyevd_bufferSize`|5.1.0| | | | |
|`cusolverDnSsyevdx`|10.1| | | |`hipsolverDnSsyevdx`|5.3.0| | | | |
|`cusolverDnSsyevdx_bufferSize`|10.1| | | |`hipsolverDnSsyevdx_bufferSize`|5.3.0| | | | |
|`cusolverDnSsyevj`|9.0| | | |`hipsolverDnSsyevj`|5.1.0| | | | |
|`cusolverDnSsyevjBatched`|9.0| | | |`hipsolverDnSsyevjBatched`|5.1.0| | | | |
|`cusolverDnSsyevjBatched_bufferSize`|9.0| | | |`hipsolverDnSsyevjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnSsyevj_bufferSize`|9.0| | | |`hipsolverDnSsyevj_bufferSize`|5.1.0| | | | |
|`cusolverDnSsygvd`|8.0| | | |`hipsolverDnSsygvd`|5.1.0| | | | |
|`cusolverDnSsygvd_bufferSize`|8.0| | | |`hipsolverDnSsygvd_bufferSize`|5.1.0| | | | |
|`cusolverDnSsygvdx`|10.1| | | |`hipsolverDnSsygvdx`|5.3.0| | | | |
|`cusolverDnSsygvdx_bufferSize`|10.1| | | |`hipsolverDnSsygvdx_bufferSize`|5.3.0| | | | |
|`cusolverDnSsygvj`|9.0| | | |`hipsolverDnSsygvj`|5.1.0| | | | |
|`cusolverDnSsygvj_bufferSize`|9.0| | | |`hipsolverDnSsygvj_bufferSize`|5.1.0| | | | |
|`cusolverDnSsytrd`| | | | |`hipsolverDnSsytrd`|5.1.0| | | | |
|`cusolverDnSsytrd_bufferSize`|8.0| | | |`hipsolverDnSsytrd_bufferSize`|5.1.0| | | | |
|`cusolverDnSsytrf`| | | | |`hipsolverDnSsytrf`|5.1.0| | | | |
|`cusolverDnSsytrf_bufferSize`| | | | |`hipsolverDnSsytrf_bufferSize`|5.1.0| | | | |
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
|`cusolverDnXgesvdjGetResidual`|9.0| | | |`hipsolverDnXgesvdjGetResidual`|5.1.0| | | | |
|`cusolverDnXgesvdjGetSweeps`|9.0| | | |`hipsolverDnXgesvdjGetSweeps`|5.1.0| | | | |
|`cusolverDnXgesvdjSetMaxSweeps`|9.0| | | |`hipsolverDnXgesvdjSetMaxSweeps`|5.1.0| | | | |
|`cusolverDnXgesvdjSetSortEig`|9.0| | | |`hipsolverDnXgesvdjSetSortEig`|5.1.0| | | | |
|`cusolverDnXgesvdjSetTolerance`|9.0| | | |`hipsolverDnXgesvdjSetTolerance`|5.1.0| | | | |
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
|`cusolverDnXsyevjGetResidual`|9.0| | | |`hipsolverDnXsyevjGetResidual`|5.1.0| | | | |
|`cusolverDnXsyevjGetSweeps`|9.0| | | |`hipsolverDnXsyevjGetSweeps`|5.1.0| | | | |
|`cusolverDnXsyevjSetMaxSweeps`|9.0| | | |`hipsolverDnXsyevjSetMaxSweeps`|5.1.0| | | | |
|`cusolverDnXsyevjSetSortEig`|9.0| | | |`hipsolverDnXsyevjSetSortEig`|5.1.0| | | | |
|`cusolverDnXsyevjSetTolerance`|9.0| | | |`hipsolverDnXsyevjSetTolerance`|5.1.0| | | | |
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
|`cusolverDnZZgels`|11.0| | | |`hipsolverDnZZgels`|5.1.0| | | | |
|`cusolverDnZZgels_bufferSize`|11.0| | | |`hipsolverDnZZgels_bufferSize`|5.1.0| | | | |
|`cusolverDnZZgesv`|10.2| | | |`hipsolverDnZZgesv`|5.1.0| | | | |
|`cusolverDnZZgesv_bufferSize`|10.2| | | |`hipsolverDnZZgesv_bufferSize`|5.1.0| | | | |
|`cusolverDnZgebrd`| | | | |`hipsolverDnZgebrd`|5.1.0| | | | |
|`cusolverDnZgebrd_bufferSize`| | | | |`hipsolverDnZgebrd_bufferSize`|5.1.0| | | | |
|`cusolverDnZgeqrf`| | | | |`hipsolverDnZgeqrf`|5.1.0| | | | |
|`cusolverDnZgeqrf_bufferSize`| | | | |`hipsolverDnZgeqrf_bufferSize`|5.1.0| | | | |
|`cusolverDnZgesvd`| | | | |`hipsolverDnZgesvd`|5.1.0| | | | |
|`cusolverDnZgesvd_bufferSize`| | | | |`hipsolverDnZgesvd_bufferSize`|5.1.0| | | | |
|`cusolverDnZgesvdaStridedBatched`|10.1| | | |`hipsolverDnZgesvdaStridedBatched`|5.4.0| | | | |
|`cusolverDnZgesvdaStridedBatched_bufferSize`|10.1| | | |`hipsolverDnZgesvdaStridedBatched_bufferSize`|5.4.0| | | | |
|`cusolverDnZgesvdj`|9.0| | | |`hipsolverDnZgesvdj`|5.1.0| | | | |
|`cusolverDnZgesvdjBatched`|9.0| | | |`hipsolverDnZgesvdjBatched`|5.1.0| | | | |
|`cusolverDnZgesvdjBatched_bufferSize`|9.0| | | |`hipsolverDnZgesvdjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnZgesvdj_bufferSize`|9.0| | | |`hipsolverDnZgesvdj_bufferSize`|5.1.0| | | | |
|`cusolverDnZgetrf`| | | | |`hipsolverDnZgetrf`|5.1.0| | | | |
|`cusolverDnZgetrf_bufferSize`| | | | |`hipsolverDnZgetrf_bufferSize`|5.1.0| | | | |
|`cusolverDnZgetrs`| | | | |`hipsolverDnZgetrs`|5.1.0| | | | |
|`cusolverDnZheevd`|8.0| | | |`hipsolverDnZheevd`|5.1.0| | | | |
|`cusolverDnZheevd_bufferSize`|8.0| | | |`hipsolverDnZheevd_bufferSize`|5.1.0| | | | |
|`cusolverDnZheevdx`|10.1| | | |`hipsolverDnZheevdx`|5.3.0| | | | |
|`cusolverDnZheevdx_bufferSize`|10.1| | | |`hipsolverDnZheevdx_bufferSize`|5.3.0| | | | |
|`cusolverDnZheevj`|9.0| | | |`hipsolverDnZheevj`|5.1.0| | | | |
|`cusolverDnZheevjBatched`|9.0| | | |`hipsolverDnZheevjBatched`|5.1.0| | | | |
|`cusolverDnZheevjBatched_bufferSize`|9.0| | | |`hipsolverDnZheevjBatched_bufferSize`|5.1.0| | | | |
|`cusolverDnZheevj_bufferSize`|9.0| | | |`hipsolverDnZheevj_bufferSize`|5.1.0| | | | |
|`cusolverDnZhegvd`|8.0| | | |`hipsolverDnZhegvd`|5.1.0| | | | |
|`cusolverDnZhegvd_bufferSize`|8.0| | | |`hipsolverDnZhegvd_bufferSize`|5.1.0| | | | |
|`cusolverDnZhegvdx`|10.1| | | |`hipsolverDnZhegvdx`|5.3.0| | | | |
|`cusolverDnZhegvdx_bufferSize`|10.1| | | |`hipsolverDnZhegvdx_bufferSize`|5.3.0| | | | |
|`cusolverDnZhegvj`|9.0| | | |`hipsolverDnZhegvj`|5.1.0| | | | |
|`cusolverDnZhegvj_bufferSize`|9.0| | | |`hipsolverDnZhegvj_bufferSize`|5.1.0| | | | |
|`cusolverDnZhetrd`|8.0| | | |`hipsolverDnZhetrd`|5.1.0| | | | |
|`cusolverDnZhetrd_bufferSize`|8.0| | | |`hipsolverDnZhetrd_bufferSize`|5.1.0| | | | |
|`cusolverDnZlaswp`| | | | | | | | | | |
|`cusolverDnZlauum`|10.1| | | | | | | | | |
|`cusolverDnZlauum_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZpotrf`| | | | |`hipsolverDnZpotrf`|5.1.0| | | | |
|`cusolverDnZpotrfBatched`|9.1| | | |`hipsolverDnZpotrfBatched`|5.1.0| | | | |
|`cusolverDnZpotrf_bufferSize`| | | | |`hipsolverDnZpotrf_bufferSize`|5.1.0| | | | |
|`cusolverDnZpotri`|10.1| | | |`hipsolverDnZpotri`|5.1.0| | | | |
|`cusolverDnZpotri_bufferSize`|10.1| | | |`hipsolverDnZpotri_bufferSize`|5.1.0| | | | |
|`cusolverDnZpotrs`| | | | |`hipsolverDnZpotrs`|5.1.0| | | | |
|`cusolverDnZpotrsBatched`|9.1| | | |`hipsolverDnZpotrsBatched`|5.1.0| | | | |
|`cusolverDnZsytrf`| | | | |`hipsolverDnZsytrf`|5.1.0| | | | |
|`cusolverDnZsytrf_bufferSize`| | | | |`hipsolverDnZsytrf_bufferSize`|5.1.0| | | | |
|`cusolverDnZsytri`|10.1| | | | | | | | | |
|`cusolverDnZsytri_bufferSize`|10.1| | | | | | | | | |
|`cusolverDnZungbr`|8.0| | | |`hipsolverDnZungbr`|5.1.0| | | | |
|`cusolverDnZungbr_bufferSize`|8.0| | | |`hipsolverDnZungbr_bufferSize`|5.1.0| | | | |
|`cusolverDnZungqr`|8.0| | | |`hipsolverDnZungqr`|5.1.0| | | | |
|`cusolverDnZungqr_bufferSize`|8.0| | | |`hipsolverDnZungqr_bufferSize`|5.1.0| | | | |
|`cusolverDnZungtr`|8.0| | | |`hipsolverDnZungtr`|5.1.0| | | | |
|`cusolverDnZungtr_bufferSize`|8.0| | | |`hipsolverDnZungtr_bufferSize`|5.1.0| | | | |
|`cusolverDnZunmqr`| | | | |`hipsolverDnZunmqr`|5.1.0| | | | |
|`cusolverDnZunmqr_bufferSize`|8.0| | | |`hipsolverDnZunmqr_bufferSize`|5.1.0| | | | |
|`cusolverDnZunmtr`|8.0| | | |`hipsolverDnZunmtr`|5.1.0| | | | |
|`cusolverDnZunmtr_bufferSize`|8.0| | | |`hipsolverDnZunmtr_bufferSize`|5.1.0| | | | |
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
|`cusolverRfAccessBundledFactorsDevice`| | | | |`hipsolverRfAccessBundledFactorsDevice`|5.6.0| | | | |
|`cusolverRfAnalyze`| | | | |`hipsolverRfAnalyze`|5.6.0| | | | |
|`cusolverRfBatchAnalyze`| | | | |`hipsolverRfBatchAnalyze`|5.6.0| | | | |
|`cusolverRfBatchRefactor`| | | | |`hipsolverRfBatchRefactor`|5.6.0| | | | |
|`cusolverRfBatchResetValues`| | | | |`hipsolverRfBatchResetValues`|5.6.0| | | | |
|`cusolverRfBatchSetupHost`| | | | |`hipsolverRfBatchSetupHost`|5.6.0| | | | |
|`cusolverRfBatchSolve`| | | | |`hipsolverRfBatchSolve`|5.6.0| | | | |
|`cusolverRfBatchZeroPivot`| | | | |`hipsolverRfBatchZeroPivot`|5.6.0| | | | |
|`cusolverRfCreate`| | | | |`hipsolverRfCreate`|5.6.0| | | | |
|`cusolverRfDestroy`| | | | |`hipsolverRfDestroy`|5.6.0| | | | |
|`cusolverRfExtractBundledFactorsHost`| | | | |`hipsolverRfExtractBundledFactorsHost`|5.6.0| | | | |
|`cusolverRfExtractSplitFactorsHost`| | | | |`hipsolverRfExtractSplitFactorsHost`|5.6.0| | | | |
|`cusolverRfGetAlgs`| | | | | | | | | | |
|`cusolverRfGetMatrixFormat`| | | | |`hipsolverRfGetMatrixFormat`|5.6.0| | | | |
|`cusolverRfGetNumericBoostReport`| | | | |`hipsolverRfGetNumericBoostReport`|5.6.0| | | | |
|`cusolverRfGetNumericProperties`| | | | |`hipsolverRfGetNumericProperties`|5.6.0| | | | |
|`cusolverRfGetResetValuesFastMode`| | | | |`hipsolverRfGetResetValuesFastMode`|5.6.0| | | | |
|`cusolverRfRefactor`| | | | |`hipsolverRfRefactor`|5.6.0| | | | |
|`cusolverRfResetValues`| | | | |`hipsolverRfResetValues`|5.6.0| | | | |
|`cusolverRfSetAlgs`| | | | |`hipsolverRfSetAlgs`|5.6.0| | | | |
|`cusolverRfSetMatrixFormat`| | | | |`hipsolverRfSetMatrixFormat`|5.6.0| | | | |
|`cusolverRfSetNumericProperties`| | | | |`hipsolverRfSetNumericProperties`|5.6.0| | | | |
|`cusolverRfSetResetValuesFastMode`| | | | |`hipsolverRfSetResetValuesFastMode`|5.6.0| | | | |
|`cusolverRfSetupDevice`| | | | |`hipsolverRfSetupDevice`|5.6.0| | | | |
|`cusolverRfSetupHost`| | | | |`hipsolverRfSetupHost`|5.6.0| | | | |
|`cusolverRfSolve`| | | | |`hipsolverRfSolve`|5.6.0| | | | |
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
|`cusolverSpCreate`| | | | |`hipsolverSpCreate`|6.1.0| | | |6.1.0|
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
|`cusolverSpDcsrlsvchol`| | | | |`hipsolverSpDcsrlsvchol`|6.1.0| | | |6.1.0|
|`cusolverSpDcsrlsvcholHost`| | | | |`hipsolverSpDcsrlsvcholHost`|6.1.0| | | |6.1.0|
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
|`cusolverSpDestroy`| | | | |`hipsolverSpDestroy`|6.1.0| | | |6.1.0|
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
|`cusolverSpScsrlsvchol`| | | | |`hipsolverSpScsrlsvchol`|6.1.0| | | |6.1.0|
|`cusolverSpScsrlsvcholHost`| | | | |`hipsolverSpScsrlsvcholHost`|6.1.0| | | |6.1.0|
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
|`cusolverSpSetStream`| | | | |`hipsolverSpSetStream`|6.1.0| | | |6.1.0|
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