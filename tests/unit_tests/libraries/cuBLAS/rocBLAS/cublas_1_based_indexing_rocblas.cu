// RUN: %run_test hipify "%s" "%t" %hipify_args 1 -roc %clang_args

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "rocblas.h"
#include "cublas_v2.h"
#define M 6
#define N 5
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
// CHECK: static __inline__ void modify(rocblas_handle handle, float *m, int ldm, int
static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int
  n, int p, int q, float alpha, float beta) {
  // CHECK: rocblas_sscal(handle, n - p + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
  // CHECK: rocblas_sscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
  cublasSscal(handle, n - p + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
  cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}
int main(void) {
  // CHECK: hipError_t cudaStat;
  // CHECK: rocblas_status stat;
  // CHECK: rocblas_handle handle;
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float* devPtrA;
  float* a = 0;
  a = (float *)malloc(M * N * sizeof(*a));
  if (!a) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      a[IDX2F(i, j, M)] = (float)((i - 1) * M + j);
    }
  }
  // CHECK: cudaStat = hipMalloc((void**)&devPtrA, M*N * sizeof(*a));
  cudaStat = cudaMalloc((void**)&devPtrA, M*N * sizeof(*a));
  // CHECK: if (cudaStat != hipSuccess) {
  if (cudaStat != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  // CHECK: stat = rocblas_create_handle(&handle);
  stat = cublasCreate(&handle);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }
  // CHECK: stat = rocblas_set_matrix(M, N, sizeof(*a), a, M, devPtrA, M);
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    // CHECK: hipFree(devPtrA);
    // CHECK: rocblas_destroy_handle(handle);
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  modify(handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  // CHECK: stat = rocblas_get_matrix(M, N, sizeof(*a), devPtrA, M, a, M);
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  // CHECK: if (stat != rocblas_status_success) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    // CHECK: hipFree(devPtrA);
    // CHECK: rocblas_destroy_handle(handle);
    cudaFree(devPtrA);
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  // CHECK: hipFree(devPtrA);
  // CHECK: rocblas_destroy_handle(handle);
  cudaFree(devPtrA);
  cublasDestroy(handle);
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      printf("%7.0f", a[IDX2F(i, j, M)]);
    }
    printf("\n");
  }
  free(a);
  return EXIT_SUCCESS;
}
