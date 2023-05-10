// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// CHECK: #include "hip/hip_runtime.h"
#include "cuda.h"
// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas_v2.h"
#include "cublas_v2.h"
// CHECK-NOT: #include "hipblas.h"
#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
static __inline__ void modify(float *m, int ldm, int n, int p, int q, float
  alpha, float beta) {
  // CHECK: hipblasHandle_t blasHandle;
  cublasHandle_t blasHandle;
  // CHECK: hipblasStatus_t blasStatus = hipblasCreate(&blasHandle);
  cublasStatus_t blasStatus = cublasCreate(&blasHandle);
  // CHECK: hipblasSscal(blasHandle, n - p, &alpha, &m[IDX2C(p, q, ldm)], ldm);
  cublasSscal(blasHandle, n - p, &alpha, &m[IDX2C(p, q, ldm)], ldm);
  // CHECK: hipblasSscal(blasHandle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1);
  cublasSscal(blasHandle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1);
  // CHECK: hipblasDestroy(blasHandle);
  cublasDestroy(blasHandle);
}
int main(void) {
  int i, j;
  // CHECK: hipblasStatus_t stat;
  cublasStatus_t stat;
  // CHECK: hipError_t result = hipSuccess;
  cudaError_t result = cudaSuccess;
  float* devPtrA;
  float* a = 0;
  a = (float *)malloc(M * N * sizeof(*a));
  if (!a) {
    printf("host memory allocation failed");
    return EXIT_FAILURE;
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      a[IDX2C(i, j, M)] = (float)(i * M + j + 1);
    }
  }
  // CHECK: hipMalloc((void**)&devPtrA, M*N*sizeof(*a));
  result = cudaMalloc((void**)&devPtrA, M*N*sizeof(*a));
  // CHECK: if (result != hipSuccess) {
  if (result != cudaSuccess) {
    printf("device memory allocation failed");
    return EXIT_FAILURE;
  }
  // CHECK: stat = hipblasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
  // CHECK: if (stat != HIPBLAS_STATUS_SUCCESS) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data download failed");
    // CHECK: hipFree(devPtrA);
    cudaFree(devPtrA);
    return EXIT_FAILURE;
  }
  modify(devPtrA, M, N, 1, 2, 16.0f, 12.0f);
  // CHECK: stat = hipblasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
  // CHECK: if (stat != HIPBLAS_STATUS_SUCCESS) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("data upload failed");
    // CHECK: hipFree(devPtrA);
    cudaFree(devPtrA);
    return EXIT_FAILURE;
  }
  // CHECK: hipFree(devPtrA);
  cudaFree(devPtrA);
  for (j = 0; j < N; j++) {
    for (i = 0; i < M; i++) {
      printf("%7.0f", a[IDX2C(i, j, M)]);
    }
    printf("\n");
  }
  free(a);
  return EXIT_SUCCESS;
}
