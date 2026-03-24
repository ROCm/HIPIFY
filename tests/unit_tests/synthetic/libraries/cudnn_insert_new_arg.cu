// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --miopen %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "miopen/miopen.h"
#include "cudnn.h"

int main() {
  printf("e_insert_new_argument: duplicate variable name guard test\n");

  // CHECK: miopenStatus_t status;
  cudnnStatus_t status;

  // CHECK: miopenHandle_t handle;
  cudnnHandle_t handle;

  // CHECK: miopenDropoutDescriptor_t DropoutDescriptor;
  cudnnDropoutDescriptor_t DropoutDescriptor;

  float dropout = 0.5f;
  void* states = nullptr;
  size_t reserveSpaceNumBytes = 0;
  unsigned long long seed = 42;

  // CUDA: cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc, cudnnHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed);
  // MIOPEN: MIOPEN_EXPORT miopenStatus_t miopenSetDropoutDescriptor(miopenDropoutDescriptor_t dropoutDesc, miopenHandle_t handle, float dropout, void* states, size_t stateSizeInBytes, unsigned long long seed, bool use_mask, bool state_evo, miopenRNGType_t rng_mode);

  // First call: base names, no suffix.
  // CHECK: bool hipify_use_mask = {};
  // CHECK-NEXT: bool hipify_state_evo = {};
  // CHECK-NEXT: miopenRNGType_t hipify_rng_mode = {};
  // CHECK-NEXT: status = miopenSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed, hipify_use_mask, hipify_state_evo, hipify_rng_mode);
  status = cudnnSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);

  // Second call in same scope: _1 suffix to avoid redefinition.
  // CHECK: bool hipify_use_mask_1 = {};
  // CHECK-NEXT: bool hipify_state_evo_1 = {};
  // CHECK-NEXT: miopenRNGType_t hipify_rng_mode_1 = {};
  // CHECK-NEXT: status = miopenSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed, hipify_use_mask_1, hipify_state_evo_1, hipify_rng_mode_1);
  status = cudnnSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);

  // Third call in same scope: _2 suffix.
  // CHECK: bool hipify_use_mask_2 = {};
  // CHECK-NEXT: bool hipify_state_evo_2 = {};
  // CHECK-NEXT: miopenRNGType_t hipify_rng_mode_2 = {};
  // CHECK-NEXT: status = miopenSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed, hipify_use_mask_2, hipify_state_evo_2, hipify_rng_mode_2);
  status = cudnnSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);

  // CHECK: return 0;
  return 0;
}
