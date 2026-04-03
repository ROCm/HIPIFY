// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --miopen %clang_args -D__CUDA_API_VERSION_INTERNAL

#pragma once
// CHECK: #include "miopen/miopen.h"
#include "cudnn.h"

inline void headerSetup(cudnnDropoutDescriptor_t DropoutDescriptor,
                        cudnnHandle_t handle, float dropout,
                        void* states, size_t reserveSpaceNumBytes,
                        unsigned long long seed) {
  // CHECK: miopenStatus_t status;
  cudnnStatus_t status;
  // CHECK: bool hipify_use_mask = {};
  // CHECK-NEXT: bool hipify_state_evo = {};
  // CHECK-NEXT: miopenRNGType_t hipify_rng_mode = {};
  // CHECK-NEXT: status = miopenSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed, hipify_use_mask, hipify_state_evo, hipify_rng_mode);
  status = cudnnSetDropoutDescriptor(DropoutDescriptor, handle, dropout, states, reserveSpaceNumBytes, seed);
}
