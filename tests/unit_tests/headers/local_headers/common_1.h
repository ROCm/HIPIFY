// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args
// CHECK: #ifndef common_1_H
// CHECK: #define common_1_H
// CHECK: void temp(
// CHECK: nSolverIters,
// CHECK: float       *v);
// CHECK: #endif
// CHECK-NOT: hip_runtime.h
// CHECK-NOT: cuda_runtime.h
#ifndef common_1_H
#define common_1_H

void temp(const float *I0,           
                     const float *I1,          
                     int          width,        
                     int          height,       
                     int          stride,       
                     float        alpha,        
                     int          nLevels,      
                     int          nWarpIters,   
                     int          nSolverIters, 
                     float       *u,            
                     float       *v);                 
#endif