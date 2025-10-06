// RUN: %run_test hipify "%s" "%t" %hipify_args --local-headers %clang_args
// CHECK: #ifndef common_2_H
// CHECK: #define common_2_H
// CHECK: void temp_1(
// CHECK: nSolverIters,
// CHECK: float       *v);
// CHECK: #endif
// CHECK-NOT: hip_runtime.h
// CHECK-NOT: cuda_runtime.h
#ifndef common_2_H
#define common_2_H

void temp_1(const float *I0,           
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