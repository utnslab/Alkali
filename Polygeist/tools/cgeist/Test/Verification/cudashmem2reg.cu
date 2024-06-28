// RUN: cgeist %s --cuda-gpu-arch=sm_60 -nocudalib -nocudainc %resourcedir --function=* --cuda-lower --cpuify="distribute" -S | FileCheck %s

#include "Inputs/cuda.h"
#include "__clang_cuda_builtin_vars.h"

#define N 2

__global__ void foo(double * w, double* s) {
  __shared__ double sumW;

  if(0 == threadIdx.x) {
    sumW = s[0];
  }

  __syncthreads();

  w[threadIdx.x] = w[threadIdx.x] / sumW;
}

double *bar(double *w, double *s) {
  foo<<< 1, N >>>(w, s);
  return w;
}

//      CHECK:  func.func @_Z3barPdS_(%[[arg0:.+]]: memref<?xf64>, %[[arg1:.+]]: memref<?xf64>)
//  CHECK-DAG:    %[[c0_i32:.+]] = arith.constant 0 : i32
//  CHECK-DAG:    %[[c1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[c2:.+]] = arith.constant 2 : index
//  CHECK-DAG:    %[[c0:.+]] = arith.constant 0 : index
//  CHECK-DAG:    %[[V0:.+]] = memref.alloca() : memref<f64>
// CHECK-NEXT:    scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c2]]) step (%[[c1]]) {
// CHECK-NEXT:      %[[V1:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:      %[[V2:.+]] = arith.cmpi eq, %[[V1]], %[[c0_i32]] : i32
// CHECK-NEXT:      scf.if %[[V2]] {
// CHECK-NEXT:        %[[V3:.+]] = memref.load %[[arg1]][%[[c0]]] : memref<?xf64>
// CHECK-NEXT:        memref.store %[[V3]], %[[V0]][] : memref<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.parallel (%[[arg2:.+]]) = (%[[c0]]) to (%[[c2]]) step (%[[c1]]) {
// CHECK-NEXT:      %[[V1:.+]] = memref.load %[[arg0]][%[[arg2]]] : memref<?xf64>
// CHECK-NEXT:      %[[V2:.+]] = memref.load %[[V0]][] : memref<f64>
// CHECK-NEXT:      %[[V3:.+]] = arith.divf %[[V1]], %[[V2]] : f64
// CHECK-NEXT:      memref.store %[[V3]], %[[arg0]][%[[arg2]]] : memref<?xf64>
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[arg0]] : memref<?xf64>
// CHECK-NEXT:  }
