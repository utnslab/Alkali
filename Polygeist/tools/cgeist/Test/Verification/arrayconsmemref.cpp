// RUN: cgeist %s --function=* -S | FileCheck %s

struct AIntDivider {
    AIntDivider() : divisor(3) {}
    unsigned int divisor;
};

void kern() {
    AIntDivider sizes_[25];
}

// CHECK:   func @_Z4kernv()  
// CHECK-DAG:     %[[c25:.+]] = arith.constant 25 : index
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<25x1xi32>
// CHECK-NEXT:     scf.for %[[arg0:.+]] = %[[c0]] to %[[c25]] step %[[c1]] {
// CHECK-NEXT:       %[[V1:.+]] = "polygeist.subindex"(%[[V0]], %[[arg0]]) : (memref<25x1xi32>, index) -> memref<?x1xi32>
// CHECK-NEXT:       call @_ZN11AIntDividerC1Ev(%[[V1]]) : (memref<?x1xi32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN11AIntDividerC1Ev(%[[arg0:.+]]: memref<?x1xi32>)  
// CHECK-NEXT:     %[[c3_i32:.+]] = arith.constant 3 : i32
// CHECK-NEXT:     affine.store %[[c3_i32]], %[[arg0]][0, 0] : memref<?x1xi32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

