// RUN: cgeist %s --function=* -S | FileCheck %s

int run();

void what() {
  for (;;) {
    if (run()) break;
  }
}

// CHECK:   func.func @what()  
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     scf.while (%[[arg0:.+]] = %[[true]]) : (i1) -> () {
// CHECK-NEXT:       scf.condition(%[[arg0]])
// CHECK-NEXT:     } do {
// CHECK-NEXT:       %[[V0:.+]] = func.call @run() : () -> i32
// CHECK-NEXT:       %[[V1:.+]] = arith.cmpi eq, %[[V0]], %[[c0_i32]] : i32
// CHECK-NEXT:       scf.yield %[[V1]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
