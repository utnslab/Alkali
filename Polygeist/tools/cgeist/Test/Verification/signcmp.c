// RUN: cgeist %s --function=* -S | FileCheck %s

void run();
unsigned int cmp(int a, int b) {
    if (a < b) {
        run();
    }
    return a < b;
}
unsigned int cmp2() {
    if (-2 < 0) {
        run();
    }
    return -2 < 0;
}

// CHECK:   func @cmp(%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 
// CHECK-NEXT:     %[[V0:.+]] = arith.cmpi slt, %[[arg0]], %[[arg1]] : i32
// CHECK-NEXT:     scf.if %[[V0]] {
// CHECK-NEXT:       call @run() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V1:.+]] = arith.extui %[[V0]] : i1 to i32
// CHECK-NEXT:     return %[[V1]] : i32
// CHECK-NEXT:   }
// CHECK:   func @cmp2() -> i32 
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     call @run() : () -> ()
// CHECK-NEXT:     return %c1_i32 : i32
// CHECK-NEXT:   }
