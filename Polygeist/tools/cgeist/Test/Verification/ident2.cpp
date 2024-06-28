// RUN: cgeist %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

struct MOperandInfo* begin();

struct MOperandInfo& inner() {
  return begin()[0];
}

// CHECK:   func @_Z5innerv() -> memref<?x2xi8>  
// CHECK-NEXT:     %[[V0:.+]] = call @_Z5beginv() : () -> memref<?x2xi8>
// CHECK-NEXT:     return %[[V0]] : memref<?x2xi8>
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @_Z5beginv() -> memref<?x2xi8> 
