EP2_BINS=./build/bin
LLVM_BINS=./llvm-project/build/bin

$EP2_BINS/ep2c tests/experiments/transport_rx.ep2.txt -o tmp.mlir
$EP2_BINS/ep2c-opt --ep2-context-infer --ep2-context-to-argument -canonicalize -cse -canonicalize \
    -ep2-buffer-reuse -ep2-dfe -ep2-dpe -ep2-canon -canonicalize -cse \
    -ep2-dpe -canonicalize -cse tmp.mlir -o final.mlir
$EP2_BINS/ep2c-opt -ep2-lower-llvm='generate=raw' -convert-cf-to-llvm \
    -canonicalize -cse -canonicalize final.mlir  -o final_llvm.mlir
$LLVM_BINS/mlir-opt --inline final_llvm.mlir -o final_inline.mlir
$LLVM_BINS/mlir-translate -mlir-to-llvmir final_inline.mlir -o final.ll
$LLVM_BINS/clang -O3 -c final.ll -o ep2.o
