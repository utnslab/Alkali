C_INPUT_FILE=tests/experiments_c/pipeline.c
C_MLIR_FILE=cinput.mlir

BIN_DIR=build/bin

$BIN_DIR/cgeist $C_INPUT_FILE -S > $C_MLIR_FILE
$BIN_DIR/ep2c-opt $C_MLIR_FILE --convert-scf-to-cf -cse -o $C_MLIR_FILE

$BIN_DIR/ep2c-opt cinput.mlir --ep2-lift-llvm --debug-only="ep2-lift-llvm,dialect-conversion" --mlir-print-ir-after-failure -cse -cse -canonicalize 2> debug.mlir > out.mlir
