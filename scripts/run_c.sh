C_INPUT_DIR=tests/experiments_c
C_PREPROCESS=cpre.c
C_LLVM_FILE=cinput.ll
C_MLIR_FILE=cinput.mlir
OUT_FILE=out.mlir

IS_SPLIT=false
OPTIONS=

while getopts ":dlo:s" opt; do
    case $opt in
        l)
            echo "Listing of example files:"
            ls $C_INPUT_DIR
            exit 0
            ;;
        o)
            OUT_FILE=${OPTARG}
            ;;
        d)  OPTIONS="--debug-only=ep2-lift-llvm,dialect-conversion --mlir-print-ir-after-failure"
            ;;
        s)
            IS_SPLIT=true
            ;;
        ?)
            echo "INVALID OPT"
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

C_INPUT_FILE=$C_INPUT_DIR/${1:-pipeline.c}
echo "Compile ${C_INPUT_FILE} to ${C_MLIR_FILE}"

BIN_DIR=build/bin
CLANG_BIN_DIR=llvm-project/build/bin
EXTRACT_PY=./scripts/extract_struct_def.py

cc -E $C_INPUT_FILE -o $C_PREPROCESS
python3 $EXTRACT_PY $C_INPUT_FILE -o cinput.struct.json
$CLANG_BIN_DIR/clang -S -emit-llvm $C_PREPROCESS -o $C_LLVM_FILE
$CLANG_BIN_DIR/mlir-translate  --import-llvm $C_LLVM_FILE -o $C_MLIR_FILE
$BIN_DIR/ep2c-opt $C_MLIR_FILE --convert-scf-to-cf -cse -o cinput2.mlir

$BIN_DIR/ep2c-opt cinput2.mlir --ep2-lift-llvm="struct-desc=cinput.struct.json" $OPTIONS -cse -cse -canonicalize --ep2-context-to-mem="transform-extern=true" -o $OUT_FILE

echo "Generation of IR success. Saved to $OUT_FILE"

if [ "$IS_SPLIT" = true ]; then
    echo "Split $OUT_FILE"
    $BIN_DIR/ep2c-opt $OUT_FILE -ep2-buffer-to-value --ep2-pipeline-handler --mlir-print-ir-after-failure -o split.mlir  2> debug.mlir
fi
