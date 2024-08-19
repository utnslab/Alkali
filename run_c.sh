C_INPUT_DIR=tests/experiments_c
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

$BIN_DIR/cgeist $C_INPUT_FILE -S > $C_MLIR_FILE
$BIN_DIR/ep2c-opt $C_MLIR_FILE --convert-scf-to-cf -cse -o $C_MLIR_FILE

$BIN_DIR/ep2c-opt cinput.mlir --ep2-lift-llvm $OPTIONS -cse -cse -canonicalize -o $OUT_FILE 2> debug.mlir

if [ "$IS_SPLIT" = true ]; then
    echo "Split $OUT_FILE"
    $BIN_DIR/ep2c-opt $OUT_FILE --ep2-pipeline-handler -ep2-buffer-to-value --mlir-print-ir-after-failure -o split.mlir
fi
