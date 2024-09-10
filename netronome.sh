#!/bin/bash

set -e

MLIR_INPUT=false
REPLICATION_OPT=""

while getopts ":mr:" opt; do
    case $opt in
        m)
            # set the input to mlir
            MLIR_INPUT=true
            ;;
        r)
            # set the replication number for mlir mode
            REPLICATION_OPT="rep=${OPTARG}"
            ;;
        ?)
            echo "INVALID OPT"
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

OUT_FILE_NAME="out-netronome-${1%.*}"
BASE_NAME="${1%.*}"
mkdir -p "outs-netronome/$OUT_FILE_NAME"

cd llvm-project && ninja -C build/ && cd -
ninja -C build/

# canonicalize from ep2 or mlir
if [ "$MLIR_INPUT" = true ]; then
  ./build/bin/ep2c-opt --ep2-pipeline-canon="mode=netronome ${REPLICATION_OPT}" -cse  \
    -ep2-context-to-mem -ep2-global-to-partition \
    $1 -o "outs-netronome/$OUT_FILE_NAME/canon.mlir" 
else
  ./build/bin/ep2c $1 --emit=mlir -o "outs-netronome/$OUT_FILE_NAME/$BASE_NAME.mlir" 
  ./build/bin/ep2c-opt -ep2-context-infer -ep2-context-to-argument -canonicalize -cse -canonicalize -ep2-buffer-reuse -ep2-dfe -ep2-dpe -ep2-canon -canonicalize -cse -ep2-dpe -canonicalize -cse -ep2-context-to-mem --ep2-controller-generation -ep2-global-to-partition -o "outs-netronome/$OUT_FILE_NAME/canon.mlir" "outs-netronome/$OUT_FILE_NAME/$BASE_NAME.mlir"
fi

./build/bin/ep2c-opt -ep2-repack -ep2-handler-repl -ep2-collect-header -ep2-lower-emitc -ep2-lower-memcpy -ep2-update-ppg -ep2-lower-noctxswap -ep2-gpr-promote -ep2-emit-netronome="basePath=outs-netronome/$OUT_FILE_NAME" "outs-netronome/$OUT_FILE_NAME/canon.mlir" -o /dev/null
echo "Generated files in outs-netronome/$OUT_FILE_NAME"
