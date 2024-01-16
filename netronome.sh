#!/bin/bash

set -e

OUT_FILE_NAME="out-netronome-${1%.*}"
BASE_NAME="${1%.*}"
mkdir -p "$OUT_FILE_NAME"

cd llvm-project && ninja -C build/ && cd -
ninja -C build/
./build/bin/ep2c $1 --emit=mlir -o "$OUT_FILE_NAME/$BASE_NAME.mlir" 
./build/bin/ep2c-opt -canonicalize -cse -ep2-context-infer -ep2-context-to-argument -canonicalize -cse -ep2-buffer-reuse -ep2-context-to-mem -o "$OUT_FILE_NAME/canon.mlir" "$OUT_FILE_NAME/$BASE_NAME.mlir"
./build/bin/ep2c-opt -ep2-handler-repl -ep2-collect-header -ep2-lower-emitc -ep2-lower-memcpy -ep2-lower-noctxswap -ep2-update-ppg -ep2-emit-netronome="basePath=$OUT_FILE_NAME" "$OUT_FILE_NAME/canon.mlir" -o /dev/null
