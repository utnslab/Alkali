#!/bin/bash

set -e

OUT_FILE_NAME="out-${1%.*}"
BASE_NAME="${1%.*}"
mkdir -p "$OUT_FILE_NAME"

cd llvm-project && ninja -C build/ && cd -
ninja -C build/
./build/bin/ep2c $1 --emit=mlir -o "$OUT_FILE_NAME/" 
./build/bin/ep2c-opt -canonicalize -ep2-context-infer -o "$OUT_FILE_NAME/canon.mlir" "$OUT_FILE_NAME/$BASE_NAME.mlir"
./build/bin/ep2c-opt -ep2-nop-elim -ep2-collect-header -ep2-lower-emitc -ep2-lower-intrinsics -ep2-emit-files "$OUT_FILE_NAME/canon.mlir" -o /dev/null
