#!/bin/bash

set -e

OUT_FILE_NAME="out-${1%.*}"

ninja -C build/
./build/bin/ep2c $1 --emit=mlir -o "./tmp.mlir" 
./build/bin/ep2c-opt -canonicalize -ep2-context-infer -ep2-emit-fpga -o "tmpopt.mlir" "tmp.mlir"
#./build/bin/ep2c-opt -canonicalize -ep2-context-infer -o "tmpopt.mlir" "tmp.mlir"
