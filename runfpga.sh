#!/bin/bash

set -e

OUT_FILE_NAME="out-${1%.*}"

ninja -C build/
./build/bin/ep2c $1 --emit=mlir -o "./tmp.mlir" 
./build/bin/ep2c-opt -canonicalize -cse --ep2-context-infer --ep2-context-to-argument -ep2-buffer-to-value -canonicalize -cse -canonicalize -cf-to-pred  -canonicalize -cse -canonicalize -ep2-emit-fpga  tmp.mlir -o tmpopt.mlir
