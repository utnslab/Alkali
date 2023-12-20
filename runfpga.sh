#!/bin/bash

set -e

OUT_FILE_NAME="out-${1%.*}"

./build/bin/ep2c $1 --emit=mlir -o "./tmp.mlir" 
./build/bin/ep2c-opt -canonicalize -ep2-context-infer -ep2-emit-fpga -o "tmpopt.mlir" "tmp.mlir"
