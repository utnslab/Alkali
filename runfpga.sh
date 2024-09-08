#!/bin/bash

set -e
set -x

OUT_FILE_NAME="out-${1%.*}"

ninja -C build/
./build/bin/ep2c $1 --emit=mlir -o "./tmp.mlir" 

#mapped
#./build/bin/ep2c-opt  --ep2-context-infer --ep2-context-to-argument -ep2-buffer-to-value -canonicalize -cse -canonicalize -ep2-atomic-id -canonicalize -cse -canonicalize -ep2-linearize  --ep2-mapping="arch-spec-file=tests/specs/fpga.json cost-model=fpga" --ep2-global-to-partition -ep2-controller-generation tmp.mlir -o mapped.mlir

#unmapped
./build/bin/ep2c-opt  --ep2-context-infer --ep2-context-to-argument -ep2-buffer-to-value -canonicalize -cse -canonicalize -ep2-atomic-id -canonicalize -cse -canonicalize -ep2-linearize  --ep2-global-to-partition  tmp.mlir -o mapped.mlir

./build/bin/ep2c-opt  -ep2-emit-fpga mapped.mlir -o tmpopt.mlir
