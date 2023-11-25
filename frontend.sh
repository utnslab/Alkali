#!/bin/bash
OUT_FILE_NAME=${1%.*}.mlir
ninja -C build/ && ./build/bin/ep2c --emit=mlir $1 &> $OUT_FILE_NAME
