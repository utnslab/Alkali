#!/bin/bash

set -e

OUT_FILE_NAME="out-${1%.*}"
mkdir -p "$OUT_FILE_NAME"

cd llvm-project && ninja -C build/ && cd -
ninja -C build/
./build/bin/ep2c $1 -o "$OUT_FILE_NAME/"
