#!/bin/bash

set -e

EXTRA_OPT=""

shift $((OPTIND-1))

OUT_FILE_NAME="out-netronome-${1%.*}"
BASE_NAME="${1%.*}"
rm -rf "outs-netronome/$OUT_FILE_NAME"
mkdir -p "outs-netronome/$OUT_FILE_NAME"

cd llvm-project && ninja -C build/ && cd -
ninja -C build/

set -x

file="$1"
# Extract the extension from the filename
extension="${file##*.}"

case "$extension" in
  ep2)
    ./build/bin/ep2c $1 --emit=mlir -o "outs-netronome/$OUT_FILE_NAME/$BASE_NAME.mlir" 
    echo "Running base command for .ep2 file..."
    ;;
  mlir)
    cp $1 "outs-netronome/$OUT_FILE_NAME/$BASE_NAME.mlir"
    ;;
  *)
    echo "Unsupported file extension: .$extension"
    exit 1
    ;;
esac

./build/bin/ep2c-opt -canonicalize -ep2-buffer-to-value --ep2-context-to-argument \
    -ep2-pipeline-handler="mode=search" \
    "outs-netronome/$OUT_FILE_NAME/$BASE_NAME.mlir" -o "outs-netronome/$OUT_FILE_NAME/cut.mlir" 

./build/bin/ep2c-opt -ep2-pipeline-canon="mode=netronome" -cse \
    -ep2-context-to-mem -ep2-global-to-partition \
    "outs-netronome/$OUT_FILE_NAME/cut.mlir" -o "outs-netronome/$OUT_FILE_NAME/canon.mlir"

./build/bin/ep2c-opt -ep2-repack -ep2-handler-repl -ep2-collect-header -ep2-lower-emitc -ep2-lower-memcpy -ep2-update-ppg -ep2-lower-noctxswap -ep2-gpr-promote -ep2-emit-netronome="basePath=outs-netronome/$OUT_FILE_NAME" "outs-netronome/$OUT_FILE_NAME/canon.mlir" -o /dev/null
echo "Generated files in outs-netronome/$OUT_FILE_NAME"
