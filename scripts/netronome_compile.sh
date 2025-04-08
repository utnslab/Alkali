#!/bin/bash

set -e

EXTRA_OPT=""

shift $((OPTIND-1))

BASE_NAME="${1%.*}"
rm -rf "netronome_out"
mkdir -p "netronome_out"

# cd llvm-project && ninja -C build/ && cd -
# ninja -C build/

set -x

file="$1"
# Extract the extension from the filename
extension="${file##*.}"

case "$extension" in
  ep2)
    ./build/bin/ep2c $1 --emit=mlir -o "netronome_out/$BASE_NAME.mlir" 
    echo "Running base command for .ep2 file..."
    ;;
  mlir)
    cp $1 "netronome_out/$BASE_NAME.mlir"
    ;;
  *)
    echo "Unsupported file extension: .$extension"
    exit 1
    ;;
esac

./build/bin/ep2c-opt -canonicalize -ep2-buffer-to-value --ep2-context-to-argument \
    -ep2-pipeline-handler="mode=search" \
    "netronome_out/$BASE_NAME.mlir" -o "netronome_out/cut.mlir" 

./build/bin/ep2c-opt -ep2-pipeline-canon="mode=netronome" -cse \
    -ep2-context-to-mem -ep2-global-to-partition \
    "netronome_out/cut.mlir" -o "netronome_out/canon.mlir"

./build/bin/ep2c-opt -ep2-repack -ep2-handler-repl -ep2-collect-header -ep2-lower-emitc -ep2-lower-memcpy -ep2-update-ppg -ep2-lower-noctxswap -ep2-gpr-promote -ep2-emit-netronome="basePath=netronome_out" "netronome_out/canon.mlir" -o /dev/null
echo "Generated files in netronome_out"
