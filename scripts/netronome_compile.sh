#!/bin/bash

set -e

EXTRA_OPT=""
ninja -C build/ -j 32
shift $((OPTIND-1))

BASE_NAME="${1%.*}"
rm -rf "netronome_out"
mkdir -p "netronome_out"

set -x

file="$1"
# Extract the extension from the filename
extension="${file##*.}"

# whether disable cut
disable_cut="$2"
# whether disable mapping
disable_map="$3"


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

# Common Optimizations
./build/bin/ep2c-opt "netronome_out/$BASE_NAME.mlir" -ep2-context-infer  -canonicalize -ep2-buffer-to-value --ep2-context-to-argument -canonicalize -cse -canonicalize -o netronome_out/commonopt.mlir


# Cut Optimization
if [ "$disable_cut" != "disable_cut" ]; then
  ./build/bin/ep2c-opt -ep2-pipeline-handler="mode=kcut knum=3" \
    "netronome_out/commonopt.mlir" -o "netronome_out/cut.mlir" 
else
  echo "Warning: cut optimization is disabled, skipping cut optimization"
  cp "netronome_out/commonopt.mlir" "netronome_out/cut.mlir"
fi

# Mapping Optimization
if [ "$disable_map" != "disable_map" ]; then
  ./build/bin/ep2c-opt -ep2-pipeline-canon="mode=netronome" -cse \
    -ep2-context-to-mem -ep2-global-to-partition \
    "netronome_out/cut.mlir" -o "netronome_out/mapped.mlir"
else
  echo "Warning: mapping optimization is disabled, skipping mapping optimization"
  cp "netronome_out/cut.mlir" "netronome_out/mapped.mlir"
fi

# Backend Codegen
./build/bin/ep2c-opt -ep2-repack -ep2-handler-repl -ep2-collect-header -ep2-lower-emitc -ep2-lower-memcpy -ep2-update-ppg -ep2-lower-noctxswap -ep2-gpr-promote -ep2-emit-netronome="basePath=netronome_out" "netronome_out/mapped.mlir" -o /dev/null

echo "Generated files in netronome_out"
