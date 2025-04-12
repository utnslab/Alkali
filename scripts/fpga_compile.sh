ninja -C build/ -j 32

rm -rf ./fpga_out
mkdir -p ./fpga_out

file="$1"
# Extract the extension from the filename
extension="${file##*.}"
BASE_NAME="${1%.*}"

# whether disable cut
disable_cut="$2"
# whether disable mapping
disable_map="$3"

case "$extension" in
  ep2)
    ./build/bin/ep2c $1 -o "fpga_out/$BASE_NAME.mlir" 
    ;;
  mlir)
    cp $1 "fpga_out/$BASE_NAME.mlir"
    ;;
  *)
    echo "Unsupported file extension: .$extension"
    exit 1
    ;;
esac

# Common Optimizations
./build/bin/ep2c-opt "fpga_out/$BASE_NAME.mlir" -ep2-context-infer --ep2-context-to-argument -ep2-buffer-to-value -canonicalize -cse -canonicalize -ep2-atomic-id -canonicalize -cse -canonicalize -ep2-linearize -o fpga_out/commonopt.mlir

# Cut Optimization
if [ "$disable_cut" != "disable_cut" ]; then
  ./build/bin/ep2c-opt "fpga_out/commonopt.mlir" --ep2-pipeline-handler="mode=search" -o "fpga_out/cut.mlir"
else
  echo "Warning: cut optimization is disabled, skipping cut optimization"
  cp "fpga_out/commonopt.mlir" "fpga_out/cut.mlir"
fi


# Mapping Optimization
if [ "$disable_map" != "disable_map" ]; then
  ./build/bin/ep2c-opt --ep2-pipeline-canon="inline-table=true" -ep2-global-to-partition "fpga_out/cut.mlir" -o "fpga_out/mapped.mlir"
else
  echo "Warning: mapping optimization is disabled, skipping mapping optimization"
  cp "fpga_out/cut.mlir" "fpga_out/mapped.mlir"
fi
# ./build/bin/ep2c-opt --ep2-pipeline-canon="inline-table=true premap-only=true" --ep2-mapping="arch-spec-file=tests/specs/fpga.json cost-model=fpga" --ep2-global-to-partition -ep2-controller-generation "fpga_out/cut.mlir" -o "fpga_out/cutfin.mlir"

# Backend Codegen
./build/bin/ep2c-opt  -ep2-emit-fpga "fpga_out/mapped.mlir" -o "fpga_out/emit.mlir"


cp ./__*.sv ./fpga_out/
rm ./__*.sv