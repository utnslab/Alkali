ninja -C build/ -j 32

rm -rf ./fpga_out
mkdir -p ./fpga_out

file="$1"
# Extract the extension from the filename
extension="${file##*.}"
BASE_NAME="${1%.*}"

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

#TODO: split FPGA performance model instead of inlining
./build/bin/ep2c-opt "fpga_out/$BASE_NAME.mlir" -canonicalize -ep2-buffer-to-value --ep2-context-to-argument -o "fpga_out/tmp.mlir" 
./build/bin/ep2c-opt "fpga_out/tmp.mlir" --ep2-pipeline-handler="mode=search" -o "fpga_out/cut.mlir"
./build/bin/ep2c-opt --ep2-pipeline-canon="inline-table=true" -ep2-global-to-partition "fpga_out/cut.mlir" -o "fpga_out/cutfin.mlir"
./build/bin/ep2c-opt  -ep2-emit-fpga "fpga_out/cutfin.mlir" -o "fpga_out/emit.mlir"

cp ./__*.sv ./fpga_out/
rm ./__*.sv