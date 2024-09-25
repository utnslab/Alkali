ninja -C build/ -j 32
 rm -rf *.sv
./build/bin/ep2c $1 -o nfchain.mlir

./build/bin/ep2c-opt nfchain.mlir -canonicalize -ep2-buffer-to-value --ep2-context-to-argument -o nfchain_value.mlir
./build/bin/ep2c-opt nfchain_value.mlir --ep2-pipeline-handler="mode=search" -o split_tmp.mlir
./build/bin/ep2c-opt --ep2-pipeline-canon="inline-table=true" -ep2-global-to-partition split_tmp.mlir -o split_fin.mlir
./build/bin/ep2c-opt  -ep2-emit-fpga split_fin.mlir -o tmpopt.mlir
rm -rf ./fpga_split_out/$1
mkdir -p ./fpga_split_out/$1
cp tmpopt.mlir ./fpga_split_out/$1/
cp split_fin.mlir ./fpga_split_out/$1/
cp ./__*.sv ./fpga_split_out/$1/
rm ./__*.sv