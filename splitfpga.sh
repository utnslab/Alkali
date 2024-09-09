 ninja -C build/
./build/bin/ep2c tests/experiments/nfchain/nfchain.ep2 -o nfchain.mlir

./build/bin/ep2c-opt nfchain.mlir -canonicalize -ep2-buffer-to-value --ep2-context-to-argument -o nfchain_value.mlir
./build/bin/ep2c-opt nfchain_value.mlir --ep2-pipeline-handler="mode=kcut knum=2" -o split_tmp.mlir
./build/bin/ep2c-opt --ep2-pipeline-canon="inline-table=true" -ep2-global-to-partition split_tmp.mlir -o split_fin.mlir
./build/bin/ep2c-opt  -ep2-emit-fpga split_fin.mlir -o tmpopt.mlir
