
#!/bin/bash

set -e

cd llvm-project && ninja -C build/ && cd -
ninja -C build/

./build/bin/ep2c-opt -ep2-pipeline-handler -o /dev/null tests/pipeline_test/nopipeline.mlir 2>&1 | tee partition.log
dot -Tpng cut.dot > cut.png
