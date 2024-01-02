# EP2: Expressive, Portable and Performant Lanaguege for SmartNICs

## Building - Building MLIR

```sh
mkdir llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
 -DLLVM_ENABLE_PROJECTS="mlir;clang" \
 -DLLVM_TARGETS_TO_BUILD="host" \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_BUILD_TYPE=DEBUG 

```

## Install the external dependencies

```sh
# current z3 version is Z3 4.8.7
sudo apt install libz3-dev
```

## Building - Component Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
 cmake -G Ninja .. -DLLVM_EXTERNAL_LIT=llvm-project/build/bin/llvm-lit -DMLIR_DIR=llvm-project/build/lib/cmake/mlir
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```

## Running Example

After `ep2c` is built, run example with
```sh
./bin/ep2c --emit=mlir ../example.ep2.txt
```

To run netronome backend, do ./run.sh XXX.ep2, and output files will be generated in tests/ folder.
