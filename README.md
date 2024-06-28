# EP2: Expressive, Portable and Performant Lanaguege for SmartNICs

## Building - Building MLIR

```sh
mkdir llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
 -DLLVM_ENABLE_PROJECTS="mlir;clang" \
 -DLLVM_TARGETS_TO_BUILD="AArch64;AMDGPU;ARM;AVR;BPF;Hexagon;Lanai;LoongArch;Mips;MSP430;NVPTX;PowerPC;RISCV;Sparc;SystemZ;VE;WebAssembly;X86;XCore" \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_BUILD_TYPE=Release 
cd .. &&  ninja -C build/
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
 cmake -G Ninja .. \
    -DLLVM_EXTERNAL_LIT=$PWD/../llvm-project/build/bin/llvm-lit \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DCMAKE_BUILD_TYPE=DEBUG
cd .. &&  ninja -C build/
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
