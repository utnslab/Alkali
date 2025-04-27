# Alkali: Portable and High-Performance SmartNIC Programs with Alkali

## Building - Building MLIR

```sh
git apply patches/translateToCpp.patch
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

## Building - Alkali Compiler Build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
 cmake -G Ninja .. \
    -DLLVM_EXTERNAL_LIT=$PWD/../llvm-project/build/bin/llvm-lit \
    -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
    -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
    -DCMAKE_BUILD_TYPE=Debug
cd .. &&  ninja -C build/
```
This will generate Alkali compiler binary in ./build/bin/ep2c

## Running Alkali Example

The source files are located in `tests/experiments_c`. First, compile the C source into Alkali IR `./scripts/run_c.sh nfchain.c`. It will generate IR into `out.mlir`. 

After that is built, you run example with
```sh
bash ./scripts/fpga_compile.sh out.mlir
```
This will compile the nfchain example for FPGA NICs. The generated Verilog code can be found in ./fpga_out. It transform the oringinal RTC handler into pipeline and data parallel FPGA codes.

Similarly, if you want to compile the nfchain example for Agilio NICs, run:
```sh
bash ./scripts/netronome_compile.sh out.mlir
```
The generated Agilio Micro C code can be found in ./netronome_out.

More other applications can be found in ./tests/
