# Portable and High-Performance SmartNIC Programs with Alkali

![Status](https://img.shields.io/badge/Version-Experimental-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Alkali is a compiler infrastructure for SmartNICs, delivering both functional and performance portability across a wide range of SmartNIC hardware. It centers on a unified intermediate representation (IR), a common set of optimization and transformation passes, and an automated network-application parallelization pipeline.


Currently, this repo contains Alkali frontend for C and αIR, the definition and implementation of IR, optimization & transformation passes, and code generation for following backends: Verilog(FPGA), MicroC(Netronome), and LLVM(ARM DPDK/RiscV).




- [Portable and High-Performance SmartNIC Programs with Alkali](#portable-and-high-performance-smartnic-programs-with-alkali)
  - [Building](#building)
    - [Building - Alkali Compiler Build](#building---alkali-compiler-build)
  - [Running Alkali Example](#running-alkali-example)
  - [Reference](#reference)
  - [Contact](#contact)



## Building

Alkali depends on Boost and Z3. You need to install the external dependencies first, and build MLIR libraries using the following command:

```sh
# current z3 version is Z3 4.8.7
sudo apt install libz3-dev

# build MLIR
git apply patches/translateToCpp.patch
mkdir llvm-project/build && cd llvm-project/build
cmake -G Ninja ../llvm \
 -DLLVM_ENABLE_PROJECTS="mlir;clang" \
 -DLLVM_TARGETS_TO_BUILD="AArch64;AMDGPU;ARM;RISCV;X86" \
 -DLLVM_ENABLE_ASSERTIONS=ON \
 -DCMAKE_BUILD_TYPE=Release 
cd .. &&  ninja -C build/
```

### Building - Alkali Compiler Build

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

Alkali binaries, `ep2c` and `ep2c-opt`, will be generated `build/bin`.

## Running Alkali Example

The source files are located in `tests`.
Alkali framework will first convert the input file (C or αIR) into MLIR representation, and then apply optimization, mapping and code generation.

After that is built, you run example with
```sh
./scripts/alkalic tests/experiments_c/nfchain.c --target rtl
```
This will compile the nfchain example for FPGA NICs. The generated Verilog code can be found in `./fpga_out`. It transforms the oringinal RTC handler into pipeline and data parallel FPGA codes.

Similarly, if you want to compile the nfchain example for Agilio NICs, run:
```sh
./scripts/alkalic tests/experiments_c/nfchain.c --target netronome
```
The generated Agilio Micro C code can be found in `./netronome_out`.

More example applications can be found in ./tests/

## Reference

If you use Alkali in your research or projects, please cite our paper [Portable and High-Performance SmartNIC Programs with Alkali](https://www.usenix.org/conference/nsdi25/presentation/lin-jiaxin):

```
@inproceedings{alkali,
  title={Portable and High-Performance SmartNIC Programs with Alkali},
  author={Lin, Jiaxin and Guo, Zhiyuan and Shah, Mihir and Ji, Tao and Zhang, Yiying and Kim, Daehyeok and Akella, Aditya}
  booktitle={22st USENIX Symposium on Networked Systems Design and Implementation (NSDI 25)},
  year={2025}
}
```

## Contact

If you encounter any issues, want to report a bug, or request a new feature, please open an issue on this repository.
For additional assistance, you can also reach out to
`z9guo [at] ucsd [dot] edu` and `jxlin [at] utexas [dot] edu`.
