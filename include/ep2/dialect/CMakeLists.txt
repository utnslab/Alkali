# add the attr definitions
set(LLVM_TARGET_DEFINITIONS EP2Ops.td)
mlir_tablegen(EP2OpsAttrDefs.h.inc -gen-attrdef-decls 
              -attrdefs-dialect=ep2)
mlir_tablegen(EP2OpsAttrDefs.cpp.inc -gen-attrdef-defs 
              -attrdefs-dialect=ep2)
add_public_tablegen_target(EP2OpsAttrDefsIncGen)

# get the rest of the definitions
add_mlir_dialect(EP2Ops ep2)

# set(LLVM_TARGET_DEFINITIONS Ops.td)
# mlir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
# add_public_tablegen_target(MLIRStandalonePassesIncGen)

# Most dialects should use add_mlir_dialect().  See examples/standalone.
